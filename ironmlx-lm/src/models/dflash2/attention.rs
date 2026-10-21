use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::{config::DFlash2Config, load_linear};

#[derive(Clone)]
pub(super) struct DFlash2KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    processed: i32,
    max_size: i32,
    batch_size: usize,
}

impl DFlash2KvCache {
    pub(super) fn new(max_size: i32, initial_offset: i32) -> Result<Self> {
        if max_size <= 0 || initial_offset < 0 {
            return Err(anyhow!(
                "DFlash2 cache requires max_size>0 and initial_offset>=0"
            ));
        }
        Ok(Self {
            keys: None,
            values: None,
            processed: initial_offset,
            max_size,
            batch_size: 1,
        })
    }

    pub(super) fn max_size(&self) -> i32 {
        self.max_size
    }

    pub(super) fn len(&self) -> i32 {
        self.keys
            .as_ref()
            .map(|keys| keys.shape().as_slice()[2])
            .unwrap_or(0)
    }

    pub(super) fn processed(&self) -> i32 {
        self.processed
    }

    pub(super) fn len_after_append(&self, append: i32) -> i32 {
        (self.len() + append).min(self.max_size)
    }

    pub(super) fn stack_rows_on(rows: &[&Self], target: StreamOrDevice) -> Result<Self> {
        let first = rows
            .first()
            .ok_or_else(|| anyhow!("DFlash2 cache row stack cannot be empty"))?;
        if rows.iter().any(|row| {
            row.processed != first.processed
                || row.max_size != first.max_size
                || row.batch_size != first.batch_size
                || row.len() != first.len()
                || row.keys.is_some() != first.keys.is_some()
                || row.values.is_some() != first.values.is_some()
        }) {
            return Err(anyhow!(
                "DFlash2 cache rows require matching position, capacity, and retained length"
            ));
        }
        let keys = first
            .keys
            .as_ref()
            .map(|_| {
                let arrays = rows
                    .iter()
                    .map(|row| row.keys.as_ref().expect("validated key presence"))
                    .collect::<Vec<_>>();
                mlx::ops::shape::concatenate_on(&arrays, 0, target)
            })
            .transpose()?;
        let values = first
            .values
            .as_ref()
            .map(|_| {
                let arrays = rows
                    .iter()
                    .map(|row| row.values.as_ref().expect("validated value presence"))
                    .collect::<Vec<_>>();
                mlx::ops::shape::concatenate_on(&arrays, 0, target)
            })
            .transpose()?;
        Ok(Self {
            keys,
            values,
            processed: first.processed,
            max_size: first.max_size,
            batch_size: rows.len() * first.batch_size,
        })
    }

    pub(super) fn row_on(&self, row: usize, target: StreamOrDevice) -> Result<Self> {
        if row >= self.batch_size {
            return Err(anyhow!(
                "DFlash2 cache row {row} is outside batch width {}",
                self.batch_size
            ));
        }
        let slice_row = |array: &Array| -> Result<Array> {
            let shape = array.shape();
            let dims = shape.as_slice();
            mlx::ops::indexing::slice_strided_on(
                array,
                &[row as i32, 0, 0, 0][..],
                &[row as i32 + 1, dims[1], dims[2], dims[3]][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )
            .map_err(Into::into)
        };
        Ok(Self {
            keys: self.keys.as_ref().map(slice_row).transpose()?,
            values: self.values.as_ref().map(slice_row).transpose()?,
            processed: self.processed,
            max_size: self.max_size,
            batch_size: 1,
        })
    }

    fn append_on(
        &mut self,
        keys: &Array,
        values: &Array,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let key_shape = keys.shape();
        let key_dims = key_shape.as_slice();
        let value_shape = values.shape();
        let value_dims = value_shape.as_slice();
        if key_dims.len() != 4 || value_dims != key_dims {
            return Err(anyhow!(
                "DFlash2 cache append requires matching rank-4 K/V, got K={key_dims:?} V={value_dims:?}"
            ));
        }
        if usize::try_from(key_dims[0])? != self.batch_size {
            return Err(anyhow!(
                "DFlash2 cache batch changed while active: previous={} next={}",
                self.batch_size,
                key_dims[0]
            ));
        }
        let append = keys.shape().as_slice()[2];
        let mut combined_keys = match &self.keys {
            Some(previous) => mlx::ops::shape::concatenate_on(&[previous, keys], 2, target)?,
            None => keys.clone(),
        };
        let mut combined_values = match &self.values {
            Some(previous) => mlx::ops::shape::concatenate_on(&[previous, values], 2, target)?,
            None => values.clone(),
        };
        let total = combined_keys.shape().as_slice()[2];
        if total > self.max_size {
            let start = total - self.max_size;
            let kshape = combined_keys.shape();
            let kdims = kshape.as_slice();
            combined_keys = mlx::ops::indexing::slice_strided_on(
                &combined_keys,
                &[0_i32, 0, start, 0][..],
                &[kdims[0], kdims[1], total, kdims[3]][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?;
            let vshape = combined_values.shape();
            let vdims = vshape.as_slice();
            combined_values = mlx::ops::indexing::slice_strided_on(
                &combined_values,
                &[0_i32, 0, start, 0][..],
                &[vdims[0], vdims[1], total, vdims[3]][..],
                &[1_i32, 1, 1, 1][..],
                target,
            )?;
        }
        self.processed = self
            .processed
            .checked_add(append)
            .ok_or_else(|| anyhow!("DFlash2 cache position overflow"))?;
        self.keys = Some(combined_keys.clone());
        self.values = Some(combined_values.clone());
        Ok((combined_keys, combined_values))
    }
}

pub(super) struct DFlash2Attention {
    input_projections: DFlash2InputProjections,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
    scale: f32,
}

enum DFlash2InputProjections {
    Separate {
        q: Linear,
        k: Linear,
        v: Linear,
    },
    Fused {
        projection: Linear,
        k: Linear,
        v: Linear,
        q_width: i32,
        k_width: i32,
    },
}

impl DFlash2Attention {
    pub(super) fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &DFlash2Config,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        let q = load_linear(loader, &format!("{prefix}.q_proj"), draft_bits)?;
        let k = load_linear(loader, &format!("{prefix}.k_proj"), draft_bits)?;
        let v = load_linear(loader, &format!("{prefix}.v_proj"), draft_bits)?;
        let input_projections = if q.quantized_parts().is_some()
            && k.quantized_parts().is_some()
            && v.quantized_parts().is_some()
        {
            let q_width = i32::try_from(q.out_features())?;
            let k_width = i32::try_from(k.out_features())?;
            let v_width = v.out_features();
            let projection =
                Linear::fuse_quantized_outputs(&[&q, &k, &v], "DFlash2 drafter fused Q/K/V")?;
            if super::m5_resident_requested() {
                // The context K/V keep their own arrays (row views would keep
                // the fused checkpoint-layout buffer alive beside its tiled
                // store).
                DFlash2InputProjections::Fused {
                    projection,
                    k,
                    v,
                    q_width,
                    k_width,
                }
            } else {
                let mut split = projection.split_quantized_outputs(
                    &[q.out_features(), k.out_features(), v_width],
                    "DFlash2 drafter split Q/K/V",
                )?;
                DFlash2InputProjections::Fused {
                    projection,
                    // Q is always consumed from the fused proposal projection;
                    // only K/V need shared Q1-compatible row views for the
                    // variable-length context projection.
                    k: {
                        let _q_view = split.remove(0);
                        split.remove(0)
                    },
                    v: split.remove(0),
                    q_width,
                    k_width,
                }
            }
        } else {
            DFlash2InputProjections::Separate { q, k, v }
        };
        Ok(Self {
            input_projections,
            o_proj: load_linear(loader, &format!("{prefix}.o_proj"), draft_bits)?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_parameters.rope_theta,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        })
    }

    /// Move every projection into a single-resident M5 store; returns the
    /// number of stores.
    pub(super) fn share_weight_layout(&mut self) -> Result<usize> {
        let mut linears: Vec<&mut Linear> = match &mut self.input_projections {
            DFlash2InputProjections::Separate { q, k, v } => vec![q, k, v],
            DFlash2InputProjections::Fused {
                projection, k, v, ..
            } => vec![projection, k, v],
        };
        linears.push(&mut self.o_proj);
        let mut stores = 0;
        for linear in linears {
            stores += usize::from(linear.share_whole()?.is_some());
        }
        Ok(stores)
    }

    /// Every projection object, including the K/V row views of a fused Q/K/V.
    pub(super) fn projections(&self) -> Vec<&Linear> {
        let mut out = match &self.input_projections {
            DFlash2InputProjections::Separate { q, k, v } => vec![q, k, v],
            DFlash2InputProjections::Fused {
                projection, k, v, ..
            } => vec![projection, k, v],
        };
        out.push(&self.o_proj);
        out
    }

    pub(super) fn forward_on(
        &self,
        x: &Array,
        context: &Array,
        mask: &Array,
        cache: &mut DFlash2KvCache,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let xshape = x.shape();
        let xdims = xshape.as_slice();
        let cshape = context.shape();
        let cdims = cshape.as_slice();
        if xdims.len() != 3 || cdims.len() != 3 || xdims[0] <= 0 || cdims[0] != xdims[0] {
            return Err(anyhow!(
                "DFlash2 attention requires matching [B,L,H] inputs with B>0"
            ));
        }
        let batch = xdims[0];
        let query_len = xdims[1];
        let context_len = cdims[1];
        let context_start = cache.processed();
        let proposal_start = context_start
            .checked_add(context_len)
            .ok_or_else(|| anyhow!("DFlash2 RoPE position overflow"))?;

        let (q, proposal_k, proposal_v) = match &self.input_projections {
            DFlash2InputProjections::Separate { q, k, v } => (
                q.forward_on(x, target)?,
                k.forward_on(x, target)?,
                v.forward_on(x, target)?,
            ),
            DFlash2InputProjections::Fused {
                projection,
                q_width,
                k_width,
                ..
            } => {
                let output = projection.forward_on(x, target)?;
                let mut parts = mlx::ops::shape::split_at_on(
                    &output,
                    &[*q_width, *q_width + *k_width],
                    -1,
                    target,
                )?;
                if parts.len() != 3 {
                    return Err(anyhow!(
                        "DFlash2 drafter fused Q/K/V returned {} parts",
                        parts.len()
                    ));
                }
                (parts.remove(0), parts.remove(0), parts.remove(0))
            }
        };
        let q = q
            .reshape_on((batch, query_len, self.num_heads, self.head_dim), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        // Context K/V uses a dynamic accepted-prefix length. Preserve the B1
        // affine accumulation tree across tensor-batch widths without routing
        // the fixed-width proposal and MLP projections through the slower
        // product-stable kernel.
        let (context_k, context_v) = {
            let _product_stable_qmm = crate::nn::product_stable_qmm::scope();
            match &self.input_projections {
                DFlash2InputProjections::Separate { k, v, .. }
                | DFlash2InputProjections::Fused { k, v, .. } => (
                    k.forward_on(context, target)?,
                    v.forward_on(context, target)?,
                ),
            }
        };
        let context_k = context_k
            .reshape_on(
                (batch, context_len, self.num_kv_heads, self.head_dim),
                target,
            )?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let context_v = context_v
            .reshape_on(
                (batch, context_len, self.num_kv_heads, self.head_dim),
                target,
            )?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let proposal_k = proposal_k
            .reshape_on((batch, query_len, self.num_kv_heads, self.head_dim), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let proposal_v = proposal_v
            .reshape_on((batch, query_len, self.num_kv_heads, self.head_dim), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;

        let q = self.q_norm.forward_on(&q, target)?;
        let context_k = self.k_norm.forward_on(&context_k, target)?;
        let proposal_k = self.k_norm.forward_on(&proposal_k, target)?;
        let q = mlx::fast::rope_on(
            &q,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            proposal_start,
            None,
            target,
        )?;
        let context_k = mlx::fast::rope_on(
            &context_k,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            context_start,
            None,
            target,
        )?;
        let proposal_k = mlx::fast::rope_on(
            &proposal_k,
            self.head_dim,
            false,
            Some(self.rope_theta),
            1.0,
            proposal_start,
            None,
            target,
        )?;

        let (cached_k, cached_v) = cache.append_on(&context_k, &context_v, target)?;
        let keys = mlx::ops::shape::concatenate_on(&[&cached_k, &proposal_k], 2, target)?;
        let values = mlx::ops::shape::concatenate_on(&[&cached_v, &proposal_v], 2, target)?;
        let output = mlx::fast::scaled_dot_product_attention_on(
            &q,
            &keys,
            &values,
            self.scale,
            "",
            Some(mask),
            None,
            target,
        )?;
        let output = output
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
            .reshape_on((batch, query_len, self.num_heads * self.head_dim), target)?;
        self.o_proj.forward_on(&output, target)
    }

    /// Rows at different draft-cache positions: the fused proposal projection
    /// and `o_proj` run once for all rows; each row's context projection,
    /// RoPE offsets, cache append and attention follow [`Self::forward_on`]
    /// with batch 1.
    pub(super) fn forward_ragged_on(
        &self,
        x: &Array,
        contexts: &[Array],
        masks: &[Array],
        caches: &mut [&mut DFlash2KvCache],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let xshape = x.shape();
        let xdims = xshape.as_slice();
        let batch = usize::try_from(xdims[0])?;
        if xdims.len() != 3
            || contexts.len() != batch
            || masks.len() != batch
            || caches.len() != batch
        {
            return Err(anyhow!(
                "DFlash2 ragged attention requires one context, mask and cache per row"
            ));
        }
        let query_len = xdims[1];
        let (q, proposal_k, proposal_v) = match &self.input_projections {
            DFlash2InputProjections::Separate { q, k, v } => (
                q.forward_on(x, target)?,
                k.forward_on(x, target)?,
                v.forward_on(x, target)?,
            ),
            DFlash2InputProjections::Fused {
                projection,
                q_width,
                k_width,
                ..
            } => {
                let output = projection.forward_on(x, target)?;
                let mut parts = mlx::ops::shape::split_at_on(
                    &output,
                    &[*q_width, *q_width + *k_width],
                    -1,
                    target,
                )?;
                if parts.len() != 3 {
                    return Err(anyhow!(
                        "DFlash2 drafter fused Q/K/V returned {} parts",
                        parts.len()
                    ));
                }
                (parts.remove(0), parts.remove(0), parts.remove(0))
            }
        };
        if draft_batched_requested()
            && contexts
                .iter()
                .all(|context| context.ndim() == 3 && context.shape().as_slice()[1] > 0)
        {
            return self.forward_ragged_batched_on(
                &q,
                &proposal_k,
                &proposal_v,
                query_len,
                contexts,
                masks,
                caches,
                target,
            );
        }
        let row_of = |array: &Array, row: i32| -> Result<Array> {
            let shape = array.shape();
            let dims = shape.as_slice();
            mlx::ops::indexing::slice_strided_on(
                array,
                &[row, 0, 0][..],
                &[row + 1, dims[1], dims[2]][..],
                &[1_i32, 1, 1][..],
                target,
            )
            .map_err(Into::into)
        };
        let mut outputs = Vec::with_capacity(batch);
        for (row, ((context, mask), cache)) in contexts
            .iter()
            .zip(masks)
            .zip(caches.iter_mut())
            .enumerate()
        {
            let row_i32 = i32::try_from(row)?;
            let context_len = context.shape().as_slice()[1];
            let context_start = cache.processed();
            let proposal_start = context_start
                .checked_add(context_len)
                .ok_or_else(|| anyhow!("DFlash2 RoPE position overflow"))?;
            let q = row_of(&q, row_i32)?
                .reshape_on((1, query_len, self.num_heads, self.head_dim), target)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
            let (context_k, context_v) = {
                let _product_stable_qmm = crate::nn::product_stable_qmm::scope();
                match &self.input_projections {
                    DFlash2InputProjections::Separate { k, v, .. }
                    | DFlash2InputProjections::Fused { k, v, .. } => (
                        k.forward_on(context, target)?,
                        v.forward_on(context, target)?,
                    ),
                }
            };
            let context_k = context_k
                .reshape_on((1, context_len, self.num_kv_heads, self.head_dim), target)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
            let context_v = context_v
                .reshape_on((1, context_len, self.num_kv_heads, self.head_dim), target)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
            let proposal_k = row_of(&proposal_k, row_i32)?
                .reshape_on((1, query_len, self.num_kv_heads, self.head_dim), target)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
            let proposal_v = row_of(&proposal_v, row_i32)?
                .reshape_on((1, query_len, self.num_kv_heads, self.head_dim), target)?
                .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
            let q = self.q_norm.forward_on(&q, target)?;
            let context_k = self.k_norm.forward_on(&context_k, target)?;
            let proposal_k = self.k_norm.forward_on(&proposal_k, target)?;
            let rope = |array: &Array, offset: i32| {
                mlx::fast::rope_on(
                    array,
                    self.head_dim,
                    false,
                    Some(self.rope_theta),
                    1.0,
                    offset,
                    None,
                    target,
                )
            };
            let q = rope(&q, proposal_start)?;
            let context_k = rope(&context_k, context_start)?;
            let proposal_k = rope(&proposal_k, proposal_start)?;
            let (cached_k, cached_v) = cache.append_on(&context_k, &context_v, target)?;
            let keys = mlx::ops::shape::concatenate_on(&[&cached_k, &proposal_k], 2, target)?;
            let values = mlx::ops::shape::concatenate_on(&[&cached_v, &proposal_v], 2, target)?;
            let output = mlx::fast::scaled_dot_product_attention_on(
                &q,
                &keys,
                &values,
                self.scale,
                "",
                Some(mask),
                None,
                target,
            )?;
            outputs.push(
                output
                    .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
                    .reshape_on((1, query_len, self.num_heads * self.head_dim), target)?,
            );
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = mlx::ops::shape::concatenate_on(&output_refs, 0, target)?;
        self.o_proj.forward_on(&output, target)
    }

    /// Ragged attention with the row-independent work batched: one context
    /// K/V projection for every row's new context, and q, proposal k and
    /// context k normalized and rotated in one dispatch each at their own
    /// positions; each row then appends its cache and attends as before.
    #[allow(clippy::too_many_arguments)]
    fn forward_ragged_batched_on(
        &self,
        q: &Array,
        proposal_k: &Array,
        proposal_v: &Array,
        query_len: i32,
        contexts: &[Array],
        masks: &[Array],
        caches: &mut [&mut DFlash2KvCache],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let batch = i32::try_from(contexts.len())?;
        let (hq, hkv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let mut q_pos = Vec::with_capacity(usize::try_from(batch * query_len)?);
        let mut ctx_pos = Vec::new();
        let mut ctx_lens = Vec::with_capacity(contexts.len());
        for (context, cache) in contexts.iter().zip(caches.iter()) {
            let context_len = context.shape().as_slice()[1];
            let context_start = cache.processed();
            let proposal_start = context_start
                .checked_add(context_len)
                .ok_or_else(|| anyhow!("DFlash2 RoPE position overflow"))?;
            q_pos.extend(proposal_start..proposal_start + query_len);
            ctx_pos.extend(context_start..context_start + context_len);
            ctx_lens.push(context_len);
        }
        let total_ctx: i32 = ctx_lens.iter().sum();
        let q_pos: Array = (&q_pos[..], &[batch * query_len][..]).try_into()?;
        let ctx_pos: Array = (&ctx_pos[..], &[total_ctx][..]).try_into()?;
        let q_all = norm_rope_on(
            &q.reshape_on((batch * query_len, hq, d), target)?,
            self.q_norm.weight(),
            self.q_norm.eps(),
            &q_pos,
            self.rope_theta,
            target,
        )?;
        let pk_all = norm_rope_on(
            &proposal_k.reshape_on((batch * query_len, hkv, d), target)?,
            self.k_norm.weight(),
            self.k_norm.eps(),
            &q_pos,
            self.rope_theta,
            target,
        )?;
        let pv_all = proposal_v.reshape_on((batch * query_len, hkv, d), target)?;
        let context_refs = contexts.iter().collect::<Vec<_>>();
        let ctx = mlx::ops::shape::concatenate_on(&context_refs, 1, target)?;
        let (ck, cv) = {
            let _product_stable_qmm = crate::nn::product_stable_qmm::scope();
            match &self.input_projections {
                DFlash2InputProjections::Separate { k, v, .. }
                | DFlash2InputProjections::Fused { k, v, .. } => {
                    (k.forward_on(&ctx, target)?, v.forward_on(&ctx, target)?)
                }
            }
        };
        let ck_all = norm_rope_on(
            &ck.reshape_on((total_ctx, hkv, d), target)?,
            self.k_norm.weight(),
            self.k_norm.eps(),
            &ctx_pos,
            self.rope_theta,
            target,
        )?;
        let cv_all = cv.reshape_on((total_ctx, hkv, d), target)?;
        // Rows [start, start + len) of a [N, H, D] array as [1, H, len, D].
        let rows = |array: &Array, start: i32, len: i32, heads: i32| -> Result<Array> {
            Ok(mlx::ops::indexing::slice_strided_on(
                array,
                &[start, 0, 0][..],
                &[start + len, heads, d][..],
                &[1_i32, 1, 1][..],
                target,
            )?
            .reshape_on((1, len, heads, d), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?)
        };
        // One fused attention dispatch for every row when all keys sit inside
        // the sliding window (no key is masked).
        let fused = contexts.len() <= 4
            && d == 128
            && q.dtype() == mlx::Dtype::Bfloat16
            && caches
                .iter()
                .zip(&ctx_lens)
                .all(|(cache, &len)| cache.len() + len + query_len <= cache.max_size());
        if fused {
            let mut ctx_off = 0_i32;
            let mut row_caches = Vec::with_capacity(contexts.len());
            for (row, cache) in caches.iter_mut().enumerate() {
                let len = ctx_lens[row];
                let ck_r = rows(&ck_all, ctx_off, len, hkv)?;
                let cv_r = rows(&cv_all, ctx_off, len, hkv)?;
                ctx_off += len;
                row_caches.push(cache.append_on(&ck_r, &cv_r, target)?);
            }
            let output = ragged_draft_attention_on(
                &q_all,
                &pk_all,
                &pv_all,
                &row_caches,
                query_len,
                self.scale,
                target,
            )?
            .reshape_on((batch, query_len, hq * d), target)?;
            return self.o_proj.forward_on(&output, target);
        }
        let mut outputs = Vec::with_capacity(contexts.len());
        let mut ctx_off = 0_i32;
        for (row, (mask, cache)) in masks.iter().zip(caches.iter_mut()).enumerate() {
            let row_i32 = i32::try_from(row)?;
            let len = ctx_lens[row];
            let q_r = rows(&q_all, row_i32 * query_len, query_len, hq)?;
            let pk_r = rows(&pk_all, row_i32 * query_len, query_len, hkv)?;
            let pv_r = rows(&pv_all, row_i32 * query_len, query_len, hkv)?;
            let ck_r = rows(&ck_all, ctx_off, len, hkv)?;
            let cv_r = rows(&cv_all, ctx_off, len, hkv)?;
            ctx_off += len;
            let (cached_k, cached_v) = cache.append_on(&ck_r, &cv_r, target)?;
            let keys = mlx::ops::shape::concatenate_on(&[&cached_k, &pk_r], 2, target)?;
            let values = mlx::ops::shape::concatenate_on(&[&cached_v, &pv_r], 2, target)?;
            let output = mlx::fast::scaled_dot_product_attention_on(
                &q_r,
                &keys,
                &values,
                self.scale,
                "",
                Some(mask),
                None,
                target,
            )?;
            outputs.push(
                output
                    .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
                    .reshape_on((1, query_len, hq * d), target)?,
            );
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = mlx::ops::shape::concatenate_on(&output_refs, 0, target)?;
        self.o_proj.forward_on(&output, target)
    }
}

/// Attention of every ragged draft row in one dispatch: `q` `[B*Lq, Hq, D]`,
/// proposal `pk`/`pv` `[B*Lq, Hkv, D]`, and each row's cache `(k, v)`
/// `[1, Hkv, P_r, D]` (context included). Returns `[B*Lq, Hq, D]` bf16. No key
/// is masked (callers keep keys inside the sliding window).
fn ragged_draft_attention_on(
    q: &Array,
    pk: &Array,
    pv: &Array,
    row_caches: &[(Array, Array)],
    query_len: i32,
    scale: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: std::sync::OnceLock<mlx::MetalKernel> = std::sync::OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        mlx::MetalKernel::builder("ironmlx_dflash2_ragged_attention_v1")
            .inputs(&["q", "pk", "pv", "k0", "v0", "k1", "v1", "k2", "v2", "k3", "v3", "params"])
            .outputs(&["out"])
            .source(
                r#"
  const uint lane = thread_index_in_simdgroup;
  const uint h = thread_position_in_grid.y;
  const uint ri = thread_position_in_grid.z;
  const uint r = ri / LQ;
  const uint hk = h / (HQ / HKV);
  const device bfloat* ck;
  const device bfloat* cv;
  int P;
  if (r == 0) { ck = (const device bfloat*)k0; cv = (const device bfloat*)v0; P = k0_shape[2]; }
  else if (r == 1) { ck = (const device bfloat*)k1; cv = (const device bfloat*)v1; P = k1_shape[2]; }
  else if (r == 2) { ck = (const device bfloat*)k2; cv = (const device bfloat*)v2; P = k2_shape[2]; }
  else { ck = (const device bfloat*)k3; cv = (const device bfloat*)v3; P = k3_shape[2]; }
  const float sc = params[0];
  const device bfloat* qp = (const device bfloat*)q + (ulong(ri) * HQ + h) * D + lane * 4;
  float qv[4];
  for (uint j = 0; j < 4; ++j) qv[j] = float(qp[j]) * sc;
  float m = -INFINITY;
  float l = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  auto step = [&](const device bfloat* kr, const device bfloat* vr) {
    float d = 0.0f;
    for (uint j = 0; j < 4; ++j) d += qv[j] * float(kr[j]);
    d = simd_sum(d);
    const float nm = max(m, d);
    const float a = metal::exp(m - nm);
    const float b = metal::exp(d - nm);
    l = l * a + b;
    for (uint j = 0; j < 4; ++j) acc[j] = acc[j] * a + b * float(vr[j]);
    m = nm;
  };
  const ulong head = ulong(hk) * ulong(P) * D;
  for (int n = 0; n < P; ++n) {
    const ulong o = head + ulong(n) * D + lane * 4;
    step(ck + o, cv + o);
  }
  for (uint n = 0; n < LQ; ++n) {
    const ulong o = (ulong(r * LQ + n) * HKV + hk) * D + lane * 4;
    step((const device bfloat*)pk + o, (const device bfloat*)pv + o);
  }
  device bfloat* op = (device bfloat*)out + (ulong(ri) * HQ + h) * D + lane * 4;
  for (uint j = 0; j < 4; ++j) op[j] = bfloat(acc[j] / l);
"#,
            )
            .build()
            .expect("build DFlash2 ragged attention kernel")
    });
    let qs = q.shape();
    let &[n, hq, d] = qs.as_slice() else {
        return Err(anyhow!("ragged draft attention q must be [N, Hq, D]"));
    };
    let hkv = pk.shape().as_slice()[1];
    anyhow::ensure!(
        (1..=4).contains(&row_caches.len()) && d == 128 && n == query_len * row_caches.len() as i32,
        "ragged draft attention supports 1..=4 rows of D=128"
    );
    let params: Array = (&[scale][..], &[1_i32][..]).try_into()?;
    let pick = |i: usize| &row_caches[i.min(row_caches.len() - 1)];
    let mut inputs: Vec<&Array> = vec![q, pk, pv];
    for i in 0..4 {
        let (k, v) = pick(i);
        inputs.push(k);
        inputs.push(v);
    }
    inputs.push(&params);
    Ok(kernel
        .dispatch_builder()
        .inputs(&inputs)
        .output_shapes(&[mlx::Shape::from(&[n, hq, d][..])])
        .output_dtypes(&[mlx::Dtype::Bfloat16])
        .grid(32, hq, n)
        .threadgroup(32, 1, 1)
        .template_int("LQ", query_len)
        .template_int("HQ", hq)
        .template_int("HKV", hkv)
        .template_int("D", d)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

/// M5 profile setting `DFLASH2_DRAFT_BATCHED`.
fn draft_batched_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::DFLASH2_DRAFT_BATCHED)
    })
}

/// Per-head RMSNorm (weight `[D]`) then non-traditional RoPE at each token's
/// own position, for `x` `[N, H, D]` bf16 and `positions` `[N]` int32. The
/// norm output is rounded to bf16 before the rotation, as the two-op path.
fn norm_rope_on(
    x: &Array,
    weight: &Array,
    eps: f32,
    positions: &Array,
    theta: f32,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: std::sync::OnceLock<mlx::MetalKernel> = std::sync::OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        mlx::MetalKernel::builder("ironmlx_dflash2_norm_rope_v1")
            .inputs(&["x", "w", "pos", "params"])
            .outputs(&["y"])
            .source(
                r#"
  constexpr uint J = D / 32;
  const uint lane = thread_position_in_grid.x;
  const uint h = thread_position_in_grid.y;
  const uint n = thread_position_in_grid.z;
  const ulong base = (ulong(n) * HH + h) * D;
  float v[J];
  float ss = 0.0f;
  for (uint j = 0; j < J; ++j) {
    v[j] = float(x[base + lane + 32 * j]);
    ss += v[j] * v[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(D) + params[0]);
  for (uint j = 0; j < J; ++j) v[j] = float(bfloat(v[j] * inv * float(w[lane + 32 * j])));
  const float p = float(pos[n]);
  const float log2_theta = metal::log2(params[1]);
  for (uint j = 0; j < J / 2; ++j) {
    const uint i = lane + 32 * j;
    const float inv_freq = metal::exp2(-(2.0f * float(i) / float(D)) * log2_theta);
    const float ang = p * inv_freq;
    const float c = metal::cos(ang);
    const float s = metal::sin(ang);
    const float a = v[j];
    const float b = v[j + J / 2];
    y[base + i] = bfloat(a * c - b * s);
    y[base + i + D / 2] = bfloat(a * s + b * c);
  }
"#,
            )
            .build()
            .expect("build DFlash2 norm+rope kernel")
    });
    let shape = x.shape();
    let &[n, heads, d] = shape.as_slice() else {
        return Err(anyhow!(
            "norm_rope expects [N, H, D], got {:?}",
            shape.as_slice()
        ));
    };
    if d % 64 != 0 || x.dtype() != mlx::Dtype::Bfloat16 {
        return Err(anyhow!("norm_rope expects bf16 with D a multiple of 64"));
    }
    let params: Array = (&[eps, theta][..], &[2_i32][..]).try_into()?;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[x, weight, positions, &params])
        .output_shapes(&[mlx::Shape::from(&[n, heads, d][..])])
        .output_dtypes(&[mlx::Dtype::Bfloat16])
        .grid(32, heads, n)
        .threadgroup(32, 1, 1)
        .template_int("D", d)
        .template_int("HH", heads)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_cache_rows_stack_and_split_without_value_or_position_loss() {
        let first_values = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let second_values = (8..16).map(|value| value as f32).collect::<Vec<_>>();
        let first: Array = (&first_values[..], &[1_i32, 2, 2, 2][..])
            .try_into()
            .expect("first cache row");
        let second: Array = (&second_values[..], &[1_i32, 2, 2, 2][..])
            .try_into()
            .expect("second cache row");
        let mut first_cache = DFlash2KvCache::new(16, 4).expect("first cache");
        let mut second_cache = DFlash2KvCache::new(16, 4).expect("second cache");
        first_cache
            .append_on(&first, &first, StreamOrDevice::default())
            .expect("append first row");
        second_cache
            .append_on(&second, &second, StreamOrDevice::default())
            .expect("append second row");

        let stacked = DFlash2KvCache::stack_rows_on(
            &[&first_cache, &second_cache],
            StreamOrDevice::default(),
        )
        .expect("stack rows");
        assert_eq!(stacked.processed(), 6);
        assert_eq!(stacked.len(), 2);
        let restored = stacked
            .row_on(1, StreamOrDevice::default())
            .expect("split row");
        assert_eq!(restored.processed(), 6);
        assert_eq!(restored.len(), 2);
        assert_eq!(
            restored
                .keys
                .expect("restored keys")
                .to_vec::<f32>()
                .expect("materialize restored keys"),
            second_values
        );
    }

    #[test]
    fn draft_cache_stack_rejects_different_positions() {
        let first = DFlash2KvCache::new(16, 4).expect("first cache");
        let second = DFlash2KvCache::new(16, 5).expect("second cache");
        let error =
            match DFlash2KvCache::stack_rows_on(&[&first, &second], StreamOrDevice::default()) {
                Ok(_) => panic!("different positions must not batch"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("matching position"));
    }

    #[test]
    fn empty_draft_cache_remembers_stacked_batch_width() {
        let first = DFlash2KvCache::new(16, 4).expect("first cache");
        let second = DFlash2KvCache::new(16, 4).expect("second cache");
        let third = DFlash2KvCache::new(16, 4).expect("third cache");
        let fourth = DFlash2KvCache::new(16, 4).expect("fourth cache");
        let stacked = DFlash2KvCache::stack_rows_on(
            &[&first, &second, &third, &fourth],
            StreamOrDevice::default(),
        )
        .expect("stack empty rows");

        let restored = stacked
            .row_on(3, StreamOrDevice::default())
            .expect("split fourth empty row");
        assert_eq!(restored.processed(), 4);
        assert_eq!(restored.len(), 0);
    }
}
