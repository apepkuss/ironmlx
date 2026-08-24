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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
    scale: f32,
}

impl DFlash2Attention {
    pub(super) fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &DFlash2Config,
        draft_bits: Option<i32>,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: load_linear(loader, &format!("{prefix}.q_proj"), draft_bits)?,
            k_proj: load_linear(loader, &format!("{prefix}.k_proj"), draft_bits)?,
            v_proj: load_linear(loader, &format!("{prefix}.v_proj"), draft_bits)?,
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

        let q = self
            .q_proj
            .forward_on(x, target)?
            .reshape_on((batch, query_len, self.num_heads, self.head_dim), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let context_k = self
            .k_proj
            .forward_on(context, target)?
            .reshape_on(
                (batch, context_len, self.num_kv_heads, self.head_dim),
                target,
            )?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let context_v = self
            .v_proj
            .forward_on(context, target)?
            .reshape_on(
                (batch, context_len, self.num_kv_heads, self.head_dim),
                target,
            )?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let proposal_k = self
            .k_proj
            .forward_on(x, target)?
            .reshape_on((batch, query_len, self.num_kv_heads, self.head_dim), target)?
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let proposal_v = self
            .v_proj
            .forward_on(x, target)?
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
