use anyhow::{anyhow, Context};
use mlx::{Array, Dtype, StreamOrDevice};
use std::time::Instant;

use crate::core::Loader;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::config::Gemma4VisionConfig;
use super::ops::{gelu_approx_mul_on, rms_norm_no_scale_on};

struct ClippableLinear {
    linear: Linear,
    input_min: Option<Array>,
    input_max: Option<Array>,
    output_min: Option<Array>,
    output_max: Option<Array>,
}

impl ClippableLinear {
    fn from_loader(loader: &Loader, prefix: &str, use_clipping: bool) -> Result<Self> {
        let linear_prefix = format!("{prefix}.linear");
        let actual_prefix = if loader.contains(&format!("{linear_prefix}.weight")) {
            linear_prefix
        } else {
            prefix.to_owned()
        };
        Ok(Self {
            linear: Linear::from_loader(loader, &actual_prefix)
                .with_context(|| format!("loading Gemma4 vision linear `{actual_prefix}`"))?,
            input_min: use_clipping
                .then(|| loader.tensor(&format!("{prefix}.input_min")).cloned())
                .transpose()?,
            input_max: use_clipping
                .then(|| loader.tensor(&format!("{prefix}.input_max")).cloned())
                .transpose()?,
            output_min: use_clipping
                .then(|| loader.tensor(&format!("{prefix}.output_min")).cloned())
                .transpose()?,
            output_max: use_clipping
                .then(|| loader.tensor(&format!("{prefix}.output_max")).cloned())
                .transpose()?,
        })
    }

    fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let x = if self.input_min.is_some() || self.input_max.is_some() {
            mlx::ops::clip_on(x, self.input_min.as_ref(), self.input_max.as_ref(), target)?
        } else {
            x.clone()
        };
        let y = self.linear.forward_on(&x, target)?;
        if self.output_min.is_some() || self.output_max.is_some() {
            Ok(mlx::ops::clip_on(
                &y,
                self.output_min.as_ref(),
                self.output_max.as_ref(),
                target,
            )?)
        } else {
            Ok(y)
        }
    }
}

pub struct MultimodalEmbedder {
    projection: Linear,
    eps: f32,
}

impl MultimodalEmbedder {
    pub fn from_loader(loader: &Loader, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            projection: Linear::from_loader(loader, &format!("{prefix}.embedding_projection"))?,
            eps,
        })
    }

    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        let x = rms_norm_no_scale_on(x, self.eps, target)?;
        self.projection.forward_on(&x, target)
    }
}

struct VisionPatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Array,
    hidden_size: i32,
    patch_size: i32,
    position_embedding_size: i32,
}

impl VisionPatchEmbedder {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &Gemma4VisionConfig) -> Result<Self> {
        Ok(Self {
            input_proj: Linear::from_loader(loader, &format!("{prefix}.input_proj"))?,
            position_embedding_table: loader
                .tensor(&format!("{prefix}.position_embedding_table"))?
                .clone(),
            hidden_size: cfg.hidden_size,
            patch_size: cfg.patch_size,
            position_embedding_size: cfg.position_embedding_size,
        })
    }

    fn forward_on(
        &self,
        pixel_values: &Array,
        pos_x: &[u32],
        pos_y: &[u32],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = pixel_values.shape();
        let dims = shape.as_slice();
        if dims.len() != 4 {
            return Err(anyhow!(
                "Gemma4Vision patch embedder expects [B,3,H,W], got {dims:?}"
            ));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if b <= 0 || c != 3 {
            return Err(anyhow!(
                "Gemma4Vision patch embedder expects RGB image batch [B,3,H,W], got {dims:?}"
            ));
        }
        let p = self.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(anyhow!(
                "Gemma4Vision patch embedder: H,W must be divisible by patch_size {p}, got {h}x{w}"
            ));
        }
        let ph = h / p;
        let pw = w / p;
        let n = ph * pw;
        if pos_x.len() != n as usize || pos_y.len() != n as usize {
            return Err(anyhow!(
                "Gemma4Vision patch positions length mismatch: x={} y={} patches={n}",
                pos_x.len(),
                pos_y.len()
            ));
        }

        let patches = pixel_values.reshape_on(&[b, c, ph, p, pw, p][..], target)?;
        let patches = patches.transpose_axes_on(&[0_i32, 2, 4, 3, 5, 1][..], target)?;
        let patches = patches.reshape_on((b, n, c * p * p), target)?;
        let patches = &(&patches - 0.5_f32) * 2.0_f32;
        let hidden = self.input_proj.forward_on(&patches, target)?;

        let pos_x_arr: Array = (pos_x, &[1_i32, n][..]).try_into()?;
        let pos_y_arr: Array = (pos_y, &[1_i32, n][..]).try_into()?;
        let table_shape = self.position_embedding_table.shape();
        let table_dims = table_shape.as_slice();
        if table_dims != [2, self.position_embedding_size, self.hidden_size] {
            return Err(anyhow!(
                "Gemma4Vision position_embedding_table shape {:?} != [2,{},{}]",
                table_dims,
                self.position_embedding_size,
                self.hidden_size
            ));
        }
        let x_table = mlx::ops::indexing::slice_strided_on(
            &self.position_embedding_table,
            &[0_i32, 0, 0][..],
            &[1_i32, self.position_embedding_size, self.hidden_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?
        .reshape_on((self.position_embedding_size, self.hidden_size), target)?;
        let y_table = mlx::ops::indexing::slice_strided_on(
            &self.position_embedding_table,
            &[1_i32, 0, 0][..],
            &[2_i32, self.position_embedding_size, self.hidden_size][..],
            &[1_i32, 1, 1][..],
            target,
        )?
        .reshape_on((self.position_embedding_size, self.hidden_size), target)?;
        let x_emb = mlx::ops::take_on(&x_table, &pos_x_arr, 0, target)?;
        let y_emb = mlx::ops::take_on(&y_table, &pos_y_arr, 0, target)?;
        Ok(&hidden + &(&x_emb + &y_emb))
    }
}

struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
}

impl VisionAttention {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &Gemma4VisionConfig) -> Result<Self> {
        let clip = cfg.use_clipped_linears;
        Ok(Self {
            q_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.q_proj"), clip)?,
            k_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.k_proj"), clip)?,
            v_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.v_proj"), clip)?,
            o_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.o_proj"), clip)?,
            q_norm: RmsNorm::from_loader(loader, &format!("{prefix}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: RmsNorm::from_loader(loader, &format!("{prefix}.k_norm"), cfg.rms_norm_eps)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward_on(
        &self,
        x: &Array,
        rope: &RopeTables,
        mask: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = x.shape();
        let dims = shape.as_slice();
        let (b, seq) = (dims[0], dims[1]);

        let q = self
            .q_proj
            .forward_on(x, target)?
            .reshape_on((b, seq, self.num_heads, self.head_dim), target)?;
        let k = self
            .k_proj
            .forward_on(x, target)?
            .reshape_on((b, seq, self.num_kv_heads, self.head_dim), target)?;
        let v = self
            .v_proj
            .forward_on(x, target)?
            .reshape_on((b, seq, self.num_kv_heads, self.head_dim), target)?;

        let q = self.q_norm.forward_on(&q, target)?;
        let k = self.k_norm.forward_on(&k, target)?;
        let v = rms_norm_no_scale_on(&v, self.rms_norm_eps, target)?;
        let q = apply_2d_rope_on(&q, rope, target)?;
        let k = apply_2d_rope_on(&k, rope, target)?;

        let q = q.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let k = k.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let v = v.transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?;
        let out =
            mlx::fast::scaled_dot_product_attention_on(&q, &k, &v, 1.0, "", mask, None, target)?;
        let out = out
            .transpose_axes_on(&[0_i32, 2, 1, 3][..], target)?
            .reshape_on((b, seq, self.num_heads * self.head_dim), target)?;
        self.o_proj.forward_on(&out, target)
    }
}

struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl VisionMlp {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &Gemma4VisionConfig) -> Result<Self> {
        let clip = cfg.use_clipped_linears;
        Ok(Self {
            gate_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.gate_proj"), clip)?,
            up_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.up_proj"), clip)?,
            down_proj: ClippableLinear::from_loader(loader, &format!("{prefix}.down_proj"), clip)?,
        })
    }

    fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        let gate = self.gate_proj.forward_on(x, target)?;
        let up = self.up_proj.forward_on(x, target)?;
        let act = gelu_approx_mul_on(&gate, &up, target)?;
        self.down_proj.forward_on(&act, target)
    }
}

struct VisionBlock {
    input_layernorm: RmsNorm,
    self_attn: VisionAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: VisionMlp,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionBlock {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &Gemma4VisionConfig) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_norm_eps,
            )?,
            self_attn: VisionAttention::from_loader(loader, &format!("{prefix}.self_attn"), cfg)?,
            post_attention_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.pre_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            mlp: VisionMlp::from_loader(loader, &format!("{prefix}.mlp"), cfg)?,
            post_feedforward_layernorm: RmsNorm::from_loader(
                loader,
                &format!("{prefix}.post_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
        })
    }

    fn forward_on(
        &self,
        x: &Array,
        rope: &RopeTables,
        mask: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward_on(x, target)?;
        let attn = self.self_attn.forward_on(&normed, rope, mask, target)?;
        let attn = self.post_attention_layernorm.forward_on(&attn, target)?;
        let h = x + &attn;

        let normed = self.pre_feedforward_layernorm.forward_on(&h, target)?;
        let ff = self.mlp.forward_on(&normed, target)?;
        let ff = self.post_feedforward_layernorm.forward_on(&ff, target)?;
        Ok(&h + &ff)
    }
}

pub struct VisionModel {
    cfg: Gemma4VisionConfig,
    patch_embedder: VisionPatchEmbedder,
    layers: Vec<VisionBlock>,
    std_bias: Option<Array>,
    std_scale: Option<Array>,
}

impl VisionModel {
    pub fn from_loader(loader: &Loader, cfg: Gemma4VisionConfig) -> Result<Self> {
        let patch_embedder =
            VisionPatchEmbedder::from_loader(loader, "vision_tower.patch_embedder", &cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers as usize {
            layers.push(VisionBlock::from_loader(
                loader,
                &format!("vision_tower.encoder.layers.{i}"),
                &cfg,
            )?);
        }
        let std_bias = if cfg.standardize {
            loader.tensor_opt("vision_tower.std_bias").cloned()
        } else {
            None
        };
        let std_scale = if cfg.standardize {
            loader.tensor_opt("vision_tower.std_scale").cloned()
        } else {
            None
        };
        Ok(Self {
            cfg,
            patch_embedder,
            layers,
            std_bias,
            std_scale,
        })
    }

    pub fn forward_on(
        &self,
        pixel_values: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();
        let profile = profile_enabled();
        let total_t0 = Instant::now();
        let shape = pixel_values.shape();
        let dims = shape.as_slice();
        if dims.len() != 4 || dims[0] <= 0 || dims[1] != 3 {
            return Err(anyhow!(
                "Gemma4VisionModel supports image batches [B,3,H,W], got {dims:?}"
            ));
        }
        let (h, w) = (dims[2], dims[3]);
        let t0 = Instant::now();
        let (positions, pos_x, pos_y, padding) = patch_positions(h, w, &self.cfg)?;
        let num_real = pos_x.len() as i32;
        let max_patches = self.cfg.max_patches();
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] vision_positions_ms={:.3} real_patches={} max_patches={}",
                t0.elapsed().as_secs_f64() * 1000.0,
                num_real,
                max_patches
            );
        }

        let t0 = Instant::now();
        let mut hidden = self
            .patch_embedder
            .forward_on(pixel_values, &pos_x, &pos_y, target)?;
        let mask: Option<Array> = None;
        profile_eval("vision_patch_embed", &[&hidden], t0, profile)?;

        let t0 = Instant::now();
        let rope = build_rope_tables(
            &positions,
            self.cfg.head_dim,
            self.cfg.rope_theta(),
            hidden.dtype(),
            target,
        )?;
        let rope_arrays = rope.arrays();
        profile_eval("vision_rope_tables", &rope_arrays, t0, profile)?;

        let t0 = Instant::now();
        for layer in &self.layers {
            hidden = layer.forward_on(&hidden, &rope, mask.as_ref(), target)?;
        }
        profile_eval("vision_layers", &[&hidden], t0, profile)?;
        let t0 = Instant::now();
        let mut pooled = self.pool_on(&hidden, &positions, &padding, target)?;
        if let (Some(bias), Some(scale)) = (&self.std_bias, &self.std_scale) {
            pooled = (&pooled - bias) * scale;
        }
        profile_eval("vision_pool", &[&pooled], t0, profile)?;
        if profile {
            tracing::info!(
                "[gemma4-vl-profile] vision_total_ms={:.3}",
                total_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(pooled)
    }

    fn pool_on(
        &self,
        hidden: &Array,
        positions: &[(i32, i32)],
        padding: &[bool],
        target: StreamOrDevice,
    ) -> Result<Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        let (b, seq, h) = (dims[0], dims[1], dims[2]);
        let hidden = if padding.iter().any(|&p| p) {
            let zero = Array::zeros_on((b, seq, h), hidden.dtype(), target)?;
            let keep: Vec<bool> = padding.iter().map(|p| !*p).collect();
            let keep_arr: Array = (keep.as_slice(), &[1_i32, seq, 1][..]).try_into()?;
            mlx::ops::where_on(&keep_arr, hidden, &zero, target)?
        } else {
            hidden.clone()
        };

        let k = self.cfg.pooling_kernel_size;
        if k <= 0 {
            return Err(anyhow!("Gemma4Vision pooler invalid kernel {k}"));
        }
        let length = self.cfg.default_output_length;
        let max_x = positions
            .iter()
            .map(|(x, _)| (*x).max(0))
            .max()
            .unwrap_or(0)
            + 1;
        let buckets_x = (max_x / k).max(1);
        let mut weights = vec![0.0_f32; (length * seq) as usize];
        let mut valid = vec![false; length as usize];
        for (idx, (x, y)) in positions.iter().enumerate() {
            if padding[idx] {
                continue;
            }
            let bx = (*x).max(0) / k;
            let by = (*y).max(0) / k;
            let bucket = bx + buckets_x * by;
            if bucket >= 0 && bucket < length {
                weights[(bucket * seq + idx as i32) as usize] = 1.0 / (k * k) as f32;
                valid[bucket as usize] = true;
            }
        }
        let valid_count = valid.iter().filter(|&&v| v).count() as i32;
        if valid_count <= 0 {
            return Err(anyhow!("Gemma4Vision pooler produced no valid soft tokens"));
        }

        let weights_arr: Array = (weights.as_slice(), &[length, seq][..]).try_into()?;
        let mut rows = Vec::with_capacity(b as usize);
        for row in 0..b {
            let hidden_row = mlx::ops::indexing::slice_strided_on(
                &hidden,
                &[row, 0, 0][..],
                &[row + 1, seq, h][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            let hidden_2d = hidden_row.reshape_on((seq, h), target)?;
            let pooled = weights_arr.matmul_on(&hidden_2d, target)?;
            let pooled = &pooled * (self.cfg.hidden_size as f32).sqrt();
            let pooled = pooled.reshape_on((1_i32, length, h), target)?;
            rows.push(mlx::ops::indexing::slice_strided_on(
                &pooled,
                &[0_i32, 0, 0][..],
                &[1_i32, valid_count, h][..],
                &[1_i32, 1, 1][..],
                target,
            )?);
        }
        if rows.len() == 1 {
            Ok(rows.pop().expect("len checked"))
        } else {
            let refs: Vec<&Array> = rows.iter().collect();
            Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
        }
    }
}

type PatchPositions = (Vec<(i32, i32)>, Vec<u32>, Vec<u32>, Vec<bool>);

fn profile_enabled() -> bool {
    std::env::var_os("IRONMLX_GEMMA4_VL_PROFILE").is_some()
}

fn profile_eval(label: &str, arrays: &[&Array], start: Instant, enabled: bool) -> Result<()> {
    if enabled {
        mlx::transforms::eval(arrays)?;
        tracing::info!(
            "[gemma4-vl-profile] {label}_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

fn patch_positions(h: i32, w: i32, cfg: &Gemma4VisionConfig) -> Result<PatchPositions> {
    let ph = h / cfg.patch_size;
    let pw = w / cfg.patch_size;
    let num_real = ph * pw;
    let max_patches = cfg.max_patches();
    if num_real > max_patches {
        return Err(anyhow!(
            "Gemma4Vision patch count {num_real} exceeds max_patches {max_patches}"
        ));
    }
    let mut positions = Vec::with_capacity(num_real as usize);
    let mut pos_x = Vec::with_capacity(num_real as usize);
    let mut pos_y = Vec::with_capacity(num_real as usize);
    let mut padding = Vec::with_capacity(num_real as usize);
    for y in 0..ph {
        for x in 0..pw {
            positions.push((x, y));
            pos_x.push(x as u32);
            pos_y.push(y as u32);
            padding.push(false);
        }
    }
    Ok((positions, pos_x, pos_y, padding))
}

struct RopeTables {
    cos: Vec<Array>,
    sin: Vec<Array>,
    channels_per_dim: i32,
}

impl RopeTables {
    fn arrays(&self) -> Vec<&Array> {
        self.cos.iter().chain(self.sin.iter()).collect()
    }
}

fn build_rope_tables(
    positions: &[(i32, i32)],
    head_dim: i32,
    base: f32,
    dtype: Dtype,
    target: StreamOrDevice,
) -> Result<RopeTables> {
    let ndim = 2;
    let channels_per_dim = 2 * (head_dim / (2 * ndim));
    if channels_per_dim <= 0 || channels_per_dim * ndim != head_dim {
        return Err(anyhow!(
            "Gemma4Vision RoPE unsupported head_dim {head_dim} for 2D split"
        ));
    }

    let seq = i32::try_from(positions.len()).context("Gemma4Vision RoPE seq length overflow")?;
    let half = channels_per_dim / 2;
    let mut cos_tables = Vec::with_capacity(ndim as usize);
    let mut sin_tables = Vec::with_capacity(ndim as usize);
    for d in 0..ndim {
        let mut cos = vec![0.0_f32; (seq * channels_per_dim) as usize];
        let mut sin = vec![0.0_f32; (seq * channels_per_dim) as usize];
        for (i, (x_pos, y_pos)) in positions.iter().enumerate() {
            let pos = if d == 0 { *x_pos } else { *y_pos };
            for j in 0..half {
                let exponent = (2.0_f32 / channels_per_dim as f32) * j as f32;
                let timescale = base.powf(exponent);
                let value = pos as f32 / timescale;
                let c = value.cos();
                let s = value.sin();
                let base_idx = i * channels_per_dim as usize;
                cos[base_idx + j as usize] = c;
                sin[base_idx + j as usize] = s;
                cos[base_idx + (half + j) as usize] = c;
                sin[base_idx + (half + j) as usize] = s;
            }
        }
        let cos_arr: Array = (cos.as_slice(), &[1_i32, seq, 1, channels_per_dim][..]).try_into()?;
        let sin_arr: Array = (sin.as_slice(), &[1_i32, seq, 1, channels_per_dim][..]).try_into()?;
        cos_tables.push(mlx::ops::astype_on(&cos_arr, dtype, target)?);
        sin_tables.push(mlx::ops::astype_on(&sin_arr, dtype, target)?);
    }
    Ok(RopeTables {
        cos: cos_tables,
        sin: sin_tables,
        channels_per_dim,
    })
}

fn apply_2d_rope_on(x: &Array, rope: &RopeTables, target: StreamOrDevice) -> Result<Array> {
    let shape = x.shape();
    let dims = shape.as_slice();
    let (_b, seq, _heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    let ndim = 2;
    let channels_per_dim = rope.channels_per_dim;
    if channels_per_dim <= 0 || channels_per_dim * ndim != head_dim {
        return Err(anyhow!(
            "Gemma4Vision RoPE unsupported head_dim {head_dim} for 2D split"
        ));
    }

    let mut parts = Vec::with_capacity(ndim as usize);
    for d in 0..ndim {
        let start = d * channels_per_dim;
        let end = start + channels_per_dim;
        let part = mlx::ops::indexing::slice_strided_on(
            x,
            &[0_i32, 0, 0, start][..],
            &[dims[0], dims[1], dims[2], end][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let half = channels_per_dim / 2;
        let first = mlx::ops::indexing::slice_strided_on(
            &part,
            &[0_i32, 0, 0, 0][..],
            &[dims[0], dims[1], dims[2], half][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let second = mlx::ops::indexing::slice_strided_on(
            &part,
            &[0_i32, 0, 0, half][..],
            &[dims[0], dims[1], dims[2], channels_per_dim][..],
            &[1_i32, 1, 1, 1][..],
            target,
        )?;
        let neg_second = -&second;
        let rotated = mlx::ops::shape::concatenate_on(&[&neg_second, &first], -1, target)?;

        let cos_arr = &rope.cos[d as usize];
        let sin_arr = &rope.sin[d as usize];
        if cos_arr.shape_at(1) != seq || sin_arr.shape_at(1) != seq {
            return Err(anyhow!(
                "Gemma4Vision RoPE table seq mismatch: table={} input={seq}",
                cos_arr.shape_at(1)
            ));
        }
        parts.push(&(&part * cos_arr) + &(&rotated * sin_arr));
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(mlx::ops::shape::concatenate_on(&refs, -1, target)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Gemma4VisionConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "gemma4_vision",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "patch_size": 16,
            "default_output_length": 280,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240
        }))
        .unwrap()
    }

    #[test]
    fn patch_positions_track_real_patches() {
        let cfg = cfg();
        let (positions, pos_x, pos_y, padding) = patch_positions(48, 48, &cfg).unwrap();
        assert_eq!(pos_x.len(), 9);
        assert_eq!(pos_y.len(), 9);
        assert_eq!(positions.len(), 9);
        assert_eq!(padding.iter().filter(|&&p| !p).count(), 9);
    }
}
