//! Qwen3.5 / Qwen3-Next GatedDeltaNet — recurrent SSM with delta rule.
//!
//! T5: the `gated_delta_step` metal_kernel (2 variants — no-mask + masked).
//! T6: the `GatedDeltaNet` main struct wiring all components.
//!
//! Mirrors mlx-lm's `_make_gated_delta_kernel(has_mask)` from
//! `/Volumes/Dev/mlx-lm/mlx_lm/models/gated_delta.py:13-115`.
//!
//! Templates: `Dk, Dv, Hk, Hv` (i32), `InT, StT` (Dtype).
//! Grid: `(32, Dv, B * Hv)`; threadgroup: `(32, 4, 1)`.

use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::ops::shape::concatenate;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::core::cache::GatedDeltaCache;
use crate::core::Loader;
use crate::nn::{Conv1d, Conv1dConfig, Linear, RmsNormGated};
use crate::Result;

/// Configuration for [`GatedDeltaNet`].
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNetConfig {
    /// Total K-side dim: `num_k_heads × head_k_dim`. Used to size q/k slices
    /// of the qkv projection output and as the K-side stride in the kernel.
    pub fn key_dim(&self) -> i32 {
        self.num_k_heads * self.head_k_dim
    }

    /// Total V-side dim: `num_v_heads × head_v_dim`. Equals the inner dim of
    /// the V projection and the input dim of `out_proj`.
    pub fn value_dim(&self) -> i32 {
        self.num_v_heads * self.head_v_dim
    }

    /// Total projection-output dim for `in_proj_qkv`:
    /// `key_dim × 2 + value_dim` — i.e. concatenated Q + K + V output.
    /// Also the channel count for the depthwise `conv1d`.
    pub fn conv_dim(&self) -> i32 {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Qwen3.5 / Qwen3-Next "linear attention" branch — recurrent SSM with
/// delta rule and scalar gating.
///
/// Mirrors mlx-lm's `Qwen3NextGatedDeltaNet`
/// (`/Volumes/Dev/mlx-lm/mlx_lm/models/qwen3_5.py:85-205`). Components:
///
/// - `in_proj_qkvz` — fused matmul (Q+K+V outputs concat'd with the gate
///   `z`); equivalent to mlx-lm's `in_proj_qkvz`. Sliced in `forward_on`.
/// - `in_proj_ba` — fused matmul (forget signal `b` + decay signal `a`);
///   equivalent to mlx-lm's `in_proj_ba`. Sliced in `forward_on`.
/// - `conv1d` — depthwise temporal mixing across the Q/K/V channels (then
///   silu via module-level fused compile cell)
/// - `norm` — `RmsNormGated`: `silu(z) * rms_norm(y)` final mixing
/// - `out_proj` — back to `hidden_size`
/// - `a_log` / `dt_bias` — per-head learned parameters for compute_g
pub struct GatedDeltaNet {
    /// Fused (qkv, z) input projection — concatenated along axis 0 at load
    /// time. Output `[B, S, conv_dim + value_dim]`; sliced in `forward_on`.
    in_proj_qkvz: Linear,
    /// Fused (b, a) input projection — concatenated along axis 0 at load
    /// time. Output `[B, S, num_v_heads * 2]`; sliced in `forward_on`.
    in_proj_ba: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,   // [num_v_heads]
    dt_bias: Array, // [num_v_heads]
    cfg: GatedDeltaNetConfig,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
}

impl GatedDeltaNet {
    /// Production constructor: load all weight tensors + a_log + dt_bias,
    /// fusing in_proj_qkv+z → in_proj_qkvz and in_proj_b+a → in_proj_ba
    /// at load time via axis-0 concatenation.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self> {
        let qmeta = loader.quant_meta().ok_or_else(|| {
            anyhow!("{prefix}: GatedDeltaNet input projections require quantized loader")
        })?;

        // Fuse in_proj_qkv + in_proj_z → in_proj_qkvz (output axis 0).
        let qkv_w = loader
            .tensor(&format!("{prefix}.in_proj_qkv.weight"))?
            .clone();
        let qkv_s = loader
            .tensor(&format!("{prefix}.in_proj_qkv.scales"))?
            .clone();
        let qkv_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_qkv.biases"))
            .cloned();
        let z_w = loader
            .tensor(&format!("{prefix}.in_proj_z.weight"))?
            .clone();
        let z_s = loader
            .tensor(&format!("{prefix}.in_proj_z.scales"))?
            .clone();
        let z_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_z.biases"))
            .cloned();

        let qkvz_weight = mlx::ops::shape::concatenate(&[&qkv_w, &z_w], 0)?;
        let qkvz_scales = mlx::ops::shape::concatenate(&[&qkv_s, &z_s], 0)?;
        let qkvz_biases = match (qkv_b_opt, z_b_opt) {
            (Some(a), Some(b)) => Some(mlx::ops::shape::concatenate(&[&a, &b], 0)?),
            (None, None) => None,
            _ => {
                return Err(anyhow!(
                    "{prefix}: in_proj_qkv.biases and in_proj_z.biases must agree on Some/None"
                ));
            }
        };

        // Eagerly evaluate the fused qkvz tensors on the loading thread so that
        // no lazy stream-tagged computation escapes into model fields that will
        // be read from other threads (e.g. tokio blocking-pool during inference).
        // MLX's CommandEncoder map is thread_local; a lazy Array whose primitive
        // carries Stream(gpu, N) will panic with "There is no Stream(gpu, N) in
        // current thread" when gpu::eval is called on a thread that never called
        // gpu::new_stream(N). The eager eval here materialises the concatenated
        // tensors so that only plain data buffers (no primitives) are stored.
        {
            let mut to_eval: Vec<&Array> = vec![&qkvz_weight, &qkvz_scales];
            if let Some(b) = &qkvz_biases {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval)
                .map_err(|e| anyhow!("{prefix}: eager eval of fused qkvz tensors failed: {e}"))?;
        }

        let in_proj_qkvz = Linear::new_quant(
            qkvz_weight,
            qkvz_scales,
            qkvz_biases,
            None,
            qmeta.group_size,
            qmeta.bits,
        );

        // Fuse in_proj_b + in_proj_a → in_proj_ba (b first, a second).
        let b_w = loader
            .tensor(&format!("{prefix}.in_proj_b.weight"))?
            .clone();
        let b_s = loader
            .tensor(&format!("{prefix}.in_proj_b.scales"))?
            .clone();
        let b_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_b.biases"))
            .cloned();
        let a_w = loader
            .tensor(&format!("{prefix}.in_proj_a.weight"))?
            .clone();
        let a_s = loader
            .tensor(&format!("{prefix}.in_proj_a.scales"))?
            .clone();
        let a_b_opt = loader
            .tensor_opt(&format!("{prefix}.in_proj_a.biases"))
            .cloned();

        let ba_weight = mlx::ops::shape::concatenate(&[&b_w, &a_w], 0)?;
        let ba_scales = mlx::ops::shape::concatenate(&[&b_s, &a_s], 0)?;
        let ba_biases = match (b_b_opt, a_b_opt) {
            (Some(p), Some(q)) => Some(mlx::ops::shape::concatenate(&[&p, &q], 0)?),
            (None, None) => None,
            _ => {
                return Err(anyhow!(
                    "{prefix}: in_proj_b.biases and in_proj_a.biases must agree on Some/None"
                ));
            }
        };

        // Same thread-crossing guard as above: eval fused ba tensors before
        // storing them so no lazy Stream(gpu, N) primitives escape to the
        // blocking-pool inference thread.
        {
            let mut to_eval: Vec<&Array> = vec![&ba_weight, &ba_scales];
            if let Some(b) = &ba_biases {
                to_eval.push(b);
            }
            mlx::transforms::eval(&to_eval)
                .map_err(|e| anyhow!("{prefix}: eager eval of fused ba tensors failed: {e}"))?;
        }

        let in_proj_ba = Linear::new_quant(
            ba_weight,
            ba_scales,
            ba_biases,
            None,
            qmeta.group_size,
            qmeta.bits,
        );

        let conv1d_cfg = Conv1dConfig {
            in_channels: cfg.conv_dim(),
            out_channels: cfg.conv_dim(),
            kernel_size: cfg.conv_kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: cfg.conv_dim(),
        };
        let conv1d = Conv1d::from_loader(loader, &format!("{prefix}.conv1d"), conv1d_cfg)?;
        let norm = RmsNormGated::from_loader(loader, &format!("{prefix}.norm"), cfg.rms_norm_eps)?;
        let out_proj = Linear::from_loader(loader, &format!("{prefix}.out_proj"))?;
        let a_log = loader.tensor(&format!("{prefix}.A_log"))?.clone();
        let dt_bias = loader.tensor(&format!("{prefix}.dt_bias"))?.clone();

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
        })
    }

    /// Test/composition seam: build from pre-built nn building blocks.
    ///
    /// `in_proj_qkvz` and `in_proj_ba` must already be the fused forms
    /// (output dim concatenated along axis 0). For tests that build
    /// separate qkv/z/a/b Linears, concat the underlying weights via
    /// `mlx::ops::shape::concatenate` first then pass a single fused
    /// Linear here.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        in_proj_qkvz: Linear,
        in_proj_ba: Linear,
        conv1d: Conv1d,
        norm: RmsNormGated,
        out_proj: Linear,
        a_log: Array,
        dt_bias: Array,
        cfg: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
        }
    }

    pub fn config(&self) -> &GatedDeltaNetConfig {
        &self.cfg
    }

    /// Forward pass with default stream.
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> Result<Array> {
        self.forward_on(x, mask, cache, ())
    }

    /// Stream-targeted forward — Qwen3-Next gated delta net algorithm.
    ///
    /// 8 steps:
    ///   1. project qkv, z, a, b
    ///   2. conv1d + silu (with conv_state from cache prepended; cache update)
    ///   3. split + reshape per-head
    ///   4. q/k rms_norm (no weight) + scale
    ///   5. compute_g via mlx::compile
    ///   6. beta = sigmoid(b)
    ///   7. dispatch gated_delta_step kernel + update recurrent cache + advance offset
    ///   8. RmsNormGated(y, z) + reshape + out_proj
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        let target = target.into();

        // Pre-flight validation. Match P3b1 Mrope's "explicit bounds > trust caller"
        // pattern — surface common misuses at the source rather than as a downstream
        // shape/MTL dispatch error.
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: x must be rank-3 [B, S, hidden]; got rank {}",
                dims.len()
            ));
        }
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "GatedDeltaNet::forward: x.last_dim={} != hidden_size={}",
                dims[2],
                self.cfg.hidden_size
            ));
        }
        if self.cfg.head_k_dim < 32 || self.cfg.head_k_dim % 32 != 0 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: head_k_dim={} must be a positive multiple of 32 \
                 (Metal kernel requires `n_per_t = Dk/32 >= 1` and full simdgroup coverage)",
                self.cfg.head_k_dim
            ));
        }
        if self.cfg.num_k_heads == 0 || self.cfg.num_v_heads % self.cfg.num_k_heads != 0 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: num_v_heads ({}) must be divisible by num_k_heads ({}) \
                 — kernel uses `hk_idx = hv_idx / (Hv/Hk)` for GQA indexing",
                self.cfg.num_v_heads,
                self.cfg.num_k_heads
            ));
        }

        let batch = dims[0];
        let seq = dims[1];

        // Step 1: fused projections + slice (was 4 quantized matmuls; now 2).
        let qkvz = self.in_proj_qkvz.forward_on(x, target)?;
        let ba = self.in_proj_ba.forward_on(x, target)?;

        let conv_dim = self.cfg.conv_dim();
        let value_dim = self.cfg.value_dim();
        let num_v_heads = self.cfg.num_v_heads;

        let qkv = mlx::ops::indexing::slice_strided(
            &qkvz,
            &[0_i32, 0, 0][..],
            &[batch, seq, conv_dim][..],
            &[1_i32, 1, 1][..],
        )?;
        let z = mlx::ops::indexing::slice_strided(
            &qkvz,
            &[0_i32, 0, conv_dim][..],
            &[batch, seq, conv_dim + value_dim][..],
            &[1_i32, 1, 1][..],
        )?;

        // Step 1b: zero out qkv at pad positions before conv1d.
        //
        // The conv1d is temporal — its output at real-token position t uses
        // input positions `t-(k-1)..t` as history. For left-padded batched
        // prefill, those history positions include pad tokens whose embeddings
        // are non-zero garbage (embed(pad_id) projected through in_proj_qkvz).
        // If we leave qkv as-is at pad positions, conv1d at the first few real
        // positions sees pad-embedding history and produces outputs that
        // diverge from the per-stream reference (which sees zero history).
        //
        // The gated_delta_step kernel's per-token mask only skips compute at
        // pad positions; it does not undo conv1d contamination of real
        // positions. Zeroing qkv at pad positions before conv1d gives real
        // tokens the same zero-history as per-stream forward_on.
        //
        // The same argument applies to `z` (used in RmsNormGated at output);
        // however, `z` is only consumed at REAL positions (gated_delta_step
        // emits zero at pad positions), so pad-position `z` values are
        // discarded anyway. We zero `qkv` only.
        let qkv = if let Some(m) = mask {
            let m_dtype = mlx::ops::cast::astype(m, qkv.dtype())?;
            let m_broadcast = m_dtype.reshape_on((batch, seq, 1), target)?;
            &qkv * &m_broadcast
        } else {
            qkv
        };

        let b = mlx::ops::indexing::slice_strided(
            &ba,
            &[0_i32, 0, 0][..],
            &[batch, seq, num_v_heads][..],
            &[1_i32, 1, 1][..],
        )?;
        let a = mlx::ops::indexing::slice_strided(
            &ba,
            &[0_i32, 0, num_v_heads][..],
            &[batch, seq, num_v_heads + num_v_heads][..],
            &[1_i32, 1, 1][..],
        )?;

        // Step 2a: prepend conv_state
        let conv_input = match cache.as_deref_mut() {
            Some(c) => concatenate(&[c.conv_state(), &qkv], 1)?,
            None => {
                // Synthesize a fresh zero conv_state of shape [B, kernel_size-1, conv_dim].
                let zeros = Array::zeros(
                    (batch, self.cfg.conv_kernel_size - 1, self.cfg.conv_dim()),
                    qkv.dtype(),
                )?;
                concatenate(&[&zeros, &qkv], 1)?
            }
        };

        // Step 2b: conv1d + silu
        let conv_out = self.conv1d.forward_on(&conv_input, target)?;
        let conv_sig = conv_out.sigmoid()?;
        let conv_out = &conv_out * &conv_sig;

        // Step 2c: update conv_state cache (last kernel_size-1 tokens of conv_input)
        if let Some(c) = cache.as_deref_mut() {
            let n_keep = self.cfg.conv_kernel_size - 1;
            // slice last n_keep tokens along axis=1
            let conv_input_dims = conv_input.shape();
            let total_len = conv_input_dims.as_slice()[1];
            let new_conv_state = mlx::ops::indexing::slice(
                &conv_input,
                vec![0_i32, total_len - n_keep, 0].as_slice(),
                vec![batch, total_len, self.cfg.conv_dim()].as_slice(),
            )?;
            c.update_conv(new_conv_state);
        }

        // Step 3: split + reshape per-head
        // conv_out shape: [B, S, conv_dim] = [B, S, key_dim*2 + value_dim]
        // Split at [key_dim, 2*key_dim] → 3 segments [B, S, key_dim], [B, S, key_dim], [B, S, value_dim]
        let split_at = vec![self.cfg.key_dim(), 2 * self.cfg.key_dim()];
        let parts = mlx::ops::shape::split_at_on(&conv_out, &split_at, -1, target)?;
        let q_flat = &parts[0]; // [B, S, num_k_heads * head_k_dim]
        let k_flat = &parts[1]; // [B, S, num_k_heads * head_k_dim]
        let v_flat = &parts[2]; // [B, S, num_v_heads * head_v_dim]

        let q_per_head = q_flat.reshape_on(
            (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
            target,
        )?;
        let k_per_head = k_flat.reshape_on(
            (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
            target,
        )?;
        let v_per_head = v_flat.reshape_on(
            (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
            target,
        )?;

        // Step 4: q/k rms_norm (no weight)
        let inv_scale = 1.0_f32 / (self.cfg.head_k_dim as f32).sqrt();
        let q_normed = mlx::fast::rms_norm_on(&q_per_head, None, 1e-6, target)?;
        let q_scaled = &q_normed * (inv_scale * inv_scale); // panic-on-err, no `?`
        let k_normed = mlx::fast::rms_norm_on(&k_per_head, None, 1e-6, target)?;
        let k_scaled = &k_normed * inv_scale; // panic-on-err, no `?`

        // Step 5: compute_g = exp(-exp(A_log) * softplus(a + dt_bias))
        // softplus stabilised: where(x > 20, x, log(1 + exp(x)))
        let x = &a + &self.dt_bias;
        let twenty: Array = (&[20.0_f32][..], ()).try_into()?;
        let zeros = a.zeros_like()?;
        let safe = zeros.logaddexp(&x)?;
        let cond = x.greater(&twenty)?;
        let sp = cond.where_(&x, &safe)?;
        let a_log_f32 = mlx::ops::cast::astype(&self.a_log, Dtype::Float32)?;
        let exp_alog = a_log_f32.exp()?;
        let neg_exp_alog = mlx::ops::binary::negative(&exp_alog)?;
        let inner = &neg_exp_alog * &sp;
        let g = inner.exp()?;

        // Step 6: beta = sigmoid(b)
        let beta = b.sigmoid_on(target)?;

        // Step 7a: build/get the appropriate kernel
        let kernel = if mask.is_some() {
            self.kernel_masked
                .get_or_init(|| build_gated_delta_kernel(true).expect("build masked kernel"))
        } else {
            self.kernel_no_mask
                .get_or_init(|| build_gated_delta_kernel(false).expect("build no-mask kernel"))
        };

        // Step 7b: get state_in from cache (or fresh zeros).
        // Note: `Array::clone()` is cheap (Arc-share refcount inc on `array_desc_`,
        // not a deep memory copy); the kernel dispatch needs an `&Array`, and
        // the cache must keep its slot for `update_recurrent` later.
        let state_in = match cache.as_deref() {
            Some(c) => c.recurrent_state().clone(),
            None => Array::zeros(
                (
                    batch,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                    self.cfg.head_k_dim,
                ),
                Dtype::Float32,
            )?,
        };

        // Step 7c: T as 0-dim int32 array
        let t_arr: Array = (&[seq][..], ()).try_into()?;

        let in_dtype = x.dtype();
        let st_dtype = Dtype::Float32;
        let y_shape = Shape::from(vec![batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim]);
        let state_shape = Shape::from(vec![
            batch,
            self.cfg.num_v_heads,
            self.cfg.head_v_dim,
            self.cfg.head_k_dim,
        ]);

        // Step 7d: dispatch
        let mut kernel_inputs: Vec<&Array> = vec![
            &q_scaled,
            &k_scaled,
            &v_per_head,
            &g,
            &beta,
            &state_in,
            &t_arr,
        ];
        if let Some(m) = mask {
            kernel_inputs.push(m);
        }

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&kernel_inputs)
            .output_shapes(&[y_shape, state_shape])
            .output_dtypes(&[in_dtype, st_dtype])
            .grid(32, self.cfg.head_v_dim, batch * self.cfg.num_v_heads)
            .threadgroup(32, 4, 1)
            .template_int("Dk", self.cfg.head_k_dim)
            .template_int("Dv", self.cfg.head_v_dim)
            .template_int("Hk", self.cfg.num_k_heads)
            .template_int("Hv", self.cfg.num_v_heads)
            .template_dtype("InT", in_dtype)
            .template_dtype("StT", st_dtype)
            .stream(target)
            .dispatch()?;

        let y = outputs.take_at(0)?; // [B, S, Hv, Dv]
        let new_state = outputs.take_at(0)?; // [B, Hv, Dv, Dk]

        // Step 7e: update cache recurrent_state, advance offset
        if let Some(c) = cache {
            c.update_recurrent(new_state);
            // TEMP(b1-p2.3c-1 Task 2): uniform per-row n = seq across all B
            // rows — replaced in Task 4 by caller-provided per_row_lens.
            let per_row_n = vec![seq; batch as usize];
            c.advance(&per_row_n)?;
        }

        // Step 8: RmsNormGated(y, z) + reshape + out_proj
        let z_per_head = z.reshape_on(
            (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
            target,
        )?;
        let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
        let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
        self.out_proj.forward_on(&normed_flat, target)
    }
}

/// Build the `gated_delta_step` MetalKernel (no-mask or masked variant).
///
/// The shader source is identical between variants except for the per-token
/// guard expression (`mask_clause`). MLX's `metal_kernel` machinery auto-injects
/// `<name>_shape` / `<name>_strides` / `<name>_ndim` for input arrays referenced
/// in the source.
///
/// `T` is passed as a 0-dim int32 array, which MLX treats as `device const
/// int32_t& T` — usable directly as an integer in the shader (e.g.
/// `for (int t = 0; t < T; ++t)`).
pub(crate) fn build_gated_delta_kernel(masked: bool) -> Result<MetalKernel> {
    let mask_clause = if masked {
        "mask[b_idx * T + t]"
    } else {
        "true"
    };
    let src = format!(
        r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_clause}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            // Note: all 32 simdgroup threads write the same zero value here
            // (no `thread_index_in_simdgroup == 0` guard). Matches mlx-lm
            // reference exactly; wasted write bandwidth is acceptable since
            // masked tokens are rare and all writes are identical.
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Advance pointers to the next time step.
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
        "#,
        mask_clause = mask_clause
    );

    let name = if masked {
        "ironmlx_gated_delta_masked"
    } else {
        "ironmlx_gated_delta"
    };

    let inputs: &[&str] = if masked {
        &["q", "k", "v", "g", "beta", "state_in", "T", "mask"]
    } else {
        &["q", "k", "v", "g", "beta", "state_in", "T"]
    };

    Ok(MetalKernel::builder(name)
        .inputs(inputs)
        .outputs(&["y", "state_out"])
        .source(&src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype, Shape};

    fn small_gdn_components() -> GatedDeltaNet {
        // Synthetic small model:
        // hidden=32, num_v_heads=4, num_k_heads=2, head_k_dim=32, head_v_dim=8,
        // conv_kernel=4, eps=1e-6
        // NOTE: head_k_dim must be >= 32 so n_per_t = Dk/32 >= 1 (Metal C++ forbids
        // zero-length arrays). Using head_k_dim=32, head_v_dim=8.
        let cfg = GatedDeltaNetConfig {
            hidden_size: 32,
            num_v_heads: 4,
            num_k_heads: 2,
            head_k_dim: 32,
            head_v_dim: 8,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        };
        // key_dim = 2*32 = 64; value_dim = 4*8 = 32
        // qkv proj output = key_dim*2 + value_dim = 64+64+32 = 160
        // conv_dim = key_dim*2 + value_dim = 160
        // out_proj input = value_dim = 32
        let conv_dim = cfg.conv_dim(); // 160
        let value_dim = cfg.value_dim(); // 32

        let qkv_w = Array::zeros((conv_dim, 32), Dtype::Float32).unwrap();
        let z_w = Array::zeros((value_dim, 32), Dtype::Float32).unwrap();
        let b_w = Array::zeros((cfg.num_v_heads, 32), Dtype::Float32).unwrap();
        let a_w = Array::zeros((cfg.num_v_heads, 32), Dtype::Float32).unwrap();
        let conv_w = Array::zeros((conv_dim, cfg.conv_kernel_size, 1), Dtype::Float32).unwrap();
        let norm_w = mlx::ops::constructors::ones((cfg.head_v_dim,), Dtype::Float32).unwrap();
        let out_w = Array::zeros((32_i32, value_dim), Dtype::Float32).unwrap();
        let a_log = Array::zeros((cfg.num_v_heads,), Dtype::Float32).unwrap();
        let dt_bias = mlx::ops::constructors::ones((cfg.num_v_heads,), Dtype::Float32).unwrap();

        // Fuse qkv+z → qkvz and b+a → ba along axis 0.
        let qkvz_w = concatenate(&[&qkv_w, &z_w], 0).unwrap();
        let ba_w = concatenate(&[&b_w, &a_w], 0).unwrap();

        GatedDeltaNet::from_components(
            crate::nn::Linear::new_fp(qkvz_w, None),
            crate::nn::Linear::new_fp(ba_w, None),
            crate::nn::Conv1d::new(
                conv_w,
                None,
                crate::nn::Conv1dConfig {
                    in_channels: conv_dim,
                    out_channels: conv_dim,
                    kernel_size: cfg.conv_kernel_size,
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: conv_dim, // depthwise
                },
            ),
            crate::nn::RmsNormGated::new(norm_w, cfg.rms_norm_eps),
            crate::nn::Linear::new_fp(out_w, None),
            a_log,
            dt_bias,
            cfg,
        )
    }

    #[test]
    fn gdn_construction_carries_config() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        assert_eq!(cfg.num_v_heads, 4);
        assert_eq!(cfg.num_k_heads, 2);
        assert_eq!(cfg.conv_kernel_size, 4);
    }

    #[test]
    fn gdn_forward_shape_dtype_no_cache() {
        let gdn = small_gdn_components();
        // x: [B=1, S=4, hidden=32] — note: small zeros so the SSM dispatch
        // succeeds even with our trivial weights.
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let out = gdn.forward(&x, None, None).expect("forward no cache");
        // out_proj maps value_dim=32 -> 32
        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        // dtype may be promoted to fp32 by rms_norm path; just verify finite
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
    }

    #[test]
    fn gdn_forward_with_cache_advances_offset() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        let mut cache = GatedDeltaCache::new_with_cap(
            1, // B
            cfg.conv_kernel_size,
            cfg.conv_dim(),
            cfg.num_v_heads,
            cfg.head_v_dim,
            cfg.head_k_dim,
            Dtype::Float32,
            16, // cap
        )
        .expect("cache");
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let _out = gdn
            .forward(&x, None, Some(&mut cache))
            .expect("forward with cache");
        assert_eq!(cache.offsets(), &[4]);
    }

    #[test]
    fn gated_delta_step_kernel_links() {
        // Dk must be >= 32 so that n_per_t = Dk/32 >= 1 (Metal C++ forbids
        // zero-length arrays). Use Dk=32, Dv=8, Hk=Hv=1, B=1, T=1.
        let kernel = build_gated_delta_kernel(false).expect("build kernel");

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 32), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch");

        let _y = outputs.take_at(0).expect("y");
        let _state = outputs.take_at(0).expect("state");
    }

    #[test]
    fn qkvz_concat_load_matches_separate_matmuls() {
        // Use Linear::new_fp (fp32, no quant) so we can exercise the concat
        // logic without needing a real quantized fixture. The concat math is
        // identical for fp and quantized weights along axis 0.
        let hidden_size = 4_i32;
        let qkv_out = 6_i32;
        let z_out = 4_i32;

        let x_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.1).collect();
        let x: Array = (x_data.as_slice(), &[1_i32, 1, hidden_size][..])
            .try_into()
            .unwrap();

        let w_qkv_data: Vec<f32> = (0..qkv_out * hidden_size)
            .map(|i| (i as f32) * 0.01 - 0.05)
            .collect();
        let w_z_data: Vec<f32> = (0..z_out * hidden_size)
            .map(|i| (i as f32) * 0.02 + 0.03)
            .collect();
        let w_qkv: Array = (w_qkv_data.as_slice(), &[qkv_out, hidden_size][..])
            .try_into()
            .unwrap();
        let w_z: Array = (w_z_data.as_slice(), &[z_out, hidden_size][..])
            .try_into()
            .unwrap();

        let lin_qkv = crate::nn::Linear::new_fp(w_qkv.clone(), None);
        let lin_z = crate::nn::Linear::new_fp(w_z.clone(), None);
        let out_qkv: Vec<f32> = lin_qkv.forward(&x).unwrap().to_vec().unwrap();
        let out_z: Vec<f32> = lin_z.forward(&x).unwrap().to_vec().unwrap();

        let w_fused = mlx::ops::shape::concatenate(&[&w_qkv, &w_z], 0).unwrap();
        let lin_fused = crate::nn::Linear::new_fp(w_fused, None);
        let out_fused = lin_fused.forward(&x).unwrap();

        let fused_qkv = mlx::ops::indexing::slice_strided(
            &out_fused,
            &[0_i32, 0, 0][..],
            &[1_i32, 1, qkv_out][..],
            &[1_i32, 1, 1][..],
        )
        .unwrap();
        let fused_z = mlx::ops::indexing::slice_strided(
            &out_fused,
            &[0_i32, 0, qkv_out][..],
            &[1_i32, 1, qkv_out + z_out][..],
            &[1_i32, 1, 1][..],
        )
        .unwrap();
        let fused_qkv_vec: Vec<f32> = fused_qkv.to_vec().unwrap();
        let fused_z_vec: Vec<f32> = fused_z.to_vec().unwrap();

        assert_eq!(fused_qkv_vec, out_qkv);
        assert_eq!(fused_z_vec, out_z);
    }

    #[test]
    fn gated_delta_step_masked_zero_path() {
        // mask=0 everywhere: output should be 0, state unchanged.
        // Use non-zero state_in to verify state isn't accidentally modified.
        // Dk=32 (minimum) so n_per_t = 32/32 = 1 (Metal C++ forbids zero-length arrays).
        let kernel = build_gated_delta_kernel(true).expect("build masked kernel");

        // Initial state has values [1.0; 8*32] (Hv=1, Dv=8, Dk=32).
        let init_state_data: Vec<f32> = (0..256).map(|_| 1.0_f32).collect();
        let state_in: Array = (init_state_data.as_slice(), (1_i32, 1, 8, 32))
            .try_into()
            .unwrap();

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();
        // mask: [B*T] = [1*1 = 1] all-zero (masked out)
        let mask = Array::zeros((1_i32,), Dtype::Bool).unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr, &mask])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch masked");

        let y = outputs.take_at(0).expect("y");
        let state_out = outputs.take_at(0).expect("state_out");

        // y must be all-zero (else branch sets `y[dv_idx] = 0`).
        let y_f32 = mlx::ops::cast::astype(&y, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_f32.to_vec().unwrap();
        assert!(
            yv.iter().all(|x| x.abs() < 1e-6),
            "masked output not zero: {:?}",
            yv
        );

        // state_out must equal state_in (no update under mask=0 — kernel writes
        // back the unchanged register-cached state at the end).
        let sv: Vec<f32> = state_out.to_vec().unwrap();
        assert!(
            sv.iter().all(|x| (x - 1.0).abs() < 1e-6),
            "state changed under mask=0: {:?}",
            sv
        );
    }
}
