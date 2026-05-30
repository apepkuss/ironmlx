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

// P5g T0: profile mode (compile-time gated by `p5g-profile` feature).
//
// Runtime mode selected by `IRONMLX_P5G_PROFILE_MODE` env var, cached once
// via OnceLock to avoid per-forward env lookup. Disabled (Mode::Off) path
// must produce zero measurable overhead beyond a single cached-flag check.
//
// `as_str()` is the single source of truth for mode names — env parser
// matches the same strings the log emits, so `IRONMLX_P5G_PROFILE_MODE=layer1`
// round-trips identically to the log line `mode=layer1`. Never log
// `{mode:?}` (would emit Debug names `Layer1` / `AblateComputeG` and
// break Phase B/C/D aggregation by env-string match).
#[cfg(feature = "p5g-profile")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileMode {
    Off,
    Layer1,
    Layer2,
    AblateComputeG,
    AblateConv,
    AblateTArr,
    // P5h T0b H2: in-place real vs substitute timing for Steps 2b/5/7c.
    // Forward path produces the real outputs (no behavior change); per step
    // the real output and a freshly-built substitute are each independently
    // eval-timed and emitted as paired records.
    H2Measure,
    // P5h T0b H3: same Step 2b passthrough as AblateConv, but Step 2c still
    // performs the real cache update from conv_input tail. Tests whether
    // skipping conv_state update accounts for AblateConv's downstream cost.
    AblateConvWithManualCacheUpdate,
    // P5h T0b H4: tight Step 7d kernel materialization timer on top of the
    // production (real compute_g) path. Cache update stays outside the inner
    // timer but inside Step 7's outer block.
    H4MeasurePhaseA,
    // P5h T0b H4: tight Step 7d timer on top of the AblateComputeG path
    // (g substituted by zeros). Combined with H4MeasurePhaseA isolates
    // kernel-output variance attributable to g's input value pattern.
    H4MeasureAblateComputeG,
}

#[cfg(feature = "p5g-profile")]
impl ProfileMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProfileMode::Off => "off",
            ProfileMode::Layer1 => "layer1",
            ProfileMode::Layer2 => "layer2",
            ProfileMode::AblateComputeG => "ablate-compute-g",
            ProfileMode::AblateConv => "ablate-conv",
            ProfileMode::AblateTArr => "ablate-t-arr",
            ProfileMode::H2Measure => "h2-measure",
            ProfileMode::AblateConvWithManualCacheUpdate => "ablate-conv-with-manual-cache-update",
            ProfileMode::H4MeasurePhaseA => "h4-measure-phase-a",
            ProfileMode::H4MeasureAblateComputeG => "h4-measure-ablate-compute-g",
        }
    }
}

#[cfg(feature = "p5g-profile")]
static PROFILE_MODE: std::sync::OnceLock<ProfileMode> = std::sync::OnceLock::new();

#[cfg(feature = "p5g-profile")]
pub(crate) fn profile_mode() -> ProfileMode {
    *PROFILE_MODE.get_or_init(
        || match std::env::var("IRONMLX_P5G_PROFILE_MODE").as_deref() {
            Ok(s) if s == ProfileMode::Layer1.as_str() => ProfileMode::Layer1,
            Ok(s) if s == ProfileMode::Layer2.as_str() => ProfileMode::Layer2,
            Ok(s) if s == ProfileMode::AblateComputeG.as_str() => ProfileMode::AblateComputeG,
            Ok(s) if s == ProfileMode::AblateConv.as_str() => ProfileMode::AblateConv,
            Ok(s) if s == ProfileMode::AblateTArr.as_str() => ProfileMode::AblateTArr,
            Ok(s) if s == ProfileMode::H2Measure.as_str() => ProfileMode::H2Measure,
            Ok(s) if s == ProfileMode::AblateConvWithManualCacheUpdate.as_str() => {
                ProfileMode::AblateConvWithManualCacheUpdate
            }
            Ok(s) if s == ProfileMode::H4MeasurePhaseA.as_str() => ProfileMode::H4MeasurePhaseA,
            Ok(s) if s == ProfileMode::H4MeasureAblateComputeG.as_str() => {
                ProfileMode::H4MeasureAblateComputeG
            }
            _ => ProfileMode::Off,
        },
    )
}

#[cfg(feature = "p5g-profile")]
fn parse_layer_idx_from_prefix(prefix: &str) -> Option<i32> {
    // Expects `model.layers.{N}.linear_attn` shape. Returns Some(N) on
    // parse success, None on naming drift (treated as "unknown layer").
    prefix.split('.').nth(2).and_then(|s| s.parse::<i32>().ok())
}

// Module-level cache for ablate-t-arr mode: avoids per-call t_arr construction
// by keying on `seq` (chunk size). Only compiled when p5g-profile feature is on.
#[cfg(feature = "p5g-profile")]
static T_ARR_ABLATION_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<i32, Array>>,
> = std::sync::OnceLock::new();

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
/// - `in_proj_qkv` — Q/K/V input projection feeding the depthwise conv.
/// - `in_proj_z` — value gate projection consumed by `RmsNormGated`.
/// - `in_proj_b` / `in_proj_a` — forget and decay signal projections for
///   the delta-rule recurrence.
/// - `conv1d` — depthwise temporal mixing across the Q/K/V channels (then
///   silu via module-level fused compile cell)
/// - `norm` — `RmsNormGated`: `silu(z) * rms_norm(y)` final mixing
/// - `out_proj` — back to `hidden_size`
/// - `a_log` / `dt_bias` — per-head learned parameters for compute_g
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,   // [num_v_heads]
    dt_bias: Array, // [num_v_heads]
    cfg: GatedDeltaNetConfig,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
    kernel_zero_state_no_mask: OnceLock<MetalKernel>,
    kernel_zero_state_masked: OnceLock<MetalKernel>,
    /// Layer index for profile log. Some(N) if parsed from `model.layers.{N}.linear_attn`
    /// prefix at `from_loader`; None for `from_components` (unit-test path) or prefix
    /// parse failure. Profile-only field (zero footprint without `p5g-profile`).
    #[cfg(feature = "p5g-profile")]
    profile_layer_idx: Option<i32>,
}

impl GatedDeltaNet {
    /// Production constructor: load all weight tensors + a_log + dt_bias.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self> {
        let in_proj_qkv = Linear::from_loader(loader, &format!("{prefix}.in_proj_qkv"))?;
        let in_proj_z = Linear::from_loader(loader, &format!("{prefix}.in_proj_z"))?;
        let in_proj_b = Linear::from_loader(loader, &format!("{prefix}.in_proj_b"))?;
        let in_proj_a = Linear::from_loader(loader, &format!("{prefix}.in_proj_a"))?;
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
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
            kernel_zero_state_no_mask: OnceLock::new(),
            kernel_zero_state_masked: OnceLock::new(),
            #[cfg(feature = "p5g-profile")]
            profile_layer_idx: parse_layer_idx_from_prefix(prefix),
        })
    }

    /// Test/composition seam: build from pre-built nn building blocks.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
        conv1d: Conv1d,
        norm: RmsNormGated,
        out_proj: Linear,
        a_log: Array,
        dt_bias: Array,
        cfg: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
            kernel_zero_state_no_mask: OnceLock::new(),
            kernel_zero_state_masked: OnceLock::new(),
            #[cfg(feature = "p5g-profile")]
            profile_layer_idx: None,
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
        // Non-decoder callers (CLI / standalone tests) — pass -1 per spec § 2.5a.
        self.forward_on(x, mask, None, cache, (), -1)
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
        per_row_lens: Option<&[i32]>,
        mut cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();
        // `layer_idx` is consumed by P5h SpanFields constructions in the
        // p5h-profile feature path (each of the 11 substeps below). In the
        // default build the parameter is unused at body level — silence the
        // unused-variable warning.
        #[cfg(not(feature = "p5h-profile"))]
        let _ = layer_idx;

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

        // P5g T0 Layer 1 entry: materialize input + cache states before timer starts.
        // Drains prior lazy ops so they're not attributed to GatedDeltaNet's forward
        // cost; also forces cache.conv_state + cache.recurrent_state to be tangible
        // so Step 2c/7e cache updates produced inside this forward are the only
        // cache-related materialization captured by the exit barrier.
        //
        // IMPORTANT: timing/logging barriers run ONLY for Layer1 and Layer2 modes.
        // AblateX modes (Phase D) must NOT trigger barriers — Phase D's "wall-time
        // vs Phase A" delta would otherwise be contaminated by barrier overhead,
        // understating the achievable upper-bound cut and possibly misranking
        // candidates. AblateX still flows through the substitute branches in their
        // respective steps (Step 0.12), which are pure shape-preserving replacements.
        #[cfg(feature = "p5g-profile")]
        let _p5g_timer_start = {
            let mode = profile_mode();
            if matches!(mode, ProfileMode::Layer1 | ProfileMode::Layer2) {
                let mut eval_set: Vec<&Array> = vec![x];
                if let Some(c) = cache.as_deref() {
                    eval_set.push(c.conv_state());
                    eval_set.push(c.recurrent_state());
                }
                if let Some(m) = mask {
                    eval_set.push(m);
                }
                mlx::transforms::eval(&eval_set[..])?;
                // Capture offset_before inside the Layer1/Layer2 arm — AblateX
                // modes never use it (exit barrier is gated on _p5g_timer_start).
                // HTTP-path B=1 invariant: offsets() always has at least one
                // element when cache exists.
                let offset_before: i32 = cache
                    .as_deref()
                    .and_then(|c| c.offsets().first().copied())
                    .unwrap_or(0);
                Some((mode, std::time::Instant::now(), offset_before))
            } else {
                None
            }
        };

        // Cache profile_mode() once per forward — avoids 16+ OnceLock reads per call.
        // mode=off path now performs exactly one cached-flag check per forward.
        #[cfg(feature = "p5g-profile")]
        let _p5g_mode = profile_mode();

        // Layer 2 per-step elapsed accumulator.
        #[cfg(feature = "p5g-profile")]
        let mut _p5g_step_elapsed: Vec<u64> = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Vec::with_capacity(12)
        } else {
            Vec::new()
        };

        // P5h T0b H2 per-step (real_us, substitute_us) pairs for Steps 2b/5/7c.
        // None outside H2Measure mode — gate exit emission on Option::is_some().
        // Each pair is filled in the corresponding step body when in H2Measure mode.
        #[cfg(feature = "p5g-profile")]
        let mut _p5h_t0b_h2_step_2b: Option<(u64, u64)> = None;
        #[cfg(feature = "p5g-profile")]
        let mut _p5h_t0b_h2_step_5: Option<(u64, u64)> = None;
        #[cfg(feature = "p5g-profile")]
        let mut _p5h_t0b_h2_step_7c: Option<(u64, u64)> = None;

        // P5h T0b H4 Step 7d tight kernel-materialization timer (us). Fires
        // under H4MeasurePhaseA or H4MeasureAblateComputeG only. Cache update
        // (c.update_recurrent + c.advance) is excluded from this timer.
        #[cfg(feature = "p5g-profile")]
        let mut _p5h_t0b_h4_step_7d: Option<u64> = None;

        // Step 1: reference-equivalent input projections.
        // Step 1a: in_proj_qkv + in_proj_z, then mask-zero qkv at pad
        // positions. The mask multiply stays bundled with qkv projection
        // because it is the immediate downstream consumer before conv1d.
        //
        // Mask-zero rationale (preserved from the prior standalone block):
        // The conv1d is temporal — its output at real-token position t uses
        // input positions `t-(k-1)..t` as history. Under right-padded batched
        // prefill, real qkv occupies positions `[0, L_i)` and the trailing
        // `[L_i, max_len)` positions are pad; conv1d output AT real positions
        // (t < L_i) only consumes earlier real positions (causal kernel), so
        // it stays clean even without zeroing pad qkv. However, conv1d output
        // AT pad positions reads back into the real-tail (positions
        // `[L_i - (k-1), L_i)`), and the kernel post-write of conv_state then
        // captures those pad-slot outputs — so we zero pad qkv up front to
        // keep pad-slot conv1d output benign and avoid leaking pad embeddings
        // (which are non-zero garbage from in_proj_qkv) into the cache
        // update path's per-row slice.
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
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_1a = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (qkv, z) = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_1a_in_proj_qkvz",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array)> {
                        let qkv = self.in_proj_qkv.forward_on(x, target)?;
                        let z = self.in_proj_z.forward_on(x, target)?;
                        let qkv = if let Some(m) = mask {
                            let m_dtype = mlx::ops::cast::astype(m, qkv.dtype())?;
                            let m_broadcast = m_dtype.reshape_on((batch, seq, 1), target)?;
                            &qkv * &m_broadcast
                        } else {
                            qkv
                        };
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&qkv, &z])?;
                        }
                        Ok((qkv, z))
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                let qkv = self.in_proj_qkv.forward_on(x, target)?;
                let z = self.in_proj_z.forward_on(x, target)?;
                let qkv = if let Some(m) = mask {
                    let m_dtype = mlx::ops::cast::astype(m, qkv.dtype())?;
                    let m_broadcast = m_dtype.reshape_on((batch, seq, 1), target)?;
                    &qkv * &m_broadcast
                } else {
                    qkv
                };
                (qkv, z)
            }
        };
        // Step 1a elapsed push: eval the actual outputs of the expanded wrap
        // (qkv post-mask + z) so the timer captures the same work the span
        // covers.
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_1a {
                mlx::transforms::eval(&[&qkv, &z])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 1b: in_proj_b + in_proj_a. Keep both projections in the same
        // profiling span because they form one logical recurrence-parameter
        // stage.
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_1b = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (b, a) = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_1b_in_proj_ba",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array)> {
                        let b = self.in_proj_b.forward_on(x, target)?;
                        let a = self.in_proj_a.forward_on(x, target)?;
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&b, &a])?;
                        }
                        Ok((b, a))
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                let b = self.in_proj_b.forward_on(x, target)?;
                let a = self.in_proj_a.forward_on(x, target)?;
                (b, a)
            }
        };
        // Step 1b elapsed push: eval the wrap's actual outputs (b, a).
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_1b {
                mlx::transforms::eval(&[&b, &a])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // P1 fix: detect ablate-conv BEFORE Steps 2a/2b so the full concat+conv1d+silu
        // graph is never constructed. The substitute (qkv.clone()) is shape-preserving:
        // qkv is [B, S, conv_dim] == conv_out's shape. Step 2c cache update is still
        // gated on ablate_conv below. Step 2a + 2b Layer 2 timers still fire to record
        // the no-op pass-through cost.
        //
        // P5h T0b H3: split into two booleans so AblateConvWithManualCacheUpdate
        // can take the Step 2b passthrough while still running the real Step 2c
        // cache update.
        //   * ablate_conv_step_2b — true under {AblateConv, AblateConvWithManualCacheUpdate}.
        //     Bypasses concat + conv1d + silu (Steps 2a/2b body); conv_out = qkv.clone().
        //   * ablate_conv_step_2c — true ONLY under AblateConv. Skips update_conv.
        // Existing AblateConv behavior is preserved (both flags true).
        #[cfg(feature = "p5g-profile")]
        let ablate_conv_step_2b = matches!(
            _p5g_mode,
            ProfileMode::AblateConv | ProfileMode::AblateConvWithManualCacheUpdate
        );
        #[cfg(not(feature = "p5g-profile"))]
        let ablate_conv_step_2b = false;

        #[cfg(feature = "p5g-profile")]
        let ablate_conv_step_2c = matches!(_p5g_mode, ProfileMode::AblateConv);
        #[cfg(not(feature = "p5g-profile"))]
        let ablate_conv_step_2c = false;

        // Step 2a: prepend conv_state
        //
        // ablate-conv early path: skip concat entirely; conv_input is unused.
        // Bind conv_input to a dummy `qkv.clone()` so the type is consistent;
        // Step 2b's cfg-on/cfg-off both skip their body when ablate_conv is on
        // and conv_input is never referenced downstream.
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_2a = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let conv_input = {
            // Step 2a is skipped only when Step 2c is ALSO skipped (i.e. nothing
            // reads conv_input downstream). Under AblateConvWithManualCacheUpdate
            // Step 2c needs the real concatenated conv_input, so we run the real
            // concat even though Step 2b bypasses conv1d.
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_2a_prepend_conv_state",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        let out = if ablate_conv_step_2c {
                            // ablate-conv early path: skip concat entirely; conv_input is unused.
                            qkv.clone()
                        } else {
                            match cache.as_deref_mut() {
                                Some(c) => concatenate(&[c.conv_state(), &qkv], 1)?,
                                None => {
                                    let zeros = Array::zeros(
                                        (batch, self.cfg.conv_kernel_size - 1, self.cfg.conv_dim()),
                                        qkv.dtype(),
                                    )?;
                                    concatenate(&[&zeros, &qkv], 1)?
                                }
                            }
                        };
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&out])?;
                        }
                        Ok(out)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                if ablate_conv_step_2c {
                    qkv.clone()
                } else {
                    match cache.as_deref_mut() {
                        Some(c) => concatenate(&[c.conv_state(), &qkv], 1)?,
                        None => {
                            let zeros = Array::zeros(
                                (batch, self.cfg.conv_kernel_size - 1, self.cfg.conv_dim()),
                                qkv.dtype(),
                            )?;
                            concatenate(&[&zeros, &qkv], 1)?
                        }
                    }
                }
            }
        };
        // Step 2a elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_2a {
                if !ablate_conv_step_2c {
                    mlx::transforms::eval(&[&conv_input])?;
                }
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 2b: conv1d + silu
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_2b = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // P5h T0b H2: separate timer for the real-body wall time. Started here
        // BEFORE the body so the timer captures construction + eval (apples to
        // apples with the substitute timer below which also covers construction).
        #[cfg(feature = "p5g-profile")]
        let _p5h_t0b_h2_start_2b_real = if matches!(_p5g_mode, ProfileMode::H2Measure) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // ablate-conv early path: bypass conv1d + silu entirely; conv_out = qkv passthrough.
        // Under AblateConv, conv_state is NOT updated (Step 2c skips update below).
        // Under AblateConvWithManualCacheUpdate, the bypass is the SAME but Step 2c
        // still runs the real cache update from conv_input tail. Diagnostic only.
        let conv_out = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_2b_conv1d_silu",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        let out = if ablate_conv_step_2b {
                            qkv.clone()
                        } else {
                            let conv_out = self.conv1d.forward_on(&conv_input, target)?;
                            let conv_sig = conv_out.sigmoid()?;
                            &conv_out * &conv_sig
                        };
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&out])?;
                        }
                        Ok(out)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                if ablate_conv_step_2b {
                    qkv.clone()
                } else {
                    let conv_out = self.conv1d.forward_on(&conv_input, target)?;
                    let conv_sig = conv_out.sigmoid()?;
                    &conv_out * &conv_sig
                }
            }
        };
        // Step 2b elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_2b {
                if !ablate_conv_step_2b {
                    mlx::transforms::eval(&[&conv_out])?;
                }
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }
        // P5h T0b H2: under H2Measure, eval the real conv_out and capture its
        // wall time, then build the substitute (qkv.clone()) and eval it. The
        // substitute is discarded; forward continues using the real conv_out.
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start_real) = _p5h_t0b_h2_start_2b_real {
                mlx::transforms::eval(&[&conv_out])?;
                let real_us = start_real.elapsed().as_micros() as u64;
                let start_sub = std::time::Instant::now();
                let substitute = qkv.clone();
                mlx::transforms::eval(&[&substitute])?;
                let substitute_us = start_sub.elapsed().as_micros() as u64;
                _p5h_t0b_h2_step_2b = Some((real_us, substitute_us));
            }
        }

        // Step 2c: update conv_state cache.
        //
        // The new conv_state for the next call must capture the last
        // `n_keep = kernel_size - 1` tokens of each row's REAL input. Under
        // right-padded batched prefill the real qkv occupies positions
        // `[k-1, k-1 + L_i)` of conv_input (= old conv_state prepended +
        // qkv with pad zeroed). For row i the real-tail window therefore
        // sits at `[k-1 + L_i - n_keep, k-1 + L_i) == [L_i, L_i + n_keep)`
        // of conv_input — uniform-length and B=1 cases collapse to the
        // last n_keep positions of conv_input (matches pre-right-pad
        // behaviour).
        //
        // When `per_row_lens` is `None` (single-stream / non-batched), we
        // fall back to the simple "last n_keep positions" slice.
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_2c = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // See the cfg-off arm below for the per-row real-tail window rationale
        // (3.45x decode slowdown root cause, take_along_axis fusable fast path,
        // bounds-check argument, and ablate-conv stale-data skip).
        {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_2c_update_conv_state",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<()> {
                        if let Some(c) = cache.as_deref_mut() {
                            if !ablate_conv_step_2c {
                                let n_keep = self.cfg.conv_kernel_size - 1;
                                let conv_input_dims = conv_input.shape();
                                let total_len = conv_input_dims.as_slice()[1];
                                let conv_dim = self.cfg.conv_dim();
                                let new_conv_state = match per_row_lens {
                                    Some(lens)
                                        if batch > 1
                                            && !lens.iter().all(|&l| l + n_keep == total_len) =>
                                    {
                                        if lens.len() as i32 != batch {
                                            return Err(anyhow!(
                                                "GatedDeltaNet::forward_on: per_row_lens.len()={} != batch={}",
                                                lens.len(),
                                                batch
                                            ));
                                        }
                                        let mut idx_flat: Vec<u32> =
                                            Vec::with_capacity((batch * n_keep) as usize);
                                        for &l in lens {
                                            for j in 0..n_keep {
                                                idx_flat.push((l + j) as u32);
                                            }
                                        }
                                        let idx: Array =
                                            (&idx_flat[..], &[batch, n_keep, 1_i32][..])
                                                .try_into()
                                                .map_err(|e| {
                                                    anyhow!(
                                                "GatedDeltaNet::forward_on: idx try_into Array failed: {e:?}"
                                            )
                                                })?;
                                        mlx::ops::indexing::take_along_axis_on(
                                            &conv_input,
                                            &idx,
                                            1,
                                            target,
                                        )?
                                    }
                                    _ => mlx::ops::indexing::slice(
                                        &conv_input,
                                        vec![0_i32, total_len - n_keep, 0].as_slice(),
                                        vec![batch, total_len, conv_dim].as_slice(),
                                    )?,
                                };
                                // T4.3: wrap GatedDeltaCache::update_conv field
                                // assignment in `cache_state_update` child span
                                // (parent: `gda_step_2c_update_conv_state`).
                                // The assignment itself is ~0us (Arc share); the
                                // span exists to attribute the mutation cost
                                // explicitly in the T5 tree so the substep's
                                // residual reflects only the new_conv_state
                                // build (slice / take_along_axis).
                                crate::core::p5h::try_with_p5h_span_from_current_trace(
                                    "cache_state_update",
                                    || crate::core::p5h::SpanFields {
                                        layer_idx: Some(layer_idx),
                                        ..Default::default()
                                    },
                                    || {
                                        c.update_conv(new_conv_state);
                                    },
                                );
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                // ablate-conv: conv was replaced with qkv passthrough, so conv_state
                // would receive stale data. Skip the update entirely.
                // AblateConvWithManualCacheUpdate (H3): keep the real update so we
                // can isolate cache-staleness cost from the passthrough body cost.
                // (ablate_conv_step_2c is defined above at Step 2a entry; no
                // re-check needed.)
                if let Some(c) = cache.as_deref_mut() {
                    if !ablate_conv_step_2c {
                        let n_keep = self.cfg.conv_kernel_size - 1;
                        let conv_input_dims = conv_input.shape();
                        let total_len = conv_input_dims.as_slice()[1];
                        let conv_dim = self.cfg.conv_dim();
                        let new_conv_state = match per_row_lens {
                            Some(lens)
                                if batch > 1 && !lens.iter().all(|&l| l + n_keep == total_len) =>
                            {
                                // Per-row real-tail window starts at position `lens[i]`
                                // in conv_input and spans `n_keep` rows. Express as a
                                // single `take_along_axis` over axis 1 with index tensor
                                // `[B, n_keep, 1]` (broadcasts to [B, n_keep, conv_dim]).
                                // This collapses the previous per-row
                                // `slice_strided_on + concatenate_on` (B+1 graph nodes
                                // per layer per call) into one fusable op — the
                                // per-row loop blocked downstream JIT fusion and
                                // caused a 3.45x decode slowdown.
                                //
                                // The match guard `!lens.iter().all(|&l| l + n_keep == total_len)`
                                // routes uniform-length batches to the seq-wide slice
                                // fast path (the `_` arm), so this arm only fires when
                                // at least one row has a true ragged tail.
                                if lens.len() as i32 != batch {
                                    return Err(anyhow!(
                                        "GatedDeltaNet::forward_on: per_row_lens.len()={} != batch={}",
                                        lens.len(),
                                        batch
                                    ));
                                }
                                // Bound check: l in [0, max_len] (= [0, total_len - n_keep]),
                                // so l + j in [0, total_len) for j in [0, n_keep). Always
                                // holds when prefill_admitted set lens[i] = prompt_lens[i]
                                // with prompt_lens[i] <= max_len = seq.
                                let mut idx_flat: Vec<u32> =
                                    Vec::with_capacity((batch * n_keep) as usize);
                                for &l in lens {
                                    for j in 0..n_keep {
                                        idx_flat.push((l + j) as u32);
                                    }
                                }
                                let idx: Array = (&idx_flat[..], &[batch, n_keep, 1_i32][..])
                                    .try_into()
                                    .map_err(|e| {
                                        anyhow!(
                                            "GatedDeltaNet::forward_on: idx try_into Array failed: {e:?}"
                                        )
                                    })?;
                                mlx::ops::indexing::take_along_axis_on(
                                    &conv_input,
                                    &idx,
                                    1,
                                    target,
                                )?
                            }
                            _ => mlx::ops::indexing::slice(
                                &conv_input,
                                vec![0_i32, total_len - n_keep, 0].as_slice(),
                                vec![batch, total_len, conv_dim].as_slice(),
                            )?,
                        };
                        c.update_conv(new_conv_state);
                    }
                }
            }
        }
        // Step 2c elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_2c {
                // P2 fix: eval the updated conv_state to force slice/take_along_axis
                // materialization. Without this the timer measured only lazy graph
                // construction and the real cost was mis-attributed to downstream steps.
                // Under AblateConv the cache block was skipped, so conv_state() still
                // holds the old (already-materialized) value — eval is a no-op there.
                // Under AblateConvWithManualCacheUpdate (H3) the update DID run, so
                // eval here forces materialization of the manually-restored cache state.
                if let Some(c) = cache.as_deref() {
                    mlx::transforms::eval(&[c.conv_state()])?;
                }
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 3: split + reshape per-head
        // conv_out shape: [B, S, conv_dim] = [B, S, key_dim*2 + value_dim]
        // Split at [key_dim, 2*key_dim] → 3 segments [B, S, key_dim], [B, S, key_dim], [B, S, value_dim]
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_3 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (q_per_head, k_per_head, v_per_head) = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_3_split_reshape_per_head",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array, Array)> {
                        let split_at = vec![self.cfg.key_dim(), 2 * self.cfg.key_dim()];
                        let parts = mlx::ops::shape::split_at_on(&conv_out, &split_at, -1, target)?;
                        let q_flat = &parts[0];
                        let k_flat = &parts[1];
                        let v_flat = &parts[2];
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
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&q_per_head, &k_per_head, &v_per_head])?;
                        }
                        Ok((q_per_head, k_per_head, v_per_head))
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
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
                (q_per_head, k_per_head, v_per_head)
            }
        };
        // Step 3 elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_3 {
                mlx::transforms::eval(&[&q_per_head, &k_per_head, &v_per_head])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 4: q/k rms_norm (no weight)
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_4 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (q_scaled, k_scaled) = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_4_qk_rmsnorm",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array)> {
                        let inv_scale = 1.0_f32 / (self.cfg.head_k_dim as f32).sqrt();
                        let q_normed = mlx::fast::rms_norm_on(&q_per_head, None, 1e-6, target)?;
                        let q_scaled = &q_normed * (inv_scale * inv_scale);
                        let k_normed = mlx::fast::rms_norm_on(&k_per_head, None, 1e-6, target)?;
                        let k_scaled = &k_normed * inv_scale;
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&q_scaled, &k_scaled])?;
                        }
                        Ok((q_scaled, k_scaled))
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                let inv_scale = 1.0_f32 / (self.cfg.head_k_dim as f32).sqrt();
                let q_normed = mlx::fast::rms_norm_on(&q_per_head, None, 1e-6, target)?;
                let q_scaled = &q_normed * (inv_scale * inv_scale); // panic-on-err, no `?`
                let k_normed = mlx::fast::rms_norm_on(&k_per_head, None, 1e-6, target)?;
                let k_scaled = &k_normed * inv_scale; // panic-on-err, no `?`
                (q_scaled, k_scaled)
            }
        };
        // Step 4 elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_4 {
                mlx::transforms::eval(&[&q_scaled, &k_scaled])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 5: compute_g = exp(-exp(A_log) * softplus(a + dt_bias))
        // softplus stabilised: where(x > 20, x, log(1 + exp(x)))
        // P1 fix: detect ablate-compute-g BEFORE the chain so the full
        // softplus/exp/mul/exp graph is never constructed. Substitute is
        // zeros_like(a) cast to Float32, matching the original chain's output
        // dtype (inner.exp() is f32) and shape ([B, S, num_v_heads]).
        // Step 5 Layer 2 timer still fires to record the no-op pass-through cost.
        #[cfg(feature = "p5g-profile")]
        let ablate_compute_g = matches!(
            _p5g_mode,
            ProfileMode::AblateComputeG | ProfileMode::H4MeasureAblateComputeG
        );
        #[cfg(not(feature = "p5g-profile"))]
        let ablate_compute_g = false;

        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_5 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // P5h T0b H2: real-body timer for Step 5 (compute_g chain).
        #[cfg(feature = "p5g-profile")]
        let _p5h_t0b_h2_start_5_real = if matches!(_p5g_mode, ProfileMode::H2Measure) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let g = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_5_compute_g",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        let g = if ablate_compute_g {
                            // see cfg-off arm below for design rationale
                            mlx::ops::cast::astype(&a.zeros_like()?, Dtype::Float32)?
                        } else {
                            let x_sp = &a + &self.dt_bias;
                            let twenty: Array = (&[20.0_f32][..], ()).try_into()?;
                            let zeros = a.zeros_like()?;
                            let safe = zeros.logaddexp(&x_sp)?;
                            let cond = x_sp.greater(&twenty)?;
                            let sp = cond.where_(&x_sp, &safe)?;
                            let a_log_f32 = mlx::ops::cast::astype(&self.a_log, Dtype::Float32)?;
                            let exp_alog = a_log_f32.exp()?;
                            let neg_exp_alog = mlx::ops::binary::negative(&exp_alog)?;
                            let inner = &neg_exp_alog * &sp;
                            inner.exp()?
                        };
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&g])?;
                        }
                        Ok(g)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                if ablate_compute_g {
                    // ablate-compute-g early path: bypass softplus/exp/mul/exp chain.
                    // g is f32 zeros with shape matching `a` ([B, S, num_v_heads]).
                    // Downstream kernel still receives a valid same-shape Array;
                    // numerics are invalid (diagnostic only).
                    mlx::ops::cast::astype(&a.zeros_like()?, Dtype::Float32)?
                } else {
                    let x_sp = &a + &self.dt_bias;
                    let twenty: Array = (&[20.0_f32][..], ()).try_into()?;
                    let zeros = a.zeros_like()?;
                    let safe = zeros.logaddexp(&x_sp)?;
                    let cond = x_sp.greater(&twenty)?;
                    let sp = cond.where_(&x_sp, &safe)?;
                    let a_log_f32 = mlx::ops::cast::astype(&self.a_log, Dtype::Float32)?;
                    let exp_alog = a_log_f32.exp()?;
                    let neg_exp_alog = mlx::ops::binary::negative(&exp_alog)?;
                    let inner = &neg_exp_alog * &sp;
                    inner.exp()?
                }
            }
        };
        // Step 5 elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_5 {
                if !ablate_compute_g {
                    mlx::transforms::eval(&[&g])?;
                }
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }
        // P5h T0b H2: under H2Measure, eval the real g (real compute_g chain)
        // and capture its wall time, then build the substitute
        // (zeros_like(a) cast to Float32) and eval it. Substitute discarded.
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start_real) = _p5h_t0b_h2_start_5_real {
                mlx::transforms::eval(&[&g])?;
                let real_us = start_real.elapsed().as_micros() as u64;
                let start_sub = std::time::Instant::now();
                let substitute = mlx::ops::cast::astype(&a.zeros_like()?, Dtype::Float32)?;
                mlx::transforms::eval(&[&substitute])?;
                let substitute_us = start_sub.elapsed().as_micros() as u64;
                _p5h_t0b_h2_step_5 = Some((real_us, substitute_us));
            }
        }

        // Step 6: beta = sigmoid(b)
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_6 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let beta = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_6_sigmoid_beta",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        let beta = b.sigmoid_on(target)?;
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&beta])?;
                        }
                        Ok(beta)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                b.sigmoid_on(target)?
            }
        };
        // Step 6 elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_6 {
                mlx::transforms::eval(&[&beta])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 7 timer: covers 7a (kernel select) through 7e (cache.advance).
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_7 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let y = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_7_kernel_and_cache_update",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        // P5h+1 T1.5 (Codex B-lite): the kernel-dispatch +
                        // materialize work (Steps 7a-7d) moves into a dedicated
                        // child span so the parent `gda_step_7_kernel_and_cache_
                        // update` becomes a thin wrapper with only two leaf
                        // children (kernel_dispatch_and_materialize +
                        // cache_state_update). Prior to this split the kernel
                        // self-time accumulated inside the parent and was
                        // synthesized as an `unattributed_*` residual leaf in
                        // the aggregator — accounting for 88-98% of total
                        // unattributed time per PP and pushing coverage_pct to
                        // 0.92-0.94 (below the 0.95 close-gate threshold).
                        let (y, new_state) =
                            crate::core::p5h::try_with_p5h_span_from_current_trace(
                                "gda_step_7_kernel_dispatch_and_materialize",
                                || crate::core::p5h::SpanFields {
                                    layer_idx: Some(layer_idx),
                                    ..Default::default()
                                },
                                || -> Result<(Array, Array)> {
                                    // Step 7a: build/get the appropriate kernel.
                                    // Initial chunks start from an all-zero
                                    // recurrent state; use a dedicated kernel
                                    // that initializes registers to zero rather
                                    // than reading a large fp32 zero buffer.
                                    let zero_state = match cache.as_deref() {
                                        Some(c) => c.offsets().iter().all(|&o| o == 0),
                                        None => true,
                                    };
                                    let kernel = match (mask.is_some(), zero_state) {
                                        (true, true) => {
                                            self.kernel_zero_state_masked.get_or_init(|| {
                                                build_gated_delta_zero_state_kernel(true)
                                                    .expect("build zero-state masked kernel")
                                            })
                                        }
                                        (false, true) => {
                                            self.kernel_zero_state_no_mask.get_or_init(|| {
                                                build_gated_delta_zero_state_kernel(false)
                                                    .expect("build zero-state no-mask kernel")
                                            })
                                        }
                                        (true, false) => self.kernel_masked.get_or_init(|| {
                                            build_gated_delta_kernel(true)
                                                .expect("build masked kernel")
                                        }),
                                        (false, false) => self.kernel_no_mask.get_or_init(|| {
                                            build_gated_delta_kernel(false)
                                                .expect("build no-mask kernel")
                                        }),
                                    };

                                    // Step 7b: get state_in only after the
                                    // stream has already advanced. See cfg-off
                                    // arm below for Arc-share / cache-slot
                                    // rationale.
                                    let state_in = if zero_state {
                                        None
                                    } else {
                                        Some(
                                            cache
                                                .as_deref()
                                                .expect("nonzero GDN state requires cache to exist")
                                                .recurrent_state()
                                                .clone(),
                                        )
                                    };

                                    // Step 7c: T as 0-dim int32 array.
                                    // see cfg-off arm below for ablate-t-arr
                                    // cache rationale.
                                    #[cfg(feature = "p5g-profile")]
                                    let t_arr: Array =
                                        if matches!(_p5g_mode, ProfileMode::AblateTArr) {
                                            let cache = T_ARR_ABLATION_CACHE.get_or_init(|| {
                                                std::sync::Mutex::new(
                                                    std::collections::HashMap::new(),
                                                )
                                            });
                                            let mut guard = cache.lock().unwrap();
                                            if let Some(arr) = guard.get(&seq) {
                                                arr.clone()
                                            } else {
                                                let arr: Array = (&[seq][..], ()).try_into()?;
                                                guard.insert(seq, arr.clone());
                                                arr
                                            }
                                        } else if matches!(_p5g_mode, ProfileMode::H2Measure) {
                                            // P5h T0b H2: time real construct,
                                            // then time substitute
                                            // (T_ARR_ABLATION_CACHE Mutex lookup
                                            // pattern). The Mutex lock is itself a
                                            // wall-time signal that's being
                                            // measured. Forward continues using
                                            // the real t_arr.
                                            let start_real = std::time::Instant::now();
                                            let real_t_arr: Array = (&[seq][..], ()).try_into()?;
                                            mlx::transforms::eval(&[&real_t_arr])?;
                                            let real_us = start_real.elapsed().as_micros() as u64;
                                            let start_sub = std::time::Instant::now();
                                            let cache = T_ARR_ABLATION_CACHE.get_or_init(|| {
                                                std::sync::Mutex::new(
                                                    std::collections::HashMap::new(),
                                                )
                                            });
                                            let mut guard = cache.lock().unwrap();
                                            let substitute: Array =
                                                if let Some(arr) = guard.get(&seq) {
                                                    arr.clone()
                                                } else {
                                                    let arr: Array = (&[seq][..], ()).try_into()?;
                                                    guard.insert(seq, arr.clone());
                                                    arr
                                                };
                                            drop(guard);
                                            mlx::transforms::eval(&[&substitute])?;
                                            let substitute_us =
                                                start_sub.elapsed().as_micros() as u64;
                                            _p5h_t0b_h2_step_7c = Some((real_us, substitute_us));
                                            real_t_arr
                                        } else {
                                            (&[seq][..], ()).try_into()?
                                        };
                                    #[cfg(not(feature = "p5g-profile"))]
                                    let t_arr: Array = (&[seq][..], ()).try_into()?;

                                    let in_dtype = x.dtype();
                                    let st_dtype = Dtype::Float32;
                                    let y_shape = Shape::from(vec![
                                        batch,
                                        seq,
                                        self.cfg.num_v_heads,
                                        self.cfg.head_v_dim,
                                    ]);
                                    let state_shape = Shape::from(vec![
                                        batch,
                                        self.cfg.num_v_heads,
                                        self.cfg.head_v_dim,
                                        self.cfg.head_k_dim,
                                    ]);

                                    // Step 7d: dispatch
                                    let mut kernel_inputs: Vec<&Array> =
                                        vec![&q_scaled, &k_scaled, &v_per_head, &g, &beta];
                                    if let Some(state_in) = state_in.as_ref() {
                                        kernel_inputs.push(state_in);
                                    }
                                    kernel_inputs.push(&t_arr);
                                    if let Some(m) = mask {
                                        kernel_inputs.push(m);
                                    }

                                    // P5h T0b H4: tight Step 7d timer fires
                                    // under H4MeasurePhaseA /
                                    // H4MeasureAblateComputeG. Covers
                                    // dispatch_builder...dispatch + take_at(0)
                                    // x2 + eval. Cache update (Step 7e) stays
                                    // OUTSIDE this timer (and OUTSIDE this
                                    // sub-span).
                                    #[cfg(feature = "p5g-profile")]
                                    let _p5h_t0b_h4_start_7d = if matches!(
                                        _p5g_mode,
                                        ProfileMode::H4MeasurePhaseA
                                            | ProfileMode::H4MeasureAblateComputeG
                                    ) {
                                        Some(std::time::Instant::now())
                                    } else {
                                        None
                                    };

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

                                    // P5h T0b H4: force-eval both kernel
                                    // outputs to materialize before capturing
                                    // elapsed; without this the timer
                                    // measures only graph construction.
                                    #[cfg(feature = "p5g-profile")]
                                    {
                                        if let Some(start) = _p5h_t0b_h4_start_7d {
                                            mlx::transforms::eval(&[&y, &new_state])?;
                                            _p5h_t0b_h4_step_7d =
                                                Some(start.elapsed().as_micros() as u64);
                                        }
                                    }

                                    // P5h+1 T1: measurement-eval probe for
                                    // BOTH kernel outputs. T1.5 (Codex B-lite)
                                    // eval(&[&y, &new_state]) — eval'ing
                                    // `new_state` here (rather than letting it
                                    // materialize lazily in the subsequent
                                    // cache_state_update child) keeps the
                                    // dispatch+take_at materialization cost
                                    // attributed to THIS sub-span instead of
                                    // bleeding into cache_state_update.
                                    if crate::core::p5h::is_measurement_eval_probes_active() {
                                        mlx::transforms::eval(&[&y, &new_state])?;
                                    }
                                    Ok((y, new_state))
                                },
                            )?;

                        // Step 7e: update cache recurrent_state, advance offset.
                        // T4.3: wrap the GatedDeltaCache::update_recurrent +
                        // advance pair in a `cache_state_update` child span
                        // (parent: `gda_step_7_kernel_and_cache_update`).
                        // Both operations are CPU-only (Arc share / per-row
                        // offset increment); the span exists to attribute the
                        // mutation cost explicitly in the T5 tree, separating
                        // it from the kernel dispatch + state construction
                        // cost (now owned by the sibling
                        // `gda_step_7_kernel_dispatch_and_materialize`
                        // sub-span per P5h+1 T1.5).
                        if let Some(c) = cache.as_deref_mut() {
                            crate::core::p5h::try_with_p5h_span_from_current_trace(
                                "cache_state_update",
                                || crate::core::p5h::SpanFields {
                                    layer_idx: Some(layer_idx),
                                    ..Default::default()
                                },
                                || -> Result<()> {
                                    c.update_recurrent(new_state);
                                    let lens_owned: Vec<i32>;
                                    let lens_ref: &[i32] = match per_row_lens {
                                        Some(l) => l,
                                        None => {
                                            // Non-batched single-stream caller:
                                            // lockstep-equivalent uniform.
                                            lens_owned = vec![seq; batch as usize];
                                            &lens_owned
                                        }
                                    };
                                    c.advance(lens_ref)?;
                                    Ok(())
                                },
                            )?;
                        }
                        Ok(y)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                // Step 7a: build/get the appropriate kernel. The initial
                // prefill chunk has a logically all-zero recurrent state; use
                // a zero-state variant so the kernel does not read or
                // materialize a large fp32 zero buffer.
                let zero_state = match cache.as_deref() {
                    Some(c) => c.offsets().iter().all(|&o| o == 0),
                    None => true,
                };
                let kernel = match (mask.is_some(), zero_state) {
                    (true, true) => self.kernel_zero_state_masked.get_or_init(|| {
                        build_gated_delta_zero_state_kernel(true)
                            .expect("build zero-state masked kernel")
                    }),
                    (false, true) => self.kernel_zero_state_no_mask.get_or_init(|| {
                        build_gated_delta_zero_state_kernel(false)
                            .expect("build zero-state no-mask kernel")
                    }),
                    (true, false) => self.kernel_masked.get_or_init(|| {
                        build_gated_delta_kernel(true).expect("build masked kernel")
                    }),
                    (false, false) => self.kernel_no_mask.get_or_init(|| {
                        build_gated_delta_kernel(false).expect("build no-mask kernel")
                    }),
                };

                // Step 7b: get state_in only after the stream has advanced.
                // Note: `Array::clone()` is cheap (Arc-share refcount inc on
                // `array_desc_`, not a deep memory copy); the regular kernel
                // dispatch needs an `&Array`, and the cache must keep its slot
                // for `update_recurrent` later.
                let state_in = if zero_state {
                    None
                } else {
                    Some(
                        cache
                            .as_deref()
                            .expect("nonzero GDN state requires cache to exist")
                            .recurrent_state()
                            .clone(),
                    )
                };

                // Step 7c: T as 0-dim int32 array.
                // ablate-t-arr: cache the Array keyed by seq to avoid per-call
                // construction overhead. Uses OnceLock<Mutex<HashMap<i32, Array>>>
                // so the same cached value is shared across all layers/calls with the
                // same chunk size.
                #[cfg(feature = "p5g-profile")]
                let t_arr: Array = if matches!(_p5g_mode, ProfileMode::AblateTArr) {
                    let cache = T_ARR_ABLATION_CACHE
                        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
                    let mut guard = cache.lock().unwrap();
                    if let Some(arr) = guard.get(&seq) {
                        arr.clone()
                    } else {
                        let arr: Array = (&[seq][..], ()).try_into()?;
                        guard.insert(seq, arr.clone());
                        arr
                    }
                } else if matches!(_p5g_mode, ProfileMode::H2Measure) {
                    // P5h T0b H2: time real construct, then time substitute
                    // (T_ARR_ABLATION_CACHE Mutex lookup pattern). The Mutex
                    // lock is itself a wall-time signal that's being measured.
                    // Forward continues using the real t_arr.
                    let start_real = std::time::Instant::now();
                    let real_t_arr: Array = (&[seq][..], ()).try_into()?;
                    mlx::transforms::eval(&[&real_t_arr])?;
                    let real_us = start_real.elapsed().as_micros() as u64;
                    let start_sub = std::time::Instant::now();
                    let cache = T_ARR_ABLATION_CACHE
                        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
                    let mut guard = cache.lock().unwrap();
                    let substitute: Array = if let Some(arr) = guard.get(&seq) {
                        arr.clone()
                    } else {
                        let arr: Array = (&[seq][..], ()).try_into()?;
                        guard.insert(seq, arr.clone());
                        arr
                    };
                    drop(guard);
                    mlx::transforms::eval(&[&substitute])?;
                    let substitute_us = start_sub.elapsed().as_micros() as u64;
                    _p5h_t0b_h2_step_7c = Some((real_us, substitute_us));
                    real_t_arr
                } else {
                    (&[seq][..], ()).try_into()?
                };

                #[cfg(not(feature = "p5g-profile"))]
                let t_arr: Array = (&[seq][..], ()).try_into()?;

                let in_dtype = x.dtype();
                let st_dtype = Dtype::Float32;
                let y_shape =
                    Shape::from(vec![batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim]);
                let state_shape = Shape::from(vec![
                    batch,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                    self.cfg.head_k_dim,
                ]);

                // Step 7d: dispatch
                let mut kernel_inputs: Vec<&Array> =
                    vec![&q_scaled, &k_scaled, &v_per_head, &g, &beta];
                if let Some(state_in) = state_in.as_ref() {
                    kernel_inputs.push(state_in);
                }
                kernel_inputs.push(&t_arr);
                if let Some(m) = mask {
                    kernel_inputs.push(m);
                }

                // P5h T0b H4: tight Step 7d timer fires under H4MeasurePhaseA /
                // H4MeasureAblateComputeG. Covers dispatch_builder...dispatch +
                // take_at(0)x2 + eval. Cache update (Step 7e) stays OUTSIDE.
                #[cfg(feature = "p5g-profile")]
                let _p5h_t0b_h4_start_7d = if matches!(
                    _p5g_mode,
                    ProfileMode::H4MeasurePhaseA | ProfileMode::H4MeasureAblateComputeG
                ) {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

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

                // P5h T0b H4: force-eval both kernel outputs to materialize
                // before capturing elapsed.
                #[cfg(feature = "p5g-profile")]
                {
                    if let Some(start) = _p5h_t0b_h4_start_7d {
                        mlx::transforms::eval(&[&y, &new_state])?;
                        _p5h_t0b_h4_step_7d = Some(start.elapsed().as_micros() as u64);
                    }
                }

                // Step 7e: update cache recurrent_state, advance offset
                if let Some(c) = cache.as_deref_mut() {
                    c.update_recurrent(new_state);
                    let lens_owned: Vec<i32>;
                    let lens_ref: &[i32] = match per_row_lens {
                        Some(l) => l,
                        None => {
                            // Non-batched single-stream caller: lockstep-equivalent uniform.
                            lens_owned = vec![seq; batch as usize];
                            &lens_owned
                        }
                    };
                    c.advance(lens_ref)?;
                }
                y
            }
        };
        // Step 7 elapsed push (after 7e c.advance).
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_7 {
                // P2 fix: eval kernel outputs to force Metal dispatch before recording
                // elapsed. MetalKernel::dispatch() returns lazy Array nodes;
                // c.update_recurrent() is assignment; c.advance() updates metadata only.
                // Without eval, the timer measured only graph construction, making
                // 7_kernel under-counted and 8_norm_proj over-counted in Layer 2 ranking.
                // new_state was moved into update_recurrent above and is inaccessible;
                // use y (kernel's first output) + c.recurrent_state() (post-update) when
                // cache exists. No-cache path uses y alone — sufficient since y and
                // new_state are co-dispatched by the same kernel launch.
                if matches!(_p5g_mode, ProfileMode::Layer2) {
                    let mut to_eval: Vec<&Array> = vec![&y];
                    if let Some(c) = cache.as_deref() {
                        to_eval.push(c.recurrent_state());
                    }
                    mlx::transforms::eval(&to_eval[..])?;
                }
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        // Step 8 timer: covers RmsNormGated + reshape + out_proj
        #[cfg(feature = "p5g-profile")]
        let _p5g_step_start_8 = if matches!(_p5g_mode, ProfileMode::Layer2) {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 8: RmsNormGated(y, z) + reshape + out_proj
        let out = {
            #[cfg(feature = "p5h-profile")]
            {
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gda_step_8_norm_proj",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<Array> {
                        let z_per_head = z.reshape_on(
                            (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
                            target,
                        )?;
                        let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
                        let normed_flat =
                            normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
                        let out = self.out_proj.forward_on(&normed_flat, target)?;
                        // P5h+1 T1: measurement-eval probe.
                        if crate::core::p5h::is_measurement_eval_probes_active() {
                            mlx::transforms::eval(&[&out])?;
                        }
                        Ok(out)
                    },
                )?
            }
            #[cfg(not(feature = "p5h-profile"))]
            {
                let z_per_head = z.reshape_on(
                    (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
                    target,
                )?;
                let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
                let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
                self.out_proj.forward_on(&normed_flat, target)?
            }
        };
        // Step 8 elapsed push
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(start) = _p5g_step_start_8 {
                mlx::transforms::eval(&[&out])?;
                _p5g_step_elapsed.push(start.elapsed().as_micros() as u64);
            }
        }

        #[cfg(feature = "p5g-profile")]
        {
            if let Some((mode, start, offset_before)) = _p5g_timer_start {
                // This block runs ONLY for Layer1 / Layer2 (entry barrier set
                // _p5g_timer_start to Some only in those modes). AblateX skips this
                // entirely — no exit eval barrier, no log emission.

                // Materialize all GDN produced outputs INCLUDING updated cache states.
                let mut eval_out: Vec<&Array> = vec![&out];
                if let Some(c) = cache.as_deref() {
                    eval_out.push(c.conv_state());
                    eval_out.push(c.recurrent_state());
                }
                mlx::transforms::eval(&eval_out[..])?;
                let elapsed_us = start.elapsed().as_micros() as u64;

                // offset_after read AFTER cache.advance() in Step 7e has executed.
                let offset_after: i32 = cache
                    .as_deref()
                    .and_then(|c| c.offsets().first().copied())
                    .unwrap_or(0);

                // Build the step_breakdown suffix iff mode == Layer2. Empty string for
                // Layer 1 so the log line is unchanged.
                let breakdown_suffix =
                    if matches!(mode, ProfileMode::Layer2) && !_p5g_step_elapsed.is_empty() {
                        let csv: Vec<String> =
                            _p5g_step_elapsed.iter().map(|us| us.to_string()).collect();
                        format!(" step_breakdown={}", csv.join(","))
                    } else {
                        String::new()
                    };

                // tracing::info! placed strictly AFTER timer-related work — no log
                // calls inside the measured window. Uses mode.as_str() for
                // env-name / log-name consistency (defined in Step 0.2).
                // Note: `batch` and `seq` are i32 locals extracted from the input
                // dims at the top of forward_on. `x` here is the function parameter
                // (the shadow introduced in the old Step 5 body has been removed).
                let layer = self.profile_layer_idx.unwrap_or(-1);
                tracing::info!(
                    "[p5g-profile] mode={} layer={} batch={} seq={} \
                     offset_before={} offset_after={} elapsed_us={}{}",
                    mode.as_str(),
                    layer,
                    batch,
                    seq,
                    offset_before,
                    offset_after,
                    elapsed_us,
                    breakdown_suffix
                );
            }
        }

        // P5h T0b H2 emission: one record per step (2b, 5, 7c) with the paired
        // (real_us, substitute_us) measurements. Only fires when the
        // corresponding step actually captured a pair (i.e. _p5g_mode ==
        // H2Measure). Field names + order match the harness parser exactly.
        #[cfg(feature = "p5g-profile")]
        {
            let layer = self.profile_layer_idx.unwrap_or(-1);
            if let Some((r2b, s2b)) = _p5h_t0b_h2_step_2b {
                tracing::info!(
                    "[p5h-t0b-h2] mode={} step=step_2b layer={} batch={} \
                     seq={} real_us={} substitute_us={}",
                    _p5g_mode.as_str(),
                    layer,
                    batch,
                    seq,
                    r2b,
                    s2b
                );
            }
            if let Some((r5, s5)) = _p5h_t0b_h2_step_5 {
                tracing::info!(
                    "[p5h-t0b-h2] mode={} step=step_5_compute_g layer={} \
                     batch={} seq={} real_us={} substitute_us={}",
                    _p5g_mode.as_str(),
                    layer,
                    batch,
                    seq,
                    r5,
                    s5
                );
            }
            if let Some((r7c, s7c)) = _p5h_t0b_h2_step_7c {
                tracing::info!(
                    "[p5h-t0b-h2] mode={} step=step_7c_t_arr layer={} \
                     batch={} seq={} real_us={} substitute_us={}",
                    _p5g_mode.as_str(),
                    layer,
                    batch,
                    seq,
                    r7c,
                    s7c
                );
            }
        }

        // P5h T0b H4 emission: one record per forward with the tight Step 7d
        // dispatch+take+eval wall time. Only fires under H4MeasurePhaseA /
        // H4MeasureAblateComputeG (the only modes that set _p5h_t0b_h4_step_7d).
        #[cfg(feature = "p5g-profile")]
        {
            if let Some(t7d) = _p5h_t0b_h4_step_7d {
                let layer = self.profile_layer_idx.unwrap_or(-1);
                tracing::info!(
                    "[p5h-t0b-h4] mode={} step=step_7d_dispatch_materialize layer={} \
                     batch={} seq={} elapsed_us={}",
                    _p5g_mode.as_str(),
                    layer,
                    batch,
                    seq,
                    t7d
                );
            }
        }

        Ok(out)
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
    build_gated_delta_kernel_impl(masked, false)
}

/// Build a `gated_delta_step` kernel for the first chunk of a stream, where
/// recurrent state is known to be all zeros. This variant intentionally has no
/// `state_in` input: it initializes the per-thread register tile to 0 and still
/// emits the same `state_out` shape as the regular kernel.
pub(crate) fn build_gated_delta_zero_state_kernel(masked: bool) -> Result<MetalKernel> {
    build_gated_delta_kernel_impl(masked, true)
}

fn build_gated_delta_kernel_impl(masked: bool, zero_state: bool) -> Result<MetalKernel> {
    let mask_clause = if masked {
        "mask[b_idx * T + t]"
    } else {
        "true"
    };
    let state_in_ptr = if zero_state {
        ""
    } else {
        "auto i_state = state_in + (n * Dv + dv_idx) * Dk;"
    };
    let state_init = if zero_state {
        "state[i] = 0.0f;"
    } else {
        "state[i] = static_cast<float>(i_state[s_idx]);"
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
        {state_in_ptr}
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          {state_init}
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
        mask_clause = mask_clause,
        state_in_ptr = state_in_ptr,
        state_init = state_init
    );

    let name = match (masked, zero_state) {
        (false, false) => "ironmlx_gated_delta",
        (true, false) => "ironmlx_gated_delta_masked",
        (false, true) => "ironmlx_gated_delta_zero_state",
        (true, true) => "ironmlx_gated_delta_zero_state_masked",
    };

    let inputs: &[&str] = match (masked, zero_state) {
        (false, false) => &["q", "k", "v", "g", "beta", "state_in", "T"],
        (true, false) => &["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
        (false, true) => &["q", "k", "v", "g", "beta", "T"],
        (true, true) => &["q", "k", "v", "g", "beta", "T", "mask"],
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
    use serial_test::serial;

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

        GatedDeltaNet::from_components(
            crate::nn::Linear::new_fp(qkv_w, None),
            crate::nn::Linear::new_fp(z_w, None),
            crate::nn::Linear::new_fp(b_w, None),
            crate::nn::Linear::new_fp(a_w, None),
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
    #[serial(mlx_metal)]
    fn gdn_construction_carries_config() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        assert_eq!(cfg.num_v_heads, 4);
        assert_eq!(cfg.num_k_heads, 2);
        assert_eq!(cfg.conv_kernel_size, 4);
    }

    #[test]
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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

    fn assert_zero_state_kernel_matches_regular(masked: bool) {
        let regular = build_gated_delta_kernel(masked).expect("regular kernel");
        let zero_state = build_gated_delta_zero_state_kernel(masked).expect("zero-state kernel");

        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.001 + 0.01).collect();
        let k_data: Vec<f32> = (0..64).map(|i| (i as f32) * -0.0007 + 0.02).collect();
        let v_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.003 + 0.1).collect();
        let g_data = [0.7_f32, 0.5];
        let beta_data = [0.25_f32, 0.4];

        let q: Array = (q_data.as_slice(), (1_i32, 2, 1, 32)).try_into().unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 2, 1, 32)).try_into().unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2, 1, 8)).try_into().unwrap();
        let g: Array = (g_data.as_slice(), (1_i32, 2, 1)).try_into().unwrap();
        let beta: Array = (beta_data.as_slice(), (1_i32, 2, 1)).try_into().unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 32), Dtype::Float32).unwrap();
        let t_arr: Array = (&[2_i32][..], ()).try_into().unwrap();
        let mask: Option<Array> = masked.then(|| {
            let mask_data = [true, false];
            (mask_data.as_slice(), (2_i32,)).try_into().unwrap()
        });

        let y_shape = Shape::from(vec![1, 2, 1, 8]);
        let state_shape = Shape::from(vec![1, 1, 8, 32]);
        let output_shapes = [y_shape.clone(), state_shape.clone()];
        let output_dtypes = [Dtype::Float32, Dtype::Float32];

        let mut regular_inputs: Vec<&Array> = vec![&q, &k, &v, &g, &beta, &state_in, &t_arr];
        if let Some(mask) = mask.as_ref() {
            regular_inputs.push(mask);
        }
        let mut regular_out = regular
            .dispatch_builder()
            .inputs(&regular_inputs)
            .output_shapes(&output_shapes)
            .output_dtypes(&output_dtypes)
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Float32)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch regular");

        let mut zero_inputs: Vec<&Array> = vec![&q, &k, &v, &g, &beta, &t_arr];
        if let Some(mask) = mask.as_ref() {
            zero_inputs.push(mask);
        }
        let mut zero_out = zero_state
            .dispatch_builder()
            .inputs(&zero_inputs)
            .output_shapes(&[y_shape, state_shape])
            .output_dtypes(&output_dtypes)
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Float32)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch zero-state");

        let regular_y = regular_out.take_at(0).expect("regular y");
        let regular_state = regular_out.take_at(0).expect("regular state");
        let zero_y = zero_out.take_at(0).expect("zero y");
        let zero_state_out = zero_out.take_at(0).expect("zero state");

        let regular_y_vec: Vec<f32> = regular_y.to_vec().unwrap();
        let zero_y_vec: Vec<f32> = zero_y.to_vec().unwrap();
        let regular_state_vec: Vec<f32> = regular_state.to_vec().unwrap();
        let zero_state_vec: Vec<f32> = zero_state_out.to_vec().unwrap();

        for (actual, expected) in zero_y_vec.iter().zip(regular_y_vec.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "zero-state y diverged: actual={actual} expected={expected}"
            );
        }
        for (actual, expected) in zero_state_vec.iter().zip(regular_state_vec.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "zero-state recurrent state diverged: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_zero_state_kernel_matches_explicit_zero_state() {
        assert_zero_state_kernel_matches_regular(false);
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_zero_state_masked_kernel_matches_explicit_zero_state() {
        assert_zero_state_kernel_matches_regular(true);
    }

    #[test]
    #[serial(mlx_metal)]
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
