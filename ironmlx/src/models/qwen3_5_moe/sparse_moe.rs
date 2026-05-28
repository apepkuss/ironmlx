//! SparseMoeBlock: routed top-k experts + shared expert with sigmoid gate.
//!
//! Implements the Qwen3.5-MoE SparseMoeBlock per the model architecture
//! defined in its config (num_experts, top_k routing, routed + shared
//! expert with sigmoid gate). The forward path uses MLX's
//! `gather_quantized_matmul` for fused per-token expert routing.
//!
//! Implementation is independent (no code from any reference). Algorithm
//! steps are derived from the Qwen3.5-MoE config + the published
//! Qwen3-MoE design. Local research notes (`.claude/p5b-research-notes.md`,
//! gitignored) capture the architecture analysis used to derive this code.
//!
//! Data flow (per forward call):
//!   x: [B, S, hidden]
//!   1. gates  = router_gate(x)                   Linear (quantized)
//!   2. probs  = softmax(gates, axis=-1)           [B, S, E]
//!   3. inds   = argpartition(probs, -k, axis=-1)[..., -k:]  [B, S, k]
//!   4. scores = take_along_axis(probs, inds, -1), optionally normalized
//!      across top-k when the model config enables norm_topk_prob
//!   5. routed = gather_quantized_matmul_on on flat [BS, H]:
//!      x_in = expand_dims(flat, [-2,-3])  → [BS, 1, 1, H]
//!      gate_out, up_out: [BS, k, 1, moe_inter]
//!      act = silu(gate_out) * up_out       [BS, k, 1, moe_inter]
//!      down_out_4d: [BS, k, 1, hidden], squeeze(-2) → [BS, k, hidden]
//!      routed_y = (down_out * scores_unsq).sum(-2)  → [BS, H]
//!   6. shared = shared_expert(x_flat)
//!      shared = sigmoid(shared_expert_gate(x_flat)) * shared
//!   7. out = routed_y + shared_gated  → reshape [B, S, H]

use anyhow::{anyhow, Context};
use mlx::compile::CompiledFn;
use mlx::ops::indexing::{slice_on, take_along_axis_on, take_on};
use mlx::ops::shape::concatenate_on;
use mlx::ops::sort::{argpartition_on, argsort_on};
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::activations::{build_swiglu, invoke_swiglu};
use crate::nn::{Linear, Mlp};
use crate::Result;

/// Minimum (batch_size * num_experts_per_tok) for the sorted-routing path.
///
/// This mirrors MLX-LM's `SwitchGLU` routing contract: sort expert indices once
/// `indices.size >= 64`. Keeping this threshold aligned is correctness-critical
/// for short Qwen3.6 MoE prefills, where the unsorted gather_qmm path can
/// diverge from the reference route packing even before any KV-cache logic runs.
const SORTED_ROUTING_MIN_BS_K: i32 = 64;
const MAX_EXACT_U32_IN_F32: i32 = 1 << 24;

fn sorted_token_indices_from_sort_perm(
    sort_perm: &Array,
    k: i32,
    bs_k: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    if k <= 0 {
        return Err(anyhow!("SparseMoeBlock: top-k must be positive, got {k}"));
    }
    if bs_k < 0 {
        return Err(anyhow!(
            "SparseMoeBlock: bs_k must be non-negative, got {bs_k}"
        ));
    }

    // sorted_token_idx[i] = sort_perm[i] / k, i.e. the original token row
    // feeding sorted slot i. For normal Qwen MoE batch/context sizes, the
    // permutation is exactly representable as f32, so keeping the division in
    // the MLX graph avoids building and uploading a Rust-side [BS*k] helper
    // array on every MoE layer.
    if bs_k <= MAX_EXACT_U32_IN_F32 {
        let sort_perm_f32 = mlx::ops::cast::astype_on(sort_perm, mlx::Dtype::Float32, target)
            .context("SparseMoeBlock: cast sort_perm to Float32")?;
        let k_scalar: Array = (&[k as f32][..], ())
            .try_into()
            .map_err(|e| anyhow!("SparseMoeBlock: build k scalar: {e}"))?;
        let div = sort_perm_f32
            .try_div_on(&k_scalar, target)
            .context("SparseMoeBlock: sort_perm / k")?;
        let sorted_token_idx_f32 = div
            .floor_on(target)
            .context("SparseMoeBlock: floor(sort_perm / k)")?;
        return mlx::ops::cast::astype_on(&sorted_token_idx_f32, mlx::Dtype::Uint32, target)
            .context("SparseMoeBlock: cast sorted_token_idx to Uint32");
    }

    // Exact fallback for oversized batches. This preserves correctness rather
    // than relying on f32 integer precision beyond 2^24.
    let bs_k_usize = usize::try_from(bs_k)
        .map_err(|e| anyhow!("SparseMoeBlock: bs_k does not fit usize: {e}"))?;
    let k_usize =
        usize::try_from(k).map_err(|e| anyhow!("SparseMoeBlock: k does not fit usize: {e}"))?;
    let token_idx_vec: Vec<u32> = (0..bs_k_usize).map(|i| (i / k_usize) as u32).collect();
    let token_idx: Array = (token_idx_vec.as_slice(), [bs_k])
        .try_into()
        .map_err(|e| anyhow!("SparseMoeBlock: build token_idx array: {e}"))?;
    take_along_axis_on(&token_idx, sort_perm, -1_i32, target)
        .context("SparseMoeBlock: take_along_axis sort token_idx")
}

fn router_topk_scores_and_indices(
    logits: &Array,
    k: i32,
    num_experts: i32,
    norm_topk_prob: bool,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let dims = logits.shape();
    let shape = dims.as_slice();
    if shape.len() != 2 {
        return Err(anyhow!(
            "SparseMoeBlock: router logits must be rank-2 [BS,E], got rank {}",
            shape.len()
        ));
    }
    let bs = shape[0];
    let probs = mlx::ops::softmax_on(logits, -1_i32, /* precise */ true, target)
        .context("SparseMoeBlock: router softmax")?;
    let part_inds =
        argpartition_on(&probs, -(k), -1, target).context("SparseMoeBlock: argpartition")?;
    let inds = mlx::ops::slice_strided_on(
        &part_inds,
        [0_i32, num_experts - k],
        [bs, num_experts],
        [1_i32, 1_i32],
        target,
    )
    .context("SparseMoeBlock: slice top-k from argpartition")?;

    let scores_raw = take_along_axis_on(&probs, &inds, -1, target)
        .context("SparseMoeBlock: take top-k probabilities")?;
    let scores = if norm_topk_prob {
        let scores_sum = mlx::ops::sum_on(&scores_raw, -1_i32, /* keepdim */ true, target)
            .context("SparseMoeBlock: sum top-k probabilities")?;
        &scores_raw / &scores_sum
    } else {
        scores_raw
    };
    let inds_u32 = mlx::ops::cast::astype_on(&inds, mlx::Dtype::Uint32, target)
        .context("SparseMoeBlock: cast indices to Uint32")?;
    Ok((scores, inds_u32))
}

#[cfg(feature = "p5h-profile")]
fn expert_occupancy_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_EXPERT_OCCUPANCY_LOG")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[cfg(feature = "p5h-profile")]
fn p5i_c_gate_up_child_spans_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRONMLX_P5I_C_GATE_UP_CHILD_SPANS")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[cfg(feature = "p5h-profile")]
fn with_p5i_c_gate_up_child_span<T>(
    enabled: bool,
    span_name: &'static str,
    layer_idx: i32,
    body: impl FnOnce() -> T,
) -> T {
    if enabled {
        crate::core::p5h::try_with_p5h_span_from_current_trace(
            span_name,
            || crate::core::p5h::SpanFields {
                layer_idx: Some(layer_idx),
                ..Default::default()
            },
            body,
        )
    } else {
        body()
    }
}

/// Source legacy gate + up weights, consumed once during lazy fused-weight
/// build then dropped to release the 2 × (E × I × H/8) buffer per layer.
struct LazyGateUpSource {
    gate_weight: Array,
    gate_scales: Array,
    gate_biases: Option<Array>,
    up_weight: Array,
    up_scales: Array,
    up_biases: Option<Array>,
}

/// Container for the fused gate+up quantized weights, lazily built on the
/// first forward thread (so the MLX Array allocation is bound to the
/// worker thread's Metal stream, not the loader thread's — see
/// scheduler_actor.rs:163 "B1-p2.5 P0 fix v2" notes).
struct FusedGateUp {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
}

/// Stacked-expert quantized weights for the routed SwiGLU.
///
/// Shape convention (4-bit, group_size=64, num_experts=E, hidden=H,
/// moe_intermediate=I):
///   gate/up (legacy source, dropped after fused build):
///                weight `[E, I, H/8]`, scales/biases `[E, I, H/64]`
///   gate_up (fused, lazy): weight `[E, 2I, H/8]`, scales/biases `[E, 2I, H/64]`
///   down:    weight `[E, H, I/8]`, scales/biases `[E, H, I/64]`
///
/// P5i.a T2: gate_proj + up_proj weights are concatenated along the
/// intermediate axis (axis=1) on the first forward call so a single
/// `gather_qmm` call replaces the prior two-call (gate then up). The fused
/// output is sliced along the last dim into gate_out / up_out before
/// SwiGLU. 4-bit affine quantization is per-row along intermediate (groups
/// are along K=last); stacking along intermediate preserves all per-row
/// scales/biases.
///
/// Lazy build is required to keep MLX Array allocations on the worker
/// thread's Metal stream — building eagerly in `from_loader` (which runs
/// on the CLI / main thread) and then evaluating in the scheduler worker
/// thread fails with "no Stream(gpu, N) in current thread" per the same
/// failure mode documented at scheduler_actor.rs:163-169 (B1-p2.5 P0 fix
/// v2). The legacy source is held in `Mutex<Option<...>>` so it can be
/// consumed and dropped once the fused build succeeds (avoids doubling
/// MoE weight footprint by ~16 GB on the 35B model).
pub struct RoutedExperts {
    /// Legacy source weights, consumed on first forward call. After
    /// lazy fused build the inner `Option` is `take()`-d to `None`,
    /// releasing the original Array refs back to MLX.
    legacy_source: std::sync::Mutex<Option<LazyGateUpSource>>,
    /// Lazy-built fused weights. Initialized on the first forward call so
    /// the resulting MLX Arrays are bound to the worker thread's Metal
    /// stream. See struct doc for failure-mode rationale.
    fused_gate_up: std::sync::OnceLock<FusedGateUp>,
    /// Per-projection intermediate size I (= gate_weight.shape(1)).
    /// Cached at load time to avoid recomputing in the hot forward path.
    pub moe_intermediate: i32,
    pub down_weight: Array,
    pub down_scales: Array,
    pub down_biases: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
    pub num_experts: i32,
}

impl RoutedExperts {
    /// Load from `{prefix}.gate_proj.*` + `up_proj.*` + `down_proj.*`.
    /// Prefix is typically `"model.layers.{i}.mlp.switch_mlp"`.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let qmeta = loader.quant_meta().ok_or_else(|| {
            anyhow!("RoutedExperts requires quantized checkpoint; loader has no QuantMeta")
        })?;

        let gate_weight = loader
            .tensor(&format!("{prefix}.gate_proj.weight"))
            .context("RoutedExperts: gate_proj.weight")?
            .clone();
        let gate_scales = loader
            .tensor(&format!("{prefix}.gate_proj.scales"))
            .context("RoutedExperts: gate_proj.scales")?
            .clone();
        let gate_biases = loader
            .tensor_opt(&format!("{prefix}.gate_proj.biases"))
            .cloned();

        let up_weight = loader
            .tensor(&format!("{prefix}.up_proj.weight"))
            .context("RoutedExperts: up_proj.weight")?
            .clone();
        let up_scales = loader
            .tensor(&format!("{prefix}.up_proj.scales"))
            .context("RoutedExperts: up_proj.scales")?
            .clone();
        let up_biases = loader
            .tensor_opt(&format!("{prefix}.up_proj.biases"))
            .cloned();

        let down_weight = loader
            .tensor(&format!("{prefix}.down_proj.weight"))
            .context("RoutedExperts: down_proj.weight")?
            .clone();
        let down_scales = loader
            .tensor(&format!("{prefix}.down_proj.scales"))
            .context("RoutedExperts: down_proj.scales")?
            .clone();
        let down_biases = loader
            .tensor_opt(&format!("{prefix}.down_proj.biases"))
            .cloned();

        let num_experts = gate_weight.shape().as_slice()[0];
        let moe_intermediate = gate_weight.shape().as_slice()[1];

        // Validate biases presence symmetry upfront (cheap, no MLX op).
        match (gate_biases.as_ref(), up_biases.as_ref()) {
            (Some(_), Some(_)) | (None, None) => {}
            (gb, ub) => {
                return Err(anyhow!(
                    "RoutedExperts: gate/up biases presence mismatch (gate={}, up={}); affine quantization requires both or neither",
                    gb.is_some(),
                    ub.is_some()
                ));
            }
        }

        let legacy_source = LazyGateUpSource {
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
        };

        Ok(Self {
            legacy_source: std::sync::Mutex::new(Some(legacy_source)),
            fused_gate_up: std::sync::OnceLock::new(),
            moe_intermediate,
            down_weight,
            down_scales,
            down_biases,
            group_size: qmeta.group_size,
            bits: qmeta.bits,
            num_experts,
        })
    }

    /// Returns the fused gate+up weights, lazily building them on first
    /// call. Must be called from the forward thread (MLX worker) so the
    /// underlying MLX Arrays are bound to that thread's Metal stream.
    /// `target` propagates the caller's stream-or-device selection to the
    /// underlying `concatenate_on` calls — typically
    /// `StreamOrDevice::default()` from the scheduler driver thread.
    ///
    /// P5i.a T2: 4-bit affine quantization stores per-(expert,row) scale +
    /// bias with groups along the K=last axis only; stacking along the
    /// intermediate axis is a mathematically exact row-wise rearrangement
    /// that preserves every per-row scale/bias. Single gather_qmm output
    /// is later sliced into (gate_out, up_out) along the last dim before
    /// SwiGLU.
    ///
    /// On successful first build the legacy source `Option` is taken,
    /// the fused tensors are explicitly materialized via `mlx::transforms::eval`
    /// (so the lazy MLX graph no longer references the source arrays), and
    /// then `source` is dropped, releasing the original gate/up Array refs
    /// (avoids doubling MoE weight footprint by ~16 GB on the 35B model).
    ///
    /// Concurrency model: the OnceLock provides a lock-free fast path once
    /// the fused weights are built. The mutex covers the ENTIRE
    /// take+build+eval+set window so concurrent callers either (a) observe
    /// the OnceLock already populated on the fast path, or (b) block on the
    /// mutex until the first builder finishes — then re-check the OnceLock
    /// inside the lock and return the populated value. There is no
    /// "raced + lost source" failure mode.
    fn fused_gate_up(&self, target: StreamOrDevice) -> Result<&FusedGateUp> {
        // Fast path: already built — no lock needed.
        if let Some(fused) = self.fused_gate_up.get() {
            return Ok(fused);
        }
        // Slow path: hold the mutex across the entire build+set window so a
        // concurrent second caller blocks here until the first builder
        // finishes, then sees the populated OnceLock on the inner re-check.
        let mut guard = self
            .legacy_source
            .lock()
            .map_err(|e| anyhow!("RoutedExperts: legacy_source mutex poisoned: {e}"))?;
        // Inner re-check: another caller may have raced through the slow
        // path while we were waiting on the mutex.
        if let Some(fused) = self.fused_gate_up.get() {
            return Ok(fused);
        }
        let source = guard.take().ok_or_else(|| {
            anyhow!(
                "RoutedExperts: legacy_source already taken but fused_gate_up never set; \
                 likely a prior panic mid-build that this struct cannot recover from"
            )
        })?;
        let weight = concatenate_on(&[&source.gate_weight, &source.up_weight], 1, target)
            .context("RoutedExperts::fused_gate_up: concatenate weights")?;
        let scales = concatenate_on(&[&source.gate_scales, &source.up_scales], 1, target)
            .context("RoutedExperts::fused_gate_up: concatenate scales")?;
        let biases = match (source.gate_biases.as_ref(), source.up_biases.as_ref()) {
            (Some(gb), Some(ub)) => Some(
                concatenate_on(&[gb, ub], 1, target)
                    .context("RoutedExperts::fused_gate_up: concatenate biases")?,
            ),
            (None, None) => None,
            _ => unreachable!("biases symmetry validated in from_loader"),
        };
        // P5i.a Codex P2 #2: `concatenate_on` returns a lazy MLX graph that
        // still references the source `gate_*` / `up_*` arrays. Dropping
        // `source` here without first materializing the fused tensors would
        // leave the source arrays alive (held by the lazy graph) and the
        // ~16 GB MoE weight doubling would NOT be avoided. Eval forces MLX
        // to compute and store the concatenated buffers, severing the
        // dependency on the source arrays so the subsequent `drop(source)`
        // can actually release them.
        //
        // This eval runs on the worker thread (same thread that built the
        // graph via lazy first-forward dispatch), so the Metal stream
        // binding matches the build site — no cross-thread stream issue.
        let mut to_eval: Vec<&Array> = vec![&weight, &scales];
        if let Some(b) = biases.as_ref() {
            to_eval.push(b);
        }
        mlx::transforms::eval(&to_eval)
            .context("RoutedExperts::fused_gate_up: eval fused tensors before releasing source")?;
        // Now safe to drop source — fused tensors are materialized and the
        // lazy graph no longer holds refs into source arrays.
        drop(source);
        let fused = FusedGateUp {
            weight,
            scales,
            biases,
        };
        // We hold the mutex AND OnceLock is empty (re-checked above), so
        // `set` cannot fail under correct usage.
        self.fused_gate_up
            .set(fused)
            .map_err(|_| anyhow!("RoutedExperts: fused_gate_up OnceLock set raced under mutex"))?;
        Ok(self
            .fused_gate_up
            .get()
            .expect("just set under mutex; OnceLock is now populated"))
    }
}

/// Sparse MoE block for Qwen3.5-MoE.
///
/// Routing: softmax → argpartition top-k, with optional top-k renormalization
/// controlled by checkpoint config.
/// Routed path: `gather_quantized_matmul_on` (G1) fused stacked SwiGLU.
/// Shared path: standard `Mlp` gated by `sigmoid(shared_expert_gate(x))`.
pub struct SparseMoeBlock {
    /// Router: Linear(hidden → num_experts) quantized 4-bit, no additive bias.
    router_gate: Linear,
    /// Stacked routed expert weights.
    routed: RoutedExperts,
    /// Shared expert: standard SwiGLU Mlp.
    shared_expert: Mlp,
    /// Linear(hidden → 1) quantized; `sigmoid(·)` gates the shared expert
    /// output independently of the routing scores.
    shared_expert_gate: Linear,
    /// Number of experts selected per token.
    num_experts_per_tok: i32,
    /// Whether selected expert scores are renormalized across top-k.
    norm_topk_prob: bool,
    swiglu: std::sync::OnceLock<CompiledFn>,
}

impl SparseMoeBlock {
    /// Construct from `{prefix}` where prefix = `"model.layers.{i}.mlp"`.
    ///
    /// Sub-paths:
    ///   `{prefix}.gate`              — router gate Linear (quantized)
    ///   `{prefix}.switch_mlp`        — routed stacked experts
    ///   `{prefix}.shared_expert`     — shared SwiGLU Mlp
    ///   `{prefix}.shared_expert_gate`— sigmoid gate Linear (quantized)
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        num_experts_per_tok: i32,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        let router_gate = Linear::from_loader(loader, &format!("{prefix}.gate"))
            .context("SparseMoeBlock: loading router gate")?;
        let routed = RoutedExperts::from_loader(loader, &format!("{prefix}.switch_mlp"))
            .context("SparseMoeBlock: loading routed experts")?;
        let shared_expert = Mlp::from_loader(loader, &format!("{prefix}.shared_expert"))
            .context("SparseMoeBlock: loading shared_expert")?;
        let shared_expert_gate =
            Linear::from_loader(loader, &format!("{prefix}.shared_expert_gate"))
                .context("SparseMoeBlock: loading shared_expert_gate")?;

        Ok(Self {
            router_gate,
            routed,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok,
            norm_topk_prob,
            swiglu: std::sync::OnceLock::new(),
        })
    }

    fn swiglu(&self) -> &CompiledFn {
        self.swiglu.get_or_init(build_swiglu)
    }

    fn swiglu_on(&self, gate: &Array, up: &Array) -> Result<Array> {
        invoke_swiglu(self.swiglu(), gate, up)
    }

    /// Forward pass: `[B, S, H]` → `[B, S, H]`.
    ///
    /// Stream-targeted. Caller is responsible for passing the correct stream;
    /// `()` selects the MLX default stream.
    ///
    /// `layer_idx` — index of the enclosing decoder block. Consumed under
    /// `#[cfg(feature = "p5h-profile")]` by the 8 MoE substep spans
    /// (router_logits_softmax_topk … moe_output_sum); inert otherwise.
    pub fn forward_on(&self, x: &Array, target: StreamOrDevice, layer_idx: i32) -> Result<Array> {
        let dims = x.shape();
        let dvec = dims.as_slice();
        if dvec.len() != 3 {
            return Err(anyhow!(
                "SparseMoeBlock::forward_on: x must be rank-3 [B,S,H], got rank {}",
                dvec.len()
            ));
        }
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let bs = b * s;
        let k = self.num_experts_per_tok;
        let num_experts = self.routed.num_experts;
        let bs_k = bs * k;
        let use_sorted = bs_k >= SORTED_ROUTING_MIN_BS_K;

        // --- Flatten [B, S, H] → [BS, H] for routing and expert kernels. ---
        // Setup before the 8 substep spans; not attributed to any substep.
        let flat_x = mlx::ops::shape::reshape(x, [bs, h])
            .context("SparseMoeBlock: reshape [B,S,H] → [BS,H]")?;

        #[cfg(feature = "p5h-profile")]
        {
            // T3.2: 8-substep instrumentation, all under the `mlp_path` wrapper
            // opened by DecoderLayerMoe::forward_on (T0a.11 step 1) per
            // decoder_layer.rs:249. Substep names per spec § 3 T3 lines 886-894
            // ("Mirrors T2 with the 8-step MoE breakdown", plan line 4641).
            //
            // The `try_` variant no-ops when no active P5H_CURRENT_TRACE
            // (CLI / standalone tests path) per Codex v12 P1 #1.
            //
            // Sorted vs default routing dispatch: `use_sorted = bs_k >=
            // SORTED_ROUTING_MIN_BS_K`. The `routing_sort_pack` and
            // `gather_qmm_gate_up` spans emit on BOTH branches — on the
            // default broadcast branch `routing_sort_pack` wraps a no-op
            // closure (inclusive_us ≈ 0) per spec hard-rule #9, and the
            // gather_qmm_gate_up closure absorbs the default-branch
            // expand_dims as part of gather_qmm input shaping. Span count
            // is invariant across branches.

            // P5i.c Phase 1 Stage α: capture gate_up child-span opt-in flag
            // once per forward pass (OnceLock cached; env var read only once).
            let gate_up_child_spans_enabled = p5i_c_gate_up_child_spans_enabled();

            // Substep 1: router_logits_softmax_topk — Linear(hidden→E) +
            // top-k selection + checkpoint-controlled score normalization +
            // cast indices to uint32 for downstream gather_qmm.
            //
            // argpartition is preferable to topk: we don't need the top-k
            // elements sorted internally (each is independently weight-summed
            // via gather_qmm); we only need to know which k indices to gather.
            // MLX argpartition is a single pass; values recovered via
            // take_along_axis. argpartition kth=-(k) places the top-k in the
            // last k positions; we slice [BS, E] → [BS, k] keeping the last k.
            let (scores, inds_u32) = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "router_logits_softmax_topk",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<(Array, Array)> {
                    let logits = self.router_gate.forward_on(&flat_x, target)?; // [BS, E]
                    let (scores, inds_u32) = router_topk_scores_and_indices(
                        &logits,
                        k,
                        num_experts,
                        self.norm_topk_prob,
                        target,
                    )?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&scores, &inds_u32])?;
                    }
                    Ok((scores, inds_u32))
                },
            )?;

            // Substep 2: routing_sort_pack — sorted branch does the real
            // argsort + token_idx build + physical gather; default branch
            // emits a zero-cost no-op span per spec hard-rule #9 so the
            // span sequence is invariant across branches.
            //
            // Sorted-flat path (BS*k >= SORTED_ROUTING_MIN_BS_K): pre-sort
            // tokens by expert id and pass `sorted_indices=true`, matching
            // MLX-LM `SwitchGLU` for `indices.size >= 64`. We physically gather
            // `flat_x` rows by the sorted token id so each
            // (token, expert-slot) is its own x-row; this changes x.shape from
            // [BS,1,1,H] to [BS*k,1,1,H] and `rhs_indices` from [BS,k] to
            // [BS*k,1] but keeps the GatherQMM output semantics identical
            // (B = BS*k either way). After down_proj we invert the permutation
            // (substep 6) to restore original token/k order.
            //
            // Returns `Some((sorted_x_4d, sorted_topk_2d, sort_perm))` on the
            // sorted branch; `None` on the default branch.
            let sort_pack_state = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "routing_sort_pack",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Option<(Array, Array, Array)>> {
                    if !use_sorted {
                        // Default broadcast branch: no sort/pack work happens
                        // here. Zero-cost span emit per hard-rule #9. No
                        // arrays produced -> nothing to eval probe.
                        return Ok(None);
                    }
                    // --- Sorted routing path. ---
                    let flat_topk = mlx::ops::shape::reshape(&inds_u32, [bs_k])
                        .context("SparseMoeBlock: reshape inds_u32 to [BS*k]")?;
                    // argsort returns the permutation that sorts flat_topk
                    // ascending by expert id. Stable per MLX semantics. Uint32.
                    let sort_perm = argsort_on(&flat_topk, -1_i32, target)
                        .context("SparseMoeBlock: argsort flat_topk")?; // [BS*k]
                    let sorted_topk_1d = take_along_axis_on(&flat_topk, &sort_perm, -1_i32, target)
                        .context("SparseMoeBlock: take_along_axis sort flat_topk")?;
                    // gather_qmm expects rhs_indices rank-2 to match x rank-4.
                    let sorted_topk_2d =
                        mlx::ops::shape::reshape(&sorted_topk_1d, [bs_k, 1_i32])
                            .context("SparseMoeBlock: reshape sorted_topk to [BS*k, 1]")?;
                    let sorted_token_idx =
                        sorted_token_indices_from_sort_perm(&sort_perm, k, bs_k, target)?;
                    // Physically gather flat_x rows in sorted order.
                    // flat_x: [BS, H], sorted_token_idx: [BS*k] -> [BS*k, H].
                    let sorted_x_2d = take_on(&flat_x, &sorted_token_idx, 0_i32, target)
                        .context("SparseMoeBlock: take flat_x by sorted_token_idx")?;
                    // Promote to rank-4 [BS*k, 1, 1, H] for gather_qmm (r+2 with r=2).
                    let sorted_x_4d = mlx::ops::shape::expand_dims_on(
                        &sorted_x_2d,
                        &[-2_i32, -3_i32][..],
                        target,
                    )
                    .context("SparseMoeBlock: expand_dims sorted_x → [BS*k,1,1,H]")?;
                    // P5h+1 T1: measurement-eval probe (sorted branch only;
                    // default broadcast branch already returned None above).
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&sorted_x_4d, &sorted_topk_2d, &sort_perm])?;
                    }
                    if expert_occupancy_log_enabled() {
                        let expert_ids = sorted_topk_2d
                            .to_vec::<u32>()
                            .context("SparseMoeBlock: expert occupancy to_vec")?;
                        let num_experts_usize = usize::try_from(num_experts)
                            .context("SparseMoeBlock: num_experts usize conversion")?;
                        let mut counts = vec![0_usize; num_experts_usize];
                        for expert_id in expert_ids {
                            let expert = expert_id as usize;
                            if expert >= counts.len() {
                                return Err(anyhow!(
                                    "SparseMoeBlock: expert occupancy id {expert} out of range \
                                     0..{}",
                                    counts.len()
                                ));
                            }
                            counts[expert] += 1;
                        }
                        let nonempty_experts = counts.iter().filter(|count| **count > 0).count();
                        let max_tokens_per_expert = counts.iter().copied().max().unwrap_or(0);
                        let mut nonempty_counts: Vec<usize> =
                            counts.iter().copied().filter(|count| *count > 0).collect();
                        nonempty_counts.sort_unstable();
                        let p95_tokens_per_expert = if nonempty_counts.is_empty() {
                            0
                        } else {
                            let idx = ((nonempty_counts.len() * 95).saturating_sub(1)) / 100;
                            nonempty_counts[idx]
                        };
                        let total_routes = counts.iter().sum::<usize>() as f64;
                        let entropy_bits = if total_routes > 0.0 {
                            counts
                                .iter()
                                .filter(|count| **count > 0)
                                .map(|count| {
                                    let p = *count as f64 / total_routes;
                                    -p * p.log2()
                                })
                                .sum::<f64>()
                        } else {
                            0.0
                        };
                        let mut top5: Vec<(usize, usize)> = counts
                            .iter()
                            .enumerate()
                            .filter_map(|(expert, count)| (*count > 0).then_some((expert, *count)))
                            .collect();
                        top5.sort_by(|left, right| {
                            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
                        });
                        top5.truncate(5);
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        std::hash::Hash::hash(&top5, &mut hasher);
                        let top5_hash = std::hash::Hasher::finish(&hasher);
                        tracing::info!(
                            target: "moe_expert_occupancy",
                            "[p5h+2-e moe_occupancy] layer={layer_idx} bs={bs} k={k} \
                             bs_k={bs_k} nonempty_experts={nonempty_experts} \
                             max_tokens_per_expert={max_tokens_per_expert} \
                             p95_tokens_per_expert={p95_tokens_per_expert} \
                             entropy_bits={entropy_bits:.6} top5_hash={top5_hash:016x} \
                             top5={top5:?}"
                        );
                    }
                    Ok(Some((sorted_x_4d, sorted_topk_2d, sort_perm)))
                },
            )?;

            // Substep 3: gather_qmm_gate_up — gate_proj + up_proj quantized
            // matmuls. Default branch absorbs the [BS,1,1,H] expand_dims on
            // flat_x (gather_qmm input shaping). Sorted branch reuses the
            // pre-packed sorted_x_4d + sorted_topk_2d from substep 2.
            //
            // Returns `(gate_out, up_out, rhs_idx_used, sorted_flag, sort_perm_opt)`
            // where rhs_idx_used + sorted_flag flow into substep 5 (down)
            // and sort_perm_opt flows into substep 6 (unsort).
            let (gate_out, up_out, rhs_idx_used, sorted_flag, sort_perm_opt) =
                crate::core::p5h::try_with_p5h_span_from_current_trace(
                    "gather_qmm_gate_up",
                    || crate::core::p5h::SpanFields {
                        layer_idx: Some(layer_idx),
                        ..Default::default()
                    },
                    || -> Result<(Array, Array, Array, bool, Option<Array>)> {
                        // P5i.a T2: single fused gate+up gather_qmm + slice.
                        // Fused weights lazily built on first forward (worker
                        // thread) for correct Metal stream binding.
                        let i = self.routed.moe_intermediate;
                        let fused = self.routed.fused_gate_up(target)?;
                        if let Some((sorted_x_4d, sorted_topk_2d, sort_perm)) = sort_pack_state {
                            let bs_k_local = sorted_topk_2d.shape().as_slice()[0];
                            // Phase 1 Stage α: cost decomposition sub-spans (sorted branch).
                            let gate_up_out = with_p5i_c_gate_up_child_span(
                                gate_up_child_spans_enabled,
                                "gate_up_gather_qmm_call",
                                layer_idx,
                                || -> Result<Array> {
                                    mlx::quantization::gather_quantized_matmul_on(
                                        &sorted_x_4d,
                                        &fused.weight,
                                        &fused.scales,
                                        fused.biases.as_ref(),
                                        None,
                                        Some(&sorted_topk_2d),
                                        true,
                                        Some(self.routed.group_size),
                                        Some(self.routed.bits),
                                        "affine",
                                        /* sorted_indices */ true,
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: gate_up gather_qmm (sorted, p5h-profile)",
                                    )
                                },
                            )?;
                            let (gate_out, up_out) = with_p5i_c_gate_up_child_span(
                                gate_up_child_spans_enabled,
                                "gate_up_slice_outputs",
                                layer_idx,
                                || -> Result<(Array, Array)> {
                                    let gate_out = slice_on(
                                        &gate_up_out,
                                        [0_i32, 0, 0, 0],
                                        [bs_k_local, 1, 1, i],
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: slice gate_out (sorted, p5h-profile)",
                                    )?;
                                    let up_out = slice_on(
                                        &gate_up_out,
                                        [0_i32, 0, 0, i],
                                        [bs_k_local, 1, 1, 2 * i],
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: slice up_out (sorted, p5h-profile)",
                                    )?;
                                    Ok((gate_out, up_out))
                                },
                            )?;
                            // P5h+1 T1: measurement-eval probe (sorted branch).
                            if crate::core::p5h::is_measurement_eval_probes_active() {
                                mlx::transforms::eval(&[
                                    &gate_out,
                                    &up_out,
                                    &sorted_topk_2d,
                                    &sort_perm,
                                ])?;
                            }
                            Ok((gate_out, up_out, sorted_topk_2d, true, Some(sort_perm)))
                        } else {
                            // --- Default broadcast path. ---
                            // Phase 1 Stage α: cost decomposition sub-spans (default branch).
                            let x_in = with_p5i_c_gate_up_child_span(
                                gate_up_child_spans_enabled,
                                "gate_up_input_shape_prep",
                                layer_idx,
                                || -> Result<Array> {
                                    mlx::ops::shape::expand_dims_on(
                                        &flat_x,
                                        &[-2_i32, -3_i32][..],
                                        target,
                                    )
                                    .context("SparseMoeBlock: expand_dims flat_x → [BS,1,1,H]")
                                },
                            )?;
                            let gate_up_out = with_p5i_c_gate_up_child_span(
                                gate_up_child_spans_enabled,
                                "gate_up_gather_qmm_call",
                                layer_idx,
                                || -> Result<Array> {
                                    mlx::quantization::gather_quantized_matmul_on(
                                        &x_in,
                                        &fused.weight,
                                        &fused.scales,
                                        fused.biases.as_ref(),
                                        None,
                                        Some(&inds_u32),
                                        true,
                                        Some(self.routed.group_size),
                                        Some(self.routed.bits),
                                        "affine",
                                        /* sorted_indices */ false,
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: gate_up gather_qmm (default, p5h-profile)",
                                    )
                                },
                            )?;
                            let (gate_out, up_out) = with_p5i_c_gate_up_child_span(
                                gate_up_child_spans_enabled,
                                "gate_up_slice_outputs",
                                layer_idx,
                                || -> Result<(Array, Array)> {
                                    let gate_out = slice_on(
                                        &gate_up_out,
                                        [0_i32, 0, 0, 0],
                                        [bs, k, 1, i],
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: slice gate_out (default, p5h-profile)",
                                    )?;
                                    let up_out = slice_on(
                                        &gate_up_out,
                                        [0_i32, 0, 0, i],
                                        [bs, k, 1, 2 * i],
                                        target,
                                    )
                                    .context(
                                        "SparseMoeBlock: slice up_out (default, p5h-profile)",
                                    )?;
                                    Ok((gate_out, up_out))
                                },
                            )?;
                            // P5h+1 T1: measurement-eval probe (default branch).
                            if crate::core::p5h::is_measurement_eval_probes_active() {
                                mlx::transforms::eval(&[&gate_out, &up_out, &inds_u32])?;
                            }
                            Ok((gate_out, up_out, inds_u32, false, None))
                        }
                    },
                )?;

            // Substep 4: swiglu_activation — silu(gate) * up.
            // gate_out, up_out shapes:
            //   sorted path:  [BS*k, 1, 1, moe_inter]
            //   default path: [BS, k, 1, moe_inter]
            // Both element-wise — same code path.
            let act = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "swiglu_activation",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let act = self.swiglu_on(&gate_out, &up_out)?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&act])?;
                    }
                    Ok(act)
                },
            )?;

            // Substep 5: gather_qmm_down — down_proj quantized matmul.
            // Output shape:
            //   sorted path:  [BS*k, 1, 1, H]  (in sorted order)
            //   default path: [BS, k, 1, H]    (in original order)
            let down_out_4d = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "gather_qmm_down",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let down_out = mlx::quantization::gather_quantized_matmul_on(
                        &act,
                        &self.routed.down_weight,
                        &self.routed.down_scales,
                        self.routed.down_biases.as_ref(),
                        None,
                        Some(&rhs_idx_used),
                        true,
                        Some(self.routed.group_size),
                        Some(self.routed.bits),
                        "affine",
                        sorted_flag,
                        target,
                    )
                    .context("SparseMoeBlock: down_proj gather_qmm")?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&down_out])?;
                    }
                    Ok(down_out)
                },
            )?;

            // Substep 6: routing_unsort_weighted_reduce — unpack (sorted
            // branch: argsort inv_perm + take + reshape; default branch:
            // squeeze) into [BS, k, H], then weight by router scores
            // and reduce across k → [BS, H].
            let routed_y = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "routing_unsort_weighted_reduce",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let down_out = if let Some(sort_perm) = sort_perm_opt {
                        // Sorted path: invert the permutation to restore
                        // original token/k order. inv_perm = argsort(sort_perm).
                        let inv_perm = argsort_on(&sort_perm, -1_i32, target)
                            .context("SparseMoeBlock: argsort inv permutation")?;
                        // Reshape over squeeze: dims are statically known
                        // singletons here, so reshape becomes a graph metadata
                        // change with no op-node, cheaper than invoking squeeze.
                        let down_out_2d = mlx::ops::shape::reshape(&down_out_4d, [bs_k, h])
                            .context("SparseMoeBlock: reshape sorted down_out to [BS*k, H]")?;
                        let unsorted_2d = take_on(&down_out_2d, &inv_perm, 0_i32, target)
                            .context("SparseMoeBlock: take inv_perm to restore order")?;
                        mlx::ops::shape::reshape(&unsorted_2d, [bs, k, h])
                            .context("SparseMoeBlock: reshape unsorted to [BS, k, H]")?
                    } else {
                        mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)
                            .context("SparseMoeBlock: squeeze down_proj dim -2")?
                    };
                    // scores: [BS, k] -> [BS, k, 1] for broadcast.
                    let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)
                        .context("SparseMoeBlock: expand scores dim")?;
                    let weighted = &down_out * &scores_unsq;
                    let routed_y = mlx::ops::sum_on(&weighted, -2_i32, false, target)
                        .context("SparseMoeBlock: sum across k")?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&routed_y])?;
                    }
                    Ok(routed_y)
                },
            )?;

            // Substep 7: shared_expert — independent LinearMLP + sigmoid gate
            // operating on the unpacked flat_x (NOT the sorted slabs).
            let shared_gated = crate::core::p5h::try_with_p5h_span_from_current_trace(
                "shared_expert",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let shared_y = self
                        .shared_expert
                        .forward_on(&flat_x, target)
                        .context("SparseMoeBlock: shared_expert forward")?; // [BS, H]
                    let gate_logit = self
                        .shared_expert_gate
                        .forward_on(&flat_x, target)
                        .context("SparseMoeBlock: shared_expert_gate forward")?; // [BS, 1]
                    let gate_sig2 = gate_logit
                        .sigmoid_on(target)
                        .context("SparseMoeBlock: shared gate sigmoid")?; // [BS, 1]
                    let shared_gated = &shared_y * &gate_sig2; // [BS, H]
                                                               // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&shared_gated])?;
                    }
                    Ok(shared_gated)
                },
            )?;

            // Substep 8: moe_output_sum — combine routed + shared and reshape
            // back to [B, S, H].
            crate::core::p5h::try_with_p5h_span_from_current_trace(
                "moe_output_sum",
                || crate::core::p5h::SpanFields {
                    layer_idx: Some(layer_idx),
                    ..Default::default()
                },
                || -> Result<Array> {
                    let out_flat = &routed_y + &shared_gated; // [BS, H]
                    let out = mlx::ops::shape::reshape(&out_flat, [b, s, h])
                        .context("SparseMoeBlock: reshape [BS,H] → [B,S,H]")?;
                    // P5h+1 T1: measurement-eval probe.
                    if crate::core::p5h::is_measurement_eval_probes_active() {
                        mlx::transforms::eval(&[&out])?;
                    }
                    Ok(out)
                },
            )
        }

        #[cfg(not(feature = "p5h-profile"))]
        {
            // Production build: layer_idx is signature-only plumbing (consumed
            // only by the p5h-profile substep spans above).
            let _ = layer_idx;

            // (1) Router: Linear -> [BS, E], then top-k expert selection.
            let logits = self.router_gate.forward_on(&flat_x, target)?; // [BS, E]

            // (2) Top-k selection via argpartition, plus score normalization
            // matching `norm_topk_prob`.
            // argpartition is preferable to topk here: we don't need the top-k
            // elements sorted internally (each is independently weight-summed via
            // gather_qmm); we only need to know which k indices to gather. MLX
            // argpartition is applied after the full router softmax so this
            // mirrors the reference inference path; normalization is then
            // applied only when the checkpoint config requests it.
            let (scores, inds_u32) = router_topk_scores_and_indices(
                &logits,
                k,
                num_experts,
                self.norm_topk_prob,
                target,
            )?;

            // (3) Routed SwiGLU via gather_quantized_matmul_on (G1 path).
            //
            // Two routing strategies are dispatched here based on `bs * k`:
            //
            //   - Sorted-flat path (BS*k >= SORTED_ROUTING_MIN_BS_K): pre-sort
            //     tokens by expert id and pass `sorted_indices=true`, matching
            //     MLX-LM `SwitchGLU` for `indices.size >= 64`. We physically
            //     gather `flat_x` rows by the sorted token id so each
            //     (token,expert-slot) is its own x-row; this changes x.shape
            //     from [BS,1,1,H] to [BS*k,1,1,H] and `rhs_indices` from [BS,k]
            //     to [BS*k,1] but keeps the GatherQMM output semantics identical
            //     (B = BS*k both cases). After down_proj we invert the permutation
            //     to restore original token/k order before the score-weighted sum.
            //
            //   - Default broadcast path (BS*k < SORTED_ROUTING_MIN_BS_K): keep `x` as [BS,1,1,H]
            //     and let MLX broadcast `lhs_indices` from [BS,1] to [BS,k].
            //     Avoids the argsort/scatter overhead on tiny decode batches
            //     below the reference sorting threshold.
            //
            // MLX gather_qmm API contract (unchanged from T0): when `rhs_indices`
            // has rank-r, the input `x` must have rank r+2 (the leading r dims
            // are broadcast against rhs_indices, the trailing 2 are matrix dims).

            let (gate_out, up_out, rhs_idx_used, sorted_flag, sort_perm_opt) = if use_sorted {
                // --- Sorted routing path. ---

                // Flatten topk indices: [BS, k] -> [BS*k].
                let flat_topk = mlx::ops::shape::reshape(&inds_u32, [bs_k])
                    .context("SparseMoeBlock: reshape inds_u32 to [BS*k]")?;

                // argsort returns the permutation that sorts flat_topk ascending
                // by expert id. Stable per MLX semantics. Dtype Uint32.
                let sort_perm = argsort_on(&flat_topk, -1_i32, target)
                    .context("SparseMoeBlock: argsort flat_topk")?; // [BS*k]

                // sorted_topk: the actual expert id per sorted slot. This is
                // what gets passed as rhs_indices so right_sorted_ is true.
                // P5i.a T1 C1: keep rank-1 (no [BS*k,1] reshape). Paired with
                // sorted_x rank-3 below so the default lhs_indices (x.shape()[..-2]
                // = [BS*k]) matches rhs_indices shape exactly; broadcast/copy is a
                // no-op. fast-path entry condition x.size()/x.shape(-2)/x.shape(-1)
                // == indices.size() ⇒ BS*k == BS*k preserved.
                let sorted_topk = take_along_axis_on(&flat_topk, &sort_perm, -1_i32, target)
                    .context("SparseMoeBlock: take_along_axis sort flat_topk")?;

                let sorted_token_idx =
                    sorted_token_indices_from_sort_perm(&sort_perm, k, bs_k, target)?;

                // Physically gather flat_x rows in sorted order.
                // flat_x: [BS, H], sorted_token_idx: [BS*k] -> [BS*k, H].
                let sorted_x_2d = take_on(&flat_x, &sorted_token_idx, 0_i32, target)
                    .context("SparseMoeBlock: take flat_x by sorted_token_idx")?;
                // P5i.a T1 C1: promote to rank-3 [BS*k, 1, H] for gather_qmm
                // (r+2 with r=1 instead of r=2). MLX gather_qmm_rhs fast path
                // still triggers: x.shape(-2)=1 ⇒ M=1; B = out.size()/M/N =
                // BS*k. Default lhs_indices shape becomes x.shape()[..-2]=[BS*k]
                // which broadcasts trivially against rhs_indices [BS*k] (also
                // rank-1, see C1 above), so x.size()/x.shape(-2)/x.shape(-1)
                // == BS*k == indices.size() — no broadcast/copy needed.
                // Replaces double expand_dims (was [-2,-3]) with single -2 axis.
                let sorted_x_3d = mlx::ops::shape::expand_dims_on(&sorted_x_2d, -2_i32, target)
                    .context("SparseMoeBlock: expand_dims sorted_x → [BS*k,1,H]")?;

                // P5i.a T2: single fused gate+up gather_qmm + slice.
                // Output shape: [BS*k, 1, 2*I]; slice along axis=-1 into
                // gate_out [BS*k, 1, I] and up_out [BS*k, 1, I].
                let fused = self.routed.fused_gate_up(target)?;
                let gate_up_out = mlx::quantization::gather_quantized_matmul_on(
                    &sorted_x_3d,
                    &fused.weight,
                    &fused.scales,
                    fused.biases.as_ref(),
                    None,
                    Some(&sorted_topk),
                    true,
                    Some(self.routed.group_size),
                    Some(self.routed.bits),
                    "affine",
                    /* sorted_indices */ true,
                    target,
                )
                .context("SparseMoeBlock: gate_up gather_qmm (sorted)")?;
                let i = self.routed.moe_intermediate;
                let gate_out = slice_on(&gate_up_out, [0_i32, 0, 0], [bs_k, 1, i], target)
                    .context("SparseMoeBlock: slice gate_out from gate_up (sorted)")?;
                let up_out = slice_on(&gate_up_out, [0_i32, 0, i], [bs_k, 1, 2 * i], target)
                    .context("SparseMoeBlock: slice up_out from gate_up (sorted)")?;

                (gate_out, up_out, sorted_topk, true, Some(sort_perm))
            } else {
                // --- Default broadcast path (Stage 1 final). ---
                let x_in = mlx::ops::shape::expand_dims_on(&flat_x, &[-2_i32, -3_i32][..], target)
                    .context("SparseMoeBlock: expand_dims flat_x → [BS,1,1,H]")?; // [BS, 1, 1, H]

                // P5i.a T2: single fused gate+up gather_qmm + slice.
                // Output shape: [BS, k, 1, 2*I]; slice along axis=-1 into
                // gate_out [BS, k, 1, I] and up_out [BS, k, 1, I].
                let fused = self.routed.fused_gate_up(target)?;
                let gate_up_out = mlx::quantization::gather_quantized_matmul_on(
                    &x_in,
                    &fused.weight,
                    &fused.scales,
                    fused.biases.as_ref(),
                    None,
                    Some(&inds_u32),
                    true,
                    Some(self.routed.group_size),
                    Some(self.routed.bits),
                    "affine",
                    false,
                    target,
                )
                .context("SparseMoeBlock: gate_up gather_qmm")?; // [BS, k, 1, 2*I]
                let i = self.routed.moe_intermediate;
                let gate_out = slice_on(&gate_up_out, [0_i32, 0, 0, 0], [bs, k, 1, i], target)
                    .context("SparseMoeBlock: slice gate_out from gate_up (default)")?;
                let up_out = slice_on(&gate_up_out, [0_i32, 0, 0, i], [bs, k, 1, 2 * i], target)
                    .context("SparseMoeBlock: slice up_out from gate_up (default)")?;

                (gate_out, up_out, inds_u32, false, None)
            };

            // SwiGLU activation: silu(gate) * up  where silu(z) = z * sigmoid(z)
            // gate_out, up_out:
            //   sorted path:  [BS*k, 1, 1, moe_inter]
            //   default path: [BS, k, 1, moe_inter]
            // Both element-wise — same code path.
            let act = self.swiglu_on(&gate_out, &up_out)?;

            let down_out_raw = mlx::quantization::gather_quantized_matmul_on(
                &act,
                &self.routed.down_weight,
                &self.routed.down_scales,
                self.routed.down_biases.as_ref(),
                None,
                Some(&rhs_idx_used),
                true,
                Some(self.routed.group_size),
                Some(self.routed.bits),
                "affine",
                sorted_flag,
                target,
            )
            .context("SparseMoeBlock: down_proj gather_qmm")?;
            // down_out_raw shape (post-T1 C1 rank-3 sorted simplification):
            //   sorted path:  [BS*k, 1, H]      (in sorted order)
            //   default path: [BS, k, 1, H]     (in original order, unchanged)

            // (6) Weight by router scores and reduce over k.
            //
            // Both branches converge on `down_out: [BS, k, H]` so the score
            // weighting + reduce is shared.
            let down_out = if let Some(sort_perm) = sort_perm_opt {
                // Sorted path: squeeze [BS*k, 1, H] -> [BS*k, H], then invert
                // the permutation to restore original token/k order before
                // reshape to [BS, k, H]. inv_perm = argsort(sort_perm).
                let inv_perm = argsort_on(&sort_perm, -1_i32, target)
                    .context("SparseMoeBlock: argsort inv permutation")?;
                // Reshape over squeeze: dims are statically known singletons here, so
                // reshape becomes a graph metadata change with no op-node, cheaper than
                // invoking squeeze.
                let down_out_2d = mlx::ops::shape::reshape(&down_out_raw, [bs_k, h])
                    .context("SparseMoeBlock: reshape sorted down_out to [BS*k, H]")?;
                let unsorted_2d = take_on(&down_out_2d, &inv_perm, 0_i32, target)
                    .context("SparseMoeBlock: take inv_perm to restore order")?;
                mlx::ops::shape::reshape(&unsorted_2d, [bs, k, h])
                    .context("SparseMoeBlock: reshape unsorted to [BS, k, H]")?
            } else {
                mlx::ops::shape::squeeze_on(&down_out_raw, &[-2_i32][..], target)
                    .context("SparseMoeBlock: squeeze down_proj dim -2")?
            };

            let routed_y = {
                // scores: [BS, k] -> [BS, k, 1] for broadcast with [BS, k, H].
                let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)
                    .context("SparseMoeBlock: expand scores dim")?;
                let weighted = &down_out * &scores_unsq;
                mlx::ops::sum_on(&weighted, -2_i32, false, target)
                    .context("SparseMoeBlock: sum across k")?
            };

            // (7) Shared expert with independent sigmoid gate.
            let shared_y = self
                .shared_expert
                .forward_on(&flat_x, target)
                .context("SparseMoeBlock: shared_expert forward")?; // [BS, H]
            let gate_logit = self
                .shared_expert_gate
                .forward_on(&flat_x, target)
                .context("SparseMoeBlock: shared_expert_gate forward")?; // [BS, 1]
            let gate_sig2 = gate_logit
                .sigmoid_on(target)
                .context("SparseMoeBlock: shared gate sigmoid")?; // [BS, 1]
            let shared_gated = &shared_y * &gate_sig2; // [BS, H]

            // (8) Combine routed + shared, then reshape back to [B, S, H].
            let out_flat = &routed_y + &shared_gated; // [BS, H]
            let out = mlx::ops::shape::reshape(&out_flat, [b, s, h])
                .context("SparseMoeBlock: reshape [BS,H] → [B,S,H]")?;

            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_routing_threshold_matches_mlx_switch_glu_contract() {
        assert_eq!(SORTED_ROUTING_MIN_BS_K, 64);
    }

    /// Compile-time check: RoutedExperts fields are public and Array can be
    /// referenced through them. Numerical correctness deferred to T5
    /// integration tests under tests/p5_qwen35_moe_*.rs. Those tests
    /// observe ironmlx output and may also record output from external
    /// reference implementations for triangulation — but ironmlx output
    /// is treated as the source of truth for its own behavior; any
    /// observed divergence from external references is informational,
    /// not a regression signal.
    #[test]
    fn routed_experts_field_access_compiles() {
        fn _accept_ref(_e: &RoutedExperts) -> i32 {
            42
        }
    }

    /// Compile-time check: SparseMoeBlock and RoutedExperts are public types
    /// that can be named from external modules. No real checkpoint needed here.
    #[test]
    fn sparse_moe_types_are_public() {
        // Trivial: ensure module-level types are accessible and the module builds.
        let _check: fn(&RoutedExperts) -> i32 = |_| 0;
    }

    #[test]
    fn sorted_token_indices_follow_sort_permutation() -> Result<()> {
        let sort_perm: Array = (&[5_u32, 0, 7, 3, 2, 1, 4, 6][..], [8])
            .try_into()
            .map_err(|e| anyhow!("build sort_perm: {e}"))?;
        let out =
            sorted_token_indices_from_sort_perm(&sort_perm, 2, 8, mlx::StreamOrDevice::default())?;

        assert_eq!(out.to_vec::<u32>()?, vec![2, 0, 3, 1, 1, 0, 2, 3]);
        Ok(())
    }

    #[test]
    fn sorted_token_indices_handle_high_exact_f32_range() -> Result<()> {
        let max = (MAX_EXACT_U32_IN_F32 - 1) as u32;
        let sort_perm: Array = (&[max, max - 1, max - 7, 15_u32][..], [4])
            .try_into()
            .map_err(|e| anyhow!("build high sort_perm: {e}"))?;
        let out = sorted_token_indices_from_sort_perm(
            &sort_perm,
            8,
            MAX_EXACT_U32_IN_F32,
            mlx::StreamOrDevice::default(),
        )?;

        assert_eq!(
            out.to_vec::<u32>()?,
            vec![max / 8, (max - 1) / 8, (max - 7) / 8, 15 / 8]
        );
        Ok(())
    }

    #[test]
    fn router_scores_do_not_renormalize_when_norm_topk_false() -> Result<()> {
        let logits: Array = (&[4.0_f32, 3.0, 1.0, 0.0][..], [1, 4])
            .try_into()
            .map_err(|e| anyhow!("build logits: {e}"))?;
        let (scores, inds) =
            router_topk_scores_and_indices(&logits, 2, 4, false, StreamOrDevice::default())?;

        let got = scores.to_vec::<f32>()?;
        let ids = inds.to_vec::<u32>()?;
        let denom = 4.0_f32.exp() + 3.0_f32.exp() + 1.0_f32.exp() + 0.0_f32.exp();
        let logits_by_id = [4.0_f32, 3.0, 1.0, 0.0];
        let want: Vec<f32> = ids
            .iter()
            .map(|&id| logits_by_id[id as usize].exp() / denom)
            .collect();

        for (got, want) in got.iter().zip(want.iter()) {
            approx::assert_abs_diff_eq!(got, want, epsilon = 1e-5);
        }
        assert!(got.iter().sum::<f32>() < 1.0);
        Ok(())
    }

    #[test]
    fn router_scores_renormalize_when_norm_topk_true() -> Result<()> {
        let logits: Array = (&[4.0_f32, 3.0, 1.0, 0.0][..], [1, 4])
            .try_into()
            .map_err(|e| anyhow!("build logits: {e}"))?;
        let (scores, inds) =
            router_topk_scores_and_indices(&logits, 2, 4, true, StreamOrDevice::default())?;

        let got = scores.to_vec::<f32>()?;
        let ids = inds.to_vec::<u32>()?;
        let logits_by_id = [4.0_f32, 3.0, 1.0, 0.0];
        let selected_exp: Vec<f32> = ids
            .iter()
            .map(|&id| logits_by_id[id as usize].exp())
            .collect();
        let denom: f32 = selected_exp.iter().sum();
        let want: Vec<f32> = selected_exp.into_iter().map(|v| v / denom).collect();

        for (got, want) in got.iter().zip(want.iter()) {
            approx::assert_abs_diff_eq!(got, want, epsilon = 1e-5);
        }
        approx::assert_abs_diff_eq!(got.iter().sum::<f32>(), 1.0, epsilon = 1e-5);
        Ok(())
    }
}
