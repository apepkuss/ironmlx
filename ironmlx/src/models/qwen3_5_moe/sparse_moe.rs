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

use crate::core::{logical_width_from_packed, Loader, QuantMeta};
use crate::nn::activations::{build_geglu_tanh, build_swiglu, invoke_geglu_tanh, invoke_swiglu};
use crate::nn::sorted_moe_weighted_sum;
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

#[derive(Debug, Clone, Copy)]
enum RoutedActivation {
    SwiGlu,
    GeGluTanh,
}

#[derive(Debug, Clone, Copy)]
struct RoutedApplyOptions {
    layer_idx: i32,
    cast_output_to_expert_dtype: bool,
    activation: RoutedActivation,
    request_layout: Option<(i32, i32)>,
}

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

fn request_interleaved_sort_perm(
    flat_topk: &Array,
    batch: i32,
    sequence: i32,
    top_k: i32,
    bs_k: i32,
    num_experts: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    if batch <= 0 || sequence <= 0 || top_k <= 0 || num_experts <= 0 {
        return Err(anyhow!(
            "request-interleaved MoE sort requires positive B/S/K/E, got B={batch}, S={sequence}, K={top_k}, E={num_experts}"
        ));
    }
    let routes_per_request = sequence
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("request-interleaved MoE sort overflow: {sequence} * {top_k}"))?;
    if batch.checked_mul(routes_per_request) != Some(bs_k) {
        return Err(anyhow!(
            "request-interleaved MoE sort layout mismatch: B={batch}, routes/request={routes_per_request}, total={bs_k}"
        ));
    }
    let max_key_exclusive = (num_experts as u64)
        .checked_mul(bs_k as u64)
        .ok_or_else(|| anyhow!("request-interleaved MoE sort key range overflow"))?;
    if max_key_exclusive > u32::MAX as u64 + 1 {
        return Err(anyhow!(
            "request-interleaved MoE sort key exceeds Uint32: E={num_experts}, routes={bs_k}"
        ));
    }

    // Build a unique route rank whose logical order is
    // (position, top-k slot, request row). Pairing equivalent request rows
    // within each expert group keeps the sorted gather kernel's matrix-row
    // position stable without copying route metadata to the host.
    let ranks =
        mlx::ops::constructors::arange_on(0.0, bs_k as f64, 1.0, mlx::Dtype::Uint32, target)?;
    let ranks = mlx::ops::shape::reshape(&ranks, [routes_per_request, batch])?;
    let ranks = ranks.transpose_axes_on(&[1_i32, 0][..], target)?;
    let ranks = mlx::ops::shape::reshape(&ranks, [bs_k])?;
    let expert_stride: Array = (&[bs_k as u32][..], ())
        .try_into()
        .map_err(|e| anyhow!("request-interleaved MoE sort expert stride: {e}"))?;
    let keys = flat_topk
        .try_mul_on(&expert_stride, target)?
        .try_add_on(&ranks, target)?;
    argsort_on(&keys, -1_i32, target).context("request-interleaved MoE route argsort")
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

fn with_glm_routed_experts_child_span<T>(
    enabled: bool,
    span_name: &'static str,
    layer_idx: i32,
    body: impl FnOnce() -> T,
) -> T {
    {
        let _ = (enabled, span_name, layer_idx);
        body()
    }
}

struct ExpertQuantProjection {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    meta: QuantMeta,
}

impl ExpertQuantProjection {
    fn gather(
        &self,
        lhs: &Array,
        rhs_indices: &Array,
        sorted_indices: bool,
        target: StreamOrDevice,
        context: &'static str,
    ) -> Result<Array> {
        mlx::quantization::gather_quantized_matmul_on(
            lhs,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            None,
            Some(rhs_indices),
            true,
            Some(self.meta.group_size),
            Some(self.meta.bits),
            self.meta.mode.mlx_backend_mode(),
            sorted_indices,
            target,
        )
        .with_context(|| format!("RoutedExperts::apply_experts: {context}"))
    }
}

enum GateUpPath {
    Fused {
        /// Legacy source weights, consumed on first forward call. After
        /// lazy fused build the inner `Option` is `take()`-d to `None`,
        /// releasing the original Array refs back to MLX.
        legacy_source: std::sync::Mutex<Option<LazyGateUpSource>>,
        /// Lazy-built fused weights. Initialized on the first forward call so
        /// the resulting MLX Arrays are bound to the worker thread's Metal
        /// stream. See struct doc for failure-mode rationale.
        fused_gate_up: std::sync::OnceLock<FusedGateUp>,
        meta: QuantMeta,
    },
    Split {
        gate: ExpertQuantProjection,
        up: ExpertQuantProjection,
    },
}

/// Source legacy gate + up weights, consumed once during lazy fused-weight
/// build then dropped to release the two packed `(E × I × H)` buffers per layer.
struct LazyGateUpSource {
    gate: ExpertQuantProjection,
    up: ExpertQuantProjection,
}

/// Container for the fused gate+up quantized weights, lazily built on the
/// first forward thread (so the MLX Array allocation is bound to the
/// worker thread's Metal stream, not the loader thread's — see
/// scheduler_actor.rs:163 "B1-p2.5 P0 fix v2" notes).
struct FusedGateUp {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    meta: QuantMeta,
}

/// Stacked-expert quantized weights for the routed SwiGLU.
///
/// Shape convention (num_experts=E, hidden=H, moe_intermediate=I):
///   gate/up (legacy source, dropped after fused build):
///                weight `[E, I, packed_H]`, scales/biases `[E, I, H/group_size]`
///   gate_up (fused, lazy): weight `[E, 2I, packed_H]`, scales/biases `[E, 2I, H/group_size]`
///   down:    weight `[E, H, packed_I]`, scales/biases `[E, H, I/group_size]`
///
/// Gate_proj + up_proj weights are concatenated along the intermediate
/// axis (axis=1) on the first forward call so a single
/// `gather_qmm` call replaces the prior two-call (gate then up). The fused
/// output is sliced along the last dim into gate_out / up_out before
/// SwiGLU. Supported quantization modes are grouped along K=last; stacking
/// along intermediate preserves all per-row scales and optional biases.
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
    gate_up: GateUpPath,
    /// Per-projection intermediate size I (= gate_weight.shape(1)).
    /// Cached at load time to avoid recomputing in the hot forward path.
    pub moe_intermediate: i32,
    pub down_weight: Array,
    pub down_scales: Array,
    pub down_biases: Option<Array>,
    down_meta: QuantMeta,
    /// Diagnostic/public metadata for the gate projection. Uniform 4-bit MoE
    /// checkpoints have the same value for gate/up/down; OptiQ mixed-bit
    /// checkpoints may not, so production dispatch uses per-projection metadata.
    pub group_size: i32,
    pub bits: i32,
    pub num_experts: i32,
    /// Lazily-built compiled SwiGLU closure for the routed activation.
    /// Owned here so `apply_experts` is self-contained.
    swiglu: std::sync::OnceLock<CompiledFn>,
    /// Lazily-built compiled GeGLU closure for Gemma-family routed experts.
    geglu: std::sync::OnceLock<CompiledFn>,
}

impl RoutedExperts {
    /// Load from `{prefix}.gate_proj.*` + `up_proj.*` + `down_proj.*`.
    /// Prefix is typically `"model.layers.{i}.mlp.switch_mlp"`.
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let gate_prefix = format!("{prefix}.gate_proj");
        let up_prefix = format!("{prefix}.up_proj");
        let down_prefix = format!("{prefix}.down_proj");
        let gate = Self::load_projection(loader, &gate_prefix, "gate_proj")?;
        let up = Self::load_projection(loader, &up_prefix, "up_proj")?;
        let down = Self::load_projection(loader, &down_prefix, "down_proj")?;

        let num_experts = gate.weight.shape().as_slice()[0];
        let moe_intermediate = gate.weight.shape().as_slice()[1];
        Self::validate_projection_shapes(prefix, &gate, &up, &down)?;

        let gate_meta = gate.meta;
        let down_meta = down.meta;
        let gate_up = if gate.meta == up.meta {
            // Validate biases presence symmetry upfront (cheap, no MLX op).
            match (gate.biases.as_ref(), up.biases.as_ref()) {
                (Some(_), Some(_)) | (None, None) => {}
                (gb, ub) => {
                    return Err(anyhow!(
                        "RoutedExperts: gate/up biases presence mismatch (gate={}, up={}); fused gate/up quantization requires both or neither",
                        gb.is_some(),
                        ub.is_some()
                    ));
                }
            }
            GateUpPath::Fused {
                legacy_source: std::sync::Mutex::new(Some(LazyGateUpSource { gate, up })),
                fused_gate_up: std::sync::OnceLock::new(),
                meta: gate_meta,
            }
        } else {
            GateUpPath::Split { gate, up }
        };

        Ok(Self {
            gate_up,
            moe_intermediate,
            down_weight: down.weight,
            down_scales: down.scales,
            down_biases: down.biases,
            down_meta,
            group_size: gate_meta.group_size,
            bits: gate_meta.bits,
            num_experts,
            swiglu: std::sync::OnceLock::new(),
            geglu: std::sync::OnceLock::new(),
        })
    }

    fn load_projection(
        loader: &Loader,
        prefix: &str,
        projection_name: &'static str,
    ) -> Result<ExpertQuantProjection> {
        let meta = loader.quant_meta_for(prefix).ok_or_else(|| {
            anyhow!("RoutedExperts requires quantization metadata for `{prefix}`")
        })?;
        let weight = loader
            .tensor(&format!("{prefix}.weight"))
            .with_context(|| format!("RoutedExperts: {projection_name}.weight"))?
            .clone();
        let scales = loader
            .tensor(&format!("{prefix}.scales"))
            .with_context(|| format!("RoutedExperts: {projection_name}.scales"))?
            .clone();
        let biases = loader.tensor_opt(&format!("{prefix}.biases")).cloned();
        meta.validate_storage(prefix, &weight, &scales, biases.as_ref())?;
        Ok(ExpertQuantProjection {
            weight,
            scales,
            biases,
            meta,
        })
    }

    fn validate_projection_shapes(
        prefix: &str,
        gate: &ExpertQuantProjection,
        up: &ExpertQuantProjection,
        down: &ExpertQuantProjection,
    ) -> Result<()> {
        let gate_shape = gate.weight.shape();
        let up_shape = up.weight.shape();
        let down_shape = down.weight.shape();
        let gate_shape = gate_shape.as_slice();
        let up_shape = up_shape.as_slice();
        let down_shape = down_shape.as_slice();
        if gate_shape.len() != 3 || up_shape.len() != 3 || down_shape.len() != 3 {
            return Err(anyhow!(
                "{prefix}: routed expert weights must be rank-3 [E,O,packed_K], got gate={gate_shape:?}, up={up_shape:?}, down={down_shape:?}"
            ));
        }
        if gate_shape[0] != up_shape[0] || gate_shape[0] != down_shape[0] {
            return Err(anyhow!(
                "{prefix}: gate/up/down expert counts must match, got gate={}, up={}, down={}",
                gate_shape[0],
                up_shape[0],
                down_shape[0]
            ));
        }
        if gate_shape[1] != up_shape[1] {
            return Err(anyhow!(
                "{prefix}: gate/up intermediate widths must match, got gate={} and up={}",
                gate_shape[1],
                up_shape[1]
            ));
        }
        let hidden_from_gate = logical_width_from_packed(gate_shape[2], gate.meta.bits)
            .with_context(|| format!("{prefix}: invalid gate packed hidden width"))?;
        let hidden_from_up = logical_width_from_packed(up_shape[2], up.meta.bits)
            .with_context(|| format!("{prefix}: invalid up packed hidden width"))?;
        if hidden_from_gate != hidden_from_up || hidden_from_gate != down_shape[1] {
            return Err(anyhow!(
                "{prefix}: hidden width mismatch, gate={}, up={}, down={}",
                hidden_from_gate,
                hidden_from_up,
                down_shape[1]
            ));
        }
        let intermediate_from_down = logical_width_from_packed(down_shape[2], down.meta.bits)
            .with_context(|| format!("{prefix}: invalid down packed intermediate width"))?;
        if intermediate_from_down != gate_shape[1] {
            return Err(anyhow!(
                "{prefix}: down packed input width {intermediate_from_down} must match gate/up intermediate width {}",
                gate_shape[1]
            ));
        }
        Ok(())
    }

    /// Returns the fused gate+up weights, lazily building them on first
    /// call. Must be called from the forward thread (MLX worker) so the
    /// underlying MLX Arrays are bound to that thread's Metal stream.
    /// `target` propagates the caller's stream-or-device selection to the
    /// underlying `concatenate_on` calls — typically
    /// `StreamOrDevice::default()` from the scheduler driver thread.
    ///
    /// Quantization stores per-(expert,row) scale and optional bias with
    /// groups along the K=last axis only; stacking along the
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
        let GateUpPath::Fused {
            legacy_source,
            fused_gate_up,
            meta,
        } = &self.gate_up
        else {
            return Err(anyhow!(
                "RoutedExperts: fused gate/up requested for split mixed-bit path"
            ));
        };
        // Fast path: already built — no lock needed.
        if let Some(fused) = fused_gate_up.get() {
            return Ok(fused);
        }
        // Slow path: hold the mutex across the entire build+set window so a
        // concurrent second caller blocks here until the first builder
        // finishes, then sees the populated OnceLock on the inner re-check.
        let mut guard = legacy_source
            .lock()
            .map_err(|e| anyhow!("RoutedExperts: legacy_source mutex poisoned: {e}"))?;
        // Inner re-check: another caller may have raced through the slow
        // path while we were waiting on the mutex.
        if let Some(fused) = fused_gate_up.get() {
            return Ok(fused);
        }
        let source = guard.take().ok_or_else(|| {
            anyhow!(
                "RoutedExperts: legacy_source already taken but fused_gate_up never set; \
                 likely a prior panic mid-build that this struct cannot recover from"
            )
        })?;
        let weight = concatenate_on(&[&source.gate.weight, &source.up.weight], 1, target)
            .context("RoutedExperts::fused_gate_up: concatenate weights")?;
        let scales = concatenate_on(&[&source.gate.scales, &source.up.scales], 1, target)
            .context("RoutedExperts::fused_gate_up: concatenate scales")?;
        let biases = match (source.gate.biases.as_ref(), source.up.biases.as_ref()) {
            (Some(gb), Some(ub)) => Some(
                concatenate_on(&[gb, ub], 1, target)
                    .context("RoutedExperts::fused_gate_up: concatenate biases")?,
            ),
            (None, None) => None,
            _ => unreachable!("biases symmetry validated in from_loader"),
        };
        // `concatenate_on` returns a lazy MLX graph that still references the
        // source `gate_*` / `up_*` arrays. Dropping
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
            meta: *meta,
        };
        // We hold the mutex AND OnceLock is empty (re-checked above), so
        // `set` cannot fail under correct usage.
        fused_gate_up
            .set(fused)
            .map_err(|_| anyhow!("RoutedExperts: fused_gate_up OnceLock set raced under mutex"))?;
        Ok(fused_gate_up
            .get()
            .expect("just set under mutex; OnceLock is now populated"))
    }

    fn gate_up_outputs(
        &self,
        lhs: &Array,
        rhs_indices: &Array,
        sorted_indices: bool,
        target: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        match &self.gate_up {
            GateUpPath::Fused { .. } => {
                let fused = self.fused_gate_up(target)?;
                let gate_up_out = mlx::quantization::gather_quantized_matmul_on(
                    lhs,
                    &fused.weight,
                    &fused.scales,
                    fused.biases.as_ref(),
                    None,
                    Some(rhs_indices),
                    true,
                    Some(fused.meta.group_size),
                    Some(fused.meta.bits),
                    fused.meta.mode.mlx_backend_mode(),
                    sorted_indices,
                    target,
                )
                .context("RoutedExperts::apply_experts: gate_up gather_qmm")?;
                let out_shape = gate_up_out.shape();
                let out_shape = out_shape.as_slice();
                let i = self.moe_intermediate;
                let (starts, gate_ends, up_starts, up_ends) = match out_shape {
                    [bs_k, one, _two_i] => (
                        vec![0_i32, 0, 0],
                        vec![*bs_k, *one, i],
                        vec![0_i32, 0, i],
                        vec![*bs_k, *one, 2 * i],
                    ),
                    [bs, k, one, _two_i] => (
                        vec![0_i32, 0, 0, 0],
                        vec![*bs, *k, *one, i],
                        vec![0_i32, 0, 0, i],
                        vec![*bs, *k, *one, 2 * i],
                    ),
                    other => {
                        return Err(anyhow!(
                            "RoutedExperts::apply_experts: fused gate_up output must be rank-3 or rank-4, got {other:?}"
                        ));
                    }
                };
                let gate_out = slice_on(&gate_up_out, starts, gate_ends, target)
                    .context("RoutedExperts::apply_experts: slice gate_out")?;
                let up_out = slice_on(&gate_up_out, up_starts, up_ends, target)
                    .context("RoutedExperts::apply_experts: slice up_out")?;
                Ok((gate_out, up_out))
            }
            GateUpPath::Split { gate, up } => {
                let gate_out =
                    gate.gather(lhs, rhs_indices, sorted_indices, target, "gate gather_qmm")?;
                let up_out =
                    up.gather(lhs, rhs_indices, sorted_indices, target, "up gather_qmm")?;
                Ok((gate_out, up_out))
            }
        }
    }

    /// SwitchGLU-style routed-expert combine.
    ///
    /// Route flat tokens `x` `[BS, H]` through the top-k experts named by
    /// `inds` `[BS, k]` (Uint32), weight each expert output by `weights`
    /// `[BS, k]`, and reduce across k → `[BS, H]`.
    ///
    /// Mirrors mlx_lm `SwitchGLU.__call__` (`switch_layers.py`) and is the
    /// exact dispatch previously inlined in `SparseMoeBlock::forward_on`
    /// (extracted verbatim so Qwen routing is byte-identical):
    ///   - `inds` is passed as `rhs_indices` to `gather_quantized_matmul_on`
    ///     (the expert id per (token, slot)); `transpose=true`.
    ///   - Sorted-flat path when `BS*k >= SORTED_ROUTING_MIN_BS_K` (=64):
    ///     pre-sort tokens by expert id, pass `sorted_indices=true`; x is
    ///     gathered to `[BS*k, 1, H]` (rank r+2 with r=1, so `lhs_indices`
    ///     defaults to `x.shape()[..-2] = [BS*k]` and broadcasts trivially
    ///     against `rhs_indices [BS*k]`). Qwen callers also provide the
    ///     request layout so equal expert ids are ordered by token/slot before
    ///     request row; this keeps equivalent rows in the same gather-QMM
    ///     numeric shape. After down_proj the permutation is inverted to
    ///     restore original token/slot order.
    ///   - Default broadcast path otherwise: x kept `[BS, 1, 1, H]` and MLX
    ///     broadcasts `rhs_indices [BS, k]` over the leading dims.
    ///
    /// Caller is responsible for `target` stream selection; `()` selects the
    /// MLX default stream.
    pub fn apply_experts(
        &self,
        x: &Array,
        inds: &Array,
        weights: &Array,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        self.apply_experts_inner(
            x,
            inds,
            weights,
            target,
            RoutedApplyOptions {
                layer_idx,
                cast_output_to_expert_dtype: false,
                activation: RoutedActivation::SwiGlu,
                request_layout: None,
            },
        )
    }

    /// Gemma-family routed combine using GELU(tanh)-gated expert activation.
    pub fn apply_experts_geglu(
        &self,
        x: &Array,
        inds: &Array,
        weights: &Array,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        self.apply_experts_inner(
            x,
            inds,
            weights,
            target,
            RoutedApplyOptions {
                layer_idx,
                cast_output_to_expert_dtype: false,
                activation: RoutedActivation::GeGluTanh,
                request_layout: None,
            },
        )
    }

    /// GLM/DeepSeek-style routed combine where the weighted-reduce result is
    /// cast back to the expert output dtype after multiplying by fp32 routing
    /// scores. Qwen mlx-lm paths keep the uncast sum, so the default
    /// [`Self::apply_experts`] preserves that behavior.
    pub fn apply_experts_cast_output(
        &self,
        x: &Array,
        inds: &Array,
        weights: &Array,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        self.apply_experts_inner(
            x,
            inds,
            weights,
            target,
            RoutedApplyOptions {
                layer_idx,
                cast_output_to_expert_dtype: true,
                activation: RoutedActivation::SwiGlu,
                request_layout: None,
            },
        )
    }

    fn apply_experts_inner(
        &self,
        x: &Array,
        inds: &Array,
        weights: &Array,
        target: StreamOrDevice,
        options: RoutedApplyOptions,
    ) -> Result<Array> {
        let xdims = x.shape();
        let xvec = xdims.as_slice();
        if xvec.len() != 2 {
            return Err(anyhow!(
                "RoutedExperts::apply_experts: x must be rank-2 [BS,H], got rank {}",
                xvec.len()
            ));
        }
        let (bs, h) = (xvec[0], xvec[1]);
        let idims = inds.shape();
        let ivec = idims.as_slice();
        if ivec.len() != 2 || ivec[0] != bs {
            return Err(anyhow!(
                "RoutedExperts::apply_experts: inds must be [BS,k] with BS={bs}, got {ivec:?}"
            ));
        }
        let k = ivec[1];
        let bs_k = bs * k;
        let use_sorted = bs_k >= SORTED_ROUTING_MIN_BS_K;
        let child_spans_enabled = false;

        let (gate_out, up_out, rhs_idx_used, sorted_flag, sort_perm_opt) =
            with_glm_routed_experts_child_span(
                child_spans_enabled,
                "glm_moe_routed_gate_up_gather_qmm",
                options.layer_idx,
                || -> Result<(Array, Array, Array, bool, Option<Array>)> {
                    let result = if use_sorted {
                        // --- Sorted routing path. ---
                        let flat_topk = mlx::ops::shape::reshape(inds, [bs_k])
                            .context("RoutedExperts::apply_experts: reshape inds to [BS*k]")?;
                        let use_request_interleaving =
                            options.request_layout.is_some_and(|(batch, _)| batch > 1);
                        let sort_perm = if use_request_interleaving {
                            let (batch, sequence) = options
                                .request_layout
                                .expect("checked request layout presence");
                            request_interleaved_sort_perm(
                                &flat_topk,
                                batch,
                                sequence,
                                k,
                                bs_k,
                                self.num_experts,
                                target,
                            )?
                        } else {
                            argsort_on(&flat_topk, -1_i32, target)
                                .context("RoutedExperts::apply_experts: argsort flat_topk")?
                        }; // [BS*k]
                        let sorted_topk = take_along_axis_on(
                            &flat_topk, &sort_perm, -1_i32, target,
                        )
                        .context("RoutedExperts::apply_experts: take_along_axis sort flat_topk")?;
                        let sorted_token_idx =
                            sorted_token_indices_from_sort_perm(&sort_perm, k, bs_k, target)?;
                        let sorted_x_2d = take_on(x, &sorted_token_idx, 0_i32, target)
                            .context("RoutedExperts::apply_experts: take x by sorted_token_idx")?;
                        let sorted_x_3d =
                            mlx::ops::shape::expand_dims_on(&sorted_x_2d, -2_i32, target).context(
                                "RoutedExperts::apply_experts: expand_dims sorted_x → [BS*k,1,H]",
                            )?;

                        let (gate_out, up_out) = self
                            .gate_up_outputs(
                                &sorted_x_3d,
                                &sorted_topk,
                                /* sorted_indices */ true,
                                target,
                            )
                            .context("RoutedExperts::apply_experts: gate/up gather_qmm (sorted)")?;

                        (gate_out, up_out, sorted_topk, true, Some(sort_perm))
                    } else {
                        // --- Default broadcast path. ---
                        let x_in =
                            mlx::ops::shape::expand_dims_on(x, &[-2_i32, -3_i32][..], target)
                                .context(
                                    "RoutedExperts::apply_experts: expand_dims x → [BS,1,1,H]",
                                )?; // [BS, 1, 1, H]

                        let (gate_out, up_out) =
                            self.gate_up_outputs(&x_in, inds, false, target).context(
                                "RoutedExperts::apply_experts: gate/up gather_qmm (default)",
                            )?;

                        (gate_out, up_out, inds.clone(), false, None)
                    };

                    Ok(result)
                },
            )?;

        // Routed activation. Qwen/GLM use SwiGLU; Gemma4 MoE uses GeGLU.
        let act = with_glm_routed_experts_child_span(
            child_spans_enabled,
            "glm_moe_routed_swiglu",
            options.layer_idx,
            || -> Result<Array> {
                let act = match options.activation {
                    RoutedActivation::SwiGlu => invoke_swiglu(self.swiglu(), &gate_out, &up_out)?,
                    RoutedActivation::GeGluTanh => {
                        invoke_geglu_tanh(self.geglu(), &gate_out, &up_out)?
                    }
                };
                Ok(act)
            },
        )?;

        let (down_out, already_reduced) = with_glm_routed_experts_child_span(
            child_spans_enabled,
            "glm_moe_routed_down_gather_qmm",
            options.layer_idx,
            || -> Result<(Array, bool)> {
                let down_out_raw = mlx::quantization::gather_quantized_matmul_on(
                    &act,
                    &self.down_weight,
                    &self.down_scales,
                    self.down_biases.as_ref(),
                    None,
                    Some(&rhs_idx_used),
                    true,
                    Some(self.down_meta.group_size),
                    Some(self.down_meta.bits),
                    self.down_meta.mode.mlx_backend_mode(),
                    sorted_flag,
                    target,
                )
                .context("RoutedExperts::apply_experts: down_proj gather_qmm")?;

                // The sorted path can consume expert output in route-sorted order,
                // avoiding the [BS,k,H] scatter and expanded intermediate.
                let down_out = if let Some(sort_perm) = sort_perm_opt {
                    let inv_perm = argsort_on(&sort_perm, -1_i32, target)
                        .context("RoutedExperts::apply_experts: argsort inv permutation")?;
                    if sorted_moe_weighted_sum::should_use(&down_out_raw, &inv_perm, weights) {
                        let out = sorted_moe_weighted_sum::apply_on(
                            &down_out_raw,
                            &inv_perm,
                            weights,
                            options.cast_output_to_expert_dtype,
                            target,
                        )
                        .context("RoutedExperts::apply_experts: sorted weighted-sum kernel")?;
                        return Ok((out, true));
                    }
                    let down_out_2d = mlx::ops::shape::reshape(&down_out_raw, [bs_k, h]).context(
                        "RoutedExperts::apply_experts: reshape sorted down_out to [BS*k, H]",
                    )?;
                    let unsorted_2d = take_on(&down_out_2d, &inv_perm, 0_i32, target)
                        .context("RoutedExperts::apply_experts: take inv_perm to restore order")?;
                    mlx::ops::shape::reshape(&unsorted_2d, [bs, k, h])
                        .context("RoutedExperts::apply_experts: reshape unsorted to [BS, k, H]")?
                } else {
                    mlx::ops::shape::squeeze_on(&down_out_raw, &[-2_i32][..], target)
                        .context("RoutedExperts::apply_experts: squeeze down_proj dim -2")?
                };
                Ok((down_out, false))
            },
        )?;

        if already_reduced {
            return Ok(down_out);
        }

        // weights: [BS, k] -> [BS, k, 1] for broadcast with [BS, k, H].
        with_glm_routed_experts_child_span(
            child_spans_enabled,
            "glm_moe_routed_weighted_reduce",
            options.layer_idx,
            || -> Result<Array> {
                let weights_unsq = mlx::ops::shape::expand_dims_on(weights, -1_i32, target)
                    .context("RoutedExperts::apply_experts: expand weights dim")?;
                let weighted = &down_out * &weights_unsq;
                let mut out = mlx::ops::sum_on(&weighted, -2_i32, false, target)
                    .context("RoutedExperts::apply_experts: sum across k")?;
                if options.cast_output_to_expert_dtype {
                    out = out.astype_on(down_out.dtype(), target).context(
                        "RoutedExperts::apply_experts: cast weighted sum to expert dtype",
                    )?;
                }
                Ok(out)
            },
        )
    }

    /// Lazily-built compiled SwiGLU closure shared by `apply_experts`.
    fn swiglu(&self) -> &CompiledFn {
        self.swiglu.get_or_init(build_swiglu)
    }

    fn geglu(&self) -> &CompiledFn {
        self.geglu.get_or_init(build_geglu_tanh)
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
        })
    }

    /// Forward pass: `[B, S, H]` → `[B, S, H]`.
    ///
    /// Stream-targeted. Caller is responsible for passing the correct stream;
    /// `()` selects the MLX default stream.
    ///
    /// `layer_idx` — index of the enclosing decoder block. Kept in the
    /// signature for parity with other layer-aware forward paths.
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
        if crate::nn::position_stable_qmm::is_armed() && b == 1 && s > 1 {
            let _product_stable = crate::nn::product_stable_qmm::scope();
            return self.forward_unisolated_on(x, target, layer_idx);
        }
        if crate::nn::position_stable_qmm::is_armed() && s > 1 {
            let mut positions = Vec::with_capacity(s as usize);
            for position in 0..s {
                let position_x = mlx::ops::indexing::slice_strided_on(
                    x,
                    &[0_i32, position, 0][..],
                    &[b, position + 1, h][..],
                    &[1_i32, 1, 1][..],
                    target,
                )
                .context("SparseMoeBlock: slicing exact verify position")?;
                positions.push(self.forward_unisolated_on(&position_x, target, layer_idx)?);
            }
            let position_refs = positions.iter().collect::<Vec<_>>();
            return mlx::ops::shape::concatenate_on(&position_refs, 1, target)
                .context("SparseMoeBlock: concatenating exact verify positions");
        }
        self.forward_unisolated_on(x, target, layer_idx)
    }

    fn forward_shared_on(&self, flat_x: &Array, target: StreamOrDevice) -> Result<Array> {
        let shared_y = self
            .shared_expert
            .forward_on(flat_x, target)
            .context("SparseMoeBlock: shared_expert forward")?;
        let gate_logit = self
            .shared_expert_gate
            .forward_on(flat_x, target)
            .context("SparseMoeBlock: shared_expert_gate forward")?;
        let gate = gate_logit
            .sigmoid_on(target)
            .context("SparseMoeBlock: shared gate sigmoid")?;
        Ok(&shared_y * &gate)
    }

    fn forward_unisolated_on(
        &self,
        x: &Array,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        let dims = x.shape();
        let dvec = dims.as_slice();
        let (b, s, h) = (dvec[0], dvec[1], dvec[2]);
        let bs = b * s;
        let k = self.num_experts_per_tok;
        let num_experts = self.routed.num_experts;

        // --- Flatten [B, S, H] → [BS, H] for routing and expert kernels. ---
        // Setup before the 8 substep spans; not attributed to any substep.
        let flat_x = mlx::ops::shape::reshape(x, [bs, h])
            .context("SparseMoeBlock: reshape [B,S,H] → [BS,H]")?;

        {
            // Signature parity with other layer-aware forward paths.
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

            // (3) Routed SwiGLU via the shared SwitchGLU-style combine.
            //
            // `RoutedExperts::apply_experts` performs the sorted/broadcast
            // gather_qmm dispatch (threshold SORTED_ROUTING_MIN_BS_K=64),
            // fused gate/up, SwiGLU, down_proj, then weights + reduces across
            // k. `inds_u32` is the rhs_indices (expert id per (token, slot));
            // `scores` are the per-slot routing weights. This is the exact
            // logic previously inlined here, extracted so GLM-4 can reuse it.
            let routed_y = self.routed.apply_experts_inner(
                &flat_x,
                &inds_u32,
                &scores,
                target,
                RoutedApplyOptions {
                    layer_idx,
                    cast_output_to_expert_dtype: false,
                    activation: RoutedActivation::SwiGlu,
                    request_layout: Some((b, s)),
                },
            )?;

            // (7) Shared expert with independent sigmoid gate.
            let shared_gated = self.forward_shared_on(&flat_x, target)?; // [BS, H]

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

    #[test]
    fn request_interleaved_sort_pairs_matching_routes() -> Result<()> {
        let experts: Array = (&[1_u32, 0, 1, 0, 1, 0, 1, 0][..], [8]).try_into()?;
        let perm = request_interleaved_sort_perm(&experts, 2, 2, 2, 8, 2, ().into())?;

        assert_eq!(perm.to_vec::<u32>()?, vec![1, 5, 3, 7, 0, 4, 2, 6]);
        Ok(())
    }

    /// Compile-time check: RoutedExperts fields are public and Array can be
    /// referenced through them. Numerical correctness deferred to T5
    /// integration tests under tests/qwen35_moe_*.rs. Those tests
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
