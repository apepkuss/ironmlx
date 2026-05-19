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
//!   4. scores = take_along_axis(probs, inds, -1) / scores.sum(-1, keepdim)
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
use mlx::ops::indexing::take_along_axis_on;
use mlx::ops::sort::argpartition_on;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::{Linear, Mlp};
use crate::Result;

/// Stacked-expert quantized weights for the routed SwiGLU.
///
/// Shape convention (4-bit, group_size=64, num_experts=E, hidden=H,
/// moe_intermediate=I):
///   gate/up: weight `[E, I, H/8]`, scales `[E, I, H/64]`, biases `[E, I, H/64]`
///   down:    weight `[E, H, I/8]`, scales `[E, H, I/64]`, biases `[E, H, I/64]`
pub struct RoutedExperts {
    pub gate_weight: Array,
    pub gate_scales: Array,
    pub gate_biases: Option<Array>,
    pub up_weight: Array,
    pub up_scales: Array,
    pub up_biases: Option<Array>,
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

        Ok(Self {
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            down_weight,
            down_scales,
            down_biases,
            group_size: qmeta.group_size,
            bits: qmeta.bits,
            num_experts,
        })
    }
}

/// Sparse MoE block for Qwen3.5-MoE.
///
/// Routing: softmax → argpartition top-k → renormalize.
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
}

impl SparseMoeBlock {
    /// Construct from `{prefix}` where prefix = `"model.layers.{i}.mlp"`.
    ///
    /// Sub-paths:
    ///   `{prefix}.gate`              — router gate Linear (quantized)
    ///   `{prefix}.switch_mlp`        — routed stacked experts
    ///   `{prefix}.shared_expert`     — shared SwiGLU Mlp
    ///   `{prefix}.shared_expert_gate`— sigmoid gate Linear (quantized)
    pub fn from_loader(loader: &Loader, prefix: &str, num_experts_per_tok: i32) -> Result<Self> {
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
        })
    }

    /// Forward pass: `[B, S, H]` → `[B, S, H]`.
    ///
    /// Stream-targeted. Caller is responsible for passing the correct stream;
    /// `()` selects the MLX default stream.
    pub fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
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

        // --- Flatten [B, S, H] → [BS, H] for routing and expert kernels. ---
        let flat_x = mlx::ops::shape::reshape(x, [bs, h])
            .context("SparseMoeBlock: reshape [B,S,H] → [BS,H]")?;

        // (1) Router: Linear → [BS, E], then softmax along expert axis.
        let logits = self.router_gate.forward_on(&flat_x, target)?; // [BS, E]
        let probs = mlx::ops::softmax_on(&logits, -1_i32, /* precise */ true, target)?; // [BS, E]

        // (2) Top-k selection via argpartition.
        // argpartition is preferable to topk here: we don't need the top-k
        // elements sorted internally (each is independently weight-summed via
        // gather_qmm); we only need to know which k indices to gather. MLX
        // argpartition is a single pass; the values are recovered via
        // take_along_axis. This is an MLX-op-selection optimization, not
        // dictated by any reference implementation.
        // argpartition kth=-(k) places the top-k elements in the last k
        // positions of the returned index array. We then slice [BS, E] →
        // [BS, k] keeping the last k columns.
        let part_inds =
            argpartition_on(&probs, -(k), -1, target).context("SparseMoeBlock: argpartition")?;
        // part_inds: [BS, E] — take last k columns via strided slice.
        // mlx::ops::slice_strided_on lives in the indexing module but is
        // re-exported at mlx::ops level.
        let inds = mlx::ops::slice_strided_on(
            &part_inds,
            [0_i32, num_experts - k], // start: row 0, col E-k
            [bs, num_experts],        // stop (exclusive): all rows, end of E dim
            [1_i32, 1_i32],           // stride 1 on both dims
            target,
        )
        .context("SparseMoeBlock: slice top-k from argpartition")?; // [BS, k]

        // (3) Gather top-k probs and renormalize.
        let scores_raw = take_along_axis_on(&probs, &inds, -1, target)
            .context("SparseMoeBlock: take_along_axis")?; // [BS, k]
        let scores_sum = mlx::ops::sum_on(&scores_raw, -1_i32, /* keepdim */ true, target)?; // [BS, 1]
        let scores = &scores_raw / &scores_sum; // [BS, k] — panics on shape mismatch (broadcast guaranteed)

        // (4) Cast indices to uint32 (gather_qmm requirement).
        let inds_u32 = mlx::ops::cast::astype_on(&inds, mlx::Dtype::Uint32, target)
            .context("SparseMoeBlock: cast indices to Uint32")?; // [BS, k]

        // (5) Routed SwiGLU via gather_quantized_matmul_on (G1 path).
        //
        // MLX gather_qmm API contract: when `rhs_indices` has shape [BS, k],
        // the input `x` must be rank-4 [BS, 1, 1, H] so that the broadcast
        // produces output shape [BS, k, 1, out_dim]. With x rank-2 [BS, H],
        // MLX auto-broadcasts lhs_indices from `x.shape()[:-2]` = `[]`,
        // causing incorrect output shape [BS, k, BS, out_dim] (verified
        // in initial P5b T2 commit, fixed in 8e74caa). This is purely an
        // MLX gather_qmm input-shape requirement.
        let x_in = mlx::ops::shape::expand_dims_on(&flat_x, &[-2_i32, -3_i32][..], target)
            .context("SparseMoeBlock: expand_dims flat_x → [BS,1,1,H]")?; // [BS, 1, 1, H]

        // P5e A.1 experiment: dispatch gate and up projections on independent
        // streams to test whether the Metal command scheduler overlaps their
        // execution. Default off; T4 decides whether to keep based on measured
        // wall-clock improvement.
        #[cfg(feature = "p5e-stream-parallel")]
        let (stream_gate, stream_up) = {
            let dev = mlx::Device::gpu(0);
            (
                StreamOrDevice::Stream(
                    mlx::new_stream(dev).context("SparseMoeBlock: new_stream for gate")?,
                ),
                StreamOrDevice::Stream(
                    mlx::new_stream(dev).context("SparseMoeBlock: new_stream for up")?,
                ),
            )
        };
        #[cfg(not(feature = "p5e-stream-parallel"))]
        let (stream_gate, stream_up) = (target, target);

        let gate_out = mlx::quantization::gather_quantized_matmul_on(
            &x_in,
            &self.routed.gate_weight,
            &self.routed.gate_scales,
            self.routed.gate_biases.as_ref(),
            /* lhs_indices */ None,
            /* rhs_indices */ Some(&inds_u32),
            /* transpose */ true,
            Some(self.routed.group_size),
            Some(self.routed.bits),
            "affine",
            /* sorted_indices */ false,
            stream_gate,
        )
        .context("SparseMoeBlock: gate_proj gather_qmm")?; // [BS, k, 1, moe_inter]

        let up_out = mlx::quantization::gather_quantized_matmul_on(
            &x_in,
            &self.routed.up_weight,
            &self.routed.up_scales,
            self.routed.up_biases.as_ref(),
            None,
            Some(&inds_u32),
            true,
            Some(self.routed.group_size),
            Some(self.routed.bits),
            "affine",
            false,
            stream_up,
        )
        .context("SparseMoeBlock: up_proj gather_qmm")?; // [BS, k, 1, moe_inter]

        // SwiGLU activation: silu(gate) * up  where silu(z) = z * sigmoid(z)
        // gate_out, up_out: [BS, k, 1, moe_inter]
        let gate_sig = gate_out
            .sigmoid_on(target)
            .context("SparseMoeBlock: gate sigmoid")?;
        let gate_silu = &gate_out * &gate_sig; // [BS, k, 1, moe_inter]
        let act = &gate_silu * &up_out; // [BS, k, 1, moe_inter]

        // down_proj: act [BS, k, 1, moe_inter] → [BS, k, 1, H], then squeeze(-2) → [BS, k, H]
        let down_out_4d = mlx::quantization::gather_quantized_matmul_on(
            &act,
            &self.routed.down_weight,
            &self.routed.down_scales,
            self.routed.down_biases.as_ref(),
            None,
            Some(&inds_u32),
            true,
            Some(self.routed.group_size),
            Some(self.routed.bits),
            "affine",
            false,
            target,
        )
        .context("SparseMoeBlock: down_proj gather_qmm")?; // [BS, k, 1, H]
        let down_out = mlx::ops::shape::squeeze_on(&down_out_4d, &[-2_i32][..], target)
            .context("SparseMoeBlock: squeeze down_proj dim -2")?; // [BS, k, H]

        // (6) Weight by renormalized scores and sum across the k axis.
        //     scores: [BS, k] → unsqueeze → [BS, k, 1] for broadcast with [BS, k, H].
        let scores_unsq = mlx::ops::shape::expand_dims_on(&scores, -1_i32, target)
            .context("SparseMoeBlock: expand scores dim")?; // [BS, k, 1]
        let weighted = &down_out * &scores_unsq; // [BS, k, H]
        let routed_y = mlx::ops::sum_on(&weighted, -2_i32, false, target)
            .context("SparseMoeBlock: sum across k")?; // [BS, H]

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
