use super::activations::silu;
use super::linear::{Linear, LinearLayer};
use super::mlp::MLP;
use crate::array::Array;
use crate::error::Result;
use crate::ops;
use crate::stream::Stream;
use crate::vector::VectorArray;

/// Single expert: gate_proj + up_proj + down_proj (SwiGLU)
pub struct Expert {
    pub gate_proj: LinearLayer,
    pub up_proj: LinearLayer,
    pub down_proj: LinearLayer,
}

impl Expert {
    pub fn new(gate_proj: LinearLayer, up_proj: LinearLayer, down_proj: LinearLayer) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let gate = self.gate_proj.forward_with_stream(x, stream)?;
        let gate = silu(&gate, stream)?;
        let up = self.up_proj.forward_with_stream(x, stream)?;
        let combined = ops::multiply(&gate, &up, stream)?;
        self.down_proj.forward_with_stream(&combined, stream)
    }
}

/// Mixture-of-Experts MLP with shared expert.
///
/// Routes each token to top-k experts, computes weighted sum of expert outputs,
/// then adds a gated shared expert contribution.
pub struct MoEMLP {
    /// Router: hidden_size -> num_experts
    pub gate: Linear,
    /// Per-expert MLPs
    pub experts: Vec<Expert>,
    /// Shared expert (standard SwiGLU MLP, always active)
    pub shared_expert: MLP,
    /// Shared expert gate: hidden_size -> 1
    pub shared_expert_gate: Linear,
    /// Number of experts selected per token
    pub num_experts_per_tok: usize,
    /// Whether to normalize top-k probabilities
    pub norm_topk_prob: bool,
}

impl MoEMLP {
    pub fn forward_with_stream(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let shape = x.shape(); // [batch, seq_len, hidden_size]
        let b = shape[0];
        let l = shape[1];
        let d = shape[2];
        let n = b * l; // total tokens
        let k = self.num_experts_per_tok as i32;

        // Flatten to [N, D] for routing
        let x_flat = ops::reshape(x, &[n, d], stream)?;

        // 1. Router: gate(x) -> softmax -> top-k selection
        let gate_logits = self.gate.forward_with_stream(&x_flat, stream)?;
        let gate_probs = ops::softmax(&gate_logits, &[-1], stream)?; // [N, num_experts]

        // Get top-k indices via argsort (descending) and top-k values
        let neg_probs = ops::neg(&gate_probs, stream)?;
        let sorted_indices = ops::argsort_axis(&neg_probs, -1, stream)?; // descending order
        let top_indices = ops::slice(&sorted_indices, &[0, 0], &[n, k], &[1, 1], stream)?; // [N, k]
        let top_values = ops::topk(&gate_probs, k, -1, stream)?; // [N, k]

        // Normalize scores if configured
        let top_scores = if self.norm_topk_prob {
            let score_sum = ops::sum(&top_values, &[-1], true, stream)?;
            ops::divide(&top_values, &score_sum, stream)?
        } else {
            top_values
        };

        // 2. Expert computation
        // Eval indices on CPU for routing decisions
        top_indices.eval()?;
        top_scores.eval()?;

        let indices_data = top_indices.to_vec_i32()?;
        let scores_data = top_scores.to_vec_f32()?;
        let k_usize = k as usize;
        let n_usize = n as usize;
        let num_experts = self.experts.len();

        // Build per-expert token assignments: expert_idx -> [(token_idx, slot_idx)]
        let mut expert_tokens: Vec<Vec<(usize, usize)>> = vec![vec![]; num_experts];
        for token_idx in 0..n_usize {
            for slot in 0..k_usize {
                let expert_idx = indices_data[token_idx * k_usize + slot] as usize;
                if expert_idx < num_experts {
                    expert_tokens[expert_idx].push((token_idx, token_idx * k_usize + slot));
                }
            }
        }

        // Accumulate weighted expert outputs per token
        // output_accum[token_idx] = sum over selected experts of (score * expert(x))
        // We build this by processing each expert's batch and adding contributions.
        //
        // Strategy: collect all (token_idx, expert_output_row, score) then build output
        // by iterating tokens. This avoids O(N*E) scatter ops.

        // For each token, collect weighted expert outputs
        let mut token_expert_results: Vec<Vec<(Array, f32)>> =
            (0..n_usize).map(|_| Vec::new()).collect();

        for (expert_idx, assignments) in expert_tokens.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }

            let expert = &self.experts[expert_idx];

            // Gather tokens for this expert
            let token_indices: Vec<i32> = assignments.iter().map(|&(t, _)| t as i32).collect();
            let idx_array = Array::from_slice_i32(&token_indices);
            let expert_input = ops::take_axis(&x_flat, &idx_array, 0, stream)?;

            // Run expert forward on the batch
            let expert_output = expert.forward_with_stream(&expert_input, stream)?;

            // Store results per token
            for (local_idx, &(token_idx, slot_idx)) in assignments.iter().enumerate() {
                let row = ops::slice(
                    &expert_output,
                    &[local_idx as i32, 0],
                    &[local_idx as i32 + 1, d],
                    &[1, 1],
                    stream,
                )?;
                let score = scores_data[slot_idx];
                token_expert_results[token_idx].push((row, score));
            }
        }

        // Build output: for each token, weighted-sum its expert results
        let mut output_rows: Vec<Array> = Vec::with_capacity(n_usize);
        for token_results in &token_expert_results {
            if token_results.is_empty() {
                // No experts selected (shouldn't happen), use zeros
                output_rows.push(ops::zeros(&[1, d], x_flat.dtype(), stream)?);
            } else {
                let mut row_sum = {
                    let (ref first_row, first_score) = token_results[0];
                    let w = Array::from_float(first_score);
                    ops::multiply(first_row, &w, stream)?
                };
                for &(ref row, score) in &token_results[1..] {
                    let w = Array::from_float(score);
                    let weighted = ops::multiply(row, &w, stream)?;
                    row_sum = ops::add(&row_sum, &weighted, stream)?;
                }
                output_rows.push(row_sum);
            }
        }

        // Stack all rows into [N, D]
        let row_refs: Vec<&Array> = output_rows.iter().collect();
        let va = VectorArray::from_arrays(&row_refs);
        let output = ops::concatenate(&va, 0, stream)?; // [N, D]

        // 3. Shared expert contribution
        let shared_out = self.shared_expert.forward_with_stream(&x_flat, stream)?;
        let shared_gate_logits = self
            .shared_expert_gate
            .forward_with_stream(&x_flat, stream)?;
        let shared_gate_val = ops::sigmoid(&shared_gate_logits, stream)?;
        let shared_contribution = ops::multiply(&shared_out, &shared_gate_val, stream)?;

        // 4. Combine: expert_output + shared_expert_output
        let output = ops::add(&output, &shared_contribution, stream)?;
        ops::reshape(&output, &[b, l, d], stream)
    }
}
