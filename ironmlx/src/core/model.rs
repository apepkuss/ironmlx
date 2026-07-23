//! Trait abstracting the inference model used by [`crate::core::scheduler::Scheduler`],
//! [`crate::core::generate::GenerationStream`], and [`crate::core::server::SchedulerActor`].
//!
//! VL-related methods (`forward_vl_chunk` / `batched_prefill_vl` / `compute_vision_embeds`)
//! intentionally remain inherent on concrete models; see spec §3.1 / §3.9.

use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::memory_budget::ModelMeta;
use crate::nn::LayerCache;
use crate::Result;

pub trait Model {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>>;

    /// Dtype expected by this model's KV cache tensors.
    fn cache_dtype(&self) -> Dtype {
        Dtype::Bfloat16
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    /// Batched prefill for rows that all occupy the full sequence dimension.
    ///
    /// With no right-padding, the model's regular causal forward is sufficient
    /// and avoids materializing a quadratic `[B, 1, T, T]` attention mask.
    #[allow(clippy::too_many_arguments)]
    fn batched_prefill_causal(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        self.forward_on(
            input_ids,
            position_ids,
            Some(per_row_lens),
            None,
            cache,
            target,
        )
    }

    /// Forward through embed + transformer + final RmsNorm, returning the
    /// hidden states (NOT projected to logits). Used by the chunked-prefill
    /// path for intermediate (non-last) chunks where only KV cache needs to
    /// be updated and logits are discarded.
    ///
    /// Signature matches `forward_on` but the return shape is `[B, S, hidden]`
    /// post-norm hidden state instead of `[B, 1, vocab]` last-position logits.
    #[allow(clippy::too_many_arguments)]
    fn forward_text_hidden(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    /// Project post-norm hidden states at every sequence position to logits.
    /// Source-independent speculative verification uses the preserved
    /// `[B, S, V]` sequence dimension to check a complete draft window.
    fn project_hidden_on(&self, _hidden: &Array, _target: StreamOrDevice) -> Result<Array> {
        Err(anyhow::anyhow!(
            "full-sequence hidden projection is not implemented for {}",
            std::any::type_name::<Self>()
        ))
    }

    /// Whether caller-built MRoPE `position_ids` are semantically consumed by
    /// this model. Models returning `false` derive positions internally and
    /// may receive a reusable placeholder Array from generation/scheduler hot
    /// paths.
    fn requires_position_ids(&self) -> bool {
        true
    }

    /// Whether this model has qualified a multi-token speculative verify
    /// forward as greedy-token-equivalent to its ordinary single-token decode
    /// path across supported batch shapes.
    ///
    /// Full KV cache trim is necessary for an exact batched verify, but it is
    /// not sufficient: quantized projection and attention kernels can change
    /// numerical shape at `Q > 1` and flip an argmax. Models must opt in only
    /// after architecture-level token-parity qualification.
    fn supports_exact_batched_speculative_verify(&self) -> bool {
        false
    }

    /// Whether q=1 speculative verification must materialize each device
    /// result before host-side cache offsets advance to the next depth.
    ///
    /// Most architectures can retain the full sequential chain lazily. A
    /// model should opt in only when its cache/shared-state execution has
    /// demonstrated that deferred materialization changes greedy results.
    fn requires_eager_sequential_speculative_verify(&self) -> bool {
        false
    }

    /// Whether speculative rollback may keep the already-computed accepted
    /// prefix by trimming only Full-cache offsets.
    ///
    /// Models whose cache or attention implementation has not qualified that
    /// invariant must restore the base snapshot and replay accepted inputs.
    fn supports_speculative_accepted_prefix_trim(&self) -> bool {
        false
    }

    /// Maximum number of requests this model wants to admit into one fresh
    /// prefill batch for a prompt of `prompt_len` tokens.
    ///
    /// The default keeps the scheduler's existing behavior. Model
    /// implementations can lower this when their B>1 long-prefill path is
    /// slower than B=1 prefill plus rolling mid-admission.
    fn fresh_prefill_batch_limit(prompt_len: usize, b_max: usize) -> usize
    where
        Self: Sized,
    {
        let _ = prompt_len;
        b_max
    }

    /// When this model takes a VL prefill (images present), should the position
    /// ids be flat **sequential** (`build_position_ids`, all three MRoPE streams
    /// identical) rather than spatial 2-D MRoPE (`build_position_ids_vl`)?
    /// MiniCPM-V-4.6 uses sequential positions even with images (mlx-vlm
    /// `_set_position_state` = arange broadcast). Default `false` preserves the
    /// spatial-MRoPE behavior of Qwen3.5-VL / Gemma.
    fn vl_positions_sequential(&self) -> bool {
        false
    }

    fn model_meta(&self) -> ModelMeta;

    fn num_hidden_layers(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::Model;

    fn _assert_trait_signature_exists<M: Model>(_: &M) {}
}
