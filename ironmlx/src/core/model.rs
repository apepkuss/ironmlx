//! Trait abstracting the inference model used by [`crate::core::scheduler::Scheduler`],
//! [`crate::core::generate::GenerationStream`], and [`crate::core::server::SchedulerActor`].
//!
//! VL-related methods (`forward_vl_chunk` / `batched_prefill_vl` / `compute_vision_embeds`)
//! intentionally remain inherent on concrete models; see spec §3.1 / §3.9.

use mlx::{Array, Dtype, StreamOrDevice};

use crate::core::cache::TurboQuantKVBits;
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

    /// Whether an equal-length text batch must preserve the single-row
    /// scheduler's `[N - 1] + [1]` prefill morphology for greedy-token parity.
    fn requires_split_batched_prefill_for_token_parity(&self) -> bool {
        false
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

    /// Whether this model instance has qualified a multi-token speculative
    /// verify forward as greedy-token-equivalent to its ordinary single-token
    /// decode path for the requested execution shape.
    ///
    /// Cache rollback policy is independent of this capability. Full KV
    /// caches may trim accepted offsets, while mixed recurrent caches restore
    /// a transaction checkpoint and replay the accepted prefix. Quantized
    /// projection and attention kernels can still change numerical shape at
    /// `Q > 1` and flip an argmax, so models must opt in only after
    /// architecture-level token-parity qualification.
    fn supports_exact_batched_speculative_verify(
        &self,
        _batch_width: usize,
        _context_tokens: usize,
        _verify_width: usize,
    ) -> bool {
        false
    }

    /// Apply cache-layout qualification on top of the model's architecture
    /// and weight-quantization profile.
    ///
    /// TurboQuant can make a multi-token cache update numerically distinct
    /// from repeated q=1 updates even when the model forward itself is
    /// qualified. Models with such a restriction must fail closed here.
    fn supports_exact_batched_speculative_verify_for_kv_cache(
        &self,
        batch_width: usize,
        context_tokens: usize,
        verify_width: usize,
        _kv_bits: Option<TurboQuantKVBits>,
    ) -> bool {
        self.supports_exact_batched_speculative_verify(batch_width, context_tokens, verify_width)
    }

    /// Whether PromptLookup may fall back to a chain of q=1 verify forwards
    /// when exact batched verification is not qualified for the current
    /// shape. Architectures may disable this when production measurements
    /// show that sequential probing is consistently more expensive than
    /// ordinary decode.
    fn supports_sequential_prompt_lookup_verify(
        &self,
        _batch_width: usize,
        _context_tokens: usize,
        _verify_width: usize,
    ) -> bool {
        true
    }

    /// Clamp PromptLookup's configured proposal width to the largest draft
    /// window this model has qualified for production verification.
    ///
    /// The limit is applied before proposal construction so shared-MTP
    /// certification metadata and bonus tokens are derived from the same
    /// window that will be verified.
    fn max_prompt_lookup_draft_tokens(&self, configured_max_draft_tokens: usize) -> usize {
        configured_max_draft_tokens
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
