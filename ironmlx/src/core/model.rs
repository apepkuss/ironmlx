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

    /// Whether caller-built MRoPE `position_ids` are semantically consumed by
    /// this model. Models returning `false` derive positions internally and
    /// may receive a reusable placeholder Array from generation/scheduler hot
    /// paths.
    fn requires_position_ids(&self) -> bool {
        true
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

    fn model_meta(&self) -> ModelMeta;

    fn num_hidden_layers(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::Model;

    fn _assert_trait_signature_exists<M: Model>(_: &M) {}
}
