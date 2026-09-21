//! Native execution request and incremental token events. No transport DTOs or scheduler ownership.

use ironmlx_core::sampler::Sampler;
use ironmlx_lm::core::cache::turboquant_kv::TurboQuantKVBits;
use ironmlx_lm::core::constrained::ConstraintPlan;
use mlx::Array;

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Tokenized prompt (after chat template rendering, if any).
    pub prompt_ids: Vec<u32>,
    /// Hard cap on tokens generated beyond the prompt.
    pub max_new_tokens: usize,
    /// Sampling configuration. Defaults to greedy if left at `Sampler::greedy()`.
    pub sampler: Sampler,
    /// Token ids that terminate the stream when produced.
    pub stop_token_ids: Vec<u32>,
    /// Max tokens per prefill forward. `0` disables chunking (entire prompt
    /// goes through a single forward). The chunked path bounds activation
    /// memory peak for long agent prompts and lets the GPU pipeline subsequent
    /// chunks; intermediate chunks update the cache only (no lm_head), the
    /// last chunk runs the full forward + lm_head.
    pub prefill_chunk_size: usize,
    /// Request-level rolling mid-admit chunk cap used while decode rows are
    /// active. Selected from the runtime scheduler profile after tokenization.
    pub decode_cadence_mid_chunk_cap: usize,
    /// Optional TurboQuant K/V bit-widths for full-attention KV cache reads.
    pub kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    /// Per-image preprocessed vision inputs in prompt order. `None` = text-only.
    ///
    /// Qwen images are fixed-size patch sequences and can be concatenated by
    /// the model implementation. Gemma4 images keep their original resized
    /// `[1, 3, H, W]` tensor per image because different images can have
    /// different `H/W`.
    pub pixel_values: Option<Vec<Array>>,
    /// Per-image `(T, H, W)` grids in the same order as `pixel_values`.
    pub image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    /// `VisionConfig.spatial_merge_size` for this model. Used to compute the
    /// MRoPE VL position-id strides; only consulted when `image_grid_thw` is
    /// `Some`. Default `2` matches Qwen3.5-VL. Sibling VL models with a
    /// different merge factor must set this explicitly.
    pub image_spatial_merge_size: i32,
    /// Token id of `<|image_pad|>` (the per-patch image placeholder). Used to
    /// locate which input_id positions get replaced with vision embeddings and
    /// to drive MRoPE VL stride boundaries. Default [`IMAGE_TOKEN_ID`](ironmlx_lm::core::model_input::IMAGE_TOKEN_ID)
    /// (`248056` for Qwen3.5-VL). Sibling VL models with a different image-pad
    /// id must set this from `Tokenizer::token_to_id("<|image_pad|>")`.
    pub image_token_id: i32,
    /// Optional immutable token-level decoding constraint for this request.
    pub constraint: Option<ConstraintPlan>,
}

#[derive(Debug, Clone)]
pub struct GenerateEvent {
    /// The token id this step produced.
    pub token: u32,
    /// Incremental decoded text since the previous event. May be empty
    /// (BPE boundary not yet reached); callers should concatenate.
    pub text: String,
    /// Some on the final event: "stop" (EOS hit) or "length" (max_new_tokens).
    pub finish_reason: Option<&'static str>,
}
