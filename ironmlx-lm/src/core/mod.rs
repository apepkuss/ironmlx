//! Model loading, input, output, and computation contracts.
pub mod audio_input;
pub mod cache;
pub mod chat_template;
pub mod constrained;
pub mod generated_output;
pub mod image_input;
pub(crate) mod image_resample;
pub mod json_schema;
pub mod load_observer;
pub mod loader;
pub mod model;
pub mod model_input;
pub mod native_output;
pub mod reasoning_budget;
pub mod sampler;
mod semantic_constraint;
pub mod speculative_model;
pub mod tokenizer;
pub mod tool_calling;
mod tool_prompt_cache;
pub mod vision;
pub mod vision_input;
pub mod weights;
pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use generated_output::{
    AudioChunk, CapabilitySupport, CollectedGeneratedOutput, GeneratedFinishReason,
    GeneratedOutputDecoder, GeneratedOutputEvent, ImageArtifact, InputCapabilityProfile,
    ModelCapabilityProfile, OutputCapabilityProfile, ToolOutputDecoderConfig,
};
pub use load_observer::{LoadCancelled, LoadObserver, LoadObserverSnapshot, WeightComponent};
pub use loader::{
    preflight_model_metadata, EosTokenId, Loader, ModelMetadataPreflight, QuantMeta, QuantMode,
    QuantizationMetadataPreflight, TokenizerConfig, WEIGHT_EVAL_CHUNK_BYTES,
};
pub use model::Model;
pub use native_output::{NativeOutputDecoderConfig, NativeOutputDialect, NativeOutputParser};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
pub use tool_calling::{AgentMessage, ToolCall, ToolDefinition, ToolDialect};
pub(crate) use weights::logical_width_from_packed;

pub mod speculative_ops;

pub mod prompt_images;
