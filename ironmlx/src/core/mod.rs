//! Generation infrastructure that's model-agnostic.

pub mod cache;
pub mod chat_template;
pub mod constrained;
pub mod dflash2;
pub mod gemma4_generation;
pub mod generate;
pub mod generated_output;
pub mod generation_types;
pub mod image_input;
pub mod loader;
pub mod memory_budget;
pub mod model;
pub mod model_input;
pub mod mtp_draft_cap_calibration;
pub mod native_output;
pub mod process_memory;
pub mod prompt_lookup;
pub mod runtime_usage;
pub mod sampler;
pub mod scheduler;
pub mod scheduler_autotune;
pub mod server;
pub mod speculative;
pub mod speculative_model;
pub mod tokenizer;
pub mod tool_calling;
pub mod vision;
pub mod weights;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use dflash2::DFlash2Metrics;
pub(crate) use dflash2::DFlash2TextGenerationStream;
pub use generate::{build_position_ids, GenerateEvent, GenerateRequest, GenerationStream};
pub use generated_output::{
    AudioChunk, CapabilitySupport, CollectedGeneratedOutput, GeneratedFinishReason,
    GeneratedOutputDecoder, GeneratedOutputEvent, ImageArtifact, InputCapabilityProfile,
    ModelCapabilityProfile, OutputCapabilityProfile, ToolOutputDecoderConfig,
};
pub use loader::{
    preflight_model_metadata, EosTokenId, Loader, ModelMetadataPreflight, QuantMeta, QuantMode,
    QuantizationMetadataPreflight, TokenizerConfig,
};
pub use model::Model;
pub use native_output::{NativeOutputDecoderConfig, NativeOutputDialect, NativeOutputParser};
pub use sampler::Sampler;
pub use scheduler::{Phase, RequestId, RequestState, Scheduler, SchedulerError, StepEvent};
pub use tokenizer::Tokenizer;
pub use tool_calling::{AgentMessage, ToolCall, ToolDefinition, ToolDialect};

pub mod vision_input;

pub mod runtime_config;

pub mod scheduler_actor;
pub mod task_execution;

pub mod diffusion_execution;

pub(crate) mod scheduler_profile_context;
pub(crate) mod scheduler_profile_store;

pub(crate) mod scheduler_resolution;

pub mod page_storage;

pub mod engine_state;
pub mod runtime_health;

pub(crate) mod model_validation;

pub mod engine_pool;
