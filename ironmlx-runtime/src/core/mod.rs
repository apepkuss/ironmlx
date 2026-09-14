//! Generation infrastructure that's model-agnostic.

pub mod cache;
pub(crate) mod chat_template;
pub(crate) mod constrained;
pub mod dflash2;
pub mod gemma4_generation;
pub mod generate;
pub(crate) mod generated_output;
pub mod generation_types;
pub(crate) mod loader;
pub mod memory_budget;
pub(crate) mod model;
pub(crate) mod model_input;
pub mod mtp_draft_cap_calibration;
pub(crate) mod native_output;
pub mod process_memory;
pub mod prompt_lookup;
pub mod runtime_usage;
pub(crate) mod sampler;
pub mod scheduler;
pub mod scheduler_autotune;
pub mod speculative;
pub(crate) mod speculative_model;
pub mod speculative_qualification;
pub(crate) mod tokenizer;
pub(crate) mod tool_calling;
pub(crate) mod vision;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use dflash2::DFlash2Metrics;
pub use dflash2::DFlash2TextGenerationStream;
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

pub(crate) mod vision_input;

pub mod runtime_config;

pub(crate) mod adaptive_admission;
pub(crate) mod dflash2_actor;
pub mod scheduler_actor;
pub mod task_execution;

pub mod diffusion_execution;

pub mod scheduler_profile_context;
pub mod scheduler_profile_store;

pub mod scheduler_resolution;

pub mod page_storage;

pub mod engine_state;
pub mod runtime_health;

pub mod model_validation;

pub mod engine_pool;
