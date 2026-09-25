//! Generation infrastructure that's model-agnostic.

pub mod cache;
pub mod dflash2;
pub mod gemma4_generation;
pub mod generate;
pub mod generation_types;
pub mod memory_budget;
pub mod mtp_draft_cap_calibration;
pub mod process_memory;
pub mod prompt_lookup;
pub mod runtime_usage;
pub mod scheduler;
pub mod scheduler_autotune;
pub mod speculative;
pub mod speculative_qualification;

pub use dflash2::DFlash2Metrics;
pub use dflash2::DFlash2TextGenerationStream;
pub use generate::{GenerateEvent, GenerateRequest, GenerationStream};
pub use scheduler::{Phase, RequestId, RequestState, Scheduler, SchedulerError, StepEvent};

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

pub mod input_preparation;
pub mod single_request;

pub mod direct_execution;

pub mod model_management;

pub mod model_capacity;

pub mod audio_execution;
pub mod decision_execution;
