//! Generation infrastructure that's model-agnostic.

pub mod cache;
pub mod chat_template;
pub mod generate;
pub mod loader;
pub mod memory_budget;
pub mod model;
pub mod mtp_draft_cap_calibration;
pub mod process_memory;
pub mod prompt_lookup;
pub mod sampler;
pub mod scheduler;
pub mod scheduler_autotune;
pub mod server;
pub mod speculative;
pub(crate) mod speculative_qualification;
pub mod tokenizer;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use generate::{build_position_ids, GenerateEvent, GenerateRequest, GenerationStream};
pub(crate) use loader::logical_width_from_packed;
pub use loader::{EosTokenId, Loader, QuantMeta, QuantMode, TokenizerConfig};
pub use model::Model;
pub use sampler::Sampler;
pub use scheduler::{Phase, RequestId, RequestState, Scheduler, SchedulerError, StepEvent};
pub use tokenizer::Tokenizer;
