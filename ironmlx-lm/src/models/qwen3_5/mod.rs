//! Qwen3.5 dense model with text and VL runtime paths.
//!
//! Hybrid 32-layer model alternating gated-full-attention (`AttnKind::Full`)
//! and gated-delta-net linear attention (`AttnKind::Linear`) by
//! `(layer_idx + 1) % full_attention_interval == 0`.

mod config;
mod model;
pub(crate) mod speculative;
mod text_model;

pub mod cross_modal;
pub mod image_processor;

pub use config::{Qwen35Config, RopeParams, VisionConfig};
pub use model::{Qwen35Model, MIN_KV_CACHE_CAP_FOR_GPU_PERF};
pub use text_model::Qwen35TextModel;
