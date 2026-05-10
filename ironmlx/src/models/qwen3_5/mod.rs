//! Qwen3.5 Dense model (text-only path).
//!
//! Hybrid 32-layer model alternating gated-full-attention (`AttnKind::Full`)
//! and gated-delta-net linear attention (`AttnKind::Linear`) by
//! `(layer_idx + 1) % full_attention_interval == 0`.

mod config;
mod model;
mod text_model;

pub mod cross_modal;
pub mod image_processor;
pub mod vision;

pub use config::{Qwen35Config, RopeParams};
pub use model::Qwen35Model;
pub use text_model::Qwen35TextModel;
