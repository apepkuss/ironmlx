//! Qwen3.6 MoE checkpoint facade.
//!
//! Downloaded Qwen3.6 MoE checkpoints declare the same HF architecture and
//! tensor graph as Qwen3.5 MoE-VL (`model_type = "qwen3_5_moe"`), so product
//! entry points dispatch them through [`crate::models::ModelArchitecture`] to
//! the shared Qwen3.5 MoE execution graph. This module remains as a validated
//! checkpoint facade for direct core API users, Qwen3.6-specific config
//! validation, and real-checkpoint regression tests.

pub mod config;
pub mod model;

pub use config::{is_qwen36_moe_config, Qwen36MoeConfig};
pub use model::Qwen36MoeModel;

pub type Qwen36MoeTextModel = crate::models::qwen3_5_moe::Qwen35MoeTextModel;
