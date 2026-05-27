//! Qwen3.6 MoE architecture package.
//!
//! The public Qwen3.6 entry point is explicit even though the downloaded
//! checkpoint declares the same HF architecture and tensor graph as Qwen3.5
//! MoE-VL. Qwen3.6-specific validation and dispatch live here; numeric
//! execution is delegated to the shared MoE-VL kernel.

pub mod config;
pub mod model;

pub use config::{is_qwen36_moe_config, Qwen36MoeConfig};
pub use model::Qwen36MoeModel;

pub type Qwen36MoeTextModel = crate::models::qwen3_5_moe::Qwen35MoeTextModel;
