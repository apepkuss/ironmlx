//! Model architectures.
//!
//! Each architecture lives in its own self-contained directory. Sharing
//! between architectures happens via [`crate::nn`] and [`crate::core`] —
//! never by reaching across model directories.
//!
//! Planned (in implementation order):
//! - **P3-P4** — `qwen3_5` (Dense): hybrid gated-delta + gated full attention,
//!   MRoPE, RMSNormGated, MTP layer, 4-bit quantized weights.
//! - **P5** — `qwen3_5_moe` (MoE variant): adds SparseMoeBlock; otherwise
//!   reuses qwen3_5 attention / norm primitives via local copies (modules
//!   stay independent — no cross-model imports).
//! - **P6** — `qwen3_5_vl` (multimodal): adds vision encoder + cross-modal
//!   token routing.

pub mod qwen3_5;
pub mod qwen3_5_moe;

pub use qwen3_5::{Qwen35Config, Qwen35Model, Qwen35TextModel, RopeParams};
pub use qwen3_5_moe::{Qwen35MoeConfig, RopeParams as MoeRopeParams};

// pub mod qwen3_5_vl;
