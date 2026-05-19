//! Qwen3.5 MoE model (text-only path). See spec
//! `docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md`.
//!
//! P5 D2: text-only LM path. VL + MTP intentionally out of scope.

pub mod config;
pub mod decoder_layer;
pub mod sparse_moe;
pub mod text_model;
// 后续 P5b task 解开：
// pub mod model;

pub use config::{Qwen35MoeConfig, RopeParams};
pub use decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};
pub use sparse_moe::{RoutedExperts, SparseMoeBlock};
pub use text_model::Qwen35MoeTextModel;
// pub use model::Qwen35MoeModel;
