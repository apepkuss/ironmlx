//! Qwen3.5 MoE model (text-only path). See spec
//! `docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md`.
//!
//! P5 D2: text-only LM path. VL + MTP intentionally out of scope.

pub mod config;
pub mod decoder_layer;
pub mod model;
pub mod sparse_moe;
pub mod text_model;

pub use config::{Qwen35MoeConfig, RopeParams};
pub use decoder_layer::{DecoderLayerMoe, DecoderLayerMoeConfig};
pub use model::{Qwen35MoeModel, MIN_KV_CACHE_CAP_FOR_GPU_PERF};
pub use sparse_moe::{RoutedExperts, SparseMoeBlock};
pub use text_model::Qwen35MoeTextModel;
