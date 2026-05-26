//! Qwen3.5 MoE model with text and VL runtime paths. See the original text
//! bring-up spec:
//! `docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md`.

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
