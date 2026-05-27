//! Gemma4 Dense text-only model support.
//!
//! Scope: `model_type=gemma4` with `text_config.enable_moe_block=false`.
//! Vision/audio towers and Gemma4 MoE are intentionally out of scope.

mod attention;
mod config;
mod decoder_layer;
mod mlp;
mod model;
mod ops;
mod rope;
mod text_model;

pub use config::{Gemma4Config, Gemma4LayerKind, Gemma4RopeParams, Gemma4TextConfig};
pub use model::Gemma4Model;
pub use text_model::Gemma4TextModel;
