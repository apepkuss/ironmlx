//! Gemma4 Dense model support.
//!
//! Scope: `model_type=gemma4` with `text_config.enable_moe_block=false`.
//! Vision supports image + text prompts. Audio/video and Gemma4 MoE are
//! intentionally out of scope.

mod attention;
mod config;
mod cross_modal;
mod decoder_layer;
pub mod image_processor;
mod mlp;
mod model;
mod ops;
mod profile;
mod rope;
mod text_model;
pub(crate) mod vision;

pub use config::{
    Gemma4Config, Gemma4LayerKind, Gemma4RopeParams, Gemma4TextConfig, Gemma4VisionConfig,
};
pub use model::Gemma4Model;
pub use text_model::Gemma4TextModel;
