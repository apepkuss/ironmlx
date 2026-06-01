//! MiniCPM-V-4.6 checkpoint facade — text-only language backbone.
//!
//! MiniCPM-V-4.6 (`model_type = "minicpmv4_6"`,
//! `architectures = ["MiniCPMV4_6ForConditionalGeneration"]`) is a
//! vision-language model. Its language backbone is Qwen3.5-text verbatim
//! (`text_config.model_type = "qwen3_5_text"`; in mlx-vlm the language model is
//! literally `class LanguageModel(Qwen35LanguageModel)` with only
//! `get_rope_index` overridden for image positions). The LM weights ship under
//! `language_model.*`, identical to a native Qwen3.5 checkpoint.
//!
//! This module therefore runs the **text path on the shared Qwen3.5 dense
//! execution graph** ([`Qwen35Model`]). It only adapts the config
//! (see [`config`]) — defaulting the omitted `mrope_section` and skipping the
//! incompatible SigLIP `vision_config`. The SigLIP vision stack is implemented
//! in the `vision` submodule (`MiniCpmV46Vision`), but is not yet wired into a
//! model/dispatch/inference path — image inputs remain out of scope until the
//! VLM model is added (P2). This `model_from_loader` facade deliberately runs
//! the text-only Qwen3.5 dense path by dropping `vision_config`.

pub mod config;
pub mod image_processor;
pub mod vision;

pub use config::text_config_from_loader;

use crate::core::Loader;
use crate::models::Qwen35Model;
use crate::Result;

/// Build a text-only [`Qwen35Model`] from a MiniCPM-V-4.6 checkpoint.
///
/// Parses the nested `text_config` into a `Qwen35Config` (defaulting the
/// omitted `mrope_section`, dropping the SigLIP `vision_config`) and constructs
/// the shared Qwen3.5 dense model. The resulting model has no vision tower;
/// callers must drive it with text-only prompts.
pub fn model_from_loader(loader: &Loader) -> Result<Qwen35Model> {
    let cfg = config::text_config_from_loader(loader)?;
    Qwen35Model::from_loader_with_config(loader, cfg)
}
