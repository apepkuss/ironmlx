//! MiniCPM-V-4.6 model module.
//!
//! MiniCPM-V-4.6 (`model_type = "minicpmv4_6"`,
//! `architectures = ["MiniCPMV4_6ForConditionalGeneration"]`) is a
//! vision-language model. Its language backbone is Qwen3.5-text verbatim
//! (`text_config.model_type = "qwen3_5_text"`). LM weights ship under
//! `language_model.*`, identical to a native Qwen3.5 checkpoint. The SigLIP
//! vision tower ships under `vision_tower.*` and is loaded when the checkpoint
//! was opened via [`Loader::open_multimodal`] AND contains
//! `vision_tower.embeddings.patch_embedding.weight`.
//!
//! `model_from_loader` builds the full [`MiniCpmV46Model`] (text + optional
//! vision). Text-only inference is the `vision = None` case (use
//! [`Loader::open`] to drop vision weights).

pub mod config;
pub mod image_processor;
pub mod model;
pub mod vision;

pub use config::text_config_from_loader;
pub use model::MiniCpmV46Model;

use crate::core::Loader;
use crate::Result;

/// Build a [`MiniCpmV46Model`] from a MiniCPM-V-4.6 checkpoint.
///
/// Parses the nested `text_config` into a `Qwen35Config`, loads the Qwen3.5
/// text backbone, and — when the loader was opened via `open_multimodal` AND
/// contains vision-tower weights — loads the SigLIP vision tower too.
/// Text-only callers use [`Loader::open`] which drops `vision_tower.*` keys;
/// the resulting model has `vision = None` and behaves identically to the
/// pure-text path.
pub fn model_from_loader(loader: &Loader) -> Result<MiniCpmV46Model> {
    MiniCpmV46Model::from_loader(loader)
}
