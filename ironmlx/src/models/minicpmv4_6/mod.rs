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

/// Build the MiniCPM-V-4.6 image placeholder string:
/// `<image>` + `<|image_pad|>` × `n` + `</image>`.
///
/// All three tokens are registered special tokens; when tokenised they produce
/// the id sequence `[248078] + [248056]*n + [248079]`, which exactly matches
/// the P2a gen-script (`gen_single_image_logits.py`, `use_image_id=False`,
/// `slice_mode=False`) fixture `expected_input_ids_img.npy`.
///
/// This is the single source of truth for image-token injection in both the
/// CLI (`cli/generate.rs`) and the HTTP serve path (`core/server/openai.rs`).
pub fn image_placeholder_string(n: usize) -> String {
    let mut out =
        String::with_capacity("<image>".len() + n * "<|image_pad|>".len() + "</image>".len());
    out.push_str("<image>");
    for _ in 0..n {
        out.push_str("<|image_pad|>");
    }
    out.push_str("</image>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_placeholder_string_format() {
        // N=3 → ids [248078, 248056×3, 248079] when tokenised.
        assert_eq!(
            image_placeholder_string(3),
            "<image><|image_pad|><|image_pad|><|image_pad|></image>"
        );
    }

    #[test]
    fn image_placeholder_string_zero() {
        assert_eq!(image_placeholder_string(0), "<image></image>");
    }
}

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
