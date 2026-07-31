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
mod qualification;
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

/// Build the MiniCPM-V-4.6 sliced-image prompt placeholder string.
///
/// `source_tokens` = source slice's image-token count = `(source_gh/4)*(source_gw/4)`.
/// `slice_tokens`  = per patch-slice image-token count = `(slice_gh/4)*(slice_gw/4)`.
/// `grid`          = `(grid_x, grid_y)` = the slice grid (grid_x columns, grid_y rows).
///
/// Produces:
/// ```text
/// <image>{<|image_pad|> × source_tokens}</image>
/// ```
/// followed, when `grid_x * grid_y > 0`, by:
/// ```text
/// for row in 0..grid_y:
///   grid_x × (<slice>{<|image_pad|> × slice_tokens}</slice>)
///   if row != grid_y - 1: "\n"
/// ```
///
/// This convention mirrors `_build_placeholder_ids_for_image` in
/// `processing_minicpmv4_6.py` (lines 1058-1064), where the newline is
/// appended after each row *except* the last.
///
/// When `grid` is `(0, 0)` or `grid_x * grid_y == 0`, the result equals
/// `image_placeholder_string(source_tokens)` (no-slice path).
pub fn sliced_image_placeholder_string(
    source_tokens: usize,
    slice_tokens: usize,
    grid: (i32, i32),
) -> String {
    let (grid_x, grid_y) = grid;
    let no_slices = grid_x <= 0 || grid_y <= 0;

    // Capacity estimate: image block + slice blocks + newlines.
    let slice_block_len = "<slice>".len() + slice_tokens * "<|image_pad|>".len() + "</slice>".len();
    let slice_count = if no_slices {
        0usize
    } else {
        (grid_x as usize) * (grid_y as usize)
    };
    let newline_count = if no_slices || grid_y <= 1 {
        0usize
    } else {
        (grid_y as usize) - 1
    };
    let capacity = "<image>".len()
        + source_tokens * "<|image_pad|>".len()
        + "</image>".len()
        + slice_count * slice_block_len
        + newline_count;

    let mut out = String::with_capacity(capacity);

    // Source / overview block — always present.
    out.push_str("<image>");
    for _ in 0..source_tokens {
        out.push_str("<|image_pad|>");
    }
    out.push_str("</image>");

    if no_slices {
        return out;
    }

    // Patch-slice blocks: grid_y rows × grid_x cols; "\n" between rows.
    for row_idx in 0..grid_y {
        for _ in 0..grid_x {
            out.push_str("<slice>");
            for _ in 0..slice_tokens {
                out.push_str("<|image_pad|>");
            }
            out.push_str("</slice>");
        }
        if row_idx != grid_y - 1 {
            out.push('\n');
        }
    }

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

    #[test]
    fn sliced_placeholder_single_row() {
        // grid (2,1): 1 row, 2 cols → 1 <image> + 2 <slice>, NO newline.
        let s = sliced_image_placeholder_string(4, 2, (2, 1));
        assert_eq!(s.matches("<image>").count(), 1);
        assert_eq!(s.matches("</image>").count(), 1);
        assert_eq!(s.matches("<slice>").count(), 2);
        assert_eq!(s.matches("</slice>").count(), 2);
        assert_eq!(s.matches("<|image_pad|>").count(), 4 + 2 * 2);
        assert_eq!(s.matches('\n').count(), 0); // single row
                                                // structure: <image> block comes before any <slice>
        assert!(s.find("<image>").unwrap() < s.find("<slice>").unwrap());
    }

    #[test]
    fn sliced_placeholder_multi_row_has_inter_row_newlines() {
        // grid (2,2): 2 rows × 2 cols → 4 <slice>, exactly 1 newline (between the 2 rows).
        let s = sliced_image_placeholder_string(4, 2, (2, 2));
        assert_eq!(s.matches("<slice>").count(), 4);
        assert_eq!(s.matches('\n').count(), 1);
    }

    #[test]
    fn sliced_placeholder_asymmetric_grid() {
        // grid (3,2): grid_x=3 cols, grid_y=2 rows → 6 <slice> blocks, exactly 1 inter-row newline.
        let s = sliced_image_placeholder_string(4, 2, (3, 2));
        assert_eq!(s.matches("<slice>").count(), 6);
        assert_eq!(s.matches("</slice>").count(), 6);
        assert_eq!(s.matches("<|image_pad|>").count(), 4 + 6 * 2); // source(4) + 6 slices × 2
        assert_eq!(s.matches('\n').count(), 1); // grid_y - 1
    }

    #[test]
    fn sliced_placeholder_no_slice_equals_image_placeholder() {
        assert_eq!(
            sliced_image_placeholder_string(7, 0, (0, 0)),
            image_placeholder_string(7)
        );
    }
}

/// One image's sliced parts for a VL request: per-slice pixel tensors (source
/// first, then patches row-major), their `(1, gh, gw)` grids, and the prompt
/// placeholder string. Single source of truth for both CLI and HTTP serve.
pub struct SlicedImageParts {
    pub pixel_values: Vec<mlx::Array>,
    pub grid_thw: Vec<(i32, i32, i32)>,
    pub placeholder: String,
}

/// Preprocess one image into its sliced VL parts.
///
/// `spatial_merge_size` is the effective vision downsample (= 4 for MiniCPM-V-4.6).
/// Token count per slice = `(gh / spatial_merge_size) * (gw / spatial_merge_size)`.
///
/// Errors if `spatial_merge_size <= 0` or if any slice grid dimension is not
/// divisible by `spatial_merge_size` — this is the single divisibility guard for
/// both the CLI and the HTTP serve path.
///
/// Push order: source first, then refine patches row-major (image-major across
/// images, matching `replace_image_tokens` scatter order).
pub fn preprocess_sliced_to_parts(
    img_bytes: &[u8],
    spatial_merge_size: i32,
) -> crate::Result<SlicedImageParts> {
    use anyhow::ensure;
    ensure!(
        spatial_merge_size > 0,
        "spatial_merge_size must be > 0, got {spatial_merge_size}"
    );

    let (slices, best_grid) =
        image_processor::preprocess_sliced_with_grid(img_bytes, image_processor::MAX_SLICE_NUMS)?;

    let tok = |gh: i32, gw: i32| -> crate::Result<usize> {
        ensure!(
            gh % spatial_merge_size == 0 && gw % spatial_merge_size == 0,
            "slice grid {gh}x{gw} is not divisible by spatial_merge_size={spatial_merge_size}"
        );
        Ok(((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize)
    };

    let (_, src_gh, src_gw) = slices[0];
    let source_tokens = tok(src_gh, src_gw)?;
    let slice_tokens = if slices.len() > 1 {
        let (_, sl_gh, sl_gw) = slices[1];
        tok(sl_gh, sl_gw)?
    } else {
        0
    };

    let grid = best_grid.unwrap_or((0, 0));
    let placeholder = sliced_image_placeholder_string(source_tokens, slice_tokens, grid);

    let mut pixel_values = Vec::with_capacity(slices.len());
    let mut grid_thw = Vec::with_capacity(slices.len());
    for (pv, gh, gw) in slices {
        pixel_values.push(pv);
        grid_thw.push((1, gh, gw));
    }

    Ok(SlicedImageParts {
        pixel_values,
        grid_thw,
        placeholder,
    })
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
