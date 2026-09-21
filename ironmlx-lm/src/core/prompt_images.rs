//! Model-specific image preprocessing and prompt placeholders for native callers.
use crate::core::model_input::IMAGE_TOKEN_ID;
use crate::models::qwen3_5::image_processor;
use crate::{Loader, Result, Tokenizer};
use anyhow::{anyhow, Context};
use mlx::Array;

/// Decoded input source bytes and a caller-supplied diagnostic label.
pub struct NamedImage {
    pub label: String,
    pub bytes: Vec<u8>,
}

pub struct PreparedImages {
    pub pixel_values: Option<Vec<Array>>,
    pub image_grid_thw: Option<Vec<(i32, i32, i32)>>,
    pub placeholders: Vec<String>,
    pub image_spatial_merge_size: i32,
    pub image_token_id: i32,
}

fn image_token_count_for_grid(grid: (i32, i32, i32), spatial_merge_size: i32) -> Result<usize> {
    let (_t, gh, gw) = grid;
    if spatial_merge_size <= 0 {
        return Err(anyhow!(
            "image_spatial_merge_size must be > 0, got {spatial_merge_size}"
        ));
    }
    if gh % spatial_merge_size != 0 || gw % spatial_merge_size != 0 {
        return Err(anyhow!(
            "image grid {gh}x{gw} is not divisible by spatial_merge_size={spatial_merge_size}"
        ));
    }
    Ok(((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize)
}

fn qwen_image_placeholder_string(token_count: usize) -> String {
    let mut out = String::with_capacity(
        "<|vision_start|>".len() + token_count * "<|image_pad|>".len() + "<|vision_end|>".len(),
    );
    out.push_str("<|vision_start|>");
    for _ in 0..token_count {
        out.push_str("<|image_pad|>");
    }
    out.push_str("<|vision_end|>");
    out
}

fn gemma4_placeholder(token_count: usize) -> String {
    let mut out = String::from("<|image>");
    for _ in 0..token_count {
        out.push_str("<|image|>");
    }
    out.push_str("<image|>");
    out
}

fn diffusion_gemma_placeholder(token_count: usize) -> String {
    gemma4_placeholder(token_count)
}

pub fn inject_image_placeholders(prompt: &str, placeholders: &[String]) -> Result<String> {
    if placeholders.is_empty() {
        return Ok(prompt.to_owned());
    }

    let marker = "<image>";
    let marker_count = prompt.match_indices(marker).count();
    if marker_count == 0 {
        let mut out = String::new();
        for placeholder in placeholders {
            out.push_str(placeholder);
        }
        out.push_str(prompt);
        return Ok(out);
    }

    if marker_count != placeholders.len() {
        return Err(anyhow!(
            "prompt contains {marker_count} <image> markers but {} --image arguments were provided",
            placeholders.len()
        ));
    }

    let mut out =
        String::with_capacity(prompt.len() + placeholders.iter().map(String::len).sum::<usize>());
    let mut rest = prompt;
    for placeholder in placeholders {
        let Some(idx) = rest.find(marker) else {
            break;
        };
        out.push_str(&rest[..idx]);
        out.push_str(placeholder);
        rest = &rest[idx + marker.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

pub fn prepare_images(
    images: impl ExactSizeIterator<Item = Result<NamedImage>>,
    loader: &Loader,
    tokenizer: &Tokenizer,
    model_type: &str,
    default_spatial_merge_size: i32,
) -> Result<PreparedImages> {
    if images.len() == 0 {
        return Ok(PreparedImages {
            pixel_values: None,
            image_grid_thw: None,
            placeholders: Vec::new(),
            image_spatial_merge_size: default_spatial_merge_size,
            image_token_id: tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(IMAGE_TOKEN_ID),
        });
    }

    let mut all_pixel_values = Vec::with_capacity(images.len());
    let mut grids = Vec::with_capacity(images.len());
    let mut placeholders = Vec::with_capacity(images.len());

    let (spatial_merge_size, image_token_id) = if model_type == "gemma4" {
        let cfg = crate::models::gemma4::Gemma4Config::from_loader(loader)
            .context("Gemma4Config::from_loader")?;
        let vision_config = cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| anyhow!("Gemma4 config has no vision_config"))?;
        for image in images {
            let NamedImage { label, bytes } = image?;
            let processed =
                crate::models::gemma4::image_processor::preprocess(&bytes, vision_config)
                    .with_context(|| format!("preprocessing {label}"))?;
            all_pixel_values.push(processed.pixel_values);
            grids.push((1, processed.grid_h, processed.grid_w));
            placeholders.push(gemma4_placeholder(processed.soft_tokens));
        }
        (
            vision_config.pooling_kernel_size,
            tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .or(cfg.image_token_id)
                .unwrap_or(258_880),
        )
    } else if model_type == "diffusion_gemma" {
        let cfg = crate::models::DiffusionGemmaConfig::from_loader(loader)
            .context("DiffusionGemmaConfig::from_loader")?;
        let vision_config = cfg
            .vision_config
            .as_ref()
            .ok_or_else(|| anyhow!("DiffusionGemma config has no vision_config"))?;
        for image in images {
            let NamedImage { label, bytes } = image?;
            let processed =
                crate::models::gemma4::image_processor::preprocess(&bytes, vision_config)
                    .with_context(|| format!("preprocessing {label}"))?;
            all_pixel_values.push(processed.pixel_values);
            grids.push((1, processed.grid_h, processed.grid_w));
            placeholders.push(diffusion_gemma_placeholder(processed.soft_tokens));
        }
        (
            vision_config.pooling_kernel_size,
            tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .or(cfg.image_token_id)
                .unwrap_or(258_880),
        )
    } else if model_type == "minicpmv4_6" {
        // MiniCPM-V-4.6: use model-config image_token_id (248056 = <|image_pad|>);
        // spatial_merge_size = 4 (2×2 Merger, "16x" downsample mode).
        // Multi-slice (LLaVA-UHD): source slice first, then refine patches row-major.
        // preprocess_sliced_to_parts is the single source of truth for the
        // divisibility guard and placeholder construction (CLI + serve share it).
        let vcfg = crate::models::minicpmv4_6::config::MiniCpmV46VisionConfig::from_loader(loader)
            .context("MiniCpmV46VisionConfig::from_loader")?;
        // image_token_id: tokenizer lookup (<|image_pad|> → 248056) first; fallback to config image_token_id.
        let image_tok_id = tokenizer
            .token_to_id("<|image_pad|>")
            .map(|id| id as i32)
            .unwrap_or(vcfg.image_token_id);
        for image in images {
            let NamedImage { label, bytes } = image?;
            let parts = crate::models::minicpmv4_6::preprocess_sliced_to_parts(
                &bytes,
                default_spatial_merge_size,
            )
            .with_context(|| format!("preprocessing {label}"))?;
            all_pixel_values.extend(parts.pixel_values);
            grids.extend(parts.grid_thw);
            placeholders.push(parts.placeholder);
        }
        (default_spatial_merge_size, image_tok_id)
    } else {
        for image in images {
            let NamedImage { label, bytes } = image?;
            let (pixel_values, gh, gw) = image_processor::preprocess(&bytes)
                .with_context(|| format!("preprocessing {label}"))?;
            let grid = (1, gh, gw);
            let token_count = image_token_count_for_grid(grid, default_spatial_merge_size)?;
            all_pixel_values.push(pixel_values);
            grids.push(grid);
            placeholders.push(qwen_image_placeholder_string(token_count));
        }
        (
            default_spatial_merge_size,
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(IMAGE_TOKEN_ID),
        )
    };

    Ok(PreparedImages {
        pixel_values: Some(all_pixel_values),
        image_grid_thw: Some(grids),
        placeholders,
        image_spatial_merge_size: spatial_merge_size,
        image_token_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_token_count_uses_spatial_merge_size() {
        assert_eq!(image_token_count_for_grid((1, 4, 6), 2).unwrap(), 6);
    }

    #[test]
    fn minicpmv46_image_token_count_uses_4x_downsample() {
        // MiniCPM-V grid (28,36) → vision tokens (28/4)*(36/4) = 63.
        assert_eq!(image_token_count_for_grid((1, 28, 36), 4).unwrap(), 63);
    }

    #[test]
    fn minicpmv46_placeholder_wraps_correct_tokens() {
        // Verify the canonical fn builds the correct <image>...<|image_pad|>...</image> string.
        let s = crate::models::minicpmv4_6::image_placeholder_string(3);
        assert_eq!(s, "<image><|image_pad|><|image_pad|><|image_pad|></image>");
    }

    #[test]
    fn diffusion_gemma_placeholder_wraps_image_soft_tokens() {
        assert_eq!(
            diffusion_gemma_placeholder(2),
            "<|image><|image|><|image|><image|>"
        );
    }

    #[test]
    fn inject_image_placeholders_replaces_markers_in_order() {
        let out = inject_image_placeholders(
            "A <image> then B <image>",
            &["[img0]".to_owned(), "[img1]".to_owned()],
        )
        .unwrap();
        assert_eq!(out, "A [img0] then B [img1]");
    }

    #[test]
    fn inject_image_placeholders_prepends_when_prompt_has_no_markers() {
        let out = inject_image_placeholders(
            "Describe this.",
            &["[img0]".to_owned(), "[img1]".to_owned()],
        )
        .unwrap();
        assert_eq!(out, "[img0][img1]Describe this.");
    }

    #[test]
    fn inject_image_placeholders_rejects_marker_count_mismatch() {
        let err =
            inject_image_placeholders("A <image>", &["[img0]".to_owned(), "[img1]".to_owned()])
                .expect_err("marker mismatch");
        assert!(err.to_string().contains("markers"));
    }

    /// Verify the slice→token-count + placeholder wiring without a real model or
    /// image decode. Synthetic slice list: source grid (28,36), two refine patches
    /// (40,28) each — matching a 640×480 coco-sample-like image with best_grid (2,1).
    ///
    /// source_tokens = (28/4)*(36/4) = 7*9 = 63
    /// slice_tokens  = (40/4)*(28/4) = 10*7 = 70
    /// grid (2,1) → 1 row × 2 cols → 2 <slice> blocks, 0 inter-row newlines
    #[test]
    fn minicpmv46_multislice_token_count_and_placeholder_wiring() {
        let spatial_merge_size = 4_i32;
        let best_grid = (2, 1);

        // Synthetic per-slice (gh, gw) pairs: [source, patch0, patch1].
        let slice_grids: Vec<(i32, i32)> = vec![(28, 36), (40, 28), (40, 28)];

        // Source tokens: slice[0].
        let (src_gh, src_gw) = slice_grids[0];
        let source_tokens =
            ((src_gh / spatial_merge_size) * (src_gw / spatial_merge_size)) as usize;
        assert_eq!(source_tokens, 63, "source_tokens = (28/4)*(36/4) = 63");

        // Slice tokens: slice[1] (first patch; all patches have the same grid).
        let slice_tokens = if slice_grids.len() > 1 {
            let (sl_gh, sl_gw) = slice_grids[1];
            ((sl_gh / spatial_merge_size) * (sl_gw / spatial_merge_size)) as usize
        } else {
            0
        };
        assert_eq!(slice_tokens, 70, "slice_tokens = (40/4)*(28/4) = 70");

        // Build the placeholder using the canonical function.
        let grid = best_grid;
        let placeholder = crate::models::minicpmv4_6::sliced_image_placeholder_string(
            source_tokens,
            slice_tokens,
            grid,
        );

        // Structural checks: 1 <image>, 2 <slice>, no inter-row newlines.
        assert_eq!(
            placeholder.matches("<image>").count(),
            1,
            "exactly one <image> block"
        );
        assert_eq!(
            placeholder.matches("</image>").count(),
            1,
            "exactly one </image>"
        );
        assert_eq!(
            placeholder.matches("<slice>").count(),
            2,
            "grid (2,1) → 2 <slice> blocks"
        );
        assert_eq!(
            placeholder.matches("</slice>").count(),
            2,
            "grid (2,1) → 2 </slice>"
        );
        assert_eq!(
            placeholder.matches('\n').count(),
            0,
            "single row → 0 inter-row newlines"
        );

        // Token counts embedded in placeholder.
        let pad_count = placeholder.matches("<|image_pad|>").count();
        assert_eq!(
            pad_count,
            source_tokens + 2 * slice_tokens,
            "total pads = 63 + 2*70 = 203"
        );
    }
}
