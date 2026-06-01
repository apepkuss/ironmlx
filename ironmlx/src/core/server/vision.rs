//! Wire-agnostic vision normalization shared by the OpenAI and Anthropic HTTP
//! handlers. Each endpoint decodes ITS OWN wire format (OpenAI `image_url`,
//! Anthropic `image`+`source`) into the neutral [`DecodedMessage`] structure;
//! [`expand_decoded_messages`] then runs the per-model preprocess + placeholder
//! rewrite shared by both. The `forward_vl` backend downstream is untouched.

use mlx::Array;

use crate::core::server::chat_format::{ChatMessage, Content};
use crate::core::server::VisionInputConfig;
use crate::core::tokenizer::Tokenizer;
use crate::models::{gemma4, qwen3_5};

/// A single content part after the endpoint's wire format has been decoded to
/// raw bytes (protocol-agnostic).
pub enum DecodedPart {
    Text(String),
    Image(Vec<u8>),
}

/// A message after wire-format decoding. Both endpoints normalize into this
/// before calling [`expand_decoded_messages`].
pub struct DecodedMessage {
    pub role: String,
    pub parts: Vec<DecodedPart>,
}

/// Qwen3.5-VL placeholder: `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>`.
fn qwen_placeholder(n: usize) -> String {
    let mut s = String::from("<|vision_start|>");
    for _ in 0..n {
        s.push_str("<|image_pad|>");
    }
    s.push_str("<|vision_end|>");
    s
}

/// Gemma4 placeholder: `<|image>` + N × `<|image|>` + `<image|>`.
fn gemma4_placeholder(n: usize) -> String {
    let mut s = String::from("<|image>");
    for _ in 0..n {
        s.push_str("<|image|>");
    }
    s.push_str("<image|>");
    s
}

/// Derive `(image_token_id, spatial_merge_size)` for the active model. Both
/// endpoints need `image_token_id` to populate `GenerateRequest`, so this is
/// the single shared source of the derivation (was inline in `openai.rs`).
pub fn derive_image_token_and_merge(
    vision_input: &VisionInputConfig,
    tokenizer: &Tokenizer,
) -> (i32, i32) {
    match vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => (
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(crate::core::generate::IMAGE_TOKEN_ID),
            *spatial_merge_size,
        ),
        VisionInputConfig::Gemma4 { vision_config } => (
            tokenizer
                .token_to_id("<|image|>")
                .map(|id| id as i32)
                .unwrap_or(258_880),
            vision_config.pooling_kernel_size,
        ),
        VisionInputConfig::MiniCpmV46 { spatial_merge_size } => (
            tokenizer
                .token_to_id("<|image_pad|>")
                .map(|id| id as i32)
                .unwrap_or(248_056),
            *spatial_merge_size,
        ),
    }
}

/// Output of [`expand_decoded_messages`]: `(flat_text_messages, pixel_values, image_grid_thw)`.
pub type ExpandedVisionInputs = (Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>);

/// For each `DecodedPart::Image`, run the `vision_input`-specific preprocess
/// (Qwen / Gemma4 / MiniCpmV46), collect `pixel_values` + `grid_thw`, and
/// rewrite every message to plain text with placeholder tokens inserted at the
/// image positions. Wire- and endpoint-agnostic.
///
/// Returns `(flat_text_messages, pixel_values, image_grid_thw)`:
/// - `flat_text_messages` feeds `render_and_encode`,
/// - `pixel_values` is `None` when there are no images (eagerly `eval`'d before
///   return so the tensors are safe to cross into `spawn_blocking`),
/// - `image_grid_thw` has one entry per image (MiniCPM-V multi-slice: one per slice).
pub fn expand_decoded_messages(
    messages: Vec<DecodedMessage>,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    let spatial_merge_size = match vision_input {
        VisionInputConfig::Qwen { spatial_merge_size } => *spatial_merge_size,
        VisionInputConfig::Gemma4 { vision_config } => vision_config.pooling_kernel_size,
        VisionInputConfig::MiniCpmV46 { spatial_merge_size } => *spatial_merge_size,
    };
    if spatial_merge_size <= 0 {
        return Err(anyhow::anyhow!(
            "expand_decoded_messages: spatial_merge_size must be > 0 (got {spatial_merge_size})"
        ));
    }

    let mut all_pixel_values: Vec<Array> = Vec::new();
    let mut grid_thw: Vec<(i32, i32, i32)> = Vec::new();
    let mut placeholders: Vec<String> = Vec::new();

    // First pass: preprocess every image part in order.
    for msg in &messages {
        for part in &msg.parts {
            if let DecodedPart::Image(img_bytes) = part {
                match vision_input {
                    VisionInputConfig::Qwen { .. } => {
                        let (pv, gh, gw) = qwen3_5::image_processor::preprocess(img_bytes)?;
                        let n = ((gh / spatial_merge_size) * (gw / spatial_merge_size)) as usize;
                        placeholders.push(qwen_placeholder(n));
                        all_pixel_values.push(pv);
                        grid_thw.push((1, gh, gw));
                    }
                    VisionInputConfig::Gemma4 { vision_config } => {
                        let processed =
                            gemma4::image_processor::preprocess(img_bytes, vision_config)?;
                        placeholders.push(gemma4_placeholder(processed.soft_tokens));
                        grid_thw.push((1, processed.grid_h, processed.grid_w));
                        all_pixel_values.push(processed.pixel_values);
                    }
                    VisionInputConfig::MiniCpmV46 { .. } => {
                        // Multi-slice (LLaVA-UHD): single source of truth for the
                        // divisibility guard, token count, and placeholder.
                        let parts = crate::models::minicpmv4_6::preprocess_sliced_to_parts(
                            img_bytes,
                            spatial_merge_size,
                        )?;
                        all_pixel_values.extend(parts.pixel_values);
                        grid_thw.extend(parts.grid_thw);
                        placeholders.push(parts.placeholder);
                    }
                }
            }
        }
    }

    // Second pass: rewrite messages to plain text with placeholders in-order.
    let mut placeholders = placeholders.into_iter();
    let flat_messages: Vec<ChatMessage> = messages
        .into_iter()
        .map(|msg| {
            let mut out = String::new();
            for part in msg.parts {
                match part {
                    DecodedPart::Text(text) => out.push_str(&text),
                    DecodedPart::Image(_) => {
                        out.push_str(&placeholders.next().unwrap_or_default());
                    }
                }
            }
            ChatMessage {
                role: msg.role,
                content: Content::Text(out),
            }
        })
        .collect();

    let pixel_values = if all_pixel_values.is_empty() {
        None
    } else {
        // Eagerly materialize on this thread before the tensor crosses into
        // spawn_blocking, where a different worker thread's default MLX stream
        // cannot evaluate this thread's lazy graph.
        for pv in &all_pixel_values {
            mlx::transforms::eval(&[pv])?;
        }
        Some(all_pixel_values)
    };

    Ok((flat_messages, pixel_values, grid_thw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_qwen_wraps_n_pads() {
        assert_eq!(
            qwen_placeholder(2),
            "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[test]
    fn placeholder_gemma4_wraps_n_pads() {
        assert_eq!(gemma4_placeholder(2), "<|image><|image|><|image|><image|>");
    }

    #[test]
    fn text_only_message_passes_through_with_no_pixels() {
        let msgs = vec![DecodedMessage {
            role: "user".to_string(),
            parts: vec![DecodedPart::Text("hello".to_string())],
        }];
        let (flat, pv, grid) = expand_decoded_messages(
            msgs,
            &VisionInputConfig::Qwen {
                spatial_merge_size: 2,
            },
        )
        .unwrap();
        assert_eq!(flat.len(), 1);
        match &flat[0].content {
            Content::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Content::Text"),
        }
        assert!(pv.is_none());
        assert!(grid.is_empty());
    }
}
