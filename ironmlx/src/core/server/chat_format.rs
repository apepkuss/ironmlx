//! Chat-template rendering shared by OpenAI and Anthropic handlers.

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::core::tokenizer::Tokenizer;
use crate::core::Message;
use crate::Result;

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

/// URL payload inside an `image_url` content part.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

/// A single part in a multimodal content array.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// OpenAI-compatible message content: either a plain string or an array of
/// typed content parts.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Flatten to a plain string, replacing image parts with a fixed
    /// placeholder token sequence.
    ///
    /// The placeholder uses Qwen3.5 vision special tokens:
    ///   `<|vision_start|>` (248053) + N × `<|image_pad|>` (248056) + `<|vision_end|>` (248054)
    ///
    /// `image_token_counts` supplies the per-image N value (= grid_h/2 * grid_w/2).
    /// Image placeholders are emitted before text, matching the Qwen VL prompt
    /// layout used by the CLI and reference mlx-vlm path.
    pub fn to_flat_string(
        &self,
        image_token_counts: &mut std::collections::VecDeque<usize>,
    ) -> String {
        match self {
            Content::Text(t) => t.clone(),
            Content::Parts(parts) => {
                let mut image_buf = String::new();
                let mut text_buf = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => text_buf.push_str(text),
                        ContentPart::ImageUrl { .. } => {
                            let n = image_token_counts.pop_front().unwrap_or(1);
                            image_buf.push_str("<|vision_start|>");
                            for _ in 0..n {
                                image_buf.push_str("<|image_pad|>");
                            }
                            image_buf.push_str("<|vision_end|>");
                        }
                    }
                }
                image_buf.push_str(&text_buf);
                image_buf
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// Subset of OpenAI/Anthropic chat-message shape that both APIs surface.
/// `content` may be either a plain string (text-only messages) or an array of
/// typed content parts (multimodal messages).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Content,
}

impl ChatMessage {
    /// Convenience constructor for text-only messages.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: Content::Text(content.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// render_and_encode
// ---------------------------------------------------------------------------

/// Apply the model's chat template to render `messages` to a single prompt
/// string, then tokenize. Returns the token ids feeding into
/// [`crate::core::generate::GenerationStream`]. `chat_template_kwargs`,
/// when present, is forwarded as additional template render-context
/// variables (e.g. `enable_thinking` for Qwen3+ thinking-mode toggle).
///
/// **Precondition:** All `ChatMessage::content` values must be `Content::Text`
/// at call time. Multimodal messages must be expanded by the caller
/// (`expand_image_parts_in_messages`) before this function is called.
/// `Content::Parts` variants will be flattened with an empty image-count
/// deque (i.e. image placeholders will have 1 pad token each), which is a
/// safe fallback but callers should prefer proper expansion.
pub fn render_and_encode(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    chat_template_kwargs: Option<&serde_json::Value>,
) -> Result<Vec<u32>> {
    if !tokenizer.has_chat_template() {
        return Err(anyhow!(
            "tokenizer has no chat_template — cannot serve /v1/chat/completions or /v1/messages"
        ));
    }
    let internal: Vec<Message> = messages
        .iter()
        .map(|m| {
            let text = match &m.content {
                Content::Text(t) => t.clone(),
                Content::Parts(_) => {
                    // Callers should have expanded parts before calling render_and_encode.
                    // Fallback: flatten without image-count info.
                    m.content
                        .to_flat_string(&mut std::collections::VecDeque::new())
                }
            };
            Message {
                role: m.role.clone(),
                content: text,
            }
        })
        .collect();
    let prompt = tokenizer.apply_chat_template(
        &internal,
        /* add_generation_prompt = */ true,
        chat_template_kwargs,
    )?;
    tokenizer.encode(&prompt, /* add_special_tokens = */ false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_deserializes_minimal_json() {
        let s = r#"{"role":"user","content":"hello"}"#;
        let m: ChatMessage = serde_json::from_str(s).unwrap();
        assert_eq!(m.role, "user");
        match &m.content {
            Content::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Content::Text"),
        }
    }

    #[test]
    fn chat_message_to_internal_message_round_trip() {
        let cm = ChatMessage::text("assistant", "ok");
        let flat = match &cm.content {
            Content::Text(t) => t.clone(),
            Content::Parts(_) => panic!("expected Text"),
        };
        let im = Message {
            role: cm.role.clone(),
            content: flat,
        };
        assert_eq!(im.role, "assistant");
        assert_eq!(im.content, "ok");
    }

    #[test]
    fn content_array_with_image_url_parsed() {
        let body = r#"
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAA="}}
            ]
        }
        "#;
        let msg: ChatMessage = serde_json::from_str(body).unwrap();
        if let Content::Parts(parts) = &msg.content {
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[0], ContentPart::Text { .. }));
            assert!(matches!(parts[1], ContentPart::ImageUrl { .. }));
        } else {
            panic!("expected Content::Parts");
        }
    }

    #[test]
    fn content_text_flat_string_identity() {
        let c = Content::Text("hello world".into());
        let result = c.to_flat_string(&mut std::collections::VecDeque::new());
        assert_eq!(result, "hello world");
    }

    #[test]
    fn content_parts_flat_string_prepends_image_placeholders() {
        let parts = vec![
            ContentPart::Text {
                text: "Look: ".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,abc".into(),
                },
            },
            ContentPart::Text {
                text: " done".into(),
            },
        ];
        let c = Content::Parts(parts);
        // Provide a count of 2 for the image
        let mut counts = std::collections::VecDeque::from([2usize]);
        let result = c.to_flat_string(&mut counts);
        assert!(result.starts_with("<|vision_start|>"));
        assert!(result.contains("<|image_pad|><|image_pad|>"));
        assert!(result.ends_with("<|vision_end|>Look:  done"));
    }
}
