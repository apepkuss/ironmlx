//! Chat-template rendering shared by OpenAI and Anthropic handlers.

use anyhow::anyhow;
use serde::Deserialize;

use crate::core::tokenizer::Tokenizer;
use crate::core::Message;
use crate::Result;

/// Subset of OpenAI/Anthropic chat-message shape that both APIs surface.
/// Both protocols accept `{"role": ..., "content": ...}`; richer content
/// (multimodal blocks, tool calls) is out of scope for P4.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Apply the model's chat template to render `messages` to a single prompt
/// string, then tokenize. Returns the token ids feeding into
/// [`crate::core::generate::GenerationStream`].
pub fn render_and_encode(tokenizer: &Tokenizer, messages: &[ChatMessage]) -> Result<Vec<u32>> {
    if !tokenizer.has_chat_template() {
        return Err(anyhow!(
            "tokenizer has no chat_template — cannot serve /v1/chat/completions or /v1/messages"
        ));
    }
    let internal: Vec<Message> = messages
        .iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    let prompt =
        tokenizer.apply_chat_template(&internal, /* add_generation_prompt = */ true)?;
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
        assert_eq!(m.content, "hello");
    }

    #[test]
    fn chat_message_to_internal_message_round_trip() {
        let cm = ChatMessage {
            role: "assistant".into(),
            content: "ok".into(),
        };
        let im = Message {
            role: cm.role.clone(),
            content: cm.content.clone(),
        };
        assert_eq!(im.role, "assistant");
        assert_eq!(im.content, "ok");
    }
}
