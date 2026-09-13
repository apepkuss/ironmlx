//! HTTP message adaptation and bounded preprocessing dispatch.

use super::chat_format::{ChatMessage, Content};
use crate::core::vision_input::VisionInputConfig;
pub use crate::core::vision_input::{derive_image_token_and_merge, DecodedMessage, DecodedPart};
use mlx::Array;

pub type ExpandedVisionInputs = (Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>);

static IMAGE_PREPROCESS_SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

pub async fn expand_decoded_messages_bounded(
    messages: Vec<DecodedMessage>,
    vision_input: VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    let permit = IMAGE_PREPROCESS_SEMAPHORE
        .acquire()
        .await
        .map_err(|_| anyhow::anyhow!("image preprocessing is unavailable"))?;
    let result =
        tokio::task::spawn_blocking(move || expand_decoded_messages(messages, &vision_input))
            .await
            .map_err(|error| anyhow::anyhow!("image preprocessing task failed: {error}"))?;
    drop(permit);
    result
}

pub fn expand_decoded_messages(
    messages: Vec<DecodedMessage>,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    let (messages, pixels, grid) =
        crate::core::vision_input::expand_decoded_messages(messages, vision_input)?;
    let messages = messages
        .into_iter()
        .map(|message| ChatMessage {
            role: message.role,
            content: Content::Text(message.content),
            reasoning_content: message.reasoning_content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        })
        .collect();
    Ok((messages, pixels, grid))
}
