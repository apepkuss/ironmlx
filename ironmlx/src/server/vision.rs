//! HTTP message adaptation and bounded preprocessing dispatch.

use super::chat_format::{ChatMessage, Content};
use ironmlx_lm::core::vision_input::DecodedMessage;
use ironmlx_lm::core::vision_input::VisionInputConfig;
use mlx::Array;

pub type ExpandedVisionInputs = (Vec<ChatMessage>, Option<Vec<Array>>, Vec<(i32, i32, i32)>);

pub async fn expand_decoded_messages_bounded(
    messages: Vec<DecodedMessage>,
    vision_input: VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    let prepared = ironmlx_runtime::core::input_preparation::expand_decoded_messages_bounded(
        messages,
        vision_input,
    )
    .await?;
    Ok(adapt_prepared_messages(prepared))
}

pub fn expand_decoded_messages(
    messages: Vec<DecodedMessage>,
    vision_input: &VisionInputConfig,
) -> anyhow::Result<ExpandedVisionInputs> {
    let (messages, pixels, grid) =
        ironmlx_lm::core::vision_input::expand_decoded_messages(messages, vision_input)?;
    Ok(adapt_prepared_messages((messages, pixels, grid)))
}

fn adapt_prepared_messages(
    (messages, pixels, grid): ironmlx_lm::core::vision_input::ExpandedVisionInputs,
) -> ExpandedVisionInputs {
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
    (messages, pixels, grid)
}
