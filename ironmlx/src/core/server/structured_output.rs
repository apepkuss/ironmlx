use anyhow::Context;
use serde::{Deserialize, Serialize};

use super::chat_format::{ChatMessage, Content, ContentPart};

/// Protocol-neutral final-answer format shared by Chat Completions and
/// Responses. Each adapter is responsible for mapping its wire shape into
/// this representation.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum StructuredOutputFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        schema: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

impl StructuredOutputFormat {
    pub(crate) fn constraint_schema(&self) -> Option<serde_json::Value> {
        match self {
            Self::Text => None,
            Self::JsonObject => Some(serde_json::json!({"type":"object"})),
            Self::JsonSchema { schema, .. } => Some(schema.clone()),
        }
    }

    pub(crate) fn prompt_instruction(&self) -> Option<String> {
        match self {
            Self::Text => None,
            Self::JsonObject => Some(
                "When producing a final answer instead of a function call, return only one valid JSON object with no Markdown or surrounding prose."
                    .to_owned(),
            ),
            Self::JsonSchema {
                name,
                description,
                schema,
                ..
            } => {
                let description = description
                    .as_deref()
                    .map(|value| format!("\nDescription: {value}"))
                    .unwrap_or_default();
                Some(format!(
                    "When producing a final answer instead of a function call, return only one JSON object matching the `{name}` schema, with no Markdown or surrounding prose.{description}\nSchema: {}",
                    serde_json::to_string(schema).expect("structured output schema serializes")
                ))
            }
        }
    }

    pub(crate) fn apply_prompt_instruction(&self, messages: &mut Vec<ChatMessage>) {
        let Some(instruction) = self.prompt_instruction() else {
            return;
        };
        messages.push(ChatMessage::text("system", instruction));
        coalesce_system_messages(messages);
    }

    pub(crate) fn validate_contract(&self, field: &str) -> anyhow::Result<()> {
        let Self::JsonSchema {
            name,
            schema,
            strict,
            ..
        } = self
        else {
            return Ok(());
        };
        crate::core::tool_calling::validate_function_name(name)
            .with_context(|| format!("invalid {field}.json_schema name"))?;
        crate::core::constrained::validate_json_output_schema(schema, strict.unwrap_or(false))
    }

    fn validate_output(&self, text: &str) -> anyhow::Result<()> {
        let value = match self {
            Self::Text => return Ok(()),
            Self::JsonObject | Self::JsonSchema { .. } => {
                serde_json::from_str::<serde_json::Value>(text.trim()).map_err(|error| {
                    anyhow::anyhow!("structured output is not valid JSON: {error}")
                })?
            }
        };
        match self {
            Self::Text => Ok(()),
            Self::JsonObject => {
                anyhow::ensure!(
                    value.is_object(),
                    "json_object output must be a JSON object"
                );
                Ok(())
            }
            Self::JsonSchema { schema, .. } => {
                crate::core::constrained::validate_schema_value(schema, &value)
                    .context("structured output does not match requested schema")
            }
        }
    }

    pub(crate) fn validate_completion(
        &self,
        text: &str,
        has_tool_calls: bool,
        finish_reason: &'static str,
    ) -> anyhow::Result<()> {
        if has_tool_calls || matches!(finish_reason, "length" | "max_tokens") {
            return Ok(());
        }
        self.validate_output(text)
    }
}

pub(crate) fn coalesce_system_messages(messages: &mut Vec<ChatMessage>) {
    let mut system_contents = Vec::new();
    messages.retain(|message| {
        if message.role == "system" {
            system_contents.push(message.content.clone());
            false
        } else {
            true
        }
    });
    if system_contents.is_empty() {
        return;
    }
    let all_text = system_contents
        .iter()
        .all(|content| matches!(content, Content::Text(_)));
    let content = if all_text {
        Content::Text(
            system_contents
                .into_iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text),
                    Content::Parts(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    } else {
        let mut parts = Vec::new();
        for (index, content) in system_contents.into_iter().enumerate() {
            if index > 0 {
                parts.push(ContentPart::Text {
                    text: "\n\n".to_owned(),
                });
            }
            match content {
                Content::Text(text) => parts.push(ContentPart::Text { text }),
                Content::Parts(content_parts) => parts.extend(content_parts),
            }
        }
        Content::Parts(parts)
    };
    messages.insert(
        0,
        ChatMessage {
            role: "system".to_owned(),
            content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer_format() -> StructuredOutputFormat {
        StructuredOutputFormat::JsonSchema {
            name: "answer".to_owned(),
            description: None,
            schema: serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    #[test]
    fn validates_complete_output_but_allows_length_truncation_and_tool_calls() {
        let format = answer_format();
        format
            .validate_completion(r#"{"answer":"ok"}"#, false, "stop")
            .unwrap();
        assert!(format
            .validate_completion(r#"{"wrong":true}"#, false, "stop")
            .is_err());
        format
            .validate_completion("{\"answer\":\"", false, "length")
            .unwrap();
        format
            .validate_completion("{\"answer\":\"", false, "max_tokens")
            .unwrap();
        format.validate_completion("", true, "tool_calls").unwrap();
    }

    #[test]
    fn structured_instruction_coalesces_existing_system_messages() {
        let mut messages = vec![
            ChatMessage::text("system", "Be concise."),
            ChatMessage::text("user", "Answer."),
        ];
        answer_format().apply_prompt_instruction(&mut messages);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(matches!(
            &messages[0].content,
            Content::Text(text)
                if text.contains("Be concise.") && text.contains("`answer` schema")
        ));
    }
}
