//! Protocol-neutral typed output events and committed-token decoding.
//!
//! Generation engines remain responsible for sampling and committing token
//! ids. This module starts only after that commit boundary, so speculative
//! drafts, rollbacks, and diffusion-canvas intermediate states can never leak
//! into API-visible output events.

use anyhow::anyhow;

use crate::core::native_output::{NativeOutputDecoderConfig, NativeOutputParser};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::tool_calling::{ToolCall, ToolCallParser, ToolDefinition, ToolDialect};
use crate::Result;

/// Whether a capability is backed by a native model/runtime contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupport {
    Unsupported,
    Native,
}

impl CapabilitySupport {
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Native)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputCapabilityProfile {
    pub text: CapabilitySupport,
    pub image: CapabilitySupport,
    pub audio: CapabilitySupport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputCapabilityProfile {
    pub text: CapabilitySupport,
    pub reasoning: CapabilitySupport,
    pub reasoning_summary: CapabilitySupport,
    pub refusal: CapabilitySupport,
    pub audio: CapabilitySupport,
    pub image: CapabilitySupport,
}

/// Capabilities derived from loaded components and exact template contracts.
///
/// Optional typed channels default to unsupported. Model-specific tasks must
/// explicitly opt in only after their native channel and round-trip semantics
/// have been implemented and validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilityProfile {
    pub input: InputCapabilityProfile,
    pub output: OutputCapabilityProfile,
    pub function_tools: CapabilitySupport,
    pub structured_output: CapabilitySupport,
}

impl ModelCapabilityProfile {
    pub fn from_loaded_contract(
        supports_image_input: bool,
        tool_dialect: Option<ToolDialect>,
        supports_structured_output: bool,
        supports_reasoning: bool,
    ) -> Self {
        Self {
            input: InputCapabilityProfile {
                text: CapabilitySupport::Native,
                image: if supports_image_input {
                    CapabilitySupport::Native
                } else {
                    CapabilitySupport::Unsupported
                },
                audio: CapabilitySupport::Unsupported,
            },
            output: OutputCapabilityProfile {
                text: CapabilitySupport::Native,
                reasoning: if supports_reasoning {
                    CapabilitySupport::Native
                } else {
                    CapabilitySupport::Unsupported
                },
                reasoning_summary: CapabilitySupport::Unsupported,
                refusal: CapabilitySupport::Unsupported,
                audio: CapabilitySupport::Unsupported,
                image: CapabilitySupport::Unsupported,
            },
            function_tools: if tool_dialect.is_some() {
                CapabilitySupport::Native
            } else {
                CapabilitySupport::Unsupported
            },
            structured_output: if supports_structured_output {
                CapabilitySupport::Native
            } else {
                CapabilitySupport::Unsupported
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageArtifact {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedFinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl GeneratedFinishReason {
    pub fn from_generation(reason: &'static str, saw_tool_call: bool) -> Result<Self> {
        if saw_tool_call {
            return Ok(Self::ToolCalls);
        }
        match reason {
            "stop" => Ok(Self::Stop),
            "length" => Ok(Self::Length),
            other => Err(anyhow!("unsupported generation finish reason `{other}`")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// Protocol-neutral output emitted after committed tokens are decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratedOutputEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ReasoningSummaryDelta(String),
    RefusalDelta(String),
    ToolCall(ToolCall),
    AudioDelta(AudioChunk),
    ImageOutput(ImageArtifact),
    Finished(GeneratedFinishReason),
}

impl GeneratedOutputEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::TextDelta(_) => "text",
            Self::ReasoningDelta(_) => "reasoning",
            Self::ReasoningSummaryDelta(_) => "reasoning_summary",
            Self::RefusalDelta(_) => "refusal",
            Self::ToolCall(_) => "tool_call",
            Self::AudioDelta(_) => "audio",
            Self::ImageOutput(_) => "image",
            Self::Finished(_) => "finished",
        }
    }
}

/// Request-local accumulation of every protocol-neutral output channel.
///
/// Protocol adapters may project this value into their own wire format, but
/// must explicitly reject channels that the protocol or endpoint cannot
/// represent. No typed output is silently folded into visible text.
#[derive(Debug, Default)]
pub struct CollectedGeneratedOutput {
    pub text: String,
    pub reasoning: String,
    pub reasoning_summary: String,
    pub refusal: String,
    pub tool_calls: Vec<ToolCall>,
    pub audio: Vec<AudioChunk>,
    pub images: Vec<ImageArtifact>,
    pub finish_reason: Option<GeneratedFinishReason>,
}

impl CollectedGeneratedOutput {
    pub fn push(&mut self, events: impl IntoIterator<Item = GeneratedOutputEvent>) -> Result<()> {
        for event in events {
            match event {
                GeneratedOutputEvent::TextDelta(delta) => self.text.push_str(&delta),
                GeneratedOutputEvent::ReasoningDelta(delta) => self.reasoning.push_str(&delta),
                GeneratedOutputEvent::ReasoningSummaryDelta(delta) => {
                    self.reasoning_summary.push_str(&delta);
                }
                GeneratedOutputEvent::RefusalDelta(delta) => self.refusal.push_str(&delta),
                GeneratedOutputEvent::ToolCall(call) => self.tool_calls.push(call),
                GeneratedOutputEvent::AudioDelta(chunk) => self.audio.push(chunk),
                GeneratedOutputEvent::ImageOutput(image) => self.images.push(image),
                GeneratedOutputEvent::Finished(reason) => {
                    anyhow::ensure!(
                        self.finish_reason.replace(reason).is_none(),
                        "generated output emitted more than one terminal event"
                    );
                }
            }
        }
        Ok(())
    }

    pub fn ensure_text_and_tools_only(&self, protocol: &str) -> Result<()> {
        anyhow::ensure!(
            self.reasoning.is_empty(),
            "{protocol} cannot represent reasoning output"
        );
        anyhow::ensure!(
            self.reasoning_summary.is_empty(),
            "{protocol} cannot represent reasoning summary output"
        );
        anyhow::ensure!(
            self.refusal.is_empty(),
            "{protocol} cannot represent refusal output"
        );
        anyhow::ensure!(
            self.audio.is_empty(),
            "{protocol} cannot represent audio output"
        );
        anyhow::ensure!(
            self.images.is_empty(),
            "{protocol} cannot represent image output"
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutputDecoderConfig {
    pub dialect: ToolDialect,
    pub response_id: String,
    pub definitions: Vec<ToolDefinition>,
    pub output_schema: Option<serde_json::Value>,
}

/// Converts committed token ids into protocol-neutral typed output events.
///
/// The decoder owns request-local detokenization and optional native tool
/// parsing. Future reasoning/refusal dialects plug in at this same boundary;
/// low-level generation events do not change.
pub struct GeneratedOutputDecoder<'a> {
    detok: Option<DecodeStream<'a>>,
    native_parser: Option<NativeOutputParser>,
    tool_parser: Option<ToolCallParser>,
    saw_tool_call: bool,
    last_token_was_reasoning: bool,
    finished: bool,
}

impl<'a> GeneratedOutputDecoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer, tool: Option<ToolOutputDecoderConfig>) -> Result<Self> {
        Self::new_with_native(tokenizer, tool, None)
    }

    pub fn new_with_native(
        tokenizer: &'a Tokenizer,
        tool: Option<ToolOutputDecoderConfig>,
        native: Option<NativeOutputDecoderConfig>,
    ) -> Result<Self> {
        let skip_special = tool
            .as_ref()
            .map(|config| config.dialect.skip_special_tokens())
            .unwrap_or(true)
            && native
                .map(|config| config.dialect.skip_special_tokens())
                .unwrap_or(true);
        let tool_parser = Self::build_tool_parser(tool)?;
        Ok(Self {
            detok: Some(tokenizer.decode_stream(skip_special)),
            native_parser: native.map(NativeOutputParser::new),
            tool_parser,
            saw_tool_call: false,
            last_token_was_reasoning: false,
            finished: false,
        })
    }

    /// Construct a decoder for an execution path that already exposes only
    /// committed, incrementally decoded text (for example block diffusion).
    pub fn from_decoded(tool: Option<ToolOutputDecoderConfig>) -> Result<Self> {
        Self::from_decoded_with_native(tool, None)
    }

    pub fn from_decoded_with_native(
        tool: Option<ToolOutputDecoderConfig>,
        native: Option<NativeOutputDecoderConfig>,
    ) -> Result<Self> {
        let tool_parser = Self::build_tool_parser(tool)?;
        Ok(Self {
            detok: None,
            native_parser: native.map(NativeOutputParser::new),
            tool_parser,
            saw_tool_call: false,
            last_token_was_reasoning: false,
            finished: false,
        })
    }

    fn build_tool_parser(tool: Option<ToolOutputDecoderConfig>) -> Result<Option<ToolCallParser>> {
        tool.map(|config| {
            ToolCallParser::new_with_output_schema(
                config.dialect,
                config.response_id,
                &config.definitions,
                config.output_schema,
            )
        })
        .transpose()
    }

    /// Decode one token that has already been committed by the execution path.
    pub fn push_token(&mut self, token: u32) -> Result<Vec<GeneratedOutputEvent>> {
        anyhow::ensure!(
            !self.finished,
            "generated output decoder is already finished"
        );
        let detok = self.detok.as_mut().ok_or_else(|| {
            anyhow!("generated output decoder was constructed for decoded text input")
        })?;
        let reasoning_before = self
            .native_parser
            .as_ref()
            .is_some_and(NativeOutputParser::is_reasoning);
        let Some(delta) = detok.step(token)? else {
            self.last_token_was_reasoning = reasoning_before;
            return Ok(Vec::new());
        };
        let events = self.push_text_delta(&delta)?;
        self.last_token_was_reasoning |= reasoning_before;
        Ok(events)
    }

    /// Feed already-decoded committed text. Primarily useful for deterministic
    /// parser tests and execution paths that own equivalent detokenization.
    pub fn push_text_delta(&mut self, delta: &str) -> Result<Vec<GeneratedOutputEvent>> {
        anyhow::ensure!(
            !self.finished,
            "generated output decoder is already finished"
        );
        if delta.is_empty() {
            self.last_token_was_reasoning = false;
            return Ok(Vec::new());
        }
        self.last_token_was_reasoning = false;
        let native_events = match self.native_parser.as_mut() {
            Some(parser) => parser.push(delta)?,
            None => vec![GeneratedOutputEvent::TextDelta(delta.to_owned())],
        };
        let events = self.route_native_events(native_events)?;
        self.last_token_was_reasoning |= self
            .native_parser
            .as_ref()
            .is_some_and(NativeOutputParser::is_reasoning)
            || events
                .iter()
                .any(|event| matches!(event, GeneratedOutputEvent::ReasoningDelta(_)));
        self.record_tool_calls(&events);
        Ok(events)
    }

    /// Flush parser state and emit one typed terminal event.
    pub fn finish(&mut self, reason: &'static str) -> Result<Vec<GeneratedOutputEvent>> {
        anyhow::ensure!(
            !self.finished,
            "generated output decoder is already finished"
        );
        self.finished = true;
        let native_events = match self.native_parser.take() {
            Some(parser) => parser.finish(reason)?,
            None => Vec::new(),
        };
        let mut events = self.route_native_events(native_events)?;
        events.extend(match self.tool_parser.take() {
            Some(parser) => {
                let (events, saw_tool_call) = parser.finish()?;
                self.saw_tool_call |= saw_tool_call;
                events
            }
            None => Vec::new(),
        });
        self.record_tool_calls(&events);
        events.push(GeneratedOutputEvent::Finished(
            GeneratedFinishReason::from_generation(reason, self.saw_tool_call)?,
        ));
        Ok(events)
    }

    pub fn saw_tool_call(&self) -> bool {
        self.saw_tool_call
    }

    pub fn last_token_was_reasoning(&self) -> bool {
        self.last_token_was_reasoning
    }

    fn record_tool_calls(&mut self, events: &[GeneratedOutputEvent]) {
        self.saw_tool_call |= events
            .iter()
            .any(|event| matches!(event, GeneratedOutputEvent::ToolCall(_)));
    }

    fn route_native_events(
        &mut self,
        native_events: Vec<GeneratedOutputEvent>,
    ) -> Result<Vec<GeneratedOutputEvent>> {
        let mut output = Vec::new();
        for event in native_events {
            match event {
                GeneratedOutputEvent::TextDelta(text) => match self.tool_parser.as_mut() {
                    Some(parser) => output.extend(parser.push(&text)?),
                    None => output.push(GeneratedOutputEvent::TextDelta(text)),
                },
                other => output.push(other),
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_owned(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city", "days"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    #[test]
    fn optional_output_capabilities_fail_closed() {
        let profile = ModelCapabilityProfile::from_loaded_contract(
            true,
            Some(ToolDialect::Qwen35),
            true,
            false,
        );
        assert!(profile.input.text.is_supported());
        assert!(profile.input.image.is_supported());
        assert!(!profile.input.audio.is_supported());
        assert!(profile.output.text.is_supported());
        assert!(!profile.output.reasoning.is_supported());
        assert!(!profile.output.reasoning_summary.is_supported());
        assert!(!profile.output.refusal.is_supported());
        assert!(!profile.output.audio.is_supported());
        assert!(!profile.output.image.is_supported());
        assert!(profile.function_tools.is_supported());
        assert!(profile.structured_output.is_supported());
    }

    #[test]
    fn finish_reason_becomes_tool_calls_when_a_call_was_seen() {
        assert_eq!(
            GeneratedFinishReason::from_generation("stop", true).unwrap(),
            GeneratedFinishReason::ToolCalls
        );
        assert_eq!(
            GeneratedFinishReason::from_generation("length", false).unwrap(),
            GeneratedFinishReason::Length
        );
        assert!(GeneratedFinishReason::from_generation("unknown", false).is_err());
    }

    #[test]
    fn decoded_text_emits_one_terminal_event_and_rejects_late_input() {
        let mut decoder = GeneratedOutputDecoder::from_decoded(None).unwrap();
        assert_eq!(
            decoder.push_text_delta("hello").unwrap(),
            vec![GeneratedOutputEvent::TextDelta("hello".to_owned())]
        );
        assert_eq!(
            decoder.finish("stop").unwrap(),
            vec![GeneratedOutputEvent::Finished(GeneratedFinishReason::Stop)]
        );
        assert!(decoder.push_text_delta("late").is_err());
        assert!(decoder.finish("stop").is_err());
    }

    #[test]
    fn decoded_qwen_call_is_typed_before_terminal_event() {
        let mut decoder = GeneratedOutputDecoder::from_decoded(Some(ToolOutputDecoderConfig {
            dialect: ToolDialect::Qwen35,
            response_id: "response".to_owned(),
            definitions: vec![weather_tool()],
            output_schema: None,
        }))
        .unwrap();
        let chunks = [
            "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>",
            "<parameter=days>2</parameter></function></tool_call>",
        ];
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.push_text_delta(chunk).unwrap());
        }
        events.extend(decoder.finish("stop").unwrap());
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_response_0".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: serde_json::json!({"city": "Tokyo", "days": 2}),
                }),
                GeneratedOutputEvent::Finished(GeneratedFinishReason::ToolCalls),
            ]
        );
    }

    #[test]
    fn collector_preserves_channels_and_rejects_duplicate_terminal_events() {
        let mut output = CollectedGeneratedOutput::default();
        output
            .push([
                GeneratedOutputEvent::TextDelta("answer".to_owned()),
                GeneratedOutputEvent::ReasoningDelta("reason".to_owned()),
                GeneratedOutputEvent::ReasoningSummaryDelta("summary".to_owned()),
                GeneratedOutputEvent::RefusalDelta("refusal".to_owned()),
                GeneratedOutputEvent::AudioDelta(AudioChunk {
                    data: vec![1, 2],
                    mime_type: "audio/pcm".to_owned(),
                    sample_rate_hz: Some(24_000),
                    channels: Some(1),
                }),
                GeneratedOutputEvent::ImageOutput(ImageArtifact {
                    data: vec![3, 4],
                    mime_type: "image/png".to_owned(),
                    width: Some(1),
                    height: Some(1),
                }),
                GeneratedOutputEvent::Finished(GeneratedFinishReason::Stop),
            ])
            .unwrap();
        assert_eq!(output.text, "answer");
        assert_eq!(output.reasoning, "reason");
        assert_eq!(output.reasoning_summary, "summary");
        assert_eq!(output.refusal, "refusal");
        assert_eq!(output.audio.len(), 1);
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.finish_reason, Some(GeneratedFinishReason::Stop));
        assert!(output.ensure_text_and_tools_only("test protocol").is_err());
        assert!(output
            .push([GeneratedOutputEvent::Finished(GeneratedFinishReason::Stop)])
            .is_err());
    }

    #[test]
    fn decoded_mode_rejects_token_input() {
        let mut decoder = GeneratedOutputDecoder::from_decoded(None).unwrap();
        assert!(decoder.push_token(1).is_err());
    }
}
