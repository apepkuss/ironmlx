//! Native model output-channel detection and incremental decoding.
//!
//! These parsers operate only on text decoded from committed tokens. They
//! split model-native reasoning channels from visible assistant text without
//! guessing from natural-language content.

use anyhow::bail;

use crate::core::generated_output::GeneratedOutputEvent;
use crate::Result;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const GEMMA_THOUGHT_OPEN: &str = "<|channel>thought\n";
const GEMMA_CHANNEL_CLOSE: &str = "<channel|>";
const MAX_PENDING_BYTES: usize = 1 << 20;

/// Exact native reasoning syntax exposed by a supported chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOutputDialect {
    Qwen35,
    Qwen36,
    Qwen38,
    Glm,
    MiniCpmV46,
    MiniCpm5,
    Gemma,
}

impl NativeOutputDialect {
    /// Detect a native output contract from the model type and exact template
    /// markers. Similar-looking templates are deliberately not enabled.
    pub fn detect(model_type: &str, template: &str) -> Option<Self> {
        if matches!(model_type, "qwen3_5" | "qwen3_5_moe") {
            let qwen38 = [
                "message.reasoning_content is string",
                "reasoning_effort|default('xhigh')",
                "resolved_reasoning_effort not in ('xhigh', 'medium', 'low')",
                "preserve_thinking is undefined or preserve_thinking is true",
                "'<think>\\n'",
                "'<think>\\n\\n</think>\\n\\n'",
            ];
            if qwen38.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Qwen38);
            }
            let common = [
                "message.reasoning_content is string",
                "content.split('</think>')",
                "'<think>\\n'",
                "'<think>\\n\\n</think>\\n\\n'",
            ];
            if common.iter().all(|needle| template.contains(needle)) {
                return Some(if template.contains("preserve_thinking is defined") {
                    Self::Qwen36
                } else {
                    Self::Qwen35
                });
            }
        }
        if model_type == "glm4_moe_lite" {
            let required = [
                "m.reasoning_content is string",
                "content.split('</think>')",
                "'<think>' + reasoning_content.strip()",
                "enable_thinking is defined and not enable_thinking",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Glm);
            }
        }
        if model_type == "minicpmv4_6" {
            let required = [
                "set enable_thinking = false",
                "message.reasoning_content is string",
                "content.split('</think>')",
                "'<think>\\n'",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::MiniCpmV46);
            }
        }
        if model_type == "llama" {
            let required = [
                "message.reasoning_content is string",
                "content.split('</think>')",
                "if enable_thinking is defined",
                "'<think>\\n'",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::MiniCpm5);
            }
        }
        if matches!(model_type, "gemma4" | "gemma4_unified" | "diffusion_gemma") {
            let required = [
                "message.get('reasoning') or message.get('reasoning_content')",
                "'<|channel>thought\\n'",
                "'<|think|>\\n'",
                "enable_thinking is defined and enable_thinking",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Gemma);
            }
        }
        None
    }

    /// Whether the template opens its native reasoning channel when callers
    /// do not explicitly supply `enable_thinking`.
    pub fn default_reasoning_enabled(self) -> bool {
        matches!(self, Self::Qwen36 | Self::Qwen38 | Self::Glm)
    }

    /// Resolve the template's effective reasoning mode from request kwargs.
    pub fn reasoning_enabled(self, kwargs: Option<&serde_json::Value>) -> Result<bool> {
        let Some(kwargs) = kwargs else {
            return Ok(self.default_reasoning_enabled());
        };
        let object = kwargs
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("chat_template_kwargs must be a JSON object"))?;
        match object.get("enable_thinking") {
            None | Some(serde_json::Value::Null) => Ok(self.default_reasoning_enabled()),
            Some(serde_json::Value::Bool(enabled)) => Ok(*enabled),
            Some(_) => bail!("chat_template_kwargs.enable_thinking must be a boolean"),
        }
    }

    /// Gemma channel delimiters and GLM's `<think>` delimiters are special
    /// tokens and must remain visible to the parser. The remaining tagged
    /// reasoning dialects encode their markers as ordinary text.
    pub fn skip_special_tokens(self) -> bool {
        !matches!(self, Self::Gemma | Self::Glm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOutputDecoderConfig {
    pub dialect: NativeOutputDialect,
    pub reasoning_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelState {
    Text,
    Reasoning,
}

/// Incrementally splits native reasoning from visible text.
pub struct NativeOutputParser {
    dialect: NativeOutputDialect,
    state: ChannelState,
    pending: String,
    trim_reasoning_prefix: bool,
    trim_text_prefix: bool,
    saw_reasoning: bool,
}

impl NativeOutputParser {
    pub fn new(config: NativeOutputDecoderConfig) -> Self {
        let state =
            if config.reasoning_enabled && !matches!(config.dialect, NativeOutputDialect::Gemma) {
                ChannelState::Reasoning
            } else {
                ChannelState::Text
            };
        Self {
            dialect: config.dialect,
            state,
            pending: String::new(),
            trim_reasoning_prefix: false,
            trim_text_prefix: false,
            saw_reasoning: matches!(state, ChannelState::Reasoning),
        }
    }

    pub fn push(&mut self, delta: &str) -> Result<Vec<GeneratedOutputEvent>> {
        self.pending.push_str(delta);
        anyhow::ensure!(
            self.pending.len() <= MAX_PENDING_BYTES,
            "native output parser pending buffer exceeds {MAX_PENDING_BYTES} bytes"
        );
        self.advance(false)
    }

    pub fn finish(mut self, finish_reason: &'static str) -> Result<Vec<GeneratedOutputEvent>> {
        let events = self.advance(true)?;
        if matches!(self.state, ChannelState::Reasoning) && finish_reason != "length" {
            bail!("unterminated native reasoning channel")
        }
        Ok(events)
    }

    fn advance(&mut self, eof: bool) -> Result<Vec<GeneratedOutputEvent>> {
        match self.dialect {
            NativeOutputDialect::Gemma => self.advance_gemma(eof),
            _ => self.advance_tagged(eof),
        }
    }

    fn advance_tagged(&mut self, eof: bool) -> Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            let marker = match self.state {
                ChannelState::Text => THINK_OPEN,
                ChannelState::Reasoning => THINK_CLOSE,
            };
            if let Some(position) = self.pending.find(marker) {
                let prefix = self.pending[..position].to_owned();
                self.emit_current(&mut events, prefix);
                self.pending.drain(..position + marker.len());
                match self.state {
                    ChannelState::Text => {
                        self.state = ChannelState::Reasoning;
                        self.saw_reasoning = true;
                        self.trim_reasoning_prefix = true;
                    }
                    ChannelState::Reasoning => {
                        self.state = ChannelState::Text;
                        self.trim_text_prefix = true;
                    }
                }
                continue;
            }
            if eof {
                let pending = std::mem::take(&mut self.pending);
                self.emit_current(&mut events, pending);
            } else {
                let keep = partial_marker_suffix_len(&self.pending, marker);
                let emit_len = self.pending.len().saturating_sub(keep);
                if emit_len > 0 {
                    let prefix = self.pending[..emit_len].to_owned();
                    self.pending.drain(..emit_len);
                    self.emit_current(&mut events, prefix);
                }
            }
            break;
        }
        Ok(events)
    }

    fn advance_gemma(&mut self, eof: bool) -> Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            let marker = match self.state {
                ChannelState::Text => GEMMA_THOUGHT_OPEN,
                ChannelState::Reasoning => GEMMA_CHANNEL_CLOSE,
            };
            if let Some(position) = self.pending.find(marker) {
                let prefix = self.pending[..position].to_owned();
                self.emit_current(&mut events, prefix);
                self.pending.drain(..position + marker.len());
                match self.state {
                    ChannelState::Text => {
                        self.state = ChannelState::Reasoning;
                        self.saw_reasoning = true;
                    }
                    ChannelState::Reasoning => {
                        self.state = ChannelState::Text;
                        self.trim_text_prefix = true;
                    }
                }
                continue;
            }
            if eof {
                let pending = std::mem::take(&mut self.pending);
                self.emit_current(&mut events, pending);
            } else {
                let keep = partial_marker_suffix_len(&self.pending, marker);
                let emit_len = self.pending.len().saturating_sub(keep);
                if emit_len > 0 {
                    let prefix = self.pending[..emit_len].to_owned();
                    self.pending.drain(..emit_len);
                    self.emit_current(&mut events, prefix);
                }
            }
            break;
        }
        Ok(events)
    }

    fn emit_current(&mut self, events: &mut Vec<GeneratedOutputEvent>, mut value: String) {
        if matches!(self.dialect, NativeOutputDialect::Glm) {
            value = sanitize_glm_controls(&value);
        }
        match self.state {
            ChannelState::Text => {
                if matches!(self.dialect, NativeOutputDialect::Gemma) {
                    value = sanitize_gemma_controls(&value);
                }
                if self.trim_text_prefix {
                    value = value.trim_start_matches(['\r', '\n']).to_owned();
                    if !value.is_empty() {
                        self.trim_text_prefix = false;
                    }
                }
                if !value.is_empty() {
                    events.push(GeneratedOutputEvent::TextDelta(value));
                }
            }
            ChannelState::Reasoning => {
                if self.trim_reasoning_prefix {
                    value = value.trim_start_matches(['\r', '\n']).to_owned();
                    if !value.is_empty() {
                        self.trim_reasoning_prefix = false;
                    }
                }
                if !value.is_empty() {
                    events.push(GeneratedOutputEvent::ReasoningDelta(value));
                }
            }
        }
    }

    pub fn saw_reasoning(&self) -> bool {
        self.saw_reasoning
    }

    pub fn is_reasoning(&self) -> bool {
        matches!(self.state, ChannelState::Reasoning)
    }
}

fn sanitize_glm_controls(value: &str) -> String {
    [
        "<|endoftext|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<sop>",
        "<eop>",
    ]
    .into_iter()
    .fold(value.to_owned(), |text, marker| text.replace(marker, ""))
}

fn sanitize_gemma_controls(value: &str) -> String {
    [
        "<|turn>model\n",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<eos>",
        "<bos>",
    ]
    .into_iter()
    .fold(value.to_owned(), |text, marker| text.replace(marker, ""))
}

fn partial_marker_suffix_len(value: &str, marker: &str) -> usize {
    let max = value.len().min(marker.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|length| {
            value.is_char_boundary(value.len() - length)
                && marker.starts_with(&value[value.len() - length..])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exact_reasoning_contracts() {
        let qwen35 = "message.reasoning_content is string content.split('</think>') '<think>\\n' '<think>\\n\\n</think>\\n\\n'";
        let qwen36 = format!("{qwen35} preserve_thinking is defined");
        let qwen38 = "message.reasoning_content is string reasoning_effort|default('xhigh') resolved_reasoning_effort not in ('xhigh', 'medium', 'low') preserve_thinking is undefined or preserve_thinking is true '<think>\\n' '<think>\\n\\n</think>\\n\\n'";
        assert_eq!(
            NativeOutputDialect::detect("qwen3_5", qwen35),
            Some(NativeOutputDialect::Qwen35)
        );
        assert_eq!(
            NativeOutputDialect::detect("qwen3_5", &qwen36),
            Some(NativeOutputDialect::Qwen36)
        );
        assert_eq!(
            NativeOutputDialect::detect("qwen3_5", qwen38),
            Some(NativeOutputDialect::Qwen38)
        );
        assert_eq!(NativeOutputDialect::detect("llama", qwen35), None);
    }

    #[test]
    fn qwen38_reasoning_is_enabled_by_default_and_honors_explicit_disable() {
        let dialect = NativeOutputDialect::Qwen38;
        assert!(dialect.reasoning_enabled(None).unwrap());
        assert!(dialect
            .reasoning_enabled(Some(&serde_json::json!({"reasoning_effort": "low"})))
            .unwrap());
        assert!(!dialect
            .reasoning_enabled(Some(&serde_json::json!({"enable_thinking": false})))
            .unwrap());
    }

    #[test]
    fn tagged_reasoning_is_split_across_fragmented_markers() {
        let mut parser = NativeOutputParser::new(NativeOutputDecoderConfig {
            dialect: NativeOutputDialect::Qwen35,
            reasoning_enabled: true,
        });
        let mut events = Vec::new();
        for chunk in ["first", " step</thi", "nk>\n\nfinal"] {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish("stop").unwrap());
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ReasoningDelta("first".into()),
                GeneratedOutputEvent::ReasoningDelta(" step".into()),
                GeneratedOutputEvent::TextDelta("final".into()),
            ]
        );
    }

    #[test]
    fn glm_terminal_role_token_is_not_visible_text() {
        let mut parser = NativeOutputParser::new(NativeOutputDecoderConfig {
            dialect: NativeOutputDialect::Glm,
            reasoning_enabled: true,
        });
        let mut events = parser
            .push("brief</think>\n\nTokyo is clear.<|user|>")
            .unwrap();
        events.extend(parser.finish("stop").unwrap());
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ReasoningDelta("brief".to_owned()),
                GeneratedOutputEvent::TextDelta("Tokyo is clear.".to_owned()),
            ]
        );
    }

    #[test]
    fn explicit_think_open_is_recognized_when_prompt_did_not_open_it() {
        let mut parser = NativeOutputParser::new(NativeOutputDecoderConfig {
            dialect: NativeOutputDialect::MiniCpm5,
            reasoning_enabled: false,
        });
        let events = parser.push("<think>\nplan</think>\nanswer").unwrap();
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ReasoningDelta("plan".into()),
                GeneratedOutputEvent::TextDelta("answer".into()),
            ]
        );
        assert!(parser.finish("stop").unwrap().is_empty());
    }

    #[test]
    fn gemma_thought_channel_is_split_and_controls_are_removed() {
        let mut parser = NativeOutputParser::new(NativeOutputDecoderConfig {
            dialect: NativeOutputDialect::Gemma,
            reasoning_enabled: true,
        });
        let mut events = Vec::new();
        for chunk in [
            "<|turn>model\n<|channel>th",
            "ought\ncheck weather<channel|>",
            "<|tool_call>call:get_weather{}<tool_call|>",
        ] {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish("stop").unwrap());
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ReasoningDelta("check weather".into()),
                GeneratedOutputEvent::TextDelta(
                    "<|tool_call>call:get_weather{}<tool_call|>".into()
                ),
            ]
        );
    }

    #[test]
    fn unclosed_reasoning_is_only_valid_for_length_termination() {
        let config = NativeOutputDecoderConfig {
            dialect: NativeOutputDialect::Glm,
            reasoning_enabled: true,
        };
        let mut stopped = NativeOutputParser::new(config);
        assert_eq!(
            stopped.push("unfinished").unwrap(),
            vec![GeneratedOutputEvent::ReasoningDelta("unfinished".into())]
        );
        assert!(stopped.finish("stop").is_err());

        let mut limited = NativeOutputParser::new(config);
        assert_eq!(
            limited.push("unfinished").unwrap(),
            vec![GeneratedOutputEvent::ReasoningDelta("unfinished".into())]
        );
        assert!(limited.finish("length").unwrap().is_empty());
    }
}
