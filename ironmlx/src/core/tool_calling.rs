//! Protocol-neutral tool definitions, conversation messages, and native
//! model-output parsing.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::constrained::{
    schema_accepts_string, schema_is_string_only, validate_schema_value,
    validate_strict_tool_schema, validate_tool_schemas,
};
use crate::core::generated_output::GeneratedOutputEvent;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const GLM_ARGUMENT_KEY_OPEN: &str = "<arg_key>";
const GLM_ARGUMENT_KEY_CLOSE: &str = "</arg_key>";
const GLM_ARGUMENT_VALUE_OPEN: &str = "<arg_value>";
const GLM_ARGUMENT_VALUE_CLOSE: &str = "</arg_value>";
const GLM_THINK_OPEN: &str = "<think>";
const GLM_THINK_CLOSE: &str = "</think>";
const GEMMA_TOOL_CALL_OPEN: &str = "<|tool_call>";
const GEMMA_TOOL_CALL_CLOSE: &str = "<tool_call|>";
pub(crate) const GEMMA_STRING_DELIMITER: &str = "<|\"|>";
const MINICPM5_FUNCTION_OPEN: &str = "<function name=\"";
const MINICPM5_NAME_CLOSE: &str = "\">";
const MINICPM5_PARAM_OPEN: &str = "<param name=\"";
const MINICPM5_PARAM_CLOSE: &str = "</param>";
const MINICPM5_CDATA_OPEN: &str = "<![CDATA[";
const MINICPM5_CDATA_PARAM_CLOSE: &str = "]]></param>";
const MAX_PENDING_BYTES: usize = 1 << 20;
const MAX_ARGUMENT_BYTES: usize = 1 << 20;

/// Native tool syntax supported by a model chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDialect {
    Qwen35,
    Gemma,
    Glm,
    Llama,
    MiniCpmV46,
    MiniCpm5,
}

impl ToolDialect {
    pub(crate) fn ordinary_constraint_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Qwen35 | Self::Llama | Self::MiniCpmV46 => &[],
            Self::MiniCpm5 => &["<function", "</function>", "<param", "</param>"],
            Self::Glm => &[
                TOOL_CALL_OPEN,
                TOOL_CALL_CLOSE,
                GLM_ARGUMENT_KEY_OPEN,
                GLM_ARGUMENT_KEY_CLOSE,
                GLM_ARGUMENT_VALUE_OPEN,
                GLM_ARGUMENT_VALUE_CLOSE,
                GLM_THINK_OPEN,
                GLM_THINK_CLOSE,
            ],
            Self::Gemma => &[
                GEMMA_TOOL_CALL_OPEN,
                GEMMA_TOOL_CALL_CLOSE,
                GEMMA_STRING_DELIMITER,
                "<|channel>",
                "<channel|>",
            ],
        }
    }

    /// Recognize exact native template contracts. Similar-looking templates
    /// are deliberately not guessed into support.
    pub fn detect(model_type: &str, template: &str) -> Option<Self> {
        if matches!(model_type, "qwen3_5" | "qwen3_5_moe") {
            let required = [
                "<tool_call>",
                "<function=",
                "<parameter=",
                "<tool_response>",
                "tool_call.function is defined",
                "tool_call.arguments",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Qwen35);
            }
        }
        if matches!(model_type, "gemma4" | "gemma4_unified" | "diffusion_gemma") {
            let required = [
                "<|tool>",
                "declaration:",
                "<|tool_call>call:",
                "<tool_call|>",
                "<|tool_response>",
                "response:",
                "tool_call_id",
                GEMMA_STRING_DELIMITER,
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Gemma);
            }
        }
        if model_type == "glm4_moe_lite" {
            let required = [
                "# Tools",
                "<tools>",
                "tool | tojson(ensure_ascii=False)",
                "<tool_call>{function-name}<arg_key>",
                "'<tool_call>' + tc.name",
                "<arg_key>{{ k }}</arg_key><arg_value>",
                "<|observation|>",
                "<tool_response>",
                "m.tool_calls",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Glm);
            }
        }
        if model_type == "llama" {
            let required = [
                "custom_tools is defined",
                "tools_in_user_message",
                "Environment: ipython",
                r#"Respond in the format {"name": function name, "parameters": dictionary"#,
                "message.tool_calls|length == 1",
                "tool_call.arguments | tojson",
                "<|start_header_id|>ipython<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|eot_id|>",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::Llama);
            }

            let minicpm5_required = [
                "You are provided with function signatures within <tools></tools> XML tags",
                "<function name=\"function-name\"><param name=\"param-name\">",
                "wrap it in a CDATA block",
                "'<function name=\"' ~ tool_call.name ~ '\">'",
                "'<param name=\"' ~ param_name ~ '\">'",
                "tool_call.arguments",
                "<tool_response>",
                "<tool_def_sep>",
            ];
            if minicpm5_required
                .iter()
                .all(|needle| template.contains(needle))
            {
                return Some(Self::MiniCpm5);
            }
        }
        if model_type == "minicpmv4_6" {
            let required = [
                "If you choose to call a function ONLY reply in the following format",
                "<tool_call>\\n<function=example_function_name>",
                "<parameter=example_parameter_1>",
                "tool_call.arguments|items",
                "<tool_response>",
                "render_content(message.content)",
                "<|image_pad|>",
                "No user query found in messages.",
            ];
            if required.iter().all(|needle| template.contains(needle)) {
                return Some(Self::MiniCpmV46);
            }
        }
        None
    }

    /// Gemma, GLM, and MiniCPM5 tool delimiters are special tokens and must
    /// remain visible to their parsers. Qwen and MiniCPM-V markers are ordinary
    /// text.
    pub fn skip_special_tokens(self) -> bool {
        !matches!(self, Self::Gemma | Self::Glm | Self::MiniCpm5)
    }
}

/// Protocol-neutral function tool definition.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A model-requested function invocation. `arguments` is structured internally;
/// protocol adapters decide whether it is exposed as JSON text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Conversation message accepted by native tool-aware chat templates.
#[derive(Debug, Clone, Serialize)]
pub struct AgentMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<TemplateToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Template-facing tool call shape used by Hugging Face chat templates.
#[derive(Debug, Clone, Serialize)]
pub struct TemplateToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: TemplateFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateFunctionCall {
    pub name: String,
    pub arguments: Value,
}

impl From<ToolCall> for TemplateToolCall {
    fn from(call: ToolCall) -> Self {
        Self {
            id: call.id,
            kind: "function",
            function: TemplateFunctionCall {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

/// Dialect-dispatched incremental parser used by protocol adapters.
pub enum ToolCallParser {
    Qwen(QwenToolCallParser),
    Gemma(GemmaToolCallParser),
    Glm(GlmToolCallParser),
    Llama(LlamaToolCallParser),
    MiniCpmV46(QwenToolCallParser),
    MiniCpm5(MiniCpm5ToolCallParser),
}

impl ToolCallParser {
    pub fn new(
        dialect: ToolDialect,
        response_id: impl Into<String>,
        tools: &[ToolDefinition],
    ) -> anyhow::Result<Self> {
        Self::new_with_output_schema(dialect, response_id, tools, None)
    }

    pub fn new_with_output_schema(
        dialect: ToolDialect,
        response_id: impl Into<String>,
        tools: &[ToolDefinition],
        output_schema: Option<Value>,
    ) -> anyhow::Result<Self> {
        let response_id = response_id.into();
        match dialect {
            ToolDialect::Qwen35 => Ok(Self::Qwen(QwenToolCallParser::new(response_id, tools)?)),
            ToolDialect::Gemma => Ok(Self::Gemma(GemmaToolCallParser::new(response_id, tools)?)),
            ToolDialect::Glm => Ok(Self::Glm(GlmToolCallParser::new(response_id, tools)?)),
            ToolDialect::Llama => Ok(Self::Llama(LlamaToolCallParser::new_with_output_schema(
                response_id,
                tools,
                output_schema,
            )?)),
            ToolDialect::MiniCpmV46 => Ok(Self::MiniCpmV46(QwenToolCallParser::new(
                response_id,
                tools,
            )?)),
            ToolDialect::MiniCpm5 => Ok(Self::MiniCpm5(MiniCpm5ToolCallParser::new(
                response_id,
                tools,
            )?)),
        }
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        match self {
            Self::Qwen(parser) => parser.push(delta),
            Self::Gemma(parser) => parser.push(delta),
            Self::Glm(parser) => parser.push(delta),
            Self::Llama(parser) => parser.push(delta),
            Self::MiniCpmV46(parser) => parser.push(delta),
            Self::MiniCpm5(parser) => parser.push(delta),
        }
    }

    pub fn finish(self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        match self {
            Self::Qwen(parser) => parser.finish(),
            Self::Gemma(parser) => parser.finish(),
            Self::Glm(parser) => parser.finish(),
            Self::Llama(parser) => parser.finish(),
            Self::MiniCpmV46(parser) => parser.finish(),
            Self::MiniCpm5(parser) => parser.finish(),
        }
    }
}

#[derive(Debug)]
enum ParseState {
    Text,
    FunctionStart,
    FunctionBody {
        name: String,
        arguments: Map<String, Value>,
    },
    ParameterValue {
        function_name: String,
        arguments: Map<String, Value>,
        parameter_name: String,
        value: String,
    },
    ToolCallClose {
        name: String,
        arguments: Map<String, Value>,
    },
}

/// Strict bounded incremental parser for the Qwen3.5/Qwen3.6 native tool
/// syntax. It never treats malformed native syntax as ordinary assistant text.
pub struct QwenToolCallParser {
    definitions: HashMap<String, ToolDefinition>,
    response_id: String,
    pending: String,
    state: ParseState,
    next_index: usize,
    saw_tool_call: bool,
}

impl QwenToolCallParser {
    pub fn new(response_id: impl Into<String>, tools: &[ToolDefinition]) -> anyhow::Result<Self> {
        validate_tool_definitions(tools)?;
        let definitions = tools
            .iter()
            .cloned()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Ok(Self {
            definitions,
            response_id: response_id.into(),
            pending: String::new(),
            state: ParseState::Text,
            next_index: 0,
            saw_tool_call: false,
        })
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        self.pending.push_str(delta);
        if self.pending.len() > MAX_PENDING_BYTES {
            bail!("tool-call parser pending buffer exceeds {MAX_PENDING_BYTES} bytes");
        }
        self.advance(false)
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        let events = self.advance(true)?;
        if !matches!(self.state, ParseState::Text) {
            bail!("incomplete native tool-call output");
        }
        if self.saw_tool_call && !self.pending.trim().is_empty() {
            bail!("unexpected text after native tool call");
        }
        Ok((events, self.saw_tool_call))
    }

    fn advance(&mut self, eof: bool) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            let state = std::mem::replace(&mut self.state, ParseState::Text);
            match state {
                ParseState::Text => {
                    if let Some(pos) = self.pending.find(TOOL_CALL_OPEN) {
                        let text = self.pending[..pos].to_owned();
                        self.pending.drain(..pos + TOOL_CALL_OPEN.len());
                        if self.saw_tool_call && !text.trim().is_empty() {
                            bail!("unexpected text between native tool calls");
                        }
                        if !text.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(text));
                        }
                        self.saw_tool_call = true;
                        self.state = ParseState::FunctionStart;
                        continue;
                    }
                    if eof {
                        if self.saw_tool_call && !self.pending.trim().is_empty() {
                            bail!("unexpected text after native tool call");
                        }
                        if !self.pending.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(std::mem::take(
                                &mut self.pending,
                            )));
                        } else {
                            self.pending.clear();
                        }
                    } else if self.saw_tool_call {
                        let candidate = self
                            .pending
                            .trim_start_matches(|character: char| character.is_whitespace());
                        if !candidate.is_empty() && !TOOL_CALL_OPEN.starts_with(candidate) {
                            bail!("unexpected text after native tool call");
                        }
                    } else {
                        let keep = partial_marker_suffix_len(&self.pending, TOOL_CALL_OPEN);
                        let emit_len = self.pending.len().saturating_sub(keep);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_owned();
                            self.pending.drain(..emit_len);
                            events.push(GeneratedOutputEvent::TextDelta(text));
                        }
                    }
                    self.state = ParseState::Text;
                    break;
                }
                ParseState::FunctionStart => {
                    trim_native_whitespace(&mut self.pending);
                    let Some(rest) = self.pending.strip_prefix(FUNCTION_OPEN) else {
                        if !eof && FUNCTION_OPEN.starts_with(&self.pending) {
                            self.state = ParseState::FunctionStart;
                            break;
                        }
                        bail!("expected `{FUNCTION_OPEN}` after `{TOOL_CALL_OPEN}`");
                    };
                    let Some(end) = rest.find('>') else {
                        self.state = ParseState::FunctionStart;
                        if eof {
                            bail!("unterminated native function name");
                        }
                        break;
                    };
                    let name = rest[..end].to_owned();
                    validate_function_name(&name)?;
                    if !self.definitions.contains_key(&name) {
                        bail!("model requested unknown tool `{name}`");
                    }
                    self.pending.drain(..FUNCTION_OPEN.len() + end + 1);
                    self.state = ParseState::FunctionBody {
                        name,
                        arguments: Map::new(),
                    };
                }
                ParseState::FunctionBody {
                    name,
                    mut arguments,
                } => {
                    trim_native_whitespace(&mut self.pending);
                    if self.pending.starts_with(FUNCTION_CLOSE) {
                        self.pending.drain(..FUNCTION_CLOSE.len());
                        validate_schema_value(
                            &self.definitions[&name].parameters,
                            &Value::Object(arguments.clone()),
                        )
                        .with_context(|| {
                            format!("tool `{name}` arguments do not match its schema")
                        })?;
                        self.state = ParseState::ToolCallClose { name, arguments };
                        continue;
                    }
                    let Some(rest) = self.pending.strip_prefix(PARAMETER_OPEN) else {
                        if !eof
                            && (PARAMETER_OPEN.starts_with(&self.pending)
                                || FUNCTION_CLOSE.starts_with(&self.pending))
                        {
                            self.state = ParseState::FunctionBody { name, arguments };
                            break;
                        }
                        bail!("expected parameter or function close for tool `{name}`");
                    };
                    let Some(end) = rest.find('>') else {
                        if eof {
                            bail!("unterminated parameter name for tool `{name}`");
                        }
                        self.state = ParseState::FunctionBody { name, arguments };
                        break;
                    };
                    let parameter_name = rest[..end].to_owned();
                    validate_parameter_name(&parameter_name)?;
                    if arguments.contains_key(&parameter_name) {
                        bail!("duplicate parameter `{parameter_name}` for tool `{name}`");
                    }
                    self.pending.drain(..PARAMETER_OPEN.len() + end + 1);
                    self.state = ParseState::ParameterValue {
                        function_name: name,
                        arguments: std::mem::take(&mut arguments),
                        parameter_name,
                        value: String::new(),
                    };
                }
                ParseState::ParameterValue {
                    function_name,
                    mut arguments,
                    parameter_name,
                    mut value,
                } => {
                    if let Some(pos) = self.pending.find(PARAMETER_CLOSE) {
                        value.push_str(&self.pending[..pos]);
                        if value.len() > MAX_ARGUMENT_BYTES {
                            bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                        }
                        self.pending.drain(..pos + PARAMETER_CLOSE.len());
                        let value = parse_argument(
                            &self.definitions[&function_name],
                            &parameter_name,
                            strip_native_value_newlines(&value),
                        )?;
                        arguments.insert(parameter_name, value);
                        self.state = ParseState::FunctionBody {
                            name: function_name,
                            arguments,
                        };
                        continue;
                    }
                    if eof {
                        bail!("unterminated parameter value for tool `{function_name}`");
                    }
                    let keep = partial_marker_suffix_len(&self.pending, PARAMETER_CLOSE);
                    let consume = self.pending.len().saturating_sub(keep);
                    value.push_str(&self.pending[..consume]);
                    if value.len() > MAX_ARGUMENT_BYTES {
                        bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                    }
                    self.pending.drain(..consume);
                    self.state = ParseState::ParameterValue {
                        function_name,
                        arguments,
                        parameter_name,
                        value,
                    };
                    break;
                }
                ParseState::ToolCallClose { name, arguments } => {
                    trim_native_whitespace(&mut self.pending);
                    if !self.pending.starts_with(TOOL_CALL_CLOSE) {
                        if !eof && TOOL_CALL_CLOSE.starts_with(&self.pending) {
                            self.state = ParseState::ToolCallClose { name, arguments };
                            break;
                        }
                        bail!("expected `{TOOL_CALL_CLOSE}` for tool `{name}`");
                    }
                    self.pending.drain(..TOOL_CALL_CLOSE.len());
                    let call = ToolCall {
                        id: format!("call_{}_{}", self.response_id, self.next_index),
                        name,
                        arguments: Value::Object(arguments),
                    };
                    self.next_index += 1;
                    events.push(GeneratedOutputEvent::ToolCall(call));
                    self.state = ParseState::Text;
                }
            }
        }
        Ok(events)
    }
}

#[derive(Debug)]
enum MiniCpm5ValueMode {
    Undecided,
    Plain,
    Cdata,
}

#[derive(Debug)]
enum MiniCpm5ParseState {
    Text,
    FunctionName,
    FunctionBody {
        name: String,
        arguments: Map<String, Value>,
    },
    ParameterValue {
        function_name: String,
        arguments: Map<String, Value>,
        parameter_name: String,
        value: String,
        mode: MiniCpm5ValueMode,
    },
}

/// Strict bounded incremental parser for MiniCPM5's native XML function
/// protocol. String values containing XML-significant characters or newlines
/// must use the checkpoint template's CDATA representation.
pub struct MiniCpm5ToolCallParser {
    definitions: HashMap<String, ToolDefinition>,
    response_id: String,
    pending: String,
    state: MiniCpm5ParseState,
    next_index: usize,
    saw_tool_call: bool,
}

impl MiniCpm5ToolCallParser {
    pub fn new(response_id: impl Into<String>, tools: &[ToolDefinition]) -> anyhow::Result<Self> {
        validate_tool_definitions(tools)?;
        let definitions = tools
            .iter()
            .cloned()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Ok(Self {
            definitions,
            response_id: response_id.into(),
            pending: String::new(),
            state: MiniCpm5ParseState::Text,
            next_index: 0,
            saw_tool_call: false,
        })
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        self.pending.push_str(delta);
        if self.pending.len() > MAX_PENDING_BYTES {
            bail!("tool-call parser pending buffer exceeds {MAX_PENDING_BYTES} bytes");
        }
        self.advance(false)
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        let events = self.advance(true)?;
        if !matches!(self.state, MiniCpm5ParseState::Text) {
            bail!("incomplete native MiniCPM5 tool-call output");
        }
        if self.saw_tool_call && !self.pending.trim().is_empty() {
            bail!("unexpected text after native MiniCPM5 tool call");
        }
        Ok((events, self.saw_tool_call))
    }

    fn advance(&mut self, eof: bool) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            let state = std::mem::replace(&mut self.state, MiniCpm5ParseState::Text);
            match state {
                MiniCpm5ParseState::Text => {
                    if let Some(pos) = self.pending.find(MINICPM5_FUNCTION_OPEN) {
                        let text = self.pending[..pos].to_owned();
                        self.pending.drain(..pos + MINICPM5_FUNCTION_OPEN.len());
                        if self.saw_tool_call && !text.trim().is_empty() {
                            bail!("unexpected text between native MiniCPM5 tool calls");
                        }
                        if !text.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(text));
                        }
                        self.saw_tool_call = true;
                        self.state = MiniCpm5ParseState::FunctionName;
                        continue;
                    }
                    if eof {
                        if self.saw_tool_call && !self.pending.trim().is_empty() {
                            bail!("unexpected text after native MiniCPM5 tool call");
                        }
                        if !self.pending.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(std::mem::take(
                                &mut self.pending,
                            )));
                        } else {
                            self.pending.clear();
                        }
                    } else {
                        let keep = partial_marker_suffix_len(&self.pending, MINICPM5_FUNCTION_OPEN);
                        let emit_len = self.pending.len().saturating_sub(keep);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_owned();
                            self.pending.drain(..emit_len);
                            if self.saw_tool_call {
                                if !text.trim().is_empty() {
                                    bail!("unexpected text after native MiniCPM5 tool call");
                                }
                            } else {
                                events.push(GeneratedOutputEvent::TextDelta(text));
                            }
                        }
                    }
                    self.state = MiniCpm5ParseState::Text;
                    break;
                }
                MiniCpm5ParseState::FunctionName => {
                    let Some(end) = self.pending.find(MINICPM5_NAME_CLOSE) else {
                        self.state = MiniCpm5ParseState::FunctionName;
                        if eof {
                            bail!("unterminated native MiniCPM5 function name");
                        }
                        break;
                    };
                    let name = self.pending[..end].to_owned();
                    validate_function_name(&name)?;
                    if !self.definitions.contains_key(&name) {
                        bail!("model requested unknown MiniCPM5 tool `{name}`");
                    }
                    self.pending.drain(..end + MINICPM5_NAME_CLOSE.len());
                    self.state = MiniCpm5ParseState::FunctionBody {
                        name,
                        arguments: Map::new(),
                    };
                }
                MiniCpm5ParseState::FunctionBody {
                    name,
                    mut arguments,
                } => {
                    trim_native_whitespace(&mut self.pending);
                    if self.pending.starts_with(FUNCTION_CLOSE) {
                        self.pending.drain(..FUNCTION_CLOSE.len());
                        validate_schema_value(
                            &self.definitions[&name].parameters,
                            &Value::Object(arguments.clone()),
                        )
                        .with_context(|| {
                            format!("tool `{name}` arguments do not match its schema")
                        })?;
                        events.push(GeneratedOutputEvent::ToolCall(ToolCall {
                            id: format!("call_{}_{}", self.response_id, self.next_index),
                            name,
                            arguments: Value::Object(arguments),
                        }));
                        self.next_index += 1;
                        self.state = MiniCpm5ParseState::Text;
                        continue;
                    }
                    let Some(rest) = self.pending.strip_prefix(MINICPM5_PARAM_OPEN) else {
                        if !eof
                            && (MINICPM5_PARAM_OPEN.starts_with(&self.pending)
                                || FUNCTION_CLOSE.starts_with(&self.pending))
                        {
                            self.state = MiniCpm5ParseState::FunctionBody { name, arguments };
                            break;
                        }
                        bail!("expected parameter or function close for MiniCPM5 tool `{name}`");
                    };
                    let Some(end) = rest.find(MINICPM5_NAME_CLOSE) else {
                        if eof {
                            bail!("unterminated parameter name for MiniCPM5 tool `{name}`");
                        }
                        self.state = MiniCpm5ParseState::FunctionBody { name, arguments };
                        break;
                    };
                    let parameter_name = rest[..end].to_owned();
                    validate_parameter_name(&parameter_name)?;
                    if arguments.contains_key(&parameter_name) {
                        bail!("duplicate parameter `{parameter_name}` for tool `{name}`");
                    }
                    self.pending
                        .drain(..MINICPM5_PARAM_OPEN.len() + end + MINICPM5_NAME_CLOSE.len());
                    self.state = MiniCpm5ParseState::ParameterValue {
                        function_name: name,
                        arguments: std::mem::take(&mut arguments),
                        parameter_name,
                        value: String::new(),
                        mode: MiniCpm5ValueMode::Undecided,
                    };
                }
                MiniCpm5ParseState::ParameterValue {
                    function_name,
                    mut arguments,
                    parameter_name,
                    mut value,
                    mode,
                } => match mode {
                    MiniCpm5ValueMode::Undecided => {
                        if self.pending.starts_with(MINICPM5_CDATA_OPEN) {
                            self.pending.drain(..MINICPM5_CDATA_OPEN.len());
                            self.state = MiniCpm5ParseState::ParameterValue {
                                function_name,
                                arguments,
                                parameter_name,
                                value,
                                mode: MiniCpm5ValueMode::Cdata,
                            };
                            continue;
                        }
                        if self.pending.starts_with(MINICPM5_PARAM_CLOSE) {
                            self.state = MiniCpm5ParseState::ParameterValue {
                                function_name,
                                arguments,
                                parameter_name,
                                value,
                                mode: MiniCpm5ValueMode::Plain,
                            };
                            continue;
                        }
                        if !eof
                            && (MINICPM5_CDATA_OPEN.starts_with(&self.pending)
                                || MINICPM5_PARAM_CLOSE.starts_with(&self.pending))
                        {
                            self.state = MiniCpm5ParseState::ParameterValue {
                                function_name,
                                arguments,
                                parameter_name,
                                value,
                                mode: MiniCpm5ValueMode::Undecided,
                            };
                            break;
                        }
                        if self.pending.starts_with('<') {
                            bail!("MiniCPM5 string values containing `<` must use a CDATA block");
                        }
                        if self.pending.is_empty() {
                            if eof {
                                bail!("unterminated parameter value for tool `{function_name}`");
                            }
                            self.state = MiniCpm5ParseState::ParameterValue {
                                function_name,
                                arguments,
                                parameter_name,
                                value,
                                mode: MiniCpm5ValueMode::Undecided,
                            };
                            break;
                        }
                        self.state = MiniCpm5ParseState::ParameterValue {
                            function_name,
                            arguments,
                            parameter_name,
                            value,
                            mode: MiniCpm5ValueMode::Plain,
                        };
                    }
                    MiniCpm5ValueMode::Plain => {
                        if let Some(pos) = self.pending.find(MINICPM5_PARAM_CLOSE) {
                            value.push_str(&self.pending[..pos]);
                            validate_minicpm5_plain_value(&value)?;
                            if value.len() > MAX_ARGUMENT_BYTES {
                                bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                            }
                            self.pending.drain(..pos + MINICPM5_PARAM_CLOSE.len());
                            let value = parse_argument(
                                &self.definitions[&function_name],
                                &parameter_name,
                                &value,
                            )?;
                            arguments.insert(parameter_name, value);
                            self.state = MiniCpm5ParseState::FunctionBody {
                                name: function_name,
                                arguments,
                            };
                            continue;
                        }
                        if eof {
                            bail!("unterminated parameter value for tool `{function_name}`");
                        }
                        let keep = partial_marker_suffix_len(&self.pending, MINICPM5_PARAM_CLOSE);
                        let consume = self.pending.len().saturating_sub(keep);
                        value.push_str(&self.pending[..consume]);
                        validate_minicpm5_plain_value(&value)?;
                        if value.len() > MAX_ARGUMENT_BYTES {
                            bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                        }
                        self.pending.drain(..consume);
                        self.state = MiniCpm5ParseState::ParameterValue {
                            function_name,
                            arguments,
                            parameter_name,
                            value,
                            mode: MiniCpm5ValueMode::Plain,
                        };
                        break;
                    }
                    MiniCpm5ValueMode::Cdata => {
                        if let Some(pos) = self.pending.find(MINICPM5_CDATA_PARAM_CLOSE) {
                            value.push_str(&self.pending[..pos]);
                            if value.len() > MAX_ARGUMENT_BYTES {
                                bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                            }
                            self.pending.drain(..pos + MINICPM5_CDATA_PARAM_CLOSE.len());
                            let value = parse_argument(
                                &self.definitions[&function_name],
                                &parameter_name,
                                &value,
                            )?;
                            arguments.insert(parameter_name, value);
                            self.state = MiniCpm5ParseState::FunctionBody {
                                name: function_name,
                                arguments,
                            };
                            continue;
                        }
                        if eof {
                            bail!("unterminated CDATA parameter value for tool `{function_name}`");
                        }
                        let keep =
                            partial_marker_suffix_len(&self.pending, MINICPM5_CDATA_PARAM_CLOSE);
                        let consume = self.pending.len().saturating_sub(keep);
                        value.push_str(&self.pending[..consume]);
                        if value.contains("]]>") {
                            bail!("MiniCPM5 CDATA value contains an early `]]>` terminator");
                        }
                        if value.len() > MAX_ARGUMENT_BYTES {
                            bail!("native tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                        }
                        self.pending.drain(..consume);
                        self.state = MiniCpm5ParseState::ParameterValue {
                            function_name,
                            arguments,
                            parameter_name,
                            value,
                            mode: MiniCpm5ValueMode::Cdata,
                        };
                        break;
                    }
                },
            }
        }
        Ok(events)
    }
}

fn validate_minicpm5_plain_value(value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value
            .chars()
            .any(|character| matches!(character, '<' | '&' | '\r' | '\n')),
        "MiniCPM5 values containing `<`, `&`, or newlines must use a CDATA block"
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlamaNativeToolCall {
    name: String,
    parameters: Value,
}

/// Strict bounded incremental parser for the Llama 3.1/3.2 custom-function
/// protocol. Native calls are one bare JSON object with exactly `name` and
/// `parameters`; ordinary assistant text is distinguished by its first
/// non-whitespace character not being `{`.
pub struct LlamaToolCallParser {
    definitions: HashMap<String, ToolDefinition>,
    response_id: String,
    output_schema: Option<Value>,
    pending: String,
    text_mode: bool,
    saw_tool_call: bool,
}

impl LlamaToolCallParser {
    pub fn new(response_id: impl Into<String>, tools: &[ToolDefinition]) -> anyhow::Result<Self> {
        Self::new_with_output_schema(response_id, tools, None)
    }

    pub fn new_with_output_schema(
        response_id: impl Into<String>,
        tools: &[ToolDefinition],
        output_schema: Option<Value>,
    ) -> anyhow::Result<Self> {
        validate_tool_definitions(tools)?;
        let definitions = tools
            .iter()
            .cloned()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Ok(Self {
            definitions,
            response_id: response_id.into(),
            output_schema,
            pending: String::new(),
            text_mode: false,
            saw_tool_call: false,
        })
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        if self.text_mode {
            return Ok((!delta.is_empty())
                .then(|| GeneratedOutputEvent::TextDelta(delta.to_owned()))
                .into_iter()
                .collect());
        }
        if self.saw_tool_call {
            anyhow::ensure!(
                delta.trim().is_empty(),
                "unexpected text after native Llama tool call"
            );
            return Ok(Vec::new());
        }
        self.pending.push_str(delta);
        if self.pending.len() > MAX_PENDING_BYTES {
            bail!("tool-call parser pending buffer exceeds {MAX_PENDING_BYTES} bytes");
        }
        self.advance(false)
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        let events = self.advance(true)?;
        Ok((events, self.saw_tool_call))
    }

    fn advance(&mut self, eof: bool) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        let Some(first) = self
            .pending
            .chars()
            .find(|character| !character.is_whitespace())
        else {
            if eof && !self.pending.is_empty() {
                self.text_mode = true;
                return Ok(vec![GeneratedOutputEvent::TextDelta(std::mem::take(
                    &mut self.pending,
                ))]);
            }
            return Ok(Vec::new());
        };
        if first != '{' {
            self.text_mode = true;
            return Ok(vec![GeneratedOutputEvent::TextDelta(std::mem::take(
                &mut self.pending,
            ))]);
        }

        if let Ok(native) = serde_json::from_str::<LlamaNativeToolCall>(&self.pending) {
            if let Some(definition) = self.definitions.get(&native.name) {
                if validate_schema_value(&definition.parameters, &native.parameters).is_ok() {
                    self.pending.clear();
                    self.saw_tool_call = true;
                    return Ok(vec![GeneratedOutputEvent::ToolCall(ToolCall {
                        id: format!("call_{}_0", self.response_id),
                        name: native.name,
                        arguments: native.parameters,
                    })]);
                }
            }
        }
        let value: Value = match serde_json::from_str(&self.pending) {
            Ok(value) => value,
            Err(error) if !eof && error.is_eof() => return Ok(Vec::new()),
            Err(error) => bail!("invalid native Llama tool-call JSON: {error}"),
        };
        if let Some(schema) = &self.output_schema {
            validate_schema_value(schema, &value)
                .context("structured Llama output does not match its response schema")?;
            self.text_mode = true;
            return Ok(vec![GeneratedOutputEvent::TextDelta(std::mem::take(
                &mut self.pending,
            ))]);
        }
        let native: LlamaNativeToolCall = serde_json::from_str(&self.pending)
            .context("native Llama JSON is not a function-call envelope")?;
        validate_function_name(&native.name)?;
        let definition = self
            .definitions
            .get(&native.name)
            .ok_or_else(|| anyhow!("model requested unknown Llama tool `{}`", native.name))?;
        validate_schema_value(&definition.parameters, &native.parameters)
            .with_context(|| format!("tool `{}` arguments do not match its schema", native.name))?;
        self.pending.clear();
        self.saw_tool_call = true;
        Ok(vec![GeneratedOutputEvent::ToolCall(ToolCall {
            id: format!("call_{}_0", self.response_id),
            name: native.name,
            arguments: native.parameters,
        })])
    }
}

#[derive(Debug)]
enum GlmParseState {
    Text,
    FunctionName,
    FunctionBody {
        name: String,
        arguments: Map<String, Value>,
    },
    ArgumentValueOpen {
        function_name: String,
        arguments: Map<String, Value>,
        argument_name: String,
    },
    ArgumentValue {
        function_name: String,
        arguments: Map<String, Value>,
        argument_name: String,
        value: String,
    },
}

/// Strict bounded incremental parser for GLM-4's native XML tool syntax:
/// `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`.
pub struct GlmToolCallParser {
    definitions: HashMap<String, ToolDefinition>,
    response_id: String,
    pending: String,
    state: GlmParseState,
    next_index: usize,
    saw_tool_call: bool,
}

impl GlmToolCallParser {
    pub fn new(response_id: impl Into<String>, tools: &[ToolDefinition]) -> anyhow::Result<Self> {
        validate_tool_definitions(tools)?;
        let definitions = tools
            .iter()
            .cloned()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Ok(Self {
            definitions,
            response_id: response_id.into(),
            pending: String::new(),
            state: GlmParseState::Text,
            next_index: 0,
            saw_tool_call: false,
        })
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        self.pending.push_str(delta);
        if self.pending.len() > MAX_PENDING_BYTES {
            bail!("tool-call parser pending buffer exceeds {MAX_PENDING_BYTES} bytes");
        }
        self.advance(false)
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        let events = self.advance(true)?;
        if !matches!(self.state, GlmParseState::Text) {
            bail!("incomplete native GLM tool-call output");
        }
        if self.saw_tool_call && !self.pending.trim().is_empty() {
            bail!("unexpected text after native GLM tool call");
        }
        Ok((events, self.saw_tool_call))
    }

    fn advance(&mut self, eof: bool) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            let state = std::mem::replace(&mut self.state, GlmParseState::Text);
            match state {
                GlmParseState::Text => {
                    let marker = [TOOL_CALL_OPEN, GLM_THINK_OPEN, GLM_THINK_CLOSE]
                        .into_iter()
                        .filter_map(|marker| self.pending.find(marker).map(|pos| (pos, marker)))
                        .min_by_key(|(pos, _)| *pos);
                    if let Some((pos, marker)) = marker {
                        let text = self.pending[..pos].to_owned();
                        self.pending.drain(..pos + marker.len());
                        if self.saw_tool_call && !text.trim().is_empty() {
                            bail!("unexpected text between native GLM tool calls");
                        }
                        if !text.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(text));
                        }
                        if marker == TOOL_CALL_OPEN {
                            self.saw_tool_call = true;
                            self.state = GlmParseState::FunctionName;
                        } else {
                            if self.saw_tool_call {
                                bail!("unexpected thinking marker after native GLM tool call");
                            }
                            self.state = GlmParseState::Text;
                        }
                        continue;
                    }
                    if eof {
                        if self.saw_tool_call && !self.pending.trim().is_empty() {
                            bail!("unexpected text after native GLM tool call");
                        }
                        if !self.pending.is_empty() && !self.saw_tool_call {
                            events.push(GeneratedOutputEvent::TextDelta(std::mem::take(
                                &mut self.pending,
                            )));
                        } else {
                            self.pending.clear();
                        }
                    } else {
                        let keep = [TOOL_CALL_OPEN, GLM_THINK_OPEN, GLM_THINK_CLOSE]
                            .into_iter()
                            .map(|marker| partial_marker_suffix_len(&self.pending, marker))
                            .max()
                            .unwrap_or(0);
                        let emit_len = self.pending.len().saturating_sub(keep);
                        if emit_len > 0 {
                            let text = self.pending[..emit_len].to_owned();
                            self.pending.drain(..emit_len);
                            if self.saw_tool_call {
                                if !text.trim().is_empty() {
                                    bail!("unexpected text after native GLM tool call");
                                }
                            } else {
                                events.push(GeneratedOutputEvent::TextDelta(text));
                            }
                        }
                    }
                    self.state = GlmParseState::Text;
                    break;
                }
                GlmParseState::FunctionName => {
                    trim_native_whitespace(&mut self.pending);
                    let Some(marker_pos) = self.pending.find('<') else {
                        self.state = GlmParseState::FunctionName;
                        if eof {
                            bail!("unterminated native GLM function name");
                        }
                        break;
                    };
                    let name = self.pending[..marker_pos].trim_end().to_owned();
                    validate_function_name(&name)?;
                    if !self.definitions.contains_key(&name) {
                        bail!("model requested unknown GLM tool `{name}`");
                    }
                    self.pending.drain(..marker_pos);
                    self.state = GlmParseState::FunctionBody {
                        name,
                        arguments: Map::new(),
                    };
                }
                GlmParseState::FunctionBody {
                    name,
                    mut arguments,
                } => {
                    trim_native_whitespace(&mut self.pending);
                    if self.pending.starts_with(TOOL_CALL_CLOSE) {
                        self.pending.drain(..TOOL_CALL_CLOSE.len());
                        validate_schema_value(
                            &self.definitions[&name].parameters,
                            &Value::Object(arguments.clone()),
                        )
                        .with_context(|| {
                            format!("GLM tool `{name}` arguments do not match its schema")
                        })?;
                        events.push(GeneratedOutputEvent::ToolCall(ToolCall {
                            id: format!("call_{}_{}", self.response_id, self.next_index),
                            name,
                            arguments: Value::Object(arguments),
                        }));
                        self.next_index += 1;
                        self.state = GlmParseState::Text;
                        continue;
                    }
                    let Some(rest) = self.pending.strip_prefix(GLM_ARGUMENT_KEY_OPEN) else {
                        if !eof
                            && (GLM_ARGUMENT_KEY_OPEN.starts_with(&self.pending)
                                || TOOL_CALL_CLOSE.starts_with(&self.pending))
                        {
                            self.state = GlmParseState::FunctionBody { name, arguments };
                            break;
                        }
                        bail!("expected GLM argument key or tool-call close for tool `{name}`");
                    };
                    let Some(end) = rest.find(GLM_ARGUMENT_KEY_CLOSE) else {
                        if eof {
                            bail!("unterminated GLM argument key for tool `{name}`");
                        }
                        self.state = GlmParseState::FunctionBody { name, arguments };
                        break;
                    };
                    let argument_name = rest[..end].to_owned();
                    validate_parameter_name(&argument_name)?;
                    if arguments.contains_key(&argument_name) {
                        bail!("duplicate argument `{argument_name}` for GLM tool `{name}`");
                    }
                    if !self.definitions[&name].parameters["properties"]
                        .as_object()
                        .expect("schema validated")
                        .contains_key(&argument_name)
                    {
                        bail!("GLM tool `{name}` emitted undeclared argument `{argument_name}`");
                    }
                    self.pending
                        .drain(..GLM_ARGUMENT_KEY_OPEN.len() + end + GLM_ARGUMENT_KEY_CLOSE.len());
                    self.state = GlmParseState::ArgumentValueOpen {
                        function_name: name,
                        arguments: std::mem::take(&mut arguments),
                        argument_name,
                    };
                }
                GlmParseState::ArgumentValueOpen {
                    function_name,
                    arguments,
                    argument_name,
                } => {
                    trim_native_whitespace(&mut self.pending);
                    if !self.pending.starts_with(GLM_ARGUMENT_VALUE_OPEN) {
                        if !eof && GLM_ARGUMENT_VALUE_OPEN.starts_with(&self.pending) {
                            self.state = GlmParseState::ArgumentValueOpen {
                                function_name,
                                arguments,
                                argument_name,
                            };
                            break;
                        }
                        bail!(
                            "expected `{GLM_ARGUMENT_VALUE_OPEN}` for GLM tool argument `{argument_name}`"
                        );
                    }
                    self.pending.drain(..GLM_ARGUMENT_VALUE_OPEN.len());
                    self.state = GlmParseState::ArgumentValue {
                        function_name,
                        arguments,
                        argument_name,
                        value: String::new(),
                    };
                }
                GlmParseState::ArgumentValue {
                    function_name,
                    mut arguments,
                    argument_name,
                    mut value,
                } => {
                    if let Some(pos) = self.pending.find(GLM_ARGUMENT_VALUE_CLOSE) {
                        value.push_str(&self.pending[..pos]);
                        if value.len() > MAX_ARGUMENT_BYTES {
                            bail!("native GLM tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                        }
                        self.pending.drain(..pos + GLM_ARGUMENT_VALUE_CLOSE.len());
                        let value = parse_argument(
                            &self.definitions[&function_name],
                            &argument_name,
                            &value,
                        )?;
                        arguments.insert(argument_name, value);
                        self.state = GlmParseState::FunctionBody {
                            name: function_name,
                            arguments,
                        };
                        continue;
                    }
                    if eof {
                        bail!("unterminated argument value for GLM tool `{function_name}`");
                    }
                    let keep = partial_marker_suffix_len(&self.pending, GLM_ARGUMENT_VALUE_CLOSE);
                    let consume = self.pending.len().saturating_sub(keep);
                    value.push_str(&self.pending[..consume]);
                    if value.len() > MAX_ARGUMENT_BYTES {
                        bail!("native GLM tool argument exceeds {MAX_ARGUMENT_BYTES} bytes");
                    }
                    self.pending.drain(..consume);
                    self.state = GlmParseState::ArgumentValue {
                        function_name,
                        arguments,
                        argument_name,
                        value,
                    };
                    break;
                }
            }
        }
        Ok(events)
    }
}

/// Strict bounded parser for Gemma's native `call:name{...}` syntax. The
/// surrounding call and string delimiters are tokenizer special tokens, so
/// callers must decode with `skip_special_tokens=false` for this dialect.
pub struct GemmaToolCallParser {
    definitions: HashMap<String, ToolDefinition>,
    response_id: String,
    pending: String,
    channel_label_pending: String,
    in_call: bool,
    next_index: usize,
    saw_tool_call: bool,
}

impl GemmaToolCallParser {
    pub fn new(response_id: impl Into<String>, tools: &[ToolDefinition]) -> anyhow::Result<Self> {
        validate_tool_definitions(tools)?;
        let definitions = tools
            .iter()
            .cloned()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Ok(Self {
            definitions,
            response_id: response_id.into(),
            pending: String::new(),
            channel_label_pending: String::new(),
            in_call: false,
            next_index: 0,
            saw_tool_call: false,
        })
    }

    pub fn push(&mut self, delta: &str) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        self.pending.push_str(delta);
        if self.pending.len() > MAX_PENDING_BYTES {
            bail!("tool-call parser pending buffer exceeds {MAX_PENDING_BYTES} bytes");
        }
        self.advance(false)
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<GeneratedOutputEvent>, bool)> {
        let events = self.advance(true)?;
        if self.in_call {
            bail!("incomplete native Gemma tool-call output");
        }
        let trailing_raw = std::mem::take(&mut self.pending);
        let trailing = self.sanitize_text(&trailing_raw, true);
        if self.saw_tool_call && !trailing.trim().is_empty() {
            bail!("unexpected text after native tool call");
        }
        Ok((events, self.saw_tool_call))
    }

    fn advance(&mut self, eof: bool) -> anyhow::Result<Vec<GeneratedOutputEvent>> {
        let mut events = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.pending.find(GEMMA_TOOL_CALL_CLOSE) else {
                    if eof {
                        bail!("unterminated native Gemma tool call");
                    }
                    break;
                };
                let body = self.pending[..end].to_owned();
                self.pending.drain(..end + GEMMA_TOOL_CALL_CLOSE.len());
                let (name, arguments) = parse_gemma_call(&body)?;
                let definition = self
                    .definitions
                    .get(&name)
                    .ok_or_else(|| anyhow!("model requested unknown tool `{name}`"))?;
                validate_schema_value(&definition.parameters, &Value::Object(arguments.clone()))
                    .with_context(|| format!("tool `{name}` arguments do not match its schema"))?;
                events.push(GeneratedOutputEvent::ToolCall(ToolCall {
                    id: format!("call_{}_{}", self.response_id, self.next_index),
                    name,
                    arguments: Value::Object(arguments),
                }));
                self.next_index += 1;
                self.saw_tool_call = true;
                self.in_call = false;
                continue;
            }

            if let Some(start) = self.pending.find(GEMMA_TOOL_CALL_OPEN) {
                let raw_text = self.pending[..start].to_owned();
                let text = self.sanitize_text(&raw_text, false);
                if self.saw_tool_call && !text.trim().is_empty() {
                    bail!("unexpected text between native tool calls");
                }
                if !text.is_empty() && !self.saw_tool_call {
                    events.push(GeneratedOutputEvent::TextDelta(text));
                }
                self.pending.drain(..start + GEMMA_TOOL_CALL_OPEN.len());
                self.in_call = true;
                continue;
            }

            if eof {
                let raw_text = std::mem::take(&mut self.pending);
                let text = self.sanitize_text(&raw_text, true);
                if self.saw_tool_call && !text.trim().is_empty() {
                    bail!("unexpected text after native tool call");
                }
                if !text.is_empty() && !self.saw_tool_call {
                    events.push(GeneratedOutputEvent::TextDelta(text));
                }
            } else {
                let keep = gemma_partial_marker_suffix_len(&self.pending);
                let emit_len = self.pending.len().saturating_sub(keep);
                if emit_len > 0 {
                    let raw_text = self.pending[..emit_len].to_owned();
                    self.pending.drain(..emit_len);
                    let text = self.sanitize_text(&raw_text, false);
                    if self.saw_tool_call && !text.trim().is_empty() {
                        bail!("unexpected text after native tool call");
                    }
                    if !text.is_empty() && !self.saw_tool_call {
                        events.push(GeneratedOutputEvent::TextDelta(text));
                    }
                }
            }
            break;
        }
        Ok(events)
    }

    fn sanitize_text(&mut self, value: &str, eof: bool) -> String {
        const CHANNEL_OPEN: &str = "<|channel>";
        const CHANNEL_LABEL: &str = "thought\n";
        let mut output = String::new();
        let mut remaining = value.to_owned();
        loop {
            if !self.channel_label_pending.is_empty() {
                self.channel_label_pending.push_str(&remaining);
                if CHANNEL_LABEL.starts_with(&self.channel_label_pending) && !eof {
                    break;
                }
                if let Some(rest) = self.channel_label_pending.strip_prefix(CHANNEL_LABEL) {
                    let rest = rest.to_owned();
                    self.channel_label_pending.clear();
                    remaining = rest;
                    continue;
                }
                output.push_str(&sanitize_gemma_controls(&self.channel_label_pending));
                self.channel_label_pending.clear();
                break;
            }
            let Some(position) = remaining.find(CHANNEL_OPEN) else {
                output.push_str(&sanitize_gemma_controls(&remaining));
                break;
            };
            output.push_str(&sanitize_gemma_controls(&remaining[..position]));
            self.channel_label_pending
                .push_str(&remaining[position + CHANNEL_OPEN.len()..]);
            remaining.clear();
        }
        output
    }
}

fn sanitize_gemma_controls(value: &str) -> String {
    [
        "<channel|>",
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

fn gemma_partial_marker_suffix_len(value: &str) -> usize {
    [
        GEMMA_TOOL_CALL_OPEN,
        "<|channel>thought\n",
        "<|channel>",
        "<channel|>",
        "<|turn>model\n",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<eos>",
        "<bos>",
    ]
    .into_iter()
    .map(|marker| partial_marker_suffix_len(value, marker))
    .max()
    .unwrap_or(0)
}

fn parse_gemma_call(body: &str) -> anyhow::Result<(String, Map<String, Value>)> {
    let body = body
        .trim()
        .strip_prefix("call:")
        .ok_or_else(|| anyhow!("expected `call:` after `{GEMMA_TOOL_CALL_OPEN}`"))?;
    let object_start = body
        .find('{')
        .ok_or_else(|| anyhow!("native Gemma tool call is missing arguments object"))?;
    let name = body[..object_start].trim().to_owned();
    validate_function_name(&name)?;
    let mut parser = GemmaValueParser::new(&body[object_start..]);
    let arguments = parser
        .parse_value()?
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("native Gemma tool arguments must be an object"))?;
    parser.skip_whitespace();
    anyhow::ensure!(
        parser.is_eof(),
        "unexpected data after Gemma tool arguments"
    );
    Ok((name, arguments))
}

struct GemmaValueParser<'a> {
    input: &'a str,
    offset: usize,
}

impl<'a> GemmaValueParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
    }

    fn is_eof(&self) -> bool {
        self.offset == self.input.len()
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.offset..]
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.remaining().chars().next() {
            if !ch.is_whitespace() {
                break;
            }
            self.offset += ch.len_utf8();
        }
    }

    fn consume(&mut self, expected: &str) -> bool {
        if self.remaining().starts_with(expected) {
            self.offset += expected.len();
            true
        } else {
            false
        }
    }

    fn parse_value(&mut self) -> anyhow::Result<Value> {
        self.skip_whitespace();
        if self.consume(GEMMA_STRING_DELIMITER) {
            let end = self
                .remaining()
                .find(GEMMA_STRING_DELIMITER)
                .ok_or_else(|| anyhow!("unterminated Gemma string argument"))?;
            let value = self.remaining()[..end].to_owned();
            self.offset += end + GEMMA_STRING_DELIMITER.len();
            return Ok(Value::String(value));
        }
        match self.remaining().chars().next() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some(_) => self.parse_scalar(),
            None => bail!("expected Gemma argument value"),
        }
    }

    fn parse_object(&mut self) -> anyhow::Result<Value> {
        anyhow::ensure!(self.consume("{"), "expected object open");
        self.skip_whitespace();
        let mut object = Map::new();
        if self.consume("}") {
            return Ok(Value::Object(object));
        }
        loop {
            let colon = self
                .remaining()
                .find(':')
                .ok_or_else(|| anyhow!("Gemma object key is missing `:`"))?;
            let key = self.remaining()[..colon].trim().to_owned();
            validate_parameter_name(&key)?;
            self.offset += colon + 1;
            let value = self.parse_value()?;
            anyhow::ensure!(
                object.insert(key.clone(), value).is_none(),
                "duplicate parameter `{key}`"
            );
            self.skip_whitespace();
            if self.consume("}") {
                break;
            }
            anyhow::ensure!(
                self.consume(","),
                "expected `,` between Gemma object fields"
            );
            self.skip_whitespace();
        }
        Ok(Value::Object(object))
    }

    fn parse_array(&mut self) -> anyhow::Result<Value> {
        anyhow::ensure!(self.consume("["), "expected array open");
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume("]") {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume("]") {
                break;
            }
            anyhow::ensure!(self.consume(","), "expected `,` between Gemma array items");
        }
        Ok(Value::Array(values))
    }

    fn parse_scalar(&mut self) -> anyhow::Result<Value> {
        let end = self
            .remaining()
            .find([',', '}', ']'])
            .unwrap_or(self.remaining().len());
        let raw = self.remaining()[..end].trim();
        anyhow::ensure!(!raw.is_empty(), "empty Gemma scalar argument");
        self.offset += end;
        serde_json::from_str(raw).context("invalid Gemma scalar argument")
    }
}

pub fn validate_tool_definitions(tools: &[ToolDefinition]) -> anyhow::Result<()> {
    if tools.is_empty() {
        bail!("tools must contain at least one function definition");
    }
    let mut names = HashSet::new();
    for tool in tools {
        validate_function_name(&tool.name)?;
        if !names.insert(tool.name.as_str()) {
            bail!("duplicate tool name `{}`", tool.name);
        }
    }
    validate_tool_schemas(tools)?;
    for tool in tools.iter().filter(|tool| tool.strict == Some(true)) {
        validate_strict_tool_schema(tool)?;
    }
    Ok(())
}

pub fn validate_function_name(name: &str) -> anyhow::Result<()> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        bail!("function name must match [A-Za-z0-9_-]{{1,64}}, got `{name}`");
    }
    Ok(())
}

fn validate_parameter_name(name: &str) -> anyhow::Result<()> {
    if name.is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
        bail!("invalid native tool parameter name");
    }
    Ok(())
}

fn parse_argument(definition: &ToolDefinition, name: &str, raw: &str) -> anyhow::Result<Value> {
    let properties = definition.parameters["properties"]
        .as_object()
        .expect("schema validated");
    let schema = properties.get(name).ok_or_else(|| {
        anyhow!(
            "tool `{}` emitted undeclared argument `{name}`",
            definition.name
        )
    })?;
    let value = if schema_is_string_only(schema) {
        Value::String(raw.to_owned())
    } else if let Ok(value) = serde_json::from_str::<Value>(raw.trim()) {
        if validate_schema_value(schema, &value).is_ok() {
            value
        } else if schema_accepts_string(schema) {
            Value::String(raw.to_owned())
        } else {
            value
        }
    } else if schema_accepts_string(schema) {
        Value::String(raw.to_owned())
    } else {
        serde_json::from_str(raw.trim()).with_context(|| {
            format!(
                "tool `{}` argument `{name}` is not valid JSON",
                definition.name
            )
        })?
    };
    validate_schema_value(schema, &value).with_context(|| {
        format!(
            "tool `{}` argument `{name}` does not match its schema",
            definition.name
        )
    })?;
    Ok(value)
}

fn trim_native_whitespace(value: &mut String) {
    let bytes = value
        .bytes()
        .take_while(|byte| byte.is_ascii_whitespace())
        .count();
    value.drain(..bytes);
}

fn strip_native_value_newlines(mut value: &str) -> &str {
    value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

fn partial_marker_suffix_len(value: &str, marker: &str) -> usize {
    let max = value.len().min(marker.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&len| {
            value.is_char_boundary(value.len() - len)
                && marker.starts_with(&value[value.len() - len..])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tool() -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city"]
            }
        }))
        .unwrap()
    }

    #[test]
    fn detects_only_complete_qwen_contract() {
        let template = "<tool_call><function=<parameter=<tool_response> tool_call.function is defined tool_call.arguments";
        assert_eq!(
            ToolDialect::detect("qwen3_5", template),
            Some(ToolDialect::Qwen35)
        );
        assert_eq!(ToolDialect::detect("minicpmv4_6", template), None);
        assert_eq!(ToolDialect::detect("qwen3_5", "<tool_call>"), None);

        let gemma = concat!(
            "<|tool>declaration:",
            "<|tool_call>call:",
            "<tool_call|>",
            "<|tool_response>response:",
            "tool_call_id",
            "<|\"|>"
        );
        assert_eq!(
            ToolDialect::detect("gemma4", gemma),
            Some(ToolDialect::Gemma)
        );
        assert_eq!(
            ToolDialect::detect("gemma4_unified", gemma),
            Some(ToolDialect::Gemma)
        );
        assert_eq!(
            ToolDialect::detect("diffusion_gemma", gemma),
            Some(ToolDialect::Gemma)
        );
        assert_eq!(ToolDialect::detect("gemma4_text", gemma), None);
        assert_eq!(ToolDialect::detect("gemma4", "<|tool_call>call:"), None);

        let glm = concat!(
            "# Tools<tools>",
            "tool | tojson(ensure_ascii=False)",
            "<tool_call>{function-name}<arg_key>",
            "'<tool_call>' + tc.name",
            "<arg_key>{{ k }}</arg_key><arg_value>",
            "<|observation|><tool_response>",
            "m.tool_calls"
        );
        assert_eq!(
            ToolDialect::detect("glm4_moe_lite", glm),
            Some(ToolDialect::Glm)
        );
        assert_eq!(ToolDialect::detect("glm4", glm), None);
        assert_eq!(
            ToolDialect::detect("glm4_moe_lite", "<tool_call><arg_key>"),
            None
        );

        let llama = concat!(
            "custom_tools is defined tools_in_user_message Environment: ipython ",
            "Respond in the format {\"name\": function name, \"parameters\": dictionary ",
            "message.tool_calls|length == 1 tool_call.arguments | tojson ",
            "<|start_header_id|>ipython<|end_header_id|> ",
            "<|start_header_id|>assistant<|end_header_id|> <|eot_id|>"
        );
        assert_eq!(
            ToolDialect::detect("llama", llama),
            Some(ToolDialect::Llama)
        );
        assert_eq!(ToolDialect::detect("minicpm5", llama), None);
        assert_eq!(
            ToolDialect::detect("llama", "Respond in the format {\"name\":"),
            None
        );

        let minicpm_v46 = concat!(
            "If you choose to call a function ONLY reply in the following format ",
            "<tool_call>\\n<function=example_function_name> ",
            "<parameter=example_parameter_1> tool_call.arguments|items ",
            "<tool_response> render_content(message.content) <|image_pad|> ",
            "No user query found in messages."
        );
        assert_eq!(
            ToolDialect::detect("minicpmv4_6", minicpm_v46),
            Some(ToolDialect::MiniCpmV46)
        );
        assert_eq!(ToolDialect::detect("qwen3_5", minicpm_v46), None);

        let minicpm5 = concat!(
            "You are provided with function signatures within <tools></tools> XML tags ",
            "<function name=\"function-name\"><param name=\"param-name\"> ",
            "wrap it in a CDATA block ",
            "'<function name=\"' ~ tool_call.name ~ '\">' ",
            "'<param name=\"' ~ param_name ~ '\">' ",
            "tool_call.arguments <tool_response> <tool_def_sep>"
        );
        assert_eq!(
            ToolDialect::detect("llama", minicpm5),
            Some(ToolDialect::MiniCpm5)
        );
        assert_eq!(ToolDialect::detect("minicpmv4_6", minicpm5), None);
    }

    #[test]
    fn incrementally_parses_unicode_and_typed_arguments() {
        let mut parser = QwenToolCallParser::new("abc", &[weather_tool()]).unwrap();
        let chunks = [
            "<tool_",
            "call>\n<function=get_weather>\n<parameter=city>\n东",
            "京\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        ];
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);
        assert!(saw_tool_call);
        assert_eq!(
            events,
            vec![GeneratedOutputEvent::ToolCall(ToolCall {
                id: "call_abc_0".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "东京", "days": 3}),
            })]
        );
    }

    #[test]
    fn minicpm5_incrementally_parses_cdata_and_typed_arguments() {
        let mut parser = MiniCpm5ToolCallParser::new("mini", &[weather_tool()]).unwrap();
        let chunks = [
            "checking\n<func",
            "tion name=\"get_weather\"><param name=\"city\"><![CDA",
            "TA[Tokyo\n<&]]></param><param name=\"days\">2</param></function>",
        ];
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);
        assert!(saw_tool_call);
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::TextDelta("checking\n".into()),
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_mini_0".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Tokyo\n<&", "days": 2}),
                }),
            ]
        );
    }

    #[test]
    fn minicpm5_parses_parallel_calls_and_rejects_non_cdata_xml_text() {
        let call = "<function name=\"get_weather\"><param name=\"city\">Tokyo</param></function>";
        let mut parser = MiniCpm5ToolCallParser::new("parallel", &[weather_tool()]).unwrap();
        let events = parser.push(&format!("{call}\n{call}")).unwrap();
        let (tail, saw_tool_call) = parser.finish().unwrap();
        assert!(saw_tool_call);
        assert!(tail.is_empty());
        assert_eq!(events.len(), 2);
        assert_eq!(
            events
                .iter()
                .map(|event| match event {
                    GeneratedOutputEvent::ToolCall(call) => call.id.as_str(),
                    GeneratedOutputEvent::TextDelta(_) => panic!("unexpected text event"),
                    other => panic!("unexpected {} event", other.kind()),
                })
                .collect::<Vec<_>>(),
            ["call_parallel_0", "call_parallel_1"]
        );

        let mut malformed = MiniCpm5ToolCallParser::new("bad", &[weather_tool()]).unwrap();
        let error = malformed
            .push(
                "<function name=\"get_weather\"><param name=\"city\">Tokyo & Osaka</param></function>",
            )
            .unwrap_err();
        assert!(error.to_string().contains("CDATA"));
    }

    #[test]
    fn parses_multiple_calls_with_stable_distinct_indexes() {
        let mut parser = QwenToolCallParser::new("abc", &[weather_tool()]).unwrap();
        let native = concat!(
            "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>",
            "</function></tool_call>",
            "<tool_call><function=get_weather><parameter=city>Osaka</parameter>",
            "<parameter=days>2</parameter></function></tool_call>"
        );
        let mut events = parser.push(native).unwrap();
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);

        assert!(saw_tool_call);
        assert_eq!(events.len(), 2);
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_abc_0".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Tokyo"}),
                }),
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_abc_1".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Osaka", "days": 2}),
                }),
            ]
        );
    }

    #[test]
    fn buffers_fragmented_whitespace_between_native_tool_calls() {
        let mut parser = QwenToolCallParser::new("fragmented", &[weather_tool()]).unwrap();
        let first = concat!(
            "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>",
            "</function></tool_call>"
        );
        let second = concat!(
            "<tool_call><function=get_weather><parameter=city>Osaka</parameter>",
            "</function></tool_call>"
        );
        let mut events = parser.push(first).unwrap();
        assert_eq!(events.len(), 1);
        assert!(parser.push("\n").unwrap().is_empty());
        assert!(parser.push("<tool_").unwrap().is_empty());
        events.extend(
            parser
                .push(&format!("call>{}", &second["<tool_call>".len()..]))
                .unwrap(),
        );
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);

        assert!(saw_tool_call);
        assert_eq!(events.len(), 2);
        assert!(events
            .iter()
            .all(|event| matches!(event, GeneratedOutputEvent::ToolCall(_))));
    }

    #[test]
    fn streams_plain_text_without_waiting_for_eof() {
        let mut parser = QwenToolCallParser::new("abc", &[weather_tool()]).unwrap();
        let events = parser.push("ordinary answer").unwrap();
        assert_eq!(
            events,
            vec![GeneratedOutputEvent::TextDelta("ordinary answer".into())]
        );
        let (tail, saw_tool_call) = parser.finish().unwrap();
        assert!(!saw_tool_call);
        assert!(tail.is_empty());
    }

    #[test]
    fn incrementally_parses_llama_bare_json_and_streams_disjoint_text() {
        let mut parser = LlamaToolCallParser::new("llama", &[weather_tool()]).unwrap();
        assert!(parser.push(" \n{\"na").unwrap().is_empty());
        let events = parser
            .push("me\":\"get_weather\",\"parameters\":{\"city\":\"东京\",\"days\":2}}")
            .unwrap();
        assert_eq!(
            events,
            vec![GeneratedOutputEvent::ToolCall(ToolCall {
                id: "call_llama_0".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "东京", "days": 2}),
            })]
        );
        assert!(parser.push(" \n").unwrap().is_empty());
        assert!(parser.finish().unwrap().1);

        let mut text = LlamaToolCallParser::new("text", &[weather_tool()]).unwrap();
        assert!(text.push("  ").unwrap().is_empty());
        assert_eq!(
            text.push("ordinary").unwrap(),
            vec![GeneratedOutputEvent::TextDelta("  ordinary".into())]
        );
        assert_eq!(
            text.push(" answer").unwrap(),
            vec![GeneratedOutputEvent::TextDelta(" answer".into())]
        );
        assert!(!text.finish().unwrap().1);
    }

    #[test]
    fn llama_parser_rejects_non_contract_json_unknown_tools_and_bad_arguments() {
        let cases = [
            r#"{"name":"unknown","parameters":{"city":"Tokyo"}}"#,
            r#"{"name":"get_weather","arguments":{"city":"Tokyo"}}"#,
            r#"{"name":"get_weather","parameters":{"days":2}}"#,
            r#"{"name":"get_weather","parameters":{"city":"Tokyo"},"extra":true}"#,
            r#"{"name":"get_weather","name":"get_weather","parameters":{"city":"Tokyo"}}"#,
        ];
        for native in cases {
            let mut parser = LlamaToolCallParser::new("bad", &[weather_tool()]).unwrap();
            assert!(parser.push(native).is_err(), "must reject {native}");
        }

        let mut parser = LlamaToolCallParser::new("bad", &[weather_tool()]).unwrap();
        parser
            .push(r#"{"name":"get_weather","parameters":{"city":"Tokyo"}"#)
            .unwrap();
        assert!(parser.finish().is_err());
    }

    #[test]
    fn llama_parser_distinguishes_structured_json_and_prioritizes_tool_envelopes() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let mut structured = LlamaToolCallParser::new_with_output_schema(
            "structured",
            &[weather_tool()],
            Some(schema.clone()),
        )
        .unwrap();
        let json = r#"{"answer":"sunny"}"#;
        assert_eq!(
            structured.push(json).unwrap(),
            vec![GeneratedOutputEvent::TextDelta(json.into())]
        );
        assert!(!structured.finish().unwrap().1);

        let overlap_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            },
            "required": ["name", "parameters"],
            "additionalProperties": false
        });
        let mut tool = LlamaToolCallParser::new_with_output_schema(
            "precedence",
            &[weather_tool()],
            Some(overlap_schema),
        )
        .unwrap();
        let events = tool
            .push(r#"{"name":"get_weather","parameters":{"city":"Tokyo"}}"#)
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [GeneratedOutputEvent::ToolCall(_)]
        ));
        assert!(tool.finish().unwrap().1);
    }

    #[test]
    fn rejects_unknown_tools_and_malformed_values() {
        let mut parser = QwenToolCallParser::new("abc", &[weather_tool()]).unwrap();
        let error = parser
            .push("<tool_call><function=nope></function></tool_call>")
            .unwrap_err();
        assert!(error.to_string().contains("unknown tool"));

        let mut parser = QwenToolCallParser::new("abc", &[weather_tool()]).unwrap();
        let error = parser
            .push("<tool_call><function=get_weather><parameter=city>x</parameter><parameter=days>many</parameter></function></tool_call>")
            .unwrap_err();
        assert!(error.to_string().contains("not valid JSON"));
    }

    #[test]
    fn incrementally_parses_glm_calls_and_typed_arguments() {
        let mut parser = GlmToolCallParser::new("glm", &[weather_tool()]).unwrap();
        let chunks = [
            "<think>需要查询。</think><tool_",
            "call>get_wea",
            "ther<arg_key>city</arg_key><arg_value>东",
            "京</arg_value><arg_key>days</arg_key><arg_value>3</arg_",
            "value></tool_call>",
        ];
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);

        assert!(saw_tool_call);
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::TextDelta("需要查询。".into()),
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_glm_0".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "东京", "days": 3}),
                }),
            ]
        );
    }

    #[test]
    fn glm_parser_handles_parallel_calls_and_rejects_invalid_arguments() {
        let native = concat!(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Osaka</arg_value>",
            "<arg_key>days</arg_key><arg_value>2</arg_value></tool_call>"
        );
        let mut parser = GlmToolCallParser::new("glm", &[weather_tool()]).unwrap();
        let mut events = parser.push(native).unwrap();
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);
        assert!(saw_tool_call);
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[1],
            GeneratedOutputEvent::ToolCall(ToolCall {
                id: "call_glm_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Osaka", "days": 2}),
            })
        );

        let mut parser = GlmToolCallParser::new("glm", &[weather_tool()]).unwrap();
        let error = parser.push("<tool_call>unknown</tool_call>").unwrap_err();
        assert!(error.to_string().contains("unknown GLM tool"));

        let mut parser = GlmToolCallParser::new("glm", &[weather_tool()]).unwrap();
        let error = parser
            .push(concat!(
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
                "<arg_key>days</arg_key><arg_value>many</arg_value></tool_call>"
            ))
            .unwrap_err();
        assert!(error.to_string().contains("not valid JSON"));

        let mut parser = GlmToolCallParser::new("glm", &[weather_tool()]).unwrap();
        let events = parser
            .push("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert!(parser.push("   \n").unwrap().is_empty());
        let (tail, saw_tool_call) = parser.finish().unwrap();
        assert!(tail.is_empty());
        assert!(saw_tool_call);
    }

    #[test]
    fn incrementally_parses_gemma_calls_and_native_values() {
        let mut parser = GemmaToolCallParser::new("gemma", &[weather_tool()]).unwrap();
        let chunks = [
            "<|channel>",
            "tho",
            "ught\n需要查询。\n<channel|><|tool_",
            "call>call:get_weather{days:3,city:<|\"|>东京<|\"|>}<tool_call|>",
        ];
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        let (tail, saw_tool_call) = parser.finish().unwrap();
        events.extend(tail);
        assert!(saw_tool_call);
        assert_eq!(
            events,
            vec![
                GeneratedOutputEvent::TextDelta("需要查询。\n".into()),
                GeneratedOutputEvent::ToolCall(ToolCall {
                    id: "call_gemma_0".into(),
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "东京", "days": 3}),
                }),
            ]
        );
    }

    #[test]
    fn parses_nested_gemma_argument_values() {
        let tool: ToolDefinition = serde_json::from_value(serde_json::json!({
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "exact": {"type": "boolean"}
                        },
                        "required": ["tags"]
                    }
                },
                "required": ["filters"]
            }
        }))
        .unwrap();
        let mut parser = GemmaToolCallParser::new("gemma", &[tool]).unwrap();
        let events = parser
            .push(concat!(
                "<|tool_call>call:search{filters:{exact:true,tags:[",
                "<|\"|>rust<|\"|>,<|\"|>mlx<|\"|>]}}<tool_call|>"
            ))
            .unwrap();
        assert!(parser.finish().unwrap().1);
        assert_eq!(
            events,
            vec![GeneratedOutputEvent::ToolCall(ToolCall {
                id: "call_gemma_0".into(),
                name: "search".into(),
                arguments: serde_json::json!({
                    "filters": {"exact": true, "tags": ["rust", "mlx"]}
                }),
            })]
        );
    }

    #[test]
    fn rejects_malformed_or_unknown_gemma_calls() {
        let mut parser = GemmaToolCallParser::new("gemma", &[weather_tool()]).unwrap();
        let error = parser
            .push("<|tool_call>call:nope{city:<|\"|>Tokyo<|\"|>}<tool_call|>")
            .unwrap_err();
        assert!(error.to_string().contains("unknown tool"));

        let mut parser = GemmaToolCallParser::new("gemma", &[weather_tool()]).unwrap();
        let error = parser
            .push("<|tool_call>call:get_weather{days:many}<tool_call|>")
            .unwrap_err();
        assert!(error.to_string().contains("invalid Gemma scalar"));
    }

    #[test]
    fn validates_strict_schema_requirements_recursively() {
        let mut valid = weather_tool();
        valid.strict = Some(true);
        valid.parameters["required"] = serde_json::json!(["city", "days"]);
        valid.parameters["additionalProperties"] = Value::Bool(false);
        validate_tool_definitions(&[valid]).unwrap();

        let mut missing_closed_object = weather_tool();
        missing_closed_object.strict = Some(true);
        assert!(validate_tool_definitions(&[missing_closed_object])
            .unwrap_err()
            .to_string()
            .contains("additionalProperties=false"));

        let mut missing_required = weather_tool();
        missing_required.strict = Some(true);
        missing_required.parameters["additionalProperties"] = Value::Bool(false);
        assert!(validate_tool_definitions(&[missing_required])
            .unwrap_err()
            .to_string()
            .contains("days"));

        let nested: ToolDefinition = serde_json::from_value(serde_json::json!({
            "name": "nested",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "options": {
                        "type": "object",
                        "properties": {"units": {"type": "string"}},
                        "required": ["units"]
                    }
                },
                "required": ["options"],
                "additionalProperties": false
            }
        }))
        .unwrap();
        let error = validate_tool_definitions(&[nested]).unwrap_err();
        assert!(error.to_string().contains("$.options"), "{error:#}");
    }
}
