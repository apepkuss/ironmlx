//! Completion-boundary semantic guards for schemas beyond the native grammar.
use std::sync::Arc;

use serde_json::Value;

use super::generated_output::GeneratedOutputEvent;
use super::json_schema::SchemaValidator;
use super::native_output::{NativeOutputDecoderConfig, NativeOutputParser};
use super::tool_calling::{ToolCallParser, ToolDefinition, ToolDialect};
use crate::Result;

#[derive(Clone, Debug)]
enum Contract {
    Json(SchemaValidator),
    Tools {
        dialect: ToolDialect,
        definitions: Vec<ToolDefinition>,
        output: Option<(Value, SchemaValidator)>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct SemanticContract {
    contract: Contract,
    native: Option<NativeOutputDecoderConfig>,
}

impl SemanticContract {
    pub(crate) fn json(
        validator: SchemaValidator,
        native: Option<NativeOutputDecoderConfig>,
    ) -> Self {
        Self {
            contract: Contract::Json(validator),
            native,
        }
    }

    pub(crate) fn tools(
        dialect: ToolDialect,
        definitions: Vec<ToolDefinition>,
        output: Option<(Value, SchemaValidator)>,
        native: Option<NativeOutputDecoderConfig>,
    ) -> Self {
        Self {
            contract: Contract::Tools {
                dialect,
                definitions,
                output,
            },
            native,
        }
    }

    fn text(&self, bytes: &[u8]) -> Result<Option<String>> {
        let Ok(text) = std::str::from_utf8(bytes) else {
            return Ok(None);
        };
        let Some(native) = self.native else {
            return Ok(Some(text.to_owned()));
        };
        let mut parser = NativeOutputParser::new(native);
        let mut events = parser.push(text)?;
        events.extend(parser.finish("stop")?);
        Ok(Some(
            events
                .into_iter()
                .filter_map(|event| match event {
                    GeneratedOutputEvent::TextDelta(text) => Some(text),
                    _ => None,
                })
                .collect(),
        ))
    }

    fn accepts(&self, text: &str, eof: bool) -> bool {
        match &self.contract {
            Contract::Json(validator) => match serde_json::from_str::<Value>(text.trim()) {
                Ok(value) => validator.is_valid(&value),
                Err(_) => !eof,
            },
            Contract::Tools {
                dialect,
                definitions,
                output,
            } => {
                let parsed = (|| -> Result<_> {
                    let mut parser = ToolCallParser::new_with_output_schema(
                        *dialect,
                        "semantic-guard",
                        definitions,
                        output.as_ref().map(|(schema, _)| schema.clone()),
                    )?;
                    let mut events = parser.push(text)?;
                    if eof {
                        events.extend(parser.finish()?.0);
                    }
                    Ok(events)
                })();
                let Ok(events) = parsed else {
                    return false;
                };
                let called = events
                    .iter()
                    .any(|event| matches!(event, GeneratedOutputEvent::ToolCall(_)));
                if called {
                    return true;
                }
                if let Some((_, output)) = output {
                    if eof || JsonPrefix::scan(text.as_bytes()).complete {
                        return serde_json::from_str::<Value>(text.trim())
                            .is_ok_and(|value| output.is_valid(&value));
                    }
                }
                true
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SemanticState {
    contract: Arc<SemanticContract>,
    bytes: Vec<u8>,
    token_ends: Vec<usize>,
}

impl SemanticState {
    pub(crate) fn new(contract: Arc<SemanticContract>) -> Self {
        Self {
            contract,
            bytes: Vec::new(),
            token_ends: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
        self.token_ends.push(self.bytes.len());
    }

    pub(crate) fn rollback(&mut self, tokens: usize) {
        self.token_ends
            .truncate(self.token_ends.len().saturating_sub(tokens));
        self.bytes
            .truncate(self.token_ends.last().copied().unwrap_or(0));
    }

    pub(crate) fn accepting(&self) -> bool {
        self.contract
            .text(&self.bytes)
            .ok()
            .flatten()
            .is_some_and(|text| self.contract.accepts(&text, true))
    }

    pub(crate) fn prepared(&self) -> PreparedSemantic<'_> {
        let prefix = if matches!(self.contract.contract, Contract::Json(_)) {
            if self.contract.native.is_none() {
                Some(JsonPrefix::scan(&self.bytes))
            } else {
                self.contract
                    .text(&self.bytes)
                    .ok()
                    .flatten()
                    .map(|text| JsonPrefix::scan(text.as_bytes()))
            }
        } else {
            None
        };
        PreparedSemantic {
            state: self,
            prefix,
        }
    }

    pub(crate) fn rejects(&self, token: &[u8], eos: bool) -> bool {
        self.prepared().rejects(token, eos)
    }
}

pub(crate) struct PreparedSemantic<'a> {
    state: &'a SemanticState,
    prefix: Option<JsonPrefix>,
}

impl PreparedSemantic<'_> {
    pub(crate) fn rejects(&self, token: &[u8], eos: bool) -> bool {
        let state = self.state;
        if eos {
            return !state.accepting();
        }
        match state.contract.contract {
            Contract::Json(_) => {
                let Some(mut prefix) = self.prefix else {
                    return false;
                };
                prefix.extend(token);
                if !prefix.complete {
                    return false;
                }
            }
            Contract::Tools { dialect, .. } => {
                let marker = match dialect {
                    ToolDialect::Qwen35 | ToolDialect::MiniCpmV46 | ToolDialect::Glm => {
                        b"</tool_call>".as_slice()
                    }
                    ToolDialect::MiniCpm5 => b"</function>".as_slice(),
                    ToolDialect::Gemma => b"<tool_call|>".as_slice(),
                    ToolDialect::Llama => b"}".as_slice(),
                };
                if !token.contains(marker.last().expect("nonempty marker")) {
                    return false;
                }
                let tail = &state.bytes[state.bytes.len().saturating_sub(marker.len())..];
                let mut boundary = tail.to_vec();
                boundary.extend_from_slice(token);
                if !boundary
                    .windows(marker.len())
                    .any(|window| window == marker)
                {
                    return false;
                }
            }
        }
        let mut candidate = state.bytes.clone();
        candidate.extend_from_slice(token);
        let Ok(Some(text)) = state.contract.text(&candidate) else {
            return false;
        };
        !state.contract.accepts(&text, false)
    }
}

/// Structural completion only; semantic validation is always delegated to the
/// production validator. Scalars that can still grow (numbers) wait for EOS.
#[derive(Default, Clone, Copy)]
struct JsonPrefix {
    depth: usize,
    string: bool,
    escape: bool,
    root_string: bool,
    started: bool,
    complete: bool,
}

impl JsonPrefix {
    fn scan(bytes: &[u8]) -> Self {
        let mut state = Self::default();
        state.extend(bytes);
        state
    }

    fn extend(&mut self, bytes: &[u8]) {
        let state = self;
        for &byte in bytes {
            if state.complete {
                break;
            }
            if !state.started {
                if byte.is_ascii_whitespace() {
                    continue;
                }
                state.started = true;
                state.root_string = byte == b'"';
            }
            if state.string {
                if state.escape {
                    state.escape = false;
                } else if byte == b'\\' {
                    state.escape = true;
                } else if byte == b'"' {
                    state.string = false;
                    if state.root_string {
                        state.complete = true;
                    }
                }
                continue;
            }
            match byte {
                b'"' => state.string = true,
                b'{' | b'[' => state.depth += 1,
                b'}' | b']' => {
                    state.depth = state.depth.saturating_sub(1);
                    state.complete = state.depth == 0;
                }
                _ => {}
            }
        }
    }
}
