use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;
use crate::engine_pool::EngineEntry;
use crate::state::AppState;
use ironmlx_core::generate::{ChatMessage as CoreChatMessage, ChatTemplate, SamplerConfig};
use ironmlx_core::media::ProcessedMedia;

// ── Thinking Mode ───────────────────────────────────────────────────────────

/// Split generated text into (content, reasoning_content).
/// Extracts text between `<think>` and `</think>` as reasoning_content.
fn split_thinking(text: &str) -> (String, Option<String>) {
    if let Some(think_start) = text.find("<think>") {
        let after_tag = think_start + "<think>".len();
        if let Some(think_end) = text.find("</think>") {
            let reasoning = text[after_tag..think_end].trim().to_string();
            let content = text[think_end + "</think>".len()..].trim().to_string();
            let reasoning_opt = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            };
            (content, reasoning_opt)
        } else {
            // <think> opened but not closed — entire text is reasoning (still generating)
            let reasoning = text[after_tag..].trim().to_string();
            (String::new(), Some(reasoning))
        }
    } else {
        (text.to_string(), None)
    }
}

/// Tracks `<think>...</think>` state across streamed tokens.
struct ThinkingState {
    buffer: String,
    in_thinking: bool,
    thinking_done: bool,
    disabled: bool,
}

impl ThinkingState {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            in_thinking: false,
            thinking_done: false,
            disabled: false,
        }
    }

    fn new_disabled() -> Self {
        Self {
            buffer: String::new(),
            in_thinking: false,
            thinking_done: true,
            disabled: true,
        }
    }

    /// Process a new token. Returns (content, reasoning_content) to send in the SSE delta.
    /// At most one of the two will be Some.
    fn process_token(&mut self, token: &str) -> (Option<String>, Option<String>) {
        if self.disabled || self.thinking_done {
            // After </think>, everything is content
            return (Some(token.to_string()), None);
        }

        self.buffer.push_str(token);

        // Check for <think> tag
        if !self.in_thinking {
            if let Some(pos) = self.buffer.find("<think>") {
                self.in_thinking = true;
                let before = self.buffer[..pos].to_string();
                self.buffer = self.buffer[pos + "<think>".len()..].to_string();
                let content = if before.trim().is_empty() {
                    None
                } else {
                    Some(before)
                };
                // Check if buffer already contains </think>
                if let Some(end_pos) = self.buffer.find("</think>") {
                    let reasoning = self.buffer[..end_pos].to_string();
                    self.buffer = self.buffer[end_pos + "</think>".len()..].to_string();
                    self.in_thinking = false;
                    self.thinking_done = true;
                    let r = if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    };
                    return (content, r);
                }
                return (content, None);
            }
            // No <think> found yet — hold up to 7 chars for partial "<think>" match
            if self.buffer.chars().count() > 7 && !self.buffer.contains('<') {
                let flushed = self.buffer.clone();
                self.buffer.clear();
                return (Some(flushed), None);
            }
            return (None, None);
        }

        // Inside <think>, look for </think>
        if let Some(end_pos) = self.buffer.find("</think>") {
            let reasoning = self.buffer[..end_pos].to_string();
            let after = self.buffer[end_pos + "</think>".len()..].to_string();
            self.buffer.clear();
            self.in_thinking = false;
            self.thinking_done = true;
            let r = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            };
            let c = if after.is_empty() { None } else { Some(after) };
            return (c, r);
        }

        // Still inside thinking — emit reasoning incrementally
        // Keep last 8 chars in buffer in case </think> spans tokens
        let char_count = self.buffer.chars().count();
        if char_count > 8 {
            let emit_count = char_count - 8;
            let emit_byte_len: usize = self
                .buffer
                .chars()
                .take(emit_count)
                .map(|c| c.len_utf8())
                .sum();
            let emit = self.buffer[..emit_byte_len].to_string();
            self.buffer = self.buffer[emit_byte_len..].to_string();
            return (None, Some(emit));
        }

        (None, None)
    }

    /// Flush any remaining buffered content (call when stream ends).
    fn flush(&mut self) -> (Option<String>, Option<String>) {
        if self.buffer.is_empty() {
            return (None, None);
        }
        let remaining = std::mem::take(&mut self.buffer);
        if self.disabled {
            // Thinking disabled — all content
            (Some(remaining), None)
        } else if self.in_thinking || !self.thinking_done {
            // Still in thinking or never entered — emit as reasoning
            (None, Some(remaining))
        } else {
            (Some(remaining), None)
        }
    }
}

// ── Tool Calling ────────────────────────────────────────────────────────────

/// Parse tool calls from model-generated text.
/// Supports Qwen-style `<tool_call>...</tool_call>` format.
/// Returns (remaining_content, tool_calls) if tool calls found.
fn parse_tool_calls_from_text(text: &str) -> (String, Option<Vec<ToolCall>>) {
    let mut tool_calls = Vec::new();
    let mut remaining = text.to_string();

    // Try Qwen-style <tool_call> tags
    let mut search_text = text;
    while let Some(start) = search_text.find("<tool_call>") {
        if let Some(end) = search_text[start..].find("</tool_call>") {
            let end_abs = start + end;
            let json_str = search_text[start + "<tool_call>".len()..end_abs].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str)
                && let (Some(name), Some(args)) = (
                    parsed.get("name").and_then(|n| n.as_str()),
                    parsed.get("arguments"),
                )
            {
                let uuid_str = uuid::Uuid::new_v4().to_string().replace('-', "");
                tool_calls.push(ToolCall {
                    id: format!("call_{}", &uuid_str[..24]),
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: args.to_string(),
                    },
                });
            }
            search_text = &search_text[end_abs + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    if !tool_calls.is_empty() {
        // Strip <tool_call>...</tool_call> blocks from remaining content
        while let Some(start) = remaining.find("<tool_call>") {
            if let Some(end) = remaining[start..].find("</tool_call>") {
                let end_abs = start + end + "</tool_call>".len();
                remaining = format!("{}{}", &remaining[..start], &remaining[end_abs..]);
            } else {
                break;
            }
        }
        remaining = remaining.trim().to_string();
        return (remaining, Some(tool_calls));
    }

    (remaining, None)
}

/// Inject tool definitions into the message list as system instructions.
fn inject_tools_into_messages(
    messages: &[ChatMessage],
    tools: &Option<Vec<ToolDefinition>>,
) -> Vec<ChatMessage> {
    let Some(tools) = tools else {
        return messages.to_vec();
    };
    if tools.is_empty() {
        return messages.to_vec();
    }

    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
    let tools_instruction = format!(
        "You have access to the following tools:\n{}\n\n\
         To call a tool, respond with a <tool_call> tag containing a JSON object with \"name\" and \"arguments\" keys.\n\
         Example: <tool_call>\n{{\"name\": \"function_name\", \"arguments\": {{\"param\": \"value\"}}}}\n</tool_call>",
        tools_json
    );

    let mut new_messages = Vec::with_capacity(messages.len() + 1);
    let mut has_system = false;
    for msg in messages {
        if msg.role == "system" && !has_system {
            has_system = true;
            let original_text = msg.content.as_text();
            new_messages.push(ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(format!(
                    "{}\n\n{}",
                    original_text, tools_instruction
                )),
                tool_call_id: None,
                tool_calls: None,
            });
        } else {
            new_messages.push(msg.clone());
        }
    }
    if !has_system {
        new_messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(tools_instruction),
                tool_call_id: None,
                tool_calls: None,
            },
        );
    }
    new_messages
}

/// POST /v1/chat/completions — supports both regular and streaming responses
/// Load saved model params and apply as defaults to the request.
fn apply_model_param_defaults(req: &mut ChatCompletionRequest) {
    let model_id = req.model.as_deref().unwrap_or("default");
    let params_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ironmlx")
        .join("config")
        .join("model_params.json");
    if !params_path.exists() {
        return;
    }
    let data = match std::fs::read_to_string(&params_path) {
        Ok(d) => d,
        Err(_) => return,
    };
    let all_params: serde_json::Map<String, serde_json::Value> = match serde_json::from_str(&data) {
        Ok(m) => m,
        Err(_) => return,
    };
    let params = match all_params.get(model_id) {
        Some(p) => p,
        None => return,
    };

    // Apply saved defaults only when request uses default values
    if let Some(mt) = params.get("max_tokens").and_then(|v| v.as_str())
        && let Ok(mt_val) = mt.parse::<usize>()
        && mt_val > 0
        && req.max_tokens == default_max_tokens()
    {
        req.max_tokens = mt_val;
    }
    if let Some(temp) = params.get("temperature").and_then(|v| v.as_str())
        && let Ok(temp_val) = temp.parse::<f32>()
        && req.temperature == default_temperature()
    {
        req.temperature = temp_val;
    }
    if let Some(tp) = params.get("top_p").and_then(|v| v.as_str())
        && let Ok(tp_val) = tp.parse::<f32>()
        && req.top_p == default_top_p()
    {
        req.top_p = tp_val;
    }
    if let Some(tk) = params.get("top_k").and_then(|v| v.as_str())
        && let Ok(tk_val) = tk.parse::<i32>()
        && req.top_k == default_top_k()
    {
        req.top_k = tk_val;
    }
    if let Some(rp) = params.get("repeat_penalty").and_then(|v| v.as_str())
        && let Ok(rp_val) = rp.parse::<f32>()
        && req.repetition_penalty == default_repetition_penalty()
    {
        req.repetition_penalty = rp_val;
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Apply saved model parameter defaults
    apply_model_param_defaults(&mut req);

    state.log_buffer.push(
        "info",
        &format!(
            "Chat request: model={}, stream={}, max_tokens={}, temp={}",
            req.model.as_deref().unwrap_or("default"),
            req.stream,
            req.max_tokens,
            req.temperature
        ),
    );
    if req.stream {
        chat_completions_stream(state, req).await
    } else {
        chat_completions_sync(state, req)
            .await
            .map(|j| j.into_response())
    }
}

async fn chat_completions_sync(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let media = extract_and_process_media(&entry, &req.messages)?;
    let messages = inject_tools_into_messages(&req.messages, &req.tools);
    let prompt = build_chat_prompt(&entry.chat_template, &messages, req.enable_thinking)?;

    // Pre-tokenization length check to prevent OOM during encode
    check_raw_prompt_length(&prompt, &entry.model_id)?;

    let prompt_tokens = encode_or_error(&entry, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;
    let model_id = entry.model_id.clone();

    // Check context window limit
    check_prompt_length(prompt_len, max_tokens, &model_id)?;

    let eos = entry.eos_token_id;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let mut token_rx = entry
        .engine
        .add_request(
            request_id.clone(),
            prompt_tokens,
            sampler,
            max_tokens,
            eos,
            media,
        )
        .await;

    let mut generated_text = String::new();
    let mut completion_len = 0usize;
    let mut finish_reason = "stop".to_string();

    while let Some(output) = token_rx.recv().await {
        if let Some(ref reason) = output.finish_reason {
            finish_reason = reason.clone();
            break;
        }
        generated_text.push_str(&output.token_text);
        completion_len += 1;
    }

    let (content, reasoning_content) = if req.enable_thinking.unwrap_or(false) {
        split_thinking(&generated_text)
    } else {
        (generated_text.clone(), None)
    };

    // Check for tool calls if tools were provided
    let (final_content, tool_calls) = if req.tools.is_some() {
        parse_tool_calls_from_text(&content)
    } else {
        (content, None)
    };

    let finish_reason = if tool_calls.is_some() {
        "tool_calls".to_string()
    } else {
        finish_reason
    };

    let resp = Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: final_content,
                reasoning_content,
                tool_calls,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    }));

    state.total_tokens.fetch_add(
        (prompt_len + completion_len) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

    resp
}

async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let media = extract_and_process_media(&entry, &req.messages)?;
    let messages = inject_tools_into_messages(&req.messages, &req.tools);
    let prompt = build_chat_prompt(&entry.chat_template, &messages, req.enable_thinking)?;

    // Pre-tokenization length check to prevent OOM during encode
    check_raw_prompt_length(&prompt, &entry.model_id)?;

    let prompt_tokens = encode_or_error(&entry, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = build_sampler(&req);
    let max_tokens = req.max_tokens;
    let model_id_check = entry.model_id.clone();

    // Check context window limit
    check_prompt_length(prompt_len, max_tokens, &model_id_check)?;

    let eos = entry.eos_token_id;
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_id = entry.model_id.clone();

    let mut token_rx = entry
        .engine
        .add_request(
            chunk_id.clone(),
            prompt_tokens,
            sampler,
            max_tokens,
            eos,
            media,
        )
        .await;

    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let chunk_id_clone = chunk_id.clone();
    let model_id_clone = model_id.clone();
    tokio::spawn(async move {
        // Send initial chunk with role
        let role_chunk = ChatCompletionChunk {
            id: chunk_id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id_clone.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        let _ = sse_tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&role_chunk).unwrap())
            ))
            .await;

        // Stream tokens with thinking mode tracking
        let enable_thinking = req.enable_thinking.unwrap_or(false);
        let mut thinking_state = if enable_thinking {
            ThinkingState::new()
        } else {
            ThinkingState::new_disabled()
        };

        while let Some(output) = token_rx.recv().await {
            if let Some(ref reason) = output.finish_reason {
                // Flush remaining buffered thinking content before final chunk
                let (flush_c, flush_r) = thinking_state.flush();
                if flush_c.is_some() || flush_r.is_some() {
                    let flush_chunk = ChatCompletionChunk {
                        id: chunk_id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: flush_c,
                                reasoning_content: flush_r,
                                tool_calls: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = sse_tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&flush_chunk).unwrap())
                        ))
                        .await;
                }

                let final_chunk = ChatCompletionChunk {
                    id: chunk_id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_id_clone.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(reason.clone()),
                    }],
                };
                let _ = sse_tx
                    .send(Ok(
                        Event::default().data(serde_json::to_string(&final_chunk).unwrap())
                    ))
                    .await;
                break;
            }

            // Route token to content or reasoning_content based on <think> state
            let (content, reasoning_content) = thinking_state.process_token(&output.token_text);

            // Skip empty deltas
            if content.is_none() && reasoning_content.is_none() {
                continue;
            }

            let chunk = ChatCompletionChunk {
                id: chunk_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id_clone.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content,
                        reasoning_content,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            if sse_tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ))
                .await
                .is_err()
            {
                break; // Client disconnected
            }
        }

        // Flush remaining buffered thinking content
        let (flush_content, flush_reasoning) = thinking_state.flush();
        if flush_content.is_some() || flush_reasoning.is_some() {
            let chunk = ChatCompletionChunk {
                id: chunk_id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id_clone.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: flush_content,
                        reasoning_content: flush_reasoning,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            let _ = sse_tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                ))
                .await;
        }

        // Send [DONE] marker
        let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let stream = ReceiverStream::new(sse_rx);
    Ok(Sse::new(stream).into_response())
}

// ── Anthropic Messages API ──────────────────────────────────────────────────

/// Convert Anthropic messages to internal ChatMessage format.
fn anthropic_to_chat_messages(req: &AnthropicMessagesRequest) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // System prompt is a top-level field in Anthropic API
    if let Some(ref system) = req.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(system.clone()),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    for msg in &req.messages {
        messages.push(ChatMessage {
            role: msg.role.clone(),
            content: MessageContent::Text(msg.content.as_text()),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    messages
}

/// Map internal finish_reason to Anthropic stop_reason.
fn to_anthropic_stop_reason(reason: &str) -> String {
    match reason {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        other => other.to_string(),
    }
}

/// POST /v1/messages — Anthropic-compatible Messages API
pub async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    if req.stream {
        anthropic_messages_stream(state, req).await
    } else {
        anthropic_messages_sync(state, req)
            .await
            .map(|j| j.into_response())
    }
}

async fn anthropic_messages_sync(
    state: Arc<AppState>,
    req: AnthropicMessagesRequest,
) -> Result<Json<AnthropicMessagesResponse>, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let messages = anthropic_to_chat_messages(&req);
    let prompt = build_chat_prompt(&entry.chat_template, &messages, None)?;
    let prompt_tokens = encode_or_error(&entry, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };
    let max_tokens = req.max_tokens;
    let eos = entry.eos_token_id;
    let request_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model_id = entry.model_id.clone();

    let mut token_rx = entry
        .engine
        .add_request(
            request_id.clone(),
            prompt_tokens,
            sampler,
            max_tokens,
            eos,
            None,
        )
        .await;

    let mut generated_text = String::new();
    let mut completion_len = 0usize;
    let mut finish_reason = "stop".to_string();

    while let Some(output) = token_rx.recv().await {
        if let Some(ref reason) = output.finish_reason {
            finish_reason = reason.clone();
            break;
        }
        generated_text.push_str(&output.token_text);
        completion_len += 1;
    }

    // Strip thinking tags — Anthropic response uses content blocks only
    let (content, _reasoning) = split_thinking(&generated_text);

    Ok(Json(AnthropicMessagesResponse {
        id: request_id,
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![AnthropicContentBlock::Text { text: content }],
        model: model_id,
        stop_reason: to_anthropic_stop_reason(&finish_reason),
        usage: AnthropicUsage {
            input_tokens: prompt_len,
            output_tokens: completion_len,
        },
    }))
}

async fn anthropic_messages_stream(
    state: Arc<AppState>,
    req: AnthropicMessagesRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let messages = anthropic_to_chat_messages(&req);
    let prompt = build_chat_prompt(&entry.chat_template, &messages, None)?;
    let prompt_tokens = encode_or_error(&entry, &prompt)?;
    let prompt_len = prompt_tokens.len();
    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };
    let max_tokens = req.max_tokens;
    let eos = entry.eos_token_id;
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model_id = entry.model_id.clone();

    let mut token_rx = entry
        .engine
        .add_request(
            msg_id.clone(),
            prompt_tokens,
            sampler,
            max_tokens,
            eos,
            None,
        )
        .await;

    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::spawn(async move {
        // 1) message_start
        let msg_start = AnthropicMessageStart {
            r#type: "message_start".to_string(),
            message: AnthropicMessageStartPayload {
                id: msg_id.clone(),
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model_id.clone(),
                usage: AnthropicUsage {
                    input_tokens: prompt_len,
                    output_tokens: 0,
                },
            },
        };
        let _ = sse_tx
            .send(Ok(Event::default()
                .event("message_start")
                .data(serde_json::to_string(&msg_start).unwrap())))
            .await;

        // 2) content_block_start
        let block_start = AnthropicContentBlockStart {
            r#type: "content_block_start".to_string(),
            index: 0,
            content_block: AnthropicContentBlock::Text {
                text: String::new(),
            },
        };
        let _ = sse_tx
            .send(Ok(Event::default()
                .event("content_block_start")
                .data(serde_json::to_string(&block_start).unwrap())))
            .await;

        // 3) content_block_delta for each token (thinking disabled for Anthropic endpoint)
        let mut thinking_state = ThinkingState::new_disabled();
        let mut output_tokens = 0usize;
        let mut finish_reason = "end_turn".to_string();

        while let Some(output) = token_rx.recv().await {
            if let Some(ref reason) = output.finish_reason {
                finish_reason = to_anthropic_stop_reason(reason);
                break;
            }
            output_tokens += 1;

            let (content, _reasoning) = thinking_state.process_token(&output.token_text);
            // Only emit content (skip reasoning for Anthropic format)
            if let Some(text) = content
                && !text.is_empty()
            {
                let delta = AnthropicContentBlockDelta {
                    r#type: "content_block_delta".to_string(),
                    index: 0,
                    delta: AnthropicTextDelta {
                        r#type: "text_delta".to_string(),
                        text,
                    },
                };
                if sse_tx
                    .send(Ok(Event::default()
                        .event("content_block_delta")
                        .data(serde_json::to_string(&delta).unwrap())))
                    .await
                    .is_err()
                {
                    return; // Client disconnected
                }
            }
        }

        // 4) content_block_stop
        let block_stop = AnthropicContentBlockStop {
            r#type: "content_block_stop".to_string(),
            index: 0,
        };
        let _ = sse_tx
            .send(Ok(Event::default()
                .event("content_block_stop")
                .data(serde_json::to_string(&block_stop).unwrap())))
            .await;

        // 5) message_delta
        let msg_delta = AnthropicMessageDelta {
            r#type: "message_delta".to_string(),
            delta: AnthropicMessageDeltaPayload {
                stop_reason: finish_reason,
            },
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens,
            },
        };
        let _ = sse_tx
            .send(Ok(Event::default()
                .event("message_delta")
                .data(serde_json::to_string(&msg_delta).unwrap())))
            .await;

        // 6) message_stop
        let msg_stop = AnthropicMessageStop {
            r#type: "message_stop".to_string(),
        };
        let _ = sse_tx
            .send(Ok(Event::default()
                .event("message_stop")
                .data(serde_json::to_string(&msg_stop).unwrap())))
            .await;
    });

    let stream = ReceiverStream::new(sse_rx);
    Ok(Sse::new(stream).into_response())
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let prompt_tokens = encode_or_error(&entry, &req.prompt)?;
    let prompt_len = prompt_tokens.len();
    let eos = entry.eos_token_id;
    let sampler = SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerConfig::default()
    };
    let max_tokens = req.max_tokens;
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model_id = entry.model_id.clone();

    let mut token_rx = entry
        .engine
        .add_request(
            request_id.clone(),
            prompt_tokens,
            sampler,
            max_tokens,
            eos,
            None,
        )
        .await;

    let mut text = String::new();
    let mut completion_len = 0usize;
    let mut finish_reason = "stop".to_string();

    while let Some(output) = token_rx.recv().await {
        if let Some(ref reason) = output.finish_reason {
            finish_reason = reason.clone();
            break;
        }
        text.push_str(&output.token_text);
        completion_len += 1;
    }

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    }))
}

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let model_ids = state.pool.list_models();
    let data = model_ids
        .into_iter()
        .map(|id| ModelInfo {
            id,
            object: "model".to_string(),
            created: 0,
            owned_by: "local".to_string(),
        })
        .collect();
    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

/// GET /health
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded = state.pool.list_models();
    let model = if loaded.is_empty() {
        "none".to_string()
    } else {
        loaded.join(", ")
    };

    let mem = ironmlx_core::memory::get_active_memory().unwrap_or(0);
    let cache = ironmlx_core::memory::get_cache_memory().unwrap_or(0);
    let peak = ironmlx_core::memory::get_peak_memory().unwrap_or(0);
    let total = ironmlx_core::memory::get_memory_size().map(|bytes| bytes as f64 / 1_048_576.0);
    let max_rec =
        ironmlx_core::memory::get_max_recommended_memory().map(|bytes| bytes as f64 / 1_048_576.0);

    let device_name = ironmlx_core::memory::get_device_name();
    let cache_stats = ironmlx_core::cache::CacheStats::current();
    let total_tokens = state
        .total_tokens
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(HealthResponse {
        status: "ok".to_string(),
        model,
        memory: Some(MemoryInfo {
            active_mb: mem as f64 / 1_048_576.0,
            cache_mb: cache as f64 / 1_048_576.0,
            peak_mb: peak as f64 / 1_048_576.0,
            total_mb: total,
            max_mb: max_rec,
        }),
        started_at: state.started_at,
        total_tokens,
        cached_tokens: cache_stats.cached_tokens,
        cache_hit_rate: cache_stats.hit_rate(),
        device_name,
    })
}

// ── Engine Pool Management ──────────────────────────────────────────────────

/// POST /v1/models/load — load a model by directory path
pub async fn load_model_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorResponse>)> {
    state
        .log_buffer
        .push("info", &format!("Loading model: {}", req.model_dir));
    let (hot_bytes, cold_bytes, cache_dir, max_seqs) = {
        let cfg = state.config.read().unwrap();
        (
            cfg.hot_cache_max_size_bytes(),
            cfg.cold_cache_max_size_bytes(),
            cfg.cache_dir.clone(),
            cfg.max_num_seqs,
        )
    };
    let model_id = state
        .pool
        .load_model(
            &req.model_dir,
            hot_bytes,
            cold_bytes,
            cache_dir.as_deref(),
            max_seqs,
            0, // auto-calculate init_cache_blocks
        )
        .map_err(|e| {
            state
                .log_buffer
                .push("error", &format!("Model load failed: {}", e));
            internal_error(&e)
        })?;
    state
        .log_buffer
        .push("info", &format!("Model loaded: {}", model_id));

    // Sync to app config so menubar shows correct model
    state.sync_default_model(&model_id);

    Ok(Json(ModelInfo {
        id: model_id,
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "local".to_string(),
    }))
}

/// POST /v1/models/unload — unload a model
pub async fn unload_model_endpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnloadModelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state.pool.unload_model(&req.model).map_err(|e| {
        state
            .log_buffer
            .push("error", &format!("Model unload failed: {}", e));
        internal_error(&e)
    })?;
    state
        .log_buffer
        .push("info", &format!("Model unloaded: {}", req.model));

    Ok(Json(serde_json::json!({
        "status": "ok",
        "unloaded": req.model,
    })))
}

/// POST /v1/models/default — set default model
pub async fn set_default_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetDefaultRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .pool
        .set_default(&req.model)
        .map_err(|e| internal_error(&e))?;

    // Sync to app config so menubar shows correct model
    state.sync_default_model(&req.model);

    Ok(Json(serde_json::json!({
        "status": "ok",
        "default_model": req.model,
    })))
}

// ── Embeddings ──────────────────────────────────────────────────────────────

/// POST /v1/embeddings — OpenAI-compatible embeddings endpoint
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| internal_error(&e))?;

    let texts = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    let mut all_embeddings = Vec::new();
    let mut total_tokens = 0usize;

    for (i, text) in texts.iter().enumerate() {
        let token_ids = entry
            .tokenizer
            .encode(text.as_str())
            .map_err(|e| internal_error(&format!("tokenize error: {}", e)))?;
        total_tokens += token_ids.len();

        // TODO: Call embedding model encode
        // Placeholder — will be connected when BertModel is ready
        let embedding = vec![0.0f32; 384];

        all_embeddings.push(EmbeddingData {
            object: "embedding".to_string(),
            index: i,
            embedding,
        });
    }

    let model_id = entry.model_id.clone();

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data: all_embeddings,
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<super::types::RerankRequest>,
) -> Result<Json<super::types::RerankResponse>, (StatusCode, String)> {
    let entry = state
        .pool
        .get(req.model.as_deref())
        .map_err(|e| (StatusCode::NOT_FOUND, e))?;

    let return_docs = req.return_documents.unwrap_or(false);

    let mut results = Vec::new();
    let mut total_tokens = 0usize;

    for (i, doc) in req.documents.iter().enumerate() {
        let text = doc.text();
        let pair_text = format!("{} [SEP] {}", req.query, text);
        let tokens = entry
            .tokenizer
            .encode(&pair_text)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize: {e}")))?;
        total_tokens += tokens.len();

        // TODO: call reranker_model.score() when integrated
        let score = 0.5f32;

        results.push(super::types::RerankResult {
            index: i,
            relevance_score: score,
            document: if return_docs {
                Some(super::types::RerankResultDocument {
                    text: text.to_string(),
                })
            } else {
                None
            },
        });
    }

    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(n) = req.top_n {
        results.truncate(n);
    }

    Ok(Json(super::types::RerankResponse {
        results,
        model: entry.model_id.clone(),
        usage: super::types::RerankUsage { total_tokens },
    }))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn build_chat_prompt(
    chat_template: &Option<ChatTemplate>,
    messages: &[ChatMessage],
    enable_thinking: Option<bool>,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    if let Some(ct) = chat_template {
        // Convert API ChatMessage to core ChatMessage for template rendering
        let core_messages: Vec<CoreChatMessage> = messages
            .iter()
            .map(|m| CoreChatMessage {
                role: m.role.clone(),
                content: m.content.as_text(),
            })
            .collect();
        ct.apply_with_thinking(&core_messages, true, enable_thinking)
            .map_err(|e| internal_error(&format!("chat template error: {}", e)))
    } else {
        // Fallback: simple concatenation
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!("{}: {}\n", msg.role, msg.content.as_text()));
        }
        prompt.push_str("assistant: ");
        Ok(prompt)
    }
}

fn build_sampler(req: &ChatCompletionRequest) -> SamplerConfig {
    SamplerConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        ..SamplerConfig::default()
    }
}

fn encode_or_error(
    entry: &EngineEntry,
    text: &str,
) -> Result<Vec<i32>, (StatusCode, Json<ErrorResponse>)> {
    entry
        .tokenizer
        .encode(text)
        .map_err(|e| internal_error(&format!("tokenize error: {}", e)))
}

/// Extract media items from multimodal messages, process them, and return ProcessedMedia.
/// Returns None if no media is found (pure text request).
fn extract_and_process_media(
    entry: &EngineEntry,
    messages: &[ChatMessage],
) -> Result<Option<Vec<ProcessedMedia>>, (StatusCode, Json<ErrorResponse>)> {
    let mut image_urls = Vec::new();

    for msg in messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                match part {
                    ContentPart::ImageUrl { image_url } => {
                        image_urls.push(image_url.url.clone());
                    }
                    ContentPart::VideoUrl { video_url: _ } => {
                        // TODO: video support via ffmpeg
                    }
                    ContentPart::Text { .. } => {}
                }
            }
        }
    }

    if image_urls.is_empty() {
        return Ok(None);
    }

    let patch_size = entry.patch_size;
    let spatial_merge_size = entry.spatial_merge_size;

    let mut all_media = Vec::new();
    for url in &image_urls {
        let bytes = ironmlx_core::media::loader::load_media(url)
            .map_err(|e| internal_error(&format!("failed to load image: {}", e)))?;

        let (pixel_values, h, w) = ironmlx_core::media::image_proc::process_image_bytes(
            &bytes,
            patch_size,
            spatial_merge_size,
        )
        .map_err(|e| internal_error(&format!("failed to process image: {}", e)))?;

        let patches_h = h / patch_size;
        let patches_w = w / patch_size;
        let grid_thw = vec![(1usize, patches_h, patches_w)];

        all_media.push(ProcessedMedia {
            pixel_values,
            grid_thw,
        });
    }

    Ok(Some(all_media))
}

/// Default max context size if not configured
const DEFAULT_MAX_CONTEXT: usize = 32768;

/// Get the max context size for a model from saved params, or use default.
fn get_max_context(model_id: &str) -> usize {
    let params_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ironmlx")
        .join("config")
        .join("model_params.json");
    if let Ok(data) = std::fs::read_to_string(&params_path)
        && let Ok(all) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&data)
        && let Some(params) = all.get(model_id)
        && let Some(ctx) = params.get("context_size").and_then(|v| v.as_str())
        && let Ok(val) = ctx.parse::<usize>()
        && val > 0
    {
        return val;
    }
    DEFAULT_MAX_CONTEXT
}

/// Quick pre-check on raw prompt string length before tokenization.
/// Rough estimate: 1 token ≈ 4 chars. Reject obviously too-long prompts early.
fn check_raw_prompt_length(
    prompt: &str,
    model_id: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let max_context = get_max_context(model_id);
    let estimated_tokens = prompt.len() / 3; // conservative: ~3 chars per token
    if estimated_tokens > max_context {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!(
                        "Prompt too long: estimated ~{} tokens (from {} chars) exceeds context window of {} tokens.",
                        estimated_tokens,
                        prompt.len(),
                        max_context
                    ),
                    r#type: "invalid_request_error".to_string(),
                },
            }),
        ));
    }
    Ok(())
}

/// Check if prompt tokens exceed the model's context window and return error if so.
fn check_prompt_length(
    prompt_len: usize,
    max_tokens: usize,
    model_id: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let max_context = get_max_context(model_id);
    if prompt_len + max_tokens > max_context {
        let available = max_context.saturating_sub(prompt_len);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!(
                        "Request exceeds context window: prompt_tokens={}, max_tokens={}, total={}, context_size={}. Reduce prompt length or max_tokens (max available for generation: {}).",
                        prompt_len,
                        max_tokens,
                        prompt_len + max_tokens,
                        max_context,
                        available
                    ),
                    r#type: "invalid_request_error".to_string(),
                },
            }),
        ));
    }
    if prompt_len > max_context {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!(
                        "Prompt too long: {} tokens exceeds context window of {} tokens.",
                        prompt_len, max_context
                    ),
                    r#type: "invalid_request_error".to_string(),
                },
            }),
        ));
    }
    Ok(())
}

fn internal_error(message: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: "server_error".to_string(),
            },
        }),
    )
}
