//! HTTP client for OpenAI-compatible `/v1/chat/completions` + hand-rolled SSE parser.
//!
//! `run_chat_completion` issues a streaming request, parses each SSE event, tracks
//! first-token time (first chunk with non-empty `delta.content`), counts content
//! chunks, captures `finish_reason` and the optional `usage` block (sent by the
//! server when `stream_options.include_usage = true`).
//!
//! The per-event handling is split out into `pub(crate) fn process_event` so unit
//! tests drive the parser without an HTTP server.

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct RequestTimings {
    pub start: Instant,
    pub first_token: Option<Instant>,
    pub token_times: Vec<Instant>,
    pub end: Instant,
}

impl RequestTimings {
    /// Time-to-first-token. Falls back to E2E if no token was emitted.
    pub fn ttft(&self) -> Duration {
        self.first_token
            .unwrap_or(self.end)
            .duration_since(self.start)
    }
    pub fn e2e(&self) -> Duration {
        self.end.duration_since(self.start)
    }
    /// Generation duration: end - first_token. Falls back to e2e if no token was emitted.
    pub fn gen_duration(&self) -> Duration {
        self.end
            .duration_since(self.first_token.unwrap_or(self.start))
    }
}

#[derive(Debug, Clone)]
pub struct RequestResult {
    pub timings: RequestTimings,
    /// Authoritative token count from server `usage` (preferred over local count).
    pub server_prompt_tokens: Option<u32>,
    pub server_completion_tokens: Option<u32>,
    /// omlx-specific extension; absent on stock OpenAI.
    pub server_cached_tokens: Option<u32>,
    /// Local fallback count of SSE chunks with non-empty `delta.content`.
    pub chunk_count: u32,
    pub finish_reason: String,
    #[allow(dead_code)]
    pub content_chars: usize,
    /// Server-emitted X-Ironmlx-Request-Id header value.
    /// `None` when `--capture-server-request-id` flag is off OR the server
    /// did not emit the header (e.g., legacy server build).
    pub request_id: Option<String>,
}

/// Parse-state mutated by `process_event`. Owned by the request loop.
#[derive(Debug, Default)]
pub struct ParseState {
    pub first_token: Option<Instant>,
    pub token_times: Vec<Instant>,
    pub chunk_count: u32,
    pub content_chars: usize,
    pub finish_reason: Option<String>,
    pub last_usage: Option<UsageBlock>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageBlock {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct SseChunk {
    #[serde(default)]
    choices: Vec<SseChoice>,
    #[serde(default)]
    usage: Option<UsageBlock>,
}

#[derive(Deserialize)]
struct SseChoice {
    #[serde(default)]
    delta: SseDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct SseDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Process one SSE event payload (the JSON body after `data: `, with `[DONE]`
/// already filtered out). Returns silently on malformed JSON (does not panic).
///
/// `now` is injected so tests can supply a deterministic first-token instant.
pub(crate) fn process_event(state: &mut ParseState, payload: &str, now: Instant) {
    let parsed: SseChunk = match serde_json::from_str(payload) {
        Ok(p) => p,
        Err(_) => return, // malformed JSON: skip silently
    };
    if let Some(c) = parsed.choices.first() {
        if let Some(content) = &c.delta.content {
            if !content.is_empty() {
                if state.first_token.is_none() {
                    state.first_token = Some(now);
                }
                state.token_times.push(now);
                state.chunk_count += 1;
                state.content_chars += content.chars().count();
            }
        }
        if let Some(reason) = &c.finish_reason {
            state.finish_reason = Some(reason.clone());
        }
    }
    if let Some(u) = parsed.usage {
        state.last_usage = Some(u);
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    stream_options: StreamOptions,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

// Qwen3+ chat template gates "thinking mode" via this kwarg. With thinking mode
// enabled, omlx buffers the entire <think>...</think> block into a single SSE
// event, which collapses gen_duration to ~0 and makes TG tok/s meaningless.
// Force it off so both engines stream token-by-token under the same protocol.
#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

/// Send one streaming chat completion request and return timing + token counts.
pub async fn run_chat_completion(
    client: &reqwest::Client,
    target_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
    capture_request_id: bool,
) -> Result<RequestResult> {
    let body = ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user",
            content: prompt,
        }],
        stream: true,
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        stream_options: StreamOptions {
            include_usage: true,
        },
        chat_template_kwargs: ChatTemplateKwargs {
            enable_thinking: false,
        },
    };

    let start = Instant::now();
    let resp = client
        .post(format!("{target_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("send to {target_url}: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body_txt = resp.text().await.unwrap_or_default();
        bail!("{target_url}: HTTP {status} — {body_txt}");
    }

    let request_id = if capture_request_id {
        resp.headers()
            .get("X-Ironmlx-Request-Id")
            .and_then(|v| v.to_str().ok())
            .map(String::from)
    } else {
        None
    };

    let mut state = ParseState::default();
    let mut buffer = String::new();

    use futures::StreamExt;
    let mut byte_stream = resp.bytes_stream();
    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.map_err(|e| anyhow!("byte_stream: {e}"))?;
        let s = std::str::from_utf8(&chunk).map_err(|e| anyhow!("non-utf8 SSE chunk: {e}"))?;
        buffer.push_str(s);
        // SSE event separator is "\n\n"; drain complete events from buffer.
        while let Some(end_idx) = buffer.find("\n\n") {
            let event = buffer[..end_idx].to_string();
            buffer.drain(..end_idx + 2);
            // Each event may contain multiple "data: " lines; OpenAI uses one.
            for line in event.lines() {
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    continue;
                }
                process_event(&mut state, payload, Instant::now());
            }
        }
    }
    let end = Instant::now();

    Ok(RequestResult {
        timings: RequestTimings {
            start,
            first_token: state.first_token,
            token_times: state.token_times,
            end,
        },
        server_prompt_tokens: state.last_usage.as_ref().and_then(|u| u.prompt_tokens),
        server_completion_tokens: state.last_usage.as_ref().and_then(|u| u.completion_tokens),
        server_cached_tokens: state.last_usage.as_ref().and_then(|u| u.cached_tokens),
        chunk_count: state.chunk_count,
        finish_reason: state.finish_reason.unwrap_or_else(|| "unknown".into()),
        content_chars: state.content_chars,
        request_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn parser_handles_role_chunk_then_content() {
        let mut state = ParseState::default();
        let now = t0();
        // role chunk first (empty content) — should NOT set first_token
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
            now,
        );
        assert!(
            state.first_token.is_none(),
            "role-chunk must not trigger first_token"
        );
        assert_eq!(state.chunk_count, 0);

        // 3 content chunks — each counted, first one sets first_token
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"Hello"}}]}"#,
            now,
        );
        assert!(
            state.first_token.is_some(),
            "first content chunk sets first_token"
        );
        assert_eq!(state.chunk_count, 1);
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":" world"}}]}"#,
            now,
        );
        assert_eq!(state.chunk_count, 2);
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}"#,
            now,
        );
        assert_eq!(state.chunk_count, 3);
        assert_eq!(state.finish_reason.as_deref(), Some("stop"));

        // usage chunk emitted at end — captures usage
        process_event(
            &mut state,
            r#"{"usage":{"prompt_tokens":12,"completion_tokens":3}}"#,
            now,
        );
        assert_eq!(
            state.last_usage.as_ref().and_then(|u| u.prompt_tokens),
            Some(12)
        );
        assert_eq!(
            state.last_usage.as_ref().and_then(|u| u.completion_tokens),
            Some(3)
        );
        assert_eq!(state.content_chars, "Hello world!".chars().count());
    }

    #[test]
    fn parser_skips_malformed_payload() {
        let mut state = ParseState::default();
        let now = t0();
        process_event(&mut state, "{garbage", now);
        // No panic, no state change beyond the no-op.
        assert_eq!(state.chunk_count, 0);
        assert!(state.finish_reason.is_none());

        // Subsequent valid event still processed.
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"ok"}}]}"#,
            now,
        );
        assert_eq!(state.chunk_count, 1);
    }

    #[test]
    fn parser_captures_cached_tokens_extension() {
        // omlx-specific cached_tokens field — make sure it round-trips.
        let mut state = ParseState::default();
        let now = t0();
        process_event(
            &mut state,
            r#"{"usage":{"prompt_tokens":50,"completion_tokens":10,"cached_tokens":40}}"#,
            now,
        );
        assert_eq!(
            state.last_usage.as_ref().and_then(|u| u.cached_tokens),
            Some(40)
        );
    }
}
