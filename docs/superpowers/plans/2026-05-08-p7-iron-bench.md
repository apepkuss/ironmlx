# P7 iron-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Rust workspace member `iron-bench` that drives any OpenAI-compatible HTTP endpoint (`/v1/chat/completions`), measures TTFT / TG decode / TPOT / PP prefill / E2E with synthetic controlled-length prompts, and reports median + p95 across N runs in Markdown / CSV / JSON for head-to-head comparison (initial use case: ironmlx vs omlx on Qwen3.5-4B-MLX-4bit single-request).

**Architecture:** New top-level workspace member `iron-bench/` (sibling of `mlx-sys`, `mlx`, `ironmlx`). Five focused source files: `main.rs` (CLI + tokio runtime), `prompt.rs` (tokenizer-aware synthetic prompt generator with per-run nonce), `client.rs` (reqwest HTTP + hand-rolled SSE parser), `runner.rs` (warmup + N timed runs per cell), `report.rs` (stats reduction + Markdown/CSV/JSON formatters). No dependency on `ironmlx` / `mlx` / `mlx-sys` crates so the harness stays engine-neutral.

**Tech Stack:** Rust 2021, workspace deps already present (tokio + reqwest 0.12 + serde + serde_json + clap + tokenizers 0.20 + anyhow + futures). Async runtime: tokio multi-thread. **Spec:** [`docs/superpowers/specs/2026-05-08-p7-iron-bench-design.md`](../specs/2026-05-08-p7-iron-bench-design.md).

---

## Conventions Recap

- **TDD per task** (where tests apply): failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before each commit** (`.claude/CLAUDE.md`):

  ```
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```

- **Test threads**: not relevant for iron-bench (no MLX GPU dispatch); standard `cargo test` is fine.
- **iron-bench has NO MLX dependency**: no `MLX_DIR` env var needed; no `mlx-sys` linkage.
- **Tokenizer fixture for tests**: `IRON_BENCH_TEST_TOKENIZER` env var pointing to a real `tokenizer.json`. When unset, tests skip (do NOT fail). Local fixture path: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/tokenizer.json`.
- **ASCII commit messages.**

---

## File Structure (after P7)

```
cxx-mlx/
├── Cargo.toml                                   # MODIFIED: workspace.members += "iron-bench"
└── iron-bench/                                  # NEW workspace member
    ├── Cargo.toml                               # NEW: binary crate manifest
    ├── README.md                                # NEW: usage + sample output
    └── src/
        ├── main.rs                              # NEW: CLI entry + tokio runtime + dispatch
        ├── prompt.rs                            # NEW: tokenizer-aware synthetic prompt generator
        ├── client.rs                            # NEW: HTTP client + SSE parser (with pub(crate) process_event helper for tests)
        ├── runner.rs                            # NEW: warmup + N timed runs per cell
        └── report.rs                            # NEW: stats reduction (median + p95) + Markdown/CSV/JSON formatters
```

---

## Task 1: Workspace member scaffold + CLI

**Files:**
- Modify: `Cargo.toml` (workspace root) — add `"iron-bench"` to `members`
- Create: `iron-bench/Cargo.toml`
- Create: `iron-bench/src/main.rs`

### Goal

Get a buildable `iron-bench` binary that parses CLI args via clap and prints a hello/dispatch summary. Locks in the dependency set + arg shape; subsequent tasks add real functionality module-by-module.

### Steps

- [ ] **Step 1.1: Add iron-bench to workspace members**

Edit [`Cargo.toml`](../../../Cargo.toml). Find the existing line:

```toml
members = ["mlx-sys", "mlx", "ironmlx"]
```

Replace with:

```toml
members = ["mlx-sys", "mlx", "ironmlx", "iron-bench"]
```

- [ ] **Step 1.2: Create `iron-bench/Cargo.toml`**

Create `iron-bench/Cargo.toml`:

```toml
[package]
name = "iron-bench"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
description = "Head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints"

[[bin]]
name = "iron-bench"
path = "src/main.rs"

[dependencies]
tokio.workspace = true
reqwest.workspace = true
futures.workspace = true
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
anyhow = "1"
```

- [ ] **Step 1.3: Create `iron-bench/src/main.rs` skeleton with full CLI args**

Create `iron-bench/src/main.rs`:

```rust
//! iron-bench — head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints.
//!
//! Drives multiple `--target name=URL` endpoints with the same synthetic-prompt matrix
//! and reports TTFT / TG decode / TPOT / PP prefill / E2E across N timed runs (median +
//! p95). Engine-neutral; no dependency on ironmlx/mlx crates.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

mod client;
mod prompt;
mod report;
mod runner;

#[derive(Parser, Debug)]
#[command(
    name = "iron-bench",
    about = "Head-to-head HTTP benchmark for OpenAI-compatible LLM endpoints",
    version
)]
struct Args {
    /// Target endpoints. Repeat for multiple targets.
    /// Format: `name=URL` (e.g., `--target ironmlx=http://localhost:8080`).
    #[arg(long, value_parser = parse_target, required = true, num_args = 1..)]
    target: Vec<(String, String)>,

    /// Path to model dir containing `tokenizer.json` (used for prompt synthesis only).
    #[arg(long)]
    model_dir: PathBuf,

    /// Model name to send in the `model` field of each JSON request.
    #[arg(long, default_value = "qwen3.5-4b")]
    model: String,

    /// Prompt token lengths to test (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![128_usize, 512, 2048])]
    prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Timed runs per cell (after warmup).
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Warmup runs per cell (excluded from stats).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,

    /// HTTP request timeout (seconds).
    #[arg(long, default_value_t = 300)]
    timeout: u64,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    Markdown,
    Csv,
    Json,
}

fn parse_target(s: &str) -> std::result::Result<(String, String), String> {
    s.split_once('=')
        .map(|(name, url)| (name.into(), url.trim_end_matches('/').into()))
        .ok_or_else(|| format!("expected name=URL, got '{s}'"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "iron-bench: {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
        args.target.len(),
        args.prompt_len,
        args.max_tokens,
        args.runs,
        args.warmup,
    );
    let _ = args; // placeholder until T2-T4 wire real dispatch

    Ok(())
}
```

- [ ] **Step 1.4: Create `prompt.rs`, `client.rs`, `runner.rs`, `report.rs` as empty stubs to satisfy `mod` references**

Create `iron-bench/src/prompt.rs`:

```rust
//! Synthetic prompt generator (tokenizer-aware). Implemented in T2.
```

Create `iron-bench/src/client.rs`:

```rust
//! HTTP client + SSE parser. Implemented in T2.
```

Create `iron-bench/src/runner.rs`:

```rust
//! Per-cell warmup + N timed runs. Implemented in T3.
```

Create `iron-bench/src/report.rs`:

```rust
//! Stats reduction + Markdown / CSV / JSON formatters. Implemented in T4.
```

- [ ] **Step 1.5: Build & smoke-test CLI**

Run:

```
cargo build --release -p iron-bench
```

Expected: clean compile.

```
cargo run --release -p iron-bench -- \
  --target a=http://localhost:1 \
  --target b=http://localhost:2 \
  --model-dir /tmp \
  --prompt-len 128,512 \
  --max-tokens 64 --runs 3 --warmup 1
```

Expected stderr line:
```
iron-bench: 2 target(s), prompt_len=[128, 512], max_tokens=64, runs=3, warmup=1
```

- [ ] **Step 1.6: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 1.7: Commit**

```bash
git add Cargo.toml iron-bench/
git commit -m "$(cat <<'EOF'
feat(iron-bench): workspace member scaffold + CLI args

New iron-bench/ workspace member alongside mlx-sys / mlx / ironmlx.
Pulls workspace deps (tokio, reqwest, futures) + serde_json, clap,
tokenizers, anyhow. main.rs parses CLI args (target=name=URL repeated,
model_dir, prompt_len comma-separated, max_tokens, runs, warmup,
format, timeout) and prints a dispatch summary. prompt/client/runner/
report modules are stubs filled in T2-T4.

iron-bench has no dependency on ironmlx/mlx/mlx-sys crates so it stays
engine-neutral and can drive any OpenAI-compatible /v1/chat/completions
endpoint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Prompt synthesis + HTTP client + SSE parser

**Files:**
- Modify: `iron-bench/src/prompt.rs` (replace stub)
- Modify: `iron-bench/src/client.rs` (replace stub)

### Goal

Build the two non-trivial primitives: (1) tokenizer-aware synthetic prompt with per-run nonce that round-trips to exactly N tokens (± small drift), and (2) HTTP `/v1/chat/completions` client that streams SSE, captures first-token timing + per-chunk content + server-side `usage` block. Extract the per-event SSE handling into a `pub(crate) fn process_event` helper so tests drive it directly without spinning up an HTTP server.

### Steps

- [ ] **Step 2.1: Implement `prompt.rs` — synthesize_prompt + 2 unit tests**

Replace `iron-bench/src/prompt.rs` with:

```rust
//! Synthetic prompt generator — tokenizer-aware, with per-run nonce.
//!
//! The output prompt encodes to exactly `target_tokens` tokens (± small
//! BPE round-trip drift) on the same tokenizer. The nonce prevents
//! prefix-cache hits across runs (omlx defaults to a tiered prefix cache;
//! without nonce, the second run's prefill would be ~0ms — invalidating
//! PP measurement).

use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

/// Synthesize a prompt that encodes to (approximately) `target_tokens` tokens.
///
/// Returns `(prompt_text, actual_token_count_local)`. The actual count is the
/// post-round-trip local tokenizer count; small BPE drift (±2 tokens) is
/// tolerated. Authoritative server-side count comes from the response
/// `usage.prompt_tokens` field if available.
pub fn synthesize_prompt(
    tokenizer: &Tokenizer,
    target_tokens: usize,
    nonce: u64,
) -> Result<(String, usize)> {
    if target_tokens == 0 {
        bail!("synthesize_prompt: target_tokens must be > 0");
    }
    let unique_prefix = format!("Benchmark request {nonce} —");
    // ~10 tokens per filler chunk for any reasonable BPE; overshoot then truncate.
    let filler = " The quick brown fox jumps over the lazy dog.";
    let approx_filler_count = target_tokens.max(10) + 8;
    let text = format!("{unique_prefix}{}", filler.repeat(approx_filler_count));

    let encoded = tokenizer
        .encode(&text[..], false)
        .map_err(|e| anyhow!("tokenizer.encode: {e}"))?;
    let ids = encoded.get_ids();
    if ids.len() < target_tokens {
        bail!(
            "synthesize_prompt: filler tokenized to {} tokens; need >= {target_tokens}. \
             Increase filler size.",
            ids.len()
        );
    }
    let truncated_ids = &ids[..target_tokens];
    let decoded = tokenizer
        .decode(truncated_ids, false)
        .map_err(|e| anyhow!("tokenizer.decode: {e}"))?;

    // Round-trip sanity: re-encode and report actual count.
    let reencoded = tokenizer
        .encode(&decoded[..], false)
        .map_err(|e| anyhow!("tokenizer.encode (verify): {e}"))?;
    let actual_tokens = reencoded.get_ids().len();

    Ok((decoded, actual_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("IRON_BENCH_TEST_TOKENIZER").ok()?;
        Tokenizer::from_file(path).ok()
    }

    #[test]
    fn synth_round_trip_target_lengths() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — skipping synth_round_trip");
            return;
        };
        for target in [32_usize, 128, 512, 2048] {
            let (text, actual) = synthesize_prompt(&tok, target, 42).expect("synth ok");
            assert!(
                actual.abs_diff(target) <= 2,
                "target={target}, actual={actual}, text len={}",
                text.len()
            );
        }
    }

    #[test]
    fn synth_zero_target_errors() {
        // No tokenizer needed — early-returns on 0 before touching tokenizer.
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set — running with stub still ok");
            // We can't construct a Tokenizer without a file; just verify error path
            // by NOT running the assertion. The 0-check is the first statement of
            // synthesize_prompt, before any tokenizer.encode call.
            return;
        };
        let r = synthesize_prompt(&tok, 0, 0);
        assert!(r.is_err(), "target_tokens=0 must return Err");
    }
}
```

- [ ] **Step 2.2: Run prompt tests — verify pass (or skip cleanly)**

```
IRON_BENCH_TEST_TOKENIZER=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/tokenizer.json \
  cargo test --release -p iron-bench prompt:: -- --nocapture
```

Expected: 2 passed (or skip lines if env var missing — should still be 2 passed).

- [ ] **Step 2.3: Implement `client.rs` — types + `process_event` helper + `run_chat_completion`**

Replace `iron-bench/src/client.rs` with:

```rust
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
    /// Generation duration: end - first_token. Falls back to a single tick if no token.
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
    pub content_chars: usize,
}

/// Parse-state mutated by `process_event`. Owned by the request loop.
#[derive(Debug, Default)]
pub struct ParseState {
    pub first_token: Option<Instant>,
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

/// Send one streaming chat completion request and return timing + token counts.
pub async fn run_chat_completion(
    client: &reqwest::Client,
    target_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
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
        stream_options: StreamOptions { include_usage: true },
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

    let mut state = ParseState::default();
    let mut buffer = String::new();

    use futures::StreamExt;
    let mut byte_stream = resp.bytes_stream();
    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.map_err(|e| anyhow!("byte_stream: {e}"))?;
        let s = std::str::from_utf8(&chunk)
            .map_err(|e| anyhow!("non-utf8 SSE chunk: {e}"))?;
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
            end,
        },
        server_prompt_tokens: state.last_usage.as_ref().and_then(|u| u.prompt_tokens),
        server_completion_tokens: state
            .last_usage
            .as_ref()
            .and_then(|u| u.completion_tokens),
        server_cached_tokens: state.last_usage.as_ref().and_then(|u| u.cached_tokens),
        chunk_count: state.chunk_count,
        finish_reason: state.finish_reason.unwrap_or_else(|| "unknown".into()),
        content_chars: state.content_chars,
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
        assert!(state.first_token.is_none(), "role-chunk must not trigger first_token");
        assert_eq!(state.chunk_count, 0);

        // 3 content chunks — each counted, first one sets first_token
        process_event(
            &mut state,
            r#"{"choices":[{"delta":{"content":"Hello"}}]}"#,
            now,
        );
        assert!(state.first_token.is_some(), "first content chunk sets first_token");
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
```

- [ ] **Step 2.4: Run client tests — verify all 3 parser tests pass**

```
cargo test --release -p iron-bench client::
```

Expected: 3 passed (`parser_handles_role_chunk_then_content`, `parser_skips_malformed_payload`, `parser_captures_cached_tokens_extension`).

- [ ] **Step 2.5: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean. If clippy warns about unused `Duration` import in `client.rs`, the import IS used (in `RequestTimings` methods); double-check before silencing.

- [ ] **Step 2.6: Commit**

```bash
git add iron-bench/src/prompt.rs iron-bench/src/client.rs
git commit -m "$(cat <<'EOF'
feat(iron-bench): synthetic prompt generator + HTTP/SSE client

prompt::synthesize_prompt round-trips a tokenizer-controlled string to
exactly `target_tokens` tokens (±2 BPE drift), with a per-run nonce in
the prefix to defeat prefix-cache hits across runs.

client::run_chat_completion sends a streaming /v1/chat/completions
request with stream_options.include_usage=true, parses SSE events via
a pub(crate) process_event helper, and reports timings + chunk count +
server-authoritative usage block (prompt_tokens, completion_tokens,
cached_tokens). 5 unit tests: 2 prompt round-trip (gated on
IRON_BENCH_TEST_TOKENIZER env var), 3 parser (role-chunk skip, malformed
payload, cached_tokens capture).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Per-cell runner

**Files:**
- Modify: `iron-bench/src/runner.rs` (replace stub)

### Goal

Per-target × per-cell driver: warmup runs (results discarded) followed by N timed runs (collected). Pure orchestration code — no new tests at this layer (T4 covers stats; client/prompt tests cover the primitives below).

### Steps

- [ ] **Step 3.1: Implement `runner.rs`**

Replace `iron-bench/src/runner.rs` with:

```rust
//! Per-cell driver: warmup + N timed runs of `run_chat_completion`.
//!
//! Each cell is one (target, prompt_len, max_tokens) combination. Warmup runs
//! materialize MLX compile graphs / allocate caches; their timings are
//! discarded. Timed runs are collected as `RunOutcome`s and reduced by the
//! `report` module.

use std::time::SystemTime;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::client::{run_chat_completion, RequestResult};
use crate::prompt::synthesize_prompt;

#[derive(Debug)]
pub struct CellResult {
    pub target_name: String,
    pub target_url: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub runs: Vec<RunOutcome>,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub run_idx: usize,
    pub prompt_tokens_local: usize,
    pub result: RequestResult,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_cell(
    client: &reqwest::Client,
    target_name: &str,
    target_url: &str,
    model: &str,
    pp: usize,
    tg: usize,
    warmup: usize,
    runs: usize,
    tokenizer: &Tokenizer,
) -> Result<CellResult> {
    eprintln!("[{target_name}] PP={pp} TG={tg}: warmup x{warmup} ...");
    for w in 0..warmup {
        let nonce = nonce_seed() ^ (w as u64);
        let (prompt, _) = synthesize_prompt(tokenizer, pp, nonce)?;
        let _ = run_chat_completion(client, target_url, model, &prompt, tg).await?;
    }

    eprintln!("[{target_name}] PP={pp} TG={tg}: timed runs x{runs} ...");
    let mut outcomes = Vec::with_capacity(runs);
    for i in 0..runs {
        let nonce = nonce_seed() ^ ((i as u64) << 8);
        let (prompt, prompt_tokens_local) = synthesize_prompt(tokenizer, pp, nonce)?;
        let result = run_chat_completion(client, target_url, model, &prompt, tg).await?;

        let ttft_ms = result.timings.ttft().as_secs_f64() * 1000.0;
        let gen_secs = result.timings.gen_duration().as_secs_f64().max(1e-9);
        let tg_count = result
            .server_completion_tokens
            .map(|n| n as f64)
            .unwrap_or(result.chunk_count as f64);
        let tg_tps = tg_count / gen_secs;
        eprintln!(
            "  [{target_name}] run {}/{runs}: TTFT={ttft_ms:.1}ms TG={tg_tps:.1} tok/s prompt={prompt_tokens_local}",
            i + 1
        );

        outcomes.push(RunOutcome {
            run_idx: i,
            prompt_tokens_local,
            result,
        });
    }

    Ok(CellResult {
        target_name: target_name.into(),
        target_url: target_url.into(),
        pp_target: pp,
        tg_target: tg,
        runs: outcomes,
    })
}

fn nonce_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
```

- [ ] **Step 3.2: Wire `run_cell` invocation into `main.rs` dispatch**

In `iron-bench/src/main.rs`, replace the body of `main()` (the placeholder hello + `let _ = args;`) with the real dispatch loop:

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "iron-bench: {} target(s), prompt_len={:?}, max_tokens={}, runs={}, warmup={}",
        args.target.len(),
        args.prompt_len,
        args.max_tokens,
        args.runs,
        args.warmup,
    );

    // Load tokenizer.json from --model-dir for synthetic prompt construction.
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load tokenizer at {}: {e}",
            tokenizer_path.display()
        )
    })?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()
        .context("reqwest::Client::build")?;

    let mut cells: Vec<runner::CellResult> = Vec::new();
    for pp in &args.prompt_len {
        for (target_name, target_url) in &args.target {
            let cell = runner::run_cell(
                &client,
                target_name,
                target_url,
                &args.model,
                *pp,
                args.max_tokens,
                args.warmup,
                args.runs,
                &tokenizer,
            )
            .await?;
            cells.push(cell);
        }
    }

    // Render output via report module (T4 fills in the formatters).
    let out = match args.format {
        OutputFormat::Markdown => report::render_markdown(&cells, &args.target, args.warmup),
        OutputFormat::Csv => report::render_csv(&cells),
        OutputFormat::Json => report::render_json(&cells, &args.target, args.warmup),
    };
    println!("{out}");

    Ok(())
}
```

> **Note**: `report::render_markdown` / `render_csv` / `render_json` don't exist yet. The build will fail until T4. To keep the commit boundary clean, T3 commit may build with `cargo build -p iron-bench` failing. **Skip the build gate at T3 commit time; T4 immediately fixes it.** Alternative: stub the three render functions in T3 returning `String::new()` and replace in T4. Pick whichever the implementer prefers; the stub-then-replace path is cleaner for git history.

- [ ] **Step 3.3: Stub the three report functions to keep the build green**

To keep T3's build green, also stub `iron-bench/src/report.rs` with the three function signatures returning empty strings:

```rust
//! Stats reduction + Markdown / CSV / JSON formatters. Stub bodies in T3;
//! filled in T4.

use crate::runner::CellResult;

pub fn render_markdown(
    _cells: &[CellResult],
    _targets: &[(String, String)],
    _warmup: usize,
) -> String {
    String::new()
}

pub fn render_csv(_cells: &[CellResult]) -> String {
    String::new()
}

pub fn render_json(
    _cells: &[CellResult],
    _targets: &[(String, String)],
    _warmup: usize,
) -> String {
    String::new()
}
```

- [ ] **Step 3.4: Build & smoke (with placeholder targets)**

```
cargo build --release -p iron-bench
```

Expected: clean.

```
cargo run --release -p iron-bench -- \
  --target a=http://127.0.0.1:1 \
  --model-dir /tmp \
  --prompt-len 32 --max-tokens 16 --runs 1 --warmup 0
```

Expected: errors out on tokenizer load (no `tokenizer.json` at `/tmp`); stderr shows the dispatch line + tokenizer-load failure. **This is OK** — we only verify the binary handles arg parsing + the early failure path. Real run requires real tokenizer + servers.

- [ ] **Step 3.5: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 3.6: Commit**

```bash
git add iron-bench/src/runner.rs iron-bench/src/main.rs iron-bench/src/report.rs
git commit -m "$(cat <<'EOF'
feat(iron-bench): per-cell runner + main dispatch loop

runner::run_cell drives W warmup runs + N timed runs against one
(target, prompt_len, max_tokens) cell, collecting RunOutcome with the
local prompt token count + the full RequestResult. main() loads the
tokenizer once from --model-dir/tokenizer.json, builds a reqwest
Client with the configured timeout, iterates the prompt_len × target
matrix, and dispatches to render_{markdown,csv,json} based on
--format. report:: render functions are stubbed (return empty String);
T4 fills them in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Stats reduction + Markdown / CSV / JSON output

**Files:**
- Modify: `iron-bench/src/report.rs` (replace stubs with real implementations)

### Goal

Reduce N `RunOutcome`s per cell into median + p95 statistics; render the result as Markdown (default human-readable table), CSV (one row per run, pandas-friendly), or JSON (nested with raw runs preserved). Includes 2 unit tests covering the stats reduction edge cases.

### Steps

- [ ] **Step 4.1: Implement `report.rs` — types + stats reduction**

Replace `iron-bench/src/report.rs` with:

```rust
//! Stats reduction (median + p95) + Markdown / CSV / JSON output formatters.

use std::time::Duration;

use crate::runner::{CellResult, RunOutcome};

/// Aggregated per-cell statistics across N timed runs.
#[derive(Debug, Clone)]
pub struct CellStats {
    pub target_name: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub n_runs: usize,
    pub ttft_ms_median: f64,
    pub ttft_ms_p95: f64,
    pub tg_tps_median: f64,
    pub tg_tps_p95: f64,
    pub tpot_ms_median: f64,
    pub pp_tps_median: f64,
    pub e2e_s_median: f64,
    pub e2e_s_p95: f64,
    pub finish_reason_summary: String,
    pub cached_tokens_warning: bool,
}

/// Reduce one cell's runs to a `CellStats`. Median + p95 over N runs.
pub fn reduce_cell(c: &CellResult) -> CellStats {
    let mut ttft_ms: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut tg_tps: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut tpot_ms: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut pp_tps: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut e2e_s: Vec<f64> = Vec::with_capacity(c.runs.len());
    let mut finish_reasons: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut cached_warning = false;

    for outcome in &c.runs {
        let r = &outcome.result;
        let ttft = r.timings.ttft();
        let gen = r.timings.gen_duration();
        let e2e = r.timings.e2e();

        let ttft_seconds = ttft.as_secs_f64().max(1e-9);
        let gen_seconds = gen.as_secs_f64().max(1e-9);

        let prompt_tokens = r
            .server_prompt_tokens
            .map(|n| n as f64)
            .unwrap_or(outcome.prompt_tokens_local as f64);
        let completion_tokens = r
            .server_completion_tokens
            .map(|n| n as f64)
            .unwrap_or(r.chunk_count as f64);

        ttft_ms.push(ttft_seconds * 1000.0);
        tg_tps.push(completion_tokens / gen_seconds);
        // TPOT excludes the first token (which is the prefill output): divide gen by (N-1) tokens.
        let tpot_div = (completion_tokens - 1.0).max(1.0);
        tpot_ms.push((gen_seconds / tpot_div) * 1000.0);
        pp_tps.push(prompt_tokens / ttft_seconds);
        e2e_s.push(e2e.as_secs_f64());

        *finish_reasons.entry(r.finish_reason.clone()).or_insert(0) += 1;
        if r.server_cached_tokens.is_some_and(|n| n > 0) {
            cached_warning = true;
        }
    }

    let finish_reason_summary = finish_reasons
        .iter()
        .map(|(k, v)| format!("{k}×{v}"))
        .collect::<Vec<_>>()
        .join(", ");

    CellStats {
        target_name: c.target_name.clone(),
        pp_target: c.pp_target,
        tg_target: c.tg_target,
        n_runs: c.runs.len(),
        ttft_ms_median: median(&mut ttft_ms.clone()),
        ttft_ms_p95: p95(&mut ttft_ms),
        tg_tps_median: median(&mut tg_tps.clone()),
        tg_tps_p95: p95(&mut tg_tps),
        tpot_ms_median: median(&mut tpot_ms),
        pp_tps_median: median(&mut pp_tps),
        e2e_s_median: median(&mut e2e_s.clone()),
        e2e_s_p95: p95(&mut e2e_s),
        finish_reason_summary,
        cached_tokens_warning: cached_warning,
    }
}

/// Median of a Vec<f64>. Mutates input (sorts in place). Empty input yields 0.0.
fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// 95th percentile (linear interpolation).
fn p95(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if xs.len() == 1 {
        return xs[0];
    }
    let rank = 0.95 * (xs.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(xs.len() - 1);
    let frac = rank - lo as f64;
    xs[lo] + frac * (xs[hi] - xs[lo])
}

/// Render Markdown tables, one per metric (TTFT, TG, E2E, PP). Each table has rows = targets,
/// columns = (PP, TG) cells.
pub fn render_markdown(
    cells: &[CellResult],
    targets: &[(String, String)],
    warmup: usize,
) -> String {
    let stats: Vec<CellStats> = cells.iter().map(reduce_cell).collect();
    if stats.is_empty() {
        return String::from("(no cells run)\n");
    }

    let target_names: Vec<&str> = targets.iter().map(|(n, _)| n.as_str()).collect();
    // Distinct (pp, tg) cell columns, in the order they appear in `cells`.
    let mut cell_cols: Vec<(usize, usize)> = Vec::new();
    for s in &stats {
        let key = (s.pp_target, s.tg_target);
        if !cell_cols.contains(&key) {
            cell_cols.push(key);
        }
    }
    let n_runs = stats.first().map(|s| s.n_runs).unwrap_or(0);

    let mut out = String::new();
    out.push_str("# iron-bench results\n\n");
    out.push_str(&format!(
        "- Targets: {}\n",
        targets
            .iter()
            .map(|(n, u)| format!("{n}={u}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("- Sampler: temperature=0, top_p=1 (greedy)\n");
    out.push_str(&format!(
        "- Runs: {n_runs} measured (after {warmup} warmup), median + p95\n\n"
    ));

    let mut table = |title: &str, value_for: &dyn Fn(&CellStats) -> String| {
        out.push_str(&format!("## {title}\n\n"));
        // Header
        out.push_str("| target |");
        for (pp, tg) in &cell_cols {
            out.push_str(&format!(" PP={pp} TG={tg} |"));
        }
        out.push('\n');
        out.push_str("|---|");
        for _ in &cell_cols {
            out.push_str("---|");
        }
        out.push('\n');
        // Rows
        for name in &target_names {
            out.push_str(&format!("| {name} |"));
            for (pp, tg) in &cell_cols {
                let cell = stats
                    .iter()
                    .find(|s| s.target_name == *name && s.pp_target == *pp && s.tg_target == *tg);
                let s = cell.map(value_for).unwrap_or_else(|| "—".into());
                out.push_str(&format!(" {s} |"));
            }
            out.push('\n');
        }
        out.push('\n');
    };

    table("TTFT (ms)", &|s| {
        format!("{:.1} (p95 {:.1})", s.ttft_ms_median, s.ttft_ms_p95)
    });
    table("Decode TG (tok/s)", &|s| {
        format!("{:.1} (p95 {:.1})", s.tg_tps_median, s.tg_tps_p95)
    });
    table("E2E (s)", &|s| {
        format!("{:.3} (p95 {:.3})", s.e2e_s_median, s.e2e_s_p95)
    });
    table("Prefill PP (tok/s, derived)", &|s| {
        format!("{:.1}", s.pp_tps_median)
    });
    table("TPOT (ms/tok)", &|s| format!("{:.2}", s.tpot_ms_median));

    let warned: Vec<String> = stats
        .iter()
        .filter(|s| s.cached_tokens_warning)
        .map(|s| {
            format!(
                "{} PP={} TG={}",
                s.target_name, s.pp_target, s.tg_target
            )
        })
        .collect();
    if warned.is_empty() {
        out.push_str("⚠ cached_tokens > 0 detected for: (none)\n");
    } else {
        out.push_str(&format!(
            "⚠ cached_tokens > 0 detected for: {}\n",
            warned.join(", ")
        ));
    }
    out
}

/// CSV output: one row per timed run. Stable column order.
pub fn render_csv(cells: &[CellResult]) -> String {
    let mut out = String::new();
    out.push_str(
        "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason\n",
    );
    for c in cells {
        for outcome in &c.runs {
            out.push_str(&csv_row(c, outcome));
            out.push('\n');
        }
    }
    out
}

fn csv_row(c: &CellResult, o: &RunOutcome) -> String {
    let r = &o.result;
    let ttft = r.timings.ttft();
    let gen = r.timings.gen_duration();
    let ttft_s = ttft.as_secs_f64().max(1e-9);
    let gen_s = gen.as_secs_f64().max(1e-9);
    let prompt_tokens = r
        .server_prompt_tokens
        .map(|n| n as f64)
        .unwrap_or(o.prompt_tokens_local as f64);
    let completion_tokens = r
        .server_completion_tokens
        .map(|n| n as f64)
        .unwrap_or(r.chunk_count as f64);
    let tpot_div = (completion_tokens - 1.0).max(1.0);

    format!(
        "{name},{pp},{tg},{idx},{ttft_ms:.3},{tg_tps:.3},{tpot_ms:.3},{pp_tps:.3},{e2e_s:.6},{p_local},{p_server},{c_server},{cached},{finish}",
        name = c.target_name,
        pp = c.pp_target,
        tg = c.tg_target,
        idx = o.run_idx,
        ttft_ms = ttft_s * 1000.0,
        tg_tps = completion_tokens / gen_s,
        tpot_ms = (gen_s / tpot_div) * 1000.0,
        pp_tps = prompt_tokens / ttft_s,
        e2e_s = r.timings.e2e().as_secs_f64(),
        p_local = o.prompt_tokens_local,
        p_server = r
            .server_prompt_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        c_server = r
            .server_completion_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        cached = r
            .server_cached_tokens
            .map(|n| n.to_string())
            .unwrap_or_default(),
        finish = r.finish_reason,
    )
}

/// JSON output: nested object with `metadata`, `stats`, and `raw_runs`.
pub fn render_json(
    cells: &[CellResult],
    targets: &[(String, String)],
    warmup: usize,
) -> String {
    let stats: Vec<CellStats> = cells.iter().map(reduce_cell).collect();
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "warmup".into(),
        serde_json::Value::from(warmup),
    );
    metadata.insert(
        "runs_measured".into(),
        serde_json::Value::from(stats.first().map(|s| s.n_runs).unwrap_or(0)),
    );
    let mut sampler = serde_json::Map::new();
    sampler.insert("temperature".into(), serde_json::Value::from(0.0_f64));
    sampler.insert("top_p".into(), serde_json::Value::from(1.0_f64));
    metadata.insert("sampler".into(), serde_json::Value::Object(sampler));
    metadata.insert(
        "targets".into(),
        serde_json::Value::Array(
            targets
                .iter()
                .map(|(n, u)| {
                    serde_json::json!({"name": n, "url": u})
                })
                .collect(),
        ),
    );

    let stats_json: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "target": s.target_name,
                "pp_target": s.pp_target,
                "tg_target": s.tg_target,
                "n_runs": s.n_runs,
                "ttft_ms_median": s.ttft_ms_median,
                "ttft_ms_p95": s.ttft_ms_p95,
                "tg_tps_median": s.tg_tps_median,
                "tg_tps_p95": s.tg_tps_p95,
                "tpot_ms_median": s.tpot_ms_median,
                "pp_tps_median": s.pp_tps_median,
                "e2e_s_median": s.e2e_s_median,
                "e2e_s_p95": s.e2e_s_p95,
                "finish_reason_summary": s.finish_reason_summary,
                "cached_tokens_warning": s.cached_tokens_warning,
            })
        })
        .collect();

    let raw_runs: Vec<serde_json::Value> = cells
        .iter()
        .flat_map(|c| {
            c.runs.iter().map(move |o| {
                let r = &o.result;
                let ttft_s = r.timings.ttft().as_secs_f64().max(1e-9);
                let gen_s = r.timings.gen_duration().as_secs_f64().max(1e-9);
                let prompt_tokens = r
                    .server_prompt_tokens
                    .map(|n| n as f64)
                    .unwrap_or(o.prompt_tokens_local as f64);
                let completion_tokens = r
                    .server_completion_tokens
                    .map(|n| n as f64)
                    .unwrap_or(r.chunk_count as f64);
                serde_json::json!({
                    "target": c.target_name,
                    "pp_target": c.pp_target,
                    "tg_target": c.tg_target,
                    "run_idx": o.run_idx,
                    "ttft_ms": ttft_s * 1000.0,
                    "tg_tps": completion_tokens / gen_s,
                    "pp_tps": prompt_tokens / ttft_s,
                    "e2e_s": r.timings.e2e().as_secs_f64(),
                    "prompt_tokens_local": o.prompt_tokens_local,
                    "prompt_tokens_server": r.server_prompt_tokens,
                    "completion_tokens_server": r.server_completion_tokens,
                    "cached_tokens": r.server_cached_tokens,
                    "finish_reason": r.finish_reason,
                })
            })
        })
        .collect();

    let root = serde_json::json!({
        "metadata": metadata,
        "stats": stats_json,
        "raw_runs": raw_runs,
    });
    serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{RequestResult, RequestTimings};
    use std::time::{Duration, Instant};

    fn fake_outcome(run_idx: usize, ttft_ms: f64, gen_ms: f64, completion_tokens: u32) -> RunOutcome {
        let start = Instant::now();
        let first_token = start + Duration::from_millis(ttft_ms as u64);
        let end = first_token + Duration::from_millis(gen_ms as u64);
        RunOutcome {
            run_idx,
            prompt_tokens_local: 128,
            result: RequestResult {
                timings: RequestTimings {
                    start,
                    first_token: Some(first_token),
                    end,
                },
                server_prompt_tokens: Some(128),
                server_completion_tokens: Some(completion_tokens),
                server_cached_tokens: Some(0),
                chunk_count: completion_tokens,
                finish_reason: "stop".into(),
                content_chars: completion_tokens as usize * 4,
            },
        }
    }

    #[test]
    fn stats_median_and_p95_with_outlier() {
        // 5 runs: TTFT = 40, 42, 45, 50, 200 ms. Median = 45, p95 = ~170 (interpolated).
        let cell = CellResult {
            target_name: "t".into(),
            target_url: "u".into(),
            pp_target: 128,
            tg_target: 128,
            runs: vec![
                fake_outcome(0, 40.0, 800.0, 100),
                fake_outcome(1, 42.0, 800.0, 100),
                fake_outcome(2, 45.0, 800.0, 100),
                fake_outcome(3, 50.0, 800.0, 100),
                fake_outcome(4, 200.0, 800.0, 100),
            ],
        };
        let s = reduce_cell(&cell);
        assert_eq!(s.n_runs, 5);
        assert!(
            (s.ttft_ms_median - 45.0).abs() < 0.5,
            "median expected ~45, got {}",
            s.ttft_ms_median
        );
        // p95 of [40,42,45,50,200] with linear interp at rank 0.95*4=3.8:
        //   xs[3] + 0.8*(xs[4]-xs[3]) = 50 + 0.8*(200-50) = 50 + 120 = 170
        assert!(
            (s.ttft_ms_p95 - 170.0).abs() < 0.5,
            "p95 expected ~170, got {}",
            s.ttft_ms_p95
        );
    }

    #[test]
    fn csv_columns_stable() {
        let cell = CellResult {
            target_name: "ironmlx".into(),
            target_url: "http://localhost:8080".into(),
            pp_target: 128,
            tg_target: 64,
            runs: vec![fake_outcome(0, 50.0, 500.0, 64)],
        };
        let csv = render_csv(&[cell]);
        let header = csv.lines().next().expect("header line");
        assert_eq!(
            header,
            "target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason"
        );
        let body = csv.lines().nth(1).expect("data line");
        assert!(body.starts_with("ironmlx,128,64,0,"), "unexpected row: {body}");
        assert!(body.ends_with(",stop"), "expected to end with finish_reason=stop, got: {body}");
    }
}
```

- [ ] **Step 4.2: Run report tests**

```
cargo test --release -p iron-bench report::
```

Expected: 2 passed (`stats_median_and_p95_with_outlier`, `csv_columns_stable`).

- [ ] **Step 4.3: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 4.4: Run full iron-bench test set**

```
IRON_BENCH_TEST_TOKENIZER=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/tokenizer.json \
  cargo test --release -p iron-bench
```

Expected: 7 passed (2 prompt + 3 client + 2 report).

- [ ] **Step 4.5: Commit**

```bash
git add iron-bench/src/report.rs
git commit -m "$(cat <<'EOF'
feat(iron-bench): stats reduction + Markdown / CSV / JSON output

reduce_cell collapses N RunOutcomes into a CellStats with median + p95
on TTFT_ms / TG_tps / E2E_s, plus medians for TPOT_ms and PP_tps.
Server-side usage.{prompt,completion}_tokens is preferred when present;
chunk_count is the local fallback.

Three render functions:
- render_markdown emits one table per metric (rows=targets, cols=cells)
- render_csv emits one row per timed run with stable header order
- render_json emits {metadata, stats[], raw_runs[]} pretty-printed

2 unit tests: median + p95 with an outlier (verifies the linear-interp
p95 calculation), and CSV header stability (regression-guards the
column order pandas consumers depend on).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Manual E2E test + README

**Files:**
- Create: `iron-bench/README.md`

### Goal

End-to-end manual smoke against the real ironmlx + omlx servers + Qwen3.5-4B-MLX-4bit checkpoint. Document the workflow + sample output in `iron-bench/README.md`.

### Steps

- [ ] **Step 5.1: Start ironmlx server in one terminal**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
MLX_DIR=$HOME/.local/mlx cargo run --release -p ironmlx -- serve --model "$SNAP" --port 8080
```

Wait for `ironmlx server listening on http://127.0.0.1:8080`.

- [ ] **Step 5.2: Start omlx server in another terminal**

```sh
omlx serve --model "$SNAP" --port 8081
```

Wait for omlx ready signal in its log.

- [ ] **Step 5.3: Run iron-bench in a third terminal**

```sh
cd /Volumes/Dev/cxx-mlx
SNAP=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir "$SNAP" \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1
```

Expected:
- ~5-10 minutes total
- Per-run progress lines on stderr (one per run)
- Markdown tables on stdout: TTFT (ms), Decode TG (tok/s), E2E (s), Prefill PP (tok/s), TPOT (ms/tok)
- TG values for both targets in the 50-150 tok/s range on M-series silicon for Qwen3.5-4B-MLX-4bit
- `cached_tokens > 0` warning empty (per-run nonce defeats prefix cache)

- [ ] **Step 5.4: Verify CSV output is pandas-compatible**

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir "$SNAP" \
  --prompt-len 128 \
  --max-tokens 32 \
  --runs 3 --warmup 1 \
  --format csv > /tmp/iron-bench-smoke.csv

/opt/homebrew/Caskroom/miniforge/base/envs/mlx/bin/python -c "
import pandas as pd
df = pd.read_csv('/tmp/iron-bench-smoke.csv')
print(df)
print('rows:', len(df), 'cols:', list(df.columns))
"
```

Expected: 6 rows (2 targets × 3 runs); 14 columns; no parse error.

- [ ] **Step 5.5: Write `iron-bench/README.md`**

Create `iron-bench/README.md`:

```markdown
# iron-bench

Head-to-head HTTP benchmark harness for OpenAI-compatible LLM endpoints.

## What it measures

Per (target, prompt_len, max_tokens) cell, across N timed runs (after W warmup):

- **TTFT (ms)** — time from request send to first non-empty content token
- **TG (tok/s)** — decode tokens per second (completion_tokens / generation_duration)
- **TPOT (ms/tok)** — time per output token (excluding prefill)
- **PP (tok/s)** — prefill tokens per second (prompt_tokens / TTFT_seconds)
- **E2E (s)** — total wall-clock time

Reports median + p95 across runs. Output formats: Markdown (default), CSV (pandas-friendly,
one row per run), JSON (nested with raw runs preserved).

## Engine-neutral

iron-bench has no dependency on the `ironmlx` / `mlx` / `mlx-sys` crates. It drives
**any OpenAI-compatible `/v1/chat/completions` endpoint** — ironmlx, omlx, mlx-lm-server,
vllm-mlx, llama.cpp, third-party cloud providers — at the same external boundary users hit.

## Methodology highlights

- **Synthetic controlled-length prompts** — uses your model's `tokenizer.json` to round-trip
  a string to exactly N tokens (±2 BPE drift); per-run nonce in the prefix prevents prefix-cache
  hits across runs (omlx defaults to enable a tiered prefix cache; without nonce, the second
  run's prefill is ~0ms, invalidating PP measurement).
- **Greedy sampler** — `temperature=0, top_p=1` for both/all targets (deterministic, no
  sampler-algorithm bias).
- **stream_options.include_usage=true** — preferred for authoritative `prompt_tokens` and
  `completion_tokens` from the server; falls back to local SSE chunk count.
- **Warmup excluded** — first N=1 run materializes MLX compile graphs / KV caches; not counted.

## Usage

```sh
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir /path/to/Qwen3.5-4B-MLX-4bit/snapshot \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1 \
  --format markdown   # or csv, json
```

`--target name=URL` can be repeated for any number of endpoints. `--prompt-len` is
comma-separated; iron-bench iterates `prompt_len × target` cells.

## Sample Markdown output

```
# iron-bench results

- Targets: ironmlx=http://localhost:8080, omlx=http://localhost:8081
- Sampler: temperature=0, top_p=1 (greedy)
- Runs: 5 measured (after 1 warmup), median + p95

## TTFT (ms)
| target  | PP=128 TG=128 | PP=512 TG=128 | PP=2048 TG=128 |
|---|---|---|---|
| ironmlx | 45.2 (p95 47.1) | 152.4 (p95 156.0) | 521.8 (p95 530.4) |
| omlx    | 42.1 (p95 43.5) | 148.7 (p95 151.2) | 510.3 (p95 518.0) |

## Decode TG (tok/s)
... (same shape)
```

## CSV consumption

The `--format csv` output is pandas-friendly:

```python
import pandas as pd
df = pd.read_csv("results.csv")
df.groupby(["target", "pp_target"])["tg_tps"].median()
```

## Limitations

- **Single-request only**. Multi-request concurrency comes in v2 once ironmlx P8b ships
  the batched scheduler.
- **HTTP overhead** (~0.1-0.5ms loopback) is included in TTFT/E2E. Both targets bear it
  equally so it cancels in head-to-head comparison.
- **No GPU memory monitoring** — the HTTP layer is opaque to the engine's memory profile.
- **OpenAI endpoint only** in v1. Anthropic `/v1/messages` is symmetric work but deferred.
```

- [ ] **Step 5.6: Project gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release
```

Expected: clean.

- [ ] **Step 5.7: Commit README**

```bash
git add iron-bench/README.md
git commit -m "$(cat <<'EOF'
docs(iron-bench): README with methodology + usage + sample output

Documents the five reported metrics (TTFT, TG, TPOT, PP, E2E),
methodology notes (synthetic controlled-length prompts, per-run nonce
to defeat prefix cache, greedy sampler, stream_options.include_usage
for authoritative token counts), the multi-target CLI shape, sample
Markdown output, pandas CSV consumption snippet, and v1 limitations
(single-request only — multi-request awaits ironmlx P8b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Acceptance

After Tasks 1-5 are complete and committed, verify the spec's § 7 acceptance criteria are met:

- [ ] **Acceptance gate**

```
cargo +nightly fmt --all -- --check && \
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && \
cargo build --release && \
IRON_BENCH_TEST_TOKENIZER=/Users/sam/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3/tokenizer.json \
  cargo test --release -p iron-bench
```

Expected: clean + 7 tests passing (2 prompt + 3 client + 2 report).

- [ ] **Spec invariants confirmed**

  1. iron-bench has zero dependency on `mlx-sys` / `mlx` / `ironmlx` crates: `grep -E '^(mlx-sys|mlx|ironmlx)' iron-bench/Cargo.toml` yields nothing.
  2. ironmlx / mlx / mlx-sys source unchanged: `git diff main..HEAD -- mlx-sys mlx ironmlx` is empty (only `Cargo.toml` workspace.members and the new `iron-bench/` directory differ).
  3. Markdown / CSV / JSON output formats all produced cleanly via `--format` flag.

- [ ] **Manual E2E** (Boss runs, see Task 5)

  Two real servers (ironmlx + omlx) bound to localhost ports; iron-bench produces a Markdown table comparing TTFT / TG / TPOT / PP / E2E across `--prompt-len 128,512,2048`. Numerical sanity: TG tok/s for both targets in 50-150 range on M-series Apple Silicon for Qwen3.5-4B-MLX-4bit.
