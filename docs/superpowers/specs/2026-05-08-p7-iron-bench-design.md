# P7 iron-bench 设计

**目标：** 构建独立的 Rust workspace binary `iron-bench`，通过 HTTP `/v1/chat/completions` 端点向多个 OpenAI 兼容引擎发同样的合成请求，输出对照表用于 ironmlx 与 omlx 的单请求 PP / TG / E2E 性能对比（以及未来其他引擎如 mlx-lm-server / vllm-mlx 的扩展）。

**作用域：**
- 新增 workspace member `iron-bench/`（与 `mlx-sys`、`mlx`、`ironmlx` 同级）
- HTTP-only methodology：harness 仅发 OpenAI 兼容 JSON 请求 + 解析 SSE 流，不与任何引擎进程内联
- 五项指标：**TTFT (ms)**、**TG decode (tok/s)**、**TPOT (ms/tok)**、**PP prefill (tok/s)** = `prompt_tokens / TTFT`、**E2E (s)**
- 多 target 并列 (`--target name=url` 重复)
- 合成 prompts，控长 token-count（用本地 `tokenizer.json` round-trip 保证两端 server tokenize 后 token 数一致）
- warmup + N 次重复，输出 median + p95
- 三种输出格式：Markdown（默认）、CSV、JSON

**Out of scope：**
- ❌ 多并发请求（P8b 范围）
- ❌ 真实 prompt 数据集 / 多领域语料（v2 可加）
- ❌ 可视化（CSV 导出后用 Python 画）
- ❌ Peak GPU memory / power 监控（HTTP 外部不可见）
- ❌ 自动启停 server（Boss 自启 server，harness 只发请求）
- ❌ Anthropic `/v1/messages` 端点（v1 仅 OpenAI；同口径足够对比）
- ❌ 鉴权 / API key（本地服务，无 token 头）

**依赖：**
- workspace 已有 deps：`tokio` (rt-multi-thread + macros)、`reqwest` (json+stream+rustls-tls)、`serde` + `serde_json`、`clap` (derive)、`tokenizers`、`anyhow`、`futures`
- iron-bench **不依赖** ironmlx / mlx / mlx-sys crate（保持工具中立 — 即使 ironmlx 没装也能驱动其他引擎）
- Qwen3.5-4B-MLX-4bit 本地 checkpoint（仅用其 `tokenizer.json` 做合成 prompt 控长；不加载模型）

---

## § 1 调研发现与决策摘要

### 1.1 omlx 性能对比生态

omlx 内部已有完整 benchmark 实现 (`/Volumes/Dev/omlx/omlx/admin/benchmark.py`)：

- 指标命名：`ttft_ms`、`tpot_ms`、`gen_tps`、`processing_tps`、`e2e_latency_s`、`peak_memory_bytes`、`prompt_tokens`、`completion_tokens`、`cached_tokens`
- 方法：`mx.reset_peak_memory()` + 流式 `engine.stream_generate()` + 首 token 时间用 `output.completion_tokens > prev_completion_tokens` 检测（不用 new_text，避免 harmony 协议 token 干扰）
- omlx 内置 cache（hot RAM + cold SSD），benchmark 后端会在 `cached_tokens > 0` 时打 warning（污染 prefill 测量）

iron-bench 借鉴指标命名 + 首 token 检测方法（用 SSE chunk 中 `delta.content` 非空判断），但**走 HTTP 外部路径**（不调 omlx 内部 API），保证两端方法论一致。

### 1.2 决策摘要

| 决策维度 | 选择 | Brainstorm 来源 |
|---|---|---|
| **Harness 驱动方式** | HTTP-only（双方各启 server，harness 只发 JSON 请求 + 解析 SSE） | Boss 选；apples-to-apples，外部口径一致 |
| **Harness 寄宿位置** | 独立 Rust workspace member `iron-bench`（不寄宿 ironmlx CLI） | Boss 选；名字中性，可比较任何 OpenAI 兼容引擎 |
| **HTTP 客户端** | `reqwest 0.12`（已是 workspace dep） | 已有，复用 |
| **SSE 解析** | 手写：`bytes_stream` + `\n\n` 分隔 + `data: ` 前缀剥离 + JSON 解析 | reqwest-eventsource 是额外依赖；手写 ~30 行；OpenAI SSE 格式简单 |
| **Prompt 合成** | 本地 tokenizer round-trip：encode 字符串 → 截至 N 个 token id → decode 回字符串作为 prompt | 保证两端 server tokenize 后正好 N 个 token |
| **Cache 防污染** | 每次 run 加 nonce 到 prompt 前缀 | omlx 默认开 prefix cache；不加 nonce 第二次 run 的 PP 几乎为 0 |
| **指标统计** | median + p95，N=5 默认（p95 在 N≥10 才精细） | 中位数对异常值鲁棒；p95 看尾延迟 |
| **完成 token 计数** | 优先用 server 返回的 `usage.completion_tokens`（最终 chunk）；fallback 数 SSE chunks with `delta.content != ""` | OpenAI 标准字段；ironmlx P4 polish 已加 usage；omlx 也支持 |
| **Sampler** | 强制 `temperature=0, top_p=1`（greedy） | deterministic，排除 sampler 算法差异 |

---

## § 2 算法 / 数据流

```mermaid
graph TD
    USER[Boss runs iron-bench] --> CLI[parse CLI args:<br/>targets, prompt-len matrix,<br/>max-tokens, runs, warmup, format]
    CLI --> TOK[load tokenizer.json from --model-dir]
    TOK --> MATRIX[for each PP_len in --prompt-len]
    MATRIX --> CELL[for each target in --target]
    CELL --> SYNTH[synthesize prompt:<br/>nonce + filler → encode → take[..N] → decode]
    SYNTH --> WARM[run warmup x W]
    WARM --> RUN[run timed x N]
    RUN --> HTTP[HTTP POST /v1/chat/completions<br/>stream=true, temperature=0]
    HTTP --> SSE[parse SSE stream:<br/>track first_token, count chunks, capture usage]
    SSE --> RESULT[RequestResult]
    RESULT --> AGG[aggregate per cell + target:<br/>median + p95]
    AGG --> OUTPUT{--format}
    OUTPUT -->|markdown| MD[Markdown table to stdout]
    OUTPUT -->|csv| CSV[CSV row-per-run to stdout]
    OUTPUT -->|json| JSON[nested JSON object to stdout]
```

**单 request 时序**：

```
client                                 server
  |                                      |
  |-- POST /v1/chat/completions -------->|
  |   {messages, stream:true, temp:0}    |
  |                                      | tokenize prompt ─┐
  |                                      | prefill          │ TTFT
  |                                      | sample first tok │
  |<-- SSE: data: {role, content:""} ----| (some servers)  ─┤
  |   (skip — empty content)             |                  │
  |<-- SSE: data: {content:"Hello"} -----| ← first_token  ─┘
  |                                      | decode loop
  |<-- SSE: data: {content:" world"} ----|              ↘ TG = N_chunks / (end - first_token)
  |   (count as 1 chunk)                 |
  |  ...                                 |
  |<-- SSE: data: {content:"!"} ---------|
  |<-- SSE: data: {finish_reason:"stop", |
  |           usage:{...}} --------------| ← capture usage if present
  |<-- SSE: data: [DONE] ----------------|
  | end_time = now ─┘ E2E = end - start
```

---

## § 3 详细设计

### 3.1 Workspace 结构

```
cxx-mlx/
├── Cargo.toml                    # MODIFIED: workspace.members += "iron-bench"
├── iron-bench/                   # NEW
│   ├── Cargo.toml                # NEW: separate binary crate
│   └── src/
│       ├── main.rs               # NEW: CLI entry + arg parsing + tokio runtime
│       ├── client.rs             # NEW: HTTP client + SSE parser
│       ├── prompt.rs             # NEW: synthetic prompt generator (tokenizer-aware)
│       ├── runner.rs             # NEW: warmup + N runs per cell
│       └── report.rs             # NEW: Markdown / CSV / JSON formatters + stats
```

`iron-bench/Cargo.toml`：

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
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
anyhow = "1"
futures.workspace = true
```

iron-bench 不依赖 `mlx-sys` / `mlx` / `ironmlx`，从而：
- 没装 ironmlx 也能驱动 omlx vs mlx-lm-server vs vllm-mlx 三方对比
- 不被 cxx-mlx 的 build 约束（无 `MLX_DIR` 要求）

### 3.2 CLI args — `main.rs`

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "iron-bench",
    about = "Head-to-head HTTP benchmark for OpenAI-compatible LLM endpoints",
    version
)]
struct Args {
    /// Target endpoints. Repeat for multiple targets.
    /// Format: name=URL (e.g., `--target ironmlx=http://localhost:8080`)
    #[arg(long, value_parser = parse_target, required = true, num_args = 1..)]
    target: Vec<(String, String)>,

    /// Path to model dir containing `tokenizer.json` (used for prompt synthesis only).
    #[arg(long)]
    model_dir: PathBuf,

    /// Model name to send in the `model` field of the JSON request.
    #[arg(long, default_value = "qwen3.5-4b")]
    model: String,

    /// Prompt token lengths to test (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![128_usize, 512, 2048])]
    prompt_len: Vec<usize>,

    /// Number of generated tokens per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Timed runs per cell.
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
enum OutputFormat { Markdown, Csv, Json }

fn parse_target(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(name, url)| (name.into(), url.trim_end_matches('/').into()))
        .ok_or_else(|| format!("expected name=URL, got '{s}'"))
}
```

### 3.3 Prompt 生成 — `prompt.rs`

**目标**：生成的 prompt 经任何使用同一 tokenizer 的 server tokenize 后正好 N 个 token；含 per-run nonce 防 prefix-cache 污染。

```rust
use anyhow::Result;
use tokenizers::Tokenizer;

/// Synthesize a prompt that tokenizes to exactly `target_tokens` tokens.
///
/// Strategy:
/// 1. Build text = nonce + repeated filler word
/// 2. Encode text → take first `target_tokens` ids
/// 3. Decode back to string (round-trip ensures server-side encode == target_tokens)
///
/// `nonce` is included to prevent prefix-cache hits across runs (omlx defaults to enable
/// tiered prefix cache; without nonce, the second run's PP would be ~0ms — invalidating
/// the prefill measurement).
pub fn synthesize_prompt(
    tokenizer: &Tokenizer,
    target_tokens: usize,
    nonce: u64,
) -> Result<(String, usize)> {
    if target_tokens == 0 {
        anyhow::bail!("synthesize_prompt: target_tokens must be > 0");
    }
    let unique_prefix = format!("Benchmark request {nonce} —");
    // Filler chosen to tokenize into ~10 tokens regardless of model
    let filler = " The quick brown fox jumps over the lazy dog.";
    let approx_filler_count = (target_tokens / 5) + 8;  // overshoot, then truncate
    let text = format!("{unique_prefix}{}", filler.repeat(approx_filler_count));

    let encoded = tokenizer
        .encode(&text[..], false)
        .map_err(|e| anyhow::anyhow!("tokenizer.encode: {e}"))?;
    let ids = encoded.get_ids();
    if ids.len() < target_tokens {
        anyhow::bail!(
            "synthesize_prompt: filler tokenized to {} tokens; need >= {target_tokens}. \
             Increase filler size.",
            ids.len()
        );
    }
    let truncated_ids = &ids[..target_tokens];
    let decoded = tokenizer
        .decode(truncated_ids, false)
        .map_err(|e| anyhow::anyhow!("tokenizer.decode: {e}"))?;

    // Round-trip sanity: encode the decoded string and confirm it's exactly target_tokens.
    // (Some tokenizers are not exactly round-trip preserving; absorb minor drift here.)
    let reencoded = tokenizer
        .encode(&decoded[..], false)
        .map_err(|e| anyhow::anyhow!("tokenizer.encode (verify): {e}"))?;
    let actual_tokens = reencoded.get_ids().len();

    Ok((decoded, actual_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test loads a real tokenizer.json (passed via env var or fixture path).
    // Verify round-trip yields target_tokens ± 2 (small drift acceptable).

    fn load_test_tokenizer() -> Option<Tokenizer> {
        let path = std::env::var("IRON_BENCH_TEST_TOKENIZER").ok()?;
        Tokenizer::from_file(path).ok()
    }

    #[test]
    fn synth_round_trip_target_lengths() {
        let Some(tok) = load_test_tokenizer() else {
            eprintln!("IRON_BENCH_TEST_TOKENIZER not set, skipping");
            return;
        };
        for target in [32, 128, 512, 2048] {
            let (text, actual) = synthesize_prompt(&tok, target, 42).unwrap();
            assert!(
                actual.abs_diff(target) <= 2,
                "target={target}, actual={actual}, drift > 2 — text len={}",
                text.len()
            );
        }
    }

    #[test]
    fn synth_zero_target_errors() {
        let Some(tok) = load_test_tokenizer() else { return; };
        assert!(synthesize_prompt(&tok, 0, 0).is_err());
    }
}
```

> **Note**：tokenizer round-trip 偶有 ±1-2 token 漂移（BPE 边界）。Production 路径忽略；测试用 `abs_diff(target) <= 2` 容忍。Server 端 tokenize 数从 `usage.prompt_tokens` 读权威值（见 § 3.4）。

### 3.4 HTTP client + SSE parser — `client.rs`

```rust
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
    pub fn ttft(&self) -> Duration {
        self.first_token
            .unwrap_or(self.end)
            .duration_since(self.start)
    }
    pub fn e2e(&self) -> Duration { self.end.duration_since(self.start) }
    pub fn gen_duration(&self) -> Duration {
        self.end.duration_since(self.first_token.unwrap_or(self.start))
    }
}

#[derive(Debug, Clone)]
pub struct RequestResult {
    pub timings: RequestTimings,
    /// Authoritative token count from server `usage` (preferred).
    pub server_prompt_tokens: Option<u32>,
    pub server_completion_tokens: Option<u32>,
    pub server_cached_tokens: Option<u32>,
    /// Local fallback: count of SSE chunks with non-empty delta.content.
    pub chunk_count: u32,
    pub finish_reason: String,
    pub content_chars: usize,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    /// Some servers (OpenAI proper) require `stream_options.include_usage=true` to
    /// emit usage in stream final chunk. ironmlx P4 polish made this implicit; omlx
    /// also emits it by default. Setting it explicitly is harmless.
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct StreamOptions { include_usage: bool }

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

#[derive(Deserialize, Clone)]
struct UsageBlock {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    /// omlx-specific extension; absent on stock OpenAI.
    #[serde(default)]
    cached_tokens: Option<u32>,
}

pub async fn run_chat_completion(
    client: &reqwest::Client,
    target_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<RequestResult> {
    let body = ChatRequest {
        model,
        messages: vec![ChatMessage { role: "user", content: prompt }],
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
        .await?;
    if !resp.status().is_success() {
        bail!("{target_url}: HTTP {} — {}", resp.status(), resp.text().await?);
    }

    let mut byte_stream = resp.bytes_stream();
    let mut first_token: Option<Instant> = None;
    let mut chunk_count: u32 = 0;
    let mut content_chars: usize = 0;
    let mut finish_reason: Option<String> = None;
    let mut last_usage: Option<UsageBlock> = None;
    let mut buffer = String::new();

    use futures::StreamExt;
    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk?;
        buffer.push_str(std::str::from_utf8(&chunk)?);
        // Drain complete SSE events (separator: \n\n)
        while let Some(end_idx) = buffer.find("\n\n") {
            let event = buffer[..end_idx].to_string();
            buffer.drain(..end_idx + 2);
            // Each event may have multiple "data: ..." lines; OpenAI uses one.
            for line in event.lines() {
                let Some(payload) = line.strip_prefix("data: ") else { continue; };
                let payload = payload.trim();
                if payload == "[DONE]" { continue; }
                let parsed: SseChunk = match serde_json::from_str(payload) {
                    Ok(p) => p,
                    Err(_) => continue,  // unknown event shape; skip silently
                };
                if let Some(c) = parsed.choices.first() {
                    if let Some(content) = &c.delta.content {
                        if !content.is_empty() {
                            if first_token.is_none() {
                                first_token = Some(Instant::now());
                            }
                            chunk_count += 1;
                            content_chars += content.chars().count();
                        }
                    }
                    if let Some(reason) = &c.finish_reason {
                        finish_reason = Some(reason.clone());
                    }
                }
                if let Some(u) = parsed.usage {
                    last_usage = Some(u);
                }
            }
        }
    }
    let end = Instant::now();
    Ok(RequestResult {
        timings: RequestTimings { start, first_token, end },
        server_prompt_tokens: last_usage.as_ref().and_then(|u| u.prompt_tokens),
        server_completion_tokens: last_usage.as_ref().and_then(|u| u.completion_tokens),
        server_cached_tokens: last_usage.as_ref().and_then(|u| u.cached_tokens),
        chunk_count,
        finish_reason: finish_reason.unwrap_or_else(|| "unknown".into()),
        content_chars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// To make the SSE parsing testable without HTTP, T2 must extract the
    /// per-event handling into a small `pub(crate) fn process_event(...)` helper
    /// taking the JSON payload string + a `&mut ParseState { chunk_count,
    /// content_chars, first_token, finish_reason, last_usage }` and applying
    /// the same logic as the inline loop in `run_chat_completion`. The tests
    /// then drive synthetic events directly without spinning up a server.
    ///
    /// Concrete test bodies (T2 will implement once `process_event` is extracted):
    ///
    /// - `parser_handles_role_chunk_then_content`: feed (role chunk with empty
    ///   content, then 3 content chunks, then `[DONE]`); assert
    ///   chunk_count == 3, first_token is Some, finish_reason == "stop"
    ///   (captured from the 3rd chunk's finish_reason field).
    ///
    /// - `parser_handles_split_event_across_chunks`: feed the same payload
    ///   bytes split mid-event (between `data:` prefix and JSON body); assert
    ///   the buffered parser yields the same final state.
    ///
    /// - `parser_skips_malformed_payload`: feed `data: {garbage` followed by
    ///   a valid event; assert no panic and the valid event is processed.
    #[test]
    fn parser_helper_exists_and_processes_events() {
        // Compile-time signal that T2 must expose `process_event`. Body filled
        // in T2 once the helper is extracted.
    }
}
```

> **`stream_options.include_usage: true`**：OpenAI 标准要求显式开启才会在 stream 末尾附 `usage`。ironmlx P4 polish 加的 usage 在 non-stream 路径，stream 路径未实装；T2 实施时如发现 ironmlx stream usage 缺失，先用 chunk_count fallback 承担。omlx 默认 emit usage（不依赖此 flag），所以 omlx 侧没问题。

### 3.5 Runner — `runner.rs`

```rust
use crate::client::{run_chat_completion, RequestResult};
use crate::prompt::synthesize_prompt;
use anyhow::Result;
use std::time::Duration;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct CellResult {
    pub target_name: String,
    pub target_url: String,
    pub pp_target: usize,    // requested target_tokens
    pub tg_target: usize,    // requested max_tokens
    pub runs: Vec<RunOutcome>,    // measured runs only (warmup excluded)
}

#[derive(Debug)]
pub struct RunOutcome {
    pub run_idx: usize,
    pub prompt_tokens_local: usize,    // post-round-trip local count
    pub result: RequestResult,
}

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
        let tg_tps = (result.server_completion_tokens.unwrap_or(result.chunk_count) as f64)
            / result.timings.gen_duration().as_secs_f64().max(1e-9);
        eprintln!(
            "  run {}/{runs}: TTFT={ttft_ms:.1}ms TG={tg_tps:.1} tok/s prompt={prompt_tokens_local}",
            i + 1
        );

        outcomes.push(RunOutcome { run_idx: i, prompt_tokens_local, result });
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
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
```

### 3.6 Stats reduction + report — `report.rs`

```rust
use crate::runner::{CellResult, RunOutcome};

pub struct CellStats {
    pub target: String,
    pub pp_target: usize,
    pub tg_target: usize,
    pub n_runs: usize,
    pub ttft_ms_median: f64,
    pub ttft_ms_p95: f64,
    pub tg_tps_median: f64,
    pub tg_tps_p95: f64,
    pub tpot_ms_median: f64,
    pub pp_tps_median: f64,    // = prompt_tokens / TTFT
    pub e2e_s_median: f64,
    pub e2e_s_p95: f64,
    pub finish_reason_summary: String,
    pub cached_tokens_warning: bool,    // if any run had cached_tokens > 0
}

pub fn reduce_cell(c: &CellResult) -> CellStats { /* sort + median + p95 */ }

pub fn render_markdown(stats: &[CellStats]) -> String { /* ... */ }
pub fn render_csv(cells: &[CellResult]) -> String { /* per-run row */ }
pub fn render_json(stats: &[CellStats], cells: &[CellResult]) -> String { /* nested */ }

#[cfg(test)]
mod tests {
    /// Fake 5 RunOutcomes with known TTFT spread → verify median + p95 calc.
    #[test]
    fn stats_median_and_p95() { /* ... */ }

    /// CSV row format stable across format flag.
    #[test]
    fn csv_columns_stable() { /* ... */ }
}
```

#### Markdown 输出示例

```markdown
# iron-bench: Qwen3.5-4B-MLX-4bit

- Date: 2026-05-08T16:35:12Z
- Targets: ironmlx=http://localhost:8080, omlx=http://localhost:8081
- Sampler: temperature=0, top_p=1 (greedy)
- Runs: 5 measured (after 1 warmup), median + p95 reported

## TTFT (ms)
| target  | PP=128 TG=128 | PP=512 TG=128 | PP=2048 TG=128 |
|---------|---------------|---------------|----------------|
| ironmlx | 45.2 (p95 47.1) | 152.4 (p95 156.0) | 521.8 (p95 530.4) |
| omlx    | 42.1 (p95 43.5) | 148.7 (p95 151.2) | 510.3 (p95 518.0) |

## Decode TG (tok/s)
... (same shape)

## E2E (s)
... (same shape)

## Prefill PP (tok/s, derived = prompt_tokens / TTFT_s)
... (same shape)

⚠ cached_tokens > 0 detected for: <none>  (otherwise lists target/cell pairs where cache was warm)
```

#### CSV 输出（`--format csv`）

每 run 一行：
```
target,pp_target,tg_target,run_idx,ttft_ms,tg_tps,tpot_ms,pp_tps,e2e_s,prompt_tokens_local,prompt_tokens_server,completion_tokens_server,cached_tokens,finish_reason
ironmlx,128,128,0,45.2,130.1,7.69,2832.0,1.039,128,128,128,0,stop
ironmlx,128,128,1,44.8,131.5,7.61,2857.1,1.026,128,128,128,0,stop
...
```

可被 `pandas.read_csv()` 直接加载。

#### JSON 输出（`--format json`）

```json
{
  "metadata": {
    "date": "2026-05-08T16:35:12Z",
    "runs_measured": 5,
    "warmup": 1,
    "sampler": {"temperature": 0.0, "top_p": 1.0},
    "targets": [
      {"name": "ironmlx", "url": "http://localhost:8080"},
      {"name": "omlx", "url": "http://localhost:8081"}
    ]
  },
  "stats": [
    {"target": "ironmlx", "pp_target": 128, "tg_target": 128,
     "ttft_ms_median": 45.2, "ttft_ms_p95": 47.1, ...}
  ],
  "raw_runs": [
    {"target": "ironmlx", "pp_target": 128, "tg_target": 128, "run_idx": 0, ...}
  ]
}
```

---

## § 4 测试策略

iron-bench 是 utility，不需要重型集成测试。仅做：

| 测试 | 模块 | 验证内容 |
|---|---|---|
| `synth_round_trip_target_lengths` | `prompt::tests` | 给 target_tokens=32/128/512/2048，round-trip 后 abs_diff ≤ 2（环境变量提供 tokenizer.json 路径，否则 skip） |
| `synth_zero_target_errors` | `prompt::tests` | target_tokens=0 → Err |
| `parser_handles_role_chunk_then_content` | `client::tests` | 合成 SSE 字节流 → 跳 role chunk + 计 3 个 content chunk + 捕获 finish_reason="stop" + 捕获 usage |
| `parser_handles_split_event_across_chunks` | `client::tests` | 单个 SSE event 跨多个 byte chunk → 正确缓冲 + 解析 |
| `parser_skips_malformed_payload` | `client::tests` | 非法 JSON payload → 不 panic，继续处理后续 events |
| `stats_median_and_p95` | `report::tests` | 5 个 fake RunOutcome（TTFT 均匀 + 1 outlier）→ median 在中间，p95 ≈ outlier |
| `csv_columns_stable` | `report::tests` | render_csv 输出第一行 header 字段顺序固定 |

合计 7 单测，全部 pure Rust（无 GPU / 无网络 / 无外部 process），CI 可跑。

**手测验收（Boss 跑）：**

```sh
# Terminal 1
ironmlx serve --model ~/.cache/.../Qwen3.5-4B-MLX-4bit/snapshots/<sha> --port 8080

# Terminal 2
omlx serve ... --port 8081

# Terminal 3
SNAP=~/.cache/.../Qwen3.5-4B-MLX-4bit/snapshots/<sha>
cargo run --release -p iron-bench -- \
  --target ironmlx=http://localhost:8080 \
  --target omlx=http://localhost:8081 \
  --model-dir "$SNAP" \
  --prompt-len 128,512,2048 \
  --max-tokens 128 \
  --runs 5 --warmup 1
```

预期：~5-10 分钟完成 30 次请求；Markdown 表显示 PP=128/512/2048 三档下两个 target 的 TTFT/TG/E2E 中位数；数字落在 Qwen3.5-4B-MLX-4bit 在 M-series 单 GPU 的合理区间（TG 估 50-150 tok/s）。

---

## § 5 风险

| 风险 | 缓解 |
|---|---|
| **omlx prefix cache 命中影响 PP 测量** | prompt 含 per-run nonce；server 返回 `usage.cached_tokens > 0` 时打 warning（标在 Markdown 输出尾） |
| **Tokenizer round-trip 漂移** | 测试容忍 abs_diff ≤ 2；server 返回 `usage.prompt_tokens` 时优先用之，本地计数仅 fallback |
| **completion token 计数粗估**（chunk count ≠ token count） | `stream_options.include_usage=true` 触发 server 返回权威 `usage.completion_tokens`；ironmlx P4 polish 已加 OpenAI 非流 usage，stream 路径如缺失则 chunk count 兜底 — 实施时确认 |
| **HTTP overhead 偏倚** | localhost loopback ~0.1-0.5ms/req，对 PP/TG 各占 < 1%，且双 target 等额承担 |
| **首 token 检测漏 role chunk** | `delta.content` 为空时不触发 first_token；测试覆盖 |
| **Sampler 不一致** | 强制 `temperature=0, top_p=1`；两端均走 greedy，确定性 |
| **Network buffering 模糊化 TTFT** | reqwest stream 默认逐 chunk 暴露；如某 server 有 buffering 偏高 TTFT，仍**双向同等承担** |
| **omlx 不发 stream-end usage** | 已确认 omlx 默认 emit；如缺失则触发 chunk-count fallback，输出 warning 提示 |
| **Tokenizers crate 加载 Qwen3.5 tokenizer.json 失败** | Qwen3.5 是 standard BPE，已知 `tokenizers` crate 0.20 + onig features 兼容（ironmlx P4 测试链路验证过）|
| **iron-bench warmup 不足导致首测异常** | `--warmup 1` 默认；MLX compile 路径触发后第二次稳定。如需要更多 warmup，提示用户 `--warmup 2` |
| **多次重复间状态污染**（KV cache 溢出 / 显存不释放） | 每次 request 独立 HTTP 连接；不复用 session；server 端清理 cache 是 server 自己的责任 |

---

## § 6 实施任务划分（建议给 writing-plans）

| Task | 内容 | 时间 |
|---|---|---|
| **T1** | Workspace 加 `iron-bench` member；Cargo.toml；scaffold `main.rs` + clap CLI args + tokio runtime；hello-world 验证编译 | 0.5 d |
| **T2** | `prompt.rs` synthetic prompt + 2 单测（round-trip + zero target err）；`client.rs` HTTP + SSE parser + 3 单测（role chunk skip、event 跨 chunk、malformed payload skip） | 1 d |
| **T3** | `runner.rs` warmup + N 次重复 + RunOutcome 收集；不需新单测（runner 是 pure 配方代码） | 0.5 d |
| **T4** | `report.rs` stats reduction (median + p95) + Markdown / CSV / JSON 三种输出 + 2 单测（stats、CSV columns） | 0.5 d |
| **T5** | E2E 手测 + iron-bench README + commit | 0.5 d |
| **合计** | | **3 天** |

---

## § 7 验收标准

1. `cargo +nightly fmt --all -- --check && cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings && cargo build --release -p iron-bench` 全清洁
2. `cargo test --release -p iron-bench` —— 7 单测全过
3. 现有 ironmlx / mlx / mlx-sys 测试无回归（iron-bench 不依赖这些 crate，仅 workspace 共享 deps；理应不影响）
4. **手测验收**（Boss 跑）：
   ```
   ironmlx serve --model <snap> --port 8080 &
   omlx ... :8081 &
   cargo run --release -p iron-bench -- \
     --target ironmlx=http://localhost:8080 \
     --target omlx=http://localhost:8081 \
     --model-dir <snap> \
     --prompt-len 128,512,2048 --max-tokens 128 --runs 5 --warmup 1
   ```
   预期：~5-10 分钟完成 30 请求；Markdown 表显示两 target 在三档 PP 下的 TTFT/TG/E2E；数字落在 Qwen3.5-4B-MLX-4bit 在 M-series 合理区间（TG 估 50-150 tok/s）
5. CSV 输出可被 `pandas.read_csv()` 直接加载；列顺序在自动化脚本里稳定
6. ironmlx / mlx / mlx-sys 源码完全不变（iron-bench 是新 workspace member，无 hard 依赖）
7. `iron-bench` 不依赖 `ironmlx` crate；可独立用于驱动 mlx-lm-server / vllm-mlx / 任何 OpenAI 兼容 endpoint

---

## § 8 后续依赖

P7 完成后解锁：
- 第一轮 ironmlx vs omlx **MTP-off 单请求**对比数据
- **P8c** 实施完成后，相同 harness 可重跑测 with-MTP 加速（无需 iron-bench 改动）
- **P8b** 实施完成后，iron-bench v2 可加 `--concurrency N` 多并发测试（v1 不在范围）
- 任何新加入的 OpenAI 兼容 endpoint（mlx-lm-server / vllm-mlx / llama.cpp / 第三方云）都可直接接入对比，无需 iron-bench 代码改动
