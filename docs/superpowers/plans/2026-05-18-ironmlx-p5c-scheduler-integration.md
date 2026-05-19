# P5c — Scheduler / Server / CLI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P5a (trait) + P5b (MoE forward) 基础上把 Qwen35MoeModel 接入 Scheduler / SchedulerActor / AppState / CLI generate / CLI serve；按 `config["model_type"]` 自动分发；MoE-aware memory budget；HTTP smoke 跑通端到端 35B-A3B-4bit 推理。

**Architecture:** CLI 入口分发为单 model_type → 生成对应 generic instantiation；non-VL HTTP endpoint 同时支持 dense / MoE；VL endpoint 仅 dense（trait bound DenseVlMethods 守护，P5a 已建立）。

**Tech Stack:** Rust 1.94 / axum 0.7 / tokio / mlx (cxx-mlx wrapper)。

**Spec reference:** [docs/superpowers/specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md](../specs/2026-05-18-ironmlx-p5-qwen35-moe-design.md) §3.8 / §3.10

---

## Pre-flight

### Step 0.1: P5b 闭环条件确认

- [ ] 在 `ironmlx-p5-moe` 分支 + P5b smoke test 已 PASS

Run: `git log --oneline -8`
Expected: 看到 `p5b-*` commits 含 close-out

- [ ] working tree clean

Run: `git status --short`
Expected: 空

### Step 0.2: 基线（dense + MoE smoke 同步绿）

- [ ] dense regression + MoE smoke

```
cargo test -p ironmlx --lib --release
IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/$(ls ~/.ironmlx/models/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots | head -1) \
  cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
```
Expected: 两个都 PASS

---

## Task 1: CLI `generate` 子命令 model_type dispatch

**Files:**
- Modify: `ironmlx/src/cli/generate.rs`

- [ ] **Step 1.1: 引入 dispatch helper**

In `ironmlx/src/cli/generate.rs`，把 `pub fn run(args: GenerateArgs) -> Result<()>` 的中段（从 `let model = Qwen35Model::from_loader(&loader)?` 开始）抽出为 generic helper:
```rust
fn run_generation_with_model<M: ironmlx::core::Model>(
    model: &M,
    tokenizer: &Tokenizer,
    args: &GenerateArgs,
) -> Result<()> {
    let prompt = if args.chat && tokenizer.has_chat_template() {
        let messages = vec![Message {
            role: "user".into(),
            content: args.prompt.clone(),
        }];
        tokenizer.apply_chat_template(&messages, true, None)?
    } else {
        args.prompt.clone()
    };
    let prompt_ids = tokenizer.encode(&prompt, false)?;

    let mut sampler = Sampler::greedy();
    if args.temperature > 0.0 {
        sampler = sampler.with_temperature(args.temperature);
    }
    if args.top_p < 1.0 { sampler = sampler.with_top_p(args.top_p); }
    if args.seed != 0 { sampler = sampler.with_seed(args.seed); }

    let request = GenerateRequest {
        prompt_ids,
        max_new_tokens: args.max_tokens,
        sampler,
        stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        prefill_chunk_size: args.prefill_chunk_size,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: crate::core::generate::IMAGE_TOKEN_ID,
    };

    let mut stream = GenerationStream::new(model, tokenizer, request)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while let Some(ev) = stream.next_token()? {
        if !ev.text.is_empty() {
            out.write_all(ev.text.as_bytes())?;
            out.flush()?;
        }
        if ev.finish_reason.is_some() { break; }
    }
    writeln!(out)?;
    Ok(())
}
```

- [ ] **Step 1.2: `run` 函数按 model_type 分发**

替换 `pub fn run` 主体：
```rust
pub fn run(args: GenerateArgs) -> Result<()> {
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "--model must point to a local directory (got '{}')", args.model
        ));
    }
    let loader = Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = Tokenizer::from_loader(&loader).context("Tokenizer::from_loader")?;

    let model_type = loader.config_raw_value()
        .get("model_type").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;

    match model_type {
        "qwen3_5" => {
            let model = crate::models::Qwen35Model::from_loader(&loader)
                .context("Qwen35Model::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &args)
        }
        "qwen3_5_moe" => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)
                .context("Qwen35MoeModel::from_loader")?;
            run_generation_with_model(&model, &tokenizer, &args)
        }
        other => Err(anyhow::anyhow!(
            "unsupported model_type: {other} (expected 'qwen3_5' or 'qwen3_5_moe')"
        )),
    }
}
```

注意：原 `Qwen35Model::from_loader` 调用涉及 VL 检测（`Loader::open_multimodal`）。本步骤的简化把所有 VL 入口都改成 dense + Loader::open；如需 VL 路径，依然走 dense Qwen35Model（VL endpoint 是 dense-only by design，见 spec §3.9）。

- [ ] **Step 1.3: 验证 build + 文档 example 不破坏**

Run: `cargo build --release -p ironmlx`
Expected: 成功

- [ ] **Step 1.4: Commit T1**

```
git add ironmlx/src/cli/generate.rs
git commit -m "$(cat <<'EOF'
feat(p5c-t1): CLI generate dispatches by config["model_type"]

run() reads model_type from config.json and constructs either
Qwen35Model or Qwen35MoeModel before delegating to a generic
run_generation_with_model<M: Model> helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: CLI `serve` 子命令 model_type dispatch

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`

- [ ] **Step 2.1: 抽 generic spawn helper**

In `ironmlx/src/cli/serve.rs`，把 `Qwen35Model::from_loader` + `spawn_scheduler_actor` 的过程改 generic:
```rust
fn serve_with_model<M>(
    model: M,
    tokenizer: Tokenizer,
    args: &ServeArgs,
) -> Result<()>
where
    M: ironmlx::core::Model + Send + Sync + 'static,
{
    let model_arc = std::sync::Arc::new(tokio::sync::Mutex::new(model));
    // ... 原 spawn_scheduler_actor + axum 启动逻辑搬过来，参数化为 <M>
    todo!("paste original body, replacing Qwen35Model with M")
}
```

> **NOTE**: 完整 body 复制自原 `serve::run()`（line 60 之后），仅类型 `Qwen35Model` → `M` + 添加 generic 参数。

- [ ] **Step 2.2: `run` 函数 dispatch**

替换 `pub fn run` 主体：
```rust
pub fn run(args: ServeArgs) -> Result<()> {
    let model_dir = std::path::PathBuf::from(&args.model);
    let loader = ironmlx::core::Loader::open(&model_dir).context("Loader::open")?;
    let tokenizer = ironmlx::core::Tokenizer::from_loader(&loader)?;

    let model_type = loader.config_raw_value()
        .get("model_type").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;

    match model_type {
        "qwen3_5" => {
            let model = crate::models::Qwen35Model::from_loader(&loader)?;
            serve_with_model(model, tokenizer, &args)
        }
        "qwen3_5_moe" => {
            let model = crate::models::Qwen35MoeModel::from_loader(&loader)?;
            serve_with_model(model, tokenizer, &args)
        }
        other => Err(anyhow::anyhow!("unsupported model_type: {other}")),
    }
}
```

- [ ] **Step 2.3: 修编译错误（VL endpoint 路径）**

VL endpoint handler (`upload_image` / multipart 处理) 在 generic 化后会因 `M: DenseVlMethods` 缺失而编译失败。处理方式：
- 把 VL endpoint 的 axum 路由注册放在 `Qwen35Model` 专属代码段，仅当 `model_type == "qwen3_5"` 时挂载该路由
- MoE 路径上 VL endpoint 完全屏蔽（请求时返回 404 或显式 405）

具体代码改动（在 `serve_with_model` 内或 `serve` 的 dispatch 处）：
```rust
// 仅 dense 挂 VL 路由
let mut app = Router::new()
    .route("/v1/chat/completions", ...)
    .route("/v1/messages", ...)
    .route("/healthz", ...);
// VL upload 路由仅给 dense 加 — 通过两段不同的 serve_with_model 分别实现
// (具体 axum trait bound 约束在 dispatch 阶段已隔离)
```

如 VL endpoint 强行 generic 不可行（trait bound 卡死），干脆把 VL endpoint 注册移到 Qwen35Model 专属的 `serve_dense(...)` 内，MoE 走 `serve_moe(...)` 不挂 VL。两个函数共享 90% 代码段（提到 free function 重用）。

- [ ] **Step 2.4: 验证 build**

Run: `cargo build --release -p ironmlx`
Expected: 成功

- [ ] **Step 2.5: Commit T2**

```
git add ironmlx/src/cli/serve.rs
git commit -m "$(cat <<'EOF'
feat(p5c-t2): CLI serve dispatches by config["model_type"]

run() picks Qwen35Model or Qwen35MoeModel; serve_with_model<M>
hosts non-VL endpoints generically. VL endpoint registration
constrained to dense Qwen35Model code path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `memory_budget::ModelMeta` MoE 实测 + admit gate 校准

**Files:**
- Modify: `ironmlx/src/core/memory_budget.rs`（可能新增 test_meta_qwen35_moe）
- Modify: `ironmlx/tests/p5_qwen35_moe_smoke.rs`（加 memory budget cross-check）

- [ ] **Step 3.1: 添加 Qwen35MoeModel test ModelMeta helper**

Append to `ironmlx/src/core/memory_budget.rs`:
```rust
/// Realistic Qwen3.5-35B-A3B-4bit ModelMeta for tests.
#[doc(hidden)]
pub fn test_meta_qwen35_moe() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 40,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        hidden_size: 2048,
        head_dim: Some(256),
        // approx_weight_bytes formula:
        //   attn = 4 * 2048 * 2048 * 40 / 2 = 335M
        //   routed = 3 * 256 * 2048 * 512 * 40 / 2 ≈ 16.1 GB
        //   shared = 3 * 2048 * 512 * 40 / 2 = 63M
        //   embed_head = 2 * 248320 * 2048 / 2 ≈ 0.5 GB
        // total ≈ 17 GB
        weight_bytes: 17 * 1024 * 1024 * 1024,
    }
}
```

- [ ] **Step 3.2: 单测验证 MoE KV 公式不退化（KV cache 应与 dense 公式相同）**

Add to `memory_budget.rs` tests module:
```rust
#[test]
fn moe_kv_bytes_per_token_matches_gqa_formula() {
    let m = test_meta_qwen35_moe();
    // 40 layers × 2 KV heads × 256 head_dim × 2 (K+V) × 2 (bf16)
    let expected = 40 * 2 * 256 * 2 * 2;
    assert_eq!(kv_bytes_per_token(&m), expected);
}

#[test]
fn moe_validate_budget_realistic_32gb() {
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "34359738368"); // 32 GiB
    // 1 stream × 8K context should fit easily
    let st = validate_startup_budget(1, 8192, &test_meta_qwen35_moe()).expect("32GB fits");
    assert!(st.soft_limit() > 0);
    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
}

#[test]
fn moe_validate_budget_rejects_overcommit_16gb() {
    std::env::set_var("IRONMLX_TOTAL_RAM_BYTES", "17179869184"); // 16 GiB
    // 16 GB - 17 GB weight - 2 GB safety < 0, must reject any cap
    let err = validate_startup_budget(1, 4096, &test_meta_qwen35_moe())
        .expect_err("16GB cannot fit 17GB MoE weights");
    let msg = format!("{err}");
    assert!(msg.contains("memory budget exceeded"));
    std::env::remove_var("IRONMLX_TOTAL_RAM_BYTES");
}
```

- [ ] **Step 3.3: 跑 memory_budget unit test**

Run: `cargo test -p ironmlx --lib --release memory_budget::tests`
Expected: 全 PASS（含新增 3 个 moe 测试）

- [ ] **Step 3.4: Commit T3**

```
git add ironmlx/src/core/memory_budget.rs
git commit -m "$(cat <<'EOF'
test(p5c-t3): memory_budget MoE-aware test fixtures + admit gate

Adds test_meta_qwen35_moe (~17GB weight estimate) + three regression
tests: KV formula identical to dense (GQA), 32GB host fits 1×8K
context, 16GB host rejects any cap (correctly surfaces OOM risk).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: HTTP smoke 集成测试（end-to-end MoE serve）

**Files:**
- Create: `ironmlx/tests/p5_qwen35_moe_http_smoke.rs`

- [ ] **Step 4.1: 写 HTTP smoke test**

Create `ironmlx/tests/p5_qwen35_moe_http_smoke.rs`:
```rust
//! P5c HTTP smoke: launch ironmlx serve with Qwen35MoeModel, post
//! a chat completion, verify SSE stream completes with valid token output.
//!
//! Run with:
//!   IRONMLX_MOE_MODEL_DIR=<snapshot> \
//!     cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --nocapture
//!
//! 实测时间预算：模型加载 ~30s + 单 prompt 100 token decode ~30-60s。

use std::process::Stdio;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn p5c_http_smoke_chat_completion() {
    let dir = std::env::var("IRONMLX_MOE_MODEL_DIR").expect("IRONMLX_MOE_MODEL_DIR set");

    // 找一个未占用端口
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let p = l.local_addr().unwrap().port();
        drop(l);
        p
    };

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_ironmlx"))
        .args([
            "serve",
            "--model", &dir,
            "--port", &port.to_string(),
            "--b-max", "1",
            "--max-cache-cap", "4096",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn().expect("spawn ironmlx serve");

    // 等待 /healthz 200（最多 90s 模型加载）
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}");
    let mut up = false;
    for _ in 0..90 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(r) = client.get(format!("{url}/healthz")).send().await {
            if r.status().is_success() { up = true; break; }
        }
    }
    assert!(up, "serve did not become healthy within 90s");

    // post 一个最小 chat completion
    let body = serde_json::json!({
        "model": "qwen3_5_moe",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": false,
    });
    let resp = client.post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send().await.expect("post");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    let v: serde_json::Value = resp.json().await.expect("json");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(!content.is_empty(), "empty completion content");

    // 优雅关停
    child.kill().ok();
    child.wait().ok();
}
```

- [ ] **Step 4.2: 本地运行（注意 35B 加载内存 + 时间）**

Run:
```
IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha> \
  cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored --nocapture 2>&1 | tail -30
```

Expected: 模型加载 ~30s（首次可能 mmap warmup 60-90s）→ /healthz 通过 → chat completion 返回 5 token，无 panic / OOM。

如出现 OOM：Step 3.3 budget 公式应该已经在 admit 阶段拒绝；如未拒绝说明公式有问题，回 T3 校准 `approx_weight_bytes`。

- [ ] **Step 4.3: Commit T4**

```
git add ironmlx/tests/p5_qwen35_moe_http_smoke.rs
git commit -m "$(cat <<'EOF'
test(p5c-t4): MoE HTTP smoke chat completion end-to-end

#[ignore] integration test spawns ironmlx serve with the MoE
snapshot, waits for /healthz, posts a /v1/chat/completions
non-streaming request, asserts non-empty content. Validates
load-time + admit gate + scheduler step path on 35B-A3B-4bit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: batched_prefill MoE 等价验证

**Files:**
- Create: `ironmlx/tests/p5_qwen35_moe_batched.rs`

- [ ] **Step 5.1: 写 batched test**

Create `ironmlx/tests/p5_qwen35_moe_batched.rs`:
```rust
//! P5c batched_prefill MoE 等价：B=2 batch 输出应与 B=1 single-stream 对每行一致。
//!
//! Run: IRONMLX_MOE_MODEL_DIR=<snap> cargo test -p ironmlx --release \
//!   --test p5_qwen35_moe_batched -- --ignored --nocapture

use mlx::Dtype;
use ironmlx::core::{Loader, Model};
use ironmlx::core::generate::{build_position_ids_batched, build_batch_attention_mask, build_batch_linear_mask};
use ironmlx::models::Qwen35MoeModel;

#[test]
#[ignore]
fn p5c_batched_prefill_b2_equals_b1_per_row() {
    let dir = std::env::var("IRONMLX_MOE_MODEL_DIR").expect("set IRONMLX_MOE_MODEL_DIR");
    let loader = Loader::open(std::path::Path::new(&dir)).unwrap();
    let model = Qwen35MoeModel::from_loader(&loader).unwrap();

    let prompt_a = vec![100_i32, 200, 300, 400];   // len 4
    let prompt_b = vec![500_i32, 600, 700];        // len 3
    let max_len = 4_i32;

    // B=1 baseline 各跑一次
    let mut cache_a = Model::make_cache(&model, 1, max_len, Dtype::Bfloat16).unwrap();
    let inp_a: mlx::Array = (&prompt_a[..], &[1_i32, 4][..]).try_into().unwrap();
    let pos_a = ironmlx::core::generate::build_position_ids(0, 4).unwrap();
    let logits_a = Model::forward_on(&model, &inp_a, &pos_a, None, None, Some(&mut cache_a), ().into()).unwrap();

    let mut cache_b = Model::make_cache(&model, 1, max_len, Dtype::Bfloat16).unwrap();
    let inp_b: mlx::Array = (&prompt_b[..], &[1_i32, 3][..]).try_into().unwrap();
    let pos_b = ironmlx::core::generate::build_position_ids(0, 3).unwrap();
    let logits_b = Model::forward_on(&model, &inp_b, &pos_b, None, None, Some(&mut cache_b), ().into()).unwrap();

    // B=2 batched
    let prompt_lens = vec![4_i32, 3_i32];
    let mut flat = vec![0_i32; 2 * max_len as usize];
    flat[..4].copy_from_slice(&prompt_a);
    flat[max_len as usize..max_len as usize + 3].copy_from_slice(&prompt_b);
    let inp_batch: mlx::Array = (&flat[..], &[2_i32, max_len][..]).try_into().unwrap();
    let pos_batch = build_position_ids_batched(&prompt_lens, max_len).unwrap();
    let attn_mask = build_batch_attention_mask(&prompt_lens, max_len, Dtype::Bfloat16).unwrap();
    let lin_mask = build_batch_linear_mask(&prompt_lens, max_len).unwrap();
    let mut cache_batch = Model::make_cache(&model, 2, max_len, Dtype::Bfloat16).unwrap();
    let logits_batch = Model::batched_prefill(
        &model, &inp_batch, &pos_batch, &attn_mask, &lin_mask,
        &prompt_lens, Some(&mut cache_batch), ().into(),
    ).unwrap();

    // 取 batched 第 0/1 行最后位置 logits 与 single-stream 各自比对
    let la: Vec<f32> = mlx::ops::cast::astype(&logits_a, Dtype::Float32).unwrap().to_vec().unwrap();
    let lb: Vec<f32> = mlx::ops::cast::astype(&logits_b, Dtype::Float32).unwrap().to_vec().unwrap();
    let lbatch: Vec<f32> = mlx::ops::cast::astype(&logits_batch, Dtype::Float32).unwrap().to_vec().unwrap();
    let vocab = la.len();
    assert_eq!(lbatch.len(), 2 * vocab);

    let mut max_diff_a = 0.0_f32;
    let mut max_diff_b = 0.0_f32;
    for i in 0..vocab {
        max_diff_a = max_diff_a.max((la[i] - lbatch[i]).abs());
        max_diff_b = max_diff_b.max((lb[i] - lbatch[vocab + i]).abs());
    }
    // argmax 必须 bit-identical；logits max_abs 容忍 bf16 round-trip
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(argmax(&la), argmax(&lbatch[..vocab]), "row 0 argmax mismatch");
    assert_eq!(argmax(&lb), argmax(&lbatch[vocab..]), "row 1 argmax mismatch");
    assert!(max_diff_a < 1e-3, "row 0 max_abs_diff = {max_diff_a}");
    assert!(max_diff_b < 1e-3, "row 1 max_abs_diff = {max_diff_b}");
}
```

- [ ] **Step 5.2: 跑测试**

```
IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored --nocapture
```
Expected: PASS

如失败说明 batched_prefill 路径与 single-stream forward_on 在 MoE 下不等价，分析点：
- routing 是否依赖 batch position（不应依赖）
- attention mask 是否影响 routing（不影响 — router 是 token-wise）
- per-row last-position 切片是否正确

- [ ] **Step 5.3: Commit T5**

```
git add ironmlx/tests/p5_qwen35_moe_batched.rs
git commit -m "$(cat <<'EOF'
test(p5c-t5): MoE batched_prefill B=2 ≡ B=1 per-row equivalence

Verifies batched_prefill argmax bit-identical to single-stream
forward_on on each row, with logits max_abs_diff < 1e-3 (bf16
roundoff tolerance). Catches batch-position-dependent routing
bugs and per-row last-position slicing regressions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 工具链 hygiene + close-out

**Files:** N/A（cross-cutting）

- [ ] **Step 6.1: 完整 build + test**

```
cargo build --release -p ironmlx
cargo test -p ironmlx --lib --release
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
```
Expected: lib unit 全 PASS（含 memory_budget 新增）；clippy 零 warning；fmt OK。

- [ ] **Step 6.2: 集成测试 sanity**

仅在本地有 snapshot 时跑：
```
export IRONMLX_MOE_MODEL_DIR=~/.ironmlx/models/.../snapshots/<sha>
cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored
cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored
```
Expected: 三个集成 test 全 PASS。

- [ ] **Step 6.3: 跑 dense p4_http_smoke 确保 P5c 改动未破坏 dense 路径**

```
export IRONMLX_MODEL_DIR=~/.ironmlx/models/.../models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/<sha>
cargo test -p ironmlx --release --test p4_http_smoke -- --ignored
```
Expected: PASS（dense regression 守门）

- [ ] **Step 6.4: close-out commit**

```
git add -A
git commit -m "$(cat <<'EOF'
chore(p5c): close-out — MoE scheduler/server/CLI integration

CLI generate + serve dispatch by model_type. memory_budget MoE
fixtures + admit gate validated on 32GB/16GB scenarios. HTTP smoke
+ batched_prefill MoE equivalence integration tests pass locally.
dense regression (p4_http_smoke) unaffected.

P5c sub-phase complete. P5d can proceed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## P5c 闭环条件

- [ ] `cargo test -p ironmlx --lib --release` PASS（含新增 memory_budget MoE tests）
- [ ] `IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_smoke -- --ignored` PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_batched -- --ignored` PASS
- [ ] `IRONMLX_MOE_MODEL_DIR=... cargo test -p ironmlx --release --test p5_qwen35_moe_http_smoke -- --ignored` PASS
- [ ] dense `p4_http_smoke` 不退化
- [ ] clippy / fmt / release build 三 hygiene PASS

满足全部 → P5d 启动。

---

## Self-Review Notes

- ✓ Spec coverage：CLI dispatch §3.10 / scheduler+server generic 已在 P5a 完成，本 phase 只补 CLI dispatch 入口；memory_budget MoE §3.8；HTTP smoke / batched equivalence 验证 §4.2
- ✓ VL endpoint 守护策略：dispatch 后仅 dense 挂 VL 路由（与 P5a DenseVlMethods 配套）
- ✓ Task 数 = 6 + Pre-flight，符合 5-7 范围
