# Scheduler Actor MTP Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `ironmlx serve --mtp-model-dir ...` to use scheduler-internal MTP for eligible single-request text Qwen requests.

**Architecture:** Keep the existing non-MTP `spawn_scheduler_actor` and `serve` path intact. Add an MTP-specific actor spawn path for `M: MtpSpeculativeModel`, route only `b_max=1` text-only greedy batches to `prefill_admitted_mtp_single` / `step_mtp_single`, and expose doc-hidden counters for tests.

**Tech Stack:** Rust, MLX, Tokio, Axum, existing `Scheduler`, `MtpSpeculativeModel`, `MtpSpeculativeConfig`, `Loader::open_mtp`, and Qwen dense/MoE model loaders.

---

## File Structure

- Modify `ironmlx/src/cli/serve.rs`
  - Add CLI args `--mtp-model-dir` and `--mtp-draft-tokens`.
  - Add serve MTP config validation.
  - Load MTP head only in Qwen dense/MoE branches.
  - Call MTP serve helper for Qwen when configured.

- Modify `ironmlx/src/core/server/mod.rs`
  - Add `MtpServeConfig<H>`.
  - Add `serve_with_mtp`.
  - Factor common serve body through a private actor spawner trait.

- Modify `ironmlx/src/core/server/scheduler_actor.rs`
  - Add actor MTP mode.
  - Add `spawn_scheduler_actor_with_mtp`.
  - Share `driver_loop` across MTP and non-MTP actor modes.
  - Increment doc-hidden MTP counters.
  - Add actor-level unit tests.

- Modify `ironmlx/src/core/scheduler.rs`
  - Add `mtp_single_active_text_greedy_eligible`.
  - Add focused eligibility tests.

---

### Task 1: Serve CLI MTP Gating RED Tests

**Files:**
- Modify: `ironmlx/src/cli/serve.rs`

- [ ] **Step 1: Add failing tests**

Add tests in `scheduler_profile_tests`:

```rust
#[test]
fn serve_mtp_args_default_off() {
    let args = base_args();

    assert!(args.mtp_model_dir.is_none());
    assert_eq!(args.mtp_draft_tokens, 1);
}

#[test]
fn serve_mtp_config_accepts_qwen_single_request_window() {
    let temp_dir = unique_temp_dir("serve-mtp-ok");
    std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
    let mut args = base_args();
    args.mtp_model_dir = Some(temp_dir.clone());
    args.mtp_draft_tokens = 2;

    let cfg = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Qwen35Dense,
        SchedulerServeConfig {
            b_max: 1,
            ..SchedulerServeConfig::default()
        },
    )
    .expect("resolve")
    .expect("enabled");

    assert_eq!(cfg.model_dir, temp_dir);
    assert_eq!(cfg.draft_tokens, 2);
    std::fs::remove_dir_all(cfg.model_dir).expect("cleanup");
}

#[test]
fn serve_mtp_config_rejects_batched_scheduler() {
    let temp_dir = unique_temp_dir("serve-mtp-bmax");
    std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
    let mut args = base_args();
    args.mtp_model_dir = Some(temp_dir.clone());

    let err = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Qwen35Dense,
        SchedulerServeConfig {
            b_max: 2,
            ..SchedulerServeConfig::default()
        },
    )
    .expect_err("b_max > 1 must be rejected");

    assert!(err.to_string().contains("--b-max 1"));
    std::fs::remove_dir_all(temp_dir).expect("cleanup");
}

#[test]
fn serve_mtp_config_rejects_non_qwen_architecture() {
    let temp_dir = unique_temp_dir("serve-mtp-non-qwen");
    std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
    let mut args = base_args();
    args.mtp_model_dir = Some(temp_dir.clone());

    let err = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Llama,
        SchedulerServeConfig {
            b_max: 1,
            ..SchedulerServeConfig::default()
        },
    )
    .expect_err("non-Qwen must be rejected");

    assert!(err.to_string().contains("Qwen"));
    std::fs::remove_dir_all(temp_dir).expect("cleanup");
}

#[test]
fn serve_mtp_config_rejects_missing_dir_and_zero_draft_tokens() {
    let mut args = base_args();
    args.mtp_model_dir = Some(PathBuf::from("/tmp/ironmlx-missing-mtp-dir"));
    let missing = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Qwen35Dense,
        SchedulerServeConfig {
            b_max: 1,
            ..SchedulerServeConfig::default()
        },
    )
    .expect_err("missing dir");
    assert!(missing.to_string().contains("local directory"));

    let temp_dir = unique_temp_dir("serve-mtp-zero-draft");
    std::fs::create_dir_all(&temp_dir).expect("create mtp dir");
    args.mtp_model_dir = Some(temp_dir.clone());
    args.mtp_draft_tokens = 0;
    let zero = resolve_serve_mtp_config(
        &args,
        crate::models::ModelArchitecture::Qwen35Dense,
        SchedulerServeConfig {
            b_max: 1,
            ..SchedulerServeConfig::default()
        },
    )
    .expect_err("zero draft tokens");
    assert!(zero.to_string().contains("max_draft_tokens must be > 0"));
    std::fs::remove_dir_all(temp_dir).expect("cleanup");
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib serve_mtp_config
```

Expected: compile/test failure because fields/functions do not exist.

- [ ] **Step 3: Implement CLI config**

Add `ServeMtpConfig`, args fields, `resolve_serve_mtp_config`, and update `base_args`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib serve_mtp_config
cargo fmt
git add ironmlx/src/cli/serve.rs
git commit -m "feat: gate serve mtp config"
```

---

### Task 2: Scheduler Eligibility RED Tests

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Add failing tests**

Add tests near existing `mtp_` scheduler tests:

```rust
#[test]
fn mtp_single_eligibility_accepts_single_text_greedy_request() {
    let model = ScriptedMtpSchedulerModel::new(3, vec![4], vec![vec![4, 5]]);
    let mut scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        16,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler");
    scheduler.admit(mk_request(vec![1, 2], 4)).expect("admit");

    assert!(scheduler.mtp_single_active_text_greedy_eligible());
    drop(model);
}

#[test]
fn mtp_single_eligibility_rejects_vl_and_non_greedy_requests() {
    let mut vl_scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        16,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler");
    let mut vl = mk_request(vec![1, 2], 4);
    vl.pixel_values = Some(vec![mlx::Array::zeros((1_i32,), mlx::Dtype::Float32).unwrap()]);
    vl.image_grid_thw = Some(vec![(1, 1, 1)]);
    vl_scheduler.admit(vl).expect("admit");
    assert!(!vl_scheduler.mtp_single_active_text_greedy_eligible());

    let mut sampling_scheduler = Scheduler::<ScriptedMtpSchedulerModel>::new(
        1,
        16,
        crate::core::memory_budget::test_meta_qwen35(),
    )
    .expect("scheduler");
    let mut sampled = mk_request(vec![1, 2], 4);
    sampled.sampler = crate::core::sampler::Sampler::greedy().with_temperature(0.7);
    sampling_scheduler.admit(sampled).expect("admit");
    assert!(!sampling_scheduler.mtp_single_active_text_greedy_eligible());
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_single_eligibility
```

Expected: compile failure because helper does not exist.

- [ ] **Step 3: Implement eligibility helper**

Add `pub(crate) fn mtp_single_active_text_greedy_eligible(&self) -> bool` to `Scheduler<M>`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_single_eligibility
cargo fmt
git add ironmlx/src/core/scheduler.rs
git commit -m "feat: expose scheduler mtp eligibility"
```

---

### Task 3: Actor MTP Mode RED Tests

**Files:**
- Modify: `ironmlx/src/core/server/scheduler_actor.rs`
- Modify: `ironmlx/src/core/server/mod.rs`

- [ ] **Step 1: Add actor MTP tests**

Add a fake `MtpSpeculativeModel` implementation for `SchedulerActorFakeModel`, then add tests:

```rust
#[test]
fn actor_mtp_mode_prefill_uses_mtp_for_eligible_request() {
    let mode = SchedulerActorMtpMode::enabled(FakeMtpHead, 1);
    assert!(mode.is_enabled());
}
```

Then expand to a direct scheduler-mode test that admits a text greedy request and calls the mode prefill/step helpers, asserting `mtp_prefill_count == 1`.

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode
```

Expected: compile failure because actor MTP mode does not exist.

- [ ] **Step 3: Implement actor MTP mode**

Add:

```rust
struct SchedulerActorNoMtp;
struct SchedulerActorMtp<H> { mtp: H, cfg: MtpSpeculativeConfig }
trait SchedulerActorMtpMode<M: Model> { ... }
```

Change `driver_loop` and `drive_empty_scheduler_handoff` to call mode prefill/step helpers.

- [ ] **Step 4: Add spawn function and counters**

Add `spawn_scheduler_actor_with_mtp` and `mtp_prefill_count` / `mtp_step_count` on `SchedulerActorHandle`.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib scheduler_actor
cargo fmt
git add ironmlx/src/core/server/scheduler_actor.rs ironmlx/src/core/server/mod.rs
git commit -m "feat: add scheduler actor mtp mode"
```

---

### Task 4: Serve Wiring

**Files:**
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/src/cli/serve.rs`

- [ ] **Step 1: Add `serve_with_mtp`**

Factor existing `serve` through a private `serve_inner` with a scheduler actor spawner. Add public `serve_with_mtp` requiring `M: MtpSpeculativeModel`.

- [ ] **Step 2: Wire Qwen branches**

In Qwen dense/MoE branches, if `resolve_serve_mtp_config` returns `Some`, load the MTP head and call `serve_with_mtp`; otherwise call existing `serve_with_model`.

- [ ] **Step 3: Verify and commit**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib serve_mtp_config
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_
cargo fmt
git add ironmlx/src/core/server/mod.rs ironmlx/src/cli/serve.rs
git commit -m "feat: wire serve mtp actor path"
```

---

### Task 5: Full Verification And Smoke

**Files:**
- No source changes expected.

- [ ] **Step 1: Required Rust checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release
```

- [ ] **Step 2: Focused tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib mtp_
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib serve_mtp_config
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib scheduler_actor
```

- [ ] **Step 3: Real serve smoke**

Start:

```bash
MLX_DIR=$HOME/.local/mlx target/release/ironmlx serve \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --host 127.0.0.1 \
  --port 18080 \
  --b-max 1 \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485 \
  --mtp-draft-tokens 1
```

Request:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Write one short sentence about MLX."}],"max_tokens":8,"stream":false}'
```

Expected: HTTP 200 JSON with `choices[0].message.content`.

- [ ] **Step 4: Negative startup smoke**

Run:

```bash
MLX_DIR=$HOME/.local/mlx target/release/ironmlx serve \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3 \
  --host 127.0.0.1 \
  --port 18081 \
  --b-max 2 \
  --mtp-model-dir /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.5-4B-MTP-4bit/snapshots/ab6f59bc6627196c611ab8851638651078170485
```

Expected: process exits with error mentioning `--b-max 1`.

- [ ] **Step 5: Final git checks and push**

Run:

```bash
git diff --check
git status --short
git push -u origin codex/mtp-phase2-actor-gating
```
