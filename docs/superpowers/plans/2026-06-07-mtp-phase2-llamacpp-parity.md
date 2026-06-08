# MTP Phase 2.5 Llama.cpp Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align ironmlx MTP runtime semantics with llama.cpp's Qwen3.5/Qwen3.6 MTP path enough to explain and reduce the Phase 2 performance gap.

**Architecture:** Keep Phase 2's single-request MTP scheduler surface, but add hidden-only MTP cache advancement so the MTP KV cache mirrors accepted target tokens instead of being restored to a stale prefix after every draft window. Add narrow timing counters to attribute draft, verify, projection, sampling, commit, and rollback costs without changing public serving behavior.

**Tech Stack:** Rust, MLX Rust bindings, ironmlx scheduler, Qwen3.5/Qwen3.6 dense/MoE MTP models, cargo tests.

---

### Task 1: Record the Phase 2.5 Scope

**Files:**
- Modify: `docs/superpowers/plans/2026-06-07-mtp-phase2-llamacpp-parity.md`

- [ ] **Step 1: Save this plan**

Run:

```bash
test -f docs/superpowers/plans/2026-06-07-mtp-phase2-llamacpp-parity.md
```

Expected: exit code 0.

### Task 2: Add Failing Tests for MTP Cache Commit Semantics

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`
- Modify: `ironmlx/src/core/speculative.rs`

- [ ] **Step 1: Add tests proving accepted verify inputs commit to MTP cache**

Add assertions to the existing scripted MTP scheduler tests so a full-accept d2 window leaves the fake MTP cache advanced by the accepted verify prefix length and a mismatch window advances only by the kept verify prefix length.

- [ ] **Step 2: Verify RED**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx scheduler_mtp
```

Expected: tests fail because current `draft_mtp_tokens_single` restores the MTP cache snapshot and no accepted-prefix commit path exists.

### Task 3: Add Hidden-Only MTP Forward Capability

**Files:**
- Modify: `ironmlx/src/core/speculative.rs`
- Modify: `ironmlx/src/models/qwen3_5/model.rs`
- Modify: `ironmlx/src/models/qwen3_5_moe/model.rs`
- Modify: `ironmlx/src/models/qwen3_6_moe/model.rs`

- [ ] **Step 1: Extend `MtpSpeculativeModel` with a required hidden-only method**

Add:

```rust
fn mtp_forward_hidden_on(
    &self,
    mtp: &Self::MtpHead,
    hidden_states: &Array,
    next_token_ids: &Array,
    position_ids: &Array,
    mask: Option<&Array>,
    mtp_cache: Option<&mut MtpCache>,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

- [ ] **Step 2: Implement model methods without vocab projection**

For dense and MoE Qwen MTP models, call the existing MTP module forward and return the hidden state directly. Keep `mtp_forward_on` as the logits-producing wrapper.

- [ ] **Step 3: Verify GREEN for model-level tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx p3b4_mtp
```

Expected: existing MTP module tests pass.

### Task 4: Commit Accepted Prefixes Into the MTP Cache

**Files:**
- Modify: `ironmlx/src/core/speculative.rs`
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Add `shift_hidden_for_mtp` helper**

Given `prev_hidden [1,1,H]` and `hidden [1,S,H]`, return `[prev_hidden, hidden[:,0:S-1,:]]`. This mirrors llama.cpp's target-hidden right shift into the MTP draft context.

- [ ] **Step 2: Add hidden-only MTP cache prefill during prompt prefill**

During prompt chunks, advance MTP cache with prompt token ids and shifted target hidden rows. Use a zero hidden row for the first prompt token, then carry the last target hidden row across chunks.

- [ ] **Step 3: Add accepted-prefix MTP cache commit after verification**

After restoring the temporary draft snapshot, replay only `verify_input[..accepted_verify_input_len]` into the MTP cache using shifted target hidden rows. Do not commit correction or bonus tokens, matching main-cache state.

- [ ] **Step 4: Verify GREEN for scheduler MTP tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx scheduler_mtp
```

Expected: new tests pass and existing scheduler MTP behavior stays lossless.

### Task 5: Add Attribution Counters

**Files:**
- Modify: `ironmlx/src/core/speculative.rs`
- Modify: `ironmlx/src/core/scheduler.rs`

- [ ] **Step 1: Extend `MtpSpeculativeStats`**

Add microsecond counters for draft, verify, projection, sampling, main rollback, MTP cache commit, and MTP cache restore.

- [ ] **Step 2: Update both scheduler and generation stream paths**

Record elapsed wall time around each phase. Keep counters additive and avoid changing request output semantics.

- [ ] **Step 3: Verify stats tests**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx mtp
```

Expected: MTP unit tests pass and stats remain non-breaking.

### Task 6: Full Rust Verification

**Files:**
- All modified Rust files

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt
```

Expected: command exits 0.

- [ ] **Step 2: Nightly format check**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
```

Expected: command exits 0.

- [ ] **Step 3: Clippy**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings
```

Expected: command exits 0.

- [ ] **Step 4: Release build**

Run:

```bash
MLX_DIR=$HOME/.local/mlx cargo build --release
```

Expected: command exits 0.
