# Gemma4 Drafter Batched Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement true Gemma4 drafter batched decode so `b_max > 1` can share drafter and verifier forward passes instead of serializing single-row schedulers.

**Architecture:** Keep this branch based on `dev` and do not merge adaptive admission logic into the implementation. Add batched external shared-KV support in the Gemma4 drafter/model layer, then replace the scheduler's per-row temp scheduler decode path with batched window fill before/after token emission. Preserve existing text, VL, paged prefix cache, prefix LRU, and active KV semantics by keeping prefill restore/save behavior unchanged and batching only the decode window path.

**Tech Stack:** Rust, MLX arrays, Gemma4 model/drafter modules, scheduler speculative decoding, existing `MtpSpeculativeStats`.

---

### Task 1: Model-Layer Batched Drafter Support

**Files:**
- Modify: `ironmlx/src/models/gemma4/drafter.rs`
- Modify: `ironmlx/src/models/gemma4/text_model.rs`

- [x] Add failing tests for batched drafter mask construction and shared-KV padding/slicing.
- [x] Implement helpers that stack per-row `Gemma4SharedKvStates` into padded batched `SharedKv` tensors.
- [x] Add batched mask construction using per-row query offsets, KV valid lengths, and per-row real KV lengths.
- [x] Add `Gemma4AssistantModel::forward_batched_on` and `Gemma4TextModel::forward_external_shared_kv_batched_on`.
- [x] Keep single-row `forward_on` behavior as a wrapper or unchanged fast path.

### Task 2: Row-Level Main KV Rollback

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [x] Add failing tests for restoring only selected compact rows from `LayerCacheSnapshot` offsets.
- [x] Implement a helper that restores selected compact cache rows without changing accepted rows.
- [x] Keep full-cache rollback helper unchanged for existing single-row/Qwen paths.

### Task 3: Scheduler Batched Drafter Window Fill

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [x] Add a batched Gemma4 drafter window-fill helper that drafts across rows per draft position.
- [x] Run one main-model batched verify over the full active cache layout, using zero per-row length for rows not being refilled.
- [x] Resolve acceptance per row, restore/replay mismatch rows only, and update per-row `last_hidden`, `shared_kv`, pending tokens, and adaptive draft budgets.
- [x] Replace `step_gemma4_drafter_batch_inner` temp single-row scheduler loop with: pre-fill empty queues, emit one token per active row, post-fill rows whose queue became empty.

### Task 4: Verification and Performance

**Files:**
- Modify tests as needed under existing Rust modules.

- [x] Run targeted unit tests for Gemma4 drafter helpers and scheduler rollback helpers.
- [x] Run `cargo fmt`.
- [x] Run `cargo +nightly fmt --all -- --check`.
- [x] Run `cargo +nightly clippy --all-features --workspace -- -D warnings`.
- [x] Run `cargo build --release`.
- [x] Benchmark Gemma4 E4B base + drafter with fixed `b_max=1`, fixed `b_max=2`, fixed `b_max=4`, and compare against the previous A/B baseline where possible.
- [x] Commit the implementation and report correctness and performance results.
