# Paged SSD Prefix Cache V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete paged SSD prefix cache coverage for Full-KV, MLA, Linear, VL, and MTP paths. Full-KV decode must continue to use the paged attention kernel.

**Architecture:** Replace the Full-KV-only SSD store schema with a v2 cache-state payload. Prefix keys include the exact cached token prefix length and an optional non-text fingerprint. Main model layers save per-layer payloads by cache kind: Full-KV as paged K/V pages, MLA as latent `c_kv/k_pe`, and Linear as conv/recurrent state. MTP stores the main cache payload plus MTP K/V layers and the prefix `last_hidden` needed to resume draft state. VL requests include image/grid/tokenizer fingerprint material in the key so token-identical prompts with different images never collide.

**Tech Stack:** Rust 1.94, MLX arrays/safetensors, existing `KVCache`, `GatedDeltaCache`, `MlaLatentCache`, `MtpCache`, scheduler single-row prefill and mid-admit paths.

### Task 1: Prefix Store V2

**Files:**
- Modify: `ironmlx/src/core/cache/prefix_store.rs`
- Modify: `ironmlx/src/core/cache/mod.rs`

- [x] Add v2 key metadata: `cached_len`, `fingerprint`, and per-layer payload kind.
- [x] Add typed payloads for Full paged pages, Linear state, MLA latent state, MTP dense K/V layers, and MTP `last_hidden`.
- [x] Save/load payload tensors through one safetensors file plus exact metadata validation.
- [x] Keep schema v2 strict; do not add v1 compatibility code.

### Task 2: Cache Export/Restore APIs

**Files:**
- Modify: `ironmlx/src/core/cache/kv_cache.rs`
- Modify: `ironmlx/src/core/cache/gated_delta.rs`
- Modify: `ironmlx/src/core/cache/mtp_cache.rs`
- Modify: `ironmlx/src/models/glm4_moe_lite/mla_cache.rs`
- Modify: `ironmlx/src/nn/decoder_layer.rs`

- [x] Export/restore Full-KV prefix pages for paged main caches.
- [x] Export/restore dense KV prefix rows for MTP caches.
- [x] Export/restore Linear conv/recurrent state with logical cached offset.
- [x] Export/restore MLA latent row prefix.
- [x] Provide mixed-layer scheduler helpers over `LayerCache`.

### Task 3: Scheduler Prefix Semantics

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`

- [x] Restore by trying token prefixes `prompt[..restore_len]` with `cached_len=restore_len`, from longest usable prefix down to 1.
- [x] Save the prompt prefix before the last token, then optionally save the full prompt after logits.
- [x] Enable VL prefix save/restore with image/grid/tokenizer fingerprint, not token-only keys.
- [x] Apply the same prefix-length semantics to fresh prefill and mid-admit.

### Task 4: MTP Prefix Payload

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs`
- Modify: `ironmlx/src/core/cache/mtp_cache.rs`

- [x] During MTP prefill, attempt to restore main cache, MTP cache, and `last_hidden`.
- [x] Split final prompt work so the `prompt_len - 1` state can be saved for exact repeat hits.
- [x] Save full MTP prefix states for future longer prompt hits.

### Task 5: Tests And Verification

**Files:**
- Modify/add focused tests near touched modules.

- [x] Prefix store v2 mixed payload round-trip and key separation tests.
- [x] Cache export/restore tests for Full-KV, Linear, MLA, and MTP dense KV.
- [x] Scheduler tests for prefix-length semantics, VL fingerprint separation, and MTP restore.
- [x] Run `MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx prefix`.
- [x] Run `MLX_DIR=$HOME/.local/mlx cargo test -p mlx --test paged_attention`.
- [x] Run `cargo fmt`.
- [x] Run `cargo +nightly fmt --all -- --check`.
- [x] Run `MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace -- -D warnings`.
- [x] Run `MLX_DIR=$HOME/.local/mlx cargo build --release`.

### Completion Notes

Completed on 2026-06-18 on branch `codex/paged-ssd-prefix-cache`.

Additional coverage added during final validation:

- B>1 MTP scheduler path: `SchedulerMtpState` is row-scoped and actor prefill/step use `prefill_admitted_mtp_batch` / `step_mtp_batch`.
- `ironmlx-core-bench --mode scheduler-text --b-max 2 --mtp-model-dir ...` now exercises the batch MTP scheduler API instead of the old B=1-only path.
- `serve --mtp-model-dir ... --b-max 2 --paged-prefix-cache-dir ...` is supported; the MTP actor now receives the paged prefix cache config.

Real-model smoke results:

- Qwen3.5-4B + Qwen3.5-MTP: `scheduler-text --b-max 2 --mtp-draft-tokens 1` produced 2 tokens with MTP stats present.
- Qwen3.5-4B server: `serve --b-max 2 --mtp-model-dir ... --paged-prefix-cache-dir ...` served two repeated OpenAI chat completions; `/healthz.mtp.prefill_count` and `/healthz.mtp.step_count` reached 2, and debug logs showed `paged SSD prefix cache MTP hit: tokens=16 restored=16`.
- GLM-4.7-Flash (MLA): `scheduler-text --b-max 2` produced 2 valid tokens.
- MiniCPM5-1B (Linear): `scheduler-text --b-max 2` produced 2 valid tokens.
- MiniCPM-V-4.6 (VL-capable): text scheduler smoke with `--b-max 2` produced 2 valid tokens; VL-specific cache key separation is covered by unit tests.
