# Gemma4 Drafter Prefix Cache Design

## Goal

Implement production support for Gemma4 assistant drafter serving with paged SSD prefix cache and prefix LRU cache for `b_max=1`. The implementation must cover both text requests and Gemma4 VL requests that enter through OpenAI and Anthropic server paths.

## Scope

- In scope: Gemma4 dense target model plus Gemma4 assistant drafter, single in-flight request (`b_max=1`), text prompt prefill, VL prompt prefill, paged SSD prefix cache, and in-memory prefix LRU cache.
- Out of scope: Gemma4 drafter continuous batching (`b_max>1`) and active KV offload. These require a separate scheduler-level drafter architecture.
- No compatibility shim: old prefix cache entries do not need migration. New fields are schema-versioned.

## Root Cause

`build_gemma4_drafter_app_state` currently rejects `paged_prefix_cache` and `prefix_lru_cache` before serving starts. Qwen MTP does not hit this because Qwen MTP uses scheduler-owned MTP cache helpers that save and restore main KV, MTP KV, and `mtp_last_hidden`. Gemma4 assistant drafter cannot reuse those fields directly because it has no independent MTP KV cache; it consumes target-model shared KV plus the previous target hidden state.

## Architecture

Add a Gemma4 drafter-specific prefix payload to `PagedPrefixEntry`:

- `main_layers`: existing target-model KV prefix layers.
- `gemma4_drafter_last_hidden`: the target hidden state at the last cached token.
- no `mtp_layers`: Gemma4 assistant shared KV is reconstructed from target main KV cache, not persisted as a second cache.

At request prefill:

1. Allocate the Gemma4 target KV cache with paged KV enabled when paged prefix cache is configured.
2. Compute a text or VL prefix fingerprint. VL fingerprints include image token id, spatial merge size, image grids, pixel tensor shapes, dtypes, and tensor bytes.
3. Try prefix LRU first, then SSD store, using the same longest-prefix search semantics as scheduler prefix cache.
4. Restore target main KV into cache row 0.
5. Rebuild `Gemma4SharedKvStates` by materializing the restored K/V-owning Gemma4 layers and inserting the last K/V source per `Gemma4LayerKind`.
6. Restore `gemma4_drafter_last_hidden`.
7. Resume prefill from `restore_len`; VL resumes with `image_pad_consumed` set to the image-pad count before the restored prefix.
8. Save reusable prefixes during cold prefill. If the next chunk would end exactly at prompt end, split at `prompt_len - 1` so the prompt's last token can be generated from a reusable prefix hit on the next request.

## Files

- `ironmlx/src/core/cache/prefix_store.rs`: add drafter payload spec, metadata, validation, safe-tensor save/load, and tests.
- `ironmlx/src/core/cache/mod.rs`: re-export the new prefix-store types.
- `ironmlx/src/models/gemma4/drafter.rs`: add prefix runtime configuration, restore/save helpers, shared-KV reconstruction, and text/VL prefill integration.
- `ironmlx/src/core/server/mod.rs`: allow paged prefix cache and prefix LRU for Gemma4 drafter at `b_max=1`; keep the active KV offload and `b_max>1` guards.
- `ironmlx/src/core/server/openai.rs`: pass drafter prefix runtime into streaming and unary Gemma4 drafter streams.
- `ironmlx/src/core/server/anthropic.rs`: pass drafter prefix runtime into streaming and unary Gemma4 drafter streams.

## Correctness Requirements

- A Gemma4 drafter prefix entry must not be keyed as Qwen MTP. The SSD key must include the drafter hidden tensor metadata.
- Prefix LRU and SSD entries must have identical hit semantics.
- Restored cache offsets must equal `restore_len` for every Gemma4 K/V-owning layer.
- Restored `Gemma4SharedKvStates` must contain every layer kind required by the assistant drafter.
- Text and VL requests must produce the same token stream with and without a prefix hit under greedy sampling.
- Existing Qwen MTP prefix cache tests must keep passing.
- Existing non-drafter prefix cache entries must not be treated as Gemma4 drafter hits.

## Validation

- Baseline before changes: `cargo build` and `cargo test --workspace --all-features`.
- Unit tests for prefix-store schema/key/validation round trip.
- Unit tests for Gemma4 shared-KV reconstruction using synthetic KV caches.
- Server/CLI tests proving Gemma4 drafter accepts paged prefix cache and prefix LRU while still rejecting `b_max>1` and active KV offload.
- Real-model smoke for local Gemma4 E4B text and VL where model files are available.
- Final Rust gate: `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.
