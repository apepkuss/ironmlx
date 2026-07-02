# Gemma4 Drafter Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade paged SSD prefix cache and prefix LRU support to Gemma4 assistant drafter serving for `b_max=1`.

**Architecture:** Extend the prefix-store schema with a Gemma4 drafter last-hidden payload, then make `Gemma4DrafterGenerationStream` restore/save target KV prefixes and rebuild Gemma4 shared KV from the restored target cache. Server state owns a drafter prefix runtime and passes it through both OpenAI and Anthropic paths.

**Tech Stack:** Rust, MLX arrays, ironmlx prefix store, Gemma4 dense model, Gemma4 assistant drafter, Axum server paths.

## Global Constraints

- Worktree: `/Users/xin/workspace/ironmlx-backend-gemma4-drafter-prefix-cache`.
- Branch: `feat/gemma4-drafter-prefix-cache`.
- Phase 1 only: `b_max=1`.
- Must cover Gemma4 text and Gemma4 VL request paths.
- Must support paged SSD prefix cache and prefix LRU cache.
- Must keep active KV offload unsupported for Gemma4 drafter.
- Must not add backward-compatibility migration code.
- Rust verification must include `cargo fmt`, `cargo +nightly fmt --all -- --check`, `cargo +nightly clippy --all-features --workspace -- -D warnings`, and `cargo build --release`.
- Do not open a PR.

---

### Task 1: Prefix Store Drafter Payload

**Files:**
- Modify: `ironmlx/src/core/cache/prefix_store.rs`
- Modify: `ironmlx/src/core/cache/mod.rs`

**Interfaces:**
- Produces: `PagedPrefixKeySpec::gemma4_drafter_last_hidden: Option<PrefixTensorSpec>`
- Produces: `PagedPrefixEntry::gemma4_drafter_last_hidden: Option<Array>`
- Produces: `PagedPrefixEntry::gemma4_drafter_last_hidden_spec(&self) -> Option<PrefixTensorSpec>`

- [ ] **Step 1: Write the failing prefix-store tests**

Add tests named:

```rust
#[test]
fn prefix_store_round_trips_gemma4_drafter_last_hidden_without_mtp_layers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = PagedPrefixStore::new(PagedPrefixCacheConfig::new(
        dir.path().to_path_buf(),
        "gemma4-drafter".to_owned(),
        4,
    ).expect("config"));
    let mut entry = single_full_paged_entry(1.0);
    let hidden: Array = (&[0.25_f32, 0.5, 0.75, 1.0][..], &[1_i32, 1_i32, 4_i32][..])
        .try_into()
        .expect("hidden");
    entry.gemma4_drafter_last_hidden = Some(hidden.clone());
    let mut spec = key_spec_for(&entry, &[1, 2, 3, 4], 4);
    spec.gemma4_drafter_last_hidden = entry.gemma4_drafter_last_hidden_spec();

    let key = store.save(&spec, &entry).expect("save");
    let loaded = store.load(&spec).expect("load").expect("entry");

    assert_eq!(key, PagedPrefixStore::key_for(&spec));
    assert_eq!(
        loaded
            .gemma4_drafter_last_hidden
            .expect("loaded hidden")
            .shape()
            .as_slice(),
        &[1, 1, 4]
    );
    assert!(loaded.mtp_layers.is_empty());
    assert!(loaded.mtp_last_hidden.is_none());
}

#[test]
fn prefix_key_distinguishes_gemma4_drafter_hidden_from_plain_prefix() {
    let entry = single_full_paged_entry(2.0);
    let plain = key_spec_for(&entry, &[9, 8, 7, 6], 4);
    let mut drafter = plain.clone();
    drafter.gemma4_drafter_last_hidden = Some(PrefixTensorSpec {
        dtype: Dtype::Float32,
        shape: vec![1, 1, 4],
    });

    assert_ne!(
        PagedPrefixStore::key_for(&plain),
        PagedPrefixStore::key_for(&drafter)
    );
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test -p ironmlx core::cache::prefix_store::tests::prefix_store_round_trips_gemma4_drafter_last_hidden_without_mtp_layers core::cache::prefix_store::tests::prefix_key_distinguishes_gemma4_drafter_hidden_from_plain_prefix`

Expected: FAIL because `gemma4_drafter_last_hidden` fields do not exist.

- [ ] **Step 3: Implement prefix-store payload**

Add the new field to key material, metadata, entry tensors, load/save, stats, and validation. Use tensor name `gemma4_drafter.last_hidden`. Keep existing MTP validation intact.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `cargo test -p ironmlx core::cache::prefix_store::tests::prefix_store_round_trips_gemma4_drafter_last_hidden_without_mtp_layers core::cache::prefix_store::tests::prefix_key_distinguishes_gemma4_drafter_hidden_from_plain_prefix`

Expected: PASS.

### Task 2: Gemma4 Shared-KV Reconstruction

**Files:**
- Modify: `ironmlx/src/models/gemma4/drafter.rs`

**Interfaces:**
- Produces: `fn gemma4_shared_kv_from_cache(model: &Gemma4Model, cache: &[LayerCache], row: usize, target: impl Into<StreamOrDevice>) -> Result<Gemma4SharedKvStates>`
- Produces: helper that materializes `LayerCache::Full` from dense, TurboQuant, or paged storage.

- [ ] **Step 1: Write failing tests**

Add synthetic tests that create a small Gemma4-like cache layout with one sliding and one full layer, restore prefix offsets, then assert `Gemma4SharedKvStates::require(Gemma4LayerKind::Sliding)` and `require(Gemma4LayerKind::Full)` both succeed.

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test -p ironmlx models::gemma4::drafter::tests::gemma4_shared_kv_rebuilds_from_dense_main_cache models::gemma4::drafter::tests::gemma4_shared_kv_rebuilds_from_paged_main_cache`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement reconstruction**

Iterate over cache layers up to `model.config().first_kv_shared_layer_idx()`, materialize `[1, heads, cached_len, dim]` K/V, and insert into `Gemma4SharedKvStates` by `model.config().layer_kind(layer_idx)`. Return an error if no required kind is present.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `cargo test -p ironmlx models::gemma4::drafter::tests::gemma4_shared_kv_rebuilds_from_dense_main_cache models::gemma4::drafter::tests::gemma4_shared_kv_rebuilds_from_paged_main_cache`

Expected: PASS.

### Task 3: Drafter Prefix Restore/Save Runtime

**Files:**
- Modify: `ironmlx/src/models/gemma4/drafter.rs`
- Modify: `ironmlx/src/core/cache/mod.rs` if extra public cache types are needed.

**Interfaces:**
- Produces: `Gemma4DrafterPrefixCacheRuntime`
- Produces: `Gemma4DrafterGenerationStream::new_with_prefix_cache(...)`
- Keeps: `Gemma4DrafterGenerationStream::new(...)` as a no-cache wrapper for CLI and existing callers.

- [ ] **Step 1: Write failing tests**

Add tests for:

```rust
#[test]
fn drafter_prefix_restore_prefers_lru_then_ssd() {
    // Build a runtime with both stores, seed SSD, then seed LRU with same spec.
    // Assert the restored key path reports LRU and returns the stored last hidden.
}

#[test]
fn drafter_prefix_save_splits_final_prompt_token() {
    // Run the prefill planning helper with prompt_len=6 and chunk_size=0.
    // Assert it saves length 5 before consuming the last prompt token.
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test -p ironmlx models::gemma4::drafter::tests::drafter_prefix_restore_prefers_lru_then_ssd models::gemma4::drafter::tests::drafter_prefix_save_splits_final_prompt_token`

Expected: FAIL because runtime and helper do not exist.

- [ ] **Step 3: Implement runtime**

Create a cloneable runtime containing `Option<PagedPrefixCacheConfig>` and `Option<Arc<Mutex<PrefixLruCache>>>`. Add restore and save helpers that use `prefix_key_spec_for_caches`, `prefix_entry_for_row`, `restore_prefix_entry_for_row`, `PagedPrefixStore`, and the new drafter last-hidden field.

- [ ] **Step 4: Integrate into prefill**

In `new_with_prefix_cache`, enable paged KV on the newly allocated cache, compute the text/VL fingerprint, restore prefix if available, resume VL image-pad cursor from restored prefix, save each reusable prefix, then construct the first generated token exactly as before.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run: `cargo test -p ironmlx models::gemma4::drafter`

Expected: PASS.

### Task 4: Server Wiring

**Files:**
- Modify: `ironmlx/src/core/server/mod.rs`
- Modify: `ironmlx/src/core/server/openai.rs`
- Modify: `ironmlx/src/core/server/anthropic.rs`

**Interfaces:**
- Produces: `Gemma4DrafterAppState::prefix_cache_runtime: Gemma4DrafterPrefixCacheRuntime`
- Consumes: `Gemma4DrafterGenerationStream::new_with_prefix_cache(...)`

- [ ] **Step 1: Write failing tests**

Add tests asserting Gemma4 drafter serve config accepts paged prefix cache and prefix LRU at `b_max=1`, and still rejects `b_max>1` plus active KV offload.

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test -p ironmlx cli::serve::scheduler_profile_tests::serve_paged_prefix_cache_accepts_gemma4_drafter_config`

Expected: FAIL because the server still rejects Gemma4 drafter paged prefix cache.

- [ ] **Step 3: Implement server runtime plumbing**

Remove only the paged-prefix and prefix-LRU guards in `build_gemma4_drafter_app_state`. Build a drafter runtime from cloned prefix config and LRU config. Pass the runtime to OpenAI and Anthropic streaming and unary calls.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `cargo test -p ironmlx cli::serve::scheduler_profile_tests::serve_paged_prefix_cache_accepts_gemma4_drafter_config`

Expected: PASS.

### Task 5: Validation and Commit

**Files:**
- Modify all files from previous tasks.

**Interfaces:**
- Produces: committed branch `feat/gemma4-drafter-prefix-cache`.

- [ ] **Step 1: Run focused prefix and drafter tests**

Run: `cargo test -p ironmlx core::cache::prefix_store::tests:: models::gemma4::drafter::tests::`

Expected: PASS.

- [ ] **Step 2: Run workspace tests**

Run: `cargo test --workspace --all-features`

Expected: PASS.

- [ ] **Step 3: Run Rust required checks**

Run:

```bash
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
```

Expected: all commands succeed. Existing MLX C++ header warnings are acceptable during build output.

- [ ] **Step 4: Run real-model smoke when local models exist**

Run a local Gemma4 E4B drafter server with paged prefix cache enabled and issue the same text request twice. Then run one VL request twice. Expected: first request saves a prefix and second request restores it without returning an error.

- [ ] **Step 5: Commit**

Run:

```bash
git status --short
git add docs/superpowers/specs/2026-07-02-gemma4-drafter-prefix-cache-design.md docs/superpowers/plans/2026-07-02-gemma4-drafter-prefix-cache.md ironmlx/src/core/cache/prefix_store.rs ironmlx/src/core/cache/mod.rs ironmlx/src/models/gemma4/drafter.rs ironmlx/src/core/server/mod.rs ironmlx/src/core/server/openai.rs ironmlx/src/core/server/anthropic.rs
git commit -m "feat(gemma4): support drafter prefix cache"
```

Expected: commit succeeds on `feat/gemma4-drafter-prefix-cache`.
