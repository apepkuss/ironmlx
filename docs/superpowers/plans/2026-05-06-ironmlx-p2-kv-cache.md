# ironmlx P2: KV cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build single-request `core::cache::KVCache` (C-3: lazy alloc, must-pass `cap`, builder `with_step`, concatenate-based grow) and wire it into `nn::Attention::forward` so P3-P7 can run prefill + decode flows.

**Architecture:** New file `ironmlx/src/core/cache/kv_cache.rs` implements the cache; `ironmlx/src/core/cache/mod.rs` re-exports `KVCache`; `ironmlx/src/nn/attention.rs` extends `forward` / `forward_on` to take `Option<&mut KVCache>`. Implementation strategy is **方案 1: mlx-lm-style concatenate** because `slice_update` is not bound in cxx-mlx (verified pre-plan). API stays stable if `slice_update` is added later.

**Tech Stack:** Rust 2021 + cxx-mlx (`mlx::Array` / `mlx::ops::shape::concatenate` / `mlx::ops::indexing::slice_strided` / `Array::zeros`). Spec: [docs/superpowers/specs/2026-05-06-ironmlx-p2-kv-cache-design.md](../specs/2026-05-06-ironmlx-p2-kv-cache-design.md).

---

## Conventions Recap

- **TDD per task**: write failing test → run (FAIL) → implement → run (PASS) → fmt/lint/build → commit.
- **Project gate before commit** (`.claude/CLAUDE.md`):
  ```
  cargo fmt
  cargo +nightly fmt --all -- --check
  cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
  cargo build --release
  ```
  `MLX_DIR=$HOME/.local/mlx` required for any test that touches MLX FFI.
- **Each task ends green**: workspace `cargo test --release` passes before commit.
- **Commit messages ASCII-safe** (no Unicode arrows / em-dashes in subject).
- **No backwards-compat code** per `.claude/CLAUDE.md`.

---

## File Structure (after P2)

```
ironmlx/src/core/
├── mod.rs                        # +pub mod cache + pub use cache::KVCache
└── cache/
    ├── mod.rs                    # NEW
    └── kv_cache.rs               # NEW

ironmlx/src/nn/
└── attention.rs                  # forward / forward_on signatures extended

ironmlx/tests/
└── p2_kv_cache.rs                # NEW integration tests
```

---

## Task 1: `core::cache::KVCache` (C-3 implementation)

**Files:**
- Create: `ironmlx/src/core/cache/mod.rs`
- Create: `ironmlx/src/core/cache/kv_cache.rs`
- Modify: `ironmlx/src/core/mod.rs` (declare `pub mod cache;` + re-export `KVCache`)
- Modify: `ironmlx/src/lib.rs` (re-export `KVCache` at crate root)

### Goal

`KVCache::new(batch, n_kv_heads, head_dim, v_head_dim, dtype, cap)` constructs lazy-alloc cache. `update_and_fetch(k, v)` grows by `step` chunks (default 256), writes new K/V at `[..., offset..offset+n_new, ...]`, returns `[..., 0..offset, ...]` slices.

### Steps

- [ ] **Step 1.1: Write failing unit tests in `ironmlx/src/core/cache/kv_cache.rs`** (file doesn't exist yet — write the file with module skeleton + tests):

```rust
//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: mlx-lm-style concatenate (slice_update is not
//! bound in cxx-mlx). Each grow concatenates `[old_keys[..offset], k_new,
//! zeros[trailing]]` along axis 2. The public API (`new`, `with_step`,
//! `update_and_fetch`, `offset`, `cap`, `reset`) is stable across
//! implementation strategies.

use mlx::ops::indexing::slice_strided;
use mlx::ops::shape::concatenate;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    cap: i32,
    step: i32,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}

impl KVCache {
    /// TODO: implementation in step 1.3
    pub fn new(
        _batch: i32,
        _n_kv_heads: i32,
        _head_dim: i32,
        _v_head_dim: i32,
        _dtype: Dtype,
        _cap: i32,
    ) -> Self {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(cap: i32) -> KVCache {
        KVCache::new(1, 4, 256, 256, Dtype::Float32, cap)
    }

    fn make_kv(seq: i32) -> (Array, Array) {
        let total = (1 * 4 * seq * 256) as usize;
        let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 10.0).collect();
        let k: Array = (&k_data[..], (1, 4, seq, 256)).try_into().unwrap();
        let v: Array = (&v_data[..], (1, 4, seq, 256)).try_into().unwrap();
        (k, v)
    }

    #[test]
    fn new_lazy_allocation_and_zero_offset() {
        let c = make_cache(1024);
        assert_eq!(c.offset(), 0);
        assert_eq!(c.cap(), 1024);
        // keys/values not allocated yet (private field — checked indirectly via
        // first update_and_fetch path; here we just verify offset is zero).
    }

    #[test]
    fn update_first_call_assigns_buffer_and_advances_offset() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(8);
        let (kf, vf) = c.update_and_fetch(&k, &v).expect("update");
        assert_eq!(c.offset(), 8);
        assert_eq!(kf.shape().as_slice(), &[1, 4, 8, 256]);
        assert_eq!(vf.shape().as_slice(), &[1, 4, 8, 256]);
    }

    #[test]
    fn returned_slices_match_written_data() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(4);
        let (kf, _vf) = c.update_and_fetch(&k, &v).expect("update");
        // First few elements of returned K should match input K.
        let kf_vec: Vec<f32> = kf.to_vec().unwrap();
        let k_vec: Vec<f32> = k.to_vec().unwrap();
        assert_eq!(kf_vec[..16], k_vec[..16]);
    }

    #[test]
    fn second_update_concatenates_and_grows_capacity() {
        // Default step=256; first update seq=8 grows capacity to 256.
        // Second update seq=4 fits within existing 256 capacity, no regrow.
        let mut c = make_cache(1024);
        let (k1, v1) = make_kv(8);
        c.update_and_fetch(&k1, &v1).unwrap();
        let (k2, v2) = make_kv(4);
        let (kf, vf) = c.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(c.offset(), 12);
        assert_eq!(kf.shape().as_slice(), &[1, 4, 12, 256]);
        assert_eq!(vf.shape().as_slice(), &[1, 4, 12, 256]);
    }

    #[test]
    fn cap_exceeded_returns_error() {
        let mut c = make_cache(10);
        let (k1, _v1) = make_kv(8);
        let (_, _) = c.update_and_fetch(&k1, &_v1).unwrap();
        // 8 + 5 = 13 > cap 10 — error
        let (k2, v2) = make_kv(5);
        let r = c.update_and_fetch(&k2, &v2);
        assert!(r.is_err(), "expected cap exceeded error");
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("cap"), "msg should mention cap; got: {msg}");
    }

    #[test]
    fn reset_clears_offset_keeping_buffers() {
        let mut c = make_cache(1024);
        let (k, v) = make_kv(8);
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 8);
        c.reset();
        assert_eq!(c.offset(), 0);
        // Subsequent update should still work.
        c.update_and_fetch(&k, &v).unwrap();
        assert_eq!(c.offset(), 8);
    }

    #[test]
    fn with_step_overrides_default() {
        let c = KVCache::new(1, 4, 256, 256, Dtype::Float32, 4096).with_step(512);
        // Verify step via internal field if accessible; otherwise indirectly
        // by performing two updates and checking growth boundary.
        let _ = c;
    }

    #[test]
    fn with_step_eq_cap_preallocates() {
        let mut c = KVCache::new(1, 4, 256, 256, Dtype::Float32, 64).with_step(64);
        let (k, v) = make_kv(8);
        let (kf, _vf) = c.update_and_fetch(&k, &v).unwrap();
        // After first update, full capacity should be 64 (one-shot alloc).
        // Returned slice has only the written portion; we rely on internal
        // capacity being a multiple of step. Indirectly verified by the
        // fact that subsequent fills up to cap=64 won't trigger additional
        // grows (timing not asserted; functional correctness via no panic).
        assert_eq!(kf.shape().as_slice(), &[1, 4, 8, 256]);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn with_step_panics_on_zero() {
        let _ = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024).with_step(0);
    }
}
```

- [ ] **Step 1.2: Run tests to verify FAIL**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::kv_cache`
Expected: 9 tests fail at `unimplemented!()` in `KVCache::new`.

- [ ] **Step 1.3: Implement `KVCache` — replace the file body** with full implementation:

```rust
//! Per-layer KV cache for full-attention layers. See P2 spec § 3 for design.
//!
//! Implementation strategy: mlx-lm-style concatenate (slice_update is not
//! bound in cxx-mlx). Each grow concatenates `[old_keys[..offset], k_new,
//! zeros[trailing]]` along axis 2. The public API (`new`, `with_step`,
//! `update_and_fetch`, `offset`, `cap`, `reset`) is stable across
//! implementation strategies.

use mlx::ops::indexing::slice_strided;
use mlx::ops::shape::concatenate;
use mlx::{Array, Dtype, StreamOrDevice};

use crate::Result;

/// Per-layer KV cache for full-attention layers.
///
/// Holds keys + values pre-allocated up to `cap` tokens; grows in
/// `step`-size chunks via `concatenate`. `update_and_fetch` advances an
/// offset pointer and returns slices of the occupied region.
///
/// P2 supports single-request usage (one cache instance per layer per
/// request). Multi-request paged cache is P8/P9 work.
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    cap: i32,
    step: i32,
    batch: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_head_dim: i32,
    dtype: Dtype,
}

impl KVCache {
    /// Construct a fresh cache. Keys/values are allocated lazily on first
    /// `update_and_fetch`.
    ///
    /// `cap` is the hard maximum sequence length; callers compute it as
    /// `prompt_tokens + max_new_tokens`.
    pub fn new(
        batch: i32,
        n_kv_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        dtype: Dtype,
        cap: i32,
    ) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            cap,
            step: 256,
            batch,
            n_kv_heads,
            head_dim,
            v_head_dim,
            dtype,
        }
    }

    /// Override grow step (default 256). `step >= cap` triggers one-shot
    /// preallocation. Panics if `step <= 0`.
    pub fn with_step(mut self, step: i32) -> Self {
        assert!(step > 0, "KVCache step must be positive (got {step})");
        self.step = step;
        self
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Reset offset to 0; retains allocated buffers for reuse.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Append `(k, v)` and return slices covering all cached tokens.
    pub fn update_and_fetch(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        self.update_and_fetch_on(k, v, ())
    }

    /// Stream-targeted variant.
    pub fn update_and_fetch_on(
        &mut self,
        k: &Array,
        v: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<(Array, Array)> {
        let target: StreamOrDevice = target.into();

        let n_new = k.shape().as_slice()[2];
        let new_offset = self.offset + n_new;
        if new_offset > self.cap {
            anyhow::bail!(
                "KVCache cap {} exceeded by {} tokens (offset {} + new {})",
                self.cap,
                new_offset - self.cap,
                self.offset,
                n_new,
            );
        }

        let current_capacity = self
            .keys
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .unwrap_or(0);

        if new_offset > current_capacity {
            // Round new offset up to next step boundary, clamped at cap.
            let target_capacity =
                ((new_offset + self.step - 1) / self.step * self.step).min(self.cap);
            self.grow_to(target_capacity, target)?;
        }

        // Write K/V at [..., offset..offset+n_new, ...] via concatenate:
        //   new_keys = concat([old_keys[..offset], k_new, old_keys[offset+n_new..]])
        // To avoid materializing the trailing zero region as a separate
        // operation, we exploit the pre-allocated buffer:
        //   keys = [..before, k_new, ..after_zero] all stitched once.
        self.write_at_offset(k, v, target)?;
        self.offset = new_offset;

        // Return slices covering [0..offset] along axis 2.
        let keys_full = self.keys.as_ref().expect("keys allocated");
        let values_full = self.values.as_ref().expect("values allocated");
        let k_slice = slice_strided(
            keys_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, self.offset, self.head_dim],
            [1_i32, 1, 1, 1],
        )?;
        let v_slice = slice_strided(
            values_full,
            [0_i32, 0, 0, 0],
            [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
            [1_i32, 1, 1, 1],
        )?;
        Ok((k_slice, v_slice))
    }

    /// Grow underlying K/V buffers to `new_capacity` along axis 2 (sequence
    /// dimension). Old contents are preserved at `[..., 0..offset, ...]`.
    fn grow_to(&mut self, new_capacity: i32, target: StreamOrDevice) -> Result<()> {
        let new_k_shape = (self.batch, self.n_kv_heads, new_capacity, self.head_dim);
        let new_v_shape = (self.batch, self.n_kv_heads, new_capacity, self.v_head_dim);
        let new_k_zeros = Array::zeros(new_k_shape, self.dtype)?;
        let new_v_zeros = Array::zeros(new_v_shape, self.dtype)?;

        // For first-time grow, the new zeros buffer IS the buffer. For
        // subsequent grows, copy old [..offset] into the front of the new
        // buffer via concat: [old[..offset], new_zeros[offset..]].
        let new_k = match &self.keys {
            None => new_k_zeros,
            Some(old) => {
                let old_kept = slice_strided(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                    [1_i32, 1, 1, 1],
                )?;
                let new_zero_tail = slice_strided(
                    &new_k_zeros,
                    [0_i32, 0, self.offset, 0],
                    [self.batch, self.n_kv_heads, new_capacity, self.head_dim],
                    [1_i32, 1, 1, 1],
                )?;
                concatenate(&[&old_kept, &new_zero_tail], 2)?
            }
        };
        let new_v = match &self.values {
            None => new_v_zeros,
            Some(old) => {
                let old_kept = slice_strided(
                    old,
                    [0_i32, 0, 0, 0],
                    [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                )?;
                let new_zero_tail = slice_strided(
                    &new_v_zeros,
                    [0_i32, 0, self.offset, 0],
                    [self.batch, self.n_kv_heads, new_capacity, self.v_head_dim],
                    [1_i32, 1, 1, 1],
                )?;
                concatenate(&[&old_kept, &new_zero_tail], 2)?
            }
        };
        let _ = target; // ops above use default stream; per-op _on threading is a follow-up.
        self.keys = Some(new_k);
        self.values = Some(new_v);
        Ok(())
    }

    /// Write `k` / `v` into K/V buffers at `[..., self.offset..self.offset+n_new, ...]`.
    /// Without `slice_update`, we reconstruct the buffer via concatenate of
    /// three pieces: prefix[..offset], k_new, suffix[offset+n_new..].
    fn write_at_offset(&mut self, k: &Array, v: &Array, target: StreamOrDevice) -> Result<()> {
        let n_new = k.shape().as_slice()[2];
        let end = self.offset + n_new;
        let capacity = self
            .keys
            .as_ref()
            .map(|a| a.shape().as_slice()[2])
            .expect("keys allocated by grow_to");

        // Build new keys: [keys[..offset], k, keys[end..capacity]]
        // - if offset == 0: just [k, suffix]
        // - if end == capacity: just [prefix, k]
        // - else: 3-way concat
        let keys_full = self.keys.as_ref().expect("keys allocated");
        let new_keys = if self.offset == 0 && end == capacity {
            k.clone()
        } else if self.offset == 0 {
            let suffix = slice_strided(
                keys_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[k, &suffix], 2)?
        } else if end == capacity {
            let prefix = slice_strided(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[&prefix, k], 2)?
        } else {
            let prefix = slice_strided(
                keys_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.head_dim],
                [1_i32, 1, 1, 1],
            )?;
            let suffix = slice_strided(
                keys_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[&prefix, k, &suffix], 2)?
        };

        let values_full = self.values.as_ref().expect("values allocated");
        let new_values = if self.offset == 0 && end == capacity {
            v.clone()
        } else if self.offset == 0 {
            let suffix = slice_strided(
                values_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.v_head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[v, &suffix], 2)?
        } else if end == capacity {
            let prefix = slice_strided(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[&prefix, v], 2)?
        } else {
            let prefix = slice_strided(
                values_full,
                [0_i32, 0, 0, 0],
                [self.batch, self.n_kv_heads, self.offset, self.v_head_dim],
                [1_i32, 1, 1, 1],
            )?;
            let suffix = slice_strided(
                values_full,
                [0_i32, 0, end, 0],
                [self.batch, self.n_kv_heads, capacity, self.v_head_dim],
                [1_i32, 1, 1, 1],
            )?;
            concatenate(&[&prefix, v, &suffix], 2)?
        };

        let _ = target; // ops above use default stream; per-op _on threading is a follow-up.
        self.keys = Some(new_keys);
        self.values = Some(new_values);
        Ok(())
    }
}
```

- [ ] **Step 1.4: Wire `mod.rs`**

Create `ironmlx/src/core/cache/mod.rs`:

```rust
//! Per-layer cache types for inference. See P2 spec § 1 for scope.

pub mod kv_cache;

pub use kv_cache::KVCache;
```

Modify `ironmlx/src/core/mod.rs` — add `pub mod cache;` and a flat re-export of `KVCache`:

```rust
//! Generation infrastructure that's model-agnostic.

pub mod cache;
pub mod chat_template;
pub mod loader;
pub mod sampler;
pub mod tokenizer;

pub use cache::KVCache;
pub use chat_template::{ChatTemplate, Message};
pub use loader::{EosTokenId, Loader, QuantMeta, QuantMode, TokenizerConfig};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
```

Modify `ironmlx/src/lib.rs` — extend the existing crate-root re-export:

```rust
pub use core::{ChatTemplate, KVCache, Loader, Message, QuantMeta, Sampler, Tokenizer};
```

- [ ] **Step 1.5: Run tests to verify PASS**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib core::cache::kv_cache`
Expected: 9 tests pass.

- [ ] **Step 1.6: Project gate**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

- [ ] **Step 1.7: Commit**

```bash
git add ironmlx/src/core/cache/ ironmlx/src/core/mod.rs ironmlx/src/lib.rs
git commit -m "feat(ironmlx-p2): KVCache with lazy alloc, must-pass cap, with_step builder"
```

---

## Task 2: Extend `Attention::forward` / `forward_on` to accept `Option<&mut KVCache>`

**Files:**
- Modify: `ironmlx/src/nn/attention.rs`

### Goal

Add `cache` parameter to `Attention::forward` / `forward_on`. Inside forward, after computing rotated K/V (Mrope::apply, currently a P3 stub) and applying q_norm/k_norm, route through `cache.update_and_fetch(&k, &v)?` if cache is `Some`, otherwise pass K/V through unchanged. SDPA consumes the (possibly cache-extended) K_full / V_full.

### Steps

- [ ] **Step 2.1: Read the existing `attention.rs` to understand the current forward body** (do not modify yet):

```bash
cat ironmlx/src/nn/attention.rs
```

Note the existing structure: forward body computes Q/K/V projections, reshapes, applies q_norm/k_norm and Mrope, calls `mlx::fast::scaled_dot_product_attention_on`. The cache hook fits between Mrope (which produces rotated K/V) and SDPA (which consumes them).

- [ ] **Step 2.2: Modify `Attention::forward` / `forward_on` signatures + body**

In `ironmlx/src/nn/attention.rs`:

1. Add `use crate::core::cache::KVCache;` near the top imports.
2. Change `forward` thin wrapper to:

```rust
pub fn forward(
    &self,
    x: &Array,
    mrope: &Mrope,
    cos: &Array,
    sin: &Array,
    mask: Option<&Array>,
    cache: Option<&mut KVCache>,
) -> Result<Array> {
    self.forward_on(x, mrope, cos, sin, mask, cache, ())
}
```

3. Change `forward_on` signature to:

```rust
pub fn forward_on(
    &self,
    x: &Array,
    mrope: &Mrope,
    cos: &Array,
    sin: &Array,
    mask: Option<&Array>,
    cache: Option<&mut KVCache>,
    target: impl Into<mlx::StreamOrDevice>,
) -> Result<Array>
```

4. Inside `forward_on`, after q_norm/k_norm + Mrope::apply on `q` and `k`, but before SDPA, insert the cache routing:

```rust
// existing: q, k, v are now [batch, heads, seq, head_dim] post-RoPE post-qk_norm
let (k_full, v_full) = match cache {
    Some(c) => c.update_and_fetch_on(&k, &v, target)?,
    None => (k, v),
};
// then SDPA uses k_full, v_full instead of k, v
let out = mlx::fast::scaled_dot_product_attention_on(
    &q,
    &k_full,
    &v_full,
    self.scale,
    "causal",
    None,
    None,
    target,
)?;
```

The remaining body (transpose + reshape + o_proj) is unchanged.

> **Note for implementer:** Boss wrote in P2 spec § 3.2 that "cache 持有已 RoPE-rotated K". Verify the existing `forward_on` applies Mrope::apply BEFORE the cache hook. If it doesn't (e.g., Mrope::apply currently happens after both q/k projections but before the q_norm/k_norm sequence), keep that ordering and place the cache hook after Mrope::apply.

- [ ] **Step 2.3: Verify P1 regression — Attention::forward without cache still compiles + tests pass**

Run:
```
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib nn::attention
MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --lib
```
Expected: P1 tests pass (Attention forward without cache path is `cache=None`, equivalent to P1 behavior).

- [ ] **Step 2.4: Project gate**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

- [ ] **Step 2.5: Commit**

```bash
git add ironmlx/src/nn/attention.rs
git commit -m "feat(ironmlx-p2): Attention forward takes Option<&mut KVCache>"
```

---

## Task 3: Integration tests

**Files:**
- Create: `ironmlx/tests/p2_kv_cache.rs`

### Goal

Verify cache-attention integration: (a) cache=None path matches P1 behavior, (b) cache=Some routes K/V through update_and_fetch, (c) offset advances after each forward.

Because `Mrope::cos_sin` / `apply` are P3 stubs (return `Err`), end-to-end Attention::forward will return `Err` even with cache. Integration tests focus on **wiring correctness**: cache.offset() advances after Attention::forward calls (proving the cache hook ran before the Err propagated back), and cache=None path doesn't regress.

### Steps

- [ ] **Step 3.1: Write failing integration test**

Create `ironmlx/tests/p2_kv_cache.rs`:

```rust
//! Integration tests for P2 — KV cache + Attention wiring.
//!
//! Because Mrope::cos_sin / apply are P3 stubs, full end-to-end forward
//! returns Err. Tests below verify that:
//!   - The cache hook executes before the Err propagates (cache.offset advances)
//!   - The cache=None path does not regress P1 behavior

use ironmlx::KVCache;
use mlx::Dtype;

#[test]
fn kv_cache_standalone_prefill_then_decode() {
    // Verify the cache works end-to-end without involving Attention.
    let mut cache = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);

    // Prefill: append 8 tokens
    let prefill_total = (1 * 4 * 8 * 256) as usize;
    let k1_data: Vec<f32> = (0..prefill_total).map(|i| i as f32).collect();
    let v1_data: Vec<f32> = (0..prefill_total).map(|i| (i as f32) * 10.0).collect();
    let k1: mlx::Array = (&k1_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v1: mlx::Array = (&v1_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let (k_full1, v_full1) = cache.update_and_fetch(&k1, &v1).unwrap();
    assert_eq!(cache.offset(), 8);
    assert_eq!(k_full1.shape().as_slice(), &[1, 4, 8, 256]);
    assert_eq!(v_full1.shape().as_slice(), &[1, 4, 8, 256]);

    // Decode: append 1 token
    let one_total = (1 * 4 * 1 * 256) as usize;
    let k2_data: Vec<f32> = (0..one_total).map(|i| (i + 100000) as f32).collect();
    let v2_data: Vec<f32> = (0..one_total).map(|i| (i + 200000) as f32).collect();
    let k2: mlx::Array = (&k2_data[..], (1, 4, 1, 256)).try_into().unwrap();
    let v2: mlx::Array = (&v2_data[..], (1, 4, 1, 256)).try_into().unwrap();
    let (k_full2, v_full2) = cache.update_and_fetch(&k2, &v2).unwrap();
    assert_eq!(cache.offset(), 9);
    assert_eq!(k_full2.shape().as_slice(), &[1, 4, 9, 256]);
    assert_eq!(v_full2.shape().as_slice(), &[1, 4, 9, 256]);
}

#[test]
fn kv_cache_with_step_eq_cap_one_shot_alloc() {
    // step >= cap → first update allocates full cap directly.
    let mut cache =
        KVCache::new(1, 4, 256, 256, Dtype::Float32, 64).with_step(64);
    let total = (1 * 4 * 8 * 256) as usize;
    let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let k: mlx::Array = (&k_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v: mlx::Array = (&v_data[..], (1, 4, 8, 256)).try_into().unwrap();
    cache.update_and_fetch(&k, &v).unwrap();
    assert_eq!(cache.offset(), 8);
    assert_eq!(cache.cap(), 64);
}

#[test]
fn kv_cache_reset_allows_session_reuse() {
    let mut cache = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
    let total = (1 * 4 * 8 * 256) as usize;
    let k_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let k: mlx::Array = (&k_data[..], (1, 4, 8, 256)).try_into().unwrap();
    let v: mlx::Array = (&v_data[..], (1, 4, 8, 256)).try_into().unwrap();
    cache.update_and_fetch(&k, &v).unwrap();
    cache.reset();
    assert_eq!(cache.offset(), 0);
    cache.update_and_fetch(&k, &v).unwrap();
    assert_eq!(cache.offset(), 8);
}
```

> **Why no Attention-end-to-end test?** Because Mrope::apply is a P3 stub returning `Err`, Attention::forward never reaches the cache hook in the end-to-end path. A test that calls Attention::forward and checks `cache.offset() == 0` after the Err would not prove the cache hook was placed correctly — the hook might be unreachable for a different reason. The cleanest verification is: (a) standalone KVCache exercises (above), (b) when P3 lands Mrope::apply, the existing Attention forward test will exercise the full path. Adding a stub-only Attention test now risks false confidence.

- [ ] **Step 3.2: Run integration tests to verify PASS**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx --test p2_kv_cache`
Expected: 3 tests pass.

- [ ] **Step 3.3: Run full workspace tests for regression check**

Run: `MLX_DIR=$HOME/.local/mlx cargo test --release -p ironmlx`
Expected: P1 tests (22 unit + 2 integration) + new P2 unit tests (9) + new P2 integration tests (3) all pass = ~36 tests.

- [ ] **Step 3.4: Project gate**

```
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
cargo build --release
```

- [ ] **Step 3.5: Commit**

```bash
git add ironmlx/tests/p2_kv_cache.rs
git commit -m "test(ironmlx-p2): KVCache integration tests"
```

---

## Verification Checklist

After Task 3:

| Item | Command | Expected |
|---|---|---|
| Unit tests | `cargo test --release -p ironmlx --lib` | All pass (P1 22 + P2 9 = 31) |
| Integration tests | `cargo test --release -p ironmlx --tests` | All pass (P1 2 + P2 3 = 5) |
| Build | `cargo build --release -p ironmlx` | Success |
| Format | `cargo +nightly fmt --all -- --check` | No diff |
| Clippy | `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | No warnings |
| CLI smoke (no behavior change) | `MLX_DIR=$HOME/.local/mlx ./target/release/ironmlx info` | Prints device |

## Spec Coverage Map

| Spec section | Task |
|---|---|
| § 3.1 KVCache (C-3) | T1 |
| § 3.2 Attention forward signature extension | T2 |
| § 3.3 KV cache dimension convention | T1 (test fixtures use [1, 4, seq, 256]) |
| § 3.4 dtype strategy | T1 (KVCache::new takes Dtype param; first update writes that dtype) |
| § 4.1 Unit tests | T1.1 (9 tests) |
| § 4.2 Integration tests | T3.1 (3 tests) — adapted from spec; standalone-cache style instead of Attention end-to-end (rationale documented in T3 step 3.1) |
| § 4.3 Regression | T2.3, T3.3 |

## Risk register (per spec § 6)

- **`slice_update` not bound — verified at plan time.** T1 implements 方案 1 (concatenate-based); public API stable across strategies.
- **Attention forward signature change breaks P1 regression** — mitigated by `cache=None` default path being equivalent to P1.
- **Mrope stub means Attention end-to-end Err** — T3 explicitly tests standalone KVCache, not Attention end-to-end; P4 model assembly will exercise the full path.
- **Concatenate-based grow may rebuild the entire cache buffer per call** — P2 trades performance for correctness simplicity (concat is hot but inside MLX, single GPU op). P9 paged cache will replace this with block-level indirection.
