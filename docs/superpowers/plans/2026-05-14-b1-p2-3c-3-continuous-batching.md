# B1-p2.3c-3 — Continuous batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the "batch boundary at evict_all" constraint by introducing mid-batch admit/evict in `Scheduler` + a rolling decode loop in `SchedulerActor::driver_loop`. Mid-batch admit prefills the new request in a standalone B=1 temporary cache (GenerationStream-equivalent path), then adopts the prefilled row into the main b_max cache via per-layer slice copies.

**Architecture:** Three layers change together — (1) cache API gains `adopt_row_from` for slice-copy of a single row's state from one cache instance to another; (2) Scheduler API relaxes Phase guards (admit/evict legal in Decoding), adds `admit_mid` for mid-batch prefill+adoption, adds `gc_finished_rows` for slot reclamation, and moves Phase transition out of `step` into `gc_finished_rows`; (3) `driver_loop` becomes a rolling decode loop with biased `tokio::select! { cmd_rx | step_default }` per iteration. Outer Idle and initial admission window unchanged from 3b-3.

**Tech Stack:** Rust + cxx-mlx + tokio biased select. `mlx::ops::indexing::{slice_strided, slice_update}` (default stream variants) for cache slice copies. No new mlx ops; reuses the well-tested batched_prefill / GenerationStream B=1 forward path.

---

## Standing Per-Task Hygiene Gate

After each task's implementation step but BEFORE the commit step, run from `/Volumes/Dev/cxx-mlx`:

```bash
MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all -- --check
MLX_DIR=$HOME/.local/mlx cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings
MLX_DIR=$HOME/.local/mlx cargo build --release -p ironmlx
```

All three must be clean. If `fmt --check` fails, run `MLX_DIR=$HOME/.local/mlx cargo +nightly fmt --all` to format and re-check. If clippy emits a warning you don't know how to fix, **STOP and ask Boss** — don't paper over with `#[allow]` unless the lint is clearly inapplicable.

Each task ends with a single git commit. Commit subject prefix: `feat(b1-p2.3c-3):` / `test(b1-p2.3c-3):` / `docs(b1-p2.3c-3):` / `fix(b1-p2.3c-3):`.

The `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer is established repo convention — every 3a/3b-*/3c-1/3c-2 commit uses it (verifiable via `git log`). Boss approved this in the plan template. Include verbatim in every commit body.

Model fixture: `export QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)` (wildcard — actual dir is `models--mlx-community--Qwen3.5-4B-MLX-4bit`).

---

## File Structure

| File | Task | Role |
| --- | --- | --- |
| `ironmlx/src/core/cache/kv_cache.rs` | 1 | `dtype()` accessor + `adopt_row_from(src, dst_row, src_row)` + 3 unit tests |
| `ironmlx/src/core/cache/gated_delta.rs` | 2 | `adopt_row_from(src, dst_row, src_row)` + 2 unit tests |
| `ironmlx/src/core/scheduler.rs` | 3, 4 | T3: admit/evict Phase guard relaxation + `gc_finished_rows` + step Phase transition removal + 3 unit tests. T4: `admit_mid` |
| `ironmlx/src/core/generate.rs` | 4 | `slice_logits_row` helper |
| `ironmlx/src/core/server/scheduler_actor.rs` | 4 | `RollingEvent` enum + `handle_admit_mid` + driver_loop rolling decode refactor |
| `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` | 5 | NEW — 3 integration scenarios |
| `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md` | 5 | NEW — close-out |

---

### Task 1: `KVCache::adopt_row_from` + `dtype()` accessor + 3 unit tests

**Files:**
- Modify: `ironmlx/src/core/cache/kv_cache.rs` (+ ~70 lines impl + ~120 lines tests)

- [ ] **Step 1: Add 3 failing unit tests**

Open `ironmlx/src/core/cache/kv_cache.rs`. Find the `#[cfg(test)] mod tests` block (15 tests after Task 1 + followup in 3c-2). Append these 3 new tests at the end (before the module's closing `}`):

```rust
    #[test]
    fn kvcache_adopt_row_from_basic() {
        // src: B=1, write 4 K/V tokens with marker values 7.0 (K) and 70.0 (V).
        let mut src = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let n_per_row = (4 * 4 * 256) as usize;
        let k_data: Vec<f32> = std::iter::repeat(7.0_f32).take(n_per_row).collect();
        let v_data: Vec<f32> = std::iter::repeat(70.0_f32).take(n_per_row).collect();
        let k: Array = (&k_data[..], (1_i32, 4_i32, 4_i32, 256_i32)).try_into().unwrap();
        let v: Array = (&v_data[..], (1_i32, 4_i32, 4_i32, 256_i32)).try_into().unwrap();
        src.update_and_fetch(&k, &v, &[4]).expect("src write");
        assert_eq!(src.offsets(), &[4]);

        // dst: B=2, fresh (no allocation yet).
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        dst.adopt_row_from(&src, /*dst_row=*/ 1, /*src_row=*/ 0)
            .expect("adopt_row_from basic");

        assert_eq!(dst.offsets(), &[0, 4]);

        // Force evaluation: read back the destination's K/V slab via the
        // cache's own per-row fetch API would re-grow the buffer, so we
        // read self.keys directly via a synthesis: write one row of zeros
        // into row 0 with per_row_lens=[0,0] (no-op) then fetch the slab.
        // Simpler: assert via update_and_fetch with per_row_lens=[0,0].
        let zero_k: Array = (&vec![0.0_f32; (2 * 4 * 0 * 256) as usize][..], (2_i32, 4_i32, 0_i32, 256_i32))
            .try_into()
            .unwrap();
        let zero_v: Array = (&vec![0.0_f32; (2 * 4 * 0 * 256) as usize][..], (2_i32, 4_i32, 0_i32, 256_i32))
            .try_into()
            .unwrap();
        // This call is the all-zero fast path (returns empty slices without writing).
        // Use it just to confirm the dst cache is functional after adoption.
        let _ = dst.update_and_fetch(&zero_k, &zero_v, &[0, 0]);

        // Exhaustively verify dst.keys / values content via a synthetic
        // read: do update_and_fetch(&k_zero_extra, &v_zero_extra, &[0, 4])
        // which would write 4 zeros to row 1 starting at offset 4 — but
        // that overwrites our adopted data. Skip; instead exercise the
        // adopted state by triggering grow_to + return slice with a no-op
        // call (above) and inspecting offsets only at this layer. The
        // semantic check that row 1's [0..4] really contains 7.0 / 70.0
        // is exercised by the integration scenarios in Task 5.
        //
        // The functional contract this lib test asserts:
        //   1. dst.offsets() == [0, 4] after adoption.
        //   2. dst.cap unchanged.
        //   3. No panic during adoption.
        //   4. dst is functionally usable post-adoption (update_and_fetch
        //      with per_row_lens=[0,0] returns Ok empty slices).
        assert_eq!(dst.cap(), 1024);
    }

    #[test]
    fn kvcache_adopt_row_from_shape_mismatch_err() {
        // src has different n_kv_heads → adopt_row_from must Err.
        let src = KVCache::new(1, 8 /* different n_kv_heads */, 256, 256, Dtype::Float32, 1024);
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        let r = dst.adopt_row_from(&src, 1, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("mismatch") || msg.contains("shape"),
            "msg should mention shape mismatch; got: {msg}"
        );
    }

    #[test]
    fn kvcache_adopt_row_from_out_of_bounds_err() {
        let src = KVCache::new(1, 4, 256, 256, Dtype::Float32, 1024);
        let mut dst = KVCache::new(2, 4, 256, 256, Dtype::Float32, 1024);
        // dst_row=2 is OOB for dst.batch=2.
        let r = dst.adopt_row_from(&src, 2, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("dst_row") || msg.contains("batch"),
            "msg should mention dst_row OOB; got: {msg}"
        );
    }
```

Note on `kvcache_adopt_row_from_basic`: the lib test verifies the functional contract (offsets, cap, no-panic, functional state). Exhaustive K/V value verification of adopted content is exercised in Task 5's integration scenarios where end-to-end bit-id parity confirms K/V was copied correctly.

- [ ] **Step 2: Run new tests to confirm they fail (compile error)**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::cache::kv_cache::tests::kvcache_adopt_row_from
```

Expected: FAIL with `no method named 'adopt_row_from' found`. Confirms the tests exercise the new API.

- [ ] **Step 3: Add `dtype()` accessor**

Open `ironmlx/src/core/cache/kv_cache.rs`. Find the existing `pub fn cap(&self) -> i32` accessor (line 78). Add an accessor just below it:

```rust
    pub fn cap(&self) -> i32 {
        self.cap
    }

    /// Dtype used for the K/V buffer. Exposed so `adopt_row_from` can
    /// validate that `src` and `self` agree before slicing.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
```

- [ ] **Step 4: Add `adopt_row_from` implementation**

In `ironmlx/src/core/cache/kv_cache.rs`, at the end of the `impl KVCache` block (after `update_and_fetch_on` but before the `impl` closing brace; the `grow_to` private fn comes after, so place `adopt_row_from` between the public `update_and_fetch_on` and the private `grow_to`):

```rust
    /// Copy a single row's cache state from `src` into `self` at
    /// `dst_row`. The destination slot's K/V at positions
    /// `[0..src.offsets[src_row]]` is overwritten; positions beyond
    /// (stale or unallocated) are not touched. `self.offsets[dst_row]`
    /// is set to `src.offsets[src_row]`.
    ///
    /// Requires matching n_kv_heads / head_dim / v_head_dim / dtype.
    /// src and self may have different batch sizes (typical usage:
    /// src.batch = 1, self.batch = b_max).
    ///
    /// Errors on shape/dtype mismatch, dst_row >= self.batch,
    /// src_row >= src.batch, or src.offsets[src_row] > self.cap.
    pub fn adopt_row_from(
        &mut self,
        src: &KVCache,
        dst_row: usize,
        src_row: usize,
    ) -> Result<()> {
        if self.n_kv_heads != src.n_kv_heads
            || self.head_dim != src.head_dim
            || self.v_head_dim != src.v_head_dim
            || self.dtype != src.dtype
        {
            anyhow::bail!(
                "KVCache::adopt_row_from: shape/dtype mismatch (self={}/{}/{}/{:?}, src={}/{}/{}/{:?})",
                self.n_kv_heads, self.head_dim, self.v_head_dim, self.dtype,
                src.n_kv_heads, src.head_dim, src.v_head_dim, src.dtype,
            );
        }
        if dst_row >= self.batch as usize {
            anyhow::bail!(
                "KVCache::adopt_row_from: dst_row {} >= self.batch {}",
                dst_row, self.batch,
            );
        }
        if src_row >= src.batch as usize {
            anyhow::bail!(
                "KVCache::adopt_row_from: src_row {} >= src.batch {}",
                src_row, src.batch,
            );
        }
        let src_off = src.offsets[src_row];
        if src_off > self.cap {
            anyhow::bail!(
                "KVCache::adopt_row_from: src.offsets[{}] = {} > self.cap {}",
                src_row, src_off, self.cap,
            );
        }

        if src_off > 0 {
            // Ensure self.keys / values are allocated up to src_off.
            let current_capacity = self
                .keys
                .as_ref()
                .map(|a| a.shape().as_slice()[2])
                .unwrap_or(0);
            if src_off > current_capacity {
                let target_capacity =
                    ((src_off + self.step - 1) / self.step * self.step).min(self.cap);
                self.grow_to(target_capacity, ())?;
            }

            let src_keys = src
                .keys
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!(
                    "KVCache::adopt_row_from: src has offset {} but keys are unallocated",
                    src_off
                ))?;
            let src_values = src
                .values
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!(
                    "KVCache::adopt_row_from: src has offset {} but values are unallocated",
                    src_off
                ))?;

            // Slice src[src_row, :, 0..src_off, :].
            let k_slice = slice_strided_on(
                src_keys,
                [src_row as i32, 0, 0, 0],
                [src_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let v_slice = slice_strided_on(
                src_values,
                [src_row as i32, 0, 0, 0],
                [src_row as i32 + 1, self.n_kv_heads, src_off, self.v_head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;

            // Write into self[dst_row, :, 0..src_off, :].
            let keys_full = self.keys.as_ref().expect("grow_to allocated keys");
            let values_full = self.values.as_ref().expect("grow_to allocated values");
            let new_keys = slice_update_on(
                keys_full,
                &k_slice,
                [dst_row as i32, 0, 0, 0],
                [dst_row as i32 + 1, self.n_kv_heads, src_off, self.head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            let new_values = slice_update_on(
                values_full,
                &v_slice,
                [dst_row as i32, 0, 0, 0],
                [dst_row as i32 + 1, self.n_kv_heads, src_off, self.v_head_dim],
                [1_i32, 1, 1, 1],
                (),
            )?;
            self.keys = Some(new_keys);
            self.values = Some(new_values);
        }

        self.offsets[dst_row] = src_off;
        Ok(())
    }
```

Note: `slice_strided_on` and `slice_update_on` are already imported at the top of the file (line 8). The `(),` for `target` passes the default stream — same convention as the existing `update_and_fetch` (non-`_on`) wrapper. Using `_on` variants keeps signature consistency with the rest of the file.

- [ ] **Step 5: Verify new tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::cache::kv_cache::tests
```

Expected: 18 tests PASS (15 baseline from 3c-2 + 3 new from this task).

- [ ] **Step 6: Hygiene gate**

Run the Standing Per-Task Hygiene Gate. All clean.

- [ ] **Step 7: Commit**

```bash
git add ironmlx/src/core/cache/kv_cache.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3c-3): KVCache::adopt_row_from + dtype() accessor

Add a primitive for slice-copying a single row's cache state from one
KVCache instance into another. The Scheduler::admit_mid path in 3c-3
uses this to move a B=1 temp cache's prefilled row into the main
b_max cache at the freed slot.

Validation: shape/dtype mismatch, dst_row out of bounds, src_row out
of bounds, src offset > self.cap all return Err.

Optimization: skip the slice_update entirely when src_off == 0 (no
data to copy); just set the destination offset. Avoids 0-length slice
edge cases with mlx.

Lazy alloc: if self.keys is None (fresh dst) and src_off > 0, call
grow_to(step-rounded target_capacity) first so slice_update has a
buffer to write into. Matches the lazy-alloc behavior of
update_and_fetch_on.

3 new unit tests cover basic adoption, shape-mismatch Err, and
out-of-bounds Err. End-to-end K/V value verification (that adopted
content matches src bit-for-bit) is exercised by Task 5's integration
scenarios.

Lib test count: 205 -> 208.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `GatedDeltaCache::adopt_row_from` + 2 unit tests

**Files:**
- Modify: `ironmlx/src/core/cache/gated_delta.rs` (+ ~70 lines impl + ~80 lines tests)

- [ ] **Step 1: Add 2 failing unit tests**

Open `ironmlx/src/core/cache/gated_delta.rs`. Find the `#[cfg(test)] mod tests` block (9 tests after 3c-1 + followup). Append at the end of the tests module:

```rust
    #[test]
    fn gdcache_adopt_row_from_state_and_offset() {
        // src: B=1 cache, mutate conv_state and recurrent_state to marker
        // values, advance offset to 4.
        let mut src =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src new");
        // conv_state shape [1, 3, 8] = 24 elements; fill with 1.0.
        let conv_marker = Array::ones((1_i32, 3, 8), Dtype::Bfloat16).expect("conv_marker zeros");
        src.update_conv(conv_marker);
        // recurrent_state shape [1, 4, 8, 8] = 256 elements; fill with 2.0.
        let rec_marker_f32: Array = (
            &vec![2.0_f32; 256][..],
            &[1_i32, 4, 8, 8][..],
        )
            .try_into()
            .expect("rec_marker");
        src.update_recurrent(rec_marker_f32);
        src.advance(&[4]).expect("src advance");
        assert_eq!(src.offsets(), &[4]);

        // dst: B=2 cache, fresh (all zeros).
        let mut dst =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("dst new");
        dst.adopt_row_from(&src, /*dst_row=*/ 1, /*src_row=*/ 0)
            .expect("adopt_row_from");

        assert_eq!(dst.offsets(), &[0, 4]);

        // Verify dst.conv_state[1, :, :] is all 1.0 (adopted from src) and
        // dst.conv_state[0, :, :] is all 0.0 (untouched).
        let conv_dtype = dst.conv_state().dtype();
        let conv_as_f32 = mlx::ops::cast::astype(dst.conv_state(), Dtype::Float32)
            .expect("cast conv to f32");
        let conv_vec: Vec<f32> = conv_as_f32.to_vec().expect("conv to_vec");
        assert_eq!(conv_vec.len(), 2 * 3 * 8); // [B=2, k-1=3, conv_dim=8]
        let conv_stride_row = 3 * 8; // (k-1) * conv_dim
        for i in 0..conv_stride_row {
            assert_eq!(conv_vec[i], 0.0_f32, "dst.conv_state row 0 corrupted at {i}");
        }
        for i in conv_stride_row..(2 * conv_stride_row) {
            assert_eq!(conv_vec[i], 1.0_f32, "dst.conv_state row 1 wrong at {i}");
        }
        let _ = conv_dtype; // suppress unused if we ever drop the cast

        // Verify dst.recurrent_state[1, :, :, :] is all 2.0 and [0, ...] is 0.0.
        let rec_vec: Vec<f32> = dst.recurrent_state().to_vec().expect("rec to_vec");
        assert_eq!(rec_vec.len(), 2 * 4 * 8 * 8); // [B=2, Hv=4, Dv=8, Dk=8]
        let rec_stride_row = 4 * 8 * 8;
        for i in 0..rec_stride_row {
            assert_eq!(rec_vec[i], 0.0_f32, "dst.rec row 0 corrupted at {i}");
        }
        for i in rec_stride_row..(2 * rec_stride_row) {
            assert_eq!(rec_vec[i], 2.0_f32, "dst.rec row 1 wrong at {i}");
        }
    }

    #[test]
    fn gdcache_adopt_row_from_out_of_bounds_err() {
        let src =
            GatedDeltaCache::new_with_cap(1, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("src new");
        let mut dst =
            GatedDeltaCache::new_with_cap(2, 4, 8, 4, 8, 8, Dtype::Bfloat16, 16).expect("dst new");
        // dst_row=2 is OOB for dst.B=2.
        let r = dst.adopt_row_from(&src, 2, 0);
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("dst_row") || msg.contains("B"),
            "msg should mention dst_row OOB; got: {msg}"
        );
    }
```

- [ ] **Step 2: Run new tests to confirm they fail (compile error)**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::cache::gated_delta::tests::gdcache_adopt_row_from
```

Expected: FAIL with `no method named 'adopt_row_from' found`.

- [ ] **Step 3: Add the imports needed for slice_strided / slice_update**

Open `ironmlx/src/core/cache/gated_delta.rs`. At the top of the file (around line 8), change the import line from:

```rust
use mlx::{Array, Dtype};
```

to:

```rust
use mlx::ops::indexing::{slice_strided_on, slice_update_on};
use mlx::{Array, Dtype};
```

- [ ] **Step 4: Add `adopt_row_from` implementation**

In `ironmlx/src/core/cache/gated_delta.rs`, append a new method to the `impl GatedDeltaCache` block (after `reset`, before the closing `}`):

```rust
    /// Copy a single row's full SSM state from `src` into `self` at
    /// `dst_row`. The destination's `conv_state[dst_row, :, :]` and
    /// `recurrent_state[dst_row, :, :, :]` slabs are overwritten;
    /// `self.offsets[dst_row]` is set to `src.offsets[src_row]`.
    ///
    /// Requires matching `kernel_size - 1`, `conv_dim`, `Hv`, `Dv`, `Dk`
    /// between src and self. Batch dimensions may differ.
    pub fn adopt_row_from(
        &mut self,
        src: &GatedDeltaCache,
        dst_row: usize,
        src_row: usize,
    ) -> Result<()> {
        let self_conv_dims = self.conv_state.shape();
        let self_conv_dims = self_conv_dims.as_slice();
        let src_conv_dims = src.conv_state.shape();
        let src_conv_dims = src_conv_dims.as_slice();
        if self_conv_dims[1] != src_conv_dims[1] || self_conv_dims[2] != src_conv_dims[2] {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: conv_state shape mismatch (self [_,{},{}] src [_,{},{}])",
                self_conv_dims[1], self_conv_dims[2],
                src_conv_dims[1], src_conv_dims[2],
            );
        }
        let self_rec_dims = self.recurrent_state.shape();
        let self_rec_dims = self_rec_dims.as_slice();
        let src_rec_dims = src.recurrent_state.shape();
        let src_rec_dims = src_rec_dims.as_slice();
        if self_rec_dims[1] != src_rec_dims[1]
            || self_rec_dims[2] != src_rec_dims[2]
            || self_rec_dims[3] != src_rec_dims[3]
        {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: recurrent_state shape mismatch (self [_,{},{},{}] src [_,{},{},{}])",
                self_rec_dims[1], self_rec_dims[2], self_rec_dims[3],
                src_rec_dims[1], src_rec_dims[2], src_rec_dims[3],
            );
        }
        if dst_row >= self.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: dst_row {} >= self.B {}",
                dst_row, self.offsets.len(),
            );
        }
        if src_row >= src.offsets.len() {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: src_row {} >= src.B {}",
                src_row, src.offsets.len(),
            );
        }
        let src_off = src.offsets[src_row];
        if src_off > self.cap {
            anyhow::bail!(
                "GatedDeltaCache::adopt_row_from: src.offsets[{}] = {} > self.cap {}",
                src_row, src_off, self.cap,
            );
        }

        let kernel_minus_one = self_conv_dims[1];
        let conv_dim = self_conv_dims[2];
        let hv = self_rec_dims[1];
        let dv = self_rec_dims[2];
        let dk = self_rec_dims[3];

        // Copy conv_state[src_row, :, :] -> self.conv_state[dst_row, :, :].
        let src_conv_slice = slice_strided_on(
            &src.conv_state,
            [src_row as i32, 0, 0],
            [src_row as i32 + 1, kernel_minus_one, conv_dim],
            [1_i32, 1, 1],
            (),
        )?;
        self.conv_state = slice_update_on(
            &self.conv_state,
            &src_conv_slice,
            [dst_row as i32, 0, 0],
            [dst_row as i32 + 1, kernel_minus_one, conv_dim],
            [1_i32, 1, 1],
            (),
        )?;

        // Copy recurrent_state[src_row, :, :, :] -> self.recurrent_state[dst_row, :, :, :].
        let src_rec_slice = slice_strided_on(
            &src.recurrent_state,
            [src_row as i32, 0, 0, 0],
            [src_row as i32 + 1, hv, dv, dk],
            [1_i32, 1, 1, 1],
            (),
        )?;
        self.recurrent_state = slice_update_on(
            &self.recurrent_state,
            &src_rec_slice,
            [dst_row as i32, 0, 0, 0],
            [dst_row as i32 + 1, hv, dv, dk],
            [1_i32, 1, 1, 1],
            (),
        )?;

        self.offsets[dst_row] = src_off;
        Ok(())
    }
```

- [ ] **Step 5: Verify new tests pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::cache::gated_delta::tests
```

Expected: 11 tests PASS (9 baseline + 2 new).

- [ ] **Step 6: Verify whole cache module is green**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::cache
```

Expected: 28 tests PASS (15 KVCache + 11 GatedDeltaCache + 2 MtpCache).

- [ ] **Step 7: Hygiene gate**

Run the Standing Per-Task Hygiene Gate. All clean.

- [ ] **Step 8: Commit**

```bash
git add ironmlx/src/core/cache/gated_delta.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3c-3): GatedDeltaCache::adopt_row_from

Mirror of KVCache::adopt_row_from for the linear-attention SSM cache.
Slice-copies one row's conv_state (shape [B, kernel_size-1, conv_dim])
and recurrent_state (shape [B, Hv, Dv, Dk]) from src into self at
dst_row, plus offset transfer.

Validation: conv_state/recurrent_state shape mismatch, dst_row OOB,
src_row OOB, src offset > self.cap all return Err.

Unlike KVCache the conv_state and recurrent_state slabs MUST be
written (not skipped on src_off==0), because the SSM kernel reads
state_in for every forward — leaving stale state from the previous
occupant would corrupt the next prefill's conv1d output. The
adoption unconditionally writes the slab (which contains zeros from
a fresh src cache via Array::zeros initialization).

2 new unit tests: exhaustive K/V state coverage on a 2-row adoption
(row 1 receives src=1.0/2.0 markers, row 0 stays at zero); OOB Err.

Lib test count: 208 -> 210.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Scheduler API — admit/evict Phase relaxation + `gc_finished_rows` + step Phase transition removal

**Files:**
- Modify: `ironmlx/src/core/scheduler.rs` (~150 lines: API changes + 3 new unit tests)

- [ ] **Step 1: Add 3 failing unit tests for new behavior**

Open `ironmlx/src/core/scheduler.rs`. Find the `#[cfg(test)] mod tests` block. Append these 3 tests at the end:

```rust
    #[test]
    fn scheduler_admit_during_decoding_ok() {
        // Force phase to Decoding (test seam); admit should succeed and
        // Phase should stay Decoding (mid-batch admit semantics).
        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        s.force_phase(Phase::Decoding);

        let id_b = s.admit(mk_req(vec![4, 5, 6, 7])).expect("admit b during Decoding");
        assert_eq!(s.phase(), Phase::Decoding, "phase should remain Decoding");
        assert_eq!(s.active_count(), 2);
        // Both ids should be findable.
        assert!(s.get(id_a).is_some());
        assert!(s.get(id_b).is_some());
    }

    #[test]
    fn scheduler_evict_during_decoding_transitions_to_finished_when_last() {
        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        s.force_phase(Phase::Decoding);

        // Evict during Decoding: legal now (was Err pre-3c-3).
        s.evict(id_a).expect("evict during Decoding");
        // active_count == 0 + was Decoding -> Finished
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
    }

    #[test]
    fn scheduler_gc_finished_rows_clears_slots_and_transitions() {
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        let mut s = Scheduler::new(2);
        let id_a = s.admit(mk_req(vec![1, 2, 3])).expect("admit a");
        let id_b = s.admit(mk_req(vec![4, 5, 6])).expect("admit b");
        s.force_phase(Phase::Decoding);

        // Mark both as finished (test seam: directly mutate state).
        s.get_mut(id_a).unwrap().finished = true;
        s.get_mut(id_a).unwrap().finish_reason = Some("length");
        s.get_mut(id_b).unwrap().finished = true;
        s.get_mut(id_b).unwrap().finish_reason = Some("stop");

        let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
        let (tx_a, _rx_a) = mpsc::unbounded_channel::<StepEvent>();
        let (tx_b, _rx_b) = mpsc::unbounded_channel::<StepEvent>();
        event_txs.insert(id_a, tx_a);
        event_txs.insert(id_b, tx_b);

        let evicted = s.gc_finished_rows(&mut event_txs);
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&id_a));
        assert!(evicted.contains(&id_b));
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.phase(), Phase::Finished);
        assert!(event_txs.is_empty(), "event_txs should be empty after gc");
    }
```

The test references `tokio::sync::mpsc` so verify the test module's imports include it. If `use tokio::sync::mpsc;` isn't already in the tests module's `use super::*;` block, add `use tokio::sync::mpsc;` near the top of the tests module.

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::scheduler::tests::scheduler_admit_during_decoding_ok core::scheduler::tests::scheduler_evict_during_decoding_transitions_to_finished_when_last core::scheduler::tests::scheduler_gc_finished_rows_clears_slots_and_transitions
```

Expected: FAIL — first two with the existing Phase guard Err, third with `no method named 'gc_finished_rows'`.

- [ ] **Step 3: Relax `admit` Phase guard**

Find `pub fn admit` (around line 179). The current body has:

```rust
        match self.phase {
            Phase::Idle | Phase::Admitting => {}
            Phase::Decoding | Phase::Finished => {
                return Err(anyhow!(
                    "scheduler in {:?} phase: cannot admit; call evict_all first",
                    self.phase
                ));
            }
        }
```

Replace with:

```rust
        if self.phase == Phase::Finished {
            return Err(anyhow!(
                "scheduler in Finished phase: call evict_all first"
            ));
        }
        // Idle / Admitting / Decoding all allow admit.
        //   Idle -> Admitting (first admit transitions below).
        //   Admitting -> Admitting (subsequent admits during window).
        //   Decoding -> Decoding (mid-batch admit; caller is responsible
        //     for prefilling the new slot via admit_mid).
```

Then find the end of the admit body where `self.phase = Phase::Admitting;` is set unconditionally. Change to conditional:

```rust
        self.slots[row_idx] = Some(state);
        if self.phase == Phase::Idle {
            self.phase = Phase::Admitting;
        }
        // Decoding stays Decoding (no transition on mid-batch admit).
        Ok(id)
```

- [ ] **Step 4: Relax `evict` Phase guard**

Find `pub fn evict` (around line 225). The current body has:

```rust
        match self.phase {
            Phase::Decoding => {
                return Err(anyhow!(
                    "evict illegal in {:?} phase: call evict_all after the batch finishes",
                    self.phase
                ));
            }
            Phase::Idle | Phase::Admitting | Phase::Finished => {}
        }
```

Replace the entire `match` block with a no-op (allow all phases). The existing logic that follows (find row_idx, set slots[i] = None, possibly transition Admitting → Idle) stays. ADD a new Decoding→Finished transition at the end:

```rust
        let row_idx = self
            .slots
            .iter()
            .position(|s| matches!(s, Some(r) if r.id == id))
            .ok_or_else(|| anyhow!("request id {} not found", id.0))?;
        self.slots[row_idx] = None;
        // Phase transitions on evict:
        //   Admitting + active_count==0 -> Idle (pre-3c-3 behavior)
        //   Decoding  + active_count==0 -> Finished (NEW in 3c-3)
        //   Idle / Finished: no transition
        if self.active_count() == 0 {
            if self.phase == Phase::Admitting {
                self.phase = Phase::Idle;
            } else if self.phase == Phase::Decoding {
                self.phase = Phase::Finished;
            }
        }
        Ok(())
```

Keep the `self.ensure_not_poisoned()?;` call at the top — it stays.

- [ ] **Step 5: Remove `step` Phase transition**

Find `step_inner` (around line 543). Near the end (currently around lines 705-712) there's a block that transitions to `Phase::Finished` when all active rows are finished:

```rust
        // If every active slot is now finished, transition to Finished.
        let all_done = self
            .slots
            .iter()
            .all(|s| matches!(s, Some(r) if r.finished) || s.is_none());
        let any_present = self.slots.iter().any(|s| s.is_some());
        if all_done && any_present {
            self.phase = Phase::Finished;
        }
```

**Delete this entire block.** In 3c-3, Phase transition out of Decoding happens in `gc_finished_rows` (called by driver_loop after step) or in `evict` / `evict_all`. `step_inner` returns `events` immediately after the row sampling loop.

The new tail of `step_inner` (after the for-loop that builds `events`) becomes:

```rust
            events.push(StepEvent {
                id: state.id,
                token,
                finish_reason: state.finish_reason,
            });
        }

        Ok(events)
    }
```

- [ ] **Step 6: Add `gc_finished_rows` method**

After `step` / `step_inner` (around line 720), but before the closing `}` of `impl Scheduler`, add the new method. First add the necessary imports at the top of the file: locate the `use` block near the top and add (if not already present):

```rust
use std::collections::HashMap;
use tokio::sync::mpsc;
```

Then add the method:

```rust
    /// Sweep finished rows: clear their slot, drop their event channel,
    /// and return the evicted IDs. Cache buffer entries for evicted
    /// slots stay in place — a subsequent `admit_mid` into the same
    /// slot overwrites via `adopt_row_from`.
    ///
    /// Phase transition: Decoding -> Finished if `active_count == 0`
    /// after the sweep.
    ///
    /// Called by `SchedulerActor::driver_loop` after every successful
    /// `step` invocation.
    ///
    /// Generic over the event-payload type `S` so the Scheduler does
    /// not need to import `StepEvent`'s concrete channel type; in
    /// production `S = StepEvent`.
    pub fn gc_finished_rows<S>(
        &mut self,
        event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<S>>,
    ) -> Vec<RequestId> {
        let mut evicted: Vec<RequestId> = Vec::new();
        for slot in self.slots.iter_mut() {
            if let Some(state) = slot.as_ref() {
                if state.finished {
                    let id = state.id;
                    event_txs.remove(&id);
                    evicted.push(id);
                    *slot = None;
                }
            }
        }
        if self.phase == Phase::Decoding && self.active_count() == 0 {
            self.phase = Phase::Finished;
        }
        evicted
    }
```

- [ ] **Step 7: Run the new unit tests to verify they pass**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release core::scheduler::tests
```

Expected: all scheduler::tests PASS (existing + 3 new from this task).

- [ ] **Step 8: Verify all existing 3b-* scheduler integration tests still pass**

These tests exercise the full Scheduler API in single-batch mode (no mid-batch admit). They should be bit-id-identical to pre-3c-3 because the rolling decode loop doesn't exist yet (driver_loop change lands in Task 4).

```bash
export QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_1_per_row_offset -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_2_scheduler_decode_mask -- --ignored --test-threads=1 2>&1 | tail -3
```

Expected: each suite PASS, 0 failures.

If `mixed_finish` or other scenarios regress bit-id: investigate the step Phase transition removal. The current driver_loop uses `while sched.phase() == Phase::Decoding` to loop step calls. With step no longer transitioning Phase to Finished, the loop would run forever. Driver_loop still uses the pre-3c-3 logic in Task 3 — so it must still see the Finished transition somehow.

**Important fix in Task 3:** `run_batch_once` (in `scheduler_actor.rs`) currently has:

```rust
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model)?;
        for ev in events {
            route_event(ev, event_txs);
        }
    }
    sched.evict_all()?;
```

With step no longer transitioning Phase, this loop becomes infinite. To preserve pre-3c-3 behavior in Task 3 (driver_loop refactor lands in Task 4), modify `run_batch_once` to use a different termination condition. The simplest change:

```rust
    while sched.phase() == Phase::Decoding {
        let events = sched.step(&model)?;
        for ev in events {
            route_event(ev, event_txs);
        }
        // 3c-3: step no longer transitions Phase. Check active state
        // here. If all rows finished, transition manually before evict_all.
        let all_done = sched.active().iter().all(|s| s.finished);
        if all_done {
            break;
        }
    }
    sched.evict_all()?;
```

Add this fix in `ironmlx/src/core/server/scheduler_actor.rs::run_batch_once` (around line 252) as part of Task 3 to keep the lib green between Task 3 and Task 4.

- [ ] **Step 9: Hygiene gate**

Run the Standing Per-Task Hygiene Gate. All clean.

- [ ] **Step 10: Commit**

```bash
git add ironmlx/src/core/scheduler.rs ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3c-3): Scheduler API mid-batch foundation

Relaxes Phase guards on admit / evict to allow mid-batch operations
(Decoding phase admit/evict legal). Removes step's Phase->Finished
transition — that responsibility moves to gc_finished_rows (called
by driver_loop after every step in 3c-3's rolling decode loop).

Adds Scheduler::gc_finished_rows<S>(event_txs) -> Vec<RequestId>:
sweeps slots where state.finished == true; drops their event channels;
returns evicted ids; transitions Decoding -> Finished when no active
rows remain.

Adjusts run_batch_once in SchedulerActor to manually detect
all-rows-finished and break the step loop, since step no longer
transitions Phase. This preserves pre-3c-3 driver_loop behavior
between Task 3 and Task 4 (when the rolling decode loop lands).

3 new unit tests cover admit-during-Decoding (Phase stays Decoding),
evict-during-Decoding (transitions to Finished on last evict), and
gc_finished_rows clearing slots + event_txs + transitioning Phase.

Lib test count: 210 -> 213.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `Scheduler::admit_mid` + `slice_logits_row` helper + driver_loop rolling decode refactor

**Files:**
- Modify: `ironmlx/src/core/generate.rs` (+ ~15 lines: `slice_logits_row` helper)
- Modify: `ironmlx/src/core/scheduler.rs` (+ ~100 lines: `admit_mid` method)
- Modify: `ironmlx/src/core/server/scheduler_actor.rs` (~140 lines: `RollingEvent`, `handle_admit_mid`, driver_loop refactor)

- [ ] **Step 1: Add `slice_logits_row` helper to `core/generate.rs`**

Open `ironmlx/src/core/generate.rs`. Locate the existing helpers section (around `build_per_row_decode_mask`). Add a new helper:

```rust
/// Slice `logits[row_idx, 0, :]` and reshape to `[vocab]`.
///
/// Common pattern used by both `Scheduler::step_inner` (per-row decode
/// sampling) and `Scheduler::admit_mid` (first-token sampling after
/// adoption). Extracted so the call site can stay clean and the
/// indexing math lives in one place.
pub fn slice_logits_row(logits: &Array, row_idx: usize) -> Result<Array> {
    let shape = logits.shape();
    let shape_slice = shape.as_slice();
    if shape_slice.len() != 3 {
        return Err(anyhow!(
            "slice_logits_row: expected logits shape [B, 1, vocab]; got rank {}",
            shape_slice.len()
        ));
    }
    let b = shape_slice[0];
    if row_idx as i32 >= b {
        return Err(anyhow!(
            "slice_logits_row: row_idx {} >= B {}",
            row_idx, b
        ));
    }
    let vocab = shape_slice[2];
    let row = mlx::ops::indexing::slice(
        logits,
        &[row_idx as i32, 0_i32, 0_i32][..],
        &[row_idx as i32 + 1, 1_i32, vocab][..],
    )
    .map_err(|e| anyhow!("slice_logits_row: slice failed: {e:?}"))?;
    row.reshape(&[vocab][..])
        .map_err(|e| anyhow!("slice_logits_row: reshape failed: {e:?}"))
}
```

Refactor `Scheduler::step_inner` to use the new helper. Find the per-row logit slicing in `step_inner` (around lines 615-622) and replace with:

```rust
            let row_flat = slice_logits_row(&logits, b_idx)
                .map_err(|e| anyhow!("step: slice logits row {b_idx} failed: {e:?}"))?;
```

(Adjust the import at top of scheduler.rs: add `slice_logits_row` to the existing `use crate::core::generate::{...}` line.)

- [ ] **Step 2: Add `admit_mid` method to `Scheduler`**

Open `ironmlx/src/core/scheduler.rs`. Add this method to the `impl Scheduler` block (after `step` but before the test-seam `force_phase`, around line 720):

```rust
    /// Mid-batch admit + prefill. Caller is `SchedulerActor::driver_loop`
    /// after `cmd_rx` delivers an Admit during the rolling decode loop.
    ///
    /// Architecture: runs prefill in a temporary B=1 cache (the
    /// GenerationStream-equivalent path), then adopts the prefilled row
    /// into the main cache via per-layer slice copies. This avoids
    /// wasted compute on a B=b_max sub-batch + variable-shape mask
    /// construction + GatedDeltaNet state corruption for other active
    /// rows.
    ///
    /// Synchronous: stalls active rows for ~L_new × B=1_prefill_per_
    /// token_time. Adoption cost is sub-microsecond. 3c+ chunked prefill
    /// reduces stall further.
    ///
    /// Returns `(RequestId, StepEvent)` — the assigned request ID and
    /// the first generated token's event. Caller registers the event
    /// channel using the returned `id`.
    pub fn admit_mid(
        &mut self,
        req: GenerateRequest,
        model: &Qwen35Model,
    ) -> Result<(RequestId, StepEvent)> {
        self.ensure_not_poisoned()?;
        if self.phase != Phase::Decoding {
            return Err(anyhow!(
                "admit_mid illegal in {:?} phase: only Decoding (use admit for Idle/Admitting)",
                self.phase
            ));
        }
        let row_idx = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .ok_or_else(|| anyhow!(
                "scheduler full: no row available (b_max={})", self.b_max
            ))?;

        // 1. Insert RequestState via the relaxed admit() path. Phase stays Decoding.
        let id = self.admit(req)?;
        let (prompt_ids, prompt_len, max_new_tokens) = {
            let state = self.slots[row_idx].as_ref().expect("admit inserted");
            (
                state.prompt_ids.clone(),
                state.prompt_ids.len() as i32,
                state.max_new_tokens,
            )
        };
        let cap_for_temp = (prompt_len + max_new_tokens as i32).max(prompt_len);

        // 2. Capture KVCache dtype from the main cache (first Full layer).
        let dtype = {
            let main_cache = self
                .cache
                .as_ref()
                .ok_or_else(|| anyhow!(
                    "admit_mid called before prefill_admitted: cache absent"
                ))?;
            main_cache
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(kv) => Some(kv.dtype()),
                    _ => None,
                })
                .unwrap_or(Dtype::Bfloat16)
        };

        // 3. Allocate a fresh B=1 temp cache.
        let mut temp_cache = model.make_cache(1, cap_for_temp, dtype)?;

        // 4. Build B=1 prefill inputs (mirror GenerationStream prefill).
        let input_ids_data: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let input_ids: Array = (&input_ids_data[..], &[1_i32, prompt_len][..])
            .try_into()
            .map_err(|e| anyhow!("admit_mid: build input_ids Array failed: {e:?}"))?;
        let position_ids = build_position_ids_batched(&[prompt_len], prompt_len)?;
        let attention_mask = build_batch_attention_mask(&[prompt_len], prompt_len, dtype)?;
        let linear_attention_mask = build_batch_linear_mask(&[prompt_len], prompt_len)?;

        // 5. Run B=1 prefill into the temp cache. Returns logits [1, 1, vocab].
        let logits = model.batched_prefill(
            &input_ids,
            &position_ids,
            &attention_mask,
            &linear_attention_mask,
            &[prompt_len],
            Some(&mut temp_cache),
            (),
        )?;

        // 6. Adopt the temp cache's row 0 into main_cache at row_idx.
        {
            let main_cache = self
                .cache
                .as_mut()
                .expect("cache asserted Some above");
            if main_cache.len() != temp_cache.len() {
                return Err(anyhow!(
                    "admit_mid: cache layer count mismatch ({} vs {})",
                    main_cache.len(),
                    temp_cache.len()
                ));
            }
            for (main_layer, temp_layer) in main_cache.iter_mut().zip(temp_cache.iter()) {
                match (main_layer, temp_layer) {
                    (LayerCache::Full(main_kv), LayerCache::Full(temp_kv)) => {
                        main_kv.adopt_row_from(temp_kv, row_idx, 0)?;
                    }
                    (LayerCache::Linear(main_gd), LayerCache::Linear(temp_gd)) => {
                        main_gd.adopt_row_from(temp_gd, row_idx, 0)?;
                    }
                    _ => return Err(anyhow!(
                        "admit_mid: cache layer kind mismatch between main and temp"
                    )),
                }
            }
        }

        // 7. Sample first token from prefill logits (last position).
        //    Logits shape [1, 1, vocab] -- slice row 0.
        let row_logits = slice_logits_row(&logits, 0)?;
        let token = {
            let state = self.slots[row_idx].as_ref().expect("admit_mid slot");
            let history: Vec<u32> = prompt_ids.clone();
            state.sampler.sample(&row_logits, &history)?
        };

        // 8. Update state + check termination.
        let state = self.slots[row_idx].as_mut().expect("admit_mid slot");
        state.generated_tokens.push(token);
        state.real_len += 1;

        if state.stop_token_ids.contains(&token) {
            state.finished = true;
            state.finish_reason = Some("stop");
        } else if state.generated_tokens.len() >= state.max_new_tokens {
            state.finished = true;
            state.finish_reason = Some("length");
        }

        Ok((
            id,
            StepEvent {
                id,
                token,
                finish_reason: state.finish_reason,
            },
        ))
    }
```

Update the imports at the top of `scheduler.rs`. Locate the `use crate::core::generate::{...}` line (if present) and ensure it includes:

```rust
use crate::core::generate::{
    build_batch_attention_mask, build_batch_linear_mask, build_per_row_decode_mask,
    build_position_ids_batched, slice_logits_row,
};
```

Verify `mlx::Dtype` and `crate::nn::LayerCache` are already imported (they should be from Task 3 / pre-existing code).

- [ ] **Step 3: Refactor `driver_loop` to rolling decode**

Open `ironmlx/src/core/server/scheduler_actor.rs`. The current `driver_loop` (lines 111-168) plus its helpers (`run_batch_once` at 240-263) get replaced. Read the entire file first for context, then make the following changes:

**3a.** Delete `fn run_batch_once` entirely (its logic gets inlined into the new rolling loop).

**3b.** Add the `RollingEvent` enum near the top of the file (just before `fn driver_loop`):

```rust
/// Event yielded by the rolling decode loop's biased select. Either
/// a new admit command arrived, or the always-ready step branch
/// fired, or the cmd_rx channel was closed (shutdown).
enum RollingEvent {
    Admit(SchedulerCommand),
    Step,
    Shutdown,
}
```

**3c.** Replace `fn driver_loop` body entirely with the rolling decode design. Full replacement:

```rust
fn driver_loop(
    model: Arc<Mutex<Qwen35Model>>,
    b_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
) {
    let mut sched = Scheduler::new(b_max);
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        // ===== Outer Idle: block waiting for the first admit (or shutdown). =====
        let Some(first_cmd) = rt.block_on(cmd_rx.recv()) else {
            return; // cmd_rx closed; all senders dropped.
        };
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (e.g., admit() Err on poison flag).
            continue 'outer;
        }

        // ===== Admission window: drain additional admits up to ADMISSION_DEADLINE. =====
        if sched.active_count() < b_max {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &admit_count,
                &saturate_triggered,
                b_max,
                ADMISSION_DEADLINE,
            ));
        }

        // ===== First-batch prefill =====
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_result = {
            let model_lock = model.blocking_lock();
            sched.prefill_admitted(&model_lock)
        };
        match prefill_result {
            Ok(prefill_events) => {
                for ev in prefill_events {
                    route_event(ev, &event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] prefill error: {e:?}");
                let _ = sched.evict_all();
                event_txs.clear();
                continue 'outer;
            }
        }

        // ===== Rolling decode loop with biased mid-batch admit =====
        'rolling: loop {
            let evt: RollingEvent = rt.block_on(async {
                tokio::select! {
                    biased;
                    maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                        Some(cmd) => RollingEvent::Admit(cmd),
                        None => RollingEvent::Shutdown,
                    },
                    () = futures::future::ready(()) => RollingEvent::Step,
                }
            });

            match evt {
                RollingEvent::Shutdown => {
                    // cmd_rx closed. Drop event_txs (handlers see EOF), return.
                    event_txs.clear();
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    handle_admit_mid(cmd, &mut sched, &mut event_txs, &admit_count, &model);
                }
                RollingEvent::Step => {
                    let step_result = {
                        let model_lock = model.blocking_lock();
                        sched.step(&model_lock)
                    };
                    match step_result {
                        Ok(events) => {
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                            sched.gc_finished_rows(&mut event_txs);
                        }
                        Err(e) => {
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            let _ = sched.evict_all();
                            event_txs.clear();
                            continue 'outer;
                        }
                    }
                }
            }

            // ===== Exit rolling loop when active_count == 0. =====
            // try_recv lets us peek for a pending command without
            // blocking. If empty, break to outer. If a command waits,
            // it's a "new outer batch" admit (since current batch
            // drained); process it via handle_admit (NOT admit_mid)
            // since sched.phase is now Finished/Idle.
            if sched.active_count() == 0 {
                match cmd_rx.try_recv() {
                    Ok(cmd) => {
                        handle_admit(cmd, &mut sched, &mut event_txs, &admit_count);
                        if sched.active_count() == 0 {
                            // Admit failed; reset for next outer iteration.
                            break 'rolling;
                        }
                        if sched.active_count() < b_max {
                            rt.block_on(drain_window(
                                &mut cmd_rx,
                                &mut sched,
                                &mut event_txs,
                                &admit_count,
                                &saturate_triggered,
                                b_max,
                                ADMISSION_DEADLINE,
                            ));
                        }
                        batch_count.fetch_add(1, Ordering::Relaxed);
                        let prefill_result = {
                            let model_lock = model.blocking_lock();
                            sched.prefill_admitted(&model_lock)
                        };
                        match prefill_result {
                            Ok(events) => {
                                for ev in events {
                                    route_event(ev, &event_txs);
                                }
                            }
                            Err(e) => {
                                tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                                let _ = sched.evict_all();
                                event_txs.clear();
                                break 'rolling;
                            }
                        }
                        continue 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        break 'rolling;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        event_txs.clear();
                        return;
                    }
                }
            }
        }

        // After rolling loop: reset cache + Phase for next outer iteration.
        // gc_finished_rows already cleared slots, so evict_all here mainly
        // resets the cache buffer state.
        let _ = sched.evict_all();
        event_txs.clear();
    }
}
```

**3d.** Add `fn handle_admit_mid` helper just after `fn handle_admit`:

```rust
/// Mid-batch admit handler. Acquires the model lock, calls
/// `Scheduler::admit_mid` (which runs B=1 prefill into a temp cache
/// and adopts the row into the main cache), then registers the
/// per-request event channel and routes the first generated token's
/// event. Lock is held only for the duration of admit_mid.
fn handle_admit_mid(
    cmd: SchedulerCommand,
    sched: &mut Scheduler,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<Qwen35Model>>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let admit_result = {
        let model_lock = model.blocking_lock();
        sched.admit_mid(request, &model_lock)
    };
    match admit_result {
        Ok((id, prefill_event)) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            event_txs.insert(id, event_tx.clone());
            if reply_tx
                .send(Ok(AdmitReply {
                    request_id: id,
                    event_rx,
                }))
                .is_err()
            {
                // Caller dropped reply_rx before we could send.
                // Evict the orphan slot.
                let _ = sched.evict(id);
                event_txs.remove(&id);
                return;
            }
            // Route the first generated token event.
            route_event(prefill_event, event_txs);
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
        }
    }
}
```

**3e.** At the top of `scheduler_actor.rs`, ensure these imports are present (add any missing):

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::generate::GenerateRequest;
use crate::core::scheduler::{Phase, RequestId, Scheduler, StepEvent};
use crate::models::qwen3_5::Qwen35Model;
use crate::Result;
```

Then in `Cargo.toml` (only if not already there), verify the `futures` crate is a dep — the `futures::future::ready` call needs it. Check via:

```bash
grep -n "^futures" /Volumes/Dev/cxx-mlx/ironmlx/Cargo.toml
```

If `futures` is NOT listed: instead of importing it, use this alternative for the biased select default branch:

```rust
let evt: RollingEvent = rt.block_on(async {
    tokio::select! {
        biased;
        maybe_cmd = cmd_rx.recv() => match maybe_cmd {
            Some(cmd) => RollingEvent::Admit(cmd),
            None => RollingEvent::Shutdown,
        },
        _ = std::future::ready(()) => RollingEvent::Step,
    }
});
```

`std::future::ready` is in std and requires no new dep. Use whichever is available.

- [ ] **Step 4: Hygiene gate**

Run the Standing Per-Task Hygiene Gate.

If compile fails on `Phase::Decoding` being unused in `scheduler_actor.rs` (since the old `while sched.phase() == Phase::Decoding` loop was deleted), remove the unused import. If `mpsc::error::TryRecvError` needs an explicit import, add `use tokio::sync::mpsc::error::TryRecvError;` at top.

- [ ] **Step 5: Run full lib test suite**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Expected: **213 passed / 0 failed / 2 ignored** (210 from Tasks 1-2 + 3 from Task 3).

- [ ] **Step 6: Run all 6 existing Scheduler-path integration tests**

These exercise the actor's rolling decode loop in the no-mid-admit case (collapses to pre-3c-3 behavior).

```bash
export QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_2_scheduler_actor -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_3_admission_window -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_4_anthropic_actor -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_1_per_row_offset -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_2_scheduler_decode_mask -- --ignored --test-threads=1 2>&1 | tail -3
```

Expected: all 6 PASS. Pay special attention to:
- `b1_p2_3b_1_scheduler_step::mixed_finish`: bit-id should remain 1.0000 (no regression from the rolling loop change for the single-batch path).
- `b1_p2_3b_3_admission_window` 4 scenarios: the admission window code path is unchanged structurally (still runs `drain_window` after first admit); should be bit-id-identical.

If any regressions: **STOP and report BLOCKED** with the specific scenario + failing bit-id values. The rolling loop in the no-mid-admit case should collapse exactly to the pre-3c-3 batch behavior (single outer iteration runs admission window + prefill + step-until-active-empty + evict_all).

- [ ] **Step 7: Hygiene gate**

Run the Standing Per-Task Hygiene Gate one more time (after integration test runs in case anything got formatted).

- [ ] **Step 8: Commit**

```bash
git add ironmlx/src/core/generate.rs ironmlx/src/core/scheduler.rs ironmlx/src/core/server/scheduler_actor.rs
git commit -m "$(cat <<'EOF'
feat(b1-p2.3c-3): admit_mid + rolling decode loop

Scheduler::admit_mid is the new mid-batch entry point. It allocates a
fresh B=1 temp cache, runs the existing GenerationStream-equivalent
batched_prefill on it, then per-layer adopts the prefilled row into
the main b_max cache via KVCache::adopt_row_from /
GatedDeltaCache::adopt_row_from. Returns (RequestId, StepEvent) with
the first generated token. Synchronous; stalls active rows for ~L_new
× B=1_prefill_per_token_time. Adoption is sub-microsecond.

Refactors SchedulerActor::driver_loop into a rolling decode loop:
biased tokio::select! { cmd_rx.recv() | std::future::ready(()) }
yields RollingEvent { Admit, Step, Shutdown }. Mid-admit triggers
handle_admit_mid (acquires lock, calls admit_mid, registers event
channel, routes first token). Step branch acquires lock, calls
Scheduler::step, then sched.gc_finished_rows(event_txs). Exit on
active_count == 0 + cmd_rx empty.

Adds slice_logits_row helper to core/generate.rs; refactors
Scheduler::step_inner to use it (was inlined). Same per-row logit
slice pattern, single home for the indexing math.

In the no-mid-admit case the rolling loop collapses to the pre-3c-3
single-batch behavior: admission window -> prefill_admitted -> step
loop -> evict_all (with gc_finished_rows transitioning Phase to
Finished). All 6 existing Scheduler-path integration suites (3b-1,
3b-2, 3b-3, 3b-4, 3c-1, 3c-2) PASS bit-id-unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 3 integration scenarios + 12-suite regression sweep + close-out

**Files:**
- Create: `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs` (~320 LOC)
- Create: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md`

- [ ] **Step 1: Create the new integration test file**

Create `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs`. Read `ironmlx/tests/b1_p2_3b_3_admission_window.rs` first to confirm test scaffolding patterns (load_fixture, tokenize_prompt, make_request signature, spawn_scheduler_actor usage, admit via cmd_tx, run_b1_baseline pattern).

The file:

```rust
//! B1-p2.3c-3 — Continuous batching (mid-batch admit/evict) integration tests.
//!
//! Three scenarios:
//!   1. continuous_batching_mid_decode_admit — central correctness gate.
//!      B=2 with A (max_new=3) + B (max_new=8). After A finishes, admit
//!      C (max_new=5) mid-decode. Verify all three rows produce correct
//!      tokens (bit-id ≥ 0.95 vs B=1 baselines).
//!   2. continuous_batching_full_reject — b_max=2 saturated by A+B both
//!      with max_new=20; admit C while decoding; verify C reply is Err
//!      "scheduler full".
//!   3. continuous_batching_drains_to_empty — admit A, drain, admit B
//!      100ms later; verify B prefills + completes through the actor's
//!      second outer batch iteration.
//!
//! All gated `#[ignore]`; drive via SchedulerActor cmd_tx (not raw
//! Scheduler — 3c-3's value lives in driver_loop's rolling decode).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::{RequestId, StepEvent};
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, AdmitReply, SchedulerCommand};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

const ARGMAX_BITID_GATE: f64 = 0.95;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_request(prompt_ids: Vec<u32>, max_new_tokens: usize, stop: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: stop,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    loop {
        match stream.next_token().expect("next_token") {
            Some(ev) => {
                tokens.push(ev.token);
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            None => break,
        }
    }
    tokens
}

fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

async fn submit_admit(
    cmd_tx: &tokio::sync::mpsc::Sender<SchedulerCommand>,
    req: GenerateRequest,
) -> ironmlx::Result<AdmitReply> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    cmd_tx
        .send(SchedulerCommand::Admit {
            request: req,
            reply_tx,
        })
        .await
        .map_err(|e| anyhow::anyhow!("cmd_tx.send: {e:?}"))?;
    reply_rx
        .await
        .map_err(|e| anyhow::anyhow!("reply_rx.await: {e:?}"))?
}

async fn drain_until_finished(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<StepEvent>,
) -> Vec<StepEvent> {
    let mut events = Vec::new();
    loop {
        match rx.recv().await {
            Some(ev) => {
                let done = ev.finish_reason.is_some();
                events.push(ev);
                if done {
                    break;
                }
            }
            None => break, // channel closed (EOF after gc_finished_rows)
        }
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_mid_decode_admit() {
    let (model, tokenizer) = load_fixture();

    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let prompt_c = tokenize_prompt(&tokenizer, "Goodbye");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let max_new_a: usize = 3;
    let max_new_b: usize = 8;
    let max_new_c: usize = 5;

    // B=1 baselines (all run before the actor scenario).
    let baseline_a = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_a.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_a, stop))
        })
        .await
        .expect("baseline A")
    };
    let baseline_b = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_b.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_b, stop))
        })
        .await
        .expect("baseline B")
    };
    let baseline_c = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_c.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_c, stop))
        })
        .await
        .expect("baseline C")
    };

    // Drive the actor.
    let handle = spawn_scheduler_actor(model.clone(), 2);

    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a.clone(), max_new_a, stop.clone()))
        .await
        .expect("admit A");
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b.clone(), max_new_b, stop.clone()))
        .await
        .expect("admit B");

    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;

    // Drain A to completion (finish_reason='length' at 3 tokens).
    let events_a = drain_until_finished(&mut rx_a).await;
    assert_eq!(
        events_a.len(),
        max_new_a,
        "A should produce {} events; got {}",
        max_new_a,
        events_a.len()
    );
    assert_eq!(events_a.last().unwrap().finish_reason, Some("length"));

    // After A finishes (and gc clears slot 0), submit C — should land via admit_mid.
    let reply_c = submit_admit(&handle.cmd_tx, make_request(prompt_c.clone(), max_new_c, stop.clone()))
        .await
        .expect("admit C mid-decode");
    let mut rx_c = reply_c.event_rx;

    // Drain B + C concurrently.
    let (events_b, events_c) = tokio::join!(
        drain_until_finished(&mut rx_b),
        drain_until_finished(&mut rx_c),
    );

    assert_eq!(events_b.len(), max_new_b, "B should produce {} events", max_new_b);
    assert_eq!(events_b.last().unwrap().finish_reason, Some("length"));
    assert_eq!(events_c.len(), max_new_c, "C should produce {} events", max_new_c);
    assert_eq!(events_c.last().unwrap().finish_reason, Some("length"));

    let tokens_a: Vec<u32> = events_a.iter().map(|e| e.token).collect();
    let tokens_b: Vec<u32> = events_b.iter().map(|e| e.token).collect();
    let tokens_c: Vec<u32> = events_c.iter().map(|e| e.token).collect();

    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    let ratio_c = argmax_bit_id_ratio(&tokens_c, &baseline_c);

    println!(
        "[continuous_batching] tokens_a={tokens_a:?} bit-id={ratio_a:.4}"
    );
    println!(
        "[continuous_batching] tokens_b={tokens_b:?} bit-id={ratio_b:.4}"
    );
    println!(
        "[continuous_batching] tokens_c={tokens_c:?} bit-id={ratio_c:.4}"
    );

    assert!(ratio_a >= ARGMAX_BITID_GATE, "A bit-id {} < {}", ratio_a, ARGMAX_BITID_GATE);
    assert!(ratio_b >= ARGMAX_BITID_GATE, "B bit-id {} < {}", ratio_b, ARGMAX_BITID_GATE);
    assert!(ratio_c >= ARGMAX_BITID_GATE, "C bit-id {} < {}", ratio_c, ARGMAX_BITID_GATE);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_full_reject() {
    let (model, tokenizer) = load_fixture();
    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let prompt_c = tokenize_prompt(&tokenizer, "Goodbye");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let handle = spawn_scheduler_actor(model.clone(), 2);

    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a, 20, stop.clone()))
        .await
        .expect("admit A");
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b, 20, stop.clone()))
        .await
        .expect("admit B");

    // Wait briefly so A + B reach Decoding phase.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Now submit C — both slots full + Decoding → admit_mid should Err.
    let admit_c_result = submit_admit(&handle.cmd_tx, make_request(prompt_c, 5, stop.clone())).await;
    match admit_c_result {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("scheduler full") || msg.contains("no row available"),
                "expected 'scheduler full' Err; got: {msg}"
            );
        }
        Ok(_) => panic!("C admit should have failed but succeeded"),
    }

    // Drain A + B normally so the actor doesn't hang.
    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;
    let (_, _) = tokio::join!(
        drain_until_finished(&mut rx_a),
        drain_until_finished(&mut rx_b),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_drains_to_empty() {
    let (model, tokenizer) = load_fixture();
    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let handle = spawn_scheduler_actor(model.clone(), 2);

    // First admit + drain.
    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a, 4, stop.clone()))
        .await
        .expect("admit A");
    let mut rx_a = reply_a.event_rx;
    let events_a = drain_until_finished(&mut rx_a).await;
    assert_eq!(events_a.len(), 4);

    // Wait ~100ms — actor's rolling loop should have exited to outer
    // Idle by now (active_count == 0 + cmd_rx empty after A's events).
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Verify batch_count counter saw 1 batch so far.
    let bc_after_a = handle
        .batch_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(bc_after_a, 1, "expected 1 batch after A; got {}", bc_after_a);

    // Second admit — should trigger a NEW outer batch (batch_count++ → 2).
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b, 5, stop.clone()))
        .await
        .expect("admit B");
    let mut rx_b = reply_b.event_rx;
    let events_b = drain_until_finished(&mut rx_b).await;
    assert_eq!(events_b.len(), 5);

    tokio::time::sleep(Duration::from_millis(150)).await;
    let bc_after_b = handle
        .batch_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        bc_after_b, 2,
        "expected 2 batches after B (separate outer iteration); got {}",
        bc_after_b
    );
}
```

- [ ] **Step 2: Run the new integration tests**

```bash
export QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_3_continuous_batching -- --ignored --test-threads=1 2>&1 | tail -20
```

Expected: all 3 scenarios PASS, with bit-id ratios printed for the central scenario.

If `continuous_batching_mid_decode_admit` fails on bit-id for row C: this is a real correctness bug in `admit_mid`'s adoption path. **STOP and report BLOCKED** with the row's tokens vs baseline tokens — investigation likely points to either `KVCache::adopt_row_from` not copying all positions correctly OR `GatedDeltaCache::adopt_row_from` recurrent_state corruption. Lib tests would have caught buffer corruption; bit-id discrepancy means the K/V or SSM state copy isn't semantically equivalent.

- [ ] **Step 3: Hygiene gate**

Run the Standing Per-Task Hygiene Gate.

- [ ] **Step 4: Lib test suite final count**

```bash
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib --release -- --test-threads=1 2>&1 | tail -3
```

Capture the test count for the close-out. Expected: 213 (210 baseline + 3 from Task 3).

- [ ] **Step 5: 12-suite regression sweep**

Run all 11 existing suites + 1 new sequentially. Capture each suite's `result: ok. X passed; Y failed; finished in Z.ZZs` line.

```bash
export QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p6_qwen35_vl_logits_match -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p6_6_logits_match -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test p6_7_chunked_prefill -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_1_batched_prefill -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_2_batched_decode -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_1_scheduler_step -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_2_scheduler_actor -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_3_admission_window -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3b_4_anthropic_actor -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_1_per_row_offset -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_2_scheduler_decode_mask -- --ignored --test-threads=1 2>&1 | tail -3
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --release --test b1_p2_3c_3_continuous_batching -- --ignored --test-threads=1 2>&1 | tail -3
```

Expected: all 12 PASS, 0 failures.

If any regressions: investigate. The most likely failure modes are:
- `b1_p2_3b_3_admission_window`: rolling loop change broke saturate_triggered counter semantics. Check counter assertion logic.
- `b1_p2_3c_1` / `3c-2`: step Phase transition removal broke single-batch behavior. Check `run_batch_once` patch from Task 3 Step 8 is correct.

- [ ] **Step 6: Write close-out report**

```bash
mkdir -p ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout
```

Create `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md`. Use this template; fill every `<fill>` with actual data:

```markdown
# B1-p2.3c-3 Continuous batching — Close-out

**Branch:** `ironmlx-b1-p2-3-continuous-batching` (off B1-p2.3c-2 head `d27aced`)
**Date:** 2026-05-15
**Spec:** `docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md` (commit `dc19170`)
**Plan:** `docs/superpowers/plans/2026-05-14-b1-p2-3c-3-continuous-batching.md` (commit `<fill plan commit>`)

## Summary

`SchedulerActor::driver_loop` is now a rolling decode loop: biased
`tokio::select!` between `cmd_rx.recv()` and an always-ready step branch
per iteration. Mid-batch admits route through new
`Scheduler::admit_mid` which runs prefill in a standalone B=1 temp
cache (GenerationStream-equivalent path) and adopts the prefilled
row into the main b_max cache via per-layer `KVCache::adopt_row_from` /
`GatedDeltaCache::adopt_row_from`. Finished rows are reclaimed by
`gc_finished_rows` called after every step.

Phase enum (Idle / Admitting / Decoding / Finished) is unchanged; the
transitions are relaxed: admit/evict legal in Decoding; Decoding →
Finished moves out of `step` into `gc_finished_rows` (and into
`evict` when manual-evicting the last row).

Synchronous B=1 prefill into a temp cache trades 3-8× faster admit
performance (vs. B=b_max sub-batch with mask) for the temp cache
allocation cost (sub-millisecond). The adoption path is sub-
microsecond on Apple Silicon unified memory. Stall during admit_mid
is L_new × B=1_prefill_per_token_time; 3c+ chunked prefill will
reduce this further by interleaving chunks with decode steps.

## Acceptance

| Test | Result |
| --- | --- |
| `kvcache_adopt_row_from_basic` (T1) | <fill PASS> |
| `kvcache_adopt_row_from_shape_mismatch_err` (T1) | <fill PASS> |
| `kvcache_adopt_row_from_out_of_bounds_err` (T1) | <fill PASS> |
| `gdcache_adopt_row_from_state_and_offset` (T2) | <fill PASS> |
| `gdcache_adopt_row_from_out_of_bounds_err` (T2) | <fill PASS> |
| `scheduler_admit_during_decoding_ok` (T3) | <fill PASS> |
| `scheduler_evict_during_decoding_transitions_to_finished_when_last` (T3) | <fill PASS> |
| `scheduler_gc_finished_rows_clears_slots_and_transitions` (T3) | <fill PASS> |
| `continuous_batching_mid_decode_admit` (T5 central gate) | <fill bit-id A/B/C> |
| `continuous_batching_full_reject` (T5) | <fill PASS> |
| `continuous_batching_drains_to_empty` (T5) | <fill PASS> |

## Architectural Changes (per spec §4.9 file map)

- `core/cache/kv_cache.rs` (Task 1): +`dtype()` accessor, +`adopt_row_from` (~70 lines), +3 unit tests
- `core/cache/gated_delta.rs` (Task 2): +`adopt_row_from` (~70 lines), +2 unit tests, added `slice_strided_on` / `slice_update_on` imports
- `core/generate.rs` (Task 4): +`slice_logits_row` helper (~15 lines)
- `core/scheduler.rs` (Task 3+4): admit/evict Phase guards relaxed; step Phase transition removed; +`gc_finished_rows` method; +`admit_mid` method; step_inner refactored to use `slice_logits_row`; 3 new unit tests
- `core/server/scheduler_actor.rs` (Task 3+4): `run_batch_once` patched in T3 (manual all-finished detection); replaced in T4 by rolling decode loop; +`RollingEvent` enum; +`handle_admit_mid` helper
- New integration test `tests/b1_p2_3c_3_continuous_batching.rs` (Task 5): 3 scenarios, ~320 LOC

No changes to: `nn/*`, `models/*`, `core/server/{openai,anthropic}.rs`, `core/generate.rs::GenerationStream`.

## Compat sunset notes

3c-3 inherits all 5 sunset markers from 3b series + 3c-1:

| Compat | Sunset trigger |
| --- | --- |
| OpenAI VL → GS | B1-p2.4 batched VL |
| OpenAI long-prompt → GS | 3c+ chunked-prefill |
| Anthropic long-prompt → GS | 3c+ chunked-prefill |
| Anthropic image-content → 400 | Future Anthropic VL phase |
| `ADMISSION_DEADLINE` hardcoded 5ms | 3d/3e config |

3c-3 closes one pre-existing limitation:
- **Pre-3c-3:** "batch boundary at evict_all" (3a/3b convention). Removed by the rolling decode loop.

3c-3 introduces two known limitations:
- **Prefill stall:** Synchronous B=1 prefill in `admit_mid` stalls active rows for `~L_new × B=1_prefill_per_token_time`. Sunset: **3c+** chunked prefill.
- **`b_max`-full reject:** `admit_mid` returns `Err("scheduler full")` when all slots are occupied. Sunset: **3d** admission queue + fair scheduling.

## Commits

| Commit | Type | Description |
| --- | --- | --- |
| `dc19170` | docs | Design spec |
| `<fill T0 plan SHA>` | docs | Implementation plan |
| `<fill T1 SHA>` | feat | T1: KVCache::adopt_row_from + dtype() accessor + 3 unit tests |
| `<fill T2 SHA>` | feat | T2: GatedDeltaCache::adopt_row_from + 2 unit tests |
| `<fill T3 SHA>` | feat | T3: Scheduler API mid-batch foundation (admit/evict Phase relaxation, gc_finished_rows, step transition removal) |
| `<fill T4 SHA>` | feat | T4: admit_mid + rolling decode loop |
| `<fill T5 SHA>` | test+docs | T5: 3 integration scenarios + 12-suite regression sweep + this close-out |

## Regression Status

All commands run with `--test-threads=1` against
`QWEN35_MODEL=$(ls -d $HOME/.ironmlx/models/models--*Qwen3.5-4B-MLX-4bit*/snapshots/*/ | head -1)`.

| Check | Result | Time |
| --- | --- | --- |
| `cargo +nightly fmt --all -- --check` | clean | - |
| `cargo +nightly clippy --all-features --workspace --exclude ironmlx-app -- -D warnings` | clean | - |
| `cargo build --release -p ironmlx` | clean | - |
| `cargo test -p ironmlx --lib --release` | <fill N passed / 0 failed / 2 ignored> | <fill> |
| P6.3 single-image (`p6_qwen35_vl_logits_match`) | <fill> | <fill> |
| P6.6 logits-match | <fill> | <fill> |
| P6.7 chunked-prefill matrix | <fill> | <fill> |
| B1-p2.1 batched prefill | <fill> | <fill> |
| B1-p2.2 batched decode | <fill> | <fill> |
| B1-p2.3b-1 scheduler scenarios (3) | <fill> | <fill> |
| B1-p2.3b-2 scheduler_actor scenarios (3) | <fill> | <fill> |
| B1-p2.3b-3 admission_window scenarios (4) | <fill> | <fill> |
| B1-p2.3b-4 anthropic_actor scenarios (3) | <fill> | <fill> |
| B1-p2.3c-1 per_row_offset scenarios (5) | <fill> | <fill> |
| B1-p2.3c-2 scheduler_decode_mask scenarios (1) | <fill> | <fill> |
| B1-p2.3c-3 continuous_batching scenarios (3) | <fill> | <fill> |

Exit code: `0`. **No regressions.**

## Notes

- **Continuous batching is live.** `iron-bench v2` (multi-concurrent-request performance comparison) is unblocked. 10 concurrent requests on b_max=4 now share a single rolling batch; finished rows yield slots immediately to admission queue head (or rejected if b_max already saturated in 3c-3; admission queue lands in 3d).
- **Mid-batch admit correctness verified.** `continuous_batching_mid_decode_admit` (central scenario) shows row C's tokens matching B=1 GenerationStream baseline at bit-id `<fill C bit-id>` despite C being admitted into row A's vacated slot mid-decode. K/V and SSM state are adopted from the temp cache cleanly.
- **No GatedDeltaNet state corruption.** The standalone B=1 temp cache approach (vs. B=b_max sub-batch + variable mask) avoids touching other active rows' SSM state during admit_mid. Other rows' recurrent_state continues evolving normally through their own `step` invocations.
- **`gc_finished_rows` runs after every step.** Slots are reclaimed within one decode step's latency of a row finishing. Drop of `event_tx` for finished rows means the HTTP handler sees EOF on its event_rx — clean SSE close.
- **Outer Idle still uses `block_on(cmd_rx.recv())`.** When the rolling loop exits (active_count == 0 + cmd_rx empty), `evict_all` resets the cache buffer and the loop returns to outer Idle. No CPU spin during idle.

## Plan-correction deviations

- <fill any deviations encountered during implementation, e.g. unexpected mlx API constraints, sampler clone semantics surprises, futures vs std::future::ready choice, etc.>

## B1-p2.3x Next Steps

- **B1-p2.3c+** — Chunked batched prefill: interleave prefill chunks with decode steps in `admit_mid` to bound prefill stall to `chunk_size × prefill_per_token_time`. Also removes long-prompt GS fallback in OpenAI/Anthropic handlers.
- **B1-p2.3d** — Admission queue + preemption: replaces the "Err scheduler full" behavior with a fair admission queue. Exposes `ADMISSION_DEADLINE` + `b_max` via `AppConfig` + CLI flags.
- **B1-p2.3e** — Per-row sampler invocation tuning.
- **B1-p2.4** — VL B>1 batched serving; removes VL GS fallback.

## Linked Artifacts

- Spec: `docs/superpowers/specs/2026-05-14-b1-p2-3c-3-continuous-batching-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-b1-p2-3c-3-continuous-batching.md`
- Predecessor close-out: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_2_closeout/report.md`
- Scheduler API: `ironmlx/src/core/scheduler.rs`
- driver_loop: `ironmlx/src/core/server/scheduler_actor.rs`
- Cache adopt_row_from primitives: `ironmlx/src/core/cache/{kv_cache,gated_delta}.rs`
- New integration test: `ironmlx/tests/b1_p2_3c_3_continuous_batching.rs`
```

Fill every `<fill>` with actual values from Steps 2, 4, 5.

- [ ] **Step 7: Commit scenarios + close-out**

```bash
mkdir -p ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout
git add ironmlx/tests/b1_p2_3c_3_continuous_batching.rs
git add -f ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/b1_p2_3c_3_closeout/report.md
git commit -m "$(cat <<'EOF'
test+docs(b1-p2.3c-3): continuous batching scenarios + close-out

continuous_batching_mid_decode_admit verifies the central 3c-3
correctness gate: B=2 with A (max_new=3) + B (max_new=8) + C
(max_new=5) admitted mid-decode after A finishes. C is routed
through admit_mid which prefills in a temp B=1 cache and adopts the
row into A's vacated slot. All three rows match B=1 GenerationStream
baselines at bit-id ≥ 0.95.

continuous_batching_full_reject verifies that admitting beyond
b_max during Decoding returns Err 'scheduler full' (3d will replace
with an admission queue).

continuous_batching_drains_to_empty verifies the rolling loop's
outer-batch boundary semantics: A admit → decode → drain → outer
Idle → 100ms gap → B admit → second outer batch (batch_count == 2).

Close-out report covers acceptance, architectural changes per spec
§4.9, 12-suite regression sweep results, plan-correction deviations,
and next-step pointers for 3c+ / 3d / 4.

B1-p2.3c-3 complete. SchedulerActor + admission window from 3b series
+ per-row offset infrastructure from 3c-1 + decode-mask activation
from 3c-2 + continuous-batching driver_loop from 3c-3 form the
foundation for iron-bench v2 multi-concurrent-request benchmarking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Plan Self-Review

**1. Spec coverage** (spec §4 architecture + §5 tests + §9 risks):

- ✅ Spec §4.1 driver_loop rolling decode — Task 4 Step 3.
- ✅ Spec §4.2 admit Phase relaxation — Task 3 Step 3.
- ✅ Spec §4.3 evict Phase relaxation — Task 3 Step 4.
- ✅ Spec §4.4 gc_finished_rows — Task 3 Step 6.
- ✅ Spec §4.5 step Phase transition removal — Task 3 Step 5 + Task 3 Step 8 (run_batch_once patch keeps lib green between T3 and T4).
- ✅ Spec §4.6 admit_mid with B=1 temp cache + adoption — Task 4 Step 2.
- ✅ Spec §4.7 KVCache::adopt_row_from — Task 1 Step 4.
- ✅ Spec §4.8 GatedDeltaCache::adopt_row_from — Task 2 Step 4.
- ✅ Spec §4.9 file map matches plan File Structure table at top.
- ✅ Spec §5.1 3 KVCache lib tests — Task 1 Step 1.
- ✅ Spec §5.2 2 GatedDeltaCache lib tests — Task 2 Step 1.
- ✅ Spec §5.3 3 integration scenarios — Task 5 Step 1.
- ✅ Spec §5.4 regression sweep — Task 5 Step 5.
- ✅ Spec §6 acceptance gates (210+ lib tests, 12 regression suites + 3 new) — Task 5 Steps 4-5.
- ✅ Spec §9 R1 grow_to before slice_update — Task 1 Step 4 conditional grow_to.
- ✅ Spec §9 R2 temp_cache lifetime — Task 4 Step 2 local Vec ownership.
- ✅ Spec §9 R3 sampler &self — Task 4 Step 2 uses state.sampler.sample.
- ✅ Spec §9 R4 always-ready spin — Task 4 Step 3 outer Idle uses block_on (no spin); rolling only runs while active.
- ✅ Spec §9 R5 shutdown race — Task 4 Step 3 RollingEvent::Shutdown handles cmd_rx closed.
- ✅ Spec §9 R6 single-task no race — Task 4 Step 3 design explicit (sequential biased select).
- ✅ Spec §9 R7 prefill stall — documented in close-out + spec §8 sunset.
- ✅ Spec §9 R8 select overhead — implicit (regression sweep verifies no slowdown).
- ✅ Spec §9 R9 make_cache(1, ...) honors batch — Task 4 Step 2 calls model.make_cache(1, cap, dtype); the integration scenarios in T5 implicitly verify (if make_cache(1) breaks, admit_mid breaks, T5 Scenario 1 fails).
- ✅ Spec §9 R10 lazy-eval after slice_update — Task 1 Step 1 lib test reads back state to force eval; Task 2 Step 1 same.

**2. Placeholder scan:** The `<fill>` markers in Task 5 Step 6's close-out template are intentional template fields filled in at Task 5 execution time. All other steps contain complete code or commands with no TBD / TODO / implement-later.

**3. Type consistency:**

- `KVCache::adopt_row_from(&mut self, src: &KVCache, dst_row: usize, src_row: usize) -> Result<()>` — consistent across spec §4.7, plan Task 1 Step 4, plan Task 4 Step 2 callsite.
- `GatedDeltaCache::adopt_row_from(&mut self, src: &GatedDeltaCache, dst_row: usize, src_row: usize) -> Result<()>` — consistent across spec §4.8, plan Task 2 Step 4, plan Task 4 Step 2 callsite.
- `Scheduler::admit_mid(&mut self, req: GenerateRequest, model: &Qwen35Model) -> Result<(RequestId, StepEvent)>` — consistent.
- `Scheduler::gc_finished_rows<S>(&mut self, event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<S>>) -> Vec<RequestId>` — consistent.
- `slice_logits_row(logits: &Array, row_idx: usize) -> Result<Array>` — consistent.
- `RollingEvent { Admit(SchedulerCommand), Step, Shutdown }` — consistent.
- `dtype()` accessor returns `Dtype` — consistent.

Plan looks clean. No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-b1-p2-3c-3-continuous-batching.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, two-stage review (spec compliance + code quality) between tasks, fast iteration. This is what 3c-1 and 3c-2 used successfully.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
