# P6.2 PatchEmbed Reshape Alignment — Design

**Status:** Approved (brainstormed 2026-05-11)
**Owner:** ironmlx
**Parent:** P6.1 (commit `91bee2e` on branch `ironmlx-p6-1-vision-diff`)
**Branch target:** continue on `ironmlx-p6-1-vision-diff` (no new branch — small fix)

## 1. Motivation

P6.1's baseline diff report (`ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/2026-05-11-1251/report.md`) shows:

- `01_patch_embed_out` max_diff = **7.0156** (99.4% of 1228800 values differ > 1e-3)
- `02_pos_embed_contrib` max_diff = 0.0312 (~identical)
- `04_rotary_freqs` max_diff = 0.0 (perfect)
- `15_block_10_out` onward escalates 177 → 269 → **4691** at block_23
- `29_merger_out` max_diff = 8.08

Investigation in brainstorming confirmed the root cause is **byte-layout misinterpretation in the P6.1 test driver**:

- mlx-vlm processor outputs `pixel_values` as `[N, 1536]` where 1536 is C-major (`(C, T, H, W)` row-major flatten).
- This is enforced by mlx-vlm's `PatchEmbed.__call__` at `/Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py:114-120`:
  ```python
  hidden_states.reshape(-1, C, T, H, W).moveaxis(1, 4)  # → [N, T, H, W, C]
  ```
- Disk weight is `[1024, 2, 16, 16, 3]` = `(Cout, kT, kH, kW, Cin)` (mlx Conv3d NHWC).
- ironmlx's internal pipeline (patchify in `image_processor.rs` → `PatchEmbed.forward`) is internally consistent: produces `[N, T, C, H, W]`, transposes to `[N, T, H, W, C]`, flattens to `[N, 1536]` (T-major), matmul with weight `[1024, 1536]` (T-major flatten of disk).
- The bug is ONLY in P6.1's diff test driver (`p6_vision_dump.rs`) and P6's Task 21 logits-match test (`p6_qwen35_vl_logits_match.rs`), which reshape `[N, 1536]` as `[N, 2, 16, 16, 3]` (T-major) instead of `[N, 3, 2, 16, 16]` (C-major).

The bytes are physically the same, but the geometric interpretation is wrong — every `(c, t, h, w)` tuple in the test is mapped to the wrong pixel. The error propagates through PatchEmbed and accumulates through 24 ViT blocks.

This means **the P6 production path is correct**; the P6.1 diff numbers were measuring our test driver's reshape bug, not a vision-encoder defect. P6.2 fixes the test driver alone.

## 2. Goals

- Fix the byte-layout misinterpretation in `p6_vision_dump.rs` and `p6_qwen35_vl_logits_match.rs`.
- Add one Rust unit test that verifies the C-major reshape against a hand-constructed reference (1536 = 3·2·16·16 with a known C-T-H-W gradient pattern).
- Re-run P6.1 pipeline and confirm `01_patch_embed_out` max_diff drops from 7.0156 to < 0.1.
- Re-run Task 21 logits-match (with the same fix applied) and confirm acceptance gates still pass (greedy first-token bit-identical; max_diff < 0.52).
- Update P6.1's committed baseline report with a post-fix run for documentation.

## 3. Non-Goals

- No changes to production code (`vision/*.rs`, `image_processor.rs`).
- No changes to mlx-vlm fork (Task 5 of P6.1 stays).
- No vision encoder dtype/numerical changes.
- No Item 3 semantic re-test gating (that's qualitative; we'll spot-check but it's not a hard gate).

## 4. The Fix (exact code change)

### 4.1 `ironmlx/tests/p6_vision_dump.rs`

Replace lines 42–46 (current incorrect reshape):

```rust
// mlx-vlm preprocesses to [N, 1536] (flattened) — reshape to [N, 2, 16, 16, 3]
// then transpose to [N, 2, 3, 16, 16] to match VisionTower input convention.
let n = pv_flat.shape().as_slice()[0];
let pv_5d = pv_flat.reshape(&[n, 2, 16, 16, 3][..]).expect("reshape pv");
let pv = pv_5d
    .transpose_axes(&[0, 1, 4, 2, 3][..])
    .expect("transpose pv");
```

with the C-major interpretation matching mlx-vlm's `PatchEmbed`:

```rust
// mlx-vlm's processor packs pixel_values as [N, 1536] where 1536 = C*T*H*W
// in C-major (C-T-H-W row-major) order — see mlx_vlm/models/qwen3_vl/vision.py:114-120
// (`reshape(-1, C, T, H, W).moveaxis(1, 4)` → `[N, T, H, W, C]`).
// We reshape that as [N, C, T, H, W] then transpose to [N, T, C, H, W] which is
// ironmlx's `VisionTower::forward` input contract.
let n = pv_flat.shape().as_slice()[0];
let pv_5d = pv_flat.reshape(&[n, 3, 2, 16, 16][..]).expect("reshape pv");
let pv = pv_5d
    .transpose_axes(&[0, 2, 1, 3, 4][..])
    .expect("transpose pv");
```

### 4.2 `ironmlx/tests/p6_qwen35_vl_logits_match.rs`

Apply the same change to the equivalent block (Task 21 testimplementation). Read the file, locate the reshape from `[N, 1536]`, apply the same two-line correction with a comment pointing at this spec.

### 4.3 New unit test: reshape semantics validator

Add to `ironmlx/src/models/qwen3_5/image_processor.rs` test module (it's where pixel-layout knowledge lives):

```rust
#[test]
fn mlxvlm_c_major_reshape_to_ironmlx_layout() {
    // Construct [1, 1536] with gradient 0..1536 (f32). mlx-vlm processor
    // packs this in (C, T, H, W) C-major. After reshape [1,3,2,16,16] +
    // transpose [0,2,1,3,4] we should get a tensor `out` such that
    // out[0, t, c, h, w] == c*2*16*16 + t*16*16 + h*16 + w.
    let flat: Vec<f32> = (0..1536).map(|i| i as f32).collect();
    let pv: mlx::Array = (flat.as_slice(), &[1, 1536][..]).try_into().unwrap();
    let pv_5d = pv.reshape(&[1, 3, 2, 16, 16][..]).unwrap();
    let pv_out = mlx::ops::shape::transpose_axes(&pv_5d, &[0, 2, 1, 3, 4][..]).unwrap();
    assert_eq!(pv_out.shape().as_slice(), &[1, 2, 3, 16, 16]);
    let v: Vec<f32> = pv_out.to_vec().unwrap();
    // Spot-check several positions against the C-major formula
    let check = |t: usize, c: usize, h: usize, w: usize| {
        let dst = ((((0 * 2) + t) * 3 + c) * 16 + h) * 16 + w;  // out[0,t,c,h,w] flat idx
        let expected = c * 2 * 16 * 16 + t * 16 * 16 + h * 16 + w;
        assert_eq!(
            v[dst] as usize,
            expected,
            "mismatch at (t={t}, c={c}, h={h}, w={w}): got {} expected {expected}",
            v[dst]
        );
    };
    check(0, 0, 0, 0);
    check(0, 0, 0, 1);  // (c=0,t=0,h=0,w=1) → byte 1
    check(0, 1, 0, 0);  // (c=1,t=0,h=0,w=0) → byte 512
    check(1, 0, 0, 0);  // (c=0,t=1,h=0,w=0) → byte 256
    check(1, 2, 15, 15);  // (c=2,t=1,h=15,w=15) → byte 1535
}
```

## 5. Acceptance Gates

1. **Numerical drop in P6.1 diff**: re-run `run_p6_1_diff.sh`; new `01_patch_embed_out` max_diff < 0.1 (currently 7.0156).
2. **Final tensor drop**: new `29_merger_out` max_diff < 1.0 (currently 8.08).
3. **Task 21 unchanged**: `p6_qwen35_vl_logits_match` test still PASSes with `max_diff < 0.52` and greedy first_token = 760 (the threshold may improve; gate is "doesn't regress + still passes").
4. **Unit test passes**: new reshape-semantics test passes.
5. **Production regression**: `cargo test -p ironmlx --lib --release -- --test-threads=1` reports 152 passed (151 baseline + 1 new), 0 failed.

If gate 1 fails (i.e., the reshape change does NOT drop max_diff), the root cause hypothesis is wrong — revert the fix commit and re-brainstorm.

## 6. Out of Scope (deferred)

- Item 3 long-decode semantic verification (qualitative; do a spot-check but not a gate).
- Vision encoder dtype work.
- Op-level (sub-block) dump granularity from P6.1's plan §13.

## 7. Files

- Modify: `ironmlx/tests/p6_vision_dump.rs` (2 lines reshape/transpose + comment update)
- Modify: `ironmlx/tests/p6_qwen35_vl_logits_match.rs` (same 2-line reshape/transpose + comment)
- Modify: `ironmlx/src/models/qwen3_5/image_processor.rs` (add 1 unit test in existing tests module)
- Update: `ironmlx/tests/fixtures/p6_qwen35_vl/diff_reports/` — commit the post-fix report under a new timestamp directory; keep the old `2026-05-11-1251/` for before/after comparison.

## 8. Effort

~1 hour single-implementer.

## 9. Rollback Plan

If acceptance gate 1 fails:

1. `git revert <fix-commit>`
2. Re-open brainstorming for P6.2 with the failed-hypothesis evidence in the new diff report.
3. Likely next investigation: dump the actual mlx-vlm processor pixel_values bytes for the first patch and confirm by hand the byte-to-pixel mapping, comparing against our ironmlx patchify output for the same image.
