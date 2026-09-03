//! Parity test for the MiniCPM-V-4.6 LLaVA-UHD MULTI-SLICE preprocessor
//! (P3 Task 2).
//!
//! Verifies that `image_processor::preprocess_sliced(coco_sample.jpg, 9)`
//! reproduces mlx-vlm's `MiniCPMVImageProcessor.preprocess` with slicing ENABLED
//! (`slice_mode=True`, `max_slice_nums=9`):
//!   - the slice COUNT matches (source overview + `gx*gy` refine patches);
//!   - each slice's grid `(grid_h, grid_w)` matches mlx-vlm's `tgt_sizes`;
//!   - each slice's packed `[1, 14, n*14, 3]` (HWC) pixel tensor is byte-close.
//!
//! Fixture image: `qwen35_vl/coco_sample.jpg` (640×480). It DOES slice:
//! ratio = 640*480/448² = 1.5306 → ceil → multiple=2 → best_grid (gx=2, gy=1)
//! → 3 slices. Source grid (gh,gw)=(28,36); each refine patch (gh,gw)=(40,28).
//!
//! Fixtures (gitignored; regenerate via
//! `tests/fixtures/minicpmv46_vl/gen_multislice_preprocess.py`):
//!   - `multislice_count.npy`      — int32 [1] total slice count.
//!   - `multislice_grids.npy`      — int32 [count, 2] per-slice (grid_h, grid_w).
//!   - `multislice_pixels_{i}.npy` — f32 [1, 14, n_i*14, 3] slice i pixel tensor.
//!
//! `#[ignore]` + fixture-gated.
//!
//! Run:
//! ```text
//! source ~/.local/mlx/mlx-env.sh && \
//!   cargo test --release -p ironmlx --test minicpmv46_multislice_preprocess_parity \
//!   -- --ignored --nocapture
//! ```

use ironmlx::models::minicpmv4_6::image_processor::{preprocess_sliced, MAX_SLICE_NUMS};

mod common;
use common::minicpmv46_parity::{load_npy_in, to_f32_vec, FIXTURE_DIR_VL};

fn load_npy(name: &str) -> mlx::Array {
    load_npy_in(FIXTURE_DIR_VL, name)
}

#[test]
#[ignore = "fixture-gated MiniCPM-V-4.6 multi-slice preprocess parity"]
fn preprocess_sliced_matches_mlx_vlm() {
    let img_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen35_vl/coco_sample.jpg"
    );
    let bytes = std::fs::read(img_path).expect("read coco_sample.jpg");

    // Expected slice count.
    let count_v: Vec<i32> = load_npy("multislice_count.npy")
        .to_vec()
        .expect("count to_vec");
    assert_eq!(count_v.len(), 1, "multislice_count.npy must be [1]");
    let exp_count = count_v[0] as usize;

    // Expected per-slice grids [count, 2].
    let grids_arr = load_npy("multislice_grids.npy");
    assert_eq!(
        grids_arr.shape().to_vec(),
        vec![exp_count as i32, 2],
        "multislice_grids.npy must be [count, 2]"
    );
    let grids: Vec<i32> = grids_arr.to_vec().expect("grids to_vec");

    // Ours.
    let slices = preprocess_sliced(&bytes, MAX_SLICE_NUMS).expect("preprocess_sliced");

    println!("slice count: ours={} expected={exp_count}", slices.len());
    assert_eq!(
        slices.len(),
        exp_count,
        "slice count mismatch (1 source + gx*gy patches)"
    );

    let mut worst_max_abs = 0.0_f32;
    let mut worst_slice = 0usize;
    for (i, (pixel_values, grid_h, grid_w)) in slices.iter().enumerate() {
        let (exp_gh, exp_gw) = (grids[i * 2], grids[i * 2 + 1]);
        println!(
            "slice {i}: grid ours=({grid_h},{grid_w}) expected=({exp_gh},{exp_gw}) \
             shape ours={:?}",
            pixel_values.shape()
        );
        assert_eq!(
            (*grid_h, *grid_w),
            (exp_gh, exp_gw),
            "slice {i} grid mismatch (slice/refine resize math off)"
        );

        let exp_arr = load_npy(&format!("multislice_pixels_{i}.npy"));
        assert_eq!(
            pixel_values.shape().to_vec(),
            exp_arr.shape().to_vec(),
            "slice {i} pixel_values shape mismatch"
        );

        let ours = to_f32_vec(pixel_values);
        let exp = to_f32_vec(&exp_arr);
        assert_eq!(ours.len(), exp.len(), "slice {i} element count mismatch");
        let max_abs = ours
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("slice {i}: pixel_values max_abs diff = {max_abs}");
        if max_abs > worst_max_abs {
            worst_max_abs = max_abs;
            worst_slice = i;
        }
    }

    println!("worst slice = {worst_slice} max_abs = {worst_max_abs}");

    // Bound rationale (same JPEG-decode floor as the single-image parity test):
    // pixel values are normalized to ~[-1, 1] (mean/std = 0.5). The slice/refine
    // resizes are byte-exact ports of PIL's bicubic resampler and the crop +
    // normalize + pack are exact integer/arithmetic reorderings, so the only
    // residual is the JPEG DECODER: the `image` crate's libjpeg vs PIL's libjpeg
    // differ by ≤3/255 on the decoded pixels BEFORE any resize (3/255 / 0.5 =
    // 0.0235 normalized). The refine path resizes that already-decoded buffer at
    // a LARGER target (784×560 here) and crops it — the decode residual does not
    // amplify under resampling (the bicubic kernel is a weighted average ≤1), so
    // the multi-slice floor stays in the same ~0.0235 band as single-image.
    // 0.05 is the spec-locked acceptance ceiling, NOT loosened post-hoc.
    assert!(
        worst_max_abs < 0.05,
        "slice {worst_slice} pixel_values max_abs {worst_max_abs} exceeds 0.05 \
         (JPEG decode floor / slice-resize / crop / normalize / pack mismatch)"
    );
}
