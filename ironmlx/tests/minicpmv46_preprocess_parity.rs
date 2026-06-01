//! Parity test for the MiniCPM-V-4.6 single-image (no-slice) preprocessor.
//!
//! Verifies that `image_processor::preprocess(coco_sample.jpg)` reproduces the
//! mlx-vlm `MiniCPMVImageProcessor` single-slice (`slice_mode=False`) output
//! byte-close: the packed `[1, 14, n*14, 3]` (HWC) pixel tensor + the
//! `[grid_h, grid_w]` grid.
//!
//! Fixtures (gitignored; regenerate via
//! `tests/fixtures/minicpmv46_vl/gen_vision_embeds.py`):
//!   - `input_pixel_values.npy` — `[1, 14, n*14, 3]` f32, the post-transpose
//!     HWC tensor the SigLIP embeddings layer consumes.
//!   - `input_grid.npy` — `[grid_h, grid_w]` i32.
//!
//! `#[ignore]` + fixture-gated (the test reads the fixtures from disk;
//! `MINICPMV46_MODEL` is only documented as the fixture origin).
//!
//! Run:
//! ```text
//! source ~/.local/mlx/mlx-env.sh && \
//!   cargo test --release -p ironmlx --test minicpmv46_preprocess_parity -- --ignored --nocapture
//! ```

use ironmlx::models::minicpmv4_6::image_processor::preprocess;

mod common;
use common::minicpmv46_parity::{load_npy_in, to_f32_vec, FIXTURE_DIR_VL};

use mlx::Array;

fn load_npy(name: &str) -> Array {
    load_npy_in(FIXTURE_DIR_VL, name)
}

#[test]
#[ignore = "fixture-gated MiniCPM-V-4.6 preprocess parity"]
fn preprocess_matches_mlx_vlm_single_slice() {
    let img_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/p6_qwen35_vl/coco_sample.jpg"
    );
    let bytes = std::fs::read(img_path).expect("read coco_sample.jpg");

    // Expected grid [grid_h, grid_w] i32.
    let grid: Vec<i32> = load_npy("input_grid.npy").to_vec().expect("grid to_vec");
    assert_eq!(grid.len(), 2, "input_grid.npy must be [grid_h, grid_w]");
    let (exp_grid_h, exp_grid_w) = (grid[0], grid[1]);

    // Expected packed pixel_values [1, 14, n*14, 3] f32.
    let exp_arr = load_npy("input_pixel_values.npy");
    let exp_shape = exp_arr.shape().to_vec();
    let exp_pix = to_f32_vec(&exp_arr);

    let (pixel_values, grid_h, grid_w) = preprocess(&bytes).expect("preprocess");

    println!(
        "grid: ours=({grid_h},{grid_w}) expected=({exp_grid_h},{exp_grid_w})  \
         shape: ours={:?} expected={exp_shape:?}",
        pixel_values.shape()
    );

    assert_eq!(
        (grid_h, grid_w),
        (exp_grid_h, exp_grid_w),
        "grid mismatch (resize math _find_best_resize/_ensure_divide is off)"
    );
    assert_eq!(
        pixel_values.shape().to_vec(),
        exp_shape,
        "pixel_values shape mismatch"
    );

    // Compare against fixture as f32 regardless of our dtype (we may emit bf16).
    let ours = to_f32_vec(&pixel_values);
    assert_eq!(ours.len(), exp_pix.len(), "element count mismatch");

    let max_abs = ours
        .iter()
        .zip(&exp_pix)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("pixel_values max_abs diff = {max_abs}");

    // Bound rationale (observed max_abs ≈ 0.0235): pixel values are normalized
    // to ~[-1, 1] (mean/std = 0.5). The resize is a byte-exact port of PIL's
    // bicubic resampler (see `image_processor::pil_bicubic_resize`), and
    // normalize + pack are exact arithmetic reorderings. The residual was
    // localized (probe, removed) to the JPEG DECODER: the `image` crate's
    // libjpeg vs PIL's libjpeg differ by ≤3/255 on ~1.4% of pixels BEFORE any
    // resize (max_abs_u8=3 → 3/255 / 0.5 = 0.0235 normalized, which exactly
    // matches the observed value). That is the irreducible decode floor; the
    // resize/normalize/pack add essentially nothing on top of it. 0.05 is the
    // spec-locked acceptance ceiling and is NOT loosened post-hoc — it sits
    // ~2× above the decode floor and well below the resample-interpolation
    // floor (CatmullRom would give 0.0549).
    assert!(
        max_abs < 0.05,
        "pixel_values max_abs {max_abs} exceeds 0.05 (JPEG decode floor / normalize / pack mismatch)"
    );
}
