//! P6.1 vision dump integration test.
//!
//! Driven by `IRONMLX_VISION_DUMP_DIR`, `QWEN35_MODEL`, and `PIXEL_VALUES_PATH`
//! env vars set by `tests/fixtures/p6_qwen35_vl/run_p6_1_diff.sh`. Reads the
//! mlx-vlm-prepared `00_pixel_values.safetensors`, drives one forward pass
//! through `VisionTower`, and as a side effect causes the 29 `dump_tensor`
//! sites in `vision/mod.rs` to write their tensors into the dump dir.
//!
//! Run with:
//! ```text
//! QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
//! PIXEL_VALUES_PATH=/path/to/00_pixel_values.safetensors \
//! IRONMLX_VISION_DUMP_DIR=/tmp/p6_diff/rust \
//! MLX_DIR=$HOME/.local/mlx \
//! cargo test -p ironmlx --features vision-dump --test p6_vision_dump --release -- --ignored
//! ```

#![cfg(feature = "vision-dump")]

use std::path::Path;

use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::Qwen35Config;
use ironmlx::models::vision::VisionTower;

#[test]
#[ignore] // requires QWEN35_MODEL + PIXEL_VALUES_PATH + IRONMLX_VISION_DUMP_DIR
fn p6_vision_dump() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let pv_path = std::env::var("PIXEL_VALUES_PATH").expect("PIXEL_VALUES_PATH env required");
    let _dump_dir =
        std::env::var("IRONMLX_VISION_DUMP_DIR").expect("IRONMLX_VISION_DUMP_DIR env required");

    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let vc = cfg.vision_config.expect("vision_config");
    let tower = VisionTower::from_loader(&loader, &vc).expect("tower");

    // mlx-vlm's processor packs pixel_values as [N, 1536] where the 1536 inner
    // dim is the (C, T, H, W) C-major row-major flatten — see
    // /Volumes/Dev/mlx-vlm/mlx_vlm/models/qwen3_vl/vision.py:114-120
    // (`reshape(-1, C, T, H, W).moveaxis(1, 4)`). We reshape that as
    // [N, C=3, T=2, H=16, W=16] then transpose [0,2,1,3,4] to land on
    // [N, T, C, H, W] = ironmlx's VisionTower::forward input contract.
    // See docs/superpowers/specs/2026-05-11-p6-2-patch-embed-reshape-design.md.
    let (mut loaded, _meta) = mlx::io::load_safetensors(&pv_path).expect("load pixel_values");
    let pv_flat = loaded.remove("tensor").expect("tensor key");
    let n = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat.reshape(&[n, 3, 2, 16, 16][..]).expect("reshape pv");
    let pv = pv_5d
        .transpose_axes(&[0, 2, 1, 3, 4][..])
        .expect("transpose pv");

    // Grid for the COCO sample: image_grid_thw = [[1, 30, 40]] (Task 20 fixture).
    let grids: Vec<(i32, i32, i32)> = vec![(1, 30, 40)];

    let embeds = tower.forward(&pv, &grids).expect("vision forward");
    // Force eval so all 29 dump calls complete before the test exits.
    mlx::transforms::eval(&[&embeds]).expect("eval embeds");

    eprintln!("[p6_vision_dump] forward complete; dumps should be in IRONMLX_VISION_DUMP_DIR");
}
