//! P6.6 multi-image vision dump integration test.
//!
//! Driven by `IRONMLX_VISION_DUMP_DIR`, `QWEN35_MODEL`, `PIXEL_VALUES_PATH`,
//! `IMAGE_GRID_THW_PATH` env vars set by `run_p6_6_diff.sh`. Reads the
//! mlx-vlm-prepared `expected_pixel_values.safetensors` (concatenated 2-image
//! patches in C-major [N_total, 1536] layout) and `expected_image_grid_thw.npy`
//! (shape `[2, 3]`), drives ONE `VisionTower::forward` over the concatenated
//! input, and dumps the output as `vision_embeds.safetensors` in the dump dir.
//! Op-level intermediate dumps fire automatically via the existing dump_tensor
//! hooks (29 module-level + 96 intra-block) when IRONMLX_VISION_DUMP_DIR is set.
//!
//! Run with:
//! ```text
//! QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit \
//! PIXEL_VALUES_PATH=/tmp/p6_diff_multi/python/expected_pixel_values.safetensors \
//! IMAGE_GRID_THW_PATH=/tmp/p6_diff_multi/python/expected_image_grid_thw.npy \
//! IRONMLX_VISION_DUMP_DIR=/tmp/p6_diff_multi/rust \
//! MLX_DIR=$HOME/.local/mlx \
//! cargo test -p ironmlx --features vision-dump --test p6_6_multi_image_dump --release -- --ignored
//! ```

#![cfg(feature = "vision-dump")]

use std::collections::HashMap;
use std::path::Path;

use mlx::Dtype;

use ironmlx::core::Loader;
use ironmlx::models::qwen3_5::vision::VisionTower;
use ironmlx::models::qwen3_5::Qwen35Config;

#[test]
#[ignore] // requires QWEN35_MODEL + PIXEL_VALUES_PATH + IMAGE_GRID_THW_PATH + IRONMLX_VISION_DUMP_DIR
fn p6_6_multi_image_dump() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env required");
    let pv_path = std::env::var("PIXEL_VALUES_PATH").expect("PIXEL_VALUES_PATH env required");
    let grid_path = std::env::var("IMAGE_GRID_THW_PATH").expect("IMAGE_GRID_THW_PATH env required");
    let dump_dir =
        std::env::var("IRONMLX_VISION_DUMP_DIR").expect("IRONMLX_VISION_DUMP_DIR env required");

    let loader = Loader::open_multimodal(Path::new(&model_dir)).expect("loader");
    let cfg = Qwen35Config::from_loader(&loader).expect("config");
    let vc = cfg.vision_config.expect("vision_config");
    let tower = VisionTower::from_loader(&loader, &vc).expect("tower");

    // Load mlx-vlm's [N_total, 1536] pixel_values (C-major) and reshape to
    // [N_total, T, C, H, W] (same conversion as p6_vision_dump.rs / P6.2 spec).
    // mlx-vlm packs as reshape(-1, C, T, H, W).moveaxis(1,4) → [N, T, H, W, C]
    // but stores as [N, 1536] C-major row-major flatten. We invert:
    //   [N_total, 1536] → reshape [N, 3, 2, 16, 16] → transpose [0,2,1,3,4]
    //   → [N_total, T=2, C=3, H=16, W=16] (ironmlx VisionTower contract).
    let (mut loaded, _meta) = mlx::io::load_safetensors(&pv_path).expect("load pixel_values");
    let pv_flat = loaded.remove("tensor").expect("tensor key in pv");
    let n_total = pv_flat.shape().as_slice()[0];
    let pv_5d = pv_flat
        .reshape(&[n_total, 3, 2, 16, 16][..])
        .expect("reshape pv");
    let pv = pv_5d
        .transpose_axes(&[0, 2, 1, 3, 4][..])
        .expect("transpose pv");

    // Load grid_thw from .npy. Shape [N, 3] (N images, each row = [T, H, W]).
    let grid_arr = mlx::io::load_npy(&grid_path).expect("load grid_thw npy");
    let grid_i32 = mlx::ops::cast::astype(&grid_arr, Dtype::Int32).expect("cast grid to i32");
    mlx::transforms::eval(&[&grid_i32]).expect("eval grid");
    let grids_flat: Vec<i32> = grid_i32.to_vec::<i32>().expect("grid to_vec");
    let grids: Vec<(i32, i32, i32)> = grids_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    assert!(!grids.is_empty(), "P6.6 expects at least 1 image, got 0");

    eprintln!(
        "[p6_6_multi_image_dump] pv.shape={:?} grids={:?}",
        pv.shape().as_slice(),
        grids
    );

    // Run vision tower forward. Op-level dumps fire via the dump_tensor hooks
    // inserted at 29 module + 96 intra-block sites (P6.1 + P6.3b).
    let embeds = tower.forward(&pv, &grids).expect("vision forward");
    mlx::transforms::eval(&[&embeds]).expect("eval embeds");

    // Save final vision_embeds for Gate 2 (concatenated 2-image merger output).
    let out_path = format!("{dump_dir}/vision_embeds.safetensors");
    let embeds_bf16 =
        mlx::ops::cast::astype(&embeds, Dtype::Bfloat16).expect("cast embeds to bf16");
    let mut tensors: HashMap<String, mlx::Array> = HashMap::new();
    tensors.insert("tensor".to_string(), embeds_bf16);
    let metadata: HashMap<String, String> = HashMap::new();
    mlx::io::save_safetensors(&out_path, &tensors, &metadata).expect("save embeds");

    eprintln!(
        "[p6_6_multi_image_dump] vision_embeds.shape={:?} saved to {}",
        embeds.shape().as_slice(),
        dump_dir
    );
}
