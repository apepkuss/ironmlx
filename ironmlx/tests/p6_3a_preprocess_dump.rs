//! P6.3a: dump ironmlx's `image_processor::preprocess` output for byte-level
//! comparison against mlx-vlm's processor output (Gate 1 diagnostic).
//!
//! Driven by:
//!   IMAGE_PATH=/path/to/coco_sample.jpg
//!   IRONMLX_PREPROCESS_DUMP_DIR=/tmp/p6_diff/ironmlx_pre
//!   MLX_DIR=$HOME/.local/mlx
//!   cargo test -p ironmlx --features vision-dump --test p6_3a_preprocess_dump --release -- --ignored

#![cfg(feature = "vision-dump")]

use std::path::Path;

use mlx::Dtype;

use ironmlx::models::qwen3_5::image_processor::preprocess;

#[test]
#[ignore]
fn p6_3a_preprocess_dump() {
    let img_path = std::env::var("IMAGE_PATH").expect("IMAGE_PATH env required");
    let out_dir = std::env::var("IRONMLX_PREPROCESS_DUMP_DIR")
        .expect("IRONMLX_PREPROCESS_DUMP_DIR env required");

    let bytes = std::fs::read(Path::new(&img_path)).expect("read image");
    let (pv_native, grid_h, grid_w) = preprocess(&bytes).expect("preprocess");

    eprintln!(
        "[p6_3a_preprocess_dump] grid_thw=[1,{grid_h},{grid_w}] native_shape={:?}",
        pv_native.shape().as_slice()
    );

    // Dump native [N, T=2, C=3, 16, 16] layout (bf16)
    {
        let path = format!("{out_dir}/00_ironmlx_pv_native.safetensors");
        let mut map = std::collections::HashMap::new();
        let pv_bf16 = mlx::ops::cast::astype(&pv_native, Dtype::Bfloat16).expect("cast");
        mlx::transforms::eval(&[&pv_bf16]).expect("eval native");
        map.insert("tensor".to_string(), pv_bf16);
        let metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        mlx::io::save_safetensors(&path, &map, &metadata).expect("save native");
    }

    // Re-shape to mlx-vlm's [N, 1536] C-major layout for direct byte-diff:
    // ironmlx is [N, T, C, H, W]; transpose [0, 2, 1, 3, 4] -> [N, C, T, H, W]
    // then reshape [N, C*T*H*W] = [N, 1536].
    let n = pv_native.shape().as_slice()[0];
    let pv_c_first =
        mlx::ops::shape::transpose_axes(&pv_native, &[0_i32, 2, 1, 3, 4][..]).expect("transpose");
    let pv_flat = pv_c_first.reshape(&[n, 1536_i32][..]).expect("reshape");
    let pv_flat_bf16 = mlx::ops::cast::astype(&pv_flat, Dtype::Bfloat16).expect("cast flat");
    mlx::transforms::eval(&[&pv_flat_bf16]).expect("eval flat");
    {
        let path = format!("{out_dir}/00_ironmlx_pv_vlmlayout.safetensors");
        let mut map = std::collections::HashMap::new();
        map.insert("tensor".to_string(), pv_flat_bf16);
        let metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        mlx::io::save_safetensors(&path, &map, &metadata).expect("save flat");
    }

    eprintln!("[p6_3a_preprocess_dump] dumped native + vlmlayout to {out_dir}");
}
