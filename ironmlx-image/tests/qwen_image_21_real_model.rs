#![cfg(target_os = "macos")]

//! Real-checkpoint acceptance gate for mlx-community/Qwen-Image-2.1-MLX-4bit.
//!
//! Run with:
//! `IRONMLX_QWEN_IMAGE_MODEL_DIR=<snapshot> cargo test -p ironmlx-image --test qwen_image_21_real_model -- --ignored --nocapture`

use std::path::PathBuf;

use ironmlx_image::models::{QwenImage21GenerationConfig, QwenImage21Pipeline};
use mlx::Dtype;

#[test]
#[ignore = "requires the pinned 10.5GB Qwen Image 2.1 MLX checkpoint"]
#[serial_test::serial(mlx_metal)]
fn loads_and_generates_rgba_image() {
    let metallib = PathBuf::from(
        std::env::var("MLX_DIR").expect("MLX_DIR must point to the local MLX install"),
    )
    .join("lib/mlx.metallib");
    mlx::metal::set_metallib_path(
        metallib
            .to_str()
            .expect("MLX_DIR/lib/mlx.metallib must be UTF-8"),
    )
    .expect("load MLX metallib");
    let model_dir = PathBuf::from(
        std::env::var("IRONMLX_QWEN_IMAGE_MODEL_DIR")
            .expect("IRONMLX_QWEN_IMAGE_MODEL_DIR must point to the model snapshot"),
    );
    let pipeline = QwenImage21Pipeline::load(&model_dir).expect("load real Qwen Image pipeline");
    let image = pipeline
        .generate(
            "A small red circle centered on a white background.",
            QwenImage21GenerationConfig {
                width: 256,
                height: 256,
                inference_steps: 2,
                seed: 7,
            },
            || false,
        )
        .expect("generate real Qwen Image output");
    assert_eq!(image.shape().as_slice(), &[1, 4, 256, 256]);
    let pixels = image
        .astype(Dtype::Float32)
        .and_then(|array| array.to_vec::<f32>())
        .expect("materialize generated pixels");
    assert!(pixels.iter().all(|value| value.is_finite()));
    assert!(pixels.iter().all(|value| (-1.0..=1.0).contains(value)));
    assert!(pixels.iter().any(|value| value.abs() > 1e-6));
}
