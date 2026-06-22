//! CLI generate + MTP real-checkpoint smoke tests.
//!
//! These tests are ignored by default because they require local MLX runtime and
//! Hugging Face checkpoint snapshots.

use std::path::PathBuf;
use std::process::Command;

fn require_env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

fn coco_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("p6_qwen35_vl")
        .join("coco_sample.jpg")
}

#[test]
#[ignore = "requires QWEN35_MODEL, QWEN35_MTP_MODEL, and MLX_DIR pointing to real local checkpoints"]
fn qwen35_vl_generate_with_mtp_accepts_image_request() {
    let model_dir = require_env_path("QWEN35_MODEL");
    let mtp_model_dir = require_env_path("QWEN35_MTP_MODEL");
    let output = Command::new(env!("CARGO_BIN_EXE_ironmlx"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("generate")
        .arg("--model")
        .arg(&model_dir)
        .arg("--mtp-model-dir")
        .arg(&mtp_model_dir)
        .arg("--mtp-draft-tokens")
        .arg("1")
        .arg("--image")
        .arg(coco_path())
        .arg("--prompt")
        .arg("Describe this image.")
        .arg("--max-tokens")
        .arg("1")
        .arg("--temperature")
        .arg("0")
        .output()
        .expect("run ironmlx generate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "generate exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
    assert!(
        !stdout.trim().is_empty(),
        "generate should emit at least one token; stderr:\n{stderr}"
    );
}
