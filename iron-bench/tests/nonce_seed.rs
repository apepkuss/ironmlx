//! Verifies clap accepts the production `--nonce-seed N` flag. Exact nonce
//! sequence semantics are covered by runner unit tests in `src/runner.rs`.

use std::process::Command;

#[test]
fn nonce_seed_accepts_flag() {
    let bin = env!("CARGO_BIN_EXE_iron-bench");
    let out = Command::new(bin)
        .args([
            "--target",
            "bogus=http://127.0.0.1:1",
            "--model",
            "x",
            "--model-dir",
            "/tmp/nonexistent",
            "--prompt-len",
            "16",
            "--max-tokens",
            "1",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--nonce-seed",
            "42",
            "--format",
            "csv",
        ])
        .output()
        .expect("iron-bench spawn");
    let code = out.status.code().unwrap_or(-1);
    assert_ne!(
        code,
        2,
        "clap arg parse rejected --nonce-seed flag: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}
