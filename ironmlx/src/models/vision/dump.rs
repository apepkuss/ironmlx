//! Compile-gated tensor dump for the P6.1 diff pipeline.
//!
//! When the `vision-dump` cargo feature is OFF (default / production builds),
//! [`dump_tensor`] is a `#[inline] fn _: () {}` no-op that the compiler erases.
//! When the feature is ON, the function reads `IRONMLX_VISION_DUMP_DIR` and,
//! if set, eagerly evaluates and saves the tensor as
//! `<dir>/<name>.safetensors`. See spec
//! `docs/superpowers/specs/2026-05-11-p6-1-vision-diff-pipeline-design.md`.

use mlx::Array;

#[cfg(feature = "vision-dump")]
pub fn dump_tensor(name: &str, t: &Array) {
    use std::collections::HashMap;
    use std::env;
    let Ok(dir) = env::var("IRONMLX_VISION_DUMP_DIR") else {
        return;
    };
    if let Err(e) = mlx::transforms::eval(&[t]) {
        eprintln!("[vision-dump] eval {name} failed: {e}");
        return;
    }
    let path = format!("{dir}/{name}.safetensors");
    let mut tensors = HashMap::new();
    tensors.insert("tensor".to_string(), t.clone());
    let metadata: HashMap<String, String> = HashMap::new();
    if let Err(e) = mlx::io::save_safetensors(&path, &tensors, &metadata) {
        eprintln!("[vision-dump] save {name} failed: {e}");
    }
}

#[cfg(not(feature = "vision-dump"))]
#[inline(always)]
pub fn dump_tensor(_: &str, _: &Array) {}
