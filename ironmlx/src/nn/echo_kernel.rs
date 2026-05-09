//! P8a stage 9 task 1 — fusion barrier pre-verification.
//!
//! Inserts an "echo" Metal kernel (input -> output, no compute) into the
//! forward path when `IRONMLX_ECHO_KERNEL=1` is set. Used to measure
//! whether `mx::fast::metal_kernel` injection itself introduces fusion
//! barrier overhead, before committing to the real self_qmm kernel.

use std::sync::OnceLock;

use mlx::{Array, MetalKernel};

use crate::Result;

/// Returns true iff `IRONMLX_ECHO_KERNEL=1` env var is set.
pub fn echo_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("IRONMLX_ECHO_KERNEL").as_deref() == Ok("1"))
}

/// Lazy MetalKernel — built once per process.
fn echo_kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(k) = KERNEL.get() {
        return Ok(k);
    }
    let src = r#"
        uint idx = thread_position_in_grid.x;
        if (idx >= (uint)total) { return; }
        out[idx] = x[idx];
    "#;
    let kernel = MetalKernel::builder("ironmlx_echo")
        .inputs(&["x"])
        .outputs(&["out"])
        .source(src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(KERNEL.get_or_init(|| kernel))
}

/// Pass `x` through an echo Metal kernel. Output is byte-identical to input.
pub fn echo(x: &Array) -> Result<Array> {
    let total = x.size() as i32;
    let kernel = echo_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x])
        .output_shapes(&[x.shape().clone()])
        .output_dtypes(&[x.dtype()])
        .grid(total, 1, 1)
        .threadgroup(256, 1, 1)
        .template_int("total", total)
        .dispatch()?;
    Ok(outputs.take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Dtype;

    #[test]
    fn echo_preserves_input() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x: Array = (data.as_slice(), (4_i32, 4)).try_into().unwrap();
        let y = echo(&x).unwrap();
        let yv: Vec<f32> = y.to_vec().unwrap();
        for (i, (a, b)) in data.iter().zip(yv.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
        assert_eq!(y.shape().as_slice(), x.shape().as_slice());
        assert_eq!(y.dtype(), Dtype::Float32);
    }
}
