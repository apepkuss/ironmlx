//! Compiled activation helpers shared by dense and routed MLP blocks.

use anyhow::anyhow;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::{ops, Array, StreamOrDevice};

use crate::Result;

// sqrt(2/π) = 0.7978845608028654  (tanh GELU approximation constant)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// GELU with tanh approximation (matches PyTorch `approximate="tanh"` / mlx-vlm `gelu_approx`).
///
/// Formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
///
/// There is no built-in `gelu_approx` in the mlx Rust bindings, so this is
/// hand-rolled from the exact polynomial formula used by mlx-vlm / PyTorch.
///
/// # Precision note
///
/// MLX Python uses `@partial(mx.compile, shapeless=True)` on `gelu_approx`, and
/// computes `x**3` via `mx.power(x, 3)`.  When the input is BF16, Python scalar
/// literals (`0.044715`, `0.5`, …) do NOT promote the result to F32 — they stay
/// in the input dtype.  In contrast, Rust `f32` scalar literals in MLX DO cause
/// BF16→F32 promotion.
///
/// To match Python exactly:
/// 1. Compute `x^3` with `x.power(&three_i32)` which matches Python's `x**3`.
/// 2. Use dtype-matched constants for the polynomial scalars so that all
///    intermediate values stay in the input dtype (BF16).  The final output
///    remains BF16, matching `gelu_approx`'s output.
pub(crate) fn gelu_tanh(x: &Array, target: StreamOrDevice) -> Result<Array> {
    // x^3 via mlx::power(x, 3) — matches Python's `x**3` exactly.
    // Using x * x * x introduces an extra BF16 rounding step that differs from
    // mx.power(x, 3) by up to 0.125 in BF16, causing downstream GELU divergence.
    let three: Array = (&[3_i32][..], ()).try_into()?;
    let x3 = x.power(&three)?;

    // Keep scalars in the input dtype to avoid BF16→F32 promotion.
    // Python float literals don't promote; Rust f32 scalars do — so we explicitly
    // create BF16 (or the input dtype) constants.
    let dtype = x.dtype();
    let c_044715: Array = ops::cast::astype(&(&[0.044_715_f32][..], ()).try_into()?, dtype)?;
    let c_sqrt2pi: Array = ops::cast::astype(&(&[SQRT_2_OVER_PI][..], ()).try_into()?, dtype)?;
    let c_half: Array = ops::cast::astype(&(&[0.5_f32][..], ()).try_into()?, dtype)?;
    let c_one: Array = ops::cast::astype(&(&[1.0_f32][..], ()).try_into()?, dtype)?;

    // inner = sqrt(2/π) * (x + 0.044715 * x^3) — all in input dtype
    let inner = (&(&x3 * &c_044715) + x) * &c_sqrt2pi;
    // tanh(inner)
    let t = inner.tanh_on(target)?;
    // 0.5 * x * (1 + t)
    let out = x * &c_half * (&t + &c_one);
    Ok(out)
}

pub(crate) fn build_swiglu() -> CompiledFn {
    compile(
        |inputs: &[&Array]| -> mlx::Result<Vec<Array>> {
            let gate = inputs[0];
            let up = inputs[1];
            let gate_sig = gate.sigmoid()?;
            let gate_silu = gate * &gate_sig;
            let out = &gate_silu * up;
            Ok(vec![out])
        },
        ShapeMode::Shapeless,
    )
    .expect("SwiGLU compile")
}

pub(crate) fn invoke_swiglu(func: &CompiledFn, gate: &Array, up: &Array) -> Result<Array> {
    let mut outs = func
        .invoke(&[gate, up])
        .map_err(|e| anyhow!("SwiGLU invoke failed: {e}"))?;
    outs.pop()
        .ok_or_else(|| anyhow!("SwiGLU returned no outputs"))
}
