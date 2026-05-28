//! Compiled activation helpers shared by dense and routed MLP blocks.

use anyhow::anyhow;
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::Array;

use crate::Result;

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
