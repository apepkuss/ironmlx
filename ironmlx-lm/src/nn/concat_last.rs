//! Concatenation of equal-shape arrays along the last axis in one dispatch.
//!
//! MLX concatenates by one strided copy per input into the shared output,
//! serialized on that output; the DFlash2 context taps (8 target layers of
//! `[1, S, H]`) pay that per window. A plain copy, so the result is
//! identical.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::ensure;
use mlx::{Array, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

fn kernel(count: usize) -> Result<MetalKernel> {
    static CACHE: OnceLock<Mutex<HashMap<usize, MetalKernel>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(kernel) = cache.get(&count) {
        return Ok(kernel.clone());
    }
    let names: Vec<String> = (0..count).map(|i| format!("x{i}")).collect();
    let mut select = String::new();
    for i in 0..count {
        select.push_str(&format!(
            "  {}if (part == {i}) out[gid] = x{i}[row * W + col];\n",
            if i == 0 { "" } else { "else " }
        ));
    }
    let source = format!(
        "  const uint gid = thread_position_in_grid.x;\n  const uint wide = W * {count};\n  const uint row = gid / wide;\n  \
         const uint rem = gid % wide;\n  const uint part = rem / W;\n  const uint col = rem % W;\n{select}"
    );
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let kernel = MetalKernel::builder(format!("ironmlx_concat_last_v1_{count}"))
        .inputs(&refs)
        .outputs(&["out"])
        .source(source)
        .build()?;
    cache.insert(count, kernel.clone());
    Ok(kernel)
}

/// `concatenate(arrays, -1)` for arrays of one shape and dtype.
pub fn concat_last_on(arrays: &[&Array], target: StreamOrDevice) -> Result<Array> {
    ensure!(!arrays.is_empty(), "concat_last: no inputs");
    let shape = arrays[0].shape();
    let dims = shape.as_slice().to_vec();
    let dtype = arrays[0].dtype();
    ensure!(
        !dims.is_empty()
            && arrays
                .iter()
                .all(|a| a.shape().as_slice() == dims.as_slice() && a.dtype() == dtype),
        "concat_last: inputs must share shape and dtype"
    );
    let w = *dims.last().unwrap();
    let rows: i32 = dims[..dims.len() - 1].iter().product();
    let count = i32::try_from(arrays.len())?;
    let total = rows * w * count;
    let flat: Vec<Array> = arrays
        .iter()
        .map(|a| a.reshape_on(&[rows * w][..], target))
        .collect::<std::result::Result<_, _>>()?;
    let inputs: Vec<&Array> = flat.iter().collect();
    let out = kernel(arrays.len())?
        .dispatch_builder()
        .inputs(&inputs)
        .output_shapes(&[Shape::from(&[total][..])])
        .output_dtypes(&[dtype])
        .grid(total, 1, 1)
        .threadgroup(256, 1, 1)
        .template_int("W", w)
        .stream(target)
        .dispatch()?
        .take_at(0)?;
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = w * count;
    Ok(out.reshape_on(out_dims.as_slice(), target)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn matches_mlx_concatenate() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let key = mlx::random::key(3)?;
        for rows in [1, 3, 8] {
            let parts = (0..8)
                .map(|i| {
                    let x = mlx::random::normal()
                        .shape(&[1, rows, 64][..])
                        .key(&mlx::random::split(&key)?.0)
                        .sample()?;
                    Ok(mlx::ops::cast::astype(
                        &(&x + (i as f32)),
                        mlx::Dtype::Bfloat16,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Array> = parts.iter().collect();
            let want = mlx::ops::shape::concatenate(&refs, -1)?;
            let got = concat_last_on(&refs, StreamOrDevice::default())?;
            assert_eq!(got.shape().as_slice(), want.shape().as_slice());
            let want = mlx::ops::cast::astype(&want, mlx::Dtype::Float32)?.to_vec::<f32>()?;
            let got = mlx::ops::cast::astype(&got, mlx::Dtype::Float32)?.to_vec::<f32>()?;
            assert_eq!(got, want, "rows {rows}");
        }
        Ok(())
    }
}
