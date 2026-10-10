//! Row-stable affine4 projection for 1..=8 activation rows.
//!
//! One fixed arithmetic for every row count, so a row's output never depends
//! on how many rows share the dispatch: decoding one token and verifying a
//! window of tokens produce identical bits per row. With `G` quantization
//! groups of 64 split into `KS` interleaved slices (`g % KS == j`):
//!
//! ```text
//! c_j[a] = 0 for a in 0..4
//! for g in slice j (ascending), s in 0..8, kk in 0..8:
//!     k   = 64 g + 8 kk + s
//!     w   = fma(scale[n][g] * 2^-4s, float(q_word[n][k] & (0xF << 4s)), bias[n][g])
//!     c_j[s % 4] = fma(w, x[m][k], c_j[s % 4])
//! c_j = ((c_j[0] + c_j[1]) + c_j[2]) + c_j[3]
//! y[m][n] = bf16(((c_0 + c_1) + c_2) + ...)
//! ```
//!
//! The `2^-4s` factor only rescales the masked (unshifted) nibble, so `w` is
//! exactly the dequantized weight `scale * q + bias`. Three kernels evaluate it: a scalar chain for one
//! row, the same chain for two or three rows with the input rows staged once
//! per threadgroup, and an 8x8x8 simdgroup MMA kernel for two to eight rows
//! (rows are padded to one 8-row tile). An Apple GPU MMA computes each
//! element as the sequential `fma` chain over its eight `k` terms, which is
//! the scalar order above; the unit test checks that all kernels agree bit
//! for bit.
//!
//! The qualified Qwen3.6 MoE affine4 target arms the route only for the
//! drafter's logit projections (through the target LM head): its arithmetic
//! differs from MLX's, which only changes draft proposals. The target's own
//! logits keep MLX's arithmetic.

use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

use super::linear::QuantizedLinearParts;
use crate::core::QuantMode;
use crate::Result;

/// Most activation rows this route takes (one MMA tile).
pub(crate) const MAX_ROWS: i32 = 8;

thread_local! {
    static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Route every eligible affine4 projection of 1..=8 rows in this scope
/// through the row-stable arithmetic.
pub(crate) fn scope() -> Scope {
    DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    Scope
}

pub(crate) fn is_armed() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

/// K slices of the arithmetic. Part of the arithmetic, the same for every
/// row count.
const K_SLICES: i32 = 8;
/// Simdgroups per one-row threadgroup. Not part of the arithmetic.
const ROW_SGS: i32 = 2;
/// Most rows the staged-row kernel evaluates; wider inputs use the MMA
/// kernel. Not part of the arithmetic.
const STAGED_MAX_ROWS: i32 = 3;
/// Output rows per lane and simdgroups per threadgroup of the staged-row
/// kernel. Not part of the arithmetic.
const STAGED_R: i32 = 2;
const STAGED_SGS: i32 = 8;

/// Output rows each one-row lane evaluates (they share one load of every
/// input group). Not part of the arithmetic.
fn row_r(n: i32, groups_per_lane: i32) -> i32 {
    let mut r = (4 / groups_per_lane).clamp(1, 4);
    while r > 1 && n % (ROW_SGS * (32 / K_SLICES) * r) != 0 {
        r /= 2;
    }
    r
}

/// Weight-side shapes this route evaluates.
pub(crate) fn weight_supported(p: &QuantizedLinearParts<'_>) -> bool {
    let w = p.weight.shape();
    let w = w.as_slice();
    p.mode == QuantMode::Affine
        && p.bits == 4
        && p.group_size == 64
        && p.weight.dtype() == Dtype::Uint32
        && p.scales.dtype() == Dtype::Bfloat16
        && p.biases.is_some_and(|b| b.dtype() == Dtype::Bfloat16)
        && w.len() == 2
        && w[0] > 0
        && w[0] % 32 == 0
        && w[1] > 0
        && (w[1] * 8) % 512 == 0
}

/// Rows of `x` (all leading dimensions) when this route takes it.
pub(crate) fn route_rows(x: &Array, p: &QuantizedLinearParts<'_>) -> Option<i32> {
    if x.dtype() != Dtype::Bfloat16 || x.ndim() < 2 || !weight_supported(p) {
        return None;
    }
    let k = *x.shape().as_slice().last()?;
    if k <= 0 || k != p.weight.shape().as_slice()[1] * 8 {
        return None;
    }
    let rows = i32::try_from(x.size() / k as usize).ok()?;
    (1..=MAX_ROWS).contains(&rows).then_some(rows)
}

/// `x`: `[..., K]` bf16 with 1..=8 rows; returns `[..., N]`.
pub(crate) fn forward_on(
    x: &Array,
    p: QuantizedLinearParts<'_>,
    rows: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    let n = p.weight.shape().as_slice()[0];
    let k = p.weight.shape().as_slice()[1] * 8;
    let biases = p
        .biases
        .ok_or_else(|| anyhow!("row-stable affine4 needs biases"))?;
    let x2 = x.reshape_on((rows, k), target)?;
    let mut out_shape = x.shape().as_slice().to_vec();
    *out_shape.last_mut().unwrap() = n;
    let (kind, r, sgs) = if rows == 1 {
        (0, row_r(n, (k / 64) / K_SLICES), ROW_SGS)
    } else if rows <= STAGED_MAX_ROWS
        && rows * k * 2 <= 16384
        && n % (STAGED_SGS * (32 / K_SLICES) * STAGED_R) == 0
    {
        (5, STAGED_R, STAGED_SGS)
    } else {
        (2, 1, 1)
    };
    let y = mlx::quantization::row_stable_affine4_matmul_on(
        &x2, p.weight, p.scales, biases, kind, K_SLICES, r, sgs, target,
    )?;
    let mut y = y.reshape_on(&out_shape[..], target)?;
    if let Some(bias) = p.bias {
        y = mlx::ops::binary::add_on(&y, bias, target)?;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn bits(a: &Array) -> Vec<u32> {
        mlx::ops::cast::astype(a, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn parts<'a>(w: &'a Array, s: &'a Array) -> QuantizedLinearParts<'a> {
        QuantizedLinearParts {
            weight: w,
            scales: s,
            biases: Some(s),
            bias: None,
            group_size: 64,
            bits: 4,
            mode: QuantMode::Affine,
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn route_declines_zero_width_and_malformed_weights() {
        let zeros = |shape: &[i32], dtype| Array::zeros(shape, dtype).unwrap();
        let s = zeros(&[32, 8], Dtype::Bfloat16);
        let w0 = zeros(&[32, 0], Dtype::Uint32);
        let x0 = zeros(&[1, 0], Dtype::Bfloat16);
        assert_eq!(route_rows(&x0, &parts(&w0, &s)), None);
        let w1 = zeros(&[64], Dtype::Uint32);
        let x = zeros(&[1, 512], Dtype::Bfloat16);
        assert_eq!(route_rows(&x, &parts(&w1, &s)), None);
        let w = zeros(&[32, 64], Dtype::Uint32);
        assert_eq!(route_rows(&x, &parts(&w, &s)), Some(1));
    }

    #[test]
    #[serial(mlx_metal)]
    fn row_and_mma_kernels_agree_bitwise_at_every_row_count() {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
        }
        let key = mlx::random::key(21).unwrap();
        for (n, k) in [(64, 2048), (256, 512), (512, 4096), (96, 1024)] {
            let w = mlx::random::normal()
                .shape(&[n, k][..])
                .dtype(Dtype::Float32)
                .scale(0.05)
                .key(&key)
                .sample()
                .unwrap();
            let w = mlx::ops::cast::astype(&w, Dtype::Bfloat16).unwrap();
            let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None).unwrap();
            let parts = QuantizedLinearParts {
                weight: &q[0],
                scales: &q[1],
                biases: Some(&q[2]),
                bias: None,
                group_size: 64,
                bits: 4,
                mode: QuantMode::Affine,
            };
            let x = mlx::random::normal()
                .shape(&[1, 8, k][..])
                .dtype(Dtype::Float32)
                .scale(3.0)
                .key(&key)
                .sample()
                .unwrap();
            let x = mlx::ops::cast::astype(&x, Dtype::Bfloat16).unwrap();
            let full = bits(&forward_on(&x, parts, 8, StreamOrDevice::default()).unwrap());
            for m in 1..=8 {
                let xm = mlx::ops::indexing::slice(&x, &[0, 0, 0][..], &[1, m, k][..]).unwrap();
                let part = bits(&forward_on(&xm, parts, m, StreamOrDevice::default()).unwrap());
                assert_eq!(part[..], full[..(m * n) as usize], "N={n} K={k} rows={m}");
            }
            // Each row alone (the scalar kernel) equals its row in the 8-row tile.
            for r in 0..8 {
                let xr = mlx::ops::indexing::slice(&x, &[0, r, 0][..], &[1, r + 1, k][..]).unwrap();
                let one = bits(&forward_on(&xr, parts, 1, StreamOrDevice::default()).unwrap());
                assert_eq!(
                    one[..],
                    full[(r * n) as usize..((r + 1) * n) as usize],
                    "N={n} K={k} row {r}"
                );
            }
        }
    }
}
