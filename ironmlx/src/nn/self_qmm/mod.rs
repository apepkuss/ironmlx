//! Self-implemented quantized matmul Metal kernel for MLX 4-bit affine
//! quantization (group_size=64). Opt-in via `IRONMLX_USE_SELF_QMM=1`.
//!
//! Stage 9: prefill (qmm_t) only, M1 Pro tuned. See
//! `docs/superpowers/specs/2026-05-09-p8a-stage9-quant-kernel-design.md`.

mod kernel;
mod lookup;

use std::sync::OnceLock;

use mlx::{Array, StreamOrDevice};

use crate::Result;

/// Returns true iff `IRONMLX_USE_SELF_QMM=1` env var is set.
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("IRONMLX_USE_SELF_QMM").as_deref() == Ok("1"))
}

/// Cached Metal device architecture string (e.g. `"apple_g13s"`).
///
/// Queried once via `mlx::metal::architecture()` and reused across
/// dispatches — the underlying MLX C++ call hits a `static` map but the
/// FFI roundtrip + string allocation still cost more than a hashed
/// `OnceLock` read, and the architecture never changes mid-process.
///
/// Stored as `std::result::Result<String, String>` (deliberately fully
/// qualified to avoid confusion with `crate::Result` / `anyhow::Result`
/// imported above) so a failed first query (e.g. Metal backend
/// unavailable in a CI build) is sticky and surfaces consistently — we
/// don't want one failure path then a second silent success on the
/// next dispatch.
fn device_arch() -> &'static std::result::Result<String, String> {
    static ARCH: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    ARCH.get_or_init(|| mlx::metal::architecture().map_err(|e| e.to_string()))
}

/// 4-bit MLX-affine quantized matmul: `x @ w^T` with per-group scales+biases.
///
/// Inputs:
/// - `x`: input tensor `[..., K]` (last dim contiguous; bf16 / fp16 / fp32)
/// - `w`: packed uint32 `[N, K/8]` (8 4-bit weights per uint32)
/// - `scales`: per-group scales `[N, K/group_size]` (same dtype as `x`)
/// - `biases`: per-group biases `[N, K/group_size]` (same dtype as `x`)
/// - `bits`: must be 4 (stage 9 single-precision)
/// - `group_size`: must be 64 (stage 9 single-group)
///
/// Output: `[..., N]` with `x.dtype()`.
///
/// Stream argument is currently consumed and ignored — the kernel runs on the
/// MLX default stream tied to the calling thread. Stage 10+ will wire it into
/// `DispatchBuilder::stream(...)` once multi-stream scheduling is exercised.
pub fn qmm_t_on(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    bits: i32,
    group_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    assert_eq!(bits, 4, "self_qmm stage 9 only supports bits=4");
    assert_eq!(
        group_size, 64,
        "self_qmm stage 9 only supports group_size=64"
    );
    let _ = target.into();

    // Compute (M, N, K). x is `[..., K]` so M = product of leading dims;
    // w is `[N, K/8]` (packed uint32, 8 nibbles per word).
    let x_dims = x.shape();
    let x_dims_slice = x_dims.as_slice();
    let k = *x_dims_slice
        .last()
        .expect("self_qmm: x must be at least 1-D");
    let m: i32 = x_dims_slice[..x_dims_slice.len() - 1].iter().product();
    let n = w.shape().as_slice()[0];

    // Pick (BM, BN, BK) from the per-arch lookup table. If the Metal
    // architecture query failed at first call, treat that as "unknown
    // device" — the lookup returns the safe default tile and warns once.
    let arch_str = match device_arch() {
        Ok(s) => s.as_str(),
        Err(_) => "",
    };
    let tile = lookup::lookup_tile(arch_str, m, n, k, bits, group_size);
    kernel::dispatch_qmm_t(x, w, scales, biases, tile.bm, tile.bn, tile.bk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{ops, Dtype};

    /// Run the self_qmm vs MLX reference comparison for a specific tile
    /// `(BM, BN, BK)`. Decision gate: max abs diff < 0.5 PASS.
    ///
    /// Calls `kernel::dispatch_qmm_t` directly (not `qmm_t_on` which is
    /// hardcoded to (64,64,32) for now). Shape is small (M=4, K=64, N=8) —
    /// this is deliberate: it exercises bound-check paths in the kernel
    /// (partial BM/BN tile coverage) without requiring large allocations.
    fn run_variant(bm: i32, bn: i32, bk: i32) {
        let m = 4_i32;
        let k = 64_i32;
        let n = 8_i32;
        let group_size = 64_i32;
        let bits = 4_i32;

        // Deterministic raw weight data (avoid random-seed test flakiness).
        // Small range (~[-0.3, 0.21]) keeps the matmul output values <~ 1.0,
        // well within bf16 precision (~7 mantissa bits ≈ 0.008 ulp at 1.0).
        let raw_data: Vec<f32> = (0..(n * k)).map(|i| (i as f32) * 0.001 - 0.3).collect();
        let raw_w_f32: Array = (raw_data.as_slice(), (n, k)).try_into().unwrap();
        let raw_w_bf16 = ops::cast::astype(&raw_w_f32, Dtype::Bfloat16).unwrap();

        let q_outs =
            mlx::quantization::quantize(&raw_w_bf16, Some(group_size), Some(bits), "affine", None)
                .expect("quantize");
        let w_packed = &q_outs[0];
        let w_scales = &q_outs[1];
        let w_biases = &q_outs[2];

        let x_data: Vec<f32> = (0..(m * k) as usize).map(|i| (i as f32) * 0.001).collect();
        let x_f32: Array = (x_data.as_slice(), (m, k)).try_into().unwrap();
        let x = ops::cast::astype(&x_f32, Dtype::Bfloat16).unwrap();

        let y_self = kernel::dispatch_qmm_t(&x, w_packed, w_scales, w_biases, bm, bn, bk)
            .expect("self_qmm dispatch");

        let y_mlx = mlx::quantization::quantized_matmul_on(
            &x,
            w_packed,
            w_scales,
            Some(w_biases),
            /* transpose = */ true,
            Some(group_size),
            Some(bits),
            "affine",
            (),
        )
        .expect("mlx quantized_matmul_on");

        assert_eq!(
            y_self.shape().as_slice(),
            y_mlx.shape().as_slice(),
            "tile (BM={bm}, BN={bn}, BK={bk}): shape mismatch self={:?} mlx={:?}",
            y_self.shape().as_slice(),
            y_mlx.shape().as_slice()
        );

        let y_self_f32 = ops::cast::astype(&y_self, Dtype::Float32).unwrap();
        let y_mlx_f32 = ops::cast::astype(&y_mlx, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_self_f32.to_vec().unwrap();
        let mv: Vec<f32> = y_mlx_f32.to_vec().unwrap();
        assert_eq!(yv.len(), mv.len());

        let max_diff = yv
            .iter()
            .zip(mv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("tile (BM={bm}, BN={bn}, BK={bk}) max abs diff = {max_diff}");
        assert!(
            max_diff < 0.5,
            "tile (BM={bm}, BN={bn}, BK={bk}): max abs diff {max_diff} >= 0.5"
        );
    }

    #[test]
    fn self_qmm_t_tile_64_64_32() {
        run_variant(64, 64, 32);
    }

    #[test]
    fn self_qmm_t_tile_64_128_32() {
        run_variant(64, 128, 32);
    }

    #[test]
    fn self_qmm_t_tile_128_128_32() {
        run_variant(128, 128, 32);
    }
}
