//! `nn::Linear` — single struct with private enum dispatch over fp and
//! quantized backends.
//!
//! Construction goes through [`Linear::from_loader`], which probes
//! `{prefix}.scales` to choose the variant. Forward computes
//! `y = x @ W^T + bias` for fp weights, or calls
//! [`mlx::quantization::quantized_matmul`] (with `transpose=true`) for
//! quantized weights, then optionally adds the (non-quantized) `bias`.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Linear projection layer. Handles both full-precision and quantized
/// weight checkpoints transparently.
pub struct Linear {
    inner: LinearImpl,
}

/// Internal backend variant. Private — callers use [`Linear`].
enum LinearImpl {
    Fp {
        /// `[out, in]` dense weight, dtype as stored in the checkpoint.
        weight: Array,
        /// Optional `[out]` bias.
        bias: Option<Array>,
    },
    Quant {
        /// Packed quantized weight (layout per `mlx::quantization`).
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group zero-points (affine quantization).
        biases: Option<Array>,
        /// Optional Linear bias term, applied after the quantized matmul.
        bias: Option<Array>,
        /// Group size from quantization metadata.
        group_size: i32,
        /// Bits per quantized weight (4 / 6 / 8).
        bits: i32,
    },
}

impl Linear {
    /// Build a `Linear` from `loader`, looking for tensors at
    /// `{prefix}.weight` (required), `{prefix}.bias` (optional),
    /// `{prefix}.scales` (signals quantized variant), and
    /// `{prefix}.biases` (optional zero-points for affine quant).
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let bias_key = format!("{prefix}.bias");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");

        let weight = loader.tensor(&weight_key)?.clone();
        let bias = loader.tensor_opt(&bias_key).cloned();

        if loader.contains(&scales_key) {
            let qmeta = loader.quant_meta_for(prefix).ok_or_else(|| {
                anyhow!(
                    "Linear `{prefix}`: `{scales_key}` present but Loader has no quantization meta"
                )
            })?;
            let scales = loader.tensor(&scales_key)?.clone();
            let biases = loader.tensor_opt(&biases_key).cloned();
            Ok(Linear {
                inner: LinearImpl::Quant {
                    weight,
                    scales,
                    biases,
                    bias,
                    group_size: qmeta.group_size,
                    bits: qmeta.bits,
                },
            })
        } else {
            Ok(Linear {
                inner: LinearImpl::Fp { weight, bias },
            })
        }
    }

    /// Test/composition seam: build an FP `Linear` from in-memory weight (and optional bias).
    /// Production code should use [`Linear::from_loader`]. This bypass lets `nn` building
    /// blocks be composed without writing a safetensors file (used by `GatedAttention`'s
    /// `from_components` constructor, unit tests, and integration tests).
    ///
    /// `weight` must be shape `[out, in]`; `bias` must be `[out]` if `Some`.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new_fp(weight: Array, bias: Option<Array>) -> Self {
        Self {
            inner: LinearImpl::Fp { weight, bias },
        }
    }

    /// Compose a quantized [`Linear`] from already-loaded Arrays. Used by
    /// callers that fuse multiple weight tensors at load time (e.g.
    /// [`GatedDeltaNet`](crate::nn::GatedDeltaNet)'s concatenated input
    /// projections). Production code that loads a single weight from a
    /// safetensors checkpoint should use [`Linear::from_loader`].
    ///
    /// `weight` is the packed quantized weight matrix; `scales` is per-group
    /// scales; `biases` is per-group zero-points (Some for affine
    /// quantization, None for symmetric); `bias` is the additive linear bias
    /// term separate from `biases` (typically None for Qwen3.5).
    /// `group_size` and `bits` are the quantization metadata (typically
    /// 64 / 4 for Qwen3.5 4-bit checkpoints).
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can
    /// use it. Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn new_quant(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        bias: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self {
            inner: LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            },
        }
    }

    /// Forward pass: `y = x @ W^T (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Number of input features (the trailing axis of the input the layer accepts).
    ///
    /// For fp weights stored as `[out, in]`, returns `weight.shape()[1]`.
    /// For quantized weights packed at `bits` bits per element into `u32`
    /// (32-bit) lanes, each stored column covers `32 / bits` logical input
    /// features, so `in_features = weight.shape()[1] * (32 / bits)`.
    pub fn in_features(&self) -> usize {
        match &self.inner {
            LinearImpl::Fp { weight, .. } => weight.shape().as_slice()[1] as usize,
            LinearImpl::Quant { weight, bits, .. } => {
                // Formula assumes power-of-2 bit width (2 / 4 / 8): each u32
                // lane packs 32/bits elements. mlx-community quants for
                // Qwen3.x are all 4-bit, so the assumption holds in practice;
                // the assert prevents silent mis-computation if a future
                // checkpoint uses non-power-of-2 bits (3 / 5 / 6, byte-packed).
                debug_assert!(
                    *bits > 0 && *bits <= 32 && (*bits as u32).is_power_of_two(),
                    "Linear::in_features: 32/bits packing assumes power-of-2 bits in {{2,4,8,16,32}}, got bits={bits}"
                );
                (weight.shape().as_slice()[1] * (32 / bits)) as usize
            }
        }
    }

    /// Number of output features (the trailing axis of the output the layer produces).
    pub fn out_features(&self) -> usize {
        match &self.inner {
            LinearImpl::Fp { weight, .. } => weight.shape().as_slice()[0] as usize,
            LinearImpl::Quant { weight, .. } => weight.shape().as_slice()[0] as usize,
        }
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        match &self.inner {
            LinearImpl::Fp { weight, bias } => {
                let wt = weight.transpose_on(target)?;
                let mut y = x.matmul_on(&wt, target)?;
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
            LinearImpl::Quant {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
            } => {
                // M-aware dispatch: route through self_qmm only when the
                // batch×seq dim is large enough that the simdgroup-MMA tile
                // pays for itself. Below the threshold most threads are idle
                // (M=1 decode would burn the SG-MMA path at <1 tok/s), so
                // fall back to mlx's compute-light qmv.
                //
                // Threshold = 2048 matches the P8a Stage 9 self_qmm
                // acceptance gate test point (M=2048, N=9216, K=2560 gate_up
                // shape; 1.32× e2e vs MLX per [project-p8a-stage9-findings]),
                // which is the empirically validated lower bound for
                // self_qmm beneficial regime. P5i.c Phase B feasibility gate
                // (2026-05-27) empirically refuted the prior `>= 32` threshold
                // by measuring 3× substep regression + 1.75-7.52% e2e
                // regression at PP=128/512 (M=128/512) when
                // IRONMLX_USE_SELF_QMM=1; per-dispatch overhead (kernel
                // launch + threadgroup setup + smem staging + encoder cost)
                // dominates at small M.
                //
                // Inflection point between 512 and 2048 is unmeasured; future
                // work can do a bench-kernel m-sweep to find it and lower
                // threshold if benefit is meaningful. For now `>= 2048` is the
                // safe + empirically validated lower bound.
                //
                // Production default unaffected (IRONMLX_USE_SELF_QMM is
                // opt-in env var; default OFF → self_qmm path never taken).
                // This threshold change only affects users who explicitly
                // set IRONMLX_USE_SELF_QMM=1 at small-PP workloads.
                let m_total: i32 = {
                    let dims = x.shape();
                    let s = dims.as_slice();
                    if s.is_empty() {
                        0
                    } else {
                        s[..s.len() - 1].iter().product()
                    }
                };
                let use_self_qmm =
                    should_dispatch_self_qmm(crate::nn::self_qmm::enabled(), m_total);
                let mut y = if use_self_qmm {
                    // Stage 9 self-quant kernel path. qmm_t_on requires
                    // affine biases (per-group zero-points); Qwen3.5
                    // mlx-community 4-bit checkpoints always carry them.
                    // Panic explicitly if missing — silent fallback would
                    // hide a checkpoint mismatch that the caller needs to
                    // know about.
                    let qbiases = biases.as_ref().expect(
                        "self_qmm requires affine biases (per-group zero-points); \
                         checkpoint has none — IRONMLX_USE_SELF_QMM=1 only supports \
                         4-bit affine-quantized weights",
                    );
                    crate::nn::self_qmm::qmm_t_on(
                        x,
                        weight,
                        scales,
                        qbiases,
                        *bits,
                        *group_size,
                        target,
                    )?
                } else {
                    // Default path — stage 8 commit 811dd36 unchanged.
                    mlx::quantization::quantized_matmul_on(
                        x,
                        weight,
                        scales,
                        biases.as_ref(),
                        /* transpose = */ true,
                        Some(*group_size),
                        Some(*bits),
                        "affine",
                        target,
                    )?
                };
                if let Some(b) = bias {
                    y = &y + b;
                }
                Ok(y)
            }
        }
    }
}

/// Pure predicate for M-aware `self_qmm` dispatch — extracted so the
/// threshold logic is unit-testable without spinning up an MLX runtime.
///
/// Production callers pass `crate::nn::self_qmm::enabled()` as the
/// `self_qmm_enabled` argument; tests pass an explicit boolean.
///
/// Threshold is `2048` per the P8a Stage 9 acceptance gate test point
/// + P5i.c Phase B empirical refutation of the prior `32` threshold (see
///   `LinearImpl::Quant` branch comment above for full rationale).
#[inline]
fn should_dispatch_self_qmm(self_qmm_enabled: bool, m_total: i32) -> bool {
    self_qmm_enabled && m_total >= 2048
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use mlx::Array;

    fn fp_linear(weight: Array, bias: Option<Array>) -> Linear {
        Linear {
            inner: LinearImpl::Fp { weight, bias },
        }
    }

    #[test]
    fn fp_forward_matches_manual_matmul() {
        // weight [out=2, in=3] = [[1,2,3],[4,5,6]]
        // x [batch=1, in=3] = [1,1,1]
        // y = x @ W^T = [[1+2+3, 4+5+6]] = [[6, 15]]
        let weight =
            Array::try_from((&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..], &[2, 3][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 1.0, 1.0][..], &[1, 3][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_eq!(v, vec![6.0, 15.0]);
        assert_eq!(y.shape().as_slice(), &[1, 2]);
    }

    #[test]
    fn fp_forward_with_bias() {
        // weight = identity [[1,0],[0,1]], x = [3, 4], bias = [10, 20]
        // y = [3*1+4*0, 3*0+4*1] + [10, 20] = [13, 24]
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let bias = Array::try_from((&[10.0f32, 20.0][..], &[2][..])).unwrap();
        let x = Array::try_from((&[3.0f32, 4.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, Some(bias));

        let y = layer.forward(&x).expect("forward");
        let v = y.to_vec::<f32>().expect("to_vec");
        assert_abs_diff_eq!(v[0], 13.0, epsilon = 1e-6);
        assert_abs_diff_eq!(v[1], 24.0, epsilon = 1e-6);
    }

    #[test]
    fn fp_dtype_preserved() {
        let weight = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0][..], &[2, 2][..])).unwrap();
        let x = Array::try_from((&[1.0f32, 2.0][..], &[1, 2][..])).unwrap();
        let layer = fp_linear(weight, None);

        let y = layer.forward(&x).expect("forward");
        assert_eq!(x.dtype(), mlx::Dtype::Float32);
        assert_eq!(y.dtype(), mlx::Dtype::Float32);
    }

    #[test]
    fn new_quant_round_trips_via_from_loader_shape() {
        // We cannot construct a real quantized weight from thin air without a
        // tokenizer / safetensors fixture. Instead verify the structural
        // contract: new_quant accepts the 6 fields exactly and stores them in
        // LinearImpl::Quant. Cross-check by inspecting in_features /
        // out_features which compute from the stored shapes.

        // Build a fake quantized weight matching MLX's packed layout for
        // 4-bit, group_size=64: weight shape [out, in/8] u32, scales shape
        // [out, in/64] f32, biases (zero-points) shape [out, in/64] f32.
        let out = 32_i32;
        let in_dim = 64_i32; // single q-group along input axis
        let weight_packed_dim = in_dim / 8; // 4 bits per weight, 8 weights per u32
        let weight_data = vec![0u32; (out * weight_packed_dim) as usize];
        let scales_data = vec![0.01_f32; (out * 1) as usize]; // in/group_size=1
        let weight: Array = (weight_data.as_slice(), &[out, weight_packed_dim][..])
            .try_into()
            .unwrap();
        let scales: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();
        let biases: Array = (scales_data.as_slice(), &[out, 1_i32][..])
            .try_into()
            .unwrap();

        let lin = Linear::new_quant(weight, scales, Some(biases), None, 64, 4);

        assert_eq!(lin.in_features(), in_dim as usize);
        assert_eq!(lin.out_features(), out as usize);
    }

    /// Regression guard for the P8a Stage 9 + P5i.c Phase B threshold
    /// `m_total >= 2048`. Phase B feasibility gate (2026-05-27) empirically
    /// refuted the prior `m_total >= 32` threshold via 3× substep regression
    /// + 1.75-7.52% e2e regression at PP=128/512 when
    /// `IRONMLX_USE_SELF_QMM=1`. This test guards against accidental
    /// lowering of the threshold below the empirically validated lower
    /// bound (Stage 9 acceptance test point M=2048).
    #[test]
    fn self_qmm_dispatch_threshold_is_2048_when_env_enabled() {
        // Env disabled → never dispatch self_qmm, regardless of M.
        assert!(!should_dispatch_self_qmm(false, 1));
        assert!(!should_dispatch_self_qmm(false, 2048));
        assert!(!should_dispatch_self_qmm(false, 100_000));

        // Env enabled but M below threshold → fall back to MLX qmv path.
        assert!(!should_dispatch_self_qmm(true, 0));
        assert!(!should_dispatch_self_qmm(true, 1)); // M=1 decode
        assert!(!should_dispatch_self_qmm(true, 32)); // prior threshold; now OFF
        assert!(!should_dispatch_self_qmm(true, 128)); // PP=128 Phase B regression band
        assert!(!should_dispatch_self_qmm(true, 512)); // PP=512 Phase B regression band
        assert!(!should_dispatch_self_qmm(true, 2047)); // just below threshold

        // Env enabled AND M at-or-above threshold → dispatch self_qmm.
        assert!(should_dispatch_self_qmm(true, 2048)); // Stage 9 acceptance gate point
        assert!(should_dispatch_self_qmm(true, 4096));
        assert!(should_dispatch_self_qmm(true, 16_384));
    }
}
