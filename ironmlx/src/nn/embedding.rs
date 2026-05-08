//! `nn::Embedding` — single struct with private enum dispatch over fp and
//! quantized lookup tables.
//!
//! Construction goes through [`Embedding::from_loader`], which probes
//! `{prefix}.scales` to choose the variant. `forward` performs row gather
//! along axis 0 (`[batch, seq] u32 -> [batch, seq, dim]`); `as_output`
//! reuses the same weight as a tied output projection (`hidden @ Wᵀ ->
//! logits`), matching Qwen3.5's lm_head-tied configuration.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::Result;

/// Embedding lookup table. Handles both full-precision and quantized
/// weight checkpoints transparently, and doubles as a tied output
/// projection via [`Embedding::as_output`].
pub struct Embedding {
    inner: EmbeddingImpl,
}

/// Internal backend variant. Private — callers use [`Embedding`].
enum EmbeddingImpl {
    Fp {
        /// `[vocab, dim]` dense weight, dtype as stored in the checkpoint.
        weight: Array,
    },
    Quant {
        /// Packed quantized weight (layout per `mlx::quantization`).
        weight: Array,
        /// Per-group scales.
        scales: Array,
        /// Per-group zero-points (affine quantization).
        biases: Option<Array>,
        /// Group size from quantization metadata.
        group_size: i32,
        /// Bits per quantized weight (4 / 6 / 8).
        bits: i32,
    },
}

impl Embedding {
    /// Build an `Embedding` from `loader`, looking for tensors at
    /// `{prefix}.weight` (required), `{prefix}.scales` (signals quantized
    /// variant), and `{prefix}.biases` (optional zero-points for affine
    /// quant).
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let scales_key = format!("{prefix}.scales");
        let biases_key = format!("{prefix}.biases");

        let weight = loader.tensor(&weight_key)?.clone();

        if loader.contains(&scales_key) {
            let qmeta = loader.quant_meta().ok_or_else(|| {
                anyhow!(
                    "Embedding `{prefix}`: `{scales_key}` present but Loader has no quantization meta"
                )
            })?;
            let scales = loader.tensor(&scales_key)?.clone();
            let biases = loader.tensor_opt(&biases_key).cloned();
            Ok(Embedding {
                inner: EmbeddingImpl::Quant {
                    weight,
                    scales,
                    biases,
                    group_size: qmeta.group_size,
                    bits: qmeta.bits,
                },
            })
        } else {
            Ok(Embedding {
                inner: EmbeddingImpl::Fp { weight },
            })
        }
    }

    /// Lookup: `tokens` (`u32`, any shape) → embeddings with `dim` appended
    /// as the last axis.
    pub fn forward(&self, tokens: &Array) -> Result<Array> {
        self.forward_on(tokens, ())
    }

    /// Stream-targeted variant of [`Embedding::forward`].
    pub fn forward_on(&self, tokens: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        match &self.inner {
            EmbeddingImpl::Fp { weight } => Ok(weight.take_on(tokens, 0, target)?),
            EmbeddingImpl::Quant {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // Per spec § 3.3: dequantize the full table then gather
                // rows. This defeats some storage savings; a fused
                // row-lookup kernel is a follow-up.
                let dequant = mlx::quantization::dequantize_on(
                    weight,
                    scales,
                    biases.as_ref(),
                    Some(*group_size),
                    Some(*bits),
                    "affine",
                    None,
                    None,
                    target,
                )?;
                Ok(dequant.take_on(tokens, 0, target)?)
            }
        }
    }

    /// Tied-embedding output: project `hidden` (`[..., dim]`) to logits
    /// (`[..., vocab]`). Equivalent to a `Linear` with `weight = embed.weight`
    /// and no bias.
    pub fn as_output(&self, hidden: &Array) -> Result<Array> {
        self.as_output_on(hidden, ())
    }

    /// Test seam — builds a fp Embedding directly from a weight Array.
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it
    /// — those tests are compiled as external crates. Hidden from rustdoc via
    /// `#[doc(hidden)]`.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn from_components_fp_for_test(weight: Array) -> Self {
        Self {
            inner: EmbeddingImpl::Fp { weight },
        }
    }

    /// Stream-targeted variant of [`Embedding::as_output`].
    pub fn as_output_on(&self, hidden: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        match &self.inner {
            EmbeddingImpl::Fp { weight } => {
                // weight: [vocab, dim]; want hidden @ Wᵀ -> [..., vocab].
                let w_t = weight.transpose_on(target)?;
                Ok(hidden.matmul_on(&w_t, target)?)
            }
            EmbeddingImpl::Quant {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => Ok(mlx::quantization::quantized_matmul_on(
                hidden,
                weight,
                scales,
                biases.as_ref(),
                /* transpose = */ true,
                Some(*group_size),
                Some(*bits),
                "affine",
                target,
            )?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::Array;

    fn fp_embedding(weight: Array) -> Embedding {
        Embedding {
            inner: EmbeddingImpl::Fp { weight },
        }
    }

    #[test]
    fn fp_forward_lookup() {
        // 4-row vocab, 3-dim embeddings: rows are [1,2,3], [4,5,6],
        // [7,8,9], [10,11,12].
        let w = Array::try_from((
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ][..],
            &[4, 3][..],
        ))
        .unwrap();
        let layer = fp_embedding(w);

        // tokens [2, 0] -> rows 2 and 0.
        let tokens = Array::try_from((&[2u32, 0][..], &[2][..])).unwrap();
        let y = layer.forward(&tokens).expect("forward");

        assert_eq!(y.shape().as_slice(), &[2, 3]);
        assert_eq!(
            y.to_vec::<f32>().expect("to_vec"),
            vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn as_output_tied_projection() {
        // weight [vocab=3, dim=2] = [[1,0],[0,1],[1,1]]
        // hidden [batch=1, seq=1, dim=2] = [2, 3]
        // logits = hidden @ Wᵀ where Wᵀ = [[1,0,1],[0,1,1]]
        //   col 0 -> 2*1 + 3*0 = 2
        //   col 1 -> 2*0 + 3*1 = 3
        //   col 2 -> 2*1 + 3*1 = 5
        let w = Array::try_from((&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0][..], &[3, 2][..])).unwrap();
        let layer = fp_embedding(w);

        let hidden = Array::try_from((&[2.0f32, 3.0][..], &[1, 1, 2][..])).unwrap();
        let logits = layer.as_output(&hidden).expect("as_output");

        assert_eq!(logits.shape().as_slice(), &[1, 1, 3]);
        assert_eq!(logits.to_vec::<f32>().expect("to_vec"), vec![2.0, 3.0, 5.0]);
    }
}
