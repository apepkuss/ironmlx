//! `nn::Embedding` — single struct with private enum dispatch over fp and
//! quantized lookup tables.
//!
//! Construction goes through [`Embedding::from_loader`], which probes
//! `{prefix}.scales` to choose the variant. `forward` performs row gather
//! along axis 0 (`[batch, seq] u32 -> [batch, seq, dim]`); `as_output`
//! reuses the same weight as a tied output projection (`hidden @ Wᵀ ->
//! logits`), matching Qwen3.5's lm_head-tied configuration.

use crate::core::{Loader, QuantMode};
use crate::Result;
use anyhow::anyhow;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

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
        /// Bits per quantized weight (2 / 4 / 8).
        bits: i32,
        /// Quantization scheme from loader metadata.
        mode: QuantMode,
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
            let qmeta = loader.quant_meta_for(prefix).ok_or_else(|| {
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
                    mode: qmeta.mode,
                },
            })
        } else {
            Ok(Embedding {
                inner: EmbeddingImpl::Fp { weight },
            })
        }
    }

    pub fn output_dtype(&self) -> Dtype {
        match &self.inner {
            EmbeddingImpl::Fp { weight, .. } => weight.dtype(),
            EmbeddingImpl::Quant { scales, biases, .. } => {
                biases.as_ref().map_or(scales.dtype(), Array::dtype)
            }
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
            EmbeddingImpl::Fp { weight, .. } => Ok(weight.take_on(tokens, 0, target)?),
            EmbeddingImpl::Quant {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => match qembedding_decode_on(
                tokens,
                QEmbeddingDecode {
                    weight,
                    scales,
                    biases: biases.as_ref(),
                    group_size: *group_size,
                    bits: *bits,
                    mode: *mode,
                },
                target,
            )? {
                Some(y) => Ok(y),
                None => {
                    // P8a-stage5: gather packed rows first, then dequantize the
                    // tiny slice — mirrors mlx-lm's QuantizedEmbedding. Per-token
                    // dequant work drops from O(vocab × dim) to O(B × S × dim).
                    // Quantization metadata is per-row (scales / biases sized
                    // along vocab axis), so axis-0 gather preserves group
                    // alignment.
                    let weight_rows = weight.take_on(tokens, 0, target)?;
                    let scales_rows = scales.take_on(tokens, 0, target)?;
                    let biases_rows = biases
                        .as_ref()
                        .map(|b| b.take_on(tokens, 0, target))
                        .transpose()?;
                    let dequant = mlx::quantization::dequantize_on(
                        &weight_rows,
                        &scales_rows,
                        biases_rows.as_ref(),
                        Some(*group_size),
                        Some(*bits),
                        mode.mlx_mode(),
                        None,
                        None,
                        target,
                    )?;
                    Ok(dequant)
                }
            },
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
                mode,
            } => Ok(mlx::quantization::quantized_matmul_on(
                hidden,
                weight,
                scales,
                biases.as_ref(),
                /* transpose = */ true,
                Some(*group_size),
                Some(*bits),
                mode.mlx_mode(),
                target,
            )?),
        }
    }

    /// Return a dense `[vocab, dim]` embedding table for diffusion
    /// self-conditioning (`probs @ weight`). Quantized checkpoints are
    /// dequantized by the caller's stream once per generation request.
    pub(crate) fn dense_weight_on(&self, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        match &self.inner {
            EmbeddingImpl::Fp { weight, .. } => Ok(weight.clone()),
            EmbeddingImpl::Quant {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Ok(mlx::quantization::dequantize_on(
                weight,
                scales,
                biases.as_ref(),
                Some(*group_size),
                Some(*bits),
                mode.mlx_mode(),
                None,
                None,
                target,
            )?),
        }
    }
}

struct QEmbeddingDecode<'a> {
    weight: &'a Array,
    scales: &'a Array,
    biases: Option<&'a Array>,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
}

fn qembedding_decode_on(
    tokens: &Array,
    params: QEmbeddingDecode<'_>,
    target: impl Into<StreamOrDevice>,
) -> Result<Option<Array>> {
    let Some(biases) = params.biases else {
        return Ok(None);
    };
    if params.group_size != 64 || params.bits != 4 || !params.mode.uses_affine_storage() {
        return Ok(None);
    }

    let weight_shape = params.weight.shape();
    let weight_dims = weight_shape.as_slice();
    if weight_dims.len() != 2 {
        return Ok(None);
    }
    let vocab = weight_dims[0];
    let packed_dim = weight_dims[1];
    let dim = packed_dim * 8;
    if vocab <= 0 || packed_dim <= 0 || dim % params.group_size != 0 {
        return Ok(None);
    }
    let sb_shape = [vocab, dim / params.group_size];
    if params.scales.shape().as_slice() != sb_shape || biases.shape().as_slice() != sb_shape {
        return Ok(None);
    }
    let output_dtype = biases.dtype();
    if params.scales.dtype() != output_dtype {
        return Ok(None);
    }

    let mut out_dims = tokens.shape().as_slice().to_vec();
    out_dims.push(dim);
    let out_shape = Shape::from(out_dims);
    let token_count = i32::try_from(tokens.shape().numel()).map_err(|_| {
        anyhow!(
            "Embedding quantized decode input too large: {} tokens",
            tokens.shape().numel()
        )
    })?;
    if token_count == 0 {
        return Ok(Some(Array::zeros_on(
            out_shape,
            output_dtype,
            target.into(),
        )?));
    }
    let target = target.into();
    let kernel = qembedding_decode_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[tokens, params.weight, params.scales, biases])
        .output_shapes(&[out_shape])
        .output_dtypes(&[output_dtype])
        .grid(token_count * dim, 1, 1)
        .threadgroup(256.min(token_count * dim), 1, 1)
        .stream(target)
        .template_int("PACKED_DIM", packed_dim)
        .template_int("GROUPS", dim / params.group_size)
        .template_int("DIM", dim)
        .template_int("TOKEN_COUNT", token_count)
        .dispatch()?;
    Ok(Some(outputs.take_at(0)?))
}

fn qembedding_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let source = r#"
        uint elem = thread_position_in_grid.x;
        if (elem >= TOKEN_COUNT * DIM) {
            return;
        }

        uint token_idx = elem / DIM;
        uint d = elem - token_idx * DIM;
        uint token = uint(tokens[token_idx]);
        uint packed_idx = d >> 3;
        uint shift = (d & 7u) << 2;
        uint q = (w[token * PACKED_DIM + packed_idx] >> shift) & 0x0fu;
        uint group = d >> 6;
        uint sb = token * GROUPS + group;
        float y = float(scales[sb]) * float(q) + float(biases[sb]);
        out[elem] = static_cast<__typeof__(*out)>(y);
    "#;

    let kernel = MetalKernel::builder("ironmlx_qembedding_decode_4bit_gs64")
        .inputs(&["tokens", "w", "scales", "biases"])
        .outputs(&["out"])
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{ops, Array, Dtype};
    use serial_test::serial;

    fn fp_embedding(weight: Array) -> Embedding {
        Embedding {
            inner: EmbeddingImpl::Fp { weight },
        }
    }

    fn assert_all_close(got: &Array, expected: &Array, tol: f32) {
        let got = ops::cast::astype(got, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = ops::cast::astype(expected, Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= tol,
                "idx={idx} got={g} expected={e} tol={tol}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
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
    #[serial(mlx_metal)]
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

    fn assert_quantized_tokens_match_dequantize(
        raw_dtype: Dtype,
        bits: i32,
        mode: QuantMode,
        token_values: &[u32],
        token_shape: &[i32],
    ) {
        let vocab = 4_i32;
        let dim = 64_i32;
        let group_size = 64_i32;

        let w_data: Vec<f32> = (0..(vocab * dim))
            .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
            .collect();
        let raw_w_f32: Array = (w_data.as_slice(), (vocab, dim)).try_into().unwrap();
        let raw_w = ops::cast::astype(&raw_w_f32, raw_dtype).unwrap();
        let q = mlx::quantization::quantize(&raw_w, Some(group_size), Some(bits), "affine", None)
            .unwrap();
        let weight = q[0].clone();
        let scales = q[1].clone();
        let biases = q[2].clone();
        let tokens: Array = (token_values, token_shape).try_into().unwrap();

        let weight_rows = weight.take(&tokens, 0).unwrap();
        let scales_rows = scales.take(&tokens, 0).unwrap();
        let biases_rows = biases.take(&tokens, 0).unwrap();
        let expected = mlx::quantization::dequantize(
            &weight_rows,
            &scales_rows,
            Some(&biases_rows),
            Some(group_size),
            Some(bits),
            "affine",
            None,
            None,
        )
        .unwrap();

        let layer = Embedding {
            inner: EmbeddingImpl::Quant {
                weight,
                scales,
                biases: Some(biases),
                group_size,
                bits,
                mode,
            },
        };
        let got = layer.forward(&tokens).unwrap();

        assert_eq!(got.shape().as_slice(), expected.shape().as_slice());
        assert_eq!(got.dtype(), expected.dtype());
        assert_all_close(&got, &expected, 0.001);
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_single_token_forward_matches_dequantize_bfloat16() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Bfloat16,
            4,
            QuantMode::Affine,
            &[2],
            &[1, 1],
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_single_token_forward_matches_dequantize_float32() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Float32,
            4,
            QuantMode::Affine,
            &[2],
            &[1, 1],
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_multi_token_forward_matches_dequantize_float32() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Float32,
            4,
            QuantMode::Affine,
            &[1, 2, 3, 0],
            &[2, 2],
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_8bit_single_token_forward_matches_dequantize_bfloat16() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Bfloat16,
            8,
            QuantMode::Affine,
            &[2],
            &[1, 1],
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn quantized_2bit_single_token_forward_matches_dequantize_float32() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Float32,
            2,
            QuantMode::Affine,
            &[2],
            &[1, 1],
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn optiq_quantized_single_token_forward_matches_dequantize_bfloat16() {
        assert_quantized_tokens_match_dequantize(
            Dtype::Bfloat16,
            4,
            QuantMode::OptiQ,
            &[2],
            &[1, 1],
        );
    }
}
