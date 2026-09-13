//! Model-independent weight access and quantized storage contracts.
//!
//! This module does not interpret model configuration or checkpoint layouts.
//! Providers resolve their own tensor names and per-prefix quantization metadata.
//! Layer construction borrows tensors; the provider need not be thread-safe and
//! the contract does not evaluate, copy, or move tensors between devices.

use anyhow::{anyhow, Context};
use mlx::{Array, Dtype};

use crate::Result;

/// Read-only weight access used when constructing neural-network layers.
///
/// Implementations own checkpoint lookup and error context. Quantization must
/// be resolved for the requested prefix, including any per-layer override.
/// Optional tensors are absent only when lookup returns `None`.
///
/// This trait has no `Send`/`Sync` bound: borrowing parameters at construction
/// time must not impose new sharing requirements on `mlx::Array`.
pub trait WeightSource {
    /// Look up a required tensor, preserving the provider's missing-key error.
    fn tensor(&self, key: &str) -> Result<&Array>;

    /// Look up an optional tensor without synthesizing a default value.
    fn tensor_opt(&self, key: &str) -> Option<&Array>;

    /// Whether a tensor exists. A scales tensor selects quantized storage.
    fn contains(&self, key: &str) -> bool {
        self.tensor_opt(key).is_some()
    }

    /// Fully resolved quantization metadata for a tensor prefix.
    /// Providers containing only full-precision weights may keep this default.
    fn quant_meta_for(&self, _prefix: &str) -> Option<QuantMeta> {
        None
    }
}

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Affine quantization (scale + offset per group).
    Affine,
    /// OptiQ mixed-bit quantization. Current MLX checkpoints use an
    /// affine-compatible packed tensor layout, but the model format and
    /// per-layer bit allocation are treated as an independent contract.
    OptiQ,
    /// OCP microscaling 4-bit floating-point quantization.
    Mxfp4,
    /// OCP microscaling 8-bit floating-point quantization.
    Mxfp8,
}

impl QuantMode {
    pub(crate) fn mlx_backend_mode(self) -> &'static str {
        match self {
            Self::Affine | Self::OptiQ => "affine",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }

    pub(crate) fn uses_affine_storage(self) -> bool {
        matches!(self, Self::Affine | Self::OptiQ)
    }

    pub(crate) fn output_dtype(self, scales_dtype: Dtype, biases_dtype: Option<Dtype>) -> Dtype {
        match self {
            Self::Affine | Self::OptiQ => biases_dtype.unwrap_or(scales_dtype),
            Self::Mxfp4 | Self::Mxfp8 => Dtype::Bfloat16,
        }
    }
}

/// Quantization metadata for a weight tensor prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantMeta {
    /// Group size for per-group quantization parameters.
    pub group_size: i32,
    /// Bits per quantized weight.
    pub bits: i32,
    /// Quantization scheme.
    pub mode: QuantMode,
}

pub(crate) fn logical_width_from_packed(packed_columns: i32, bits: i32) -> Result<i32> {
    if packed_columns <= 0 {
        return Err(anyhow!(
            "packed quantized width must be positive, got {packed_columns}"
        ));
    }
    if bits <= 0 {
        return Err(anyhow!("quantized bit width must be positive, got {bits}"));
    }

    let packed_bits = packed_columns
        .checked_mul(32)
        .ok_or_else(|| anyhow!("packed quantized width overflows i32: {packed_columns} * 32"))?;
    if packed_bits % bits != 0 {
        return Err(anyhow!(
            "packed quantized width {packed_columns} does not encode an integral logical width at {bits} bits"
        ));
    }

    Ok(packed_bits / bits)
}

impl QuantMeta {
    pub(crate) fn validate_storage(
        self,
        prefix: &str,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
    ) -> Result<()> {
        match self.mode {
            QuantMode::Affine => {
                self.validate_affine_compatible_storage(prefix, weight, scales, biases, "affine")?;
            }
            QuantMode::OptiQ => {
                self.validate_affine_compatible_storage(
                    prefix,
                    weight,
                    scales,
                    biases,
                    "OptiQ affine-compatible",
                )?;
            }
            QuantMode::Mxfp4 | QuantMode::Mxfp8 => {
                if weight.dtype() != Dtype::Uint32 {
                    return Err(anyhow!(
                        "{prefix}: {} packed weight must have dtype uint32, got {:?}",
                        self.mode.mlx_backend_mode(),
                        weight.dtype()
                    ));
                }
                if scales.dtype() != Dtype::Uint8 {
                    return Err(anyhow!(
                        "{prefix}: {} scales must have dtype uint8, got {:?}",
                        self.mode.mlx_backend_mode(),
                        scales.dtype()
                    ));
                }
                if biases.is_some() {
                    return Err(anyhow!(
                        "{prefix}: {} storage must not contain affine quantization biases",
                        self.mode.mlx_backend_mode()
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_affine_compatible_storage(
        self,
        prefix: &str,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
        scheme_name: &str,
    ) -> Result<()> {
        if weight.dtype() != Dtype::Uint32 {
            return Err(anyhow!(
                "{prefix}: {scheme_name} packed weight must have dtype uint32, got {:?}",
                weight.dtype()
            ));
        }
        let biases = biases.ok_or_else(|| {
            anyhow!("{prefix}: {scheme_name} storage requires quantization biases")
        })?;
        if !is_supported_affine_parameter_dtype(scales.dtype())
            || !is_supported_affine_parameter_dtype(biases.dtype())
        {
            return Err(anyhow!(
                "{prefix}: {scheme_name} scales and biases must use a supported real floating dtype, got {:?} and {:?}",
                scales.dtype(),
                biases.dtype()
            ));
        }
        match self.mode {
            QuantMode::OptiQ if self.group_size != 64 => {
                return Err(anyhow!(
                    "{prefix}: unsupported {scheme_name} group_size {}; supported value is 64",
                    self.group_size
                ));
            }
            QuantMode::Affine if !matches!(self.group_size, 32 | 64 | 128) => {
                return Err(anyhow!(
                    "{prefix}: unsupported {scheme_name} group_size {}; supported values are 32, 64, and 128",
                    self.group_size
                ));
            }
            _ => {}
        }

        let weight_shape = weight.shape();
        let scales_shape = scales.shape();
        let biases_shape = biases.shape();
        if weight_shape.len() < 2 || scales_shape.len() != weight_shape.len() {
            return Err(anyhow!(
                "{prefix}: {scheme_name} weight and scales must have matching rank of at least 2, got {:?} and {:?}",
                weight_shape.as_slice(),
                scales_shape.as_slice()
            ));
        }
        if scales_shape != biases_shape {
            return Err(anyhow!(
                "{prefix}: {scheme_name} scales and biases must have the same shape, got {:?} and {:?}",
                scales_shape.as_slice(),
                biases_shape.as_slice()
            ));
        }

        let trailing_axis = weight_shape.len() - 1;
        if weight_shape.as_slice()[..trailing_axis] != scales_shape.as_slice()[..trailing_axis] {
            return Err(anyhow!(
                "{prefix}: {scheme_name} weight and parameters must have identical leading dimensions, got {:?} and {:?}",
                &weight_shape.as_slice()[..trailing_axis],
                &scales_shape.as_slice()[..trailing_axis]
            ));
        }

        let logical_width =
            logical_width_from_packed(weight_shape.as_slice()[trailing_axis], self.bits)
                .with_context(|| format!("{prefix}: invalid {scheme_name} packed weight shape"))?;
        if logical_width % self.group_size != 0 {
            return Err(anyhow!(
                "{prefix}: {scheme_name} logical width {logical_width} is not divisible by group_size {}",
                self.group_size
            ));
        }
        let expected_groups = logical_width / self.group_size;
        let actual_groups = scales_shape.as_slice()[trailing_axis];
        if actual_groups != expected_groups {
            return Err(anyhow!(
                "{prefix}: {scheme_name} parameter trailing width must be {expected_groups}, got {actual_groups}"
            ));
        }

        Ok(())
    }
}

fn is_supported_affine_parameter_dtype(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16)
}
