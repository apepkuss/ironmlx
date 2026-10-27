//! SwiGLU MLP — `down( silu(gate(x)) * up(x) )`.

use std::sync::OnceLock;

use mlx::compile::CompiledFn;
use mlx::{Array, StreamOrDevice};

use crate::core::weights::WeightSource;
use crate::nn::activations::{build_swiglu, invoke_swiglu};
use crate::nn::Linear;
use crate::Result;

/// SwiGLU feed-forward block, as used by Llama / Qwen / Mistral families.
///
/// Computes `down( silu(gate(x)) * up(x) )` where `silu(z) = z * sigmoid(z)`.
pub struct Mlp {
    gate_up: GateUp,
    down: Linear,
    swiglu: OnceLock<CompiledFn>,
}

enum GateUp {
    Separate { gate: Linear, up: Linear },
    Fused(Box<FusedGateUp>),
}

struct FusedGateUp {
    projection: Linear,
    gate: Linear,
    up: Linear,
}

/// One half's MLP under the half-chunk pipeline, the ANE's channels pending.
pub(crate) struct AneHalf {
    split: &'static std::sync::Mutex<super::ane_ffn::AneFfn>,
    pending: super::ane_ffn::Pending,
    /// The GPU's channels; submitted before the ANE's output is awaited.
    pub(crate) partial: Array,
    dims: Vec<i32>,
}

impl Mlp {
    /// Build an `Mlp` from `loader`, expecting the three sub-projections at
    /// `{prefix}.gate_proj`, `{prefix}.up_proj`, and `{prefix}.down_proj`.
    pub fn from_loader(loader: &(impl WeightSource + ?Sized), prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_up: GateUp::Separate {
                gate: Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?,
                up: Linear::from_loader(loader, &format!("{prefix}.up_proj"))?,
            },
            down: Linear::from_loader(loader, &format!("{prefix}.down_proj"))?,
            swiglu: OnceLock::new(),
        })
    }

    pub(crate) fn from_loader_dflash2(
        loader: &(impl WeightSource + ?Sized),
        prefix: &str,
    ) -> Result<Self> {
        let gate = Linear::from_loader(loader, &format!("{prefix}.gate_proj"))?;
        let up = Linear::from_loader(loader, &format!("{prefix}.up_proj"))?;
        let down = Linear::from_loader(loader, &format!("{prefix}.down_proj"))?;
        Self::from_components_dflash2(gate, up, down)
    }

    pub(crate) fn from_components_dflash2(gate: Linear, up: Linear, down: Linear) -> Result<Self> {
        if gate.out_features() != up.out_features() {
            anyhow::bail!(
                "DFlash2 fused MLP requires matching gate/up widths, got {} and {}",
                gate.out_features(),
                up.out_features()
            );
        }
        if gate.quantized_parts().is_none() && up.quantized_parts().is_none() {
            return Ok(Self::from_components(gate, up, down));
        }
        let projection =
            Linear::fuse_quantized_outputs(&[&gate, &up], "DFlash2 fused MLP gate/up")?;
        let mut separate = projection.split_quantized_outputs(
            &[gate.out_features(), up.out_features()],
            "DFlash2 split MLP gate/up",
        )?;
        Ok(Self {
            gate_up: GateUp::Fused(Box::new(FusedGateUp {
                projection,
                gate: separate.remove(0),
                up: separate.remove(0),
            })),
            down,
            swiglu: OnceLock::new(),
        })
    }

    /// Projections applied to activations (fused storage when fused).
    pub(crate) fn activation_projections(&self) -> Vec<&Linear> {
        let mut out = match &self.gate_up {
            GateUp::Separate { gate, up } => vec![gate, up],
            GateUp::Fused(fused) => vec![&fused.projection],
        };
        out.push(&self.down);
        out
    }

    /// Experimental shared weight layout: move gate/up (fused storage plus
    /// its row views) and down into single-resident stores. Returns the
    /// number of stores created.
    pub(crate) fn share_weight_layout(&mut self) -> Result<usize> {
        let mut stores = 0;
        if let GateUp::Fused(fused) = &mut self.gate_up {
            let FusedGateUp {
                projection,
                gate,
                up,
            } = fused.as_mut();
            if let Some(store) = projection.share_whole()? {
                let gate_width = gate.out_features();
                gate.share_rows(&store, 0)?;
                up.share_rows(&store, gate_width)?;
                stores += 1;
            }
        }
        if self.down.share_whole()?.is_some() {
            stores += 1;
        }
        Ok(stores)
    }

    /// Test/composition seam: build an `Mlp` from pre-built sub-projections.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx-lm/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn from_components(gate: Linear, up: Linear, down: Linear) -> Self {
        Self {
            gate_up: GateUp::Separate { gate, up },
            down,
            swiglu: OnceLock::new(),
        }
    }

    fn swiglu(&self) -> &CompiledFn {
        self.swiglu.get_or_init(build_swiglu)
    }

    fn swiglu_on(&self, gate: &Array, up: &Array) -> Result<Array> {
        invoke_swiglu(self.swiglu(), gate, up)
    }

    /// Forward pass on the default stream.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(x, ())
    }

    /// Timing diagnostic of a split FFN (`IRONMLX_DIAGNOSTIC_DENSE_FFN_GPU_UNITS`
    /// = U, Dense long prefill only): the GPU computes only the first U x 512
    /// intermediate channels (gate, up, SwiGLU, down over those inputs) after
    /// a synchronization of the layer input, as a split with another engine
    /// would. The output is NOT the FFN; for feasibility timing only.
    fn forward_split_timing_on(
        &self,
        x: &Array,
        units: i32,
        target: StreamOrDevice,
    ) -> Result<Option<Array>> {
        let GateUp::Fused(fused) = &self.gate_up else {
            return Ok(None);
        };
        let Some((gw, gs, gb)) = fused.projection.native_affine4_arrays(target)? else {
            return Ok(None);
        };
        let Some((dw, ds, db)) = self.down.native_affine4_arrays(target)? else {
            return Ok(None);
        };
        let inter = gw.shape().as_slice()[0] / 2;
        let keep = (units * 512).min(inter);
        mlx::transforms::eval(&[x])?;
        let rows = |a: &Array, start: i32, len: i32| -> Result<Array> {
            let w = a.shape().as_slice()[1];
            Ok(mlx::ops::indexing::slice_on(
                a,
                &[start, 0][..],
                &[start + len, w][..],
                target,
            )?)
        };
        let project = |w: &Array, s: &Array, b: &Array, input: &Array| -> Result<Array> {
            let wd = mlx::quantization::dequantize_on(
                w,
                s,
                Some(b),
                Some(64),
                Some(4),
                "affine",
                None,
                Some(mlx::Dtype::Bfloat16),
                target,
            )?;
            Ok(mlx::ops::matmul::matmul_on(
                input,
                &mlx::ops::shape::transpose_on(&wd, target)?,
                target,
            )?)
        };
        let g = project(
            &rows(&gw, 0, keep)?,
            &rows(&gs, 0, keep)?,
            &rows(&gb, 0, keep)?,
            x,
        )?;
        let u = project(
            &rows(&gw, inter, keep)?,
            &rows(&gs, inter, keep)?,
            &rows(&gb, inter, keep)?,
            x,
        )?;
        let activated = self.swiglu_on(&g, &u)?;
        let cols = |a: &Array, len: i32| -> Result<Array> {
            let h = a.shape().as_slice()[0];
            Ok(mlx::ops::shape::contiguous_on(
                &mlx::ops::indexing::slice_on(a, &[0, 0][..], &[h, len][..], target)?,
                false,
                target,
            )?)
        };
        Ok(Some(project(
            &cols(&dw, keep / 8)?,
            &cols(&ds, keep / 64)?,
            &cols(&db, keep / 64)?,
            &activated,
        )?))
    }

    /// Half `half` (0, then 1, of each layer) of a full prefill chunk under
    /// the half-chunk pipeline (`DENSE_PREFILL_ANE_FFN_PIPELINE`): hands the
    /// Neural Engine its channels and queues the GPU's (the tiled prefill
    /// kernels); [`Self::ane_half_finish_on`] adds the two. `None` when the
    /// shapes or weights do not take it or the ANE is unavailable.
    pub(crate) fn ane_half_begin_on(
        &self,
        x: &Array,
        half: usize,
        target: StreamOrDevice,
    ) -> Result<Option<AneHalf>> {
        let units = super::dense_fast_path::prefill_ane_ffn_units();
        let dims = x.shape().as_slice().to_vec();
        let hidden = *dims.last().unwrap_or(&0);
        let rows = super::dense_fast_path::PREFILL_DEQUANT_ROWS / 2;
        if !super::dense_fast_path::ane_armed()
            || !super::dense_fast_path::prefill_ane_ffn_pipelined()
            || units == 0
            || x.dtype() != mlx::Dtype::Bfloat16
            || hidden == 0
            || x.size() / hidden as usize != rows
        {
            return Ok(None);
        }
        let GateUp::Fused(fused) = &self.gate_up else {
            return Ok(None);
        };
        let started = fused.projection.with_tiled_prepared(target, |gate_up| {
            self.down.with_tiled_prepared(target, |down| {
                let inter = gate_up.weight.shape().as_slice()[0] / 2;
                if gate_up.weight.shape().as_slice()[1] * 8 != hidden
                    || down.weight.shape().as_slice() != [hidden, inter / 8]
                {
                    return Ok(None);
                }
                let Some(split) = super::ane_ffn::pipelined_instance(
                    rows,
                    hidden as usize,
                    inter as usize,
                    units,
                ) else {
                    return Ok(None);
                };
                super::ane_ffn::check()?;
                let mut guard = split
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ANE FFN split poisoned"))?;
                let x2 = mlx::ops::shape::reshape_on(x, &[rows as i32, hidden][..], target)?;
                let pending = guard.begin_tiled_half(&x2, gate_up, down, half, target)?;
                let g = guard.gpu_channels();
                let activated = super::dense_prefill_kernels::prefill_tiled_swiglu_cols(
                    &pending.ready,
                    gate_up,
                    g,
                    target,
                )?;
                let partial = super::dense_prefill_kernels::prefill_tiled_gemm_leading(
                    &activated, down, target,
                )?;
                Ok(Some(AneHalf {
                    split,
                    pending,
                    partial,
                    dims: dims.clone(),
                }))
            })
        })?;
        Ok(started.flatten().flatten())
    }

    /// The MLP output of a half started by [`Self::ane_half_begin_on`].
    pub(crate) fn ane_half_finish_on(
        &self,
        half: AneHalf,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let out = half
            .split
            .lock()
            .map_err(|_| anyhow::anyhow!("ANE FFN split poisoned"))?
            .finish(half.pending, &half.partial, target)?;
        Ok(mlx::ops::shape::reshape_on(&out, &half.dims[..], target)?)
    }

    /// A full prefill chunk split between the GPU (leading intermediate
    /// channels, the BF16 dequantize + matmul route for gate/up, the quantized
    /// matmul for down) and the Neural Engine (`ane_ffn`); `None` when the
    /// shapes or weights do not take it or the ANE is unavailable.
    fn forward_ane_split_on(&self, x: &Array, target: StreamOrDevice) -> Result<Option<Array>> {
        let units = super::dense_fast_path::prefill_ane_ffn_units();
        // The pipeline shares only the halves of its interleaved chunks
        // (`ane_half_begin_on`).
        if !super::dense_fast_path::ane_armed()
            || super::dense_fast_path::prefill_ane_ffn_pipelined()
        {
            return Ok(None);
        }
        let dims = x.shape();
        let dims = dims.as_slice();
        let hidden = *dims.last().unwrap_or(&0);
        if units == 0
            || x.dtype() != mlx::Dtype::Bfloat16
            || hidden == 0
            || x.size() / hidden as usize != super::dense_fast_path::PREFILL_DEQUANT_ROWS
        {
            return Ok(None);
        }
        let GateUp::Fused(fused) = &self.gate_up else {
            return Ok(None);
        };
        // Single-resident tiled stores: the GPU's channels take the tiled
        // prefill kernels (SwiGLU in the gate / up epilogue, the down
        // projection over its leading inputs) and the ANE's operands are read
        // from the same layout, so no store converts to the native layout.
        let rows = super::dense_fast_path::PREFILL_DEQUANT_ROWS;
        let tiled = fused.projection.with_tiled_prepared(target, |gate_up| {
            self.down.with_tiled_prepared(target, |down| {
                let inter = gate_up.weight.shape().as_slice()[0] / 2;
                if gate_up.weight.shape().as_slice()[1] * 8 != hidden
                    || down.weight.shape().as_slice() != [hidden, inter / 8]
                {
                    return Ok(None);
                }
                let Some(split) =
                    super::ane_ffn::instance(rows, hidden as usize, inter as usize, units)
                else {
                    return Ok(None);
                };
                super::ane_ffn::check()?;
                let mut split = split
                    .lock()
                    .map_err(|_| anyhow::anyhow!("ANE FFN split poisoned"))?;
                let x2 = mlx::ops::shape::reshape_on(x, &[rows as i32, hidden][..], target)?;
                let pending = split.begin_tiled(&x2, gate_up, down, target)?;
                let g = split.gpu_channels();
                let activated = super::dense_prefill_kernels::prefill_tiled_swiglu_cols(
                    &pending.ready,
                    gate_up,
                    g,
                    target,
                )?;
                let partial = super::dense_prefill_kernels::prefill_tiled_gemm_leading(
                    &activated, down, target,
                )?;
                let out = split.finish(pending, &partial, target)?;
                Ok(Some(mlx::ops::shape::reshape_on(&out, dims, target)?))
            })
        })?;
        match tiled {
            Some(Some(Some(out))) => return Ok(Some(out)),
            Some(Some(None)) => return Ok(None),
            _ => {}
        }
        let Some((gw, gs, gb)) = fused.projection.native_affine4_arrays(target)? else {
            return Ok(None);
        };
        let Some((dw, ds, db)) = self.down.native_affine4_arrays(target)? else {
            return Ok(None);
        };
        let inter = gw.shape().as_slice()[0] / 2;
        if gw.shape().as_slice()[1] * 8 != hidden || dw.shape().as_slice() != [hidden, inter / 8] {
            return Ok(None);
        }
        let rows = super::dense_fast_path::PREFILL_DEQUANT_ROWS;
        let Some(split) = super::ane_ffn::instance(rows, hidden as usize, inter as usize, units)
        else {
            return Ok(None);
        };
        super::ane_ffn::check()?;
        let mut split = split
            .lock()
            .map_err(|_| anyhow::anyhow!("ANE FFN split poisoned"))?;
        let x2 = mlx::ops::shape::reshape_on(x, &[rows as i32, hidden][..], target)?;
        let pending = split.begin(&x2, (&gw, &gs, &gb), (&dw, &ds, &db), target)?;
        let g = split.gpu_channels();
        let slice_rows = |a: &Array, start: i32, len: i32| -> Result<Array> {
            let w = a.shape().as_slice()[1];
            Ok(mlx::ops::indexing::slice_on(
                a,
                &[start, 0][..],
                &[start + len, w][..],
                target,
            )?)
        };
        let dequant_gemm = |start: i32| -> Result<Array> {
            let w = mlx::quantization::dequantize_on(
                &slice_rows(&gw, start, g)?,
                &slice_rows(&gs, start, g)?,
                Some(&slice_rows(&gb, start, g)?),
                Some(64),
                Some(4),
                "affine",
                None,
                Some(mlx::Dtype::Bfloat16),
                target,
            )?;
            Ok(mlx::ops::matmul::matmul_on(
                &pending.ready,
                &mlx::ops::shape::transpose_on(&w, target)?,
                target,
            )?)
        };
        let gate = dequant_gemm(0)?;
        let up = dequant_gemm(inter)?;
        let activated = self.swiglu_on(&gate, &up)?;
        let cols = |a: &Array, len: i32| -> Result<Array> {
            Ok(mlx::ops::shape::contiguous_on(
                &mlx::ops::indexing::slice_on(a, &[0, 0][..], &[hidden, len][..], target)?,
                false,
                target,
            )?)
        };
        let partial = mlx::quantization::quantized_matmul_on(
            &activated,
            &cols(&dw, g / 8)?,
            &cols(&ds, g / 64)?,
            Some(&cols(&db, g / 64)?),
            true,
            Some(64),
            Some(4),
            "affine",
            target,
        )?;
        let mut out = split.finish(pending, &partial, target)?;
        // Diagnostic (`IRONMLX_DIAGNOSTIC_ANE_FFN_GPU_REST=1`): the ANE's
        // channels computed on the GPU instead (the split's summation, no
        // int8): the noise floor of splitting alone.
        if std::env::var("IRONMLX_DIAGNOSTIC_ANE_FFN_GPU_REST").as_deref() == Ok("1") {
            let a = inter - g;
            let rest = |start: i32| -> Result<Array> {
                let w = mlx::quantization::dequantize_on(
                    &slice_rows(&gw, start, a)?,
                    &slice_rows(&gs, start, a)?,
                    Some(&slice_rows(&gb, start, a)?),
                    Some(64),
                    Some(4),
                    "affine",
                    None,
                    Some(mlx::Dtype::Bfloat16),
                    target,
                )?;
                Ok(mlx::ops::matmul::matmul_on(
                    &x2,
                    &mlx::ops::shape::transpose_on(&w, target)?,
                    target,
                )?)
            };
            let act = self.swiglu_on(&rest(g)?, &rest(inter + g)?)?;
            let tail = |q: &Array, from: i32, len: i32| -> Result<Array> {
                Ok(mlx::ops::shape::contiguous_on(
                    &mlx::ops::indexing::slice_on(
                        q,
                        &[0, from][..],
                        &[hidden, from + len][..],
                        target,
                    )?,
                    false,
                    target,
                )?)
            };
            let down_rest = mlx::quantization::quantized_matmul_on(
                &act,
                &tail(&dw, g / 8, a / 8)?,
                &tail(&ds, g / 64, a / 64)?,
                Some(&tail(&db, g / 64, a / 64)?),
                true,
                Some(64),
                Some(4),
                "affine",
                target,
            )?;
            let sum = &mlx::ops::cast::astype_on(&partial, mlx::Dtype::Float32, target)?
                + &mlx::ops::cast::astype_on(&down_rest, mlx::Dtype::Float32, target)?;
            out = mlx::ops::cast::astype_on(&sum, mlx::Dtype::Bfloat16, target)?;
        }
        // Diagnostic (`IRONMLX_DIAGNOSTIC_ANE_FFN_COMPARE=1`): each split's
        // error against the GPU route over all channels (synchronizes).
        if std::env::var("IRONMLX_DIAGNOSTIC_ANE_FFN_COMPARE").as_deref() == Ok("1") {
            let f32 = |a: &Array| mlx::ops::cast::astype_on(a, mlx::Dtype::Float32, target);
            let full_rows = |start: i32| -> Result<Array> {
                let w = mlx::quantization::dequantize_on(
                    &slice_rows(&gw, start, inter)?,
                    &slice_rows(&gs, start, inter)?,
                    Some(&slice_rows(&gb, start, inter)?),
                    Some(64),
                    Some(4),
                    "affine",
                    None,
                    Some(mlx::Dtype::Bfloat16),
                    target,
                )?;
                Ok(mlx::ops::matmul::matmul_on(
                    &x2,
                    &mlx::ops::shape::transpose_on(&w, target)?,
                    target,
                )?)
            };
            let act = self.swiglu_on(&full_rows(0)?, &full_rows(inter)?)?;
            let full = mlx::quantization::quantized_matmul_on(
                &act,
                &dw,
                &ds,
                Some(&db),
                true,
                Some(64),
                Some(4),
                "affine",
                target,
            )?;
            let rel = |got: &Array, want: &Array| -> Result<f32> {
                let d = &f32(got)? - &f32(want)?;
                let num = mlx::ops::reduction::sum(&(&d * &d), mlx::ops::reduction::All, false)?
                    .item::<f32>()?;
                let w = f32(want)?;
                let den = mlx::ops::reduction::sum(&(&w * &w), mlx::ops::reduction::All, false)?
                    .item::<f32>()?;
                Ok((num / den).sqrt())
            };
            let e_all = rel(&out, &full)?;
            let e_ane = rel(
                &(&f32(&out)? - &f32(&partial)?),
                &(&f32(&full)? - &f32(&partial)?),
            )?;
            eprintln!("ANE_FFN_COMPARE units={units} rel_all={e_all:.5} rel_ane={e_ane:.5}");
        }
        Ok(Some(mlx::ops::shape::reshape_on(&out, dims, target)?))
    }

    /// Stream-targeted forward pass.
    pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
        let target = target.into();
        if super::dense_fast_path::prefill_dequant_armed() {
            if let Some(y) = self.forward_ane_split_on(x, target)? {
                return Ok(y);
            }
        }
        if super::dense_fast_path::prefill_dequant_armed()
            && x.size() / (*x.shape().as_slice().last().unwrap_or(&1) as usize) >= 512
        {
            static UNITS: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
            if let Some(units) = *UNITS.get_or_init(|| {
                std::env::var("IRONMLX_DIAGNOSTIC_DENSE_FFN_GPU_UNITS")
                    .ok()
                    .and_then(|v| v.parse().ok())
            }) {
                if let Some(y) = self.forward_split_timing_on(x, units, target)? {
                    return Ok(y);
                }
            }
        }
        let (g, u) = match &self.gate_up {
            GateUp::Separate { gate, up } => {
                (gate.forward_on(x, target)?, up.forward_on(x, target)?)
            }
            GateUp::Fused(fused) => {
                let FusedGateUp {
                    projection,
                    gate,
                    up,
                } = fused.as_ref();
                if super::m5_affine4::armed() {
                    if let Some(activated) = projection.prefill_swiglu_on(x, target)? {
                        super::decoder_layer::prefill_stage("mlp_gate_up", &[&activated])?;
                        return self.down.forward_on(&activated, target);
                    }
                }
                if super::product_stable_qmm::is_armed()
                    || super::dflash2_drafter_fusion::is_armed()
                    || super::m5_affine4::armed()
                {
                    let output = projection.forward_on(x, target)?;
                    let mut parts = mlx::ops::shape::split_n_on(&output, 2, -1, target)?;
                    if parts.len() != 2 {
                        anyhow::bail!("DFlash2 fused MLP gate/up returned {} parts", parts.len());
                    }
                    (parts.remove(0), parts.remove(0))
                } else {
                    (gate.forward_on(x, target)?, up.forward_on(x, target)?)
                }
            }
        };
        super::decoder_layer::prefill_stage("mlp_gate_up", &[&g, &u])?;
        let activated = self.swiglu_on(&g, &u)?;
        super::decoder_layer::prefill_stage("mlp_swiglu", &[&activated])?;
        self.down.forward_on(&activated, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    #[test]
    #[serial(mlx_metal)]
    fn swiglu_forward_matches_manual_silu_mul() {
        let identity: Array = (&[1.0_f32, 0.0, 0.0, 1.0][..], (2, 2)).try_into().unwrap();
        let mlp = Mlp::from_components(
            Linear::new_fp(identity.clone(), None),
            Linear::new_fp(identity.clone(), None),
            Linear::new_fp(identity, None),
        );
        let x: Array = (&[0.5_f32, -1.0][..], (1, 2)).try_into().unwrap();

        let y = mlp.forward(&x).expect("forward");
        let got: Vec<f32> = y.to_vec().expect("to_vec");
        let want: Vec<f32> = x
            .to_vec::<f32>()
            .expect("input to_vec")
            .into_iter()
            .map(|v| v * (1.0 / (1.0 + (-v).exp())) * v)
            .collect();

        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-5);
        }
    }
}
