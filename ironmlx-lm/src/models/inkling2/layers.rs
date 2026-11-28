//! Inkling attention, convolutions, routed experts and cross-pass message gates.
//! Adapted from Apple's mlx-lm Inkling 2 reference (MIT, 2026).
use super::{
    config::TextConfig,
    metal::{relative_logits, static_conv4, ActivationTables},
    ops::*,
};
use crate::core::{
    cache::{GatedDeltaCache, KVCache},
    Loader,
};
use crate::nn::{Linear, RmsNorm};
use anyhow::{ensure, Result};
use mlx::ops::{constructors::arange_on, indexing, shape, sort};
use mlx::{Array, Dtype, StreamOrDevice};

thread_local! {
    pub(super) static FAST_OQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static VERIFY_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) fn with_checkpoint_layout<T>(oq: bool, run: impl FnOnce() -> Result<T>) -> Result<T> {
    struct RestoreLayout(bool);
    impl Drop for RestoreLayout {
        fn drop(&mut self) {
            FAST_OQ.set(self.0);
        }
    }
    let _restore = RestoreLayout(FAST_OQ.replace(oq));
    run()
}

pub(super) fn with_verify_rows<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    struct Scope(bool);
    impl Drop for Scope {
        fn drop(&mut self) {
            VERIFY_ROWS.set(self.0);
        }
    }
    let _scope = Scope(VERIFY_ROWS.replace(true));
    let _stable = crate::nn::product_stable_qmm::scope();
    let _linear = crate::nn::position_stable_linear::scope();
    run()
}

pub(super) struct ExpertLinear {
    weight: Array,
    scales: Option<Array>,
    biases: Option<Array>,
    global: Option<Array>,
    mode: &'static str,
    bits: i32,
    group: i32,
}
impl ExpertLinear {
    pub fn load(
        loader: &(impl crate::core::weights::WeightSource + ?Sized),
        prefix: &str,
        input: i32,
        output: i32,
        experts: i32,
    ) -> Result<Self> {
        let weight = loader.tensor(&format!("{prefix}.weight"))?.clone();
        let scales = loader.tensor_opt(&format!("{prefix}.scales")).cloned();
        let biases = loader.tensor_opt(&format!("{prefix}.biases")).cloned();
        let global = loader
            .tensor_opt(&format!("{prefix}.global_scale"))
            .cloned();
        let mut mode = "affine";
        let mut bits = 0;
        let mut group = 0;
        ensure!(
            weight.ndim() == 3 && weight.shape_at(0) == experts && weight.shape_at(1) == output,
            "{prefix}: invalid expert weight shape"
        );
        if let Some(s) = &scales {
            ensure!(
                weight.dtype() == Dtype::Uint32
                    && s.ndim() == 3
                    && s.shape_at(0) == experts
                    && s.shape_at(1) == output
                    && s.shape_at(2) > 0
                    && input % s.shape_at(2) == 0
                    && weight.shape_at(2) * 32 % input == 0,
                "{prefix}: invalid packed expert storage"
            );
            bits = weight.shape_at(2) * 32 / input;
            group = input / s.shape_at(2);
            if s.dtype() == Dtype::Uint8 {
                mode = match (bits, group) {
                    (4, 16) => "nvfp4",
                    (4, 32) => "mxfp4",
                    (8, 32) => "mxfp8",
                    _ => anyhow::bail!("{prefix}: unsupported expert quantization {bits}/{group}"),
                };
                ensure!(
                    biases.is_none(),
                    "{prefix}: floating-point quantization cannot have affine biases"
                );
                if mode == "nvfp4" {
                    ensure!(
                        global.as_ref().is_some_and(
                            |g| g.dtype() == Dtype::Float32 && g.size() == experts as usize
                        ),
                        "{prefix}: NVFP4 requires one float32 global scale per expert"
                    );
                } else {
                    ensure!(global.is_none(), "{prefix}: global scales require NVFP4");
                }
            } else {
                ensure!(
                    global.is_none(),
                    "{prefix}: affine experts cannot have global scales"
                );
                let meta = loader
                    .quant_meta_for(prefix)
                    .ok_or_else(|| anyhow::anyhow!("{prefix}: missing affine metadata"))?;
                ensure!(
                    meta.bits == bits && meta.group_size == group,
                    "{prefix}: inconsistent affine expert metadata"
                );
                meta.validate_storage(prefix, &weight, s, biases.as_ref())?;
            }
        } else {
            ensure!(
                weight.shape_at(2) == input
                    && matches!(
                        weight.dtype(),
                        Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16
                    ),
                "{prefix}: invalid unquantized expert weight"
            );
        }
        Ok(Self {
            weight,
            scales,
            biases,
            global,
            mode,
            bits,
            group,
        })
    }
    /// The single-token NVFP4 product with the routed activation applied in
    /// the same dispatch, where [`Self::forward`] would use the decode kernel
    /// with an even row tile and return no separate scale.
    fn forward_activated(
        &self,
        x: &Array,
        indices: &Array,
        tables: &ActivationTables,
        target: StreamOrDevice,
    ) -> Result<Option<Array>> {
        let (Some(scales), Some(global)) = (&self.scales, &self.global) else {
            return Ok(None);
        };
        if !(self.mode == "nvfp4"
            && FAST_OQ.get()
            && !VERIFY_ROWS.get()
            && x.dtype() == Dtype::Bfloat16
            && indices.ndim() == 3
            && indices.shape().as_slice() == [1, 1, 8]
            && x.ndim() == 5
            && x.shape_at(1) == 1
            && x.shape_at(2) == 1
            && x.shape_at(4) == 2304
            && self.weight.shape_at(1) == 2304
            && is_gpu(target))
        {
            return Ok(None);
        }
        Ok(Some(super::nvfp4::decode_activated(
            x,
            &self.weight,
            scales,
            global,
            indices,
            4,
            Some(tables.tables()),
            target,
        )?))
    }

    fn forward(
        &self,
        x: &Array,
        indices: &Array,
        sorted: bool,
        defer_global_scale: bool,
        target: StreamOrDevice,
    ) -> Result<(Array, Option<Array>)> {
        let dtype = x.dtype();
        if let (Some(scales), Some(global)) = (&self.scales, &self.global) {
            if self.mode == "nvfp4"
                && dtype == Dtype::Bfloat16
                && indices.ndim() == 3
                && indices.shape_at(0) == 1
                && indices.shape_at(2) == 8
                && (indices.shape_at(1) == 1 || VERIFY_ROWS.get())
                && x.ndim() == 5
                && x.shape_at(1) == indices.shape_at(1)
                && matches!((x.shape_at(2), x.shape_at(4)), (1, 2304) | (8, 1152))
                && self.weight.shape_at(1) == 2304
                && is_gpu(target)
                && (m5_available() || FAST_OQ.get())
            {
                return Ok((
                    super::nvfp4::decode(
                        x,
                        &self.weight,
                        scales,
                        global,
                        indices,
                        if !FAST_OQ.get() {
                            1
                        } else if m5_available() || x.shape_at(-1) == 2304 {
                            4
                        } else {
                            2
                        },
                        target,
                    )?,
                    None,
                ));
            }
        }
        let input = if self.global.is_some() && !(sorted && FAST_OQ.get()) {
            x.astype_on(Dtype::Float32, target)?
        } else {
            x.clone()
        };
        let mut y = if let Some(scales) = &self.scales {
            mlx::quantization::gather_quantized_matmul_on(
                &input,
                &self.weight,
                scales,
                self.biases.as_ref(),
                None,
                Some(indices),
                true,
                Some(self.group),
                Some(self.bits),
                self.mode,
                sorted,
                target,
            )?
        } else {
            let w = indexing::take_on(&self.weight, indices, 0, target)?;
            let mut axes: Vec<i32> = (0..w.ndim() as i32).collect();
            let n = axes.len();
            axes.swap(n - 1, n - 2);
            input.matmul_on(&w.transpose_axes_on(axes.as_slice(), target)?, target)?
        };
        if let Some(global) = &self.global {
            let g = indexing::take_on(&global.reshape_on((-1,), target)?, indices, 0, target)?;
            // MLX's NVFP4 global scale is the tensor amax. The stored E4M3
            // block scales cover E2M1_MAX * E4M3_MAX = 6 * 448.
            let g = scale(&g, 1.0 / (6.0 * 448.0), target)?;
            if defer_global_scale {
                return Ok((y, Some(g)));
            }
            y = y.try_mul_on(&g.expand_dims_on([-1, -2], target)?, target)?;
        }
        Ok((y.astype_on(dtype, target)?, None))
    }

    fn supports_grouped_prefill(&self, mode: &str) -> bool {
        self.scales.is_some()
            && self.mode == mode
            && self.weight.shape_at(1) % 64 == 0
            && self.weight.shape_at(2) % 8 == 0
    }
}

pub(super) fn is_gpu(target: StreamOrDevice) -> bool {
    match target {
        StreamOrDevice::Default => mlx::default_device(),
        StreamOrDevice::Device(device) => device,
        StreamOrDevice::Stream(stream) => stream.device,
        StreamOrDevice::ThreadLocalStream(stream) => stream.device,
    }
    .device_type
        == mlx::DeviceType::Gpu
}

pub(super) fn m5_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::env::var_os("IRONMLX_INKLING_M5").as_deref() != Some(std::ffi::OsStr::new("0"))
            && mlx::metal::nax_available().unwrap_or(false)
    })
}

fn grouped_prefill_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("IRONMLX_INKLING_M5_GROUPED_PREFILL").as_deref()
            != Some(std::ffi::OsStr::new("0"))
    })
}

struct Moe {
    router: Array,
    bias: Array,
    tables: Option<ActivationTables>,
    gate_up: ExpertLinear,
    down: ExpertLinear,
    k: i32,
    gate_cap: f32,
    up_cap: f32,
    route_scale: f32,
}
impl Moe {
    fn load(l: &Loader, p: &str, c: &TextConfig, tables: Option<ActivationTables>) -> Result<Self> {
        Ok(Self {
            tables,
            router: l.tensor(&format!("{p}.gate.weight"))?.clone(),
            bias: l
                .tensor(&format!("{p}.gate.e_score_correction_bias"))?
                .clone(),
            gate_up: ExpertLinear::load(
                l,
                &format!("{p}.experts.gate_up_proj"),
                c.hidden_size,
                2 * c.intermediate_size,
                c.n_routed_experts,
            )?,
            down: ExpertLinear::load(
                l,
                &format!("{p}.experts.down_proj"),
                c.intermediate_size,
                c.hidden_size,
                c.n_routed_experts,
            )?,
            k: c.num_experts_per_tok,
            gate_cap: c.mlp_gate_softcap,
            up_cap: c.mlp_up_softcap,
            route_scale: c.route_scale,
        })
    }
    /// `wide` is `x` already widened to float32, when the caller has it.
    fn forward(&self, x: &Array, wide: Option<&Array>, t: StreamOrDevice) -> Result<MoeOutput> {
        let input = match wide {
            Some(wide) => wide.clone(),
            None => x.astype_on(Dtype::Float32, t)?,
        };
        let router = self.router.transpose_on(t)?;
        let tokens = x.shape_at(1);
        let logits = row_matmul(&input, &router, t)?;
        let (scores, indices) = if FAST_OQ.get()
            && x.shape_at(0) == 1
            && (tokens == 1 || VERIFY_ROWS.get())
            && logits.dtype() == Dtype::Float32
            && logits.shape_at(2) == 128
            && self.k == 8
            && self.bias.dtype() == Dtype::Float32
            && self.bias.shape().as_slice() == [128]
            && is_gpu(t)
        {
            super::metal::route_decode(&logits, &self.bias, t)?
        } else {
            let scores = logits.sigmoid_on(t)?;
            let choices = scores.try_add_on(&self.bias, t)?.try_neg_on(t)?;
            let indices = slice(
                &sort::argpartition_on(&choices, self.k - 1, -1, t)?,
                2,
                0,
                self.k,
                t,
            )?;
            (scores, indices)
        };
        let weights = if self.k == 8
            && x.shape_at(0) == 1
            && (tokens == 1 || VERIFY_ROWS.get())
            && x.dtype() == Dtype::Bfloat16
            && x.shape_at(2) % 32 == 0
            && is_gpu(t)
            && (m5_available() || FAST_OQ.get())
        {
            None
        } else {
            let weights = indexing::take_along_axis_on(&scores, &indices, -1, t)?;
            Some(scale(
                &weights.try_div_on(
                    &weights
                        .sum_on(-1, true, t)?
                        .try_add_on(&scalar(1e-20)?, t)?,
                    t,
                )?,
                self.route_scale,
                t,
            )?)
        };
        let routes = tokens * self.k;
        // MLX's grouped matrix kernel needs at least four routed rows per
        // expert on average. NVFP4 retains FP32 products and global scales.
        // Restrict MXFP8 NAX to full-size chunks. Smaller chunks take the
        // ordinary path, whose reduction order is stable for short prompts.
        let sorted = x.shape_at(0) == 1
            && routes >= 16
            && routes / self.gate_up.weight.shape_at(0) >= 4
            && matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16)
            && is_gpu(t)
            && ((self.gate_up.supports_grouped_prefill("nvfp4")
                && self.down.supports_grouped_prefill("nvfp4"))
                || (FAST_OQ.get()
                    && self.gate_up.supports_grouped_prefill("affine")
                    && self.down.supports_grouped_prefill("affine"))
                || (tokens >= 256
                    && self.gate_up.supports_grouped_prefill("mxfp8")
                    && self.down.supports_grouped_prefill("mxfp8")
                    && m5_available()
                    && grouped_prefill_enabled()));
        let fused_sort =
            FAST_OQ.get() && self.k == 8 && self.tables.is_some() && x.dtype() == Dtype::Bfloat16;
        let mut fused_activation = false;
        let (gate_up, expert_indices, inverse) = if sorted {
            let flat = indices.reshape_on((routes,), t)?;
            let order = sort::argsort_on(&flat, 0, t)?;
            let experts = indexing::take_on(&flat, &order, 0, t)?;
            let (token_indices, inverse) = if fused_sort {
                super::metal::sorted_route_positions(&order, self.k, t)?
            } else {
                let tokens_by_route = arange_on(0.0, f64::from(tokens), 1.0, Dtype::Uint32, t)?
                    .reshape_on((tokens, 1), t)?
                    .broadcast_to_on((tokens, self.k), t)?
                    .reshape_on((routes,), t)?;
                (
                    indexing::take_on(&tokens_by_route, &order, 0, t)?,
                    sort::argsort_on(&order, 0, t)?,
                )
            };
            let input = indexing::take_on(&x.squeeze_on(0, t)?, &token_indices, 0, t)?
                .expand_dims_on(-2, t)?;
            (
                self.gate_up
                    .forward(&input, &experts, true, fused_sort, t)?,
                experts,
                Some(inverse),
            )
        } else if let Some(activated) = match &self.tables {
            Some(tables) if x.shape_at(0) == 1 => self.gate_up.forward_activated(
                &x.expand_dims_on([-2, -3], t)?,
                &indices,
                tables,
                t,
            )?,
            _ => None,
        } {
            fused_activation = true;
            ((activated, None), indices, None)
        } else {
            (
                self.gate_up
                    .forward(&x.expand_dims_on([-2, -3], t)?, &indices, false, false, t)?,
                indices,
                None,
            )
        };
        let (gate_up, gate_scale) = gate_up;
        let activation = if fused_activation {
            gate_up
        } else if let Some(tables) = self
            .tables
            .as_ref()
            .filter(|_| x.shape_at(0) == 1 && x.dtype() == Dtype::Bfloat16 && is_gpu(t))
        {
            tables.apply_scaled(&gate_up, gate_scale.as_ref(), t)?
        } else {
            let dims = gate_up.shape().as_slice().to_vec();
            let last = dims.len() - 1;
            let mut starts = vec![0; dims.len()];
            let mut steps = vec![1; dims.len()];
            steps[last] = 2;
            let gate = indexing::slice_strided_on(
                &gate_up,
                starts.as_slice(),
                dims.as_slice(),
                steps.as_slice(),
                t,
            )?
            .astype_on(Dtype::Float32, t)?;
            starts[last] = 1;
            let up = indexing::slice_strided_on(
                &gate_up,
                starts.as_slice(),
                dims.as_slice(),
                steps.as_slice(),
                t,
            )?
            .astype_on(Dtype::Float32, t)?;
            cap(&gate, self.gate_cap, t)?
                .try_mul_on(&gate.sigmoid_on(t)?, t)?
                .try_mul_on(&cap(&up, self.up_cap, t)?, t)?
                .astype_on(x.dtype(), t)?
        };
        let (out, down_scale) = self.down.forward(
            &activation,
            &expert_indices,
            sorted,
            sorted && fused_sort,
            t,
        )?;
        let out = out.squeeze_on(-2, t)?;
        let Some(weights) = weights else {
            return Ok(MoeOutput::Routed {
                experts: out,
                scores,
                indices: expert_indices,
                route_scale: self.route_scale,
            });
        };
        if let Some(inverse) = inverse.as_ref().filter(|_| fused_sort) {
            return Ok(MoeOutput::Mixed(super::metal::mix_sorted_routes(
                &out,
                &weights,
                inverse,
                down_scale.as_ref(),
                t,
            )?));
        }
        let out = out.astype_on(Dtype::Float32, t)?;
        let out = if let Some(inverse) = inverse {
            indexing::take_on(&out, &inverse, 0, t)?
                .reshape_on((1, tokens, self.k, x.shape_at(2)), t)?
        } else {
            out
        };
        Ok(MoeOutput::Mixed(
            out.try_mul_on(&weights.expand_dims_on(-1, t)?, t)?
                .sum_on(-2, false, t)?
                .astype_on(x.dtype(), t)?,
        ))
    }
}

/// The routed experts' outputs, mixed or (for one decoded token) still
/// separate so that the mixture can be fused with what follows it.
enum MoeOutput {
    Mixed(Array),
    Routed {
        experts: Array,
        scores: Array,
        indices: Array,
        route_scale: f32,
    },
}
impl MoeOutput {
    fn mixed(&self, t: StreamOrDevice) -> Result<Array> {
        match self {
            Self::Mixed(out) => Ok(out.clone()),
            Self::Routed {
                experts,
                scores,
                indices,
                route_scale,
            } => super::metal::mix_routed_decode(experts, scores, indices, *route_scale, t),
        }
    }
}

/// `weight @ x` per row, for `x: [gates, rows, in]` and `weight` stored
/// row-major as `[gates, out, in]`. With the vector on the right of a matrix
/// in this layout MLX splits each output's sum over many threads. For the
/// `[in, out]` layout it adds `in / 8` products in sequence on each of eight
/// lanes, 55 us per call at `in = 4608`. The two orders round differently,
/// so single-token decode and verification rows must both come through here.
pub(super) fn row_major_matmul(x: &Array, weight: &Array, t: StreamOrDevice) -> Result<Array> {
    Ok(weight
        .expand_dims_on(1, t)?
        .matmul_on(&x.expand_dims_on(-1, t)?, t)?
        .squeeze_on(-1, t)?)
}

/// Attention of one decode row over the keys visible to it, with MLX's fused
/// single-query kernel. `q: [1, 1, heads, d]`, `k, v: [1, kv_heads, keys, d]`,
/// `bias: [1, heads, 1, extent]` indexed by backward distance and
/// `sinks: [1, heads, 1]`; returns `[1, heads, 1, d]`.
///
/// The kernel adds a float mask to the scores and takes one sink logit per
/// head, which for a single query is the per-query sink. Past the first
/// kilobyte of keys it splits them into blocks reduced in parallel; the
/// separate score, softmax and value steps added `keys / 8` products in
/// sequence for the value product and cost 6.5 ms per token at 28K keys.
#[cfg(test)]
pub(super) fn fused_row_attention_for_tests(
    q: &Array,
    k: &Array,
    v: &Array,
    bias: &Array,
    sinks: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    fused_row_attention(q, k, v, bias, sinks, t)
}

fn fused_row_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    bias: &Array,
    sinks: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    let (heads, d, keys) = (q.shape_at(2), q.shape_at(3), k.shape_at(2));
    // `bias` is indexed by distance from the far end of the extent, so key
    // `j`, `keys - 1 - j` back, reads the tail in key order. Beyond the
    // extent it adds zero.
    let extent = bias.shape_at(3);
    let band = keys.min(extent);
    let near = slice(bias, 3, extent - band, extent, t)?;
    let mask = if keys > band {
        cat(
            &[
                &Array::zeros((1, heads, 1, keys - band), bias.dtype())?,
                &near,
            ],
            -1,
            t,
        )?
    } else {
        near
    };
    let sinks = sinks.reshape_on((heads,), t)?.astype_on(q.dtype(), t)?;
    Ok(mlx::fast::scaled_dot_product_attention_on(
        &q.transpose_axes_on([0, 2, 1, 3], t)?,
        k,
        v,
        (d as f32).powf(-0.5),
        "",
        Some(&mask),
        Some(&sinks),
        t,
    )?)
}

/// Keep one vector per batch item during verification. Putting the shared
/// matrix on the left prevents MLX from folding the token dimension into GEMM.
fn row_matmul(x: &Array, weight: &Array, t: StreamOrDevice) -> Result<Array> {
    if !VERIFY_ROWS.get() || x.shape_at(-2) == 1 {
        return Ok(x.matmul_on(weight, t)?);
    }
    let transposed = if weight.ndim() == 2 {
        weight.transpose_on(t)?
    } else {
        weight.transpose_axes_on([0, 2, 1], t)?
    };
    let w = transposed.expand_dims_on(-3, t)?;
    Ok(w.matmul_on(&x.expand_dims_on(-1, t)?, t)?
        .squeeze_on(-1, t)?)
}

struct MessageGate {
    down: Linear,
    forget: Linear,
    gain: Linear,
    dtype: Dtype,
}
impl MessageGate {
    fn load(l: &Loader, p: &str) -> Result<Self> {
        Ok(Self {
            down: Linear::from_loader(l, &format!("{p}.down_proj"))?,
            forget: Linear::from_loader(l, &format!("{p}.forget_gate_proj"))?,
            gain: Linear::from_loader(l, &format!("{p}.message_gate_proj"))?,
            dtype: l.tensor(&format!("{p}.down_proj.weight"))?.dtype(),
        })
    }
    fn forward(&self, x: &Array, message: &Array, t: StreamOrDevice) -> Result<Array> {
        let input = cat(&[&norm(x, t)?, &norm(message, t)?], -1, t)?.astype_on(self.dtype, t)?;
        let state = self
            .down
            .forward_on(&input, t)?
            .astype_on(Dtype::Float32, t)?;
        let forget = self.forget.forward_on(&state, t)?;
        let gain = self.gain.forward_on(&state, t)?;
        if x.shape_at(0) == 1
            && x.dtype() == Dtype::Float32
            && message.dtype() == Dtype::Float32
            && forget.dtype() == Dtype::Float32
            && gain.dtype() == Dtype::Float32
            && [message, &forget, &gain]
                .iter()
                .all(|v| v.shape() == x.shape())
            && is_gpu(t)
        {
            return super::metal::mix_message(x, message, &forget, &gain, t);
        }
        let forget = forget.tanh_on(t)?;
        let gain = gain.tanh_on(t)?;
        Ok(scalar(1.0)?
            .try_sub_on(&forget, t)?
            .try_mul_on(x, t)?
            .try_add_on(&gain.try_mul_on(message, t)?, t)?)
    }
}

enum LoopGates {
    Separate(Vec<MessageGate>),
    Packed {
        down: Array,
        up: Array,
        /// `down` with each gate's matrix stored row-major as `[out, in]`.
        down_rows: Array,
    },
}

impl LoopGates {
    fn input_messages(
        &self,
        inputs: &[Array],
        messages: &Array,
        t: StreamOrDevice,
    ) -> Result<Vec<Array>> {
        match self {
            Self::Separate(gates) => (0..3)
                .map(|i| {
                    gates[i as usize].forward(
                        &inputs[i as usize],
                        &slice(messages, 2, i, i + 1, t)?.squeeze_on(2, t)?,
                        t,
                    )
                })
                .collect(),
            Self::Packed {
                down,
                up,
                down_rows,
            } => {
                let values = shape::stack_on(&inputs.iter().collect::<Vec<_>>(), 0, t)?;
                let messages = slice(messages, 2, 0, 3, t)?.transpose_axes_on([2, 0, 1, 3], t)?;
                let output = Self::project(
                    &values,
                    &messages,
                    &slice(down, 0, 0, 3, t)?,
                    &slice(up, 0, 0, 3, t)?,
                    &slice(down_rows, 0, 0, 3, t)?,
                    t,
                )?;
                (0..3)
                    .map(|i| Ok(slice(&output, 0, i, i + 1, t)?.squeeze_on(0, t)?))
                    .collect()
            }
        }
    }

    fn mlp_message(&self, input: &Array, messages: &Array, t: StreamOrDevice) -> Result<Array> {
        let message = slice(messages, 2, 3, 4, t)?.squeeze_on(2, t)?;
        match self {
            Self::Separate(gates) => gates[3].forward(input, &message, t),
            Self::Packed {
                down,
                up,
                down_rows,
            } => Ok(Self::project(
                &input.expand_dims_on(0, t)?,
                &message.expand_dims_on(0, t)?,
                &slice(down, 0, 3, 4, t)?,
                &slice(up, 0, 3, 4, t)?,
                &slice(down_rows, 0, 3, 4, t)?,
                t,
            )?
            .squeeze_on(0, t)?),
        }
    }

    fn project(
        x: &Array,
        msg: &Array,
        down: &Array,
        up: &Array,
        down_rows: &Array,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let (gates, b, s, h) = (x.shape_at(0), x.shape_at(1), x.shape_at(2), x.shape_at(3));
        let input = if b == 1
            && s <= 8
            && h <= 4096
            && x.dtype() == Dtype::Float32
            && msg.dtype() == Dtype::Float32
            && is_gpu(t)
        {
            super::metal::rms_pair_cast(x, msg, down.dtype(), t)?
        } else if b == 1 && s <= 8 && is_gpu(t) {
            let both = cat(&[x, msg], -1, t)?.reshape_on((gates, s, 2, h), t)?;
            norm(&both, t)?.reshape_on((gates, s, 2 * h), t)?
        } else {
            cat(&[&norm(x, t)?, &norm(msg, t)?], -1, t)?.reshape_on((gates, b * s, 2 * h), t)?
        }
        .astype_on(down.dtype(), t)?;
        let low = if b == 1 && s <= 8 && is_gpu(t) {
            row_major_matmul(&input, down_rows, t)?
        } else {
            row_matmul(&input, down, t)?
        }
        .astype_on(Dtype::Float32, t)?;
        let high = row_matmul(&low, up, t)?.reshape_on((gates, b, s, 2 * h), t)?;
        if b == 1
            && s <= 8
            && is_gpu(t)
            && x.dtype() == Dtype::Float32
            && msg.dtype() == Dtype::Float32
        {
            return super::metal::mix_packed_messages(x, msg, &high, t);
        }
        let gain = slice(&high, 3, 0, h, t)?.tanh_on(t)?;
        let forget = slice(&high, 3, h, 2 * h, t)?.tanh_on(t)?;
        Ok(scalar(1.0)?
            .try_sub_on(&forget, t)?
            .try_mul_on(x, t)?
            .try_add_on(&gain.try_mul_on(msg, t)?, t)?)
    }
}

#[derive(Clone, Copy)]
enum ConvolutionKind {
    Static,
    Dynamic,
    PackedDynamic,
}

struct Convolution {
    weight: Array,
    dynamic: Option<Linear>,
    heads: i32,
    kernel: i32,
}
impl Convolution {
    fn load(l: &Loader, p: &str, heads: i32, kernel: i32, kind: ConvolutionKind) -> Result<Self> {
        let weight_name = match kind {
            ConvolutionKind::Static => "conv1d.weight",
            ConvolutionKind::Dynamic | ConvolutionKind::PackedDynamic => "static_weight",
        };
        Ok(Self {
            weight: l.tensor(&format!("{p}.{weight_name}"))?.clone(),
            dynamic: if matches!(kind, ConvolutionKind::Dynamic) {
                Some(Linear::from_loader(l, &format!("{p}.weight_proj"))?)
            } else {
                None
            },
            heads,
            kernel,
        })
    }
    fn forward(
        &self,
        x: &Array,
        dynamic_input: &Array,
        coefficients: Option<&Array>,
        state: &Array,
        lens: &[i32],
        t: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        let (b, s, c) = (x.shape_at(0), x.shape_at(1), x.shape_at(2));
        let k = self.kernel;
        if b == 1
            && lens[0] == s
            && k == 4
            && self.dynamic.is_none()
            && coefficients.is_none()
            && x.dtype() == Dtype::Bfloat16
            && self.weight.dtype() == Dtype::Bfloat16
            && self.weight.shape().as_slice() == [c, 4, 1]
            && is_gpu(t)
        {
            return static_conv4(x, &self.weight, state, t);
        }
        let dynamic = if let Some(coefficients) = coefficients {
            Some(coefficients.clone())
        } else {
            self.dynamic
                .as_ref()
                .map(|p| p.forward_on(dynamic_input, t))
                .transpose()?
        };
        if let Some(dynamic) = dynamic.as_ref().filter(|_| {
            b == 1
                && lens[0] == s
                && k == 4
                && x.dtype() == Dtype::Bfloat16
                && self.weight.dtype() == Dtype::Bfloat16
                && self.weight.shape().as_slice() == [4, c]
                && is_gpu(t)
                && FAST_OQ.get()
        }) {
            return super::metal::dynamic_conv4(x, &self.weight, dynamic, state, self.heads, t);
        }
        let window = cat(&[state, &x.astype_on(Dtype::Float32, t)?], 1, t)?;
        let out = if let Some(dynamic) = dynamic {
            let dynamic = dynamic
                .astype_on(Dtype::Float32, t)?
                .reshape_on(&[b, s, self.heads, 1, k][..], t)?;
            let static_w = self
                .weight
                .astype_on(Dtype::Float32, t)?
                .transpose_on(t)?
                .reshape_on((self.heads, -1, k), t)?;
            let weights = dynamic.try_add_on(&static_w, t)?;
            let lagged: Vec<Array> = (0..k)
                .map(|lag| slice(&window, 1, k - 1 - lag, k - 1 - lag + s, t))
                .collect::<Result<_>>()?;
            let windows = shape::stack_on(&lagged.iter().collect::<Vec<_>>(), -1, t)?
                .reshape_on(&[b, s, self.heads, -1, k][..], t)?;
            let out = windows
                .try_mul_on(&weights, t)?
                .sum_on(-1, false, t)?
                .reshape_on((b, s, c), t)?;
            scale(
                &out.try_add_on(&x.astype_on(Dtype::Float32, t)?, t)?,
                1.0 / ((2 * k + 1) as f32).sqrt(),
                t,
            )?
        } else {
            let out = mlx::ops::conv::conv1d_on(
                &window.astype_on(self.weight.dtype(), t)?,
                &self.weight,
                1,
                0,
                1,
                c,
                t,
            )?
            .astype_on(Dtype::Float32, t)?;
            scale(
                &out.try_add_on(&x.astype_on(Dtype::Float32, t)?, t)?,
                1.0 / ((k + 1) as f32).sqrt(),
                t,
            )?
        };
        // For right-padded or inactive rows, preserve precisely the valid suffix.
        let state = if b == 1 {
            slice(&window, 1, lens[0], lens[0] + k - 1, t)?
        } else {
            let positions: Vec<i32> = lens
                .iter()
                .flat_map(|n| (0..k - 1).map(move |i| n + i))
                .collect();
            let positions: Array = (positions.as_slice(), &[b, k - 1, 1][..]).try_into()?;
            indexing::take_along_axis_on(&window, &positions, 1, t)?
        };
        Ok((out.astype_on(x.dtype(), t)?, state))
    }

    /// The expert mixture of one decoded token and this residual convolution
    /// in one dispatch, where [`Self::residual`] would take its fast path.
    #[allow(clippy::too_many_arguments)]
    fn mix_routed_residual(
        &self,
        experts: &Array,
        scores: &Array,
        indices: &Array,
        route_scale: f32,
        state: &Array,
        lens: &[i32],
        residual: &Array,
        output: Dtype,
        t: StreamOrDevice,
    ) -> Result<Option<(Array, Array)>> {
        let channels = experts.shape_at(-1);
        if !(FAST_OQ.get()
            && experts.ndim() == 4
            && experts.shape_at(0) == 1
            && experts.shape_at(1) == 1
            && experts.shape_at(2) == 8
            && lens == [1]
            && self.kernel == 4
            && experts.dtype() == Dtype::Bfloat16
            && self.dynamic.is_none()
            && self.weight.dtype() == Dtype::Bfloat16
            && self.weight.shape().as_slice() == [channels, 4, 1]
            && is_gpu(t))
        {
            return Ok(None);
        }
        Ok(Some(super::metal::mix_routed_conv4_decode(
            experts,
            scores,
            indices,
            route_scale,
            &self.weight,
            state,
            residual,
            output,
            t,
        )?))
    }

    fn residual(
        &self,
        x: &Array,
        state: &Array,
        lens: &[i32],
        residual: Option<&Array>,
        output: Dtype,
        t: StreamOrDevice,
    ) -> Result<(Array, Array)> {
        if FAST_OQ.get()
            && x.shape_at(0) == 1
            && lens[0] == x.shape_at(1)
            && self.kernel == 4
            && x.dtype() == Dtype::Bfloat16
            && self.dynamic.is_none()
            && self.weight.dtype() == Dtype::Bfloat16
            && self.weight.shape().as_slice() == [x.shape_at(2), 4, 1]
            && is_gpu(t)
        {
            return super::metal::static_conv4_with_residual(
                x,
                &self.weight,
                state,
                residual,
                output,
                t,
            );
        }
        let (value, state) = self.forward(x, x, None, state, lens, t)?;
        let value = if let Some(residual) = residual {
            residual
                .astype_on(Dtype::Float32, t)?
                .try_add_on(&value.astype_on(Dtype::Float32, t)?, t)?
        } else {
            value
        };
        Ok((value.astype_on(output, t)?, state))
    }
}

enum AttentionInputs {
    Separate(Box<[Linear; 6]>),
    PackedLocal {
        packed: Linear,
        separate: Box<[Linear; 6]>,
    },
    PackedGlobal {
        query: Linear,
        key: Linear,
        value: Linear,
    },
}

struct Attention {
    projections: AttentionInputs,
    o: Linear,
    q_scale: Array,
    r_scale: Array,
    relative: Array,
    reversed_relative: Array,
    k_conv: Convolution,
    v_conv: Convolution,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    relative_dim: i32,
    window: i32,
    eps: f32,
    q_clip: f32,
    r_clip: f32,
    log_floor: i32,
    log_alpha: f32,
}
impl Attention {
    fn load(l: &Loader, p: &str, c: &TextConfig, i: usize) -> Result<Self> {
        let (heads, kv_heads, dim) = c.attention_dims(i);
        let local = c.is_local(i);
        let linear = |n| Linear::from_loader(l, &format!("{p}.{n}_proj"));
        let projections = if l.contains(&format!("{p}.packed_input.weight")) {
            let packed = Linear::from_loader(l, &format!("{p}.packed_input"))?;
            let q = (heads * dim) as usize;
            let kv = (kv_heads * dim) as usize;
            let widths = [q, kv, kv, (heads * c.d_rel) as usize, q, heads as usize];
            let separate = packed.split_quantized_outputs(&widths, p)?;
            let separate = separate
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid Inkling projection count"))?;
            AttentionInputs::PackedLocal {
                packed,
                separate: Box::new(separate),
            }
        } else if l.contains(&format!("{p}.packed_query.weight")) {
            AttentionInputs::PackedGlobal {
                query: Linear::from_loader(l, &format!("{p}.packed_query"))?,
                key: Linear::from_loader(l, &format!("{p}.packed_k"))?,
                value: Linear::from_loader(l, &format!("{p}.packed_v"))?,
            }
        } else {
            AttentionInputs::Separate(Box::new([
                linear("q")?,
                linear("k")?,
                linear("v")?,
                linear("r")?,
                linear("gate")?,
                linear("sink")?,
            ]))
        };
        let relative = l.tensor(&format!("{p}.rel_logits_proj.proj"))?.clone();
        let conv_kind = if local {
            ConvolutionKind::Static
        } else if matches!(&projections, AttentionInputs::PackedGlobal { .. }) {
            ConvolutionKind::PackedDynamic
        } else {
            ConvolutionKind::Dynamic
        };
        Ok(Self {
            projections,
            o: linear("o")?,
            q_scale: l.tensor(&format!("{p}.q_l2_clip.scale"))?.clone(),
            r_scale: l.tensor(&format!("{p}.rel_l2_clip.scale"))?.clone(),
            relative: relative.clone(),
            reversed_relative: {
                let width = relative.shape_at(-1);
                let order: Vec<u32> = (0..width as u32).rev().collect();
                let order: Array = (order.as_slice(), &[width][..]).try_into()?;
                let reversed = indexing::take(&relative, &order, -1)?;
                mlx::transforms::eval(&[&reversed])?;
                reversed
            },
            k_conv: Convolution::load(
                l,
                &format!("{p}.k_sconv"),
                kv_heads,
                c.sconv_kernel_size,
                conv_kind,
            )?,
            v_conv: Convolution::load(
                l,
                &format!("{p}.v_sconv"),
                kv_heads,
                c.sconv_kernel_size,
                conv_kind,
            )?,
            heads,
            kv_heads,
            dim,
            relative_dim: c.d_rel,
            window: if local { c.sliding_window_size } else { 0 },
            eps: c.rms_norm_eps,
            q_clip: c.qk_l2_clip_threshold,
            r_clip: c.rel_l2_clip_threshold,
            log_floor: c.log_scaling_n_floor,
            log_alpha: c.log_scaling_alpha,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        inputs: &[Array],
        shared_input: bool,
        states: &mut [Array],
        kv: &mut KVCache,
        lens: &[i32],
        conv_inputs: Option<&mut Vec<Array>>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let x = &inputs[0];
        let (b, s) = (x.shape_at(0), x.shape_at(1));
        let h = self.heads;
        let d = self.dim;
        let offsets = kv.offsets().to_vec();
        let dq = h * d;
        let dkv = self.kv_heads * d;
        let rel = h * self.relative_dim;
        let (q, k, v, r, gate, sinks, dynamic_k, dynamic_v) = match &self.projections {
            AttentionInputs::PackedLocal { packed, .. } if shared_input => {
                let y = packed.forward_on(x, t)?;
                let (q, k, v, r, gate, sink) = (
                    slice(&y, 2, 0, dq, t)?,
                    slice(&y, 2, dq, dq + dkv, t)?,
                    slice(&y, 2, dq + dkv, dq + 2 * dkv, t)?,
                    slice(&y, 2, dq + 2 * dkv, dq + 2 * dkv + rel, t)?,
                    slice(&y, 2, dq + 2 * dkv + rel, 2 * dq + 2 * dkv + rel, t)?,
                    slice(&y, 2, 2 * dq + 2 * dkv + rel, 2 * dq + 2 * dkv + rel + h, t)?,
                );
                (q, k, v, r, gate, sink, None, None)
            }
            AttentionInputs::PackedGlobal { query, key, value } => {
                let q = query.forward_on(x, t)?;
                let k = key.forward_on(&inputs[1], t)?;
                let v = value.forward_on(&inputs[2], t)?;
                (
                    slice(&q, 2, 0, dq, t)?,
                    slice(&k, 2, 0, dkv, t)?,
                    slice(&v, 2, 0, dkv, t)?,
                    slice(&q, 2, dq, dq + rel, t)?,
                    slice(&q, 2, dq + rel, 2 * dq + rel, t)?,
                    slice(&q, 2, 2 * dq + rel, 2 * dq + rel + h, t)?,
                    Some(slice(&k, 2, dkv, k.shape_at(2), t)?),
                    Some(slice(&v, 2, dkv, v.shape_at(2), t)?),
                )
            }
            AttentionInputs::Separate(parts)
            | AttentionInputs::PackedLocal {
                separate: parts, ..
            } => (
                parts[0].forward_on(x, t)?,
                parts[1].forward_on(&inputs[1], t)?,
                parts[2].forward_on(&inputs[2], t)?,
                parts[3].forward_on(x, t)?,
                parts[4].forward_on(x, t)?,
                parts[5].forward_on(x, t)?,
                None,
                None,
            ),
        };
        if let Some(inputs) = conv_inputs {
            inputs.extend([k.clone(), v.clone()]);
        }
        let (k, kstate) =
            self.k_conv
                .forward(&k, &inputs[1], dynamic_k.as_ref(), &states[0], lens, t)?;
        let (v, vstate) =
            self.v_conv
                .forward(&v, &inputs[2], dynamic_v.as_ref(), &states[1], lens, t)?;
        states[0] = kstate;
        states[1] = vstate;
        let mut q = l2_clip(
            &q.reshape_on((b, s, h, d), t)?,
            Some(&self.q_scale),
            self.q_clip,
            t,
        )?;
        let k = l2_clip(
            &k.reshape_on((b, s, self.kv_heads, d), t)?,
            None,
            self.q_clip,
            t,
        )?
        .transpose_axes_on([0, 2, 1, 3], t)?;
        let v = v.reshape_on((b, s, self.kv_heads, d), t)?;
        let v = if FAST_OQ.get()
            && b == 1
            && (s == 1 || VERIFY_ROWS.get())
            && v.dtype() == Dtype::Bfloat16
        {
            mlx::fast::rms_norm_on(&v, None, self.eps, t)?
        } else {
            mlx::fast::rms_norm_on(&v.astype_on(Dtype::Float32, t)?, None, self.eps, t)?
                .astype_on(x.dtype(), t)?
        }
        .transpose_axes_on([0, 2, 1, 3], t)?;
        let mut r = l2_clip(
            &r.reshape_on((b, s, h, self.relative_dim), t)?,
            Some(&self.r_scale),
            self.r_clip,
            t,
        )?;
        if self.window == 0 && offsets.iter().any(|o| o + s > self.log_floor) {
            let tau: Vec<f32> = offsets
                .iter()
                .flat_map(|o| {
                    (0..s).map(move |j| {
                        1.0 + self.log_alpha
                            * (((o + j + 1) as f32 / self.log_floor as f32).max(1.0)).ln()
                    })
                })
                .collect();
            let tau: Array = (tau.as_slice(), &[b, s, 1, 1][..]).try_into()?;
            q = q
                .astype_on(Dtype::Float32, t)?
                .try_mul_on(&tau, t)?
                .astype_on(x.dtype(), t)?;
            r = r
                .astype_on(Dtype::Float32, t)?
                .try_mul_on(&tau, t)?
                .astype_on(x.dtype(), t)?;
        }
        let bias = r
            .matmul_on(&self.relative.astype_on(r.dtype(), t)?, t)?
            .transpose_axes_on([0, 2, 1, 3], t)?;
        // The same products with the extent reversed, for the fused rows.
        let reversed_bias = || -> Result<Array> {
            Ok(
                r.matmul_on(&self.reversed_relative.astype_on(r.dtype(), t)?, t)?
                    .transpose_axes_on([0, 2, 1, 3], t)?,
            )
        };
        let (mut k, mut v) = kv.update_and_fetch_on(&k, &v, lens, t)?;
        // The cache returns each row from its base on; a sliding-window
        // layer's cache has dropped what came before.
        let bases = kv.bases().to_vec();
        // The first query sees the oldest key of this chunk's visible window.
        // Keep the underlying cache layout for snapshot and prefix consumers.
        let key_start = if b == 1 && self.window > 0 && lens[0] > 0 {
            (offsets[0] - self.window + 1).max(bases[0])
        } else if b == 1 {
            bases[0]
        } else {
            0
        };
        if b == 1 && key_start > bases[0] {
            let end = k.shape_at(2);
            k = slice(&k, 2, key_start - bases[0], end, t)?;
            v = slice(&v, 2, key_start - bases[0], end, t)?;
        }
        let n = k.shape_at(2);
        let extent = bias.shape_at(-1);
        let scores = || -> Result<Array> {
            let q = q
                .transpose_axes_on([0, 2, 1, 3], t)?
                .astype_on(Dtype::Float32, t)?
                .reshape_on(&[b, self.kv_heads, h / self.kv_heads, s, d][..], t)?;
            let keys = k
                .astype_on(Dtype::Float32, t)?
                .transpose_axes_on([0, 1, 3, 2], t)?
                .expand_dims_on(2, t)?;
            scale(&q.matmul_on(&keys, t)?, (d as f32).powf(-0.5), t)
        };
        // The fused row kernel takes the sinks in the query's type; going
        // through float32 and back is exact but costs two dispatches a layer.
        let row_sinks = sinks.transpose_axes_on([0, 2, 1], t)?;
        let sinks = sinks
            .astype_on(Dtype::Float32, t)?
            .transpose_axes_on([0, 2, 1], t)?;

        let regular = b == 1 && lens[0] == s && n == offsets[0] + s - key_start && is_gpu(t);
        // One fused MLX call per row: scores, bias, sink, softmax and the
        // value product together. A verification row passes exactly what
        // the decode step at its position does, so both round alike.
        let fused_rows = FAST_OQ.get()
            && regular
            && (s == 1 || VERIFY_ROWS.get())
            && d == 128
            && h % self.kv_heads == 0
            && h / self.kv_heads <= 32
            && [&q, &k, &v, &bias]
                .iter()
                .all(|a| a.dtype() == Dtype::Bfloat16);
        // 8-bit global K/V: the fused rows read the packed buffers directly.
        let int8_sides = kv
            .int8()
            .and_then(|storage| storage.sides())
            .map(|(keys, values)| (keys.clone(), values.clone()));
        let out = if fused_rows {
            let bias = reversed_bias()?;
            let mut outputs = Vec::with_capacity(s as usize);
            for i in 0..s {
                let end = offsets[0] + i + 1;
                let start = if self.window > 0 {
                    (end - self.window).max(0)
                } else {
                    0
                };
                outputs.push(match &int8_sides {
                    Some((keys, values)) => super::metal::int8_row_attention(
                        &slice(&q, 1, i, i + 1, t)?,
                        keys,
                        values,
                        start,
                        end,
                        &slice(&bias, 2, i, i + 1, t)?,
                        &slice(&row_sinks, 2, i, i + 1, t)?,
                        t,
                    )?,
                    None => fused_row_attention(
                        &slice(&q, 1, i, i + 1, t)?,
                        &slice(&k, 2, start - key_start, end - key_start, t)?,
                        &slice(&v, 2, start - key_start, end - key_start, t)?,
                        &slice(&bias, 2, i, i + 1, t)?,
                        &slice(&row_sinks, 2, i, i + 1, t)?,
                        t,
                    )?,
                });
            }
            if s == 1 {
                outputs.remove(0)
            } else {
                cat(&outputs.iter().collect::<Vec<_>>(), 2, t)?
            }
        } else if regular && s > 1 && VERIFY_ROWS.get() {
            let mut outputs = Vec::with_capacity(s as usize);
            for i in 0..s {
                let end = offsets[0] + i + 1;
                let start = if self.window > 0 {
                    (end - self.window).max(0)
                } else {
                    0
                };
                let nk = end - start;
                let vi = slice(&v, 2, start - key_start, end - key_start, t)?;
                let weights = {
                    let logits = {
                        let qi = slice(&q, 1, i, i + 1, t)?;
                        let ki = slice(&k, 2, start - key_start, end - key_start, t)?;
                        let ri = slice(&bias, 2, i, i + 1, t)?;
                        let si = slice(&sinks, 2, i, i + 1, t)?;
                        if d == 128
                            && nk >= 9
                            && qi.dtype() == Dtype::Bfloat16
                            && ki.dtype() == Dtype::Bfloat16
                            && h % self.kv_heads == 0
                            && (m5_available() || FAST_OQ.get())
                        {
                            super::metal::decode_relative_logits(
                                &qi,
                                &ki,
                                &ri,
                                &si,
                                self.window,
                                t,
                            )?
                        } else {
                            let qi = qi
                                .transpose_axes_on([0, 2, 1, 3], t)?
                                .astype_on(Dtype::Float32, t)?
                                .reshape_on(&[1, self.kv_heads, h / self.kv_heads, 1, d][..], t)?;
                            let ki = ki
                                .astype_on(Dtype::Float32, t)?
                                .transpose_axes_on([0, 1, 3, 2], t)?
                                .expand_dims_on(2, t)?;
                            let scores = scale(&qi.matmul_on(&ki, t)?, (d as f32).powf(-0.5), t)?;
                            relative_logits(&scores, &ri, &si, self.window, t)?
                        }
                    };
                    slice(&logits.softmax_on(-1, true, t)?, 4, 0, nk, t)?
                        .astype_on(vi.dtype(), t)?
                };
                outputs.push(
                    weights
                        .matmul_on(&vi.expand_dims_on(2, t)?, t)?
                        .reshape_on((1, h, 1, d), t)?,
                );
            }
            cat(&outputs.iter().collect::<Vec<_>>(), 2, t)?
        } else if regular
            && s >= 32
            && q.dtype() == Dtype::Bfloat16
            && bias.dtype() == Dtype::Bfloat16
            && FAST_OQ.get()
            && (self.window == 0 || self.window == extent)
        {
            // The fused kernel reads the bias by backward distance and one
            // sink per query. A mask with one value per (query, key) is
            // 2.75 GB per layer for a 2048-token block over 28K keys, and
            // writing and reading it was a fifth of a 30K prefill.
            mlx::fast::scaled_dot_product_attention_on(
                &q.transpose_axes_on([0, 2, 1, 3], t)?,
                &k,
                &v,
                (d as f32).powf(-0.5),
                if self.window > 0 {
                    "band_window"
                } else {
                    "band"
                },
                Some(&bias),
                Some(&sinks.astype_on(q.dtype(), t)?),
                t,
            )?
        } else if regular && s >= 32 && q.dtype() == Dtype::Bfloat16 && FAST_OQ.get() {
            let mask = super::metal::relative_mask(&bias, &sinks, n, self.window, t)?;
            // The sink is query-dependent. Represent it as a zero-valued
            // extra key whose additive mask supplies the learned score.
            let zero = Array::zeros((1, self.kv_heads, 1, d), k.dtype())?;
            mlx::fast::scaled_dot_product_attention_on(
                &q.transpose_axes_on([0, 2, 1, 3], t)?,
                &cat(&[&k, &zero], 2, t)?,
                &cat(&[&v, &zero], 2, t)?,
                (d as f32).powf(-0.5),
                "",
                Some(&mask),
                None,
                t,
            )?
        } else {
            let logits = if regular
                && s == 1
                && d == 128
                && n >= 9
                && q.dtype() == Dtype::Bfloat16
                && k.dtype() == Dtype::Bfloat16
                && h % self.kv_heads == 0
                && (m5_available() || FAST_OQ.get())
            {
                super::metal::decode_relative_logits(&q, &k, &bias, &sinks, self.window, t)?
            } else if regular {
                relative_logits(&scores()?, &bias, &sinks, self.window, t)?
            } else {
                let mut distances = Vec::with_capacity((b * s * n) as usize);
                let mut visible = Vec::with_capacity((b * s * n) as usize);
                let mut has_bias = Vec::with_capacity((b * s * n) as usize);
                for (row, offset) in offsets.iter().enumerate() {
                    for j in 0..s {
                        for key in 0..n {
                            let absolute_key = if b == 1 { key_start } else { bases[row] } + key;
                            let delta = offset + j - absolute_key;
                            distances.push(delta.clamp(0, extent - 1));
                            has_bias.push(delta < extent);
                            visible.push(
                                delta >= 0
                                    && absolute_key < offset + lens[row]
                                    && (self.window == 0 || delta < self.window),
                            );
                        }
                    }
                }
                let distances: Array = (distances.as_slice(), &[b, 1, s, n][..]).try_into()?;
                let visible: Array = (visible.as_slice(), &[b, 1, s, n][..]).try_into()?;
                let has_bias: Array = (has_bias.as_slice(), &[b, 1, s, n][..]).try_into()?;
                let bias = indexing::take_along_axis_on(
                    &bias,
                    &distances.broadcast_to_on((b, h, s, n), t)?,
                    -1,
                    t,
                )?
                .astype_on(Dtype::Float32, t)?;
                let bias = has_bias.where_on(&bias, &scalar(0.0)?, t)?;
                let mask = visible
                    .where_on(&bias, &scalar(f32::NEG_INFINITY)?, t)?
                    .reshape_on(&[b, self.kv_heads, h / self.kv_heads, s, n][..], t)?;

                let scores = scores()?.try_add_on(&mask, t)?;
                let sinks =
                    sinks.reshape_on(&[b, self.kv_heads, h / self.kv_heads, s, 1][..], t)?;
                cat(&[&scores, &sinks], -1, t)?
            };
            let weights = logits.softmax_on(-1, true, t)?;
            let weights = slice(&weights, 4, 0, n, t)?.astype_on(v.dtype(), t)?;
            weights.matmul_on(&v.expand_dims_on(2, t)?, t)?
        };
        if FAST_OQ.get()
            && regular
            && (s == 1 || VERIFY_ROWS.get())
            && gate.dtype() == Dtype::Bfloat16
            && out.dtype() == Dtype::Bfloat16
        {
            let gated = super::metal::gate_attention(&out.reshape_on((b, h, s, d), t)?, &gate, t)?;
            return self.o.forward_on(&gated, t);
        }
        let out = out
            .reshape_on((b, h, s, d), t)?
            .transpose_axes_on([0, 2, 1, 3], t)?
            .reshape_on((b, s, h * d), t)?
            .astype_on(Dtype::Float32, t)?;
        self.o.forward_on(
            &out.try_mul_on(&gate.astype_on(Dtype::Float32, t)?.sigmoid_on(t)?, t)?
                .astype_on(gate.dtype(), t)?,
            t,
        )
    }
}

/// An RMSNorm that also keeps its scale in float32. MLX promotes the scale
/// to a float32 input's type on every call, one extra dispatch each time;
/// the promoted copy is exact, so passing it in directly computes the same.
struct Norm {
    scale: RmsNorm,
    float32: RmsNorm,
}
impl Norm {
    fn load(l: &Loader, p: &str, eps: f32) -> Result<Self> {
        let scale = RmsNorm::from_loader(l, p, eps)?;
        let weight = scale.weight().astype(Dtype::Float32)?;
        mlx::transforms::eval(&[&weight])?;
        Ok(Self {
            float32: RmsNorm::new(weight, eps),
            scale,
        })
    }
    fn forward_on(&self, x: &Array, t: StreamOrDevice) -> Result<Array> {
        Ok(if x.dtype() == Dtype::Float32 {
            self.float32.forward_on(x, t)?
        } else {
            self.scale.forward_on(x, t)?
        })
    }
    /// The norm of a float32 `x` cast to `dtype`, in one dispatch on the GPU.
    fn forward_cast_on(&self, x: &Array, dtype: Dtype, t: StreamOrDevice) -> Result<Array> {
        if x.dtype() == Dtype::Float32 && x.shape_at(-1) <= 4096 && is_gpu(t) {
            return super::metal::rms_cast(
                x,
                Some(self.float32.weight()),
                self.float32.eps(),
                dtype,
                t,
            );
        }
        Ok(self.forward_on(x, t)?.astype_on(dtype, t)?)
    }
    /// [`Self::forward_cast_on`] and, from the same dispatch, its result
    /// widened back to float32 when the fused kernel applies.
    fn forward_cast_dual_on(
        &self,
        x: &Array,
        dtype: Dtype,
        t: StreamOrDevice,
    ) -> Result<(Array, Option<Array>)> {
        if x.dtype() == Dtype::Float32 && x.shape_at(-1) <= 4096 && is_gpu(t) {
            let (narrow, wide) = super::metal::rms_cast_dual(
                x,
                Some(self.float32.weight()),
                self.float32.eps(),
                dtype,
                t,
            )?;
            return Ok((narrow, Some(wide)));
        }
        Ok((self.forward_cast_on(x, dtype, t)?, None))
    }
}

pub(super) struct DecoderLayer {
    attention: Attention,
    attention_query: Option<Array>,
    mlp_query: Array,
    input_norms: Vec<Norm>,
    mlp_norm: Norm,
    gates: LoopGates,
    moe: Moe,
    attention_conv: Convolution,
    mlp_conv: Convolution,
    boundary: bool,
    conv_widths: [i32; 4],
}
impl DecoderLayer {
    pub fn load(
        l: &Loader,
        c: &TextConfig,
        i: usize,
        tables: Option<ActivationTables>,
    ) -> Result<Self> {
        let p = format!("model.language_model.layers.{i}");
        let local = c.is_local(i);
        let norm_names = if local {
            vec!["input_layernorm"]
        } else {
            vec![
                "q_input_layernorm",
                "k_input_layernorm",
                "v_input_layernorm",
            ]
        };
        let input_norms = norm_names
            .iter()
            .map(|n| Norm::load(l, &format!("{p}.{n}"), c.rms_norm_eps))
            .collect::<Result<_>>()?;
        let gates = if l.contains(&format!("{p}.packed_loop_down")) {
            let down = l.tensor(&format!("{p}.packed_loop_down"))?.clone();
            let down_rows = down.transpose_axes([0, 2, 1])?.contiguous(false)?;
            mlx::transforms::eval(&[&down_rows])?;
            LoopGates::Packed {
                down,
                up: l.tensor(&format!("{p}.packed_loop_out"))?.clone(),
                down_rows,
            }
        } else if c
            .loop_config
            .as_ref()
            .is_some_and(|q| (q.core_start..q.core_end).contains(&i))
        {
            LoopGates::Separate(
                ["q", "k", "v", "mlp"]
                    .iter()
                    .map(|n| MessageGate::load(l, &format!("{p}.{n}_message_gate")))
                    .collect::<Result<_>>()?,
            )
        } else {
            LoopGates::Separate(Vec::new())
        };
        let (_, kv, d) = c.attention_dims(i);
        Ok(Self {
            attention: Attention::load(l, &format!("{p}.self_attn"), c, i)?,
            attention_query: if i > 0 {
                Some(l.tensor(&format!("{p}.attention_input.query"))?.clone())
            } else {
                None
            },
            mlp_query: l.tensor(&format!("{p}.mlp_input.query"))?.clone(),
            input_norms,
            mlp_norm: Norm::load(l, &format!("{p}.post_attention_layernorm"), c.rms_norm_eps)?,
            gates,
            moe: Moe::load(l, &format!("{p}.mlp"), c, tables)?,
            attention_conv: Convolution::load(
                l,
                &format!("{p}.post_attention_convolution"),
                1,
                c.sconv_kernel_size,
                ConvolutionKind::Static,
            )?,
            mlp_conv: Convolution::load(
                l,
                &format!("{p}.post_mlp_convolution"),
                1,
                c.sconv_kernel_size,
                ConvolutionKind::Static,
            )?,
            boundary: i.is_multiple_of(c.n_layers_per_attn_res_block),
            conv_widths: [kv * d, kv * d, c.hidden_size, c.hidden_size],
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        blocks: &mut Vec<Array>,
        messages: Option<&Array>,
        kv: &mut KVCache,
        conv: &mut GatedDeltaCache,
        lens: &[i32],
        verify_input: Option<&mut Vec<Array>>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        let dtype = x.dtype();
        let normalized = if x.shape_at(-1) <= 4096 && is_gpu(t) {
            super::metal::rms_cast(x, None, 1e-6, Dtype::Float32, t)?
        } else {
            norm(&x.astype_on(Dtype::Float32, t)?, t)?
        };
        let attention_input = self
            .attention_query
            .as_ref()
            .map(|q| depth_attention(blocks, &normalized, q, t))
            .transpose()?;
        let mut inputs: Vec<Array> = Vec::with_capacity(3);
        for i in 0..3 {
            let input = match &attention_input {
                Some(a) => slice(
                    a,
                    2,
                    if self.input_norms.len() == 1 { 0 } else { i },
                    if self.input_norms.len() == 1 {
                        1
                    } else {
                        i + 1
                    },
                    t,
                )?
                .squeeze_on(2, t)?,
                None => x.clone(),
            };
            inputs.push(input);
        }
        if let Some(messages) = messages {
            inputs = self.gates.input_messages(&inputs, messages, t)?;
        }
        let shared_input = self.input_norms.len() == 1 && messages.is_none();
        for i in 0..inputs.len() {
            inputs[i] = if shared_input && i > 0 {
                inputs[0].clone()
            } else {
                self.input_norms[if self.input_norms.len() == 1 { 0 } else { i }]
                    .forward_cast_on(&inputs[i], dtype, t)?
            };
        }
        let mut start = 0;
        let mut states = Vec::new();
        for width in self.conv_widths {
            states.push(slice(conv.conv_state(), 2, start, start + width, t)?);
            start += width;
        }
        let mut conv_inputs = verify_input.as_ref().map(|_| Vec::with_capacity(4));
        let attn = self.attention.forward(
            &inputs,
            shared_input,
            &mut states,
            kv,
            lens,
            conv_inputs.as_mut(),
            t,
        )?;
        if let Some(inputs) = &mut conv_inputs {
            inputs.push(attn.clone());
        }
        let (residual, state) = self.attention_conv.residual(
            &attn,
            &states[2],
            lens,
            (!self.boundary).then_some(x),
            Dtype::Float32,
            t,
        )?;
        states[2] = state;
        if self.boundary {
            blocks.push(normalized);
        }
        let mut input =
            depth_attention(blocks, &norm(&residual, t)?, &self.mlp_query, t)?.squeeze_on(2, t)?;
        if let Some(m) = messages {
            input = self.gates.mlp_message(&input, m, t)?;
        }
        let (normed, wide) = self.mlp_norm.forward_cast_dual_on(&input, dtype, t)?;
        let moe = self.moe.forward(&normed, wide.as_ref(), t)?;
        if let (
            MoeOutput::Routed {
                experts,
                scores,
                indices,
                route_scale,
            },
            None,
        ) = (&moe, verify_input.as_ref())
        {
            if let Some((output, state)) = self.mlp_conv.mix_routed_residual(
                experts,
                scores,
                indices,
                *route_scale,
                &states[3],
                lens,
                &residual,
                dtype,
                t,
            )? {
                states[3] = state;
                conv.update_conv(cat(&states.iter().collect::<Vec<_>>(), -1, t)?);
                conv.advance(lens)?;
                return Ok(output);
            }
        }
        let mlp = moe.mixed(t)?;
        if let (Some(mut inputs), Some(verify_input)) = (conv_inputs, verify_input) {
            inputs.push(mlp.clone());
            verify_input
                .push(cat(&inputs.iter().collect::<Vec<_>>(), 2, t)?.astype_on(Dtype::Float32, t)?);
        }
        let (output, state) =
            self.mlp_conv
                .residual(&mlp, &states[3], lens, Some(&residual), dtype, t)?;
        states[3] = state;
        conv.update_conv(cat(&states.iter().collect::<Vec<_>>(), -1, t)?);
        conv.advance(lens)?;
        Ok(output)
    }
}
