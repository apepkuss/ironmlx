//! Single-request Metal kernels for Inkling attention, convolution and experts.
use super::ops::cap;
use anyhow::Result;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};
use std::sync::OnceLock;

pub(super) fn kernel(
    slot: &'static OnceLock<MetalKernel>,
    build: impl FnOnce() -> mlx::Result<MetalKernel>,
) -> Result<&'static MetalKernel> {
    if let Some(k) = slot.get() {
        return Ok(k);
    }
    let k = build()?;
    Ok(slot.get_or_init(|| k))
}

/// Align learned relative bias and append the per-query sink in a single write.
/// Scores already contain the FP32 QK scale. Softmax and the value product keep
/// their native MLX reduction order and the checkpoint's intermediate casts.
pub(super) fn relative_logits(
    scores: &Array,
    bias: &Array,
    sinks: &Array,
    window: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_relative_logits")
            .inputs(&["scores", "bias", "sinks"])
            .outputs(&["out"])
            .ensure_row_contiguous(true)
            .source(r#"
                const uint key = thread_position_in_grid.x;
                const uint q = thread_position_in_grid.y;
                const uint h = thread_position_in_grid.z;
                const uint n = scores_shape[4], queries = scores_shape[3];
                const size_t row = size_t(h) * queries + q;
                const int distance = int(n - queries + q) - int(key);
                float value;
                if (key == n) {
                    value = sinks[row];
                } else if (distance < 0 || (WINDOW > 0 && distance >= WINDOW)) {
                    value = -INFINITY;
                } else {
                    const uint extent = bias_shape[3];
                    const float r = uint(distance) < extent ? float(bias[row * extent + uint(distance)]) : 0.0f;
                    value = scores[row * n + key] + r;
                }
                out[row * (n + 1) + key] = value;
            "#)
            .build()
    })?;
    let mut dims = scores.shape().as_slice().to_vec();
    dims[4] += 1;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[scores, bias, sinks])
        .output_shapes(&[Shape::from(dims.as_slice())])
        .output_dtypes(&[Dtype::Float32])
        .template_int("WINDOW", window)
        .grid(dims[4], dims[3], dims[1] * dims[2])
        .threadgroup(128, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

pub(super) fn relative_mask(
    bias: &Array,
    sinks: &Array,
    keys: i32,
    window: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_relative_mask")
            .inputs(&["bias", "sinks"])
            .outputs(&["mask"])
            .ensure_row_contiguous(false)
            .source(r#"
                const int key = thread_position_in_grid.x;
                const int q = thread_position_in_grid.y;
                const int h = thread_position_in_grid.z;
                const int queries = bias_shape[2];
                const int distance = KEYS - queries + q - key;
                T value = T(0);
                if (key == KEYS) {
                    value = T(sinks[h * sinks_strides[1] + q * sinks_strides[2]]);
                } else if (distance < 0 || (WINDOW > 0 && distance >= WINDOW)) {
                    value = T(-INFINITY);
                } else if (distance < bias_shape[3]) {
                    value = bias[h * bias_strides[1] + q * bias_strides[2] + distance * bias_strides[3]];
                }
                mask[(size_t(h) * queries + q) * (KEYS + 1) + key] = value;
            "#)
            .build()
    })?;
    let (heads, queries) = (bias.shape_at(1), bias.shape_at(2));
    let width = keys + 1;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[bias, sinks])
        .output_shapes(&[Shape::from(&[1, heads, queries, width][..])])
        .output_dtypes(&[bias.dtype()])
        .template_dtype("T", bias.dtype())
        .template_int("KEYS", keys)
        .template_int("WINDOW", window)
        .grid(width, queries, heads)
        .threadgroup(128, 1, 1)
        .stream(target)
        .dispatch()?
        .take_at(0)?)
}

/// Single-query GQA score with relative bias, evaluated directly from BF16 cache views.
pub(super) fn decode_relative_logits(
    q: &Array,
    k: &Array,
    bias: &Array,
    sinks: &Array,
    window: i32,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_decode_relative_logits")
            .inputs(&["q", "k", "bias", "sinks"])
            .outputs(&["out"])
            .ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint lane = thread_index_in_simdgroup;
                const uint key = thread_position_in_grid.x / 32;
                const uint head = thread_position_in_grid.y;
                const uint n = k_shape[2];
                if (key == n) {
                    if (lane == 0) out[head * (n + 1) + n] = float(sinks[head * sinks_strides[1]]);
                    return;
                }
                const uint distance = n - 1 - key;
                const uint kv_head = head / (q_shape[2] / k_shape[1]);
                const size_t q_base = head * q_strides[2];
                const size_t k_base = kv_head * k_strides[1] + key * k_strides[2];
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) {
                    const uint d = 4 * lane + i;
                    acc = fma(float(q[q_base + d * q_strides[3]]), float(k[k_base + d * k_strides[3]]), acc);
                }
                // Match the MLX FP32 GEMV shuffle tree before adding relative bias.
                #pragma clang loop unroll(full)
                for (ushort gap = 16; gap > 0; gap >>= 1) {
                    acc += simd_shuffle_down(acc, gap);
                }
                const float dot = acc;
                if (lane == 0) {
                    const float rel = distance < uint(bias_shape[3])
                        ? float(bias[head * bias_strides[1] + distance * bias_strides[3]]) : 0.0f;
                    out[head * (n + 1) + key] = WINDOW > 0 && distance >= WINDOW
                        ? -INFINITY : dot * 0.08838834764831844f + rel;
                }
            "#).build()
    })?;
    let n = k.shape_at(2);
    let h = q.shape_at(2);
    let kv = k.shape_at(1);
    Ok(kernel
        .dispatch_builder()
        .inputs(&[q, k, bias, sinks])
        .output_shapes(&[Shape::from(&[1, kv, h / kv, 1, n + 1][..])])
        .output_dtypes(&[Dtype::Float32])
        .template_int("WINDOW", window)
        .grid(32 * (n + 1), h, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Compute one learned depth query directly from up to eight normalized
/// source rows. Preserve the eight partial sums of MLX's FP32 GEMV.
pub(super) fn single_depth_query(
    blocks: &[Array],
    partial: &Array,
    query: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_single_depth_query")
            .inputs(&["s0","s1","s2","s3","s4","s5","s6","s7","q"])
            .outputs(&["out"]).ensure_row_contiguous(true)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint lane = thread_index_in_simdgroup;
                const uint simd = simdgroup_index_in_threadgroup;
                const uint row = threadgroup_position_in_grid.y;
                const uint local_index = thread_position_in_threadgroup.x;
                const device float* sources[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
                threadgroup float partials[8][8];
                threadgroup float weights[8];
                float dots[SOURCES] = {};
                for (uint column = simd * 128 + 4 * lane; column < 2304; column += 1024) {
                    for (uint j = 0; j < 4; ++j) {
                        const float coeff = q[column + j];
                        for (uint i = 0; i < SOURCES; ++i) {
                            dots[i] = fma(sources[i][row * 2304 + column + j], coeff, dots[i]);
                        }
                    }
                }
                for (uint i = 0; i < SOURCES; ++i) {
                    for (ushort gap = 16; gap; gap >>= 1) dots[i] += simd_shuffle_down(dots[i], gap);
                    if (lane == 0) partials[simd][i] = dots[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd == 0) {
                    float score = -INFINITY;
                    if (lane < SOURCES) {
                        score = partials[0][lane];
                        for (uint j = 1; j < 8; ++j) score += partials[j][lane];
                        score *= as_type<float>(SCALE);
                    }
                    const float numerator = precise::exp(score - simd_max(score));
                    float denominator = 0;
                    for (ushort i = 0; i < SOURCES; ++i) denominator += simd_shuffle(numerator, i);
                    if (lane < SOURCES) weights[lane] = precise::divide(numerator, denominator);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint c = local_index; c < 2304; c += 256) {
                    float first = 0, tail = 0;
                    for (uint i = 0; i < SOURCES; ++i) {
                        const float v = sources[i][row * 2304 + c];
                        if (i < 4) first = fma(weights[i], v, first);
                        else tail = fma(weights[i], v, tail);
                    }
                    out[row * 2304 + c] = first + tail;
                }
            "#).build()
    })?;
    let mut inputs = [partial; 9];
    for (slot, block) in inputs.iter_mut().zip(blocks) {
        *slot = block;
    }
    inputs[8] = query;
    let rows = partial.shape_at(1);
    Ok(kernel
        .dispatch_builder()
        .inputs(&inputs)
        .output_shapes(&[Shape::from(&[1, rows, 1, 2304][..])])
        .output_dtypes(&[Dtype::Float32])
        .template_int("SOURCES", (blocks.len() + 1) as i32)
        .template_int("SCALE", (2304f32.powf(-0.5)).to_bits() as i32)
        .grid(256, rows, 1)
        .threadgroup(256, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Normalize the depth logits and mix the sources in a single dispatch.
/// A one-query MLX GEMV accumulates groups of four sources independently.
pub(super) fn mix_depth(
    logits: &Array,
    states: &Array,
    scale: f32,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_depth")
            .inputs(&["logits", "x"]).outputs(&["out"]).ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint channel = thread_position_in_grid.x;
                const uint query = thread_position_in_grid.y;
                const uint row = thread_position_in_grid.z;
                const uint lane = thread_index_in_simdgroup;
                const uint sources = x_shape[2];
                float score = -INFINITY;
                if (lane < sources) {
                    score = logits[row * logits_strides[1] + lane * logits_strides[2] + query * logits_strides[3]]
                        * as_type<float>(SCALE);
                }
                const float peak = simd_max(score);
                const float probability = precise::exp(score - peak);
                float total = 0;
                for (ushort i = 0; i < sources; ++i) total += simd_shuffle(probability, i);
                const float weight = precise::divide(probability, total);
                float first = 0, tail = 0;
                for (ushort i = 0; i < sources; ++i) {
                    const float v = x[row * x_strides[1] + uint(i) * x_strides[2] + channel * x_strides[3]];
                    const float w = simd_shuffle(weight, i);
                    if (Q == 1 && i >= 4) tail = fma(w, v, tail);
                    else first = fma(w, v, first);
                }
                out[(row * Q + query) * x_shape[3] + channel] = first + tail;
            "#).build()
    })?;
    let (rows, queries, hidden) = (states.shape_at(1), logits.shape_at(-1), states.shape_at(-1));
    Ok(kernel
        .dispatch_builder()
        .inputs(&[logits, states])
        .output_shapes(&[Shape::from(&[1, rows, queries, hidden][..])])
        .output_dtypes(&[Dtype::Float32])
        .template_int("Q", queries)
        .template_int("SCALE", scale.to_bits() as i32)
        .grid(hidden, queries, rows)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Rank 128 routed experts for each decode row. Match Metal argpartition's
/// stable sort, including the original expert order when probabilities tie.
pub(super) fn route_decode(
    logits: &Array,
    bias: &Array,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_route_decode")
            .inputs(&["logits", "bias"])
            .outputs(&["scores", "indices"])
            .ensure_row_contiguous(false)
            .source(
                r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint row = thread_position_in_grid.y;
                const uint lane = thread_index_in_simdgroup;
                float keys[4];
                for (uint i = 0; i < 4; ++i) {
                    const uint id = i * 32 + lane;
                    const float x = logits[row * logits_strides[1] + id * logits_strides[2]];
                    const float low = 1.0f / (1.0f + metal::precise::exp(abs(x)));
                    const float score = x < 0.0f ? low : 1.0f - low;
                    scores[row * 128 + id] = score;
                    keys[i] = score + bias[id * bias_strides[0]];
                }
                for (uint rank = 0; rank < 8; ++rank) {
                    uint best = 0;
                    for (uint i = 1; i < 4; ++i)
                        if (keys[i] > keys[best]) best = i;
                    const float maximum = simd_max(keys[best]);
                    const uint id = best * 32 + lane;
                    const uint winner = simd_min(keys[best] == maximum ? id : 0xffffffffu);
                    if (id == winner) keys[best] = -INFINITY;
                    if (lane == 0) indices[row * 8 + rank] = winner;
                }
            "#,
            )
            .build()
    })?;
    let rows = logits.shape_at(1);
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[logits, bias])
        .output_shapes(&[logits.shape(), Shape::from(&[1, rows, 8][..])])
        .output_dtypes(&[Dtype::Float32, Dtype::Uint32])
        .grid(32, rows, 1)
        .threadgroup(32, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}

/// Apply the output gate while converting head-major values to token-major rows.
pub(super) fn gate_attention(values: &Array, gate: &Array, t: StreamOrDevice) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_gate_attention")
            .inputs(&["v","gate"]).outputs(&["out"]).ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint c = thread_position_in_grid.x;
                const uint row = thread_position_in_grid.y;
                const uint dim = v_shape[3];
                const float x = float(gate[row * gate_strides[1] + c * gate_strides[2]]);
                const float low = 1.0f / (1.0f + metal::precise::exp(abs(x)));
                const float scale = x < 0.0f ? low : 1.0f - low;
                const float value = float(v[(c / dim) * v_strides[1] + row * v_strides[2] + (c % dim) * v_strides[3]]);
                out[row * gate_shape[2] + c] = bfloat(value * scale);
            "#).build()
    })?;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[values, gate])
        .output_shapes(&[gate.shape()])
        .output_dtypes(&[gate.dtype()])
        .grid(gate.shape_at(2), gate.shape_at(1), 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Gather eight expert probabilities and apply the normalized mixture in one dispatch.
pub(super) fn mix_routed_decode(
    experts: &Array,
    scores: &Array,
    indices: &Array,
    route_scale: f32,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_routed_decode")
            .inputs(&["experts", "scores", "indices"])
            .outputs(&["out"])
            .ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint c = thread_position_in_grid.x;
                const uint token = thread_position_in_grid.y;
                const uint lane = thread_index_in_simdgroup;
                float prob = 0;
                if (lane < 8) {
                    const uint expert = indices[token * indices_strides[1] + lane * indices_strides[2]];
                    prob = float(scores[token * scores_strides[1] + expert * scores_strides[2]]);
                }
                float total = 0;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) total += simd_shuffle(prob, j);
                const float weight = (prob / (total + 1e-20f)) * as_type<float>(SCALE);
                float result = 0;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    result += float(experts[token * experts_strides[1] + j * experts_strides[2] + c * experts_strides[3]]) * simd_shuffle(weight, j);
                }
                out[token * experts_shape[3] + c] = bfloat(result);
            "#)
            .build()
    })?;
    let d = experts.shape_at(-1);
    let tokens = experts.shape_at(1);
    Ok(kernel
        .dispatch_builder()
        .inputs(&[experts, scores, indices])
        .output_shapes(&[Shape::from(&[1, tokens, d][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .template_int("SCALE", route_scale.to_bits() as i32)
        .grid(d, tokens, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// [`mix_routed_decode`] followed by [`static_conv4_with_residual`] for one
/// token, in one dispatch: each channel's mixture is rounded to BF16 exactly
/// as the separate kernel writes it, then convolved with the channel's state.
#[allow(clippy::too_many_arguments)]
pub(super) fn mix_routed_conv4_decode(
    experts: &Array,
    scores: &Array,
    indices: &Array,
    route_scale: f32,
    weight: &Array,
    state: &Array,
    residual: &Array,
    dtype: Dtype,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_routed_conv4_decode")
            .inputs(&["experts", "scores", "indices", "w", "state", "residual"])
            .outputs(&["out", "next_state"])
            .ensure_row_contiguous(false)
            .source(r#"
                const uint c = thread_position_in_grid.x;
                const uint lane = thread_index_in_simdgroup;
                const uint channels = experts_shape[3];
                float result = 0;
                {
                    #pragma clang fp contract(off) reassociate(off)
                    float prob = 0;
                    if (lane < 8) {
                        const uint expert = indices[lane * indices_strides[2]];
                        prob = float(scores[expert * scores_strides[2]]);
                    }
                    float total = 0;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) total += simd_shuffle(prob, j);
                    const float weight = (prob / (total + 1e-20f)) * as_type<float>(SCALE);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) {
                        result += float(experts[j * experts_strides[2] + c * experts_strides[3]]) * simd_shuffle(weight, j);
                    }
                }
                if (c >= channels) return;
                const float x0 = float(bfloat(result));
                float sum = 0;
                for (int tap = 0; tap < 4; ++tap) {
                    const int i = tap - 3;
                    const float v = (i < 0)
                        ? float(bfloat(state[(i + 3) * state_strides[1] + c * state_strides[2]]))
                        : x0;
                    sum = fma(v, float(w[c * w_strides[0] + tap * w_strides[1]]), sum);
                }
                float value = float(bfloat((float(bfloat(sum)) + x0) * 0.4472135954999579f));
                value += float(residual[c * residual_strides[2]]);
                out[c] = static_cast<O>(value);
                next_state[c] = state[1 * state_strides[1] + c * state_strides[2]];
                next_state[channels + c] = state[2 * state_strides[1] + c * state_strides[2]];
                next_state[2 * channels + c] = x0;
            "#)
            .build()
    })?;
    let d = experts.shape_at(-1);
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[experts, scores, indices, weight, state, residual])
        .output_shapes(&[Shape::from(&[1, 1, d][..]), Shape::from(&[1, 3, d][..])])
        .output_dtypes(&[dtype, Dtype::Float32])
        .template_int("SCALE", route_scale.to_bits() as i32)
        .template_dtype("O", dtype)
        .grid((d + 31) / 32 * 32, 1, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}

/// Recover the source token and inverse permutation for expert-sorted routes.
pub(super) fn sorted_route_positions(
    order: &Array,
    experts_per_token: i32,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_sorted_route_positions")
            .inputs(&["order"])
            .outputs(&["tokens", "inverse"])
            .source(
                r#"
                const uint i = thread_position_in_grid.x;
                const uint source = order[i];
                tokens[i] = source / EXPERTS;
                inverse[source] = i;
            "#,
            )
            .build()
    })?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[order])
        .output_shapes(&[order.shape(), order.shape()])
        .output_dtypes(&[Dtype::Uint32, Dtype::Uint32])
        .template_int("EXPERTS", experts_per_token)
        .grid(order.shape_at(0), 1, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}

/// Accumulate the routed outputs in token order without a reordered buffer.
pub(super) fn mix_sorted_routes(
    experts: &Array,
    weights: &Array,
    inverse: &Array,
    scales: Option<&Array>,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_sorted_routes")
            .inputs(&["experts", "weights", "inverse", "scales"])
            .outputs(&["output"])
            .ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint channel = thread_position_in_grid.x;
                const uint token = thread_position_in_grid.y;
                float sum = 0.f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    const uint expert = inverse[token * 8 + j];
                    float value = float(experts[expert * experts_strides[0] + channel * experts_strides[1]]);
                    if (SCALED) value = float(bfloat(value * scales[expert * scales_strides[0]]));
                    sum += value * weights[token * weights_strides[1] + j * weights_strides[2]];
                }
                output[token * experts_shape[1] + channel] = bfloat(sum);
            "#)
            .build()
    })?;
    let (tokens, channels) = (weights.shape_at(1), experts.shape_at(1));
    Ok(kernel
        .dispatch_builder()
        .inputs(&[experts, weights, inverse, scales.unwrap_or(inverse)])
        .output_shapes(&[Shape::from(&[1, tokens, channels][..])])
        .output_dtypes(&[Dtype::Bfloat16])
        .template_bool("SCALED", scales.is_some())
        .grid(channels, tokens, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Fuse the dynamic depthwise coefficients, causal taps and state advance.
pub(super) fn dynamic_conv4(
    x: &Array,
    weight: &Array,
    coefficients: &Array,
    state: &Array,
    heads: i32,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_dynamic_conv4")
            .inputs(&["x", "w", "coefficients", "state"])
            .outputs(&["out", "next_state"])
            .ensure_row_contiguous(false)
            .source(r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint c = thread_position_in_grid.x;
                const int pos = int(thread_position_in_grid.y);
                const int s = x_shape[1];
                const uint channels = x_shape[2];
                if (pos < s) {
                    const uint head = c / (channels / HEADS);
                    float sum = 0;
                    for (int lag = 0; lag < 4; ++lag) {
                        const int i = pos - lag;
                        const float v = i < 0
                            ? state[(i + 3) * state_strides[1] + c * state_strides[2]]
                            : float(x[i * x_strides[1] + c * x_strides[2]]);
                        const float coefficient = float(w[lag * w_strides[0] + c * w_strides[1]])
                            + float(coefficients[pos * coefficients_strides[1] + (4 * head + lag) * coefficients_strides[2]]);
                        sum += v * coefficient;
                    }
                    const float input = float(x[pos * x_strides[1] + c * x_strides[2]]);
                    out[pos * channels + c] = bfloat((sum + input) * (1.0f / 3.0f));
                }
                if (pos < 3) {
                    const int i = s + pos - 3;
                    next_state[pos * channels + c] = i < 0
                        ? state[(i + 3) * state_strides[1] + c * state_strides[2]]
                        : float(x[i * x_strides[1] + c * x_strides[2]]);
                }
            "#)
            .build()
    })?;
    let mut y = kernel
        .dispatch_builder()
        .inputs(&[x, weight, coefficients, state])
        .output_shapes(&[x.shape(), state.shape()])
        .output_dtypes(&[x.dtype(), state.dtype()])
        .template_int("HEADS", heads)
        .grid(x.shape_at(2), x.shape_at(1).max(3), 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((y.take_at(0)?, y.take_at(0)?))
}

/// Four-tap depthwise convolution with the same BF16 convolution boundary as
/// MLX conv1d. Read the packed cache view directly and advance all three taps.
pub(super) fn static_conv4(
    x: &Array,
    weight: &Array,
    state: &Array,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static_conv4_with_residual(x, weight, state, None, x.dtype(), t)
}

pub(super) fn static_conv4_with_residual(
    x: &Array,
    weight: &Array,
    state: &Array,
    residual: Option<&Array>,
    dtype: Dtype,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_static_conv4")
            .inputs(&["x", "w", "state", "residual"])
            .outputs(&["out", "next_state"])
            .ensure_row_contiguous(false)
            .source(r#"
                const uint c = thread_position_in_grid.x;
                const int pos = int(thread_position_in_grid.y);
                const int s = x_shape[1];
                const uint channels = x_shape[2];
                if (pos < s) {
                    float sum = 0;
                    for (int tap = 0; tap < 4; ++tap) {
                        const int i = pos + tap - 3;
                        const float v = (i < 0)
                            ? float(bfloat(state[(i + 3) * state_strides[1] + c * state_strides[2]]))
                            : float(x[i * x_strides[1] + c * x_strides[2]]);
                        sum = fma(v, float(w[c * w_strides[0] + tap * w_strides[1]]), sum);
                    }
                    const float skip = float(x[pos * x_strides[1] + c * x_strides[2]]);
                    float value = float(bfloat((float(bfloat(sum)) + skip) * 0.4472135954999579f));
                    if (ADD_RESIDUAL)
                        value += float(residual[pos * residual_strides[1] + c * residual_strides[2]]);
                    out[pos * channels + c] = static_cast<O>(value);
                }
                if (pos < 3) {
                    const int i = s + pos - 3;
                    next_state[pos * channels + c] = (i < 0)
                        ? state[(i + 3) * state_strides[1] + c * state_strides[2]]
                        : float(x[i * x_strides[1] + c * x_strides[2]]);
                }
            "#)
            .build()
    })?;
    let shape = x.shape();
    let c = x.shape_at(2);
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, weight, state, residual.unwrap_or(x)])
        .output_shapes(&[shape, Shape::from(&[1, 3, c][..])])
        .output_dtypes(&[dtype, Dtype::Float32])
        .template_dtype("O", dtype)
        .template_bool("ADD_RESIDUAL", residual.is_some())
        .grid(c, x.shape_at(1).max(3), 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}

/// Tables contain the original FP32 gate and up functions for every possible
/// BF16 input. All layers share the same tables and round after the product.
#[derive(Clone)]
pub(super) struct ActivationTables {
    gate: Array,
    up: Array,
}
impl ActivationTables {
    pub(super) fn new(gate_cap: f32, up_cap: f32, t: StreamOrDevice) -> Result<Self> {
        let values: Vec<f32> = (0..=u16::MAX)
            .map(|v| f32::from_bits(u32::from(v) << 16))
            .collect();
        let values: Array = (values.as_slice(), &[65536][..]).try_into()?;
        let gate = cap(&values, gate_cap, t)?.try_mul_on(&values.sigmoid_on(t)?, t)?;
        let up = cap(&values, up_cap, t)?;
        mlx::transforms::eval(&[&gate, &up])?;
        Ok(Self { gate, up })
    }

    pub(super) fn tables(&self) -> (&Array, &Array) {
        (&self.gate, &self.up)
    }

    pub(super) fn apply_scaled(
        &self,
        x: &Array,
        scales: Option<&Array>,
        t: StreamOrDevice,
    ) -> Result<Array> {
        static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
        let kernel = kernel(&KERNEL, || {
            MetalKernel::builder("ironmlx_inkling_routed_activation")
                .inputs(&["x","gate","up","scales"])
                .outputs(&["out"])
                .source(r#"
                    const size_t i = size_t(thread_position_in_grid.y) * threads_per_grid.x + thread_position_in_grid.x;
                    const float scale = SCALED ? float(scales[thread_position_in_grid.y]) : 1.f;
                    const ushort g = as_type<ushort>(bfloat(float(x[2*i]) * scale));
                    const ushort u = as_type<ushort>(bfloat(float(x[2*i+1]) * scale));
                    out[i] = bfloat(gate[g] * up[u]);
                "#)
                .build()
        })?;
        let mut dims = x.shape().as_slice().to_vec();
        let width = x.shape_at(-1) / 2;
        *dims.last_mut().expect("expert projection rank") = width;
        let rows = i32::try_from(x.size() / x.shape_at(-1) as usize)?;
        Ok(kernel
            .dispatch_builder()
            .inputs(&[x, &self.gate, &self.up, scales.unwrap_or(&self.gate)])
            .output_shapes(&[Shape::from(dims.as_slice())])
            .output_dtypes(&[Dtype::Bfloat16])
            .template_bool("SCALED", scales.is_some())
            .grid(width, rows, 1)
            .threadgroup(128, 1, 1)
            .stream(t)
            .dispatch()?
            .take_at(0)?)
    }
}

/// One SIMD group normalizes one head. The relative and K/Q projections each
/// round only after clipping, so keep the reduction and scaling in FP32.
pub(super) fn clip_heads(
    x: &Array,
    weights: Option<&Array>,
    threshold: f32,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_clip_heads")
            .inputs(&["x", "w"])
            .outputs(&["out"])
            .source(
                r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint lane = thread_index_in_simdgroup;
                const uint head = thread_position_in_grid.y;
                const uint token = thread_position_in_grid.z;
                const size_t base = (size_t(token) * x_shape[2] + head) * D;
                // MLX's FP32 row sum uses four consecutive inputs per lane
                // for D=128, and a serial row reduction for D=16.
                const uint width = D == 128 ? 4 : 1;
                float values[width];
                float sum = 0.f;
                for (uint j = 0; j < width; ++j) {
                    const uint i = lane * width + j;
                    float v = i < D ? float(x[base+i]) : 0.f;
                    if (WEIGHTED && i < D) v *= float(w[head * D + i]);
                    values[j] = v;
                    sum += v*v;
                }
                if (D == 16) {
                    const float square = sum;
                    sum = 0.f;
                    for (uint i = 0; i < 16; ++i) sum += simd_shuffle(square, i);
                } else {
                    sum = simd_sum(sum);
                }
                const float norm = precise::sqrt(sum) + 1e-6f;
                const float ratio = min(1.f, as_type<float>(THRESHOLD) / norm);
                for (uint j = 0; j < width; ++j) {
                    const uint i = lane * width + j;
                    if (i < D) out[base+i] = bfloat(values[j] * ratio);
                }
            "#,
            )
            .build()
    })?;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[x, weights.unwrap_or(x)])
        .output_shapes(&[x.shape()])
        .output_dtypes(&[x.dtype()])
        .template_int("D", x.shape_at(-1))
        .template_int("THRESHOLD", threshold.to_bits() as i32)
        .template_bool("WEIGHTED", weights.is_some())
        .grid(32, x.shape_at(2), x.shape_at(1))
        .threadgroup(32, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// MLX's single-row RMSNorm (`rms_single_row`, four reads per thread) over a
/// float32 view of `x`, with an optional float32 scale, written in `out_dtype`.
/// It computes what `rms_norm(x.astype(f32), w)` and a cast of its result do,
/// in one dispatch instead of up to three.
pub(super) fn rms_cast(
    x: &Array,
    weight: Option<&Array>,
    eps: f32,
    out_dtype: Dtype,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_rms_cast")
            .inputs(&["x", "w"])
            .outputs(&["out"])
            .source(
                r#"
                const uint gid = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                constexpr uint N_READS = 4;
                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[32];
                float acc = 0;
                float thread_x[N_READS];
                const size_t row = size_t(gid) * AXIS;
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    thread_x[i] = c < AXIS ? float(x[row + c]) : 0;
                    acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) {
                    local_sums[simd_lane_id] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) {
                    local_sums[simd_group_id] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(acc / AXIS + as_type<float>(EPS));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    if (c < AXIS) {
                        const float y = thread_x[i] * local_inv_mean[0];
                        out[row + c] = HAS_W ? static_cast<out_t>(float(w[c]) * y) : static_cast<out_t>(y);
                    }
                }
            "#,
            )
            .build()
    })?;
    let axis = x.shape_at(-1);
    let rows = i32::try_from(x.size())? / axis;
    let threads = (axis + 3) / 4;
    let threads = (threads + 31) / 32 * 32;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[x, weight.unwrap_or(x)])
        .output_shapes(&[x.shape()])
        .output_dtypes(&[out_dtype])
        .template_int("AXIS", axis)
        .template_int("EPS", eps.to_bits() as i32)
        .template_bool("HAS_W", weight.is_some())
        .template_dtype("out_t", out_dtype)
        .grid(threads * rows, 1, 1)
        .threadgroup(threads, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// [`rms_cast`] that also returns its result widened back to float32.
pub(super) fn rms_cast_dual(
    x: &Array,
    weight: Option<&Array>,
    eps: f32,
    out_dtype: Dtype,
    t: StreamOrDevice,
) -> Result<(Array, Array)> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_rms_cast_dual")
            .inputs(&["x", "w"])
            .outputs(&["out", "wide"])
            .source(
                r#"
                const uint gid = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                constexpr uint N_READS = 4;
                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[32];
                float acc = 0;
                float thread_x[N_READS];
                const size_t row = size_t(gid) * AXIS;
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    thread_x[i] = c < AXIS ? float(x[row + c]) : 0;
                    acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) {
                    local_sums[simd_lane_id] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) {
                    local_sums[simd_group_id] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(acc / AXIS + as_type<float>(EPS));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    if (c < AXIS) {
                        const float y = thread_x[i] * local_inv_mean[0];
                        const out_t narrow = HAS_W ? static_cast<out_t>(float(w[c]) * y) : static_cast<out_t>(y);
                        out[row + c] = narrow;
                        wide[row + c] = float(narrow);
                    }
                }
            "#,
            )
            .build()
    })?;
    let axis = x.shape_at(-1);
    let rows = i32::try_from(x.size())? / axis;
    let threads = (axis + 3) / 4;
    let threads = (threads + 31) / 32 * 32;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[x, weight.unwrap_or(x)])
        .output_shapes(&[x.shape(), x.shape()])
        .output_dtypes(&[out_dtype, Dtype::Float32])
        .template_int("AXIS", axis)
        .template_int("EPS", eps.to_bits() as i32)
        .template_bool("HAS_W", weight.is_some())
        .template_dtype("out_t", out_dtype)
        .grid(threads * rows, 1, 1)
        .threadgroup(threads, 1, 1)
        .stream(t)
        .dispatch()?;
    Ok((outputs.take_at(0)?, outputs.take_at(0)?))
}

/// `rms_norm(cat([x, msg], -1).reshape(gates, s, 2, h))` cast to `out_dtype`
/// and laid out as `[gates, s, 2h]`, read from strided `[gates, 1, s, h]`
/// inputs. Each half row reduces as MLX's `rms_single_row` does.
pub(super) fn rms_pair_cast(
    x: &Array,
    msg: &Array,
    out_dtype: Dtype,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_rms_pair_cast")
            .inputs(&["x", "msg"])
            .outputs(&["out"])
            .ensure_row_contiguous(false)
            .source(
                r#"
                const uint gid = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                constexpr uint N_READS = 4;
                const uint steps = x_shape[2];
                const uint half_row = gid % 2;
                const uint pos = (gid / 2) % steps;
                const uint g = gid / (2 * steps);
                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[32];
                float acc = 0;
                float thread_x[N_READS];
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    float v = 0;
                    if (c < AXIS) {
                        v = half_row == 0
                            ? float(x[g * x_strides[0] + pos * x_strides[2] + c * x_strides[3]])
                            : float(msg[g * msg_strides[0] + pos * msg_strides[2] + c * msg_strides[3]]);
                    }
                    thread_x[i] = v;
                    acc += thread_x[i] * thread_x[i];
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) {
                    local_sums[simd_lane_id] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_lane_id == 0) {
                    local_sums[simd_group_id] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(acc / AXIS + 1e-6f);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const size_t row = size_t(gid) * AXIS;
                for (uint i = 0; i < N_READS; i++) {
                    const uint c = lid * N_READS + i;
                    if (c < AXIS) {
                        out[row + c] = static_cast<out_t>(thread_x[i] * local_inv_mean[0]);
                    }
                }
            "#,
            )
            .build()
    })?;
    let (gates, steps, axis) = (x.shape_at(0), x.shape_at(2), x.shape_at(3));
    let threads = ((axis + 3) / 4 + 31) / 32 * 32;
    Ok(kernel
        .dispatch_builder()
        .inputs(&[x, msg])
        .output_shapes(&[Shape::from(&[gates, steps, 2 * axis][..])])
        .output_dtypes(&[out_dtype])
        .template_int("AXIS", axis)
        .template_dtype("out_t", out_dtype)
        .grid(threads * gates * steps * 2, 1, 1)
        .threadgroup(threads, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// One query row's attention over 8-bit affine K/V (groups of 64 along the
/// head dimension; see `crate::core::cache::int8_kv`), keys `[start, end)`,
/// with the relative bias `bias` (`[.., heads, .., EXTENT]`, indexed by
/// distance from the far end of the extent) over the nearest keys and one
/// sink logit per head. Two passes: each threadgroup reduces one KV head
/// over one block of keys, then one threadgroup per head merges the blocks.
/// `q` is `[1, 1, heads, 128]`; returns `[1, heads, 1, 128]` in `q`'s type.
#[allow(clippy::too_many_arguments)]
pub(super) fn int8_row_attention(
    q: &Array,
    keys: &crate::core::cache::int8_kv::Int8KvSide,
    values: &crate::core::cache::int8_kv::Int8KvSide,
    start: i32,
    end: i32,
    bias: &Array,
    sinks: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    static PASS1: OnceLock<MetalKernel> = OnceLock::new();
    static PASS2: OnceLock<MetalKernel> = OnceLock::new();
    let pass1 = kernel(&PASS1, || {
        MetalKernel::builder("ironmlx_inkling_int8_row_attention")
            .inputs(&["q", "kw", "ks", "kb", "vw", "vs", "vb", "bias", "range"])
            .outputs(&["pm", "pl", "po"])
            .ensure_row_contiguous(false)
            .source(
                r#"
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint block = threadgroup_position_in_grid.x;
                const uint kvh = threadgroup_position_in_grid.y;
                constexpr uint SG = 8;
                const uint start = range[0];
                const uint end = range[1];
                float qv[G][4];
                for (uint h = 0; h < G; ++h)
                    for (uint i = 0; i < 4; ++i)
                        qv[h][i] = float(q[(kvh * G + h) * q_strides[2] + (lane * 4 + i) * q_strides[3]])
                            * as_type<float>(SCALE);
                float m[G], l[G], o[G][4];
                for (uint h = 0; h < G; ++h) {
                    m[h] = -INFINITY;
                    l[h] = 0;
                    for (uint i = 0; i < 4; ++i) o[h][i] = 0;
                }
                const uint count = end - start;
                const uint per_block = (count + BLOCKS - 1) / BLOCKS;
                const uint first = start + block * per_block;
                const uint last = min(end, first + per_block);
                const uint group = lane / 16;
                // Two keys a step, in the same order as one at a time: the
                // reductions of the second overlap those of the first.
                constexpr uint NK = 2;
                for (uint key0 = first + sg; key0 < last; key0 += SG * NK) {
                    float kd[NK][4], vd[NK][4];
                    uint keys[NK];
                    for (uint n = 0; n < NK; ++n) {
                        const uint key = key0 + n * SG < last ? key0 + n * SG : key0;
                        keys[n] = key;
                        const size_t kw_row = size_t(kvh) * kw_strides[1] + size_t(key) * kw_strides[2];
                        const size_t ks_row = size_t(kvh) * ks_strides[1] + size_t(key) * ks_strides[2];
                        const size_t vw_row = size_t(kvh) * vw_strides[1] + size_t(key) * vw_strides[2];
                        const size_t vs_row = size_t(kvh) * vs_strides[1] + size_t(key) * vs_strides[2];
                        const uint kword = kw[kw_row + lane];
                        const float ksc = float(ks[ks_row + group]);
                        const float kbi = float(kb[ks_row + group]);
                        const uint vword = vw[vw_row + lane];
                        const float vsc = float(vs[vs_row + group]);
                        const float vbi = float(vb[vs_row + group]);
                        for (uint i = 0; i < 4; ++i) {
                            kd[n][i] = float((kword >> (8 * i)) & 255u) * ksc + kbi;
                            vd[n][i] = float((vword >> (8 * i)) & 255u) * vsc + vbi;
                        }
                    }
                    float scores[NK][G];
                    for (uint n = 0; n < NK; ++n)
                        for (uint h = 0; h < G; ++h) {
                            float partial = 0;
                            for (uint i = 0; i < 4; ++i) partial += qv[h][i] * kd[n][i];
                            scores[n][h] = simd_sum(partial);
                        }
                    for (uint n = 0; n < NK; ++n) {
                        if (key0 + n * SG >= last) break;
                        const uint key = keys[n];
                        const bool near = key + EXTENT >= end;
                        for (uint h = 0; h < G; ++h) {
                            float score = scores[n][h];
                            if (near) score += float(bias[(kvh * G + h) * bias_strides[1] + (key + EXTENT - end) * bias_strides[3]]);
                            const float next = max(m[h], score);
                            const float factor = fast::exp(m[h] - next);
                            const float p = fast::exp(score - next);
                            m[h] = next;
                            l[h] = l[h] * factor + p;
                            for (uint i = 0; i < 4; ++i) o[h][i] = o[h][i] * factor + p * vd[n][i];
                        }
                    }
                }
                threadgroup float tm[SG][G];
                threadgroup float tl[SG][G];
                threadgroup float to[SG][G][128];
                for (uint h = 0; h < G; ++h) {
                    if (lane == 0) {
                        tm[sg][h] = m[h];
                        tl[sg][h] = l[h];
                    }
                    for (uint i = 0; i < 4; ++i) to[sg][h][lane * 4 + i] = o[h][i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint h = sg; h < G; h += SG) {
                    float mm = -INFINITY;
                    for (uint j = 0; j < SG; ++j) mm = max(mm, tm[j][h]);
                    float ll = 0;
                    float acc[4] = {0, 0, 0, 0};
                    for (uint j = 0; j < SG; ++j) {
                        const float f = tm[j][h] == -INFINITY ? 0.f : fast::exp(tm[j][h] - mm);
                        ll += tl[j][h] * f;
                        for (uint i = 0; i < 4; ++i) acc[i] += to[j][h][lane * 4 + i] * f;
                    }
                    const uint row = (kvh * G + h) * BLOCKS + block;
                    if (lane == 0) {
                        pm[row] = mm;
                        pl[row] = ll;
                    }
                    for (uint i = 0; i < 4; ++i) po[row * 128 + lane * 4 + i] = acc[i];
                }
            "#,
            )
            .build()
    })?;
    let pass2 = kernel(&PASS2, || {
        MetalKernel::builder("ironmlx_inkling_int8_row_attention_merge")
            .inputs(&["pm", "pl", "po", "sinks"])
            .outputs(&["out"])
            .ensure_row_contiguous(false)
            .source(
                r#"
                const uint d = thread_position_in_threadgroup.x;
                const uint qh = threadgroup_position_in_grid.x;
                const float sink = float(sinks[qh * sinks_strides[1]]);
                float mm = sink;
                for (uint b = 0; b < BLOCKS; ++b) mm = max(mm, pm[qh * BLOCKS + b]);
                float total = fast::exp(sink - mm);
                float acc = 0;
                for (uint b = 0; b < BLOCKS; ++b) {
                    const float pmb = pm[qh * BLOCKS + b];
                    const float f = pmb == -INFINITY ? 0.f : fast::exp(pmb - mm);
                    total += pl[qh * BLOCKS + b] * f;
                    acc += po[(qh * BLOCKS + b) * 128 + d] * f;
                }
                out[qh * 128 + d] = static_cast<out_t>(acc / total);
            "#,
            )
            .build()
    })?;
    let (heads, dim) = (q.shape_at(2), q.shape_at(3));
    let kv_heads = keys.words.shape_at(1);
    anyhow::ensure!(
        dim == 128 && heads % kv_heads == 0 && end > start && start >= 0,
        "8-bit row attention shape"
    );
    let count = end - start;
    let blocks = (((count + 63) / 64).max(1) as u32)
        .next_power_of_two()
        .clamp(16, 128) as i32;
    let range: Array = (&[start as u32, end as u32][..], &[2][..]).try_into()?;
    let scale = (dim as f32).powf(-0.5);
    let rows = heads * blocks;
    let mut partial = pass1
        .dispatch_builder()
        .inputs(&[
            q,
            &keys.words,
            &keys.scales,
            &keys.biases,
            &values.words,
            &values.scales,
            &values.biases,
            bias,
            &range,
        ])
        .output_shapes(&[
            Shape::from(&[rows][..]),
            Shape::from(&[rows][..]),
            Shape::from(&[rows, 128][..]),
        ])
        .output_dtypes(&[Dtype::Float32, Dtype::Float32, Dtype::Float32])
        .template_int("G", heads / kv_heads)
        .template_int("BLOCKS", blocks)
        .template_int("EXTENT", bias.shape_at(3))
        .template_int("SCALE", scale.to_bits() as i32)
        .grid(256 * blocks, kv_heads, 1)
        .threadgroup(256, 1, 1)
        .stream(t)
        .dispatch()?;
    let (pm, pl, po) = (
        partial.take_at(0)?,
        partial.take_at(0)?,
        partial.take_at(0)?,
    );
    Ok(pass2
        .dispatch_builder()
        .inputs(&[&pm, &pl, &po, sinks])
        .output_shapes(&[Shape::from(&[1, heads, 1, 128][..])])
        .output_dtypes(&[q.dtype()])
        .template_int("BLOCKS", blocks)
        .template_dtype("out_t", q.dtype())
        .grid(128 * heads, 1, 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// Mix the packed loop controller outputs without materializing gate slices.
pub(super) fn mix_packed_messages(
    x: &Array,
    message: &Array,
    high: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let k = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_packed_messages")
            .inputs(&["x", "msg", "high"])
            .outputs(&["out"])
            .ensure_row_contiguous(false)
            .source(
                r#"
                #pragma clang fp contract(off) reassociate(off)
                const uint c = thread_position_in_grid.x;
                const uint pos = thread_position_in_grid.y;
                const uint g = thread_position_in_grid.z;
                const uint h = x_shape[3];
                const size_t j = g*high_strides[0] + pos*high_strides[2] + c*high_strides[3];
                const float gain = precise::tanh(high[j]);
                const float forget = precise::tanh(high[j+h*high_strides[3]]);
                const float a = x[g*x_strides[0] + pos*x_strides[2] + c*x_strides[3]];
                const float b = msg[g*msg_strides[0] + pos*msg_strides[2] + c*msg_strides[3]];
                out[(g*x_shape[2]+pos)*h+c] = (1.f-forget)*a + gain*b;
            "#,
            )
            .build()
    })?;
    Ok(k.dispatch_builder()
        .inputs(&[x, message, high])
        .output_shapes(&[x.shape()])
        .output_dtypes(&[Dtype::Float32])
        .grid(x.shape_at(3), x.shape_at(2), x.shape_at(0))
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

/// The loop combines two already-projected FP32 gates with FP32 messages.
pub(super) fn mix_message(
    x: &Array,
    message: &Array,
    forget: &Array,
    gain: &Array,
    t: StreamOrDevice,
) -> Result<Array> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let k = kernel(&KERNEL, || {
        MetalKernel::builder("ironmlx_inkling_mix_message")
        .inputs(&["x","message","forget","gain"])
        .outputs(&["out"])
        .source(r#"
            #pragma clang fp contract(off) reassociate(off)
            const size_t i = size_t(thread_position_in_grid.y) * threads_per_grid.x + thread_position_in_grid.x;
            out[i] = (1.f - precise::tanh(forget[i])) * x[i] + precise::tanh(gain[i]) * message[i];
        "#).build()
    })?;
    Ok(k.dispatch_builder()
        .inputs(&[x, message, forget, gain])
        .output_shapes(&[x.shape()])
        .output_dtypes(&[Dtype::Float32])
        .grid(x.shape_at(-1), x.shape_at(1), 1)
        .threadgroup(128, 1, 1)
        .stream(t)
        .dispatch()?
        .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::{indexing, shape};

    fn metal_stream() -> Result<StreamOrDevice> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        Ok(StreamOrDevice::default())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn decode_router_preserves_probabilities_and_ties() -> Result<()> {
        let t = metal_stream()?;
        let logits = mlx::random::normal()
            .shape(&[1, 3, 128][..])
            .key(&mlx::random::key(759)?)
            .sample()?;
        let bias = Array::zeros((128,), Dtype::Float32)?;
        for input in [&logits, &Array::zeros((1, 3, 128), Dtype::Float32)?] {
            let (scores, indices) = route_decode(input, &bias, t)?;
            let reference = input.sigmoid_on(t)?;
            let ordered = mlx::ops::sort::argpartition_on(&reference.try_neg_on(t)?, 7, -1, t)?;
            let expected = super::super::ops::slice(&ordered, 2, 0, 8, t)?;
            assert_eq!(scores.to_vec::<f32>()?, reference.to_vec::<f32>()?);
            assert_eq!(indices.to_vec::<u32>()?, expected.to_vec::<u32>()?);
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn fused_depth_matches_serial_source_reductions() -> Result<()> {
        let t = metal_stream()?;
        let hidden = 2304;
        let scale = (hidden as f32).powf(-0.5);
        for rows in [1, 3] {
            for sources in 1..=8 {
                let key = mlx::random::key((rows * 8 + sources) as u64)?;
                let states = mlx::random::normal()
                    .shape(&[1, rows, sources, hidden][..])
                    .key(&key)
                    .sample()?;
                for queries in [1, 3, 4] {
                    let logits = mlx::random::normal()
                        .shape(&[1, rows, sources, queries][..])
                        .key(&key)
                        .sample()?;
                    let scaled = super::super::ops::scale(&logits, scale, t)?;
                    let weights = scaled
                        .softmax_on(-2, true, t)?
                        .transpose_axes_on([0, 1, 3, 2], t)?;
                    let reference = weights.matmul_on(&states, t)?;
                    let fused = mix_depth(&logits, &states, scale, t)?;
                    assert_eq!(
                        fused
                            .not_equal(&reference)?
                            .astype(Dtype::Int32)?
                            .sum(mlx::ops::All, false)?
                            .item::<i32>()?,
                        0,
                        "rows {rows}, sources {sources}, queries {queries}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn single_depth_query_matches_stacked_sources() -> Result<()> {
        let t = metal_stream()?;
        let key = mlx::random::key(215)?;
        let query = mlx::random::normal()
            .shape(&[1, 2304][..])
            .key(&key)
            .sample()?;
        for rows in [1, 3, 8] {
            let mut blocks = Vec::new();
            for count in 1..=8 {
                let partial = mlx::random::normal()
                    .shape(&[1, rows, 2304][..])
                    .key(&mlx::random::key(count as u64)?)
                    .sample()?;
                let reference = super::super::ops::depth_attention(&blocks, &partial, &query, t)?;
                let fused = single_depth_query(&blocks, &partial, &query, t)?;
                assert_eq!(
                    fused
                        .not_equal(&reference)?
                        .astype(Dtype::Int32)?
                        .sum(mlx::ops::All, false)?
                        .item::<i32>()?,
                    0,
                    "rows {rows}, sources {count}"
                );
                blocks.push(partial);
            }
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn activation_table_preserves_reference_rounding_for_bfloat16_pairs() -> Result<()> {
        let t = metal_stream()?;
        let values: Vec<f32> = (0..=u16::MAX)
            .map(|bits| f32::from_bits(u32::from(bits) << 16))
            .filter(|v| v.is_finite())
            .collect();
        let gates: Array = (values.as_slice(), &[values.len() as i32][..]).try_into()?;
        let ups = indexing::take_on(
            &gates,
            &mlx::ops::constructors::arange_on(
                (values.len() - 1) as f64,
                -1.0,
                -1.0,
                Dtype::Int32,
                t,
            )?,
            0,
            t,
        )?;
        let x = mlx::ops::shape::stack_on(&[&gates, &ups], -1, t)?.astype(Dtype::Bfloat16)?;
        let want = cap(&gates, 4.0, t)?
            .try_mul(&gates.sigmoid()?)?
            .try_mul(&cap(&ups, 25.0, t)?)?
            .astype(Dtype::Bfloat16)?;
        let got = ActivationTables::new(4.0, 25.0, t)?
            .apply_scaled(&x, None, t)?
            .squeeze_on(-1, t)?;
        let different = got
            .not_equal(&want)?
            .astype(Dtype::Int32)?
            .sum(mlx::ops::All, false)?
            .item::<i32>()?;
        assert_eq!(different, 0);
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn row_major_gate_projection_is_row_stable_and_close_to_column_major() -> Result<()> {
        let t = metal_stream()?;
        let rand = |shape: &[i32], seed| -> anyhow::Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(seed)?)
                .sample()?
                .astype(Dtype::Bfloat16)?)
        };
        for gates in [1, 3] {
            // The checkpoint stores BF16 `[gates, in, out]`.
            let down = rand(&[gates, 4608, 64], 91)?;
            let rows = down.transpose_axes([0, 2, 1])?.contiguous(false)?;
            for tokens in [1, 3, 8] {
                let x = rand(&[gates, tokens, 4608], 17 + tokens as u64)?;
                let got = super::super::layers::row_major_matmul(&x, &rows, t)?;
                for token in 0..tokens {
                    let one = super::super::ops::slice(&x, 1, token, token + 1, t)?;
                    let row = super::super::ops::slice(&got, 1, token, token + 1, t)?;
                    // Verification rows must reproduce single-token decode
                    // bit for bit.
                    let alone = super::super::layers::row_major_matmul(&one, &rows, t)?;
                    let different = row
                        .not_equal(&alone)?
                        .astype(Dtype::Int32)?
                        .sum(mlx::ops::All, false)?
                        .item::<i32>()?;
                    assert_eq!(different, 0, "gates={gates} tokens={tokens} token={token}");
                    // The `[in, out]` layout sums the same products in
                    // another order: at most one BF16 step apart.
                    let reference = one.matmul_on(&down, t)?.astype_on(Dtype::Float32, t)?;
                    let error = row
                        .astype_on(Dtype::Float32, t)?
                        .try_sub_on(&reference, t)?
                        .abs()?
                        .try_div_on(
                            &reference
                                .abs()?
                                .try_add_on(&super::super::ops::scalar(1.0)?, t)?,
                            t,
                        )?
                        .max(mlx::ops::All, false)?
                        .item::<f32>()?;
                    assert!(error < 0.02, "gates={gates} tokens={tokens} error={error}");
                }
            }
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn rms_pair_cast_matches_the_concatenated_norm_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        for (gates, steps) in [(3, 1), (1, 1), (3, 4)] {
            let key = mlx::random::key((gates * 10 + steps) as u64)?;
            let x = mlx::random::normal()
                .shape(&[gates, 1, steps, 2304][..])
                .key(&key)
                .sample()?;
            // A strided message, as the packed controller slices it.
            let msg = mlx::random::normal()
                .shape(&[1, steps, gates + 1, 2304][..])
                .key(&mlx::random::split(&key)?.0)
                .sample()?;
            let msg = msg
                .reshape(&[1, steps, gates + 1, 2304][..])?
                .transpose_axes([2, 0, 1, 3])?;
            let msg = indexing::slice(&msg, &[0, 0, 0, 0], &[gates, 1, steps, 2304])?;
            let expected = mlx::fast::rms_norm(
                &shape::concatenate(&[&x, &msg], -1)?.reshape(&[gates, steps, 2, 2304][..])?,
                None,
                1e-6,
            )?
            .reshape(&[gates, steps, 4608][..])?
            .astype(Dtype::Bfloat16)?;
            let actual = rms_pair_cast(&x, &msg, Dtype::Bfloat16, t)?;
            let diffs = actual
                .not_equal(&expected)?
                .astype(Dtype::Int32)?
                .sum(mlx::ops::All, false)?
                .item::<i32>()?;
            assert_eq!(diffs, 0, "gates={gates} steps={steps}");
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn mix_routed_conv4_matches_the_two_kernels_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        let d = 2304;
        let normal = |shape: &[i32], seed: u64| -> Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(77 + seed)?)
                .sample()?)
        };
        let experts = normal(&[1, 1, 8, d], 0)?.astype(Dtype::Bfloat16)?;
        let scores = normal(&[1, 1, 128], 1)?.sigmoid()?;
        let indices: Array = (&[3u32, 90, 17, 64, 5, 127, 40, 8][..], &[1, 1, 8][..]).try_into()?;
        let weight = normal(&[d, 4, 1], 2)?.astype(Dtype::Bfloat16)?;
        let state = normal(&[1, 3, d], 3)?;
        let residual = normal(&[1, 1, d], 4)?;
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let mixed = mix_routed_decode(&experts, &scores, &indices, 8.0, t)?;
            let (want, want_state) =
                static_conv4_with_residual(&mixed, &weight, &state, Some(&residual), dtype, t)?;
            let (got, got_state) = mix_routed_conv4_decode(
                &experts, &scores, &indices, 8.0, &weight, &state, &residual, dtype, t,
            )?;
            for (a, b, what) in [(&got, &want, "out"), (&got_state, &want_state, "state")] {
                let diffs = a
                    .not_equal(b)?
                    .astype(Dtype::Int32)?
                    .sum(mlx::ops::All, false)?
                    .item::<i32>()?;
                assert_eq!(diffs, 0, "{what} {dtype:?}");
            }
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn int8_row_attention_matches_attention_over_the_dequantized_cache() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        let normal = |shape: &[i32], seed: u64| -> Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(700 + seed)?)
                .sample()?
                .astype(Dtype::Bfloat16)?)
        };
        for (keys, cap) in [(9, 256), (700, 1024), (3000, 4096)] {
            let mut cache = crate::core::cache::KVCache::new(1, 8, 128, 128, Dtype::Bfloat16, 8192)
                .with_step(256);
            assert!(cache.enable_int8());
            let k = normal(&[1, 8, keys, 128], 1)?;
            let v = normal(&[1, 8, keys, 128], 2)?;
            let (dk, dv) = cache.update_and_fetch(&k, &v, &[keys])?;
            let _ = cap;
            let qerr = dk
                .astype(Dtype::Float32)?
                .try_sub(&k.astype(Dtype::Float32)?)?
                .abs()?
                .max(mlx::ops::All, false)?
                .item::<f32>()?;
            // Affine 8-bit groups of 64 keep normal values within a few
            // hundredths.
            assert!(qerr < 0.05, "keys={keys} dequantization error {qerr}");
            let (ks, vs) = cache.int8().unwrap().sides().unwrap();
            // q as the attention layer slices it: [b, s, h, d] with s > 1.
            let q_all = normal(&[1, 2, 24, 128], 3)?;
            let q = indexing::slice(&q_all, &[0, 1, 0, 0], &[1, 2, 24, 128])?;
            let bias_all = normal(&[1, 24, 2, 1024], 4)?;
            let bias = indexing::slice(&bias_all, &[0, 0, 1, 0], &[1, 24, 2, 1024])?;
            let sinks_all = normal(&[1, 2, 24], 5)?.transpose_axes([0, 2, 1])?;
            let sinks = indexing::slice(&sinks_all, &[0, 0, 1], &[1, 24, 2])?;
            let got = int8_row_attention(&q, ks, vs, 0, keys, &bias, &sinks, t)?;
            let want = super::super::layers::fused_row_attention_for_tests(
                &q, &dk, &dv, &bias, &sinks, t,
            )?;
            let err = got
                .astype(Dtype::Float32)?
                .try_sub(&want.astype(Dtype::Float32)?)?
                .abs()?
                .max(mlx::ops::All, false)?
                .item::<f32>()?;
            assert!(err < 2e-2, "keys={keys} max error {err}");
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn rms_cast_matches_mlx_rms_norm_bitwise() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let t = StreamOrDevice::default();
        for rows in [1, 3, 8] {
            let key = mlx::random::key(rows as u64)?;
            let x = mlx::random::normal()
                .shape(&[1, rows, 2304][..])
                .key(&key)
                .sample()?
                .astype(Dtype::Bfloat16)?;
            let w = mlx::random::normal()
                .shape(&[2304][..])
                .key(&mlx::random::split(&key)?.0)
                .sample()?
                .astype(Dtype::Bfloat16)?
                .astype(Dtype::Float32)?;
            let x32 = x.astype(Dtype::Float32)?;
            let cases = [
                (
                    rms_cast(&x, None, 1e-6, Dtype::Float32, t)?,
                    mlx::fast::rms_norm_on(&x32, None, 1e-6, t)?,
                ),
                (
                    rms_cast(&x32, Some(&w), 1e-6, Dtype::Bfloat16, t)?,
                    mlx::fast::rms_norm_on(&x32, Some(&w), 1e-6, t)?.astype(Dtype::Bfloat16)?,
                ),
                (
                    rms_cast(&x32, Some(&w), 1e-6, Dtype::Float32, t)?,
                    mlx::fast::rms_norm_on(&x32, Some(&w), 1e-6, t)?,
                ),
            ];
            for (i, (actual, expected)) in cases.iter().enumerate() {
                let diffs = actual
                    .not_equal(expected)?
                    .astype(Dtype::Int32)?
                    .sum(mlx::ops::All, false)?
                    .item::<i32>()?;
                assert_eq!(diffs, 0, "rows={rows} case={i}");
            }
        }
        Ok(())
    }

    /// Throughput of the fused attention kernels on long keys. Prints only.
    #[test]
    #[ignore]
    #[serial_test::serial(mlx_metal)]
    fn attention_kernel_throughput() -> Result<()> {
        let t = metal_stream()?;
        let rand = |shape: &[i32], seed, dtype| -> anyhow::Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(seed)?)
                .sample()?
                .astype(dtype)?)
        };
        let (heads, kv, d, extent, queries) = (24, 8, 128, 1024, 2048);
        let scale = (d as f32).powf(-0.5);
        let time = |run: &dyn Fn() -> anyhow::Result<Array>| -> anyhow::Result<f64> {
            let mut best = f64::MAX;
            for _ in 0..6 {
                let out = run()?;
                let start = std::time::Instant::now();
                mlx::transforms::eval(&[&out])?;
                best = best.min(start.elapsed().as_secs_f64());
            }
            Ok(best)
        };
        for (keys, dtype) in [
            (16384, Dtype::Bfloat16),
            (16384, Dtype::Float16),
            (16384, Dtype::Float32),
            (61440, Dtype::Bfloat16),
            (61440, Dtype::Float16),
        ] {
            let q = rand(&[1, heads, queries, d], 11, dtype)?;
            let k = rand(&[1, kv, keys, d], 12, dtype)?;
            let v = rand(&[1, kv, keys, d], 13, dtype)?;
            let bias = rand(&[1, heads, queries, extent], 14, dtype)?;
            let sinks = rand(&[1, heads, queries], 15, dtype)?;
            mlx::transforms::eval(&[&q, &k, &v, &bias, &sinks])?;
            let band = time(&|| {
                Ok(mlx::fast::scaled_dot_product_attention_on(
                    &q,
                    &k,
                    &v,
                    scale,
                    "band",
                    Some(&bias),
                    Some(&sinks),
                    t,
                )?)
            })?;
            let causal = time(&|| {
                Ok(mlx::fast::scaled_dot_product_attention_on(
                    &q, &k, &v, scale, "causal", None, None, t,
                )?)
            })?;
            // The scores alone as one plain matrix product per KV head.
            let qg = q.reshape_on(&[kv, 3 * queries, d][..], t)?;
            let kt = k
                .reshape_on(&[kv, keys, d][..], t)?
                .transpose_axes_on([0, 2, 1], t)?;
            mlx::transforms::eval(&[&qg, &kt])?;
            let product = time(&|| Ok(qg.matmul_on(&kt, t)?.sum(mlx::ops::All, false)?))?;
            // What a full additive mask costs on top (one value per query
            // and key), as the reference implementation passes it.
            let masked = if keys <= 16384 {
                let mask = rand(&[1, heads, queries, keys], 16, dtype)?;
                mlx::transforms::eval(&[&mask])?;
                time(&|| {
                    Ok(mlx::fast::scaled_dot_product_attention_on(
                        &q,
                        &k,
                        &v,
                        scale,
                        "",
                        Some(&mask),
                        None,
                        t,
                    )?)
                })?
            } else {
                f64::NAN
            };
            // One decode step: a single query with a bias per key and a
            // sink per head, and the time to merely read K and V once.
            let q1 = rand(&[1, heads, 1, d], 17, dtype)?;
            let bias1 = rand(&[1, heads, 1, keys], 18, dtype)?;
            let sinks1 = rand(&[heads], 19, dtype)?;
            mlx::transforms::eval(&[&q1, &bias1, &sinks1])?;
            let step = time(&|| {
                Ok(mlx::fast::scaled_dot_product_attention_on(
                    &q1,
                    &k,
                    &v,
                    scale,
                    "",
                    Some(&bias1),
                    Some(&sinks1),
                    t,
                )?)
            })?;
            let read = time(&|| {
                Ok(k.sum(mlx::ops::All, false)?
                    .try_add_on(&v.sum(mlx::ops::All, false)?, t)?)
            })?;
            let kv_bytes = 2.0 * (kv * keys * d) as f64 * dtype.byte_size() as f64;
            println!(
                "{dtype:?} keys {keys:6}: decode step {:6.2} ms ({:5.1} GB/s of K+V)  read K+V once {:6.2} ms ({:5.1} GB/s)  full-mask prefill {:7.1} ms",
                step * 1e3,
                kv_bytes / step / 1e9,
                read * 1e3,
                kv_bytes / read / 1e9,
                masked * 1e3,
            );
            // Two products of heads x queries x keys x d multiply-adds.
            let flops = 2.0 * 2.0 * (heads * queries) as f64 * keys as f64 * d as f64;
            println!(
                "{dtype:?} keys {keys:6}: band {:7.1} ms ({:5.2} TFLOP/s)  causal {:7.1} ms ({:5.2})  scores-only matmul {:7.1} ms ({:5.2})",
                band * 1e3,
                flops / band / 1e12,
                causal * 1e3,
                flops / causal / 1e12,
                product * 1e3,
                flops / 2.0 / product / 1e12,
            );
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn banded_attention_matches_the_dense_mask() -> Result<()> {
        let t = metal_stream()?;
        let rand = |shape: &[i32], seed, dtype| -> anyhow::Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(seed)?)
                .sample()?
                .astype(dtype)?)
        };
        let (heads, kv, d, extent) = (24, 8, 128, 1024);
        let scale = (d as f32).powf(-0.5);
        // (queries, keys, window): unaligned and aligned blocks, keys inside
        // and far beyond the band, and a sliding window.
        for (queries, keys, window) in [
            // Eight queries or fewer take MLX's unfused graph.
            (4, 40, 0),
            (8, 1500, 0),
            (8, 1031, 1024),
            (64, 64, 0),
            (96, 200, 0),
            (70, 1500, 0),
            (128, 4096, 0),
            (96, 1119, 1024),
            (64, 1087, 1024),
        ] {
            let q = rand(&[1, heads, queries, d], 11, Dtype::Bfloat16)?;
            let k = rand(&[1, kv, keys, d], 12, Dtype::Bfloat16)?;
            let v = rand(&[1, kv, keys, d], 13, Dtype::Bfloat16)?;
            let bias = rand(&[1, heads, queries, extent], 14, Dtype::Bfloat16)?;
            let sinks = rand(&[1, heads, queries], 15, Dtype::Float32)?;
            let banded = mlx::fast::scaled_dot_product_attention_on(
                &q,
                &k,
                &v,
                scale,
                if window > 0 { "band_window" } else { "band" },
                Some(&bias),
                Some(&sinks.astype(Dtype::Bfloat16)?),
                t,
            )?;
            let mask = relative_mask(&bias, &sinks, keys, window, t)?;
            let zero = Array::zeros((1, kv, 1, d), Dtype::Bfloat16)?;
            let dense = mlx::fast::scaled_dot_product_attention_on(
                &q,
                &mlx::ops::shape::concatenate_on(&[&k, &zero], 2, t)?,
                &mlx::ops::shape::concatenate_on(&[&v, &zero], 2, t)?,
                scale,
                "",
                Some(&mask),
                None,
                t,
            )?;
            let a = banded.astype(Dtype::Float32)?;
            let b = dense.astype(Dtype::Float32)?;
            let error = a
                .try_sub_on(&b, t)?
                .abs()?
                .max(mlx::ops::All, false)?
                .item::<f32>()?;
            let size = b.abs()?.max(mlx::ops::All, false)?.item::<f32>()?;
            // The unfused graph multiplies in BF16; the kernels accumulate
            // in FP32.
            let tolerance = if queries <= 8 { 0.05 } else { 0.02 };
            assert!(
                error <= tolerance * size,
                "queries={queries} keys={keys} window={window} error={error} size={size}"
            );
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn decode_scores_match_mlx_reduction() -> Result<()> {
        let t = metal_stream()?;
        let rand = |shape: &[i32], seed| -> anyhow::Result<Array> {
            Ok(mlx::random::normal()
                .shape(shape)
                .key(&mlx::random::key(seed)?)
                .sample()?
                .astype(Dtype::Bfloat16)?)
        };
        let q = rand(&[1, 1, 24, 128], 123)?;
        let bias = rand(&[1, 1, 24, 64], 127)?.transpose_axes_on([0, 2, 1, 3], t)?;
        let sinks = rand(&[1, 24, 1], 207)?.astype(Dtype::Float32)?;
        for (n, window) in [
            (9, 0),
            (47, 0),
            (55, 0),
            (128, 64),
            (240, 0),
            (1024, 1024),
            (4020, 0),
            (4097, 0),
            (32020, 0),
        ] {
            let k = rand(&[1, 8, n + 11, 128], 332)?;
            let k = super::super::ops::slice(&k, 2, 7, n + 7, t)?;
            let q32 = q
                .transpose_axes_on([0, 2, 1, 3], t)?
                .astype_on(Dtype::Float32, t)?
                .reshape_on(&[1, 8, 3, 1, 128][..], t)?;
            let kt = k
                .astype_on(Dtype::Float32, t)?
                .transpose_axes_on([0, 1, 3, 2], t)?
                .expand_dims_on(2, t)?;
            let scores = super::super::ops::scale(&q32.matmul_on(&kt, t)?, (128f32).powf(-0.5), t)?;
            let reference = relative_logits(&scores, &bias, &sinks, window, t)?;
            let actual = decode_relative_logits(&q, &k, &bias, &sinks, window, t)?;
            let mismatch = actual
                .not_equal(&reference)?
                .astype(Dtype::Int32)?
                .sum(mlx::ops::All, false)?
                .item::<i32>()?;
            assert_eq!(mismatch, 0, "n={n}");
        }
        Ok(())
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn routed_mix_matches_native_order_and_rounding() -> Result<()> {
        let t = metal_stream()?;
        let indices: Array = (&[7u32, 25, 1, 63, 82, 2, 127, 31][..], &[1, 1, 8][..]).try_into()?;
        for seed in [3u64, 27, 1341, 78554] {
            let experts = mlx::random::normal()
                .shape(&[1, 1, 8, 2304][..])
                .key(&mlx::random::key(seed)?)
                .sample()?
                .astype(Dtype::Bfloat16)?;
            let scores = mlx::random::normal()
                .shape(&[1, 1, 128][..])
                .key(&mlx::random::key(seed + 1)?)
                .sample()?
                .sigmoid_on(t)?;
            let selected = indexing::take_along_axis_on(&scores, &indices, -1, t)?;
            let denom = selected
                .sum_on(-1, true, t)?
                .try_add_on(&super::super::ops::scalar(1e-20)?, t)?;
            for scale in [1.0, 1.25, 2.5] {
                let normalized =
                    super::super::ops::scale(&selected.try_div_on(&denom, t)?, scale, t)?
                        .expand_dims_on(-1, t)?;
                let reference = experts
                    .astype(Dtype::Float32)?
                    .try_mul_on(&normalized, t)?
                    .sum_on(-2, false, t)?
                    .astype(Dtype::Bfloat16)?;
                let actual = mix_routed_decode(&experts, &scores, &indices, scale, t)?;
                let mismatch = actual
                    .not_equal(&reference)?
                    .astype(Dtype::Int32)?
                    .sum(mlx::ops::All, false)?
                    .item::<i32>()?;
                assert_eq!(mismatch, 0, "seed={seed} route_scale={scale}");
            }
        }
        Ok(())
    }
}
