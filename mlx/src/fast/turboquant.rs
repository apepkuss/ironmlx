//! TurboQuant GPU primitives built on MLX custom Metal kernels.

use std::{sync::OnceLock, time::Instant};

use crate::ops::unary::softmax_on;
use crate::{Array, Dtype, Error, MetalKernel, Result, Shape, StreamOrDevice};

pub const TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD: i32 = 128;
pub const TURBOQUANT_MULTIROW_MAX_QUERY_ROWS: i32 = 4;
pub const TURBOQUANT_MULTIROW_MIN_SEQ_LEN: i32 = 2048;
const PARALLEL_DECODE_V_CHUNK_SIZE: i32 = 256;
const QK_SIMDGROUPS_PER_THREADGROUP: i32 = 4;
const QK_POSITIONS_PER_SIMDGROUP: i32 = 4;
const V_CHUNK_DIMS_PER_THREADGROUP: i32 = 16;
const V_Q_HEADS_PER_THREADGROUP: i32 = 4;

fn weighted_v_chunk_branchless_shape_supported(q_per_kv: i32, head_dim: i32) -> bool {
    q_per_kv == V_Q_HEADS_PER_THREADGROUP && head_dim % V_CHUNK_DIMS_PER_THREADGROUP == 0
}

#[derive(Clone, Copy)]
struct TurboquantAttnProfileEvent<'a> {
    stage: &'a str,
    elapsed_us: u128,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    k_bits: u8,
    v_bits: u8,
    v_chunks: i32,
}

fn format_turboquant_attn_profile_line(event: TurboquantAttnProfileEvent<'_>) -> String {
    format!(
        "{{\"event\":\"turboquant_attn_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"batch\":{},\"q_heads\":{},\"kv_heads\":{},\"seq_len\":{},\"head_dim\":{},\"k_bits\":{},\"v_bits\":{},\"v_chunks\":{}}}",
        event.stage,
        event.elapsed_us,
        event.batch,
        event.q_heads,
        event.kv_heads,
        event.seq_len,
        event.head_dim,
        event.k_bits,
        event.v_bits,
        event.v_chunks
    )
}

#[derive(Clone, Copy)]
struct TurboquantAttnProfile {
    enabled: bool,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    k_bits: u8,
    v_bits: u8,
    v_chunks: i32,
}

impl TurboquantAttnProfile {
    fn start(self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn eval_stage(
        self,
        stage: &'static str,
        arrays: &[&Array],
        start: Option<Instant>,
    ) -> Result<()> {
        let Some(start) = start else {
            return Ok(());
        };
        crate::transforms::eval(arrays)?;
        eprintln!(
            "{}",
            format_turboquant_attn_profile_line(TurboquantAttnProfileEvent {
                stage,
                elapsed_us: start.elapsed().as_micros(),
                batch: self.batch,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                seq_len: self.seq_len,
                head_dim: self.head_dim,
                k_bits: self.k_bits,
                v_bits: self.v_bits,
                v_chunks: self.v_chunks,
            })
        );
        Ok(())
    }
}

const TURBO_QUANTIZE_SOURCE: &str = r#"
uint vec_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

threadgroup float values[HEAD_DIM];
threadgroup uint indices[HEAD_DIM];

uint input_base = vec_idx * HEAD_DIM;
uint packed_base = vec_idx * PACKED_DIM;

float x_val = (float)x[input_base + lid];
values[lid] = x_val * x_val;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
        values[lid] += values[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lid == 0) {
    values[0] = sqrt(values[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float norm = values[0];
float safe_norm = norm > 1.0e-10f ? norm : 1.0f;
values[lid] = (x_val / safe_norm) * (float)signs[lid];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float rotated = values[lid] / sqrt((float)HEAD_DIM);
uint idx = 0;
for (uint c = 0; c < LEVELS - 1; ++c) {
    float boundary = 0.5f * ((float)codebook[c] + (float)codebook[c + 1]);
    if (rotated > boundary) {
        idx += 1;
    } else {
        break;
    }
}
indices[lid] = idx;

float centroid = (float)codebook[idx];
values[lid] = centroid * centroid;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
        values[lid] += values[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lid == 0) {
    float recon_norm = sqrt(values[0]);
    float safe_recon_norm = recon_norm > 1.0e-10f ? recon_norm : 1.0e-10f;
    norms[vec_idx] = norm / safe_recon_norm;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lid < PACKED_DIM) {
    uint word = 0;
    for (uint i = 0; i < VALUES_PER_WORD; ++i) {
        uint src = lid * VALUES_PER_WORD + i;
        if (src < HEAD_DIM) {
            word |= (indices[src] & ((1u << BITS) - 1u)) << (i * BITS);
        }
    }
    packed[packed_base + lid] = word;
}
"#;

const TURBO_DEQUANTIZE_SOURCE: &str = r#"
uint vec_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

threadgroup float values[HEAD_DIM];

uint packed_base = vec_idx * PACKED_DIM;
uint word_idx = lid / VALUES_PER_WORD;
uint word_offset = lid - word_idx * VALUES_PER_WORD;
uint word = packed[packed_base + word_idx];
uint idx = (word >> (word_offset * BITS)) & ((1u << BITS) - 1u);

values[lid] = (float)codebook[idx] * (float)norms[vec_idx];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float recovered = (values[lid] / sqrt((float)HEAD_DIM)) * (float)signs[lid];
y[vec_idx * HEAD_DIM + lid] = static_cast<__typeof__(*y)>(recovered);
"#;

const TURBOQUANT_SDPA_DECODE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

uint q_head = group_idx % Q_HEADS;
uint batch = group_idx / Q_HEADS;
uint kv_head = q_head / Q_PER_KV;

threadgroup float q_rot[HEAD_DIM];
threadgroup float scratch[HEAD_DIM];
threadgroup float score_value;
threadgroup float max_value;
threadgroup float denom_value;
threadgroup float weight_value;

uint q_base = ((batch * Q_HEADS + q_head) * HEAD_DIM);
q_rot[lid] = (float)queries[q_base + lid] * (float)k_signs[lid];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = q_rot[left];
        float b = q_rot[right];
        q_rot[left] = a + b;
        q_rot[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
q_rot[lid] = q_rot[lid] / sqrt((float)HEAD_DIM);
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lid == 0) {
    max_value = -3.4028234663852886e38f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint pos = 0; pos < SEQ_LEN; ++pos) {
    uint k_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    uint k_word_idx = lid / K_VALUES_PER_WORD;
    uint k_word_offset = lid - k_word_idx * K_VALUES_PER_WORD;
    uint k_word = k_packed[k_vec * K_PACKED_DIM + k_word_idx];
    uint k_idx = (k_word >> (k_word_offset * K_BITS)) & ((1u << K_BITS) - 1u);
    scratch[lid] = q_rot[lid] * (float)k_codebook[k_idx] * (float)k_norms[k_vec];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        float mask_value = 0.0f;
        if (HAS_MASK) {
            uint mask_head = MASK_HEADS == 1 ? 0 : q_head;
            uint mask_idx = ((batch * MASK_HEADS + mask_head) * SEQ_LEN + pos);
            mask_value = (float)mask_arr[mask_idx];
        }
        score_value = scratch[0] * (float)scale_arr + mask_value;
        max_value = max(max_value, score_value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lid == 0) {
    denom_value = 0.0f;
}
float acc = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint pos = 0; pos < SEQ_LEN; ++pos) {
    uint k_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    uint k_word_idx = lid / K_VALUES_PER_WORD;
    uint k_word_offset = lid - k_word_idx * K_VALUES_PER_WORD;
    uint k_word = k_packed[k_vec * K_PACKED_DIM + k_word_idx];
    uint k_idx = (k_word >> (k_word_offset * K_BITS)) & ((1u << K_BITS) - 1u);
    scratch[lid] = q_rot[lid] * (float)k_codebook[k_idx] * (float)k_norms[k_vec];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        float mask_value = 0.0f;
        if (HAS_MASK) {
            uint mask_head = MASK_HEADS == 1 ? 0 : q_head;
            uint mask_idx = ((batch * MASK_HEADS + mask_head) * SEQ_LEN + pos);
            mask_value = (float)mask_arr[mask_idx];
        }
        score_value = scratch[0] * (float)scale_arr + mask_value;
        weight_value = exp(score_value - max_value);
        denom_value += weight_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint v_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    uint v_word_idx = lid / V_VALUES_PER_WORD;
    uint v_word_offset = lid - v_word_idx * V_VALUES_PER_WORD;
    uint v_word = v_packed[v_vec * V_PACKED_DIM + v_word_idx];
    uint v_idx = (v_word >> (v_word_offset * V_BITS)) & ((1u << V_BITS) - 1u);
    acc += weight_value * (float)v_codebook[v_idx] * (float)v_norms[v_vec];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float safe_denom = denom_value > 1.0e-20f ? denom_value : 1.0f;
q_rot[lid] = acc / safe_denom;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = q_rot[left];
        float b = q_rot[right];
        q_rot[left] = a + b;
        q_rot[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float recovered = (q_rot[lid] / sqrt((float)HEAD_DIM)) * (float)v_signs[lid];
out[q_base + lid] = static_cast<__typeof__(*out)>(recovered);
"#;

const TURBOQUANT_QUERY_ROTATE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

threadgroup float values[HEAD_DIM];

uint q_base = group_idx * HEAD_DIM;
values[lid] = (float)queries[q_base + lid] * (float)k_signs[lid];
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

q_rot[q_base + lid] = values[lid] / sqrt((float)HEAD_DIM);
"#;

const TURBOQUANT_SDPA_MULTIROW_PASS1_SOURCE: &str = r#"
uint kv_head = threadgroup_position_in_grid.x;
uint batch = threadgroup_position_in_grid.y;
uint block = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup;
uint repeat = thread_position_in_threadgroup.y;
uint q_head = kv_head * Q_PER_KV + repeat;

thread float q_values[Q_ROWS][ELEMENTS_PER_LANE];
thread float out_values[Q_ROWS][ELEMENTS_PER_LANE];
thread float max_scores[Q_ROWS];
thread float sum_exp_scores[Q_ROWS];

#pragma clang loop unroll(full)
for (uint row = 0; row < Q_ROWS; ++row) {
    uint q_base = (((batch * Q_HEADS + q_head) * Q_ROWS + row) * HEAD_DIM);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
        uint dim = lane + i * 32;
        q_values[row][i] = (float)q_rot[q_base + dim];
        out_values[row][i] = 0.0f;
    }
    max_scores[row] = -INFINITY;
    sum_exp_scores[row] = 0.0f;
}

for (uint pos = block; pos < SEQ_LEN; pos += BLOCKS) {
    uint kv_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    float k_norm = (float)k_norms[kv_vec];
    float v_norm = (float)v_norms[kv_vec];
    thread float k_values[ELEMENTS_PER_LANE];
    thread float v_values[ELEMENTS_PER_LANE];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
        uint dim = lane + i * 32;
        uint k_word_idx = dim / K_VALUES_PER_WORD;
        uint k_word_offset = dim - k_word_idx * K_VALUES_PER_WORD;
        uint k_word = k_packed[kv_vec * K_PACKED_DIM + k_word_idx];
        uint k_idx = (k_word >> (k_word_offset * K_BITS)) & ((1u << K_BITS) - 1u);
        k_values[i] = (float)k_codebook[k_idx] * k_norm;

        uint v_word_idx = dim / V_VALUES_PER_WORD;
        uint v_word_offset = dim - v_word_idx * V_VALUES_PER_WORD;
        uint v_word = v_packed[kv_vec * V_PACKED_DIM + v_word_idx];
        uint v_idx = (v_word >> (v_word_offset * V_BITS)) & ((1u << V_BITS) - 1u);
        v_values[i] = (float)v_codebook[v_idx] * v_norm;
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < Q_ROWS; ++row) {
        float dot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
            dot += q_values[row][i] * k_values[i];
        }
        float score = simd_sum(dot) * (float)scale_arr;
        uint query_len = (uint)query_lens[batch];
        uint kv_len = (uint)kv_lens[batch];
        bool visible = query_len > 0 && query_len <= Q_ROWS
            && kv_len >= query_len && kv_len <= SEQ_LEN
            && row < query_len && pos <= kv_len - query_len + row;
        if (HAS_MASK) {
            uint mask_head = MASK_HEADS == 1 ? 0 : q_head;
            uint mask_row = MASK_ROWS == 1 ? 0 : row;
            uint mask_idx = (((batch * MASK_HEADS + mask_head) * MASK_ROWS + mask_row) * SEQ_LEN + pos);
            float mask_value = (float)mask_arr[mask_idx];
            score += mask_value;
            visible = visible && mask_value > -1.0e30f;
        }

        if (visible) {
            float new_max = max(max_scores[row], score);
            float old_factor = fast::exp(max_scores[row] - new_max);
            float new_factor = fast::exp(score - new_max);
            max_scores[row] = new_max;
            sum_exp_scores[row] = sum_exp_scores[row] * old_factor + new_factor;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
                out_values[row][i] = out_values[row][i] * old_factor
                    + new_factor * v_values[i];
            }
        }
    }
}

#pragma clang loop unroll(full)
for (uint row = 0; row < Q_ROWS; ++row) {
    uint output_row = ((batch * Q_HEADS + q_head) * Q_ROWS + row);
    uint partial_row = output_row * BLOCKS + block;
    if (lane == 0) {
        partial_sums[partial_row] = sum_exp_scores[row];
        partial_maxs[partial_row] = max_scores[row];
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
        uint dim = lane + i * 32;
        partial_acc[partial_row * HEAD_DIM + dim] = out_values[row][i];
    }
}
"#;

const TURBOQUANT_SDPA_MULTIROW_PASS2_SOURCE: &str = r#"
uint output_row = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;

threadgroup float values[HEAD_DIM];
threadgroup float reduce_scratch[1024];
threadgroup float simd_maxs[32];
threadgroup float simd_sums[32];
threadgroup float global_max;
threadgroup float global_sum;

thread float acc[ELEMENTS_PER_LANE];
#pragma clang loop unroll(full)
for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
    acc[i] = 0.0f;
}
float local_max = -INFINITY;
float local_sum = 0.0f;
for (uint block = simd_gid; block < BLOCKS; block += 32) {
    uint partial_row = output_row * BLOCKS + block;
    float block_max = (float)partial_maxs[partial_row];
    float block_sum = (float)partial_sums[partial_row];
    if (block_max <= -1.0e30f) {
        continue;
    }
    float new_max = max(local_max, block_max);
    float old_factor = fast::exp(local_max - new_max);
    float block_factor = fast::exp(block_max - new_max);
    local_sum = local_sum * old_factor + block_sum * block_factor;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
        uint dim = simd_lid + i * 32;
        acc[i] = acc[i] * old_factor
            + (float)partial_acc[partial_row * HEAD_DIM + dim] * block_factor;
    }
    local_max = new_max;
}

if (simd_lid == 0) {
    simd_maxs[simd_gid] = local_max;
    simd_sums[simd_gid] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_gid == 0) {
    float group_max = simd_maxs[simd_lid];
    float row_max = simd_max(group_max);
    float factor = group_max > -1.0e30f ? fast::exp(group_max - row_max) : 0.0f;
    float row_sum = simd_sum(simd_sums[simd_lid] * factor);
    if (simd_lid == 0) {
        global_max = row_max;
        global_sum = row_sum;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float local_factor = local_max > -1.0e30f ? fast::exp(local_max - global_max) : 0.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < ELEMENTS_PER_LANE; ++i) {
    reduce_scratch[simd_lid * 32 + simd_gid] = acc[i] * local_factor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint group = 0; group < 32; ++group) {
            total += reduce_scratch[simd_lid * 32 + group];
        }
        uint dim = simd_lid + i * 32;
        values[dim] = global_sum > 1.0e-20f ? total / global_sum : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lid < HEAD_DIM) {
    float recovered = (values[lid] / sqrt((float)HEAD_DIM)) * (float)v_signs[lid];
    out[output_row * HEAD_DIM + lid] = static_cast<__typeof__(*out)>(recovered);
}
"#;

const TURBOQUANT_QK_DECODE_SOURCE: &str = r#"
uint tile_idx = threadgroup_position_in_grid.y;
uint sgid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint block_idx = tile_idx * QK_SIMDGROUPS + sgid;
uint blocks_per_head = (SEQ_LEN + QK_POSITIONS_PER_SIMDGROUP - 1) / QK_POSITIONS_PER_SIMDGROUP;
uint total_blocks = BATCH * Q_HEADS * blocks_per_head;
if (block_idx >= total_blocks) {
    return;
}

uint pos_block = block_idx % blocks_per_head;
uint q_head = (block_idx / blocks_per_head) % Q_HEADS;
uint batch = block_idx / (blocks_per_head * Q_HEADS);
uint kv_head = q_head / Q_PER_KV;
uint pos_base = pos_block * QK_POSITIONS_PER_SIMDGROUP;

uint q_base = (batch * Q_HEADS + q_head) * HEAD_DIM;
thread float acc[QK_POSITIONS_PER_SIMDGROUP];
for (uint i = 0; i < QK_POSITIONS_PER_SIMDGROUP; ++i) {
    acc[i] = 0.0f;
}

for (uint dim = lane; dim < HEAD_DIM; dim += 32) {
    float q_value = (float)q_rot[q_base + dim];
    uint k_word_idx = dim / K_VALUES_PER_WORD;
    uint k_word_offset = dim - k_word_idx * K_VALUES_PER_WORD;
    for (uint i = 0; i < QK_POSITIONS_PER_SIMDGROUP; ++i) {
        uint pos = pos_base + i;
        if (pos < SEQ_LEN) {
            uint k_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
            uint k_word = k_packed[k_vec * K_PACKED_DIM + k_word_idx];
            uint k_idx = (k_word >> (k_word_offset * K_BITS)) & ((1u << K_BITS) - 1u);
            acc[i] += q_value * (float)k_codebook[k_idx];
        }
    }
}

thread float score_acc[QK_POSITIONS_PER_SIMDGROUP];
for (uint i = 0; i < QK_POSITIONS_PER_SIMDGROUP; ++i) {
    score_acc[i] = simd_sum(acc[i]);
}

if (lane == 0) {
    for (uint i = 0; i < QK_POSITIONS_PER_SIMDGROUP; ++i) {
        uint pos = pos_base + i;
        if (pos < SEQ_LEN) {
            uint k_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
            float score = score_acc[i] * (float)k_norms[k_vec];
            float mask_value = 0.0f;
            if (HAS_MASK) {
                uint mask_head = MASK_HEADS == 1 ? 0 : q_head;
                uint mask_idx = ((batch * MASK_HEADS + mask_head) * SEQ_LEN + pos);
                mask_value = (float)mask_arr[mask_idx];
            }
            uint score_idx = ((batch * Q_HEADS + q_head) * SEQ_LEN + pos);
            scores[score_idx] = score * (float)scale_arr + mask_value;
        }
    }
}
"#;

const TURBOQUANT_WEIGHTED_V_DECODE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

uint q_head = group_idx % Q_HEADS;
uint batch = group_idx / Q_HEADS;
uint kv_head = q_head / Q_PER_KV;

threadgroup float values[HEAD_DIM];

float acc = 0.0f;
for (uint pos = 0; pos < SEQ_LEN; ++pos) {
    uint weight_idx = ((batch * Q_HEADS + q_head) * SEQ_LEN + pos);
    float weight = (float)weights[weight_idx];
    uint v_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    uint v_word_idx = lid / V_VALUES_PER_WORD;
    uint v_word_offset = lid - v_word_idx * V_VALUES_PER_WORD;
    uint v_word = v_packed[v_vec * V_PACKED_DIM + v_word_idx];
    uint v_idx = (v_word >> (v_word_offset * V_BITS)) & ((1u << V_BITS) - 1u);
    acc += weight * (float)v_codebook[v_idx] * (float)v_norms[v_vec];
}

values[lid] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

uint out_base = ((batch * Q_HEADS + q_head) * HEAD_DIM);
float recovered = (values[lid] / sqrt((float)HEAD_DIM)) * (float)v_signs[lid];
out[out_base + lid] = static_cast<__typeof__(*out)>(recovered);
"#;

const TURBOQUANT_WEIGHTED_V_CHUNK_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint dim_group = group_idx % V_DIM_GROUPS;
uint chunk = (group_idx / V_DIM_GROUPS) % V_CHUNKS;
uint q_group = (group_idx / (V_DIM_GROUPS * V_CHUNKS)) % Q_GROUPS_PER_KV;
uint kv_head = (group_idx / (V_DIM_GROUPS * V_CHUNKS * Q_GROUPS_PER_KV)) % KV_HEADS;
uint batch = group_idx / (V_DIM_GROUPS * V_CHUNKS * Q_GROUPS_PER_KV * KV_HEADS);
uint pos = chunk * V_CHUNK_SIZE + lid;
uint dim_base = dim_group * V_DIMS_PER_GROUP;
uint q_offset_base = q_group * V_Q_HEADS_PER_GROUP;

threadgroup float scratch[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP * V_CHUNK_SIMDGROUPS];

thread float acc[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP];
#pragma clang loop unroll(full)
for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        acc[q * V_DIMS_PER_GROUP + i] = 0.0f;
    }
}
if (pos < SEQ_LEN) {
    uint v_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    float v_norm = (float)v_norms[v_vec];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        uint dim = dim_base + i;
        if (dim < HEAD_DIM) {
            uint v_word_idx = dim / V_VALUES_PER_WORD;
            uint v_word_offset = dim - v_word_idx * V_VALUES_PER_WORD;
            uint v_word = v_packed[v_vec * V_PACKED_DIM + v_word_idx];
            uint v_idx = (v_word >> (v_word_offset * V_BITS)) & ((1u << V_BITS) - 1u);
            float v_value = v_norm * (float)v_codebook[v_idx];
            #pragma clang loop unroll(full)
            for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
                uint q_offset = q_offset_base + q;
                if (q_offset < Q_PER_KV) {
                    uint q_head = kv_head * Q_PER_KV + q_offset;
                    uint weight_idx = ((batch * Q_HEADS + q_head) * SEQ_LEN + pos);
                    float weight = (float)weights[weight_idx];
                    acc[q * V_DIMS_PER_GROUP + i] = weight * v_value;
                }
            }
        }
    }
}

#pragma clang loop unroll(full)
for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        float simd_acc = simd_sum(acc[q * V_DIMS_PER_GROUP + i]);
        if (lane == 0) {
            uint scratch_idx = ((q * V_DIMS_PER_GROUP + i) * V_CHUNK_SIMDGROUPS + sgid);
            scratch[scratch_idx] = simd_acc;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sgid == 0) {
    #pragma clang loop unroll(full)
    for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
        uint q_offset = q_offset_base + q;
        if (q_offset < Q_PER_KV) {
            uint q_head = kv_head * Q_PER_KV + q_offset;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
                uint dim = dim_base + i;
                if (dim < HEAD_DIM) {
                    uint scratch_idx = ((q * V_DIMS_PER_GROUP + i) * V_CHUNK_SIMDGROUPS + lane);
                    float chunk_acc = lane < V_CHUNK_SIMDGROUPS ? scratch[scratch_idx] : 0.0f;
                    chunk_acc = simd_sum(chunk_acc);
                    if (lane == 0) {
                        uint partial_idx = (((batch * Q_HEADS + q_head) * V_CHUNKS + chunk) * HEAD_DIM + dim);
                        v_partial[partial_idx] = chunk_acc;
                    }
                }
            }
        }
    }
}
"#;

const TURBOQUANT_WEIGHTED_V_CHUNK_BRANCHLESS_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint dim_group = group_idx % V_DIM_GROUPS;
uint chunk = (group_idx / V_DIM_GROUPS) % V_CHUNKS;
uint kv_head = (group_idx / (V_DIM_GROUPS * V_CHUNKS)) % KV_HEADS;
uint batch = group_idx / (V_DIM_GROUPS * V_CHUNKS * KV_HEADS);
uint pos = chunk * V_CHUNK_SIZE + lid;
uint dim_base = dim_group * V_DIMS_PER_GROUP;

threadgroup float scratch[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP * V_CHUNK_SIMDGROUPS];

thread float acc[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP];
#pragma clang loop unroll(full)
for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        acc[q * V_DIMS_PER_GROUP + i] = 0.0f;
    }
}
if (pos < SEQ_LEN) {
    uint v_vec = ((batch * KV_HEADS + kv_head) * SEQ_LEN + pos);
    float v_norm = (float)v_norms[v_vec];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        uint dim = dim_base + i;
        uint v_word_idx = dim / V_VALUES_PER_WORD;
        uint v_word_offset = dim - v_word_idx * V_VALUES_PER_WORD;
        uint v_word = v_packed[v_vec * V_PACKED_DIM + v_word_idx];
        uint v_idx = (v_word >> (v_word_offset * V_BITS)) & ((1u << V_BITS) - 1u);
        float v_value = v_norm * (float)v_codebook[v_idx];
        #pragma clang loop unroll(full)
        for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
            uint q_head = kv_head * Q_PER_KV + q;
            uint weight_idx = ((batch * Q_HEADS + q_head) * SEQ_LEN + pos);
            float weight = (float)weights[weight_idx];
            acc[q * V_DIMS_PER_GROUP + i] = weight * v_value;
        }
    }
}

#pragma clang loop unroll(full)
for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
        float simd_acc = simd_sum(acc[q * V_DIMS_PER_GROUP + i]);
        if (lane == 0) {
            uint scratch_idx = ((q * V_DIMS_PER_GROUP + i) * V_CHUNK_SIMDGROUPS + sgid);
            scratch[scratch_idx] = simd_acc;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sgid == 0) {
    #pragma clang loop unroll(full)
    for (uint q = 0; q < V_Q_HEADS_PER_GROUP; ++q) {
        uint q_head = kv_head * Q_PER_KV + q;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < V_DIMS_PER_GROUP; ++i) {
            uint dim = dim_base + i;
            uint scratch_idx = ((q * V_DIMS_PER_GROUP + i) * V_CHUNK_SIMDGROUPS + lane);
            float chunk_acc = lane < V_CHUNK_SIMDGROUPS ? scratch[scratch_idx] : 0.0f;
            chunk_acc = simd_sum(chunk_acc);
            if (lane == 0) {
                uint partial_idx = (((batch * Q_HEADS + q_head) * V_CHUNKS + chunk) * HEAD_DIM + dim);
                v_partial[partial_idx] = chunk_acc;
            }
        }
    }
}
"#;

const TURBOQUANT_WEIGHTED_V_REDUCE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

uint q_head = group_idx % Q_HEADS;
uint batch = group_idx / Q_HEADS;

threadgroup float values[HEAD_DIM];

float acc = 0.0f;
for (uint chunk = 0; chunk < V_CHUNKS; ++chunk) {
    uint partial_idx = (((batch * Q_HEADS + q_head) * V_CHUNKS + chunk) * HEAD_DIM + lid);
    acc += (float)v_partial[partial_idx];
}

values[lid] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint width = 1; width < HEAD_DIM; width <<= 1) {
    uint pair = lid;
    if (pair < HEAD_DIM / 2) {
        uint block = pair / width;
        uint offset = pair - block * width;
        uint left = block * width * 2 + offset;
        uint right = left + width;
        float a = values[left];
        float b = values[right];
        values[left] = a + b;
        values[right] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

uint out_base = ((batch * Q_HEADS + q_head) * HEAD_DIM);
float recovered = (values[lid] / sqrt((float)HEAD_DIM)) * (float)v_signs[lid];
out[out_base + lid] = static_cast<__typeof__(*out)>(recovered);
"#;

fn cached_turbo_quantize_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turbo_quantize")
        .inputs(&["x", "signs", "codebook"])
        .outputs(&["packed", "norms"])
        .source(TURBO_QUANTIZE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turbo_dequantize_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turbo_dequantize")
        .inputs(&["packed", "norms", "signs", "codebook"])
        .outputs(&["y"])
        .source(TURBO_DEQUANTIZE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_sdpa_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_sdpa_decode")
        .inputs(&[
            "queries",
            "k_packed",
            "k_norms",
            "v_packed",
            "v_norms",
            "k_signs",
            "k_codebook",
            "v_signs",
            "v_codebook",
            "scale_arr",
            "mask_arr",
        ])
        .outputs(&["out"])
        .source(TURBOQUANT_SDPA_DECODE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_query_rotate_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_query_rotate")
        .inputs(&["queries", "k_signs"])
        .outputs(&["q_rot"])
        .source(TURBOQUANT_QUERY_ROTATE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_sdpa_multirow_pass1_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_sdpa_multirow_pass1")
        .inputs(&[
            "q_rot",
            "k_packed",
            "k_norms",
            "k_codebook",
            "v_packed",
            "v_norms",
            "v_codebook",
            "scale_arr",
            "query_lens",
            "kv_lens",
            "mask_arr",
        ])
        .outputs(&["partial_acc", "partial_sums", "partial_maxs"])
        .source(TURBOQUANT_SDPA_MULTIROW_PASS1_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_sdpa_multirow_pass2_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_sdpa_multirow_pass2")
        .inputs(&["partial_acc", "partial_sums", "partial_maxs", "v_signs"])
        .outputs(&["out"])
        .source(TURBOQUANT_SDPA_MULTIROW_PASS2_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_qk_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_qk_decode")
        .inputs(&[
            "q_rot",
            "k_packed",
            "k_norms",
            "k_codebook",
            "scale_arr",
            "mask_arr",
        ])
        .outputs(&["scores"])
        .source(TURBOQUANT_QK_DECODE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_weighted_v_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_weighted_v_decode")
        .inputs(&["weights", "v_packed", "v_norms", "v_signs", "v_codebook"])
        .outputs(&["out"])
        .source(TURBOQUANT_WEIGHTED_V_DECODE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_weighted_v_chunk_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_weighted_v_chunk")
        .inputs(&["weights", "v_packed", "v_norms", "v_codebook"])
        .outputs(&["v_partial"])
        .source(TURBOQUANT_WEIGHTED_V_CHUNK_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_weighted_v_chunk_branchless_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_weighted_v_chunk_branchless")
        .inputs(&["weights", "v_packed", "v_norms", "v_codebook"])
        .outputs(&["v_partial"])
        .source(TURBOQUANT_WEIGHTED_V_CHUNK_BRANCHLESS_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_turboquant_weighted_v_reduce_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_turboquant_weighted_v_reduce")
        .inputs(&["v_partial", "v_signs"])
        .outputs(&["out"])
        .source(TURBOQUANT_WEIGHTED_V_REDUCE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

/// Quantize the last dimension of `input` with TurboQuant.
///
/// Input shape is `[..., D]`, where `D` must be a power of two. `signs` is
/// `[D]`, `codebook` is `[8]` for 3-bit or `[16]` for 4-bit. The return value
/// is `(packed, adjusted_norms)`:
///
/// - `packed`: `[..., ceil(D / values_per_word)]` `uint32`
/// - `adjusted_norms`: `[...]` `float32`, storing `norm / recon_norm`
pub fn turbo_quantize(
    input: &Array,
    signs: &Array,
    codebook: &Array,
    bits: u8,
) -> Result<(Array, Array)> {
    turbo_quantize_on(input, signs, codebook, bits, ())
}

/// Stream-targeted variant of [`turbo_quantize`].
pub fn turbo_quantize_on(
    input: &Array,
    signs: &Array,
    codebook: &Array,
    bits: u8,
    target: impl Into<StreamOrDevice>,
) -> Result<(Array, Array)> {
    let input_shape = input.shape();
    let dims = input_shape.as_slice();
    if dims.is_empty() {
        return Err(Error::Mlx(
            "turbo_quantize: input must have at least one dimension".to_owned(),
        ));
    }

    let head_dim = *dims.last().expect("checked non-empty");
    if head_dim <= 0 {
        return Err(Error::Mlx(format!(
            "turbo_quantize: head_dim must be positive (got {head_dim})"
        )));
    }
    let head_dim_usize = head_dim as usize;
    if !head_dim_usize.is_power_of_two() {
        return Err(Error::Mlx(format!(
            "turbo_quantize: head_dim {head_dim} must be a power of two"
        )));
    }

    let (values_per_word, levels) = match bits {
        3 => (10_i32, 8_i32),
        4 => (8_i32, 16_i32),
        _ => {
            return Err(Error::Mlx(format!(
                "turbo_quantize: unsupported bit-width {bits} (expected 3 or 4)"
            )));
        }
    };

    let signs_shape = signs.shape();
    if signs_shape.as_slice() != [head_dim] {
        return Err(Error::Mlx(format!(
            "turbo_quantize: signs shape must be [{head_dim}] (got {signs_shape})"
        )));
    }

    let codebook_shape = codebook.shape();
    if codebook_shape.as_slice() != [levels] {
        return Err(Error::Mlx(format!(
            "turbo_quantize: codebook shape must be [{levels}] for {bits}-bit (got {codebook_shape})"
        )));
    }

    let vector_count = dims[..dims.len() - 1].iter().product::<i32>();
    let packed_dim = (head_dim + values_per_word - 1) / values_per_word;

    let mut packed_shape = dims[..dims.len() - 1].to_vec();
    packed_shape.push(packed_dim);
    let norms_shape = Shape::from(dims[..dims.len() - 1].to_vec());
    let packed_shape = Shape::from(packed_shape);

    let kernel = cached_turbo_quantize_kernel()?;
    let grid_x = vector_count * head_dim;
    let target = target.into();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[input, signs, codebook])
        .output_shapes(&[packed_shape, norms_shape])
        .output_dtypes(&[Dtype::Uint32, Dtype::Float32])
        .grid(grid_x, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("BITS", i32::from(bits))
        .template_int("VALUES_PER_WORD", values_per_word)
        .template_int("PACKED_DIM", packed_dim)
        .template_int("LEVELS", levels)
        .dispatch()?;

    let packed = outputs.take_at(0)?;
    let norms = outputs.take_at(0)?;
    Ok((packed, norms))
}

/// Dequantize TurboQuant packed data back to dense vectors.
///
/// Input shapes:
///
/// - `packed`: `[..., ceil(head_dim / values_per_word)]` `uint32`
/// - `norms`: `[...]` `float32`, produced by [`turbo_quantize`]
/// - `signs`: `[head_dim]`
/// - `codebook`: `[8]` for 3-bit or `[16]` for 4-bit
///
/// The return shape is `[..., head_dim]` with dtype `output_dtype`.
pub fn turbo_dequantize(
    packed: &Array,
    norms: &Array,
    signs: &Array,
    codebook: &Array,
    bits: u8,
    head_dim: i32,
    output_dtype: Dtype,
) -> Result<Array> {
    turbo_dequantize_on(
        packed,
        norms,
        signs,
        codebook,
        bits,
        head_dim,
        output_dtype,
        (),
    )
}

/// Stream-targeted variant of [`turbo_dequantize`].
#[allow(clippy::too_many_arguments)]
pub fn turbo_dequantize_on(
    packed: &Array,
    norms: &Array,
    signs: &Array,
    codebook: &Array,
    bits: u8,
    head_dim: i32,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if !matches!(
        output_dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: output_dtype must be f32, f16, or bf16 (got {output_dtype})"
        )));
    }
    if head_dim <= 0 {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: head_dim must be positive (got {head_dim})"
        )));
    }
    let head_dim_usize = head_dim as usize;
    if !head_dim_usize.is_power_of_two() {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: head_dim {head_dim} must be a power of two"
        )));
    }

    let (values_per_word, levels) = match bits {
        3 => (10_i32, 8_i32),
        4 => (8_i32, 16_i32),
        _ => {
            return Err(Error::Mlx(format!(
                "turbo_dequantize: unsupported bit-width {bits} (expected 3 or 4)"
            )));
        }
    };
    let packed_dim = (head_dim + values_per_word - 1) / values_per_word;

    let packed_shape = packed.shape();
    let packed_dims = packed_shape.as_slice();
    if packed_dims.is_empty() {
        return Err(Error::Mlx(
            "turbo_dequantize: packed must have at least one dimension".to_owned(),
        ));
    }
    let actual_packed_dim = *packed_dims.last().expect("checked non-empty");
    if actual_packed_dim != packed_dim {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: packed last dim must be {packed_dim} for head_dim={head_dim} bits={bits} (got {actual_packed_dim})"
        )));
    }

    let norms_shape = norms.shape();
    if norms_shape.as_slice() != &packed_dims[..packed_dims.len() - 1] {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: norms shape must match packed prefix {:?} (got {norms_shape})",
            &packed_dims[..packed_dims.len() - 1]
        )));
    }

    let signs_shape = signs.shape();
    if signs_shape.as_slice() != [head_dim] {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: signs shape must be [{head_dim}] (got {signs_shape})"
        )));
    }

    let codebook_shape = codebook.shape();
    if codebook_shape.as_slice() != [levels] {
        return Err(Error::Mlx(format!(
            "turbo_dequantize: codebook shape must be [{levels}] for {bits}-bit (got {codebook_shape})"
        )));
    }

    let vector_count = packed_dims[..packed_dims.len() - 1].iter().product::<i32>();
    let mut output_shape = packed_dims[..packed_dims.len() - 1].to_vec();
    output_shape.push(head_dim);
    let output_shape = Shape::from(output_shape);

    let kernel = cached_turbo_dequantize_kernel()?;
    let grid_x = vector_count * head_dim;
    let target = target.into();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[packed, norms, signs, codebook])
        .output_shapes(&[output_shape])
        .output_dtypes(&[output_dtype])
        .grid(grid_x, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("BITS", i32::from(bits))
        .template_int("VALUES_PER_WORD", values_per_word)
        .template_int("PACKED_DIM", packed_dim)
        .dispatch()?;

    outputs.take_at(0)
}

/// Causal multi-row attention over TurboQuant-packed K/V.
///
/// This path is specialized for MTP verification. `queries` must be
/// `[B, Hq, Q, D]` with `2 <= Q <= 4`; every K/V token is unpacked once per
/// GQA repeat and reused across all query rows. `query_lens` and `kv_lens` are
/// `[B]` int32 arrays describing each batch row's valid query count and total
/// cache length. Valid row `r` attends through position
/// `kv_lens[b] - query_lens[b] + r`; padded query rows produce zero output.
/// The output is `[B, Hq, Q, D]`.
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_multirow(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    query_lens: &Array,
    kv_lens: &Array,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
) -> Result<Array> {
    turboquant_sdpa_multirow_on(
        queries,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_signs,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        query_lens,
        kv_lens,
        mask_arr,
        output_dtype,
        (),
    )
}

/// Stream-targeted variant of [`turboquant_sdpa_multirow`].
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_multirow_on(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    query_lens: &Array,
    kv_lens: &Array,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    const OP: &str = "turboquant_sdpa_multirow";
    if !matches!(
        output_dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(Error::Mlx(format!(
            "{OP}: output_dtype must be f32, f16, or bf16 (got {output_dtype})"
        )));
    }

    let queries_shape = queries.shape();
    let queries_dims = queries_shape.as_slice();
    if queries_dims.len() != 4 {
        return Err(Error::Mlx(format!(
            "{OP}: queries must be rank-4 [B,Hq,Q,D] (got {queries_shape})"
        )));
    }
    let batch = queries_dims[0];
    let q_heads = queries_dims[1];
    let q_rows = queries_dims[2];
    let head_dim = queries_dims[3];
    if batch <= 0 || q_heads <= 0 {
        return Err(Error::Mlx(format!(
            "{OP}: batch and q_heads must be positive (got B={batch}, Hq={q_heads})"
        )));
    }
    if !(2..=TURBOQUANT_MULTIROW_MAX_QUERY_ROWS).contains(&q_rows) {
        return Err(Error::Mlx(format!(
            "{OP}: query rows must be in [2, {}] (got {q_rows})",
            TURBOQUANT_MULTIROW_MAX_QUERY_ROWS
        )));
    }
    validate_head_dim(OP, head_dim)?;
    if head_dim < 32 || head_dim % 32 != 0 {
        return Err(Error::Mlx(format!(
            "{OP}: head_dim must be a multiple of 32 and at least 32 (got {head_dim})"
        )));
    }

    let (k_values_per_word, k_levels) = bit_layout(OP, "k_bits", k_bits)?;
    let (v_values_per_word, v_levels) = bit_layout(OP, "v_bits", v_bits)?;
    let k_packed_dim = (head_dim + k_values_per_word - 1) / k_values_per_word;
    let v_packed_dim = (head_dim + v_values_per_word - 1) / v_values_per_word;

    let k_packed_shape = k_packed.shape();
    let k_packed_dims = k_packed_shape.as_slice();
    if k_packed_dims.len() != 4 {
        return Err(Error::Mlx(format!(
            "{OP}: k_packed must be rank-4 [B,Hkv,S,packed_D] (got {k_packed_shape})"
        )));
    }
    let kv_heads = k_packed_dims[1];
    let seq_len = k_packed_dims[2];
    if k_packed_dims[0] != batch
        || k_packed_dims[3] != k_packed_dim
        || seq_len < q_rows
        || kv_heads <= 0
    {
        return Err(Error::Mlx(format!(
            "{OP}: k_packed shape {k_packed_shape} is incompatible with queries {queries_shape}, k_bits={k_bits}"
        )));
    }
    if q_heads % kv_heads != 0 {
        return Err(Error::Mlx(format!(
            "{OP}: q_heads {q_heads} must be divisible by kv_heads {kv_heads}"
        )));
    }
    let q_per_kv = q_heads / kv_heads;
    if q_per_kv > 32 {
        return Err(Error::Mlx(format!(
            "{OP}: q_heads/kv_heads must not exceed 32 (got {q_per_kv})"
        )));
    }

    let expected_norms = [batch, kv_heads, seq_len];
    let k_norms_shape = k_norms.shape();
    if k_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "{OP}: k_norms shape must be {:?} (got {k_norms_shape})",
            expected_norms
        )));
    }
    let v_packed_shape = v_packed.shape();
    if v_packed_shape.as_slice() != [batch, kv_heads, seq_len, v_packed_dim] {
        return Err(Error::Mlx(format!(
            "{OP}: v_packed shape must be [{batch}, {kv_heads}, {seq_len}, {v_packed_dim}] for v_bits={v_bits} (got {v_packed_shape})"
        )));
    }
    let v_norms_shape = v_norms.shape();
    if v_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "{OP}: v_norms shape must be {:?} (got {v_norms_shape})",
            expected_norms
        )));
    }

    validate_vector_shape(OP, "k_signs", k_signs, head_dim)?;
    validate_vector_shape(OP, "v_signs", v_signs, head_dim)?;
    validate_codebook_shape(OP, "k_codebook", k_codebook, k_levels, k_bits)?;
    validate_codebook_shape(OP, "v_codebook", v_codebook, v_levels, v_bits)?;

    for (name, lens) in [("query_lens", query_lens), ("kv_lens", kv_lens)] {
        let lens_shape = lens.shape();
        if lens.dtype() != Dtype::Int32 || lens_shape.as_slice() != [batch] {
            return Err(Error::Mlx(format!(
                "{OP}: {name} must be int32 [{batch}] (got dtype={}, shape={lens_shape})",
                lens.dtype()
            )));
        }
    }

    let (mask_heads, mask_rows) = match mask_arr {
        Some(mask) => {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != batch
                || !(mask_dims[2] == 1 || mask_dims[2] == q_rows)
                || mask_dims[3] != seq_len
                || !(mask_dims[1] == 1 || mask_dims[1] == q_heads)
            {
                return Err(Error::Mlx(format!(
                    "{OP}: mask must be [B,1|Hq,1|Q,S] for B={batch}, Hq={q_heads}, Q={q_rows}, S={seq_len} (got {mask_shape})"
                )));
            }
            (mask_dims[1], mask_dims[2])
        }
        None => (1, 1),
    };

    let blocks = if seq_len <= 8_192 {
        64
    } else if seq_len <= 32_768 {
        128
    } else if seq_len <= 65_536 {
        256
    } else {
        512
    };
    let target = target.into();
    let q_rot_shape = Shape::from(vec![batch, q_heads, q_rows, head_dim]);
    let mut q_rot_outputs = cached_turboquant_query_rotate_kernel()?
        .dispatch_builder()
        .inputs(&[queries, k_signs])
        .output_shapes(&[q_rot_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(batch * q_heads * q_rows * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .dispatch()?;
    let q_rot = q_rot_outputs.take_at(0)?;

    let scale_arr: Array = (&[scale][..], &[][..]).try_into()?;
    let mask_input = mask_arr.unwrap_or(queries);
    let output_rows = batch * q_heads * q_rows;
    let partial_acc_shape = Shape::from(vec![output_rows, blocks, head_dim]);
    let partial_stats_shape = Shape::from(vec![output_rows, blocks]);
    let mut pass1_outputs = cached_turboquant_sdpa_multirow_pass1_kernel()?
        .dispatch_builder()
        .inputs(&[
            &q_rot, k_packed, k_norms, k_codebook, v_packed, v_norms, v_codebook, &scale_arr,
            query_lens, kv_lens, mask_input,
        ])
        .output_shapes(&[
            partial_acc_shape,
            partial_stats_shape.clone(),
            partial_stats_shape,
        ])
        .output_dtypes(&[Dtype::Float32, Dtype::Float32, Dtype::Float32])
        .grid(kv_heads * 32, batch * q_per_kv, blocks)
        .threadgroup(32, q_per_kv, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("ELEMENTS_PER_LANE", head_dim / 32)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("Q_ROWS", q_rows)
        .template_int("SEQ_LEN", seq_len)
        .template_int("BLOCKS", blocks)
        .template_int("K_BITS", i32::from(k_bits))
        .template_int("V_BITS", i32::from(v_bits))
        .template_int("K_VALUES_PER_WORD", k_values_per_word)
        .template_int("V_VALUES_PER_WORD", v_values_per_word)
        .template_int("K_PACKED_DIM", k_packed_dim)
        .template_int("V_PACKED_DIM", v_packed_dim)
        .template_bool("HAS_MASK", mask_arr.is_some())
        .template_int("MASK_HEADS", mask_heads)
        .template_int("MASK_ROWS", mask_rows)
        .dispatch()?;
    let partial_acc = pass1_outputs.take_at(0)?;
    let partial_sums = pass1_outputs.take_at(0)?;
    let partial_maxs = pass1_outputs.take_at(0)?;

    let output_shape = Shape::from(vec![batch, q_heads, q_rows, head_dim]);
    let mut output = cached_turboquant_sdpa_multirow_pass2_kernel()?
        .dispatch_builder()
        .inputs(&[&partial_acc, &partial_sums, &partial_maxs, v_signs])
        .output_shapes(&[output_shape])
        .output_dtypes(&[output_dtype])
        .grid(output_rows * 1024, 1, 1)
        .threadgroup(1024, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("ELEMENTS_PER_LANE", head_dim / 32)
        .template_int("BLOCKS", blocks)
        .dispatch()?;
    output.take_at(0)
}

/// Fused decode attention over TurboQuant-packed K/V.
///
/// This is the decode-specialized counterpart to
/// [`scaled_dot_product_attention`](super::scaled_dot_product_attention):
/// `queries` must be `[B, Hq, 1, D]`, `k_packed`/`v_packed` must be
/// `[B, Hkv, S, packed_D]`, and `Hq` must be divisible by `Hkv`.
/// The output is `[B, Hq, 1, D]`.
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
) -> Result<Array> {
    turboquant_sdpa_decode_on_impl(
        queries,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_signs,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        (),
        false,
    )
}

/// Explicitly run the parallel long-sequence TurboQuant decode attention path.
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode_parallel(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
) -> Result<Array> {
    turboquant_sdpa_decode_parallel_on(
        queries,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_signs,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        (),
    )
}

/// Explicitly run the parallel long-sequence TurboQuant decode attention path
/// with queries already rotated into the TurboQuant key basis.
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode_parallel_pre_rotated(
    q_rot: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
) -> Result<Array> {
    turboquant_sdpa_decode_parallel_pre_rotated_on(
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        (),
    )
}

/// Stream-targeted variant of [`turboquant_sdpa_decode_parallel_pre_rotated`].
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode_parallel_pre_rotated_on(
    q_rot: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    turboquant_sdpa_decode_parallel_pre_rotated_on_impl(
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        target,
    )
}

/// Stream-targeted variant of [`turboquant_sdpa_decode_parallel`].
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode_parallel_on(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    turboquant_sdpa_decode_on_impl(
        queries,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_signs,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        target,
        true,
    )
}

/// Stream-targeted variant of [`turboquant_sdpa_decode`].
#[allow(clippy::too_many_arguments)]
pub fn turboquant_sdpa_decode_on(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    turboquant_sdpa_decode_on_impl(
        queries,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_signs,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        output_dtype,
        target,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_on_impl(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
    force_parallel: bool,
) -> Result<Array> {
    if !matches!(
        output_dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: output_dtype must be f32, f16, or bf16 (got {output_dtype})"
        )));
    }

    let queries_shape = queries.shape();
    let queries_dims = queries_shape.as_slice();
    if queries_dims.len() != 4 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: queries must be rank-4 [B,Hq,1,D] (got {queries_shape})"
        )));
    }
    let batch = queries_dims[0];
    let q_heads = queries_dims[1];
    let q_seq = queries_dims[2];
    let head_dim = queries_dims[3];
    if q_seq != 1 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: queries seq dim must be 1 (got {q_seq})"
        )));
    }
    validate_head_dim("turboquant_sdpa_decode", head_dim)?;

    let (k_values_per_word, k_levels) = bit_layout("turboquant_sdpa_decode", "k_bits", k_bits)?;
    let (v_values_per_word, v_levels) = bit_layout("turboquant_sdpa_decode", "v_bits", v_bits)?;
    let k_packed_dim = (head_dim + k_values_per_word - 1) / k_values_per_word;
    let v_packed_dim = (head_dim + v_values_per_word - 1) / v_values_per_word;

    let k_packed_shape = k_packed.shape();
    let k_packed_dims = k_packed_shape.as_slice();
    if k_packed_dims.len() != 4 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: k_packed must be rank-4 [B,Hkv,S,packed_D] (got {k_packed_shape})"
        )));
    }
    let kv_heads = k_packed_dims[1];
    let seq_len = k_packed_dims[2];
    if k_packed_dims[0] != batch
        || k_packed_dims[3] != k_packed_dim
        || seq_len <= 0
        || kv_heads <= 0
    {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: k_packed shape {k_packed_shape} is incompatible with queries {queries_shape}, k_bits={k_bits}"
        )));
    }
    if q_heads % kv_heads != 0 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: q_heads {q_heads} must be divisible by kv_heads {kv_heads}"
        )));
    }

    let expected_norms = [batch, kv_heads, seq_len];
    let k_norms_shape = k_norms.shape();
    if k_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: k_norms shape must be {:?} (got {k_norms_shape})",
            expected_norms
        )));
    }

    let v_packed_shape = v_packed.shape();
    let v_packed_dims = v_packed_shape.as_slice();
    if v_packed_dims != [batch, kv_heads, seq_len, v_packed_dim] {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: v_packed shape must be [{batch}, {kv_heads}, {seq_len}, {v_packed_dim}] for v_bits={v_bits} (got {v_packed_shape})"
        )));
    }
    let v_norms_shape = v_norms.shape();
    if v_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode: v_norms shape must be {:?} (got {v_norms_shape})",
            expected_norms
        )));
    }

    validate_vector_shape("turboquant_sdpa_decode", "k_signs", k_signs, head_dim)?;
    validate_vector_shape("turboquant_sdpa_decode", "v_signs", v_signs, head_dim)?;
    validate_codebook_shape(
        "turboquant_sdpa_decode",
        "k_codebook",
        k_codebook,
        k_levels,
        k_bits,
    )?;
    validate_codebook_shape(
        "turboquant_sdpa_decode",
        "v_codebook",
        v_codebook,
        v_levels,
        v_bits,
    )?;

    let mask_heads = match mask_arr {
        Some(mask) => {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != batch
                || mask_dims[2] != 1
                || mask_dims[3] != seq_len
                || !(mask_dims[1] == 1 || mask_dims[1] == q_heads)
            {
                return Err(Error::Mlx(format!(
                    "turboquant_sdpa_decode: mask must be [B,1,1,S] or [B,Hq,1,S] for B={batch}, Hq={q_heads}, S={seq_len} (got {mask_shape})"
                )));
            }
            mask_dims[1]
        }
        None => 1,
    };

    let scale_arr: Array = (&[scale][..], &[][..]).try_into()?;
    let mask_input = mask_arr.unwrap_or(queries);
    let output_shape = Shape::from(vec![batch, q_heads, 1, head_dim]);
    let target = target.into();
    if force_parallel || seq_len >= TURBOQUANT_PARALLEL_DECODE_SEQ_THRESHOLD {
        return turboquant_sdpa_decode_parallel_dispatch(
            queries,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            k_signs,
            k_codebook,
            v_signs,
            v_codebook,
            scale,
            k_bits,
            v_bits,
            mask_arr,
            output_dtype,
            target,
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            q_heads / kv_heads,
            k_values_per_word,
            v_values_per_word,
            k_packed_dim,
            v_packed_dim,
            mask_heads,
        );
    }

    let kernel = cached_turboquant_sdpa_decode_kernel()?;
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[
            queries, k_packed, k_norms, v_packed, v_norms, k_signs, k_codebook, v_signs,
            v_codebook, &scale_arr, mask_input,
        ])
        .output_shapes(&[output_shape])
        .output_dtypes(&[output_dtype])
        .grid(batch * q_heads * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_heads / kv_heads)
        .template_int("SEQ_LEN", seq_len)
        .template_int("K_BITS", i32::from(k_bits))
        .template_int("V_BITS", i32::from(v_bits))
        .template_int("K_VALUES_PER_WORD", k_values_per_word)
        .template_int("V_VALUES_PER_WORD", v_values_per_word)
        .template_int("K_PACKED_DIM", k_packed_dim)
        .template_int("V_PACKED_DIM", v_packed_dim)
        .template_bool("HAS_MASK", mask_arr.is_some())
        .template_int("MASK_HEADS", mask_heads)
        .dispatch()?;

    outputs.take_at(0)
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_parallel_pre_rotated_on_impl(
    q_rot: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    if !matches!(
        output_dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: output_dtype must be f32, f16, or bf16 (got {output_dtype})"
        )));
    }

    let q_rot_shape = q_rot.shape();
    let q_rot_dims = q_rot_shape.as_slice();
    if q_rot_dims.len() != 3 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: q_rot must be rank-3 [B,Hq,D] (got {q_rot_shape})"
        )));
    }
    let batch = q_rot_dims[0];
    let q_heads = q_rot_dims[1];
    let head_dim = q_rot_dims[2];
    validate_head_dim("turboquant_sdpa_decode_parallel_pre_rotated", head_dim)?;

    let (k_values_per_word, k_levels) = bit_layout(
        "turboquant_sdpa_decode_parallel_pre_rotated",
        "k_bits",
        k_bits,
    )?;
    let (v_values_per_word, v_levels) = bit_layout(
        "turboquant_sdpa_decode_parallel_pre_rotated",
        "v_bits",
        v_bits,
    )?;
    let k_packed_dim = (head_dim + k_values_per_word - 1) / k_values_per_word;
    let v_packed_dim = (head_dim + v_values_per_word - 1) / v_values_per_word;

    let k_packed_shape = k_packed.shape();
    let k_packed_dims = k_packed_shape.as_slice();
    if k_packed_dims.len() != 4 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: k_packed must be rank-4 [B,Hkv,S,packed_D] (got {k_packed_shape})"
        )));
    }
    let kv_heads = k_packed_dims[1];
    let seq_len = k_packed_dims[2];
    if k_packed_dims[0] != batch
        || k_packed_dims[3] != k_packed_dim
        || seq_len <= 0
        || kv_heads <= 0
    {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: k_packed shape {k_packed_shape} is incompatible with q_rot {q_rot_shape}, k_bits={k_bits}"
        )));
    }
    if q_heads % kv_heads != 0 {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: q_heads {q_heads} must be divisible by kv_heads {kv_heads}"
        )));
    }

    let expected_norms = [batch, kv_heads, seq_len];
    let k_norms_shape = k_norms.shape();
    if k_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: k_norms shape must be {:?} (got {k_norms_shape})",
            expected_norms
        )));
    }

    let v_packed_shape = v_packed.shape();
    let v_packed_dims = v_packed_shape.as_slice();
    if v_packed_dims != [batch, kv_heads, seq_len, v_packed_dim] {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: v_packed shape must be [{batch}, {kv_heads}, {seq_len}, {v_packed_dim}] for v_bits={v_bits} (got {v_packed_shape})"
        )));
    }
    let v_norms_shape = v_norms.shape();
    if v_norms_shape.as_slice() != expected_norms {
        return Err(Error::Mlx(format!(
            "turboquant_sdpa_decode_parallel_pre_rotated: v_norms shape must be {:?} (got {v_norms_shape})",
            expected_norms
        )));
    }

    validate_vector_shape(
        "turboquant_sdpa_decode_parallel_pre_rotated",
        "v_signs",
        v_signs,
        head_dim,
    )?;
    validate_codebook_shape(
        "turboquant_sdpa_decode_parallel_pre_rotated",
        "k_codebook",
        k_codebook,
        k_levels,
        k_bits,
    )?;
    validate_codebook_shape(
        "turboquant_sdpa_decode_parallel_pre_rotated",
        "v_codebook",
        v_codebook,
        v_levels,
        v_bits,
    )?;

    let mask_heads = match mask_arr {
        Some(mask) => {
            let mask_shape = mask.shape();
            let mask_dims = mask_shape.as_slice();
            if mask_dims.len() != 4
                || mask_dims[0] != batch
                || mask_dims[2] != 1
                || mask_dims[3] != seq_len
                || !(mask_dims[1] == 1 || mask_dims[1] == q_heads)
            {
                return Err(Error::Mlx(format!(
                    "turboquant_sdpa_decode_parallel_pre_rotated: mask must be [B,1,1,S] or [B,Hq,1,S] for B={batch}, Hq={q_heads}, S={seq_len} (got {mask_shape})"
                )));
            }
            mask_dims[1]
        }
        None => 1,
    };

    let target = target.into();
    let mask_input = mask_arr.unwrap_or(q_rot);
    if std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_some() {
        let v_chunks = (seq_len + PARALLEL_DECODE_V_CHUNK_SIZE - 1) / PARALLEL_DECODE_V_CHUNK_SIZE;
        let profile = TurboquantAttnProfile {
            enabled: true,
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            k_bits,
            v_bits,
            v_chunks,
        };
        return turboquant_sdpa_decode_parallel_dispatch_rotated_profiled(
            q_rot,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            k_codebook,
            v_signs,
            v_codebook,
            scale,
            k_bits,
            v_bits,
            mask_arr,
            mask_input,
            output_dtype,
            target,
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            q_heads / kv_heads,
            k_values_per_word,
            v_values_per_word,
            k_packed_dim,
            v_packed_dim,
            mask_heads,
            profile,
        );
    }

    turboquant_sdpa_decode_parallel_dispatch_rotated(
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        mask_input,
        output_dtype,
        target,
        batch,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
        q_heads / kv_heads,
        k_values_per_word,
        v_values_per_word,
        k_packed_dim,
        v_packed_dim,
        mask_heads,
    )
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_parallel_dispatch(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: StreamOrDevice,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    q_per_kv: i32,
    k_values_per_word: i32,
    v_values_per_word: i32,
    k_packed_dim: i32,
    v_packed_dim: i32,
    mask_heads: i32,
) -> Result<Array> {
    if std::env::var_os("IRONMLX_TURBOQUANT_ATTN_PROFILE").is_some() {
        return turboquant_sdpa_decode_parallel_dispatch_profiled(
            queries,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            k_signs,
            k_codebook,
            v_signs,
            v_codebook,
            scale,
            k_bits,
            v_bits,
            mask_arr,
            output_dtype,
            target,
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            q_per_kv,
            k_values_per_word,
            v_values_per_word,
            k_packed_dim,
            v_packed_dim,
            mask_heads,
        );
    }

    let q_rot_shape = Shape::from(vec![batch, q_heads, head_dim]);
    let mut q_rot_outputs = cached_turboquant_query_rotate_kernel()?
        .dispatch_builder()
        .inputs(&[queries, k_signs])
        .output_shapes(&[q_rot_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(batch * q_heads * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .dispatch()?;
    let q_rot = q_rot_outputs.take_at(0)?;

    let mask_input = mask_arr.unwrap_or(queries);
    turboquant_sdpa_decode_parallel_dispatch_rotated(
        &q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        mask_input,
        output_dtype,
        target,
        batch,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
        q_per_kv,
        k_values_per_word,
        v_values_per_word,
        k_packed_dim,
        v_packed_dim,
        mask_heads,
    )
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_parallel_dispatch_rotated(
    q_rot: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    mask_input: &Array,
    output_dtype: Dtype,
    target: StreamOrDevice,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    q_per_kv: i32,
    k_values_per_word: i32,
    v_values_per_word: i32,
    k_packed_dim: i32,
    v_packed_dim: i32,
    mask_heads: i32,
) -> Result<Array> {
    let scale_arr: Array = (&[scale][..], &[][..]).try_into()?;
    let scores_shape = Shape::from(vec![batch, q_heads, 1, seq_len]);
    let qk_blocks_per_head =
        (seq_len + QK_POSITIONS_PER_SIMDGROUP - 1) / QK_POSITIONS_PER_SIMDGROUP;
    let qk_total_blocks = batch * q_heads * qk_blocks_per_head;
    let qk_block_tiles =
        (qk_total_blocks + QK_SIMDGROUPS_PER_THREADGROUP - 1) / QK_SIMDGROUPS_PER_THREADGROUP;
    let mut score_outputs = cached_turboquant_qk_decode_kernel()?
        .dispatch_builder()
        .inputs(&[q_rot, k_packed, k_norms, k_codebook, &scale_arr, mask_input])
        .output_shapes(&[scores_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(32, qk_block_tiles * QK_SIMDGROUPS_PER_THREADGROUP, 1)
        .threadgroup(32, QK_SIMDGROUPS_PER_THREADGROUP, 1)
        .stream(target)
        .template_int("BATCH", batch)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("SEQ_LEN", seq_len)
        .template_int("QK_SIMDGROUPS", QK_SIMDGROUPS_PER_THREADGROUP)
        .template_int("QK_POSITIONS_PER_SIMDGROUP", QK_POSITIONS_PER_SIMDGROUP)
        .template_int("K_BITS", i32::from(k_bits))
        .template_int("K_VALUES_PER_WORD", k_values_per_word)
        .template_int("K_PACKED_DIM", k_packed_dim)
        .template_bool("HAS_MASK", mask_arr.is_some())
        .template_int("MASK_HEADS", mask_heads)
        .dispatch()?;
    let scores = score_outputs.take_at(0)?;
    let weights = softmax_on(&scores, &[-1_i32][..], false, target)?;

    let output_shape = Shape::from(vec![batch, q_heads, 1, head_dim]);
    if seq_len < PARALLEL_DECODE_V_CHUNK_SIZE {
        let mut output = cached_turboquant_weighted_v_decode_kernel()?
            .dispatch_builder()
            .inputs(&[&weights, v_packed, v_norms, v_signs, v_codebook])
            .output_shapes(&[output_shape])
            .output_dtypes(&[output_dtype])
            .grid(batch * q_heads * head_dim, 1, 1)
            .threadgroup(head_dim, 1, 1)
            .stream(target)
            .template_int("HEAD_DIM", head_dim)
            .template_int("Q_HEADS", q_heads)
            .template_int("KV_HEADS", kv_heads)
            .template_int("Q_PER_KV", q_per_kv)
            .template_int("SEQ_LEN", seq_len)
            .template_int("V_BITS", i32::from(v_bits))
            .template_int("V_VALUES_PER_WORD", v_values_per_word)
            .template_int("V_PACKED_DIM", v_packed_dim)
            .dispatch()?;

        return output.take_at(0);
    }

    let v_chunks = (seq_len + PARALLEL_DECODE_V_CHUNK_SIZE - 1) / PARALLEL_DECODE_V_CHUNK_SIZE;
    let v_dim_groups = (head_dim + V_CHUNK_DIMS_PER_THREADGROUP - 1) / V_CHUNK_DIMS_PER_THREADGROUP;
    let v_q_groups_per_kv = (q_per_kv + V_Q_HEADS_PER_THREADGROUP - 1) / V_Q_HEADS_PER_THREADGROUP;
    let use_branchless_weighted_v = weighted_v_chunk_branchless_shape_supported(q_per_kv, head_dim);
    let weighted_v_chunk_kernel = if use_branchless_weighted_v {
        cached_turboquant_weighted_v_chunk_branchless_kernel()?
    } else {
        cached_turboquant_weighted_v_chunk_kernel()?
    };
    let weighted_v_q_groups_per_kv = if use_branchless_weighted_v {
        1
    } else {
        v_q_groups_per_kv
    };
    let partial_shape = Shape::from(vec![batch, q_heads, v_chunks, head_dim]);
    let mut partial_outputs = weighted_v_chunk_kernel
        .dispatch_builder()
        .inputs(&[&weights, v_packed, v_norms, v_codebook])
        .output_shapes(&[partial_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(
            batch
                * kv_heads
                * weighted_v_q_groups_per_kv
                * v_chunks
                * v_dim_groups
                * PARALLEL_DECODE_V_CHUNK_SIZE,
            1,
            1,
        )
        .threadgroup(PARALLEL_DECODE_V_CHUNK_SIZE, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("SEQ_LEN", seq_len)
        .template_int("V_BITS", i32::from(v_bits))
        .template_int("V_VALUES_PER_WORD", v_values_per_word)
        .template_int("V_PACKED_DIM", v_packed_dim)
        .template_int("V_CHUNKS", v_chunks)
        .template_int("V_CHUNK_SIZE", PARALLEL_DECODE_V_CHUNK_SIZE)
        .template_int("V_CHUNK_SIMDGROUPS", PARALLEL_DECODE_V_CHUNK_SIZE / 32)
        .template_int("V_DIM_GROUPS", v_dim_groups)
        .template_int("V_DIMS_PER_GROUP", V_CHUNK_DIMS_PER_THREADGROUP)
        .template_int("Q_GROUPS_PER_KV", weighted_v_q_groups_per_kv)
        .template_int("V_Q_HEADS_PER_GROUP", V_Q_HEADS_PER_THREADGROUP)
        .dispatch()?;
    let v_partial = partial_outputs.take_at(0)?;

    let mut output = cached_turboquant_weighted_v_reduce_kernel()?
        .dispatch_builder()
        .inputs(&[&v_partial, v_signs])
        .output_shapes(&[output_shape])
        .output_dtypes(&[output_dtype])
        .grid(batch * q_heads * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("V_CHUNKS", v_chunks)
        .dispatch()?;

    output.take_at(0)
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_parallel_dispatch_profiled(
    queries: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_signs: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    output_dtype: Dtype,
    target: StreamOrDevice,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    q_per_kv: i32,
    k_values_per_word: i32,
    v_values_per_word: i32,
    k_packed_dim: i32,
    v_packed_dim: i32,
    mask_heads: i32,
) -> Result<Array> {
    let v_chunks = (seq_len + PARALLEL_DECODE_V_CHUNK_SIZE - 1) / PARALLEL_DECODE_V_CHUNK_SIZE;
    let profile = TurboquantAttnProfile {
        enabled: true,
        batch,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
        k_bits,
        v_bits,
        v_chunks,
    };

    let q_rot_shape = Shape::from(vec![batch, q_heads, head_dim]);
    let q_rotate_start = profile.start();
    let mut q_rot_outputs = cached_turboquant_query_rotate_kernel()?
        .dispatch_builder()
        .inputs(&[queries, k_signs])
        .output_shapes(&[q_rot_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(batch * q_heads * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .dispatch()?;
    let q_rot = q_rot_outputs.take_at(0)?;
    profile.eval_stage("q_rotate", &[&q_rot], q_rotate_start)?;

    let mask_input = mask_arr.unwrap_or(queries);
    turboquant_sdpa_decode_parallel_dispatch_rotated_profiled(
        &q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        k_codebook,
        v_signs,
        v_codebook,
        scale,
        k_bits,
        v_bits,
        mask_arr,
        mask_input,
        output_dtype,
        target,
        batch,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
        q_per_kv,
        k_values_per_word,
        v_values_per_word,
        k_packed_dim,
        v_packed_dim,
        mask_heads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn turboquant_sdpa_decode_parallel_dispatch_rotated_profiled(
    q_rot: &Array,
    k_packed: &Array,
    k_norms: &Array,
    v_packed: &Array,
    v_norms: &Array,
    k_codebook: &Array,
    v_signs: &Array,
    v_codebook: &Array,
    scale: f32,
    k_bits: u8,
    v_bits: u8,
    mask_arr: Option<&Array>,
    mask_input: &Array,
    output_dtype: Dtype,
    target: StreamOrDevice,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    q_per_kv: i32,
    k_values_per_word: i32,
    v_values_per_word: i32,
    k_packed_dim: i32,
    v_packed_dim: i32,
    mask_heads: i32,
    profile: TurboquantAttnProfile,
) -> Result<Array> {
    let v_chunks = (seq_len + PARALLEL_DECODE_V_CHUNK_SIZE - 1) / PARALLEL_DECODE_V_CHUNK_SIZE;
    let scale_arr: Array = (&[scale][..], &[][..]).try_into()?;
    let scores_shape = Shape::from(vec![batch, q_heads, 1, seq_len]);
    let qk_blocks_per_head =
        (seq_len + QK_POSITIONS_PER_SIMDGROUP - 1) / QK_POSITIONS_PER_SIMDGROUP;
    let qk_total_blocks = batch * q_heads * qk_blocks_per_head;
    let qk_block_tiles =
        (qk_total_blocks + QK_SIMDGROUPS_PER_THREADGROUP - 1) / QK_SIMDGROUPS_PER_THREADGROUP;
    let qk_start = profile.start();
    let mut score_outputs = cached_turboquant_qk_decode_kernel()?
        .dispatch_builder()
        .inputs(&[q_rot, k_packed, k_norms, k_codebook, &scale_arr, mask_input])
        .output_shapes(&[scores_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(32, qk_block_tiles * QK_SIMDGROUPS_PER_THREADGROUP, 1)
        .threadgroup(32, QK_SIMDGROUPS_PER_THREADGROUP, 1)
        .stream(target)
        .template_int("BATCH", batch)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("SEQ_LEN", seq_len)
        .template_int("QK_SIMDGROUPS", QK_SIMDGROUPS_PER_THREADGROUP)
        .template_int("QK_POSITIONS_PER_SIMDGROUP", QK_POSITIONS_PER_SIMDGROUP)
        .template_int("K_BITS", i32::from(k_bits))
        .template_int("K_VALUES_PER_WORD", k_values_per_word)
        .template_int("K_PACKED_DIM", k_packed_dim)
        .template_bool("HAS_MASK", mask_arr.is_some())
        .template_int("MASK_HEADS", mask_heads)
        .dispatch()?;
    let scores = score_outputs.take_at(0)?;
    profile.eval_stage("qk", &[&scores], qk_start)?;

    let softmax_start = profile.start();
    let weights = softmax_on(&scores, &[-1_i32][..], false, target)?;
    profile.eval_stage("softmax", &[&weights], softmax_start)?;

    let output_shape = Shape::from(vec![batch, q_heads, 1, head_dim]);
    if seq_len < PARALLEL_DECODE_V_CHUNK_SIZE {
        let weighted_v_start = profile.start();
        let mut output = cached_turboquant_weighted_v_decode_kernel()?
            .dispatch_builder()
            .inputs(&[&weights, v_packed, v_norms, v_signs, v_codebook])
            .output_shapes(&[output_shape])
            .output_dtypes(&[output_dtype])
            .grid(batch * q_heads * head_dim, 1, 1)
            .threadgroup(head_dim, 1, 1)
            .stream(target)
            .template_int("HEAD_DIM", head_dim)
            .template_int("Q_HEADS", q_heads)
            .template_int("KV_HEADS", kv_heads)
            .template_int("Q_PER_KV", q_per_kv)
            .template_int("SEQ_LEN", seq_len)
            .template_int("V_BITS", i32::from(v_bits))
            .template_int("V_VALUES_PER_WORD", v_values_per_word)
            .template_int("V_PACKED_DIM", v_packed_dim)
            .dispatch()?;

        let output = output.take_at(0)?;
        profile.eval_stage("weighted_v_serial", &[&output], weighted_v_start)?;
        return Ok(output);
    }

    let v_dim_groups = (head_dim + V_CHUNK_DIMS_PER_THREADGROUP - 1) / V_CHUNK_DIMS_PER_THREADGROUP;
    let v_q_groups_per_kv = (q_per_kv + V_Q_HEADS_PER_THREADGROUP - 1) / V_Q_HEADS_PER_THREADGROUP;
    let use_branchless_weighted_v = weighted_v_chunk_branchless_shape_supported(q_per_kv, head_dim);
    let weighted_v_chunk_kernel = if use_branchless_weighted_v {
        cached_turboquant_weighted_v_chunk_branchless_kernel()?
    } else {
        cached_turboquant_weighted_v_chunk_kernel()?
    };
    let weighted_v_q_groups_per_kv = if use_branchless_weighted_v {
        1
    } else {
        v_q_groups_per_kv
    };
    let partial_shape = Shape::from(vec![batch, q_heads, v_chunks, head_dim]);
    let weighted_v_chunk_start = profile.start();
    let mut partial_outputs = weighted_v_chunk_kernel
        .dispatch_builder()
        .inputs(&[&weights, v_packed, v_norms, v_codebook])
        .output_shapes(&[partial_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(
            batch
                * kv_heads
                * weighted_v_q_groups_per_kv
                * v_chunks
                * v_dim_groups
                * PARALLEL_DECODE_V_CHUNK_SIZE,
            1,
            1,
        )
        .threadgroup(PARALLEL_DECODE_V_CHUNK_SIZE, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("KV_HEADS", kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("SEQ_LEN", seq_len)
        .template_int("V_BITS", i32::from(v_bits))
        .template_int("V_VALUES_PER_WORD", v_values_per_word)
        .template_int("V_PACKED_DIM", v_packed_dim)
        .template_int("V_CHUNKS", v_chunks)
        .template_int("V_CHUNK_SIZE", PARALLEL_DECODE_V_CHUNK_SIZE)
        .template_int("V_CHUNK_SIMDGROUPS", PARALLEL_DECODE_V_CHUNK_SIZE / 32)
        .template_int("V_DIM_GROUPS", v_dim_groups)
        .template_int("V_DIMS_PER_GROUP", V_CHUNK_DIMS_PER_THREADGROUP)
        .template_int("Q_GROUPS_PER_KV", weighted_v_q_groups_per_kv)
        .template_int("V_Q_HEADS_PER_GROUP", V_Q_HEADS_PER_THREADGROUP)
        .dispatch()?;
    let v_partial = partial_outputs.take_at(0)?;
    profile.eval_stage("weighted_v_chunk", &[&v_partial], weighted_v_chunk_start)?;

    let weighted_v_reduce_start = profile.start();
    let mut output = cached_turboquant_weighted_v_reduce_kernel()?
        .dispatch_builder()
        .inputs(&[&v_partial, v_signs])
        .output_shapes(&[output_shape])
        .output_dtypes(&[output_dtype])
        .grid(batch * q_heads * head_dim, 1, 1)
        .threadgroup(head_dim, 1, 1)
        .stream(target)
        .template_int("HEAD_DIM", head_dim)
        .template_int("Q_HEADS", q_heads)
        .template_int("V_CHUNKS", v_chunks)
        .dispatch()?;

    let output = output.take_at(0)?;
    profile.eval_stage("weighted_v_reduce", &[&output], weighted_v_reduce_start)?;
    Ok(output)
}

fn validate_head_dim(op: &str, head_dim: i32) -> Result<()> {
    if head_dim <= 0 {
        return Err(Error::Mlx(format!(
            "{op}: head_dim must be positive (got {head_dim})"
        )));
    }
    let head_dim_usize = head_dim as usize;
    if !head_dim_usize.is_power_of_two() {
        return Err(Error::Mlx(format!(
            "{op}: head_dim {head_dim} must be a power of two"
        )));
    }
    Ok(())
}

fn bit_layout(op: &str, name: &str, bits: u8) -> Result<(i32, i32)> {
    match bits {
        3 => Ok((10, 8)),
        4 => Ok((8, 16)),
        _ => Err(Error::Mlx(format!(
            "{op}: unsupported {name} {bits} (expected 3 or 4)"
        ))),
    }
}

fn validate_vector_shape(op: &str, name: &str, array: &Array, head_dim: i32) -> Result<()> {
    let shape = array.shape();
    if shape.as_slice() != [head_dim] {
        return Err(Error::Mlx(format!(
            "{op}: {name} shape must be [{head_dim}] (got {shape})"
        )));
    }
    Ok(())
}

fn validate_codebook_shape(
    op: &str,
    name: &str,
    array: &Array,
    levels: i32,
    bits: u8,
) -> Result<()> {
    let shape = array.shape();
    if shape.as_slice() != [levels] {
        return Err(Error::Mlx(format!(
            "{op}: {name} shape must be [{levels}] for {bits}-bit (got {shape})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turboquant_attn_profile_line_is_stable_jsonl() {
        let event = TurboquantAttnProfileEvent {
            stage: "qk",
            elapsed_us: 1234,
            batch: 1,
            q_heads: 28,
            kv_heads: 4,
            seq_len: 32_768,
            head_dim: 128,
            k_bits: 3,
            v_bits: 4,
            v_chunks: 128,
        };

        assert_eq!(
            format_turboquant_attn_profile_line(event),
            "{\"event\":\"turboquant_attn_stage\",\"stage\":\"qk\",\"elapsed_us\":1234,\"batch\":1,\"q_heads\":28,\"kv_heads\":4,\"seq_len\":32768,\"head_dim\":128,\"k_bits\":3,\"v_bits\":4,\"v_chunks\":128}"
        );
    }

    #[test]
    fn weighted_v_dim_group_tuning_constant_matches_retained_variant() {
        assert_eq!(V_CHUNK_DIMS_PER_THREADGROUP, 16);
    }

    #[test]
    fn qk_positions_per_simdgroup_tuning_constant_matches_candidate() {
        assert_eq!(QK_POSITIONS_PER_SIMDGROUP, 4);
    }

    #[test]
    fn weighted_v_q_head_group_tuning_constant_matches_candidate() {
        assert_eq!(V_Q_HEADS_PER_THREADGROUP, 4);
    }

    #[test]
    fn weighted_v_chunk_branchless_shape_requires_full_q_and_dim_groups() {
        assert!(weighted_v_chunk_branchless_shape_supported(4, 256));
        assert!(!weighted_v_chunk_branchless_shape_supported(3, 256));
        assert!(!weighted_v_chunk_branchless_shape_supported(4, 260));
    }
}
