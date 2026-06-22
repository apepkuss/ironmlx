use std::{sync::OnceLock, time::Instant};

use crate::ops::unary::softmax_on;
use crate::{Array, Dtype, Error, Result, Shape, StreamOrDevice};

use super::MetalKernel;

const PAGED_SDPA_DECODE_SIMD_WIDTH: i32 = 32;
const PAGED_SDPA_DECODE_MAX_VALUES_PER_THREAD: i32 = 32;
const PAGED_SDPA_PARALLEL_DECODE_SEQ_THRESHOLD: i32 = 128;
const PAGED_QK_SIMDGROUPS_PER_THREADGROUP: i32 = 4;
const PAGED_QK_POSITIONS_PER_SIMDGROUP: i32 = 8;
const PAGED_V_CHUNK_SIZE: i32 = 256;
const PAGED_V_DIMS_PER_THREADGROUP: i32 = 16;
const PAGED_V_Q_HEADS_PER_GROUP: i32 = 4;

#[derive(Clone, Copy)]
struct PagedAttnProfileEvent<'a> {
    stage: &'a str,
    elapsed_us: u128,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks: i32,
    v_chunks: i32,
}

fn format_paged_attn_profile_line(event: PagedAttnProfileEvent<'_>) -> String {
    format!(
        "{{\"event\":\"paged_attn_stage\",\"stage\":\"{}\",\"elapsed_us\":{},\"batch\":{},\"q_heads\":{},\"kv_heads\":{},\"seq_len\":{},\"head_dim\":{},\"block_size\":{},\"max_blocks\":{},\"v_chunks\":{}}}",
        event.stage,
        event.elapsed_us,
        event.batch,
        event.q_heads,
        event.kv_heads,
        event.seq_len,
        event.head_dim,
        event.block_size,
        event.max_blocks,
        event.v_chunks,
    )
}

#[derive(Clone, Copy)]
struct PagedAttnProfile {
    enabled: bool,
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    seq_len: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks: i32,
    v_chunks: i32,
}

impl PagedAttnProfile {
    fn for_shape(shape: DecodeShape) -> Self {
        let seq_len = shape.max_seq_len();
        let v_chunks = (seq_len + PAGED_V_CHUNK_SIZE - 1) / PAGED_V_CHUNK_SIZE;
        Self {
            enabled: std::env::var_os("IRONMLX_PAGED_ATTN_PROFILE").is_some(),
            batch: shape.batch,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            seq_len,
            head_dim: shape.head_dim,
            block_size: shape.block_size,
            max_blocks: shape.max_blocks,
            v_chunks,
        }
    }

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
            format_paged_attn_profile_line(PagedAttnProfileEvent {
                stage,
                elapsed_us: start.elapsed().as_micros(),
                batch: self.batch,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                seq_len: self.seq_len,
                head_dim: self.head_dim,
                block_size: self.block_size,
                max_blocks: self.max_blocks,
                v_chunks: self.v_chunks,
            })
        );
        Ok(())
    }
}

const PAGED_SDPA_DECODE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;

uint q_head = group_idx % uint(H_Q);
uint batch = group_idx / uint(H_Q);
uint q_group = uint(H_Q / H_KV);
uint kv_head = q_head / q_group;
uint q_base = ((batch * uint(H_Q) + q_head) * uint(HEAD_DIM));

int len_i = lengths[batch];
thread float acc[VALUES_PER_THREAD];
#pragma clang loop unroll(full)
for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
    acc[i] = 0.0f;
}

if (len_i <= 0) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
        uint d = lane + i * uint(SIMD_WIDTH);
        if (d < uint(HEAD_DIM)) {
            out[q_base + d] = static_cast<__typeof__(*out)>(0.0f);
        }
    }
    return;
}

uint len = uint(len_i);
float scale = float(scale_arr[0]);
float max_value = -3.4028234663852886e38f;
float denom_value = 0.0f;

for (uint t = 0; t < len; ++t) {
    uint block_col = t / uint(BLOCK_SIZE);
    uint in_block = t - block_col * uint(BLOCK_SIZE);
    int page_i = block_table[batch * uint(MAX_BLOCKS) + block_col];
    uint page = uint(page_i);
    uint k_base = (((page * uint(H_KV) + kv_head) * uint(BLOCK_SIZE) + in_block) * uint(HEAD_DIM));

    float dot = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
        uint d = lane + i * uint(SIMD_WIDTH);
        if (d < uint(HEAD_DIM)) {
            dot += float(queries[q_base + d]) * float(k_pages[k_base + d]);
        }
    }

    float score = simd_sum(dot) * scale;
    float old_max = max_value;
    float new_max = max(old_max, score);
    float rescale_value = exp(old_max - new_max);
    float weight_value = exp(score - new_max);
    denom_value = denom_value * rescale_value + weight_value;
    max_value = new_max;

    uint v_base = (((page * uint(H_KV) + kv_head) * uint(BLOCK_SIZE) + in_block) * uint(HEAD_DIM));
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
        uint d = lane + i * uint(SIMD_WIDTH);
        if (d < uint(HEAD_DIM)) {
            acc[i] = acc[i] * rescale_value + weight_value * float(v_pages[v_base + d]);
        }
    }
}

float safe_denom = denom_value > 1.0e-20f ? denom_value : 1.0f;
#pragma clang loop unroll(full)
for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
    uint d = lane + i * uint(SIMD_WIDTH);
    if (d < uint(HEAD_DIM)) {
        out[q_base + d] = static_cast<__typeof__(*out)>(acc[i] / safe_denom);
    }
}
"#;

const PAGED_SDPA_QK_DECODE_SOURCE: &str = r#"
uint tile_idx = threadgroup_position_in_grid.y;
uint sgid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint block_idx = tile_idx * uint(QK_SIMDGROUPS) + sgid;
uint blocks_per_head = (uint(MAX_SEQ_LEN) + uint(QK_POSITIONS_PER_SIMDGROUP) - 1) / uint(QK_POSITIONS_PER_SIMDGROUP);
uint total_blocks = uint(BATCH) * uint(H_Q) * blocks_per_head;
if (block_idx >= total_blocks) {
    return;
}

uint pos_block = block_idx % blocks_per_head;
uint q_head = (block_idx / blocks_per_head) % uint(H_Q);
uint batch = block_idx / (blocks_per_head * uint(H_Q));
uint q_group = uint(H_Q / H_KV);
uint kv_head = q_head / q_group;
uint pos_base = pos_block * uint(QK_POSITIONS_PER_SIMDGROUP);
uint q_base = (batch * uint(H_Q) + q_head) * uint(HEAD_DIM);
int len_i = lengths[batch];
uint len = len_i > 0 ? uint(len_i) : 0;

thread float acc[QK_POSITIONS_PER_SIMDGROUP];
#pragma clang loop unroll(full)
for (uint i = 0; i < uint(QK_POSITIONS_PER_SIMDGROUP); ++i) {
    acc[i] = 0.0f;
}

for (uint dim = lane; dim < uint(HEAD_DIM); dim += 32) {
    float q_value = float(queries[q_base + dim]);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(QK_POSITIONS_PER_SIMDGROUP); ++i) {
        uint pos = pos_base + i;
        if (pos < len) {
            uint block_col = pos / uint(BLOCK_SIZE);
            uint in_block = pos - block_col * uint(BLOCK_SIZE);
            int page_i = block_table[batch * uint(MAX_BLOCKS) + block_col];
            if (page_i >= 0) {
                uint page = uint(page_i);
                uint k_base = (((page * uint(H_KV) + kv_head) * uint(BLOCK_SIZE) + in_block) * uint(HEAD_DIM));
                acc[i] += q_value * float(k_pages[k_base + dim]);
            }
        }
    }
}

thread float score_acc[QK_POSITIONS_PER_SIMDGROUP];
#pragma clang loop unroll(full)
for (uint i = 0; i < uint(QK_POSITIONS_PER_SIMDGROUP); ++i) {
    score_acc[i] = simd_sum(acc[i]);
}

if (lane == 0) {
    float scale = float(scale_arr[0]);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(QK_POSITIONS_PER_SIMDGROUP); ++i) {
        uint pos = pos_base + i;
        if (pos < uint(MAX_SEQ_LEN)) {
            float score = -3.4028234663852886e38f;
            if (pos < len) {
                score = score_acc[i] * scale;
            } else if (len_i <= 0 && pos == 0) {
                score = 0.0f;
            }
            uint score_idx = ((batch * uint(H_Q) + q_head) * uint(MAX_SEQ_LEN) + pos);
            scores[score_idx] = score;
        }
    }
}
"#;

const PAGED_SDPA_WEIGHTED_V_CHUNK_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;
uint sgid = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

uint dim_group = group_idx % uint(V_DIM_GROUPS);
uint chunk = (group_idx / uint(V_DIM_GROUPS)) % uint(V_CHUNKS);
uint q_group = (group_idx / (uint(V_DIM_GROUPS) * uint(V_CHUNKS))) % uint(Q_GROUPS_PER_KV);
uint kv_head = (group_idx / (uint(V_DIM_GROUPS) * uint(V_CHUNKS) * uint(Q_GROUPS_PER_KV))) % uint(H_KV);
uint batch = group_idx / (uint(V_DIM_GROUPS) * uint(V_CHUNKS) * uint(Q_GROUPS_PER_KV) * uint(H_KV));
uint pos = chunk * uint(V_CHUNK_SIZE) + lid;
uint dim_base = dim_group * uint(V_DIMS_PER_GROUP);
uint q_base_offset = q_group * uint(V_Q_HEADS_PER_GROUP);

threadgroup float scratch[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP * V_CHUNK_SIMDGROUPS];
thread float acc[V_Q_HEADS_PER_GROUP * V_DIMS_PER_GROUP];
#pragma clang loop unroll(full)
for (uint q = 0; q < uint(V_Q_HEADS_PER_GROUP); ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(V_DIMS_PER_GROUP); ++i) {
        acc[q * uint(V_DIMS_PER_GROUP) + i] = 0.0f;
    }
}

int len_i = lengths[batch];
uint len = len_i > 0 ? uint(len_i) : 0;
if (pos < len) {
    uint block_col = pos / uint(BLOCK_SIZE);
    uint in_block = pos - block_col * uint(BLOCK_SIZE);
    int page_i = block_table[batch * uint(MAX_BLOCKS) + block_col];
    if (page_i >= 0) {
        uint page = uint(page_i);
        uint v_base = (((page * uint(H_KV) + kv_head) * uint(BLOCK_SIZE) + in_block) * uint(HEAD_DIM));
        #pragma clang loop unroll(full)
        for (uint i = 0; i < uint(V_DIMS_PER_GROUP); ++i) {
            uint dim = dim_base + i;
            if (dim < uint(HEAD_DIM)) {
                float v_value = float(v_pages[v_base + dim]);
                #pragma clang loop unroll(full)
                for (uint q = 0; q < uint(V_Q_HEADS_PER_GROUP); ++q) {
                    uint q_offset = q_base_offset + q;
                    if (q_offset < uint(Q_PER_KV)) {
                        uint q_head = kv_head * uint(Q_PER_KV) + q_offset;
                        uint weight_idx = ((batch * uint(H_Q) + q_head) * uint(MAX_SEQ_LEN) + pos);
                        float weight = float(weights[weight_idx]);
                        acc[q * uint(V_DIMS_PER_GROUP) + i] = weight * v_value;
                    }
                }
            }
        }
    }
}

#pragma clang loop unroll(full)
for (uint q = 0; q < uint(V_Q_HEADS_PER_GROUP); ++q) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(V_DIMS_PER_GROUP); ++i) {
        float simd_acc = simd_sum(acc[q * uint(V_DIMS_PER_GROUP) + i]);
        if (lane == 0) {
            uint scratch_idx = ((q * uint(V_DIMS_PER_GROUP) + i) * uint(V_CHUNK_SIMDGROUPS) + sgid);
            scratch[scratch_idx] = simd_acc;
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sgid == 0) {
    #pragma clang loop unroll(full)
    for (uint q = 0; q < uint(V_Q_HEADS_PER_GROUP); ++q) {
        uint q_offset = q_base_offset + q;
        if (q_offset < uint(Q_PER_KV)) {
            uint q_head = kv_head * uint(Q_PER_KV) + q_offset;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < uint(V_DIMS_PER_GROUP); ++i) {
                uint dim = dim_base + i;
                if (dim < uint(HEAD_DIM)) {
                    uint scratch_idx = ((q * uint(V_DIMS_PER_GROUP) + i) * uint(V_CHUNK_SIMDGROUPS) + lane);
                    float chunk_acc = lane < uint(V_CHUNK_SIMDGROUPS) ? scratch[scratch_idx] : 0.0f;
                    chunk_acc = simd_sum(chunk_acc);
                    if (lane == 0) {
                        uint partial_idx = (((batch * uint(H_Q) + q_head) * uint(V_CHUNKS) + chunk) * uint(HEAD_DIM) + dim);
                        v_partial[partial_idx] = chunk_acc;
                    }
                }
            }
        }
    }
}
"#;

const PAGED_SDPA_WEIGHTED_V_REDUCE_SOURCE: &str = r#"
uint group_idx = threadgroup_position_in_grid.x;
uint lid = thread_index_in_threadgroup;

uint q_head = group_idx % uint(H_Q);
uint batch = group_idx / uint(H_Q);

float acc = 0.0f;
for (uint chunk = 0; chunk < uint(V_CHUNKS); ++chunk) {
    uint partial_idx = (((batch * uint(H_Q) + q_head) * uint(V_CHUNKS) + chunk) * uint(HEAD_DIM) + lid);
    acc += float(v_partial[partial_idx]);
}

uint out_base = ((batch * uint(H_Q) + q_head) * uint(HEAD_DIM));
out[out_base + lid] = static_cast<__typeof__(*out)>(acc);
"#;

/// Decode-only paged scaled dot-product attention.
///
/// `queries` shape: `[B, Hq, 1, D]`; page tensors shape:
/// `[page_count, Hkv, block_size, D]`; `block_table` shape:
/// `[B, max_blocks]`; `lengths` shape: `[B]`.
#[allow(clippy::too_many_arguments)]
pub fn paged_scaled_dot_product_attention_decode(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    scale: f32,
    block_size: i32,
) -> Result<Array> {
    paged_scaled_dot_product_attention_decode_on(
        queries,
        k_pages,
        v_pages,
        block_table,
        lengths,
        scale,
        block_size,
        (),
    )
}

/// Stream-targeted variant of [`paged_scaled_dot_product_attention_decode`].
#[allow(clippy::too_many_arguments)]
pub fn paged_scaled_dot_product_attention_decode_on(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    scale: f32,
    block_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let shape =
        validate_decode_inputs(queries, k_pages, v_pages, block_table, lengths, block_size)?;
    let target = target.into();
    if shape.max_seq_len() >= PAGED_SDPA_PARALLEL_DECODE_SEQ_THRESHOLD {
        return paged_scaled_dot_product_attention_decode_parallel_on(
            queries,
            k_pages,
            v_pages,
            block_table,
            lengths,
            scale,
            shape,
            target,
        );
    }

    paged_scaled_dot_product_attention_decode_serial_on(
        queries,
        k_pages,
        v_pages,
        block_table,
        lengths,
        scale,
        shape,
        target,
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_scaled_dot_product_attention_decode_serial_on(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    scale: f32,
    shape: DecodeShape,
    target: StreamOrDevice,
) -> Result<Array> {
    let profile = PagedAttnProfile::for_shape(shape);
    let scale_arr: Array = (&[scale][..], (1_i32,)).try_into()?;
    let kernel = cached_paged_sdpa_decode_kernel()?;
    let output_shape = Shape::from((shape.batch, shape.q_heads, 1_i32, shape.head_dim));
    let serial_start = profile.start();
    let mut outputs = kernel
        .dispatch_builder()
        .inputs(&[queries, k_pages, v_pages, block_table, lengths, &scale_arr])
        .output_shapes(&[output_shape])
        .output_dtypes(&[queries.dtype()])
        .grid(
            shape.batch * shape.q_heads * PAGED_SDPA_DECODE_SIMD_WIDTH,
            1,
            1,
        )
        .threadgroup(PAGED_SDPA_DECODE_SIMD_WIDTH, 1, 1)
        .template_int("BATCH", shape.batch)
        .template_int("H_Q", shape.q_heads)
        .template_int("H_KV", shape.kv_heads)
        .template_int("HEAD_DIM", shape.head_dim)
        .template_int("SIMD_WIDTH", PAGED_SDPA_DECODE_SIMD_WIDTH)
        .template_int("VALUES_PER_THREAD", shape.values_per_thread)
        .template_int("BLOCK_SIZE", shape.block_size)
        .template_int("MAX_BLOCKS", shape.max_blocks)
        .stream(target)
        .dispatch()?;

    let output = outputs.take_at(0)?;
    profile.eval_stage("serial", &[&output], serial_start)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn paged_scaled_dot_product_attention_decode_parallel_on(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    scale: f32,
    shape: DecodeShape,
    target: StreamOrDevice,
) -> Result<Array> {
    let max_seq_len = shape.max_seq_len();
    let profile = PagedAttnProfile::for_shape(shape);
    let scale_arr: Array = (&[scale][..], (1_i32,)).try_into()?;
    let scores_shape = Shape::from((shape.batch, shape.q_heads, 1_i32, max_seq_len));
    let qk_blocks_per_head =
        (max_seq_len + PAGED_QK_POSITIONS_PER_SIMDGROUP - 1) / PAGED_QK_POSITIONS_PER_SIMDGROUP;
    let qk_total_blocks = shape.batch * shape.q_heads * qk_blocks_per_head;
    let qk_block_tiles = (qk_total_blocks + PAGED_QK_SIMDGROUPS_PER_THREADGROUP - 1)
        / PAGED_QK_SIMDGROUPS_PER_THREADGROUP;
    let qk_start = profile.start();
    let mut score_outputs = cached_paged_sdpa_qk_decode_kernel()?
        .dispatch_builder()
        .inputs(&[queries, k_pages, block_table, lengths, &scale_arr])
        .output_shapes(&[scores_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(
            PAGED_SDPA_DECODE_SIMD_WIDTH,
            qk_block_tiles * PAGED_QK_SIMDGROUPS_PER_THREADGROUP,
            1,
        )
        .threadgroup(
            PAGED_SDPA_DECODE_SIMD_WIDTH,
            PAGED_QK_SIMDGROUPS_PER_THREADGROUP,
            1,
        )
        .template_int("BATCH", shape.batch)
        .template_int("H_Q", shape.q_heads)
        .template_int("H_KV", shape.kv_heads)
        .template_int("HEAD_DIM", shape.head_dim)
        .template_int("BLOCK_SIZE", shape.block_size)
        .template_int("MAX_BLOCKS", shape.max_blocks)
        .template_int("MAX_SEQ_LEN", max_seq_len)
        .template_int("QK_SIMDGROUPS", PAGED_QK_SIMDGROUPS_PER_THREADGROUP)
        .template_int(
            "QK_POSITIONS_PER_SIMDGROUP",
            PAGED_QK_POSITIONS_PER_SIMDGROUP,
        )
        .stream(target)
        .dispatch()?;
    let scores = score_outputs.take_at(0)?;
    profile.eval_stage("qk", &[&scores], qk_start)?;

    let softmax_start = profile.start();
    let weights = softmax_on(&scores, &[-1_i32][..], false, target)?;
    profile.eval_stage("softmax", &[&weights], softmax_start)?;

    let v_chunks = (max_seq_len + PAGED_V_CHUNK_SIZE - 1) / PAGED_V_CHUNK_SIZE;
    let v_dim_groups =
        (shape.head_dim + PAGED_V_DIMS_PER_THREADGROUP - 1) / PAGED_V_DIMS_PER_THREADGROUP;
    let q_per_kv = shape.q_heads / shape.kv_heads;
    let v_q_groups_per_kv = (q_per_kv + PAGED_V_Q_HEADS_PER_GROUP - 1) / PAGED_V_Q_HEADS_PER_GROUP;
    let partial_shape = Shape::from((shape.batch, shape.q_heads, v_chunks, shape.head_dim));
    let weighted_v_chunk_start = profile.start();
    let mut partial_outputs = cached_paged_sdpa_weighted_v_chunk_kernel()?
        .dispatch_builder()
        .inputs(&[&weights, v_pages, block_table, lengths])
        .output_shapes(&[partial_shape])
        .output_dtypes(&[Dtype::Float32])
        .grid(
            shape.batch
                * shape.kv_heads
                * v_q_groups_per_kv
                * v_chunks
                * v_dim_groups
                * PAGED_V_CHUNK_SIZE,
            1,
            1,
        )
        .threadgroup(PAGED_V_CHUNK_SIZE, 1, 1)
        .template_int("H_Q", shape.q_heads)
        .template_int("H_KV", shape.kv_heads)
        .template_int("Q_PER_KV", q_per_kv)
        .template_int("HEAD_DIM", shape.head_dim)
        .template_int("BLOCK_SIZE", shape.block_size)
        .template_int("MAX_BLOCKS", shape.max_blocks)
        .template_int("MAX_SEQ_LEN", max_seq_len)
        .template_int("V_CHUNKS", v_chunks)
        .template_int("V_CHUNK_SIZE", PAGED_V_CHUNK_SIZE)
        .template_int(
            "V_CHUNK_SIMDGROUPS",
            PAGED_V_CHUNK_SIZE / PAGED_SDPA_DECODE_SIMD_WIDTH,
        )
        .template_int("V_DIM_GROUPS", v_dim_groups)
        .template_int("V_DIMS_PER_GROUP", PAGED_V_DIMS_PER_THREADGROUP)
        .template_int("Q_GROUPS_PER_KV", v_q_groups_per_kv)
        .template_int("V_Q_HEADS_PER_GROUP", PAGED_V_Q_HEADS_PER_GROUP)
        .stream(target)
        .dispatch()?;
    let v_partial = partial_outputs.take_at(0)?;
    profile.eval_stage("weighted_v_chunk", &[&v_partial], weighted_v_chunk_start)?;

    let output_shape = Shape::from((shape.batch, shape.q_heads, 1_i32, shape.head_dim));
    let weighted_v_reduce_start = profile.start();
    let mut outputs = cached_paged_sdpa_weighted_v_reduce_kernel()?
        .dispatch_builder()
        .inputs(&[&v_partial])
        .output_shapes(&[output_shape])
        .output_dtypes(&[queries.dtype()])
        .grid(shape.batch * shape.q_heads * shape.head_dim, 1, 1)
        .threadgroup(shape.head_dim, 1, 1)
        .template_int("H_Q", shape.q_heads)
        .template_int("HEAD_DIM", shape.head_dim)
        .template_int("V_CHUNKS", v_chunks)
        .stream(target)
        .dispatch()?;

    let output = outputs.take_at(0)?;
    profile.eval_stage("weighted_v_reduce", &[&output], weighted_v_reduce_start)?;
    Ok(output)
}

#[derive(Clone, Copy)]
struct DecodeShape {
    batch: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    values_per_thread: i32,
    block_size: i32,
    max_blocks: i32,
}

impl DecodeShape {
    fn max_seq_len(self) -> i32 {
        self.max_blocks * self.block_size
    }
}

fn validate_decode_inputs(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    block_size: i32,
) -> Result<DecodeShape> {
    if block_size <= 0 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode requires positive block_size, got {block_size}"
        )));
    }
    if queries.ndim() != 4 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode queries rank must be 4, got {}",
            queries.ndim()
        )));
    }
    if k_pages.ndim() != 4 || v_pages.ndim() != 4 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode page ranks must be 4, got k={} v={}",
            k_pages.ndim(),
            v_pages.ndim()
        )));
    }
    if block_table.ndim() != 2 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode block_table rank must be 2, got {}",
            block_table.ndim()
        )));
    }
    if lengths.ndim() != 1 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode lengths rank must be 1, got {}",
            lengths.ndim()
        )));
    }
    if block_table.dtype() != Dtype::Int32 {
        return Err(Error::DtypeMismatch {
            expected: Dtype::Int32,
            actual: block_table.dtype(),
        });
    }
    if lengths.dtype() != Dtype::Int32 {
        return Err(Error::DtypeMismatch {
            expected: Dtype::Int32,
            actual: lengths.dtype(),
        });
    }
    if k_pages.dtype() != queries.dtype() {
        return Err(Error::DtypeMismatch {
            expected: queries.dtype(),
            actual: k_pages.dtype(),
        });
    }
    if v_pages.dtype() != queries.dtype() {
        return Err(Error::DtypeMismatch {
            expected: queries.dtype(),
            actual: v_pages.dtype(),
        });
    }

    let q_shape = queries.shape();
    let k_shape = k_pages.shape();
    let v_shape = v_pages.shape();
    let table_shape = block_table.shape();
    let len_shape = lengths.shape();

    if k_shape != v_shape {
        return Err(Error::ShapeMismatch {
            expected: k_shape,
            actual: v_shape,
        });
    }

    let batch = q_shape[0];
    let q_heads = q_shape[1];
    let q_len = q_shape[2];
    let head_dim = q_shape[3];
    let kv_heads = k_shape[1];
    let page_block_size = k_shape[2];
    let page_head_dim = k_shape[3];

    if head_dim <= 0 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode requires positive head_dim, got {head_dim}"
        )));
    }
    if q_len != 1 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode requires query sequence length 1, got {q_len}"
        )));
    }
    if page_block_size != block_size {
        return Err(Error::Mlx(format!(
            "paged SDPA decode block_size argument {block_size} does not match page dim {page_block_size}"
        )));
    }
    if page_head_dim != head_dim {
        return Err(Error::Mlx(format!(
            "paged SDPA decode head_dim mismatch: q={head_dim} pages={page_head_dim}"
        )));
    }
    if table_shape[0] != batch || len_shape[0] != batch {
        return Err(Error::Mlx(format!(
            "paged SDPA decode batch mismatch: q={batch} table={} lengths={}",
            table_shape[0], len_shape[0]
        )));
    }
    if kv_heads <= 0 || q_heads % kv_heads != 0 {
        return Err(Error::Mlx(format!(
            "paged SDPA decode requires Hq divisible by Hkv, got Hq={q_heads} Hkv={kv_heads}"
        )));
    }
    let values_per_thread = values_per_thread_i32(head_dim)?;

    Ok(DecodeShape {
        batch,
        q_heads,
        kv_heads,
        head_dim,
        values_per_thread,
        block_size,
        max_blocks: table_shape[1],
    })
}

fn values_per_thread_i32(head_dim: i32) -> Result<i32> {
    let value = (head_dim + PAGED_SDPA_DECODE_SIMD_WIDTH - 1) / PAGED_SDPA_DECODE_SIMD_WIDTH;
    if value > PAGED_SDPA_DECODE_MAX_VALUES_PER_THREAD {
        return Err(Error::Mlx(format!(
            "paged SDPA decode head_dim {head_dim} requires {value} values per SIMD lane, exceeding {}",
            PAGED_SDPA_DECODE_MAX_VALUES_PER_THREAD
        )));
    }
    Ok(value)
}

fn cached_paged_sdpa_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_paged_sdpa_decode")
        .inputs(&[
            "queries",
            "k_pages",
            "v_pages",
            "block_table",
            "lengths",
            "scale_arr",
        ])
        .outputs(&["out"])
        .source(PAGED_SDPA_DECODE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_paged_sdpa_qk_decode_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_paged_sdpa_qk_decode")
        .inputs(&["queries", "k_pages", "block_table", "lengths", "scale_arr"])
        .outputs(&["scores"])
        .source(PAGED_SDPA_QK_DECODE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_paged_sdpa_weighted_v_chunk_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_paged_sdpa_weighted_v_chunk")
        .inputs(&["weights", "v_pages", "block_table", "lengths"])
        .outputs(&["v_partial"])
        .source(PAGED_SDPA_WEIGHTED_V_CHUNK_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}

fn cached_paged_sdpa_weighted_v_reduce_kernel() -> Result<&'static MetalKernel> {
    static CELL: OnceLock<MetalKernel> = OnceLock::new();
    if let Some(kernel) = CELL.get() {
        return Ok(kernel);
    }

    let kernel = MetalKernel::builder("mlx_fast_paged_sdpa_weighted_v_reduce")
        .inputs(&["v_partial"])
        .outputs(&["out"])
        .source(PAGED_SDPA_WEIGHTED_V_REDUCE_SOURCE)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(CELL.get_or_init(|| kernel))
}
