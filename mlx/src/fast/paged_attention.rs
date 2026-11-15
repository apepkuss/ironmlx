use std::sync::OnceLock;

use crate::{Array, Dtype, Error, Result, Shape, StreamOrDevice};

use super::MetalKernel;

const PAGED_SDPA_DECODE_SIMD_WIDTH: i32 = 32;
const PAGED_SDPA_DECODE_MAX_VALUES_PER_THREAD: i32 = 32;
/// Query heads one threadgroup serves; even one head per simdgroup stays within 1024 threads.
const PAGED_SDPA_MAX_HEADS_PER_THREADGROUP: i32 = 32;
/// Query heads one simdgroup keeps in registers and serves from one load.
const PAGED_SDPA_MAX_HEADS_PER_SIMDGROUP: i32 = 2;
/// Fewest positions a split covers.
const PAGED_SDPA_MIN_SPLIT_LEN: i32 = 32;
/// Split the sequence until the first pass has about this many threadgroups.
const PAGED_SDPA_TARGET_THREADGROUPS: i32 = 256;
/// Simdgroups that share one query head's splits in the second pass.
const PAGED_SDPA_REDUCE_SIMDGROUPS: i32 = 8;

/// First pass: each threadgroup reads one KV head's keys and values over one
/// split of the positions, and each simdgroup keeps `HEADS_PER_SG` query
/// heads of that KV head in registers, so a key/value row it loads serves all
/// of them. Every head runs an online softmax; the pass writes the
/// unnormalized weighted sum with the split's max score and exp-sum.
const PAGED_SDPA_SPLIT_SOURCE: &str = r#"
#define DIM_OK(d) (CONTIGUOUS || (d) < uint(HEAD_DIM))
#define LANE_DIM(lane, i) (CONTIGUOUS ? (lane) * uint(VALUES_PER_THREAD) + (i) : (lane) + (i) * 32)
uint tg = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint max_blocks = uint(params[0]);
uint splits = uint(params[1]);
uint split_len = uint(params[2]);

uint split = tg % splits;
uint rest = tg / splits;
uint head_group = rest % uint(HEAD_GROUPS);
rest = rest / uint(HEAD_GROUPS);
uint kv_head = rest % uint(H_KV);
uint batch = rest / uint(H_KV);
uint first_row = batch * uint(H_Q) + kv_head * uint(Q_PER_KV)
    + head_group * uint(HEADS_PER_TG) + sg * uint(HEADS_PER_SG);

int len_i = lengths[batch];
uint len = len_i > 0 ? uint(len_i) : 0;
int start_i = starts[batch];
uint start = (start_i > 0 && len > 0) ? min(uint(start_i), len - 1) : 0;
uint s0 = max(start, split * split_len);
uint s1 = min(len, split * split_len + split_len);

float scale = float(scale_arr[0]);
thread float q[HEADS_PER_SG][VALUES_PER_THREAD];
thread float o[HEADS_PER_SG][VALUES_PER_THREAD];
thread float max_score[HEADS_PER_SG];
thread float sum_exp[HEADS_PER_SG];
#pragma clang loop unroll(full)
for (uint h = 0; h < uint(HEADS_PER_SG); ++h) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
        uint d = LANE_DIM(lane, i);
        q[h][i] = DIM_OK(d)
            ? scale * float(queries[(first_row + h) * uint(HEAD_DIM) + d])
            : 0.0f;
        o[h][i] = 0.0f;
    }
    max_score[h] = -3.4028234663852886e38f;
    sum_exp[h] = 0.0f;
}

uint t = s0;
while (t < s1) {
    uint block_col = t / uint(BLOCK_SIZE);
    uint block_end = min(s1, (block_col + 1) * uint(BLOCK_SIZE));
    int page_i = block_table[batch * max_blocks + block_col];
    if (page_i >= 0) {
        uint base = ((uint(page_i) * uint(H_KV) + kv_head) * uint(BLOCK_SIZE)
            + (t - block_col * uint(BLOCK_SIZE))) * uint(HEAD_DIM);
        for (uint p = t; p < block_end; ++p, base += uint(HEAD_DIM)) {
            float k_reg[VALUES_PER_THREAD];
            float v_reg[VALUES_PER_THREAD];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
                uint d = LANE_DIM(lane, i);
                k_reg[i] = DIM_OK(d) ? float(k_pages[base + d]) : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
                uint d = LANE_DIM(lane, i);
                v_reg[i] = DIM_OK(d) ? float(v_pages[base + d]) : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint h = 0; h < uint(HEADS_PER_SG); ++h) {
                float dot = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
                    dot += q[h][i] * k_reg[i];
                }
                float score = simd_sum(dot);
                float new_max = max(max_score[h], score);
                float factor = exp(max_score[h] - new_max);
                float weight = exp(score - new_max);
                max_score[h] = new_max;
                sum_exp[h] = sum_exp[h] * factor + weight;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
                    o[h][i] = o[h][i] * factor + weight * v_reg[i];
                }
            }
        }
    }
    t = block_end;
}

#pragma clang loop unroll(full)
for (uint h = 0; h < uint(HEADS_PER_SG); ++h) {
    uint part_row = (first_row + h) * splits + split;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < uint(VALUES_PER_THREAD); ++i) {
        uint d = LANE_DIM(lane, i);
        if (DIM_OK(d)) {
            partials[part_row * uint(HEAD_DIM) + d] = o[h][i];
        }
    }
    if (lane == 0) {
        maxs[part_row] = max_score[h];
        sums[part_row] = sum_exp[h];
    }
}
"#;

/// Second pass: one threadgroup per query head and 32 dimensions. Each
/// simdgroup rescales a strided subset of the splits to the global max (one
/// dimension per lane), then simdgroup 0 adds them up and normalizes. A row
/// with no positions yields zeros.
const PAGED_SDPA_REDUCE_SOURCE: &str = r#"
uint tg = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint splits = uint(params[1]);
uint dim_groups = (uint(HEAD_DIM) + 31) / 32;
uint row = tg / dim_groups;
uint d = (tg % dim_groups) * 32 + lane;
uint base = row * splits;

float global_max = -3.4028234663852886e38f;
for (uint s = lane; s < splits; s += 32) {
    global_max = max(global_max, maxs[base + s]);
}
global_max = simd_max(global_max);

float denom = 0.0f;
float acc = 0.0f;
for (uint s = sg; s < splits; s += uint(REDUCE_SIMDGROUPS)) {
    float split_sum = sums[base + s];
    if (split_sum > 0.0f) {
        float w = exp(maxs[base + s] - global_max);
        denom += w * split_sum;
        if (d < uint(HEAD_DIM)) {
            acc += w * partials[(base + s) * uint(HEAD_DIM) + d];
        }
    }
}

threadgroup float acc_sh[REDUCE_SIMDGROUPS * 32];
threadgroup float denom_sh[REDUCE_SIMDGROUPS];
acc_sh[sg * 32 + lane] = acc;
if (lane == 0) {
    denom_sh[sg] = denom;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float total = 0.0f;
    float total_denom = 0.0f;
    for (uint i = 0; i < uint(REDUCE_SIMDGROUPS); ++i) {
        total += acc_sh[i * 32 + lane];
        total_denom += denom_sh[i];
    }
    if (d < uint(HEAD_DIM)) {
        out[row * uint(HEAD_DIM) + d] =
            static_cast<__typeof__(*out)>(total_denom > 0.0f ? total / total_denom : 0.0f);
    }
}
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
    paged_scaled_dot_product_attention_decode_from_on(
        queries,
        k_pages,
        v_pages,
        block_table,
        lengths,
        None,
        scale,
        block_size,
        target,
    )
}

/// [`paged_scaled_dot_product_attention_decode_on`] over part of each row:
/// row `b` attends to positions `starts[b]..lengths[b]` only (`starts`:
/// `[B]` int32; `None` is all zeros). This is how a sliding-window layer
/// reads a cache that holds more than its window.
#[allow(clippy::too_many_arguments)]
pub fn paged_scaled_dot_product_attention_decode_from_on(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    starts: Option<&Array>,
    scale: f32,
    block_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let shape =
        validate_decode_inputs(queries, k_pages, v_pages, block_table, lengths, block_size)?;
    let zeros;
    let starts = match starts {
        Some(starts) => {
            if starts.dtype() != Dtype::Int32 || starts.shape().as_slice() != [shape.batch] {
                return Err(Error::Mlx(format!(
                    "paged SDPA decode starts must be int32 [{}], got {:?} {:?}",
                    shape.batch,
                    starts.dtype(),
                    starts.shape().as_slice()
                )));
            }
            starts
        }
        None => {
            zeros = Array::zeros((shape.batch,), Dtype::Int32)?;
            &zeros
        }
    };
    paged_scaled_dot_product_attention_decode_split_on(
        queries,
        k_pages,
        v_pages,
        block_table,
        lengths,
        starts,
        scale,
        shape,
        target.into(),
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_scaled_dot_product_attention_decode_split_on(
    queries: &Array,
    k_pages: &Array,
    v_pages: &Array,
    block_table: &Array,
    lengths: &Array,
    starts: &Array,
    scale: f32,
    shape: DecodeShape,
    target: StreamOrDevice,
) -> Result<Array> {
    let q_per_kv = shape.q_heads / shape.kv_heads;
    let heads_per_tg = heads_per_threadgroup(q_per_kv);
    let head_groups = q_per_kv / heads_per_tg;
    let heads_per_sg = heads_per_simdgroup(heads_per_tg);
    let simdgroups = heads_per_tg / heads_per_sg;
    let plan = SplitPlan::new(
        shape.batch * shape.kv_heads * head_groups,
        shape.max_seq_len(),
    );
    let scale_arr: Array = (&[scale][..], (1_i32,)).try_into()?;
    let params: Array = (
        &[shape.max_blocks, plan.splits, plan.split_len][..],
        (3_i32,),
    )
        .try_into()?;
    let rows = shape.batch * shape.q_heads;
    let threadgroups = shape.batch * shape.kv_heads * head_groups * plan.splits;

    let mut split_outputs = cached_kernel(
        &SPLIT_KERNEL,
        "mlx_fast_paged_sdpa_decode_split",
        &[
            "queries",
            "k_pages",
            "v_pages",
            "block_table",
            "lengths",
            "starts",
            "scale_arr",
            "params",
        ],
        &["partials", "maxs", "sums"],
        PAGED_SDPA_SPLIT_SOURCE,
    )?
    .dispatch_builder()
    .inputs(&[
        queries,
        k_pages,
        v_pages,
        block_table,
        lengths,
        starts,
        &scale_arr,
        &params,
    ])
    .output_shapes(&[
        Shape::from((rows, plan.splits, shape.head_dim)),
        Shape::from((rows, plan.splits)),
        Shape::from((rows, plan.splits)),
    ])
    .output_dtypes(&[Dtype::Float32, Dtype::Float32, Dtype::Float32])
    .grid(threadgroups * PAGED_SDPA_DECODE_SIMD_WIDTH, simdgroups, 1)
    .threadgroup(PAGED_SDPA_DECODE_SIMD_WIDTH, simdgroups, 1)
    .template_int("H_Q", shape.q_heads)
    .template_int("H_KV", shape.kv_heads)
    .template_int("Q_PER_KV", q_per_kv)
    .template_int("HEADS_PER_TG", heads_per_tg)
    .template_int("HEAD_GROUPS", head_groups)
    .template_int("HEAD_DIM", shape.head_dim)
    .template_int("VALUES_PER_THREAD", shape.values_per_thread)
    .template_int("BLOCK_SIZE", shape.block_size)
    .template_int("HEADS_PER_SG", heads_per_sg)
    .template_int(
        "CONTIGUOUS",
        i32::from(shape.head_dim % PAGED_SDPA_DECODE_SIMD_WIDTH == 0),
    )
    .stream(target)
    .dispatch()?;
    let partials = split_outputs.take_at(0)?;
    let maxs = split_outputs.take_at(0)?;
    let sums = split_outputs.take_at(0)?;

    let dim_groups =
        (shape.head_dim + PAGED_SDPA_DECODE_SIMD_WIDTH - 1) / PAGED_SDPA_DECODE_SIMD_WIDTH;
    let mut outputs = cached_kernel(
        &REDUCE_KERNEL,
        "mlx_fast_paged_sdpa_decode_reduce",
        &["partials", "maxs", "sums", "params"],
        &["out"],
        PAGED_SDPA_REDUCE_SOURCE,
    )?
    .dispatch_builder()
    .inputs(&[&partials, &maxs, &sums, &params])
    .output_shapes(&[Shape::from((
        shape.batch,
        shape.q_heads,
        1_i32,
        shape.head_dim,
    ))])
    .output_dtypes(&[queries.dtype()])
    .grid(
        rows * dim_groups * PAGED_SDPA_DECODE_SIMD_WIDTH,
        PAGED_SDPA_REDUCE_SIMDGROUPS,
        1,
    )
    .threadgroup(
        PAGED_SDPA_DECODE_SIMD_WIDTH,
        PAGED_SDPA_REDUCE_SIMDGROUPS,
        1,
    )
    .template_int("HEAD_DIM", shape.head_dim)
    .template_int("REDUCE_SIMDGROUPS", PAGED_SDPA_REDUCE_SIMDGROUPS)
    .stream(target)
    .dispatch()?;
    outputs.take_at(0)
}

/// Query heads one simdgroup keeps in registers: the most that divide the
/// threadgroup's heads, up to the cap.
fn heads_per_simdgroup(heads_per_tg: i32) -> i32 {
    (1..=heads_per_tg.min(PAGED_SDPA_MAX_HEADS_PER_SIMDGROUP))
        .rev()
        .find(|h| heads_per_tg % h == 0)
        .unwrap_or(1)
}

/// Largest divisor of `q_per_kv` that fits one threadgroup.
fn heads_per_threadgroup(q_per_kv: i32) -> i32 {
    (1..=q_per_kv.min(PAGED_SDPA_MAX_HEADS_PER_THREADGROUP))
        .rev()
        .find(|h| q_per_kv % h == 0)
        .unwrap_or(1)
}

/// How the table's positions are cut into splits. Positions past a row's
/// length leave their splits empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitPlan {
    splits: i32,
    split_len: i32,
}

impl SplitPlan {
    fn new(base_threadgroups: i32, max_seq_len: i32) -> Self {
        let max_seq_len = max_seq_len.max(1);
        let wanted = (PAGED_SDPA_TARGET_THREADGROUPS + base_threadgroups - 1) / base_threadgroups;
        let split_len = ((max_seq_len + wanted - 1) / wanted).max(PAGED_SDPA_MIN_SPLIT_LEN);
        let splits = (max_seq_len + split_len - 1) / split_len;
        Self { splits, split_len }
    }
}

static SPLIT_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static REDUCE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn cached_kernel(
    cell: &'static OnceLock<MetalKernel>,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    source: &str,
) -> Result<&'static MetalKernel> {
    if let Some(kernel) = cell.get() {
        return Ok(kernel);
    }
    let kernel = MetalKernel::builder(name)
        .inputs(inputs)
        .outputs(outputs)
        .source(source)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?;
    Ok(cell.get_or_init(|| kernel))
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
