//! Routed + shared experts on the M5 tensor unit (Metal 4 MPP `matmul2d`
//! reading 4-bit codes), for every batch width (prefill, verify, decode).
//!
//! Each MoE layer keeps one resident copy of its expert weights, permuted
//! (exact byte permutation) into the M5 tiled layout of `m5_affine4` per
//! expert at start-up (or on first use). Every route of the batch, including the shared
//! expert (expert `E` for every token), is grouped by expert into blocks of
//! at most 16 rows; routes keep compact rows (no padding to the blocks). The
//! rows are gathered, then each block streams its expert's codes once per
//! column tile. Five dispatches per layer: group, gather (+ group sums),
//! gate/up + SwiGLU, down, weighted combine (+ the decoder residual).
//!
//! A row's output depends only on its own input and on the batch's width
//! class (the tensor op is row invariant and every reduction runs in a fixed
//! order): calls of fewer than 256 tokens (decode, verify, short prefill)
//! split each pass's K across threadgroup slices, wider calls (prefill) do
//! not. Within a class results do not depend on batch width, block
//! composition or route order.
//!
//! M5 profile setting `MOE_M5_EXPERTS`; affine 4-bit, group 64 only.

use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, ensure};
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

const ROWS: i32 = 16;
/// Columns per tensor op: the gate/up pass computes 32 gate and the matching
/// 32 up outputs in one 64-wide op; the down pass 32 outputs.
const GATE_UP_TILE: i32 = 64;
const DOWN_TILE: i32 = 32;
/// K slices per threadgroup (best in a dependent 40-layer chain, M5 Max).
const GATE_UP_SPLIT_K: i32 = 4;
const DOWN_SPLIT_K: i32 = 2;
/// K slices per threadgroup of wide calls (at least [`WIDE_TOKENS`]): one
/// slice keeps more threadgroups per block resident. Dependent 40-layer
/// chain, uniform routing, M5 Max: 2048 tokens 3.95 -> 3.51 ms per layer,
/// 1024 2.24 -> 2.00, 512 1.36 -> 1.26, 256-384 equal, 136 0.90 -> 0.93
/// (so narrower calls keep the split). In-process prefill: six 30K prompts
/// TTFT -0.52 s (-5.3%), natural 291-5934-token prompts 0 to -4.5%. The
/// changed summation order moves
/// teacher-forced NLL as any other valid split does (natural English,
/// Chinese, Rust: mean +0.0003 / +0.084 / -0.040 against 2/2 +0.064 /
/// -0.021 / -0.079 and 4/1 +0.022 / +0.019 / -0.071; top-1 agreement
/// 92-94% for all three, MoE routing amplifies the last-bit differences).
const WIDE_GATE_UP_SPLIT_K: i32 = 1;
const WIDE_DOWN_SPLIT_K: i32 = 1;
const HEADER: i32 = 4;
/// Widest token slice per dispatch chain; wider batches run in slices
/// (row invariance keeps the result identical).
const MAX_SLICE_TOKENS: i32 = 2048;
/// Fewest tokens of a wide call: the split-K counts above drop to one and
/// the down pass reads precomputed activation group sums (one dispatch more,
/// every column tile's sums saved). Decided per call, so the slices of a
/// call wider than [`MAX_SLICE_TOKENS`] share the class.
const WIDE_TOKENS: i32 = 256;

static AB_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Diagnostic in-process A/B (benchmarks only): with
/// `IRONMLX_DIAGNOSTIC_MOE_M5_AB=1` at load both expert layouts stay
/// resident and `set_ab_off(true)` routes experts through the checkpoint
/// layout again.
#[doc(hidden)]
pub fn set_ab_off(off: bool) {
    AB_OFF.store(off, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn ab_off() -> bool {
    AB_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn ab_both_resident() -> bool {
    static BOTH: OnceLock<bool> = OnceLock::new();
    *BOTH.get_or_init(|| std::env::var("IRONMLX_DIAGNOSTIC_MOE_M5_AB").as_deref() == Ok("1"))
}

/// M5 profile setting `MOE_M5_EXPERTS` on a GPU with the tensor unit.
pub(crate) fn requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::MOE_M5_EXPERTS)
            && crate::nn::m5_affine4::tensor_unit_available()
    })
}

/// One affine 4-bit matrix in MLX's layout: codes `[.., N, K / 8]` uint32,
/// scales and biases `[.., N, K / 64]` bf16.
pub(crate) struct Native {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

/// Checkpoint weights of one sparse MoE block.
pub(crate) struct NativeExperts {
    /// Routed `[E, I, H / 8]`.
    pub gate: Native,
    pub up: Native,
    /// Routed `[E, H, I / 8]`.
    pub down: Native,
    /// Shared `[I, H / 8]`, `[I, H / 8]`, `[H, I / 8]`.
    pub shared_gate: Native,
    pub shared_up: Native,
    pub shared_down: Native,
}

/// Tiled codes `[B, N, K / 8]` (per matrix `[N/32][K/64][32][8]` words) and
/// `(scale, bias)` pairs `[B, K / 64, N, 2]`.
#[derive(Clone)]
struct Tiled {
    weight: Array,
    sb: Array,
}

#[derive(Clone)]
struct TiledExperts {
    /// Routed gate rows then up rows: `[E, 2I, ..]`.
    gate_up: Tiled,
    down: Tiled,
    /// Shared gate rows then up rows: `[1, 2I, ..]`.
    shared_gate_up: Tiled,
    shared_down: Tiled,
}

enum State {
    Native(Box<NativeExperts>),
    Tiled(TiledExperts),
    Poisoned,
}

/// Single-resident expert weights of one MoE layer.
pub(crate) struct Store {
    state: Mutex<State>,
    experts: i32,
    hidden: i32,
    intermediate: i32,
}

fn affine4(n: &Native, rows: i32, k: i32, batch: Option<i32>) -> bool {
    let w = n.weight.shape();
    let s = n.scales.shape();
    let w = w.as_slice();
    let s = s.as_slice();
    let lead_ok = |d: &[i32]| match batch {
        Some(b) => d.len() == 3 && d[0] == b,
        None => d.len() == 2,
    };
    n.weight.dtype() == Dtype::Uint32
        && n.scales.dtype() == Dtype::Bfloat16
        && n.biases.dtype() == Dtype::Bfloat16
        && lead_ok(w)
        && lead_ok(s)
        && w[w.len() - 2] == rows
        && w[w.len() - 1] == k / 8
        && s[s.len() - 2] == rows
        && s[s.len() - 1] == k / 64
        && n.biases.shape().as_slice() == s
}

impl Store {
    /// `None` when the weights are not affine 4-bit group 64 of supported
    /// shapes (the caller keeps MLX's routed experts).
    pub(crate) fn new(native: NativeExperts) -> Option<Self> {
        let g = native.gate.weight.shape();
        let g = g.as_slice();
        if g.len() != 3 {
            return None;
        }
        let (e, i) = (g[0], g[1]);
        let h = native.down.weight.shape().as_slice().get(1).copied()?;
        let ok = h % 512 == 0
            && i % 64 == 0
            && (1..1024).contains(&e)
            && affine4(&native.gate, i, h, Some(e))
            && affine4(&native.up, i, h, Some(e))
            && affine4(&native.down, h, i, Some(e))
            && affine4(&native.shared_gate, i, h, None)
            && affine4(&native.shared_up, i, h, None)
            && affine4(&native.shared_down, h, i, None);
        ok.then(|| Self {
            state: Mutex::new(State::Native(Box::new(native))),
            experts: e,
            hidden: h,
            intermediate: i,
        })
    }

    fn tiled(&self, target: StreamOrDevice) -> Result<TiledExperts> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("moe m5 experts: store poisoned"))?;
        if let State::Tiled(tiled) = &*state {
            return Ok(tiled.clone());
        }
        let State::Native(native) = std::mem::replace(&mut *state, State::Poisoned) else {
            return Err(anyhow!("moe m5 experts: an earlier conversion failed"));
        };
        let tiled = convert(&native, target)?;
        drop(native);
        mlx::transforms::clear_cache();
        *state = State::Tiled(tiled.clone());
        Ok(tiled)
    }

    /// Convert to the tiled layout now (start-up), not in the first request.
    pub(crate) fn prepare(&self, target: StreamOrDevice) -> Result<()> {
        self.tiled(target).map(|_| ())
    }
}

/// Tiled layout of `[B, N, K/8]` codes and `[B, N, K/64]` scales / biases.
// Layout conversion, one pass per output array (no intermediate arrays).
// Codes: output word `(e, t, g, r, w)` of the tiled layout `[N/NT][K/64][NT][8]`
// per expert; with interleaving (HALF < NT) output row `t * NT + r` takes
// rows of A and B alternately in HALF-row chunks.
const TILE_CODES: &str = r#"
  const ulong idx = thread_position_in_grid.x;
  if (idx >= ulong(E) * NO * KW) return;
  const ulong e = idx / (ulong(NO) * KW);
  const uint rem = uint(idx % (ulong(NO) * KW));
  constexpr uint KG8 = KW / 8;
  const uint w = rem % 8;
  const uint r = (rem / 8) % NT;
  const uint g = (rem / (8 * NT)) % KG8;
  const uint t = rem / (8 * NT * KG8);
  const uint n = t * NT + r;
  const uint chunk = n / HALF;
  const bool from_b = HALF < NT && (chunk % 2) == 1;
  const uint row = HALF < NT ? (chunk / 2) * HALF + n % HALF : n;
  const device uint32_t* src = from_b ? wb : wa;
  out[idx] = src[(e * NA + row) * KW + g * 8 + w];
"#;

// (scale, bias) pairs `[E, K/64, NO, 2]` from `[E, NA, K/64]` scales/biases.
const TILE_SB: &str = r#"
  const ulong idx = thread_position_in_grid.x;
  if (idx >= ulong(E) * KG * NO) return;
  const ulong e = idx / (ulong(KG) * NO);
  const uint g = uint((idx / NO) % KG);
  const uint n = uint(idx % NO);
  const uint chunk = n / HALF;
  const bool from_b = HALF < NT && (chunk % 2) == 1;
  const uint row = HALF < NT ? (chunk / 2) * HALF + n % HALF : n;
  const ulong src = (e * NA + row) * KG + g;
  out[2 * idx] = from_b ? sb_[src] : sa[src];
  out[2 * idx + 1] = from_b ? bb[src] : ba[src];
"#;

/// Tiled layout of `[E, N, K/8]` codes (`b` interleaved after `a` in
/// `width / 2` row chunks when given) and their scales / biases.
fn tile(a: &Native, b: Option<&Native>, width: i32, target: StreamOrDevice) -> Result<Tiled> {
    use mlx::ops::shape::expand_dims_on;
    let lift = |x: &Array| -> Result<Array> {
        Ok(if x.ndim() == 2 {
            expand_dims_on(x, 0, target)?
        } else {
            x.clone()
        })
    };
    let (wa, sa, ba) = (lift(&a.weight)?, lift(&a.scales)?, lift(&a.biases)?);
    let other = b.unwrap_or(a);
    let (wb, sbb, bb) = (
        lift(&other.weight)?,
        lift(&other.scales)?,
        lift(&other.biases)?,
    );
    let shape = wa.shape();
    let &[e, na, kw] = shape.as_slice() else {
        return Err(anyhow!("moe m5 experts: codes must be [E, N, K/8]"));
    };
    let kg = kw / 8;
    let (no, half) = if b.is_some() {
        (2 * na, width / 2)
    } else {
        (na, width)
    };
    static CODES: OnceLock<MetalKernel> = OnceLock::new();
    static SB: OnceLock<MetalKernel> = OnceLock::new();
    let total = |n: i64| i32::try_from(n).map_err(|_| anyhow!("moe m5 experts: layout too large"));
    let weight = kernel(
        &CODES,
        "ironmlx_moe_m5_tile_codes_v1",
        &["wa", "wb"],
        &["out"],
        TILE_CODES.to_string(),
        false,
    )?
    .dispatch_builder()
    .inputs(&[&wa, &wb])
    .output_shapes(&[Shape::from(&[e, no, kw][..])])
    .output_dtypes(&[Dtype::Uint32])
    .grid(total(i64::from(e) * i64::from(no) * i64::from(kw))?, 1, 1)
    .threadgroup(256, 1, 1)
    .template_int("E", e)
    .template_int("NA", na)
    .template_int("NO", no)
    .template_int("KW", kw)
    .template_int("NT", width)
    .template_int("HALF", half)
    .stream(target)
    .dispatch()?
    .take_at(0)?;
    let sb = kernel(
        &SB,
        "ironmlx_moe_m5_tile_sb_v1",
        &["sa", "sb_", "ba", "bb"],
        &["out"],
        TILE_SB.to_string(),
        false,
    )?
    .dispatch_builder()
    .inputs(&[&sa, &sbb, &ba, &bb])
    .output_shapes(&[Shape::from(&[e, kg, no, 2][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(total(i64::from(e) * i64::from(kg) * i64::from(no))?, 1, 1)
    .threadgroup(256, 1, 1)
    .template_int("E", e)
    .template_int("NA", na)
    .template_int("NO", no)
    .template_int("KG", kg)
    .template_int("NT", width)
    .template_int("HALF", half)
    .stream(target)
    .dispatch()?
    .take_at(0)?;
    Ok(Tiled { weight, sb })
}

fn convert(n: &NativeExperts, target: StreamOrDevice) -> Result<TiledExperts> {
    let gate_up = tile(&n.gate, Some(&n.up), GATE_UP_TILE, target)?;
    mlx::transforms::eval(&[&gate_up.weight, &gate_up.sb])?;
    let down = tile(&n.down, None, DOWN_TILE, target)?;
    mlx::transforms::eval(&[&down.weight, &down.sb])?;
    let shared_gate_up = tile(&n.shared_gate, Some(&n.shared_up), GATE_UP_TILE, target)?;
    let shared_down = tile(&n.shared_down, None, DOWN_TILE, target)?;
    mlx::transforms::eval(&[
        &shared_gate_up.weight,
        &shared_gate_up.sb,
        &shared_down.weight,
        &shared_down.sb,
    ])?;
    Ok(TiledExperts {
        gate_up,
        down,
        shared_gate_up,
        shared_down,
    })
}

// Packed grouping buffer (uint32):
//   [0] block count, [1] T, [2] KT, [3] MB | block expert [MB] |
//   block rows [MB] | block first row [MB] | route row [R]
// Rows are compact: an expert's routes take consecutive rows (no padding to
// the 16-row blocks), so the gathered inputs and outputs have R rows.
// A route is r < T * KT for token r / KT's slot r % KT, and T * KT + t for
// token t's shared expert (expert E). R = T * (KT + 1); MB bounds the block
// count: sum over experts of ceil(count / 16) <= min(R, R / 16 + E + 1).
const GROUP: &str = r#"
  const uint lid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint T = inds_shape[0];
  const uint KT = inds_shape[1];
  const uint R = T * (KT + 1);
  const uint MB = min(R, R / 16 + uint(E) + 1);
  threadgroup atomic_uint counts[NT];
  threadgroup uint block_start[NT];
  threadgroup uint row_start[NT];
  threadgroup uint simd_totals[NT / 32];
  threadgroup uint simd_rows[NT / 32];
  device uint* block_expert = packed + 4;
  device uint* block_rows = block_expert + MB;
  device uint* block_row0 = block_rows + MB;
  device uint* route_row = block_row0 + MB;
  atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = lid; r < R; r += NT) {
    const uint e = r < T * KT ? inds[r] : uint(E);
    atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint count = atomic_load_explicit(&counts[lid], memory_order_relaxed);
  const uint blocks = (count + 15) / 16;
  const uint within = simd_prefix_exclusive_sum(blocks);
  const uint rwithin = simd_prefix_exclusive_sum(count);
  if (lane == 31) {
    simd_totals[sg] = within + blocks;
    simd_rows[sg] = rwithin + count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = 0, rbefore = 0;
  for (uint s = 0; s < sg; ++s) {
    before += simd_totals[s];
    rbefore += simd_rows[s];
  }
  const uint start = before + within;
  const uint rstart = rbefore + rwithin;
  block_start[lid] = start;
  row_start[lid] = rstart;
  for (uint j = 0; j < blocks; ++j) {
    block_expert[start + j] = lid;
    block_rows[start + j] = min(16u, count - 16 * j);
    block_row0[start + j] = rstart + 16 * j;
  }
  if (lid == NT - 1) {
    packed[0] = start + blocks;
    packed[1] = T;
    packed[2] = KT;
    packed[3] = MB;
  }
  atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // A route's row in its expert's blocks; rows are computed independently,
  // so their order does not change any output.
  for (uint r = lid; r < R; r += NT) {
    const uint e = r < T * KT ? inds[r] : uint(E);
    const uint rank = atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    route_row[r] = row_start[e] + rank;
  }
"#;

const GATHER: &str = r#"
  const uint c = thread_position_in_grid.x;
  const uint r = thread_position_in_grid.y;
  const uint T = packed[1];
  const uint KT = packed[2];
  const uint MB = packed[3];
  if (c >= H / 8 || r >= T * (KT + 1)) return;
  const uint row = packed[4 + 3 * MB + r];
  const uint tok = r < T * KT ? r / KT : r - T * KT;
  const vec<bfloat, 8> v = ((const device vec<bfloat, 8>*)x)[ulong(tok) * (H / 8) + c];
  ((device vec<bfloat, 8>*)xg)[ulong(row) * (H / 8) + c] = v;
  // Group sums of 64 inputs: eight in order per thread, then a fixed tree
  // over the group's eight threads.
  float sum = 0.0f;
  for (int i = 0; i < 8; i++) sum += float(v[i]);
  sum += simd_shuffle_down(sum, 4);
  sum += simd_shuffle_down(sum, 2);
  sum += simd_shuffle_down(sum, 1);
  if (c % 8 == 0) xs[ulong(row) * (H / 64) + c / 8] = sum;
"#;

// Shared prologue of the two expert passes: fragment coordinates, block
// lookup and the K slice of this simdgroup.
const PROLOGUE: &str = r#"
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  constexpr int GS = 64, NF = NT / 16;
  constexpr int KG = K / GS;
  const uint blk = threadgroup_position_in_grid.y;
  if (blk >= packed[0]) return;
  const uint MB = packed[3];
  const uint expert = packed[4 + blk];
  const int rows = int(packed[4 + MB + blk]);
  const int64_t row0 = packed[4 + 2 * MB + blk];        // the block's first compact row
  const int tx = threadgroup_position_in_grid.x;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + row0 * K, dextents<int32_t, 2>(K, rows));
  // Group sums of the block's rows fm and fm + 8 (x enters as s * (q * x) + b * sum(x)).
  auto xsum = [&](int g, thread float& xs0, thread float& xs1) {
    xs0 = 0.0f; xs1 = 0.0f;
    if (fm < rows) {
      const device vec<bfloat, 8>* xp = (const device vec<bfloat, 8>*)((const device bfloat*)X + (row0 + fm) * K + g * GS);
      for (int w = 0; w < 8; w++) { const vec<bfloat, 8> v = xp[w]; for (int i = 0; i < 8; i++) xs0 += float(v[i]); }
    }
    if (fm + 8 < rows) {
      const device vec<bfloat, 8>* xp = (const device vec<bfloat, 8>*)((const device bfloat*)X + (row0 + fm + 8) * K + g * GS);
      for (int w = 0; w < 8; w++) { const vec<bfloat, 8> v = xp[w]; for (int i = 0; i < 8; i++) xs1 += float(v[i]); }
    }
  };
"#;

// Gate/up pass: a threadgroup owns gate columns [32 tx, 32 tx + 32) and the
// matching up columns; output act = silu(g) * u (g, u rounded to bf16 first).
const GATE_UP: &str = r#"
  constexpr int N2 = 2 * I;
  constexpr int HF = NF / 2;                            // gate fragments, then up fragments
  const bool routed = expert < uint(E);
  const device uchar* wb = routed ? (const device uchar*)Wgu + (int64_t)expert * (N2 * (K / 2)) : (const device uchar*)Wsgu;
  const device uint4* sbv = routed ? (const device uint4*)SBgu + (int64_t)expert * (KG * N2 / 4) : (const device uint4*)SBsgu;
  const int c0 = tx * NT;                               // first interleaved column of this tile
  const int n0 = tx * (NT / 2);                         // its first gate (and up) output
  float C[NF * 8];
  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
  // (scale, bias) pairs of this lane's columns, one group ahead.
  uint4 cs[NF], ns[NF];
  for (int f = 0; f < NF; f++) cs[f] = sbv[(g_begin * N2 + c0 + f * 16 + fn) / 4];
  const device float* xsr0 = XS + (row0 + fm) * KG;
  const device float* xsr1 = xsr0 + 8 * KG;
  for (int g = g_begin; g < g_end; g++) {
    if (g + 1 < g_end)
      for (int f = 0; f < NF; f++) ns[f] = sbv[((g + 1) * N2 + c0 + f * 16 + fn) / 4];
    const float xs0 = fm < rows ? xsr0[g] : 0.0f;
    const float xs1 = fm + 8 < rows ? xsr1[g] : 0.0f;
    auto a = tA.slice(g * GS, 0);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)wb + (int64_t)(tx * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int f = 0; f < NF; f++) {
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(cs[f]);
      for (int r = 0; r < 2; r++)
        for (int j = 0; j < 4; j++) {
          const int i = f * 8 + r * 4 + j;
          C[i] = fma(float(v[2 * j]), P[i], fma(float(v[2 * j + 1]), r ? xs1 : xs0, C[i]));
        }
    }
    for (int f = 0; f < NF; f++) cs[f] = ns[f];
  }
  // K slices are added in slice order.
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  if (SK > 1) {
    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
      for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
  }
  if (sg == 0)
    for (int f = 0; f < HF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        if (m >= rows) continue;
        const int n = n0 + f * 16 + fn;
        for (int j = 0; j < 4; j++) {
          const float gv = float(bfloat(C[f * 8 + r * 4 + j]));
          const float uv = float(bfloat(C[(f + HF) * 8 + r * 4 + j]));
          act[(row0 + m) * I + n + j] = bfloat(gv / (1.0f + metal::exp(-gv)) * uv);
        }
      }
"#;

const DOWN: &str = r#"
  const bool routed = expert < uint(E);
  const device uchar* wb = routed ? (const device uchar*)Wd + (int64_t)expert * (N * (K / 2)) : (const device uchar*)Wsd;
  const device uint4* sbv = routed ? (const device uint4*)SBd + (int64_t)expert * (KG * N / 4) : (const device uint4*)SBsd;
  const int n0 = tx * NT;
  float C[NF * 8];
  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
  uint4 cs[NF], ns[NF];
  for (int f = 0; f < NF; f++) cs[f] = sbv[(g_begin * N + n0 + f * 16 + fn) / 4];
  for (int g = g_begin; g < g_end; g++) {
    if (g + 1 < g_end)
      for (int f = 0; f < NF; f++) ns[f] = sbv[((g + 1) * N + n0 + f * 16 + fn) / 4];
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(cs[f]);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    for (int f = 0; f < NF; f++) cs[f] = ns[f];
    auto a = tA.slice(g * GS, 0);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)wb + (int64_t)(tx * KG + g) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT));
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    float xs0, xs1;
#ifdef IRONMLX_DOWN_SUMS
    xs0 = fm < rows ? XS[(row0 + fm) * KG + g] : 0.0f;
    xs1 = fm + 8 < rows ? XS[(row0 + fm + 8) * KG + g] : 0.0f;
#else
    xsum(g, xs0, xs1);
#endif
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++)
        for (int j = 0; j < 4; j++) {
          const int i = f * 8 + r * 4 + j;
          C[i] = fma(s[f][j], P[i], fma(bb[f][j], r ? xs1 : xs0, C[i]));
        }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  if (SK > 1) {
    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
      for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
  }
  if (sg == 0)
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        if (m >= rows) continue;
        const int n = n0 + f * 16 + fn;
        for (int j = 0; j < 4; j++) dn[(row0 + m) * N + n + j] = bfloat(C[f * 8 + r * 4 + j]);
      }
"#;

// Group sums of the activations for the down pass: per row and group of 64,
// eight vectors of eight added in order from zero, as the pass's own sum.
const ACT_SUMS: &str = r#"
  const uint g = thread_position_in_grid.x;
  const uint r = thread_position_in_grid.y;
  constexpr uint KG = I / 64;
  if (g >= KG) return;
  const device vec<bfloat, 8>* xp = (const device vec<bfloat, 8>*)((const device bfloat*)act + ulong(r) * I + g * 64);
  float acc = 0.0f;
  for (int w = 0; w < 8; w++) { const vec<bfloat, 8> v = xp[w]; for (int i = 0; i < 8; i++) acc += float(v[i]); }
  sums[ulong(r) * KG + g] = acc;
"#;

const COMBINE: &str = r#"
  const uint h = thread_position_in_grid.x;
  const uint t = thread_position_in_grid.y;
  const uint T = packed[1];
  const uint KT = packed[2];
  const uint MB = packed[3];
  if (h >= H || t >= T) return;
  const device uint* route_row = packed + 4 + 3 * MB;
  float acc = 0.0f;
  for (uint j = 0; j < KT; ++j) {
    const uint r = t * KT + j;
    acc += float(scores[r]) * float(dn[ulong(route_row[r]) * H + h]);
  }
  acc += float(shared_gate[t]) * float(dn[ulong(route_row[T * KT + t]) * H + h]);
  // RES: the decoder layer's residual add, as the separate bf16 add rounds.
  if (RES) y[ulong(t) * H + h] = ((const device bfloat*)res)[ulong(t) * H + h] + bfloat(acc);
  else y[ulong(t) * H + h] = bfloat(acc);
"#;

const MPP_HEADER: &str = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;";

fn kernel(
    slot: &'static OnceLock<MetalKernel>,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    source: String,
    mpp: bool,
) -> Result<&'static MetalKernel> {
    if slot.get().is_none() {
        let mut builder = MetalKernel::builder(name)
            .inputs(inputs)
            .outputs(outputs)
            .source(source);
        if mpp {
            builder = builder.header(MPP_HEADER);
        }
        let _ = slot.set(builder.build()?);
    }
    Ok(slot.get().expect("kernel built"))
}

/// Routed + shared expert output `[T, H]` (bf16) for `x` `[T, H]` bf16,
/// `inds` `[T, k]` uint32 expert ids, `scores` `[T, k]` routing weights and
/// `shared_gate` `[T, 1]` (the shared expert's sigmoid gate).
/// With `residual` `[T, H]`, returns `residual + experts` (the decoder
/// layer's residual add, same rounding).
pub(crate) fn forward_on(
    store: &Store,
    x: &Array,
    inds: &Array,
    scores: &Array,
    shared_gate: &Array,
    residual: Option<&Array>,
    target: StreamOrDevice,
) -> Result<Array> {
    let xs = x.shape();
    let &[t, h] = xs.as_slice() else {
        return Err(anyhow!(
            "moe m5 experts: x must be [T, H], got {:?}",
            xs.as_slice()
        ));
    };
    let is = inds.shape();
    let &[ti, kt] = is.as_slice() else {
        return Err(anyhow!(
            "moe m5 experts: inds must be [T, k], got {:?}",
            is.as_slice()
        ));
    };
    ensure!(
        ti == t
            && t >= 1
            && h == store.hidden
            && x.dtype() == Dtype::Bfloat16
            && inds.dtype() == Dtype::Uint32
            && scores.shape().as_slice() == [t, kt]
            && shared_gate.size() == t as usize,
        "moe m5 experts: unsupported shapes x={:?} inds={:?}",
        xs.as_slice(),
        is.as_slice()
    );
    let tiled = store.tiled(target)?;
    let wide = t >= WIDE_TOKENS;
    if t <= MAX_SLICE_TOKENS {
        return run(
            store,
            &tiled,
            x,
            inds,
            scores,
            shared_gate,
            residual,
            wide,
            target,
        );
    }
    use mlx::ops::indexing::slice_on;
    let mut parts = Vec::new();
    let mut start = 0;
    while start < t {
        let stop = (start + MAX_SLICE_TOKENS).min(t);
        let rows = |a: &Array, w: i32| slice_on(a, &[start, 0][..], &[stop, w][..], target);
        let gate = mlx::ops::shape::reshape_on(shared_gate, &[t, 1][..], target)?;
        let res = residual.map(|r| rows(r, h)).transpose()?;
        parts.push(run(
            store,
            &tiled,
            &rows(x, h)?,
            &rows(inds, kt)?,
            &rows(scores, kt)?,
            &rows(&gate, 1)?,
            res.as_ref(),
            wide,
            target,
        )?);
        start = stop;
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(mlx::ops::shape::concatenate_on(&refs, 0, target)?)
}

#[allow(clippy::too_many_arguments)]
fn run(
    store: &Store,
    w: &TiledExperts,
    x: &Array,
    inds: &Array,
    scores: &Array,
    shared_gate: &Array,
    residual: Option<&Array>,
    wide: bool,
    target: StreamOrDevice,
) -> Result<Array> {
    let (e, h, i) = (store.experts, store.hidden, store.intermediate);
    let (gate_up_sk, down_sk) = if wide {
        (WIDE_GATE_UP_SPLIT_K, WIDE_DOWN_SPLIT_K)
    } else {
        (GATE_UP_SPLIT_K, DOWN_SPLIT_K)
    };
    let shape = inds.shape();
    let (t, kt) = (shape.as_slice()[0], shape.as_slice()[1]);
    let routes = t * (kt + 1);
    let mb = routes.min(routes / ROWS + e + 1);
    let group_threads = (e + 1 + 31) / 32 * 32;
    ensure!(group_threads <= 1024, "moe m5 experts: too many experts");

    static GROUP_K: OnceLock<MetalKernel> = OnceLock::new();
    static GATHER_K: OnceLock<MetalKernel> = OnceLock::new();
    static GATE_UP_K: OnceLock<MetalKernel> = OnceLock::new();
    static DOWN_K: OnceLock<MetalKernel> = OnceLock::new();
    static ACT_SUM_K: OnceLock<MetalKernel> = OnceLock::new();
    static DOWN_SUMS_K: OnceLock<MetalKernel> = OnceLock::new();
    static COMBINE_K: OnceLock<MetalKernel> = OnceLock::new();

    let packed = kernel(
        &GROUP_K,
        "ironmlx_moe_m5_group_v1",
        &["inds"],
        &["packed"],
        GROUP.to_string(),
        false,
    )?
    .dispatch_builder()
    .inputs(&[inds])
    .output_shapes(&[Shape::from(&[HEADER + 3 * mb + routes][..])])
    .output_dtypes(&[Dtype::Uint32])
    .grid(group_threads, 1, 1)
    .threadgroup(group_threads, 1, 1)
    .template_int("E", e)
    .template_int("NT", group_threads)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    let mut gathered = kernel(
        &GATHER_K,
        "ironmlx_moe_m5_gather_v2",
        &["x", "packed"],
        &["xg", "xs"],
        GATHER.to_string(),
        false,
    )?
    .dispatch_builder()
    .inputs(&[x, &packed])
    .output_shapes(&[
        Shape::from(&[routes, h][..]),
        Shape::from(&[routes, h / 64][..]),
    ])
    .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
    .grid(h / 8, routes, 1)
    .threadgroup((h / 8).min(256), 1, 1)
    .template_int("H", h)
    .stream(target)
    .dispatch()?;
    let xg = gathered.take_at(0)?;
    let xsums = gathered.take_at(0)?;

    let act = kernel(
        &GATE_UP_K,
        "ironmlx_moe_m5_gate_up_v3",
        &["X", "XS", "packed", "Wgu", "SBgu", "Wsgu", "SBsgu"],
        &["act"],
        format!("{PROLOGUE}{GATE_UP}"),
        true,
    )?
    .dispatch_builder()
    .inputs(&[
        &xg,
        &xsums,
        &packed,
        &w.gate_up.weight,
        &w.gate_up.sb,
        &w.shared_gate_up.weight,
        &w.shared_gate_up.sb,
    ])
    .output_shapes(&[Shape::from(&[routes, i][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(2 * i / GATE_UP_TILE * 32 * gate_up_sk, mb, 1)
    .threadgroup(32 * gate_up_sk, 1, 1)
    .template_int("K", h)
    .template_int("I", i)
    .template_int("E", e)
    .template_int("SK", gate_up_sk)
    .template_int("NT", GATE_UP_TILE)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    // Wide calls: the activation group sums once per row, not per column
    // tile of the down pass.
    let act_sums = wide
        .then(|| -> Result<Array> {
            kernel(
                &ACT_SUM_K,
                "ironmlx_moe_m5_act_sums_v1",
                &["act"],
                &["sums"],
                ACT_SUMS.to_string(),
                false,
            )?
            .dispatch_builder()
            .inputs(&[&act])
            .output_shapes(&[Shape::from(&[routes, i / 64][..])])
            .output_dtypes(&[Dtype::Float32])
            .grid(i / 64, routes, 1)
            .threadgroup((i / 64).min(256), 1, 1)
            .template_int("I", i)
            .stream(target)
            .dispatch()?
            .take_at(0)
            .map_err(Into::into)
        })
        .transpose()?;
    let dn = kernel(
        if act_sums.is_some() {
            &DOWN_SUMS_K
        } else {
            &DOWN_K
        },
        if act_sums.is_some() {
            "ironmlx_moe_m5_down_sums_v1"
        } else {
            "ironmlx_moe_m5_down_v2"
        },
        &["X", "packed", "Wd", "SBd", "Wsd", "SBsd", "XS"],
        &["dn"],
        format!(
            "{}{PROLOGUE}{DOWN}",
            if act_sums.is_some() {
                "#define IRONMLX_DOWN_SUMS 1\n"
            } else {
                ""
            }
        ),
        true,
    )?
    .dispatch_builder()
    .inputs(&[
        &act,
        &packed,
        &w.down.weight,
        &w.down.sb,
        &w.shared_down.weight,
        &w.shared_down.sb,
        act_sums.as_ref().unwrap_or(&act),
    ])
    .output_shapes(&[Shape::from(&[routes, h][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(h / DOWN_TILE * 32 * down_sk, mb, 1)
    .threadgroup(32 * down_sk, 1, 1)
    .template_int("K", i)
    .template_int("N", h)
    .template_int("E", e)
    .template_int("SK", down_sk)
    .template_int("NT", DOWN_TILE)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    // Without a residual, `y` itself stands in for the unread input slot.
    let res = residual.unwrap_or(x);
    Ok(kernel(
        &COMBINE_K,
        "ironmlx_moe_m5_combine_v2",
        &["dn", "packed", "scores", "shared_gate", "res"],
        &["y"],
        COMBINE.to_string(),
        false,
    )?
    .dispatch_builder()
    .inputs(&[&dn, &packed, scores, shared_gate, res])
    .template_int("RES", i32::from(residual.is_some()))
    .output_shapes(&[Shape::from(&[t, h][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(h, t, 1)
    .threadgroup(256, 1, 1)
    .template_int("H", h)
    .stream(target)
    .dispatch()?
    .take_at(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::ops::cast::astype;
    use serial_test::serial;

    fn quantized(shape: &[i32], seed: u64) -> Result<Native> {
        let key = mlx::random::key(seed)?;
        let w = mlx::random::normal()
            .shape(shape)
            .scale(0.05)
            .key(&key)
            .sample()?;
        let w = astype(&w, Dtype::Bfloat16)?;
        let q = mlx::quantization::quantize(&w, Some(64), Some(4), "affine", None)?;
        mlx::transforms::eval(&[&q[0], &q[1], &q[2]])?;
        Ok(Native {
            weight: q[0].clone(),
            scales: q[1].clone(),
            biases: q[2].clone(),
        })
    }

    fn native(e: i32, h: i32, i: i32) -> Result<NativeExperts> {
        Ok(NativeExperts {
            gate: quantized(&[e, i, h], 1)?,
            up: quantized(&[e, i, h], 2)?,
            down: quantized(&[e, h, i], 3)?,
            shared_gate: quantized(&[i, h], 4)?,
            shared_up: quantized(&[i, h], 5)?,
            shared_down: quantized(&[h, i], 6)?,
        })
    }

    fn dequant(n: &Native) -> Result<Array> {
        let d = mlx::quantization::dequantize(
            &n.weight,
            &n.scales,
            Some(&n.biases),
            Some(64),
            Some(4),
            "affine",
            None,
            None,
        )?;
        Ok(astype(&d, Dtype::Float32)?)
    }

    /// Float32 reference of the routed + shared SwiGLU MoE.
    fn reference(
        n: &NativeExperts,
        x: &[f32],
        inds: &[u32],
        scores: &[f32],
        gates: &[f32],
        h: usize,
        i: usize,
        kt: usize,
    ) -> Result<Vec<f32>> {
        use mlx::ops::indexing::take;
        let t = x.len() / h;
        let xa: Array = (x, &[t as i32, h as i32][..]).try_into()?;
        let (gw, uw, dw) = (dequant(&n.gate)?, dequant(&n.up)?, dequant(&n.down)?);
        let (sgw, suw, sdw) = (
            dequant(&n.shared_gate)?,
            dequant(&n.shared_up)?,
            dequant(&n.shared_down)?,
        );
        let swiglu = |g: &Array, u: &Array| -> Result<Array> {
            let s = g * &mlx::ops::unary::sigmoid(g)?;
            Ok(&s * u)
        };
        let mut out = vec![0.0f32; t * h];
        for tok in 0..t {
            let xr = mlx::ops::indexing::slice(
                &xa,
                &[tok as i32, 0][..],
                &[tok as i32 + 1, h as i32][..],
            )?;
            let mut acc: Option<Array> = None;
            let mut add = |y: Array, wgt: f32| {
                let y = &y * wgt;
                acc = Some(match acc.take() {
                    Some(a) => &a + &y,
                    None => y,
                });
            };
            for j in 0..kt {
                let ex: Array = (&[inds[tok * kt + j] as i32][..], &[1][..]).try_into()?;
                let pick = |w: &Array| -> Result<Array> {
                    Ok(mlx::ops::shape::squeeze(&take(w, &ex, 0)?, 0)?)
                };
                let g = mlx::ops::matmul::matmul(&xr, &mlx::ops::shape::transpose(&pick(&gw)?)?)?;
                let u = mlx::ops::matmul::matmul(&xr, &mlx::ops::shape::transpose(&pick(&uw)?)?)?;
                let a = swiglu(&g, &u)?;
                add(
                    mlx::ops::matmul::matmul(&a, &mlx::ops::shape::transpose(&pick(&dw)?)?)?,
                    scores[tok * kt + j],
                );
            }
            let g = mlx::ops::matmul::matmul(&xr, &mlx::ops::shape::transpose(&sgw)?)?;
            let u = mlx::ops::matmul::matmul(&xr, &mlx::ops::shape::transpose(&suw)?)?;
            let a = swiglu(&g, &u)?;
            add(
                mlx::ops::matmul::matmul(&a, &mlx::ops::shape::transpose(&sdw)?)?,
                gates[tok],
            );
            let row = acc.expect("routes").to_vec::<f32>()?;
            out[tok * h..(tok + 1) * h].copy_from_slice(&row);
        }
        let _ = i;
        Ok(out)
    }

    #[test]
    #[serial(mlx_metal)]
    fn matches_reference_and_is_batch_invariant() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !crate::nn::m5_affine4::tensor_unit_available() {
            return Ok(());
        }
        let (e, h, i, kt) = (16_i32, 512_i32, 128_i32, 4_usize);
        let store = Store::new(native(e, h, i)?).expect("supported");
        let reference_weights = native(e, h, i)?;
        let t = 40_usize;
        let x: Vec<f32> = (0..t * h as usize)
            .map(|v| (((v * 7919) % 211) as f32 - 105.0) * 0.01)
            .collect();
        // Distinct experts per token; two token classes share four experts
        // each, so every routed expert and the shared expert span 2-3 blocks.
        let inds: Vec<u32> = (0..t)
            .flat_map(|tok| (0..kt).map(move |j| ((tok % 2 + 3 * j) % e as usize) as u32))
            .collect();
        let scores: Vec<f32> = (0..t * kt).map(|v| 0.05 + (v % 7) as f32 * 0.03).collect();
        let gates: Vec<f32> = (0..t).map(|v| 0.3 + (v % 5) as f32 * 0.1).collect();
        let run_rows = |rows: std::ops::Range<usize>| -> Result<Vec<f32>> {
            let n = rows.len();
            let xa: Array = (
                &x[rows.start * h as usize..rows.end * h as usize],
                &[n as i32, h][..],
            )
                .try_into()?;
            let xa = astype(&xa, Dtype::Bfloat16)?;
            let ia: Array = (
                &inds[rows.start * kt..rows.end * kt],
                &[n as i32, kt as i32][..],
            )
                .try_into()?;
            let sa: Array = (
                &scores[rows.start * kt..rows.end * kt],
                &[n as i32, kt as i32][..],
            )
                .try_into()?;
            let ga: Array = (&gates[rows.clone()], &[n as i32, 1][..]).try_into()?;
            let ga = astype(&ga, Dtype::Bfloat16)?;
            let y = forward_on(&store, &xa, &ia, &sa, &ga, None, StreamOrDevice::default())?;
            Ok(astype(&y, Dtype::Float32)?.to_vec::<f32>()?)
        };
        let all = run_rows(0..t)?;
        // Batch invariance: every token alone, and a narrow slice.
        for tok in [0, 1, 17, t - 1] {
            let one = run_rows(tok..tok + 1)?;
            assert_eq!(
                one.as_slice(),
                &all[tok * h as usize..(tok + 1) * h as usize],
                "token {tok}"
            );
        }
        let mid = run_rows(5..13)?;
        assert_eq!(mid.as_slice(), &all[5 * h as usize..13 * h as usize]);
        // Accuracy against a float32 reference (bf16 inputs).
        let xb: Vec<f32> = {
            let xa: Array = (x.as_slice(), &[t as i32, h][..]).try_into()?;
            astype(&astype(&xa, Dtype::Bfloat16)?, Dtype::Float32)?.to_vec::<f32>()?
        };
        let gb: Vec<f32> = {
            let ga: Array = (gates.as_slice(), &[t as i32][..]).try_into()?;
            astype(&astype(&ga, Dtype::Bfloat16)?, Dtype::Float32)?.to_vec::<f32>()?
        };
        let want = reference(
            &reference_weights,
            &xb,
            &inds,
            &scores,
            &gb,
            h as usize,
            i as usize,
            kt,
        )?;
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let err = (all
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / want.len() as f32)
            .sqrt();
        eprintln!("moe m5 experts rel_rms_error={}", err / rms);
        assert!(err / rms < 0.02, "relative rms error {}", err / rms);
        Ok(())
    }

    /// Wide calls (one K slice, down pass reading precomputed activation
    /// group sums) match the float32 reference as narrow calls do, give
    /// every token the same bits in any wide call, and keep the class for
    /// the slices of a call wider than `MAX_SLICE_TOKENS` (its 60-token
    /// tail slice included).
    #[test]
    #[serial(mlx_metal)]
    fn wide_calls_match_reference_and_are_batch_invariant() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        if !crate::nn::m5_affine4::tensor_unit_available() {
            return Ok(());
        }
        let (e, h, i, kt) = (16_i32, 512_i32, 128_i32, 4_usize);
        let store = Store::new(native(e, h, i)?).expect("supported");
        let reference_weights = native(e, h, i)?;
        let t = (MAX_SLICE_TOKENS + 60) as usize;
        let x: Vec<f32> = (0..t * h as usize)
            .map(|v| (((v * 7919) % 211) as f32 - 105.0) * 0.01)
            .collect();
        let inds: Vec<u32> = (0..t)
            .flat_map(|tok| {
                (0..kt).map(move |j| {
                    let hot = (tok * 13 + j * 5) % 23;
                    (if hot < 9 {
                        j
                    } else {
                        (tok * 7 + j * 3) % e as usize
                    }) as u32
                })
            })
            .collect();
        let scores: Vec<f32> = (0..t * kt).map(|v| 0.05 + (v % 7) as f32 * 0.03).collect();
        let gates: Vec<f32> = (0..t).map(|v| 0.3 + (v % 5) as f32 * 0.1).collect();
        let run_rows = |rows: std::ops::Range<usize>| -> Result<Vec<f32>> {
            let n = rows.len() as i32;
            let xa: Array = (
                &x[rows.start * h as usize..rows.end * h as usize],
                &[n, h][..],
            )
                .try_into()?;
            let ia: Array =
                (&inds[rows.start * kt..rows.end * kt], &[n, kt as i32][..]).try_into()?;
            let sa: Array =
                (&scores[rows.start * kt..rows.end * kt], &[n, kt as i32][..]).try_into()?;
            let ga: Array = (&gates[rows.clone()], &[n, 1][..]).try_into()?;
            let y = forward_on(
                &store,
                &astype(&xa, Dtype::Bfloat16)?,
                &ia,
                &sa,
                &astype(&ga, Dtype::Bfloat16)?,
                None,
                StreamOrDevice::default(),
            )?;
            Ok(astype(&y, Dtype::Float32)?.to_vec::<f32>()?)
        };
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        let wide = 520_usize;
        assert!(wide as i32 >= 2 * WIDE_TOKENS);
        let all = run_rows(0..wide)?;
        let mut halves = run_rows(0..wide / 2)?;
        halves.extend(run_rows(wide / 2..wide)?);
        assert!(bits(&all) == bits(&halves), "wide calls differ by width");
        // Sliced call: the 60-token tail slice stays in the wide class.
        let sliced = run_rows(0..t)?;
        let last = t - WIDE_TOKENS as usize;
        assert!(
            bits(&sliced[last * h as usize..]) == bits(&run_rows(last..t)?),
            "a slice of a wide call left the wide class"
        );
        assert!(bits(&sliced[..wide * h as usize]) == bits(&all));
        // Accuracy against a float32 reference (bf16 inputs).
        let xb: Vec<f32> = {
            let xa: Array = (&x[..wide * h as usize], &[wide as i32, h][..]).try_into()?;
            astype(&astype(&xa, Dtype::Bfloat16)?, Dtype::Float32)?.to_vec::<f32>()?
        };
        let gb: Vec<f32> = {
            let ga: Array = (&gates[..wide], &[wide as i32][..]).try_into()?;
            astype(&astype(&ga, Dtype::Bfloat16)?, Dtype::Float32)?.to_vec::<f32>()?
        };
        let want = reference(
            &reference_weights,
            &xb,
            &inds[..wide * kt],
            &scores[..wide * kt],
            &gb,
            h as usize,
            i as usize,
            kt,
        )?;
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let err = (all
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / want.len() as f32)
            .sqrt();
        eprintln!("moe m5 experts wide rel_rms_error={}", err / rms);
        assert!(err / rms < 0.02, "relative rms error {}", err / rms);
        Ok(())
    }

    /// Diagnostic: a 40-layer dependent chain (each layer's output is the
    /// next layer's input) of this route versus the expert-tile route on
    /// Qwen3.6 MoE shapes with rotating expert sets. Run with --ignored.
    #[test]
    #[ignore]
    #[serial(mlx_metal)]
    fn chained_layers_versus_tile_route() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        use crate::nn::moe_tile_experts::{self, Quant, Weights};
        use mlx::ops::shape::concatenate;
        let (e, h, i) = (256, 2048, 512);
        let n = native(e, h, i)?;
        let gu_w = concatenate(&[&n.gate.weight, &n.up.weight], 1)?;
        let gu_s = concatenate(&[&n.gate.scales, &n.up.scales], 1)?;
        let gu_b = concatenate(&[&n.gate.biases, &n.up.biases], 1)?;
        mlx::transforms::eval(&[&gu_w, &gu_s, &gu_b])?;
        fn q(a: &Native) -> Quant<'_> {
            Quant {
                weight: &a.weight,
                scales: &a.scales,
                biases: &a.biases,
            }
        }
        let weights = Weights {
            gate_up: Quant {
                weight: &gu_w,
                scales: &gu_s,
                biases: &gu_b,
            },
            down: q(&n.down),
            shared_gate: q(&n.shared_gate),
            shared_up: q(&n.shared_up),
            shared_down: q(&n.shared_down),
        };
        let store = Store::new(native(e, h, i)?).expect("supported");
        let target = StreamOrDevice::default();
        let _tile = moe_tile_experts::scope();
        let layers = 40;
        for (label, t, unique) in [
            ("B1q8", 8, 33),
            ("B1q4", 4, 20),
            ("B1q1", 1, 8),
            ("B4q8", 32, 90),
        ] {
            let kt = 8;
            let mut routes = Vec::new();
            for layer in 0..layers {
                let experts: Vec<i32> = (0..unique).map(|j| (layer * 41 + j * 7) % e).collect();
                let inds: Vec<u32> = (0..t * kt)
                    .map(|r| experts[(r % unique) as usize] as u32)
                    .collect();
                let inds: Array = (inds.as_slice(), &[t, kt][..]).try_into()?;
                let scores: Vec<f32> = (0..t * kt).map(|v| 0.02 + (v % 5) as f32 * 0.01).collect();
                let scores: Array = (scores.as_slice(), &[t, kt][..]).try_into()?;
                let gate: Vec<f32> = (0..t).map(|v| 0.2 + (v % 3) as f32 * 0.1).collect();
                let gate: Array = (gate.as_slice(), &[t, 1][..]).try_into()?;
                let gate = astype(&gate, Dtype::Bfloat16)?;
                mlx::transforms::eval(&[&inds, &scores, &gate])?;
                routes.push((inds, scores, gate));
            }
            let x0: Vec<f32> = (0..t * 2048)
                .map(|v| ((v % 17) as f32 - 8.0) * 0.05)
                .collect();
            let x0: Array = (x0.as_slice(), &[t, 2048][..]).try_into()?;
            let x0 = astype(&x0, Dtype::Bfloat16)?;
            mlx::transforms::eval(&[&x0])?;
            let mut ms = [Vec::new(), Vec::new()];
            for round in 0..13 {
                for order in 0..2 {
                    let which = (round + order) % 2;
                    let started = std::time::Instant::now();
                    let mut x = x0.clone();
                    for (inds, scores, gate) in &routes {
                        let y = if which == 0 {
                            forward_on(&store, &x, inds, scores, gate, None, target)?
                        } else {
                            moe_tile_experts::forward_on(&x, inds, scores, gate, &weights, target)?
                        };
                        // Residual-like dependency keeps the chain sequential.
                        x = &x + &y;
                    }
                    mlx::transforms::eval(&[&x])?;
                    let v = started.elapsed().as_secs_f64() * 1e3;
                    if round > 0 {
                        ms[which].push(v);
                    }
                }
            }
            for v in &mut ms {
                v.sort_by(f64::total_cmp);
            }
            let (m5, tile) = (ms[0][ms[0].len() / 2], ms[1][ms[1].len() / 2]);
            eprintln!(
                "N={label} 40-layer chain m5_ms={m5:.3} tile_ms={tile:.3} tile/m5={:.3}",
                tile / m5
            );
        }
        Ok(())
    }
}
