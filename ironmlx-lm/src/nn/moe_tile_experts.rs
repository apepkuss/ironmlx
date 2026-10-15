//! Expert-tile MoE for narrow batches (decode and speculative verify).
//!
//! MLX's sorted `gather_qmm` costs roughly in proportion to the routed rows
//! rather than to the experts' weight bytes, so a batch of a few dozen tokens
//! spends most of its MoE time re-reading expert weights. Here every route of
//! the batch, including the shared expert (as expert `E` for every token), is
//! grouped by expert into tiles of at most 8 rows, and each tile streams its
//! expert's 4-bit weights once through 8x8 simdgroup MMAs
//! (`moe_tile_experts.metal`). Four dispatches per layer: group,
//! gate/up + SwiGLU (reading x by token), down, weighted combine.
//!
//! M5 profile setting `MOE_TILE_EXPERTS`; affine 4-bit, group 64 weights
//! only.

use std::sync::OnceLock;

use anyhow::ensure;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

/// Widest batch (tokens) the tile route takes; wider batches keep MLX's
/// sorted gather, whose cost amortizes over many rows per expert.
pub(crate) const MAX_TOKENS: i32 = 64;

const ROWS: i32 = 8;
const GATE_UP_SIMDGROUPS: i32 = 4;
const COLUMNS_PER_SIMDGROUP: i32 = 8;

thread_local! {
    static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Arms the route on this thread (DFlash2 verify windows); prefill and other
/// forwards keep MLX's routed-expert arithmetic.
pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

pub(crate) fn scope() -> Scope {
    DEPTH.with(|d| d.set(d.get() + 1));
    Scope
}

pub(crate) fn armed() -> bool {
    DEPTH.with(|d| d.get() > 0)
}

/// M5 profile setting `MOE_TILE_EXPERTS`.
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::MOE_TILE_EXPERTS)
    })
}

/// One affine 4-bit matrix: codes `[.., N, K / 8]` uint32, scales and biases
/// `[.., N, K / 64]` bf16.
#[derive(Clone, Copy)]
pub(crate) struct Quant<'a> {
    pub weight: &'a Array,
    pub scales: &'a Array,
    pub biases: &'a Array,
}

/// Weights of one sparse MoE block: routed gate/up fused along the output
/// axis (`[E, 2I, H/8]`, gate rows first), routed down `[E, H, I/8]`, and the
/// shared expert's gate `[I, H/8]`, up `[I, H/8]` and down `[H, I/8]`.
pub(crate) struct Weights<'a> {
    pub gate_up: Quant<'a>,
    pub down: Quant<'a>,
    pub shared_gate: Quant<'a>,
    pub shared_up: Quant<'a>,
    pub shared_down: Quant<'a>,
}

const GROUP: &str = r#"
  const uint lid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint T = inds_shape[0];
  const uint KT = inds_shape[1];
  const uint R = T * (KT + 1);
  const uint MT = R;
  threadgroup uint route_expert[kMaxRoutes];
  threadgroup atomic_uint counts[NT];
  threadgroup uint tile_start[NT];
  threadgroup uint simd_totals[NT / 32];
  device uint* tile_experts = packed + kHeader;
  device uint* grouped = tile_experts + MT;
  device uint* route_rows = grouped + MT * kRows;
  for (uint i = lid; i < MT * kRows; i += NT) grouped[i] = kNone;
  atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
  for (uint r = lid; r < R; r += NT) route_expert[r] = r < T * KT ? inds[r] : uint(E);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = lid; r < R; r += NT)
    atomic_fetch_add_explicit(&counts[route_expert[r]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint count = atomic_load_explicit(&counts[lid], memory_order_relaxed);
  const uint tiles = (count + kRows - 1) / kRows;
  const uint within = simd_prefix_exclusive_sum(tiles);
  if (lane == 31) simd_totals[sg] = within + tiles;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = 0;
  for (uint s = 0; s < sg; ++s) before += simd_totals[s];
  const uint start = before + within;
  tile_start[lid] = start;
  for (uint j = 0; j < tiles; ++j) tile_experts[start + j] = lid;
  if (lid == NT - 1) {
    packed[0] = start + tiles;
    packed[1] = T;
    packed[2] = KT;
    packed[3] = MT;
  }
  atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  // A route's row inside its expert's tiles; rows of a tile are computed
  // independently, so their order does not change any output.
  for (uint r = lid; r < R; r += NT) {
    const uint e = route_expert[r];
    const uint rank = atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    const uint row = tile_start[e] * kRows + rank;
    grouped[row] = r;
    route_rows[r] = row;
  }
"#;

const GATE_UP: &str = r#"
  constexpr uint BLOCKS = I / (SG * 8);
  const uint items = packed[0] * BLOCKS;
  const uint T = packed[1];
  const uint KT = packed[2];
  const uint MT = packed[3];
  const uint lane = thread_index_in_simdgroup;
  const ushort fm = frag_m(lane);
  const ushort fn = frag_n(lane);
  for (uint item = threadgroup_position_in_grid.x; item < items; item += threadgroups_per_grid.x) {
  const uint tile = item / BLOCKS;
  const uint expert = packed[kHeader + tile];
  const uint n0 = (item % BLOCKS) * (SG * 8) + simdgroup_index_in_threadgroup * 8;
  // Input rows fn and fn + 1 of the tile, read from x by token (padding rows
  // reuse the tile's first route; their outputs are never read).
  const device uint* grouped = packed + kHeader + MT + tile * kRows;
  auto token = [&](uint row) {
    uint route = grouped[row];
    route = route == kNone ? grouped[0] : route;
    return route < T * KT ? route / KT : route - T * KT;
  };
  const device half* xr0 = (const device half*)x + ulong(token(fn)) * H;
  const device half* xr1 = (const device half*)x + ulong(token(fn + 1)) * H;
  constexpr uint KB = H / 2;
  constexpr uint KG = H / 64;
  const device uchar* w0; const device bfloat* s0; const device bfloat* b0;
  const device uchar* w1; const device bfloat* s1; const device bfloat* b1;
  if (expert < E) {
    const ulong r0 = ulong(expert) * (2 * I) + n0;
    const ulong r1 = r0 + I;
    w0 = (const device uchar*)gu_w + r0 * KB; s0 = (const device bfloat*)gu_s + r0 * KG;
    b0 = (const device bfloat*)gu_b + r0 * KG;
    w1 = (const device uchar*)gu_w + r1 * KB; s1 = (const device bfloat*)gu_s + r1 * KG;
    b1 = (const device bfloat*)gu_b + r1 * KG;
  } else {
    w0 = (const device uchar*)sg_w + ulong(n0) * KB; s0 = (const device bfloat*)sg_s + ulong(n0) * KG;
    b0 = (const device bfloat*)sg_b + ulong(n0) * KG;
    w1 = (const device uchar*)su_w + ulong(n0) * KB; s1 = (const device bfloat*)su_s + ulong(n0) * KG;
    b1 = (const device bfloat*)su_b + ulong(n0) * KG;
  }
  simdgroup_float8x8 c0;
  simdgroup_float8x8 c1;
  moe_tile_block<true, 1>(xr0, xr1, H, w0, s0, b0, w1, s1, b1, lane, c0, c1);
  const vec<float, 2> g2 = reinterpret_cast<thread vec<float, 2>&>(c0.thread_elements());
  const vec<float, 2> u2 = reinterpret_cast<thread vec<float, 2>&>(c1.thread_elements());
  for (ushort i = 0; i < 2; ++i) {
    const float g = float(bfloat(g2[i]));
    const float u = float(bfloat(u2[i]));
    // Stored as half for the down pass; clamped to half's range.
    act[(ulong(tile) * kRows + fn + i) * I + n0 + fm] =
        half(clamp(g / (1.0f + metal::exp(-g)) * u, -65000.0f, 65000.0f));
  }
  }
"#;

const DOWN: &str = r#"
  constexpr uint BLOCKS = H / (SG * 8);
  const uint items = packed[0] * BLOCKS;
  const uint lane = thread_index_in_simdgroup;
  const ushort fm = frag_m(lane);
  const ushort fn = frag_n(lane);
  for (uint item = threadgroup_position_in_grid.x; item < items; item += threadgroups_per_grid.x) {
  const uint tile = item / BLOCKS;
  const uint expert = packed[kHeader + tile];
  const uint n0 = (item % BLOCKS) * (SG * 8) + simdgroup_index_in_threadgroup * 8;
  constexpr uint KB = I / 2;
  constexpr uint KG = I / 64;
  const ulong r0 = expert < E ? ulong(expert) * H + n0 : ulong(n0);
  const device uchar* w = (const device uchar*)(expert < E ? d_w : sd_w) + r0 * KB;
  const device bfloat* s = (const device bfloat*)(expert < E ? d_s : sd_s) + r0 * KG;
  const device bfloat* b = (const device bfloat*)(expert < E ? d_b : sd_b) + r0 * KG;
  simdgroup_float8x8 c0;
  simdgroup_float8x8 c1;
  const device half* ar = (const device half*)act + (ulong(tile) * kRows + fn) * I;
  moe_tile_block<false, 4>(ar, ar + I, I, w, s, b, w, s, b, lane, c0, c1);
  const vec<float, 2> d2 = reinterpret_cast<thread vec<float, 2>&>(c0.thread_elements());
  for (ushort i = 0; i < 2; ++i)
    ((device bfloat*)dn)[(ulong(tile) * kRows + fn + i) * H + n0 + fm] = bfloat(d2[i]);
  }
"#;

const COMBINE: &str = r#"
  const uint h = thread_position_in_grid.x;
  const uint t = thread_position_in_grid.y;
  const uint T = packed[1];
  const uint KT = packed[2];
  const uint MT = packed[3];
  if (h >= H || t >= T) return;
  const device uint* route_rows = packed + kHeader + MT + MT * kRows;
  const device bfloat* d = (const device bfloat*)dn;
  float acc = 0.0f;
  for (uint j = 0; j < KT; ++j) {
    const uint r = t * KT + j;
    acc += float(scores[r]) * float(d[ulong(route_rows[r]) * H + h]);
  }
  acc += float(shared_gate[t]) * float(d[ulong(route_rows[T * KT + t]) * H + h]);
  y[ulong(t) * H + h] = bfloat(acc);
"#;

/// Threadgroups of the expert passes; each loops over (tile, column block)
/// items, so the grid does not depend on the batch's tile count.
fn threadgroups() -> i32 {
    static N: OnceLock<i32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("IRONMLX_DIAGNOSTIC_MOE_TILE_THREADGROUPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2560)
    })
}

fn header() -> String {
    include_str!("moe_tile_experts.metal").to_string()
}

fn kernel(
    slot: &'static OnceLock<MetalKernel>,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    source: &str,
) -> Result<&'static MetalKernel> {
    if slot.get().is_none() {
        let built = MetalKernel::builder(name)
            .inputs(inputs)
            .outputs(outputs)
            .header(header())
            .source(source)
            .build()?;
        let _ = slot.set(built);
    }
    Ok(slot.get().expect("kernel built"))
}

fn check_quant(q: &Quant<'_>, rows: i32, k: i32, what: &str) -> Result<()> {
    let w = q.weight.shape();
    let s = q.scales.shape();
    let w = w.as_slice();
    let s = s.as_slice();
    ensure!(
        q.weight.dtype() == Dtype::Uint32
            && q.scales.dtype() == Dtype::Bfloat16
            && q.biases.dtype() == Dtype::Bfloat16
            && w.len() >= 2
            && w[w.len() - 2] == rows
            && w[w.len() - 1] == k / 8
            && s[s.len() - 1] == k / 64
            && q.biases.shape().as_slice() == s,
        "moe tile experts: {what} must be affine 4-bit group 64 [.., {rows}, {}], got {w:?} / {s:?}",
        k / 8
    );
    Ok(())
}

/// Routed + shared expert output `[T, H]` (bf16) for `x` `[T, H]` bf16,
/// `inds` `[T, k]` uint32 expert ids, `scores` `[T, k]` routing weights and
/// `shared_gate` `[T, 1]` (the shared expert's sigmoid gate).
pub(crate) fn forward_on(
    x: &Array,
    inds: &Array,
    scores: &Array,
    shared_gate: &Array,
    w: &Weights<'_>,
    target: StreamOrDevice,
) -> Result<Array> {
    let xs = x.shape();
    let &[t, h] = xs.as_slice() else {
        anyhow::bail!(
            "moe tile experts: x must be [T, H], got {:?}",
            xs.as_slice()
        );
    };
    let is = inds.shape();
    let &[ti, kt] = is.as_slice() else {
        anyhow::bail!(
            "moe tile experts: inds must be [T, k], got {:?}",
            is.as_slice()
        );
    };
    let gu = w.gate_up.weight.shape();
    let gu = gu.as_slice();
    ensure!(
        gu.len() == 3,
        "moe tile experts: gate/up must be [E, 2I, H/8], got {gu:?}"
    );
    let (e, i) = (gu[0], gu[1] / 2);
    ensure!(
        ti == t
            && (1..=MAX_TOKENS).contains(&t)
            && t * (kt + 1) <= 1024
            && x.dtype() == Dtype::Bfloat16
            && inds.dtype() == Dtype::Uint32
            && scores.shape().as_slice() == [t, kt]
            && shared_gate.size() == t as usize
            && h % 512 == 0
            && i % 512 == 0
            && i % (GATE_UP_SIMDGROUPS * COLUMNS_PER_SIMDGROUP) == 0
            && e < 1024,
        "moe tile experts: unsupported shapes x={:?} inds={:?} E={e} I={i}",
        xs.as_slice(),
        is.as_slice()
    );
    check_quant(&w.gate_up, 2 * i, h, "gate/up")?;
    check_quant(&w.down, h, i, "down")?;
    check_quant(&w.shared_gate, i, h, "shared gate")?;
    check_quant(&w.shared_up, i, h, "shared up")?;
    check_quant(&w.shared_down, h, i, "shared down")?;

    let routes = t * (kt + 1);
    let max_tiles = routes;
    let group_threads = (e + 1 + 31) / 32 * 32;
    let packed_len = 4 + max_tiles + max_tiles * ROWS + routes;
    let tile_rows = max_tiles * ROWS;

    static GROUP_K: OnceLock<MetalKernel> = OnceLock::new();
    static GATE_UP_K: OnceLock<MetalKernel> = OnceLock::new();
    static DOWN_K: OnceLock<MetalKernel> = OnceLock::new();
    static COMBINE_K: OnceLock<MetalKernel> = OnceLock::new();

    let packed = kernel(
        &GROUP_K,
        "ironmlx_moe_tile_group_v4",
        &["inds"],
        &["packed"],
        GROUP,
    )?
    .dispatch_builder()
    .inputs(&[inds])
    .output_shapes(&[Shape::from(&[packed_len][..])])
    .output_dtypes(&[Dtype::Uint32])
    .grid(group_threads, 1, 1)
    .threadgroup(group_threads, 1, 1)
    .template_int("E", e)
    .template_int("NT", group_threads)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    let threads = GATE_UP_SIMDGROUPS * 32;
    let x_half = x.astype_on(Dtype::Float16, target)?;
    let act = kernel(
        &GATE_UP_K,
        "ironmlx_moe_tile_gate_up_v4",
        &[
            "x", "packed", "gu_w", "gu_s", "gu_b", "sg_w", "sg_s", "sg_b", "su_w", "su_s", "su_b",
        ],
        &["act"],
        GATE_UP,
    )?
    .dispatch_builder()
    .inputs(&[
        &x_half,
        &packed,
        w.gate_up.weight,
        w.gate_up.scales,
        w.gate_up.biases,
        w.shared_gate.weight,
        w.shared_gate.scales,
        w.shared_gate.biases,
        w.shared_up.weight,
        w.shared_up.scales,
        w.shared_up.biases,
    ])
    .output_shapes(&[Shape::from(&[tile_rows, i][..])])
    .output_dtypes(&[Dtype::Float16])
    .grid(threadgroups() * threads, 1, 1)
    .threadgroup(threads, 1, 1)
    .template_int("E", e)
    .template_int("H", h)
    .template_int("I", i)
    .template_int("SG", GATE_UP_SIMDGROUPS)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    let dn = kernel(
        &DOWN_K,
        "ironmlx_moe_tile_down_v4",
        &["act", "packed", "d_w", "d_s", "d_b", "sd_w", "sd_s", "sd_b"],
        &["dn"],
        DOWN,
    )?
    .dispatch_builder()
    .inputs(&[
        &act,
        &packed,
        w.down.weight,
        w.down.scales,
        w.down.biases,
        w.shared_down.weight,
        w.shared_down.scales,
        w.shared_down.biases,
    ])
    .output_shapes(&[Shape::from(&[tile_rows, h][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(threadgroups() * threads, 1, 1)
    .threadgroup(threads, 1, 1)
    .template_int("E", e)
    .template_int("H", h)
    .template_int("I", i)
    .template_int("SG", GATE_UP_SIMDGROUPS)
    .stream(target)
    .dispatch()?
    .take_at(0)?;

    let y = kernel(
        &COMBINE_K,
        "ironmlx_moe_tile_combine_v4",
        &["dn", "packed", "scores", "shared_gate"],
        &["y"],
        COMBINE,
    )?
    .dispatch_builder()
    .inputs(&[&dn, &packed, scores, shared_gate])
    .output_shapes(&[Shape::from(&[t, h][..])])
    .output_dtypes(&[Dtype::Bfloat16])
    .grid(h, t, 1)
    .threadgroup(256, 1, 1)
    .template_int("H", h)
    .stream(target)
    .dispatch()?
    .take_at(0)?;
    Ok(y)
}
