//! Long-context prefill recurrence of the gated delta rule.
//!
//! The same token-by-token recurrence as the prefill kernel in
//! `gated_delta_net` (S = g S; m = S k; S += k (v - m) beta; o = S q, with an
//! f32 state), laid out for long chunks: eight lanes share a state row and
//! each owns sixteen key columns, so a key-dimension dot product costs three
//! shuffle steps instead of a 32-lane reduction, and blocks of 16 tokens of
//! q, k, v, g and beta are staged in threadgroup memory once for all rows of
//! the threadgroup. Only the summation order of the dot products differs from
//! the prefill kernel, so callers use it from a fixed token position on.
//!
//! Inputs (any float type, read as f32): q, k `[B, T, Hk, Dk]`, v
//! `[B, T, Hv, Dv]`, g and beta `[B, T, Hv]`, state_in `[B, Hv, Dv, Dk]`;
//! T (int32 scalar).
//! Outputs: y `[B, T, Hv, Dv]` (bf16), state_out `[B, Hv, Dv, Dk]` (f32).

use std::sync::OnceLock;

use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::Result;

/// State rows per threadgroup.
const ROWS: i32 = 16;
/// Lanes per state row (each owns 128 / LANES key columns).
const LANES: i32 = 8;

const SOURCE: &str = r#"
  constexpr int C = Dk / LPR;            // key columns per lane
  constexpr int NTH = ROWS * LPR;        // threads per threadgroup
  threadgroup ST keys[BT * Dk];
  threadgroup ST queries[BT * Dk];
  threadgroup float values[BT * ROWS];
  threadgroup float gates[BT];
  threadgroup float betas[BT];

  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint n = threadgroup_position_in_grid.z;
  const uint b_idx = n / Hv;
  const uint hv = n % Hv;
  const uint hk = hv / (Hv / Hk);
  const uint row_base = threadgroup_position_in_grid.x * ROWS;
  const uint row = sg * (32 / LPR) + lane / LPR;
  const uint col = (lane % LPR) * C;
  const int T_ = T;

  const ulong sbase = ((ulong(n) * Dv) + row_base + row) * Dk + col;
  float s[C];
  for (int c = 0; c < C; c++) s[c] = float(state_in[sbase + c]);

  for (int t0 = 0; t0 < T_; t0 += BT) {
    const int nt = min(BT, T_ - t0);
    // Stage the block: q and k (BT x Dk), v (BT x ROWS), g and beta.
    for (uint e = tid * 8; e < uint(BT * Dk); e += NTH * 8) {
      const int tok = int(e) / Dk;
      const int dim = int(e) % Dk;
      if (tok < nt) {
        const ulong at = ((ulong(b_idx) * T_ + t0 + tok) * Hk + hk) * Dk + dim;
        for (int i = 0; i < 8; i++) {
          keys[e + i] = static_cast<ST>(k[at + i]);
          queries[e + i] = static_cast<ST>(q[at + i]);
        }
      }
    }
    for (uint e = tid; e < uint(BT * ROWS); e += NTH) {
      const int tok = int(e) / ROWS;
      const int r = int(e) % ROWS;
      if (tok < nt)
        values[e] = float(v[((ulong(b_idx) * T_ + t0 + tok) * Hv + hv) * Dv + row_base + r]);
    }
    if (tid < uint(BT) && int(tid) < nt) {
      const ulong at = (ulong(b_idx) * T_ + t0 + tid) * Hv + hv;
      gates[tid] = float(g[at]);
      betas[tid] = float(beta[at]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < nt; i++) {
      const float gi = gates[i];
      const threadgroup ST* kr = keys + i * Dk + col;
      const threadgroup ST* qr = queries + i * Dk + col;
      float m = 0.0f;
      for (int c = 0; c < C; c++) {
        s[c] = s[c] * gi;
        m = fma(s[c], float(kr[c]), m);
      }
      for (ushort x = 1; x < LPR; x <<= 1) m += simd_shuffle_xor(m, x);
      const float delta = (values[i * ROWS + row] - m) * betas[i];
      float o = 0.0f;
      for (int c = 0; c < C; c++) {
        s[c] = fma(float(kr[c]), delta, s[c]);
        o = fma(s[c], float(qr[c]), o);
      }
      for (ushort x = 1; x < LPR; x <<= 1) o += simd_shuffle_xor(o, x);
      if (lane % LPR == 0)
        y[((ulong(b_idx) * T_ + t0 + i) * Hv + hv) * Dv + row_base + row] = static_cast<bfloat>(o);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int c = 0; c < C; c += 4)
    *(device float4*)(state_out + sbase + c) = float4(s[c], s[c + 1], s[c + 2], s[c + 3]);
"#;

fn kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    if KERNEL.get().is_none() {
        let built = MetalKernel::builder("ironmlx_gdn_long_scan_v1")
            .inputs(&["q", "k", "v", "g", "beta", "state_in", "T"])
            .outputs(&["y", "state_out"])
            .source(SOURCE)
            .ensure_row_contiguous(true)
            .atomic_outputs(false)
            .build()?;
        let _ = KERNEL.set(built);
    }
    Ok(KERNEL.get().expect("kernel built"))
}

/// Run the long-context recurrence over one chunk; returns `(y, state_out)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_on(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    target: StreamOrDevice,
) -> Result<(Array, Array)> {
    let shape = v.shape();
    let (batch, seq) = (shape.as_slice()[0], shape.as_slice()[1]);
    anyhow::ensure!(
        head_k_dim == 128 && head_v_dim % ROWS == 0 && num_v_heads % num_k_heads == 0,
        "gdn long scan: unsupported head layout"
    );
    let t_arr: Array = (&[seq][..], ()).try_into()?;
    // bf16 q and k stage exactly as bf16, 32 tokens per block; other types
    // as f32, 16 tokens (threadgroup memory).
    let (block, stage) = if q.dtype() == Dtype::Bfloat16 && k.dtype() == Dtype::Bfloat16 {
        (32, Dtype::Bfloat16)
    } else {
        (16, Dtype::Float32)
    };
    let mut out = kernel()?
        .dispatch_builder()
        .inputs(&[q, k, v, g, beta, state_in, &t_arr])
        .output_shapes(&[
            Shape::from(vec![batch, seq, num_v_heads, head_v_dim]),
            Shape::from(vec![batch, num_v_heads, head_v_dim, head_k_dim]),
        ])
        .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
        .grid(head_v_dim / ROWS * (ROWS * LANES), 1, batch * num_v_heads)
        .threadgroup(ROWS * LANES, 1, 1)
        .template_int("Dk", head_k_dim)
        .template_int("Dv", head_v_dim)
        .template_int("Hk", num_k_heads)
        .template_int("Hv", num_v_heads)
        .template_int("ROWS", ROWS)
        .template_int("LPR", LANES)
        .template_int("BT", block)
        .template_dtype("ST", stage)
        .stream(target)
        .dispatch()?;
    let y = out.take_at(0)?;
    let state = out.take_at(0)?;
    Ok((y, state))
}
