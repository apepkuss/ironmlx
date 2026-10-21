// Diagnostic instantiation of Apple's installed MLX 0.32.3 D256 NAX kernel.
// Upstream implementation copyright Apple Inc.; MLX is MIT licensed.
// No installed MLX library or rival application is modified.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

// Two row groups of 16 query rows per threadgroup (BQ 32): the head-dim
// halves' score exchange synchronises four simdgroups instead of eight. Each
// row's arithmetic is that of the 64-row tile.
instantiate_kernel("ironmlx_probe_dsplit_bf16", attention_nax_dsplit,
                   bfloat, 32, 32, 256, 2, 2, bfloat, float)

// Split-K BF16 D256 NAX attention for short query blocks (decode and
// speculative verify) over long caches. Pass 1: one threadgroup per (key
// block, KV head, batch) runs the head-dim split NAX tile above over the
// GQA heads x query rows of that KV head (packed as contiguous rows of the
// row-major query) and writes unnormalized f32 partials with their row max
// and sum. Pass 2 combines the blocks of each row in block order. Key blocks
// are absolute (multiples of KB), each row masks keys past its own causal
// limit and skips blocks wholly past it, so a row's result does not depend
// on how many other rows share the call.
struct IronSplitKParams {
  int N;        // keys
  int R;        // query rows per head
  int rows;     // GQA factor * R
  int KB;       // keys per block (multiple of 32)
  int NB;       // key blocks
  int kv_heads;
  int causal;
  int parts;    // row parts per key block (one threadgroup each)
  int lo;       // row j sees keys >= lo + (j % R) (sliding window); very
                // negative when unbounded
  int tail_from; // tree: keys >= tail_from come from the row part's tail
  int TL;        // tree: keys per part tail
  float scale2; // scale * log2(e)
  int64_t Q_batch;
  int64_t K_strides[3];
  int64_t V_strides[3];
};

// TREE: a flat draft tree verified in one call. Query rows are node-major
// (row j = node slot j / R, R = GQA factor) and each row part (two node
// slots) lies on one root-to-leaf path. Keys before `tail_from` are the
// shared prompt; from `tail_from` (chunk aligned) every part reads its own
// tail: the prompt keys up to the tree base, then its path's nodes at the
// positions a linear verify of that path gives them. A node row's limit is
// LIM[slot], so each row computes exactly its linear-verify row.
template <int WM, int BD, bool TREE>
[[kernel, max_total_threads_per_threadgroup(WM * 2 * 32)]] void ironmlx_splitk_pass1(
    const device bfloat* Q [[buffer(0)]],
    const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]],
    device float* Opart [[buffer(3)]],
    device float* Mpart [[buffer(4)]],
    device float* Spart [[buffer(5)]],
    const constant IronSplitKParams& p [[buffer(6)]],
    const device bfloat* TailK [[buffer(7)]],
    const device bfloat* TailV [[buffer(8)]],
    const device int* LIM [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using T = bfloat;
  using A = float;
  constexpr short kU = 16;
  constexpr short WN = 2;
  constexpr short BK = 32;
  constexpr short TK = BK / kU;
  constexpr short TDh = BD / kU / WN;
  constexpr short BDh = BD / WN;

  // Threadgroup x: key block, then its row part (16 * WM rows each),
  // side by side so a block's parts read its keys while cached.
  const int blk = tid.x / p.parts, row_part = tid.x % p.parts, part_row0 = row_part * 16 * WM;
  const int h = tid.y, z = tid.z;
  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;
  const short tm = part_row0 + kU * row_group;
  const int rows = p.rows;
  const short lim_rows_q = short(rows - tm);

  Q += z * p.Q_batch + (long(h) * rows + tm) * BD + d_group * BDh;
  const int k0 = blk * p.KB;
  const int k1 = min(p.N, k0 + p.KB);
  const int ks = int(p.K_strides[2]);
  const int vs = int(p.V_strides[2]);
  K += z * p.K_strides[0] + h * p.K_strides[1] + long(k0) * ks + d_group * BDh;
  V += z * p.V_strides[0] + h * p.V_strides[1] + long(k0) * vs + d_group * BDh;
  const long tail_base = TREE ? ((long(z) * p.kv_heads + h) * p.parts + row_part) * p.TL * BD + d_group * BDh : 0;

  using otile_t = NAXTile<A, 1, TDh>;
  using stile_t = NAXTile<A, 1, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  constexpr short kRowsPT = otile_t::kRowsPerThread;
  constexpr auto neg_inf = Limits<A>::finite_min;

  otile_t Otile;
  Otile.clear();
  metal::vec<A, kRowsPT> max_score;
  metal::vec<A, kRowsPT> sum_score{0};
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = neg_inf;
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  int row_lim[kRowsPT];
  int row_lo[kRowsPT];
  for (short i = 0; i < kRowsPT; ++i) {
    const int j = tm + sm + i * stile_t::kFragRowsJump;
    if (TREE) {
      row_lim[i] = j < rows ? LIM[j / p.R] : 0;
      row_lo[i] = p.lo;
    } else {
      row_lim[i] = p.causal ? p.N - p.R + (j % p.R) + 1 : p.N;
      row_lo[i] = p.lo + (j % p.R);
    }
  }
  const int min_lim = TREE ? p.tail_from + 1 : (p.causal ? p.N - p.R + 1 : p.N);
  const int max_lo = TREE ? p.lo : p.lo + p.R - 1;

  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    Qtiles[id].load_rows(Q + id * kU, BD, lim_rows_q);
  }

  threadgroup A s_xchg[WM][WN][TK * kEPF * 32];
  const short base = short(simd_lane_id) * (TK * kEPF);

  for (int kc = k0; kc < k1; kc += BK) {
    const short lim_k = short(min(int(BK), k1 - kc));
    const bool full = lim_k == BK;
    const bool tail = TREE && kc >= p.tail_from;
    const device T* Kc = tail ? TailK + tail_base + long(kc - p.tail_from) * BD : K;
    const device T* Vc = tail ? TailV + tail_base + long(kc - p.tail_from) * BD : V;
    const int kst = tail ? BD : ks;
    const int vst = tail ? BD : vs;

    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int off = ik * kU * kst + id * kU;
        if (full) {
          Ktile.load(Kc + off, kst);
        } else {
          Ktile.load_rows(Kc + off, kst, short(lim_k - ik * kU));
        }
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }

    // Sum the two head-dim halves in a fixed order.
    threadgroup A* slot = s_xchg[row_group][d_group];
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + ik * kEPF + i] = s[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s[i] = s_xchg[row_group][0][base + ik * kEPF + i] +
            s_xchg[row_group][1][base + ik * kEPF + i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= p.scale2;
    }

    // Keys past a row's causal limit (and past the cache end) or before its
    // window start are masked.
    if (kc + BK > min_lim || kc < max_lo) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const int c = kc + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (c < row_lim[ii] && c >= row_lo[ii]) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    metal::vec<A, kRowsPT> new_max;
    metal::vec<A, kRowsPT> factor;
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int off = ik * kU * vst + id * kU;
        if (full) {
          Vtile.load(Vc + off, vst);
        } else {
          Vtile.load_rows(Vc + off, vst, short(lim_k - ik * kU));
        }
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    K += BK * ks;
    V += BK * vs;
  }

  const long part = (long(z) * p.kv_heads + h) * p.NB + blk;
  if (lim_rows_q > 0) {
    Otile.store_rows(Opart + (part * rows + tm) * BD + d_group * BDh, BD, lim_rows_q);
  }
  if (d_group == 0 && sn == 0) {
    for (short i = 0; i < kRowsPT; ++i) {
      const int j = tm + sm + i * stile_t::kFragRowsJump;
      if (j < rows) {
        Mpart[part * rows + j] = max_score[i];
        Spart[part * rows + j] = sum_score[i];
      }
    }
  }
}

template <int BD, bool TREE>
[[kernel]] void ironmlx_splitk_pass2(
    const device float* Opart [[buffer(0)]],
    const device float* Mpart [[buffer(1)]],
    const device float* Spart [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    const constant IronSplitKParams& p [[buffer(4)]],
    const device int* LIM [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
  const int j = tid.x, h = tid.y, z = tid.z;
  const int rows = p.rows;
  const int lim = TREE ? LIM[j / p.R] : (p.causal ? p.N - p.R + (j % p.R) + 1 : p.N);
  const int nb = (lim + p.KB - 1) / p.KB;
  // Blocks wholly before the row's window start hold no key of the row.
  const int b0 = TREE ? 0 : max(0, p.lo + (j % p.R)) / p.KB;
  const long part0 = (long(z) * p.kv_heads + h) * p.NB;
  float m = Limits<float>::finite_min;
  for (int b = b0; b < nb; b++) {
    m = max(m, Mpart[(part0 + b) * rows + j]);
  }
  float s = 0;
  float acc = 0;
  for (int b = b0; b < nb; b++) {
    const long row = (part0 + b) * rows + j;
    const float w = fast::exp2(Mpart[row] - m);
    s += Spart[row] * w;
    acc += Opart[row * BD + lid] * w;
  }
  O[((long(z) * p.kv_heads + h) * rows + j) * BD + lid] = static_cast<bfloat>(acc / s);
}

instantiate_kernel("ironmlx_splitk_d256_pass1_wm1", ironmlx_splitk_pass1, 1, 256, false)
instantiate_kernel("ironmlx_splitk_d256_pass2", ironmlx_splitk_pass2, 256, false)
instantiate_kernel("ironmlx_splitk_d256_tree_pass1_wm1", ironmlx_splitk_pass1, 1, 256, true)
instantiate_kernel("ironmlx_splitk_d256_tree_pass2", ironmlx_splitk_pass2, 256, true)
instantiate_kernel("ironmlx_splitk_d128_pass1_wm1", ironmlx_splitk_pass1, 1, 128, false)
instantiate_kernel("ironmlx_splitk_d128_pass2", ironmlx_splitk_pass2, 128, false)



