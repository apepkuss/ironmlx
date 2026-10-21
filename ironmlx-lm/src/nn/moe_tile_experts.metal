// Expert-tile MoE for narrow (decode / verify) batches. Every route of the
// batch (top-k routed experts per token, then the shared expert as expert E
// for every token) is grouped by expert into tiles of at most 8 rows; each
// tile streams its expert's 4-bit weights once through 8x8 simdgroup MMAs.
//
// Packed grouping buffer (uint32):
//   [0] tile count, [1] T, [2] KT, [3] MT | tile experts [MT] | grouped routes
//   [MT * 8] (padding = kNone) | route rows [R]
// A route is r < T * KT for token r / KT's slot r % KT, and T * KT + t for
// token t's shared expert. R = MT = T * (KT + 1).

constant constexpr uint kRows = 8;
constant constexpr uint kGroup = 64;
constant constexpr uint kNone = 0xffffffffu;
constant constexpr uint kHeader = 4;
constant constexpr uint kMaxRoutes = 1024;

// Lane coordinates in an 8x8 simdgroup fragment.
inline ushort frag_m(uint lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
inline ushort frag_n(uint lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

// One simdgroup's 8-column block of an 8-row tile (xr0 / xr1: the input rows
// fn and fn + 1 of the lane, half): Y^T (8 columns x 8 rows) = W (8 columns
// x K) X^T in half 8x8x8 MMAs with fp32 accumulation, over spans of
// SPAN = 128 * NB inputs. In a span a lane loads NB * 16 consecutive code bytes
// of its weight row, then runs NB batches of 16 steps; the fragment's logical
// k = kappa at batch b, step j maps to span input
// (kappa / 2) * (SPAN / 4) + 32 b + 2 j + kappa % 2. A code byte's two nibbles
// become the half2 (1024 + lo, 1024 + hi) by OR-ing them into 0x6400, minus
// 1024 (exact), then scale * q + bias in half2.
template <bool GateUp, ushort NB>
inline void moe_tile_block(
    const device half* xr0, const device half* xr1, uint K,
    const device uchar* w0, const device bfloat* s0, const device bfloat* b0,
    const device uchar* w1, const device bfloat* s1, const device bfloat* b1,
    uint lane, thread simdgroup_float8x8& c0, thread simdgroup_float8x8& c1) {
  constexpr uint SPAN = 128 * NB;
  const ushort fm = frag_m(lane);
  const ushort fn = frag_n(lane);
  const uint groups = K / kGroup;
  c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const device uchar* row0 = w0 + ulong(fm) * (K / 2) + (fn / 2) * (SPAN / 8);
  const device uchar* row1 = w1 + ulong(fm) * (K / 2) + (fn / 2) * (SPAN / 8);
  const uint x_lane = (fm / 2) * (SPAN / 4) + (fm % 2);
  const uint g_lane = (fn / 2) * (SPAN / 4);
  auto run = [&](thread simdgroup_float8x8& c, uint4 q, half sc, half bi,
                 thread simdgroup_half8x8 (&bm)[16]) __attribute__((always_inline)) {
    const half2 s2 = half2(sc);
    const half2 b2 = half2(bi);
    for (ushort j = 0; j < 16; ++j) {
      const uint word = q[j / 4] >> (8 * (j % 4));
      const uint pair = (word & 0xfu) | ((word & 0xf0u) << 12) | 0x64006400u;
      simdgroup_half8x8 am;
      reinterpret_cast<thread half2&>(am.thread_elements()) =
          fma(s2, as_type<half2>(pair) - half2(1024.0h), b2);
      simdgroup_multiply_accumulate(c, am, bm[j], c);
    }
  };
  for (uint k0 = 0; k0 < K; k0 += SPAN) {
    uint4 q0[NB];
    uint4 q1[NB];
    for (ushort v = 0; v < NB; ++v) q0[v] = ((const device uint4*)(row0 + k0 / 2))[v];
    if (GateUp)
      for (ushort v = 0; v < NB; ++v) q1[v] = ((const device uint4*)(row1 + k0 / 2))[v];
    for (ushort bt = 0; bt < NB; ++bt) {
      simdgroup_half8x8 bm[16];
      {
        const uint off = k0 + x_lane + bt * 32;
        const device half* x0 = xr0 + off;
        const device half* x1 = xr1 + off;
        for (ushort j = 0; j < 16; ++j) {
          reinterpret_cast<thread half2&>(bm[j].thread_elements()) = half2(x0[2 * j], x1[2 * j]);
        }
      }
      const uint g = (k0 + g_lane + bt * 32) / kGroup;
      run(c0, q0[bt], half(s0[ulong(fm) * groups + g]), half(b0[ulong(fm) * groups + g]), bm);
      if (GateUp)
        run(c1, q1[bt], half(s1[ulong(fm) * groups + g]), half(b1[ulong(fm) * groups + g]), bm);
    }
  }
}
