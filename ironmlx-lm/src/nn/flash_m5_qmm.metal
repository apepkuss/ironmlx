// Adapted from TensorFold v0.3.6.2 lane_qmm.py. See LICENSE.tensorfold.

  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  // one op for all TMR 16-row blocks: each row gets the 16-row op's bits
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  // Native scales and biases remain in checkpoint row-major order.
  // (s, b) BF16 pairs in checkpoint row-major layout.
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      for (int j=0;j<4;j++) {
        const int col=n0+f*16+fn+j;
        s[f][j]=col<N ? float(S[col*KG+g]) : 0.0f;
        bb[f][j]=col<N ? float(B[col*KG+g]) : 0.0f;
      }
    }
    auto a = tA.slice(g * GS, 0);
    auto b = tB.slice(g*GS,n0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const bool live = !EDGE || rb + t * 16 < MP;     // EDGE: the last 32-row block passes MP, where XS ends
      float xs0 = 0.0f, xs1 = 0.0f;
      if (FXS) {
        // Group sums of x in-kernel, in the separate xsum kernel's order.
        const int r0 = rb + t * 16 + fm;
        if (r0 < M) {
          const device vec<bfloat, 8>* xp = (const device vec<bfloat, 8>*)((const device bfloat*)X + (int64_t)r0 * K + g * GS);
          for (int w = 0; w < GS/8; w++) { const vec<bfloat, 8> v = xp[w]; for (int i = 0; i < 8; i++) xs0 += float(v[i]); }
        }
        if (r0 + 8 < M) {
          const device vec<bfloat, 8>* xp = (const device vec<bfloat, 8>*)((const device bfloat*)X + (int64_t)(r0 + 8) * K + g * GS);
          for (int w = 0; w < GS/8; w++) { const vec<bfloat, 8> v = xp[w]; for (int i = 0; i < 8; i++) xs1 += float(v[i]); }
        }
      } else {
        xs0 = live ? 0.0f : 0.0f;
        xs1 = live ? 0.0f : 0.0f;
      }
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
