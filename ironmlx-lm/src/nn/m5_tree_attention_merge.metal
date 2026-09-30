// Adapted from TensorFold v0.3.6.2 stream_attention.py; B1 specialization. See LICENSE.tensorfold.

  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;               // node * G + g
  const int NCB = meta[3], W = meta[4], RPA = meta[5], CA = meta[6];
  const int st = nodes[2 * (int(r) / G) + 1];
  const int PT = meta[MG + st * MS + 0], NCBS = meta[MG + st * MS + 5];
  const int ra = meta[MG + st * MS + 3] * 16 + (int(r) / G - meta[MG + st * MS + 6]) * G + int(r) % G;
  const int CT = PT / CK;                                      // chunks the shared kernel finished
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {                               // committed chunks, in order
    const int64_t row = ((int64_t)hk * CA + c) * RPA + ra;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POA[row * D + lane * DP + i] * f2;
    m = nm;
  }
  for (int c = 0; c < NCBS; c++) {                              // then the window's chunks
    const int64_t row = (((int64_t)hk * NCB + c) * W + node) * 16 + g;
    const float mc = PMB[row];
    if (mc == -INFINITY) continue;
    const float lc = PLB[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POB[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * W + node) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);

