/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Copyright © 2024 Apple Inc. Adapted from MLX sdpa_vector.h at 65383d68c.
// Keep logical token-to-partial assignment and BF16 intermediate rounding.
const int head = int(threadgroup_position_in_grid.x);
const int block = int(threadgroup_position_in_grid.z);
const int lane = int(thread_index_in_simdgroup);
constexpr int E = D / 32;
float q[E]; float o[E] = {0};
for (int j=0;j<E;j++) q[j] = float(scale[0]) * float(queries[head*D+lane*E+j]);
float maximum = -3.4028234663852886e38f;
float sum = 0;
for (int i=block;i<int(n_keys[0]);i+=BLOCKS) {
    int page = int(page_map[i / RATIO]);
    if (page < 0) continue;
    int pos = page * RATIO + i % RATIO;
    int offset = ((head/GQA)*int(packed_size[0])+pos)*D+lane*E;
    float score = 0;
    for (int j=0;j<E;j++) score += q[j] * float(keys[offset+j]);
    score = simd_sum(score);
    float next = max(maximum, score);
    float factor = fast::exp(maximum-next);
    float exp_score = fast::exp(score-next);
    maximum = next; sum = sum * factor + exp_score;
    for (int j=0;j<E;j++) o[j] = o[j] * factor + exp_score * float(values[offset+j]);
}
if (lane == 0) { sums[head*BLOCKS+block] = sum; maxs[head*BLOCKS+block] = maximum; }
for (int j=0;j<E;j++) partials[(head*BLOCKS+block)*D+lane*E+j] = T(o[j]);

// PASS TWO
uint3 tid = threadgroup_position_in_grid;
uint3 tpg = threadgroups_per_grid;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
const int blocks = int(n_blocks[0]);
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = -3.4028234663852886e38f;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
