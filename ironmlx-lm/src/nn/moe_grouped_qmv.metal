// Expert-grouped gather qmv. One threadgroup owns one run of rows that share
// an expert (rows are sorted by expert) and eight output features. Each block
// of an expert's packed weights is unpacked once into exact integer terms and
// reused by every row of the run; each row then evaluates the MLX qdot
// expression over those terms in the original order, so a row's value does not
// depend on how many rows are grouped.
typedef float U;
constexpr int packs_per_thread = 2;
constexpr int results_per_simdgroup = 4;
constexpr int row_chunk = 4;
constexpr int pack_factor = grouped_pack_factor<BITS>();
constexpr int bytes_per_pack = grouped_bytes_per_pack<BITS>();
constexpr int values_per_thread = pack_factor * packs_per_thread;
constexpr int block_size = values_per_thread * 32;
constexpr int scale_step_per_thread = GS / values_per_thread;
constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
constexpr int in_vec_size_g = K / GS;
constexpr int terms = grouped_term_count<BITS, values_per_thread>();

const uint rows = experts_shape[0];
const uint start = threadgroup_position_in_grid.x;
const uint expert = experts[start];
if (start > 0 && experts[start - 1] == expert) {
  return;
}
uint run = 1;
while (run < MAXRUN && start + run < rows && experts[start + run] == expert) {
  run++;
}

const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const int out_row = threadgroup_position_in_grid.y * (2 * results_per_simdgroup) +
    simd_gid * results_per_simdgroup;

const device uint8_t* w_base = (const device uint8_t*)w +
    (size_t)expert * N * in_vec_size_w + out_row * in_vec_size_w +
    simd_lid * packs_per_thread * bytes_per_pack;
const device T* s_base = scales + (size_t)expert * N * in_vec_size_g +
    out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
const device T* b_base = biases + (size_t)expert * N * in_vec_size_g +
    out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

thread U x_thread[values_per_thread];
thread ushort unpacked[results_per_simdgroup][terms];

for (uint chunk = 0; chunk < run; chunk += row_chunk) {
  const uint chunk_rows = min((uint)row_chunk, run - chunk);
  const device T* xs = x + (size_t)(start + chunk) * K + simd_lid * values_per_thread;
  const device uint8_t* ws = w_base;
  const device T* sc = s_base;
  const device T* bi = b_base;
  thread U result[row_chunk][results_per_simdgroup];
  for (int r = 0; r < row_chunk; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  for (int k = 0; k < K; k += block_size) {
    U s[results_per_simdgroup];
    U b[results_per_simdgroup];
    for (int row = 0; row < results_per_simdgroup; row++) {
      grouped_unpack<BITS, values_per_thread>(ws + row * in_vec_size_w, unpacked[row]);
      s[row] = sc[row * in_vec_size_g];
      b[row] = bi[row * in_vec_size_g];
    }
    for (uint r = 0; r < chunk_rows; r++) {
      U sum = grouped_load_vector<T, U, values_per_thread, BITS>(
          xs + (size_t)r * K + k, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[r][row] += grouped_qdot_unpacked<U, values_per_thread, BITS>(
            unpacked[row], x_thread, s[row], b[row], sum);
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / GS;
    bi += block_size / GS;
  }

  for (uint r = 0; r < chunk_rows; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U value = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        y[(size_t)(start + chunk + r) * N + out_row + row] = static_cast<T>(value);
      }
    }
  }
}
