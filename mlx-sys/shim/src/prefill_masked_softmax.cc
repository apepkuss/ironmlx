#include "cxx_mlx_shim/prefill_masked_softmax.h"
#include "cxx_mlx_shim/experiment_config.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

// Softmax bodies follow MLX 0.32.2 kernels/softmax.h (MIT, Copyright Apple
// Inc.): softmax_single_row / softmax_looped, T=bfloat, AccT=float, N_READS=4.
// The only change: a key past the query's causal limit loads bf16 lowest,
// exactly what where(mask, scores, finfo(bf16).min) would have stored.
// Each element is read and written by the same thread, so in-place is safe.
constexpr const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
constant constexpr int NR = 4;
constant constexpr int SIMD_SIZE = 32;

inline float masked_load(const device bfloat* in, int i, int visible) {
  return (i <= visible) ? float(in[i]) : float(as_type<bfloat>(ushort(0xFF7F)));
}

[[kernel]] void ironmlx_masked_softmax_single(
    const device bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],
    constant int& query_rows [[buffer(3)]],
    constant int& causal_offset [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;
  const int visible = causal_offset + int(gid % uint(query_rows));
  threadgroup float local_max[SIMD_SIZE];
  threadgroup float local_normalizer[SIMD_SIZE];
  float ld[NR];
  in += gid * size_t(axis_size);
  const int base = lid * NR;
  if (base + NR <= axis_size) {
    for (int i = 0; i < NR; i++) ld[i] = masked_load(in, base + i, visible);
  } else {
    for (int i = 0; i < NR; i++)
      ld[i] = (base + i < axis_size) ? masked_load(in, base + i, visible)
                                     : -metal::numeric_limits<float>::infinity();
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = -metal::numeric_limits<float>::max();
  for (int i = 0; i < NR; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) local_max[0] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < NR; i++) {
    float exp_x = fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) local_normalizer[simd_group_id] = normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) local_normalizer[0] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  out += gid * size_t(axis_size);
  if (base + NR <= axis_size) {
    for (int i = 0; i < NR; i++) out[base + i] = bfloat(ld[i] * normalizer);
  } else {
    for (int i = 0; i < NR; i++)
      if (base + i < axis_size) out[base + i] = bfloat(ld[i] * normalizer);
  }
}

[[kernel]] void ironmlx_masked_softmax_looped(
    const device bfloat* in [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant int& axis_size [[buffer(2)]],
    constant int& query_rows [[buffer(3)]],
    constant int& causal_offset [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  const int visible = causal_offset + int(gid % uint(query_rows));
  in += gid * size_t(axis_size);
  threadgroup float local_max[SIMD_SIZE];
  threadgroup float local_normalizer[SIMD_SIZE];
  float prevmax;
  float maxval = -metal::numeric_limits<float>::max();
  float normalizer = 0;
  const int rounds = (axis_size + NR * int(lsize) - 1) / (NR * int(lsize));
  for (int r = 0; r < rounds; r++) {
    int offset = r * lsize * NR + lid * NR;
    float vals[NR];
    if (offset + NR <= axis_size) {
      for (int i = 0; i < NR; i++) vals[i] = masked_load(in, offset + i, visible);
    } else {
      for (int i = 0; i < NR; i++)
        vals[i] = (offset + i < axis_size) ? masked_load(in, offset + i, visible)
                                           : -metal::numeric_limits<float>::infinity();
    }
    prevmax = maxval;
    for (int i = 0; i < NR; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < NR; i++) normalizer += fast::exp(vals[i] - maxval);
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  prevmax = maxval;
  if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) local_normalizer[simd_group_id] = normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);
  normalizer = 1 / normalizer;
  out += gid * size_t(axis_size);
  for (int r = 0; r < rounds; r++) {
    int offset = r * lsize * NR + lid * NR;
    if (offset + NR <= axis_size) {
      for (int i = 0; i < NR; i++)
        out[offset + i] = bfloat(fast::exp(masked_load(in, offset + i, visible) - maxval) * normalizer);
    } else {
      for (int i = 0; i < NR; i++)
        if (offset + i < axis_size)
          out[offset + i] = bfloat(fast::exp(masked_load(in, offset + i, visible) - maxval) * normalizer);
    }
  }
}
)METAL";

constexpr int kLoopedLimit = 4096;   // MLX SOFTMAX_LOOPED_LIMIT
constexpr int kLoopedThreads = 1024; // MLX looped pipeline width on M5

class ExperimentalMaskedCausalSoftmax final : public mx::UnaryPrimitive {
 public:
  // causal_offset: last visible key index of the block's first query row.
  ExperimentalMaskedCausalSoftmax(mx::Stream stream, int causal_offset)
      : UnaryPrimitive(stream), causal_offset_(causal_offset) {}
  const char* name() const override { return "ExperimentalMaskedCausalSoftmax"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Experimental masked softmax requires GPU");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& in = inputs[0];
    if (!in.flags().row_contiguous)
      throw std::runtime_error("Experimental masked softmax requires row-contiguous scores");
    if (in.is_donatable()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(mx::allocator::malloc(out.nbytes()));
    }
    const int axis = in.shape(-1);
    const int query_rows = in.shape(-2);
    const int rows = static_cast<int>(in.size() / axis);
    const bool looped = axis > kLoopedLimit;
    auto& device = mx::metal::device(stream().device);
    auto lib = device.get_library("ironmlx_experimental_masked_softmax_v2",
                                  [] { return std::string(kSource); });
    auto kernel = device.get_kernel(
        looped ? "ironmlx_masked_softmax_looped" : "ironmlx_masked_softmax_single", lib);
    size_t threads;
    if (looped) {
      if (kernel->maxTotalThreadsPerThreadgroup() < kLoopedThreads)
        throw std::runtime_error("Experimental masked softmax pipeline narrower than MLX's");
      threads = kLoopedThreads;
    } else {
      const size_t needed = (axis + 3) / 4;
      threads = 32 * ((needed + 31) / 32);
    }
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(axis, 2);
    encoder.set_bytes(query_rows, 3);
    encoder.set_bytes(causal_offset_, 4);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads, 1, 1));
  }

 private:
  int causal_offset_;
};

bool requested() {
  return prefill_masked_softmax_requested();
}
}  // namespace

mlx::core::array experimental_masked_causal_softmax(
    const mx::array& scores, int causal_offset, mx::StreamOrDevice target) {
  const auto stream = mx::to_stream(target);
  return mx::array(scores.shape(), scores.dtype(),
                   std::make_shared<ExperimentalMaskedCausalSoftmax>(stream, causal_offset),
                   {scores});
}

std::optional<mlx::core::array> experimental_masked_causal_sdpa(
    const mx::array& queries, const mx::array& keys, const mx::array& values,
    float scale, mx::StreamOrDevice s) {
  if (!requested()) return {};
  if (queries.ndim() != 4 || keys.ndim() != 4 || values.ndim() != 4 ||
      queries.dtype() != mx::bfloat16 || keys.dtype() != mx::bfloat16 ||
      values.dtype() != mx::bfloat16 || queries.shape(-2) <= 128 ||
      keys.shape(-2) < queries.shape(-2) || keys.shape(-3) != values.shape(-3) ||
      queries.shape(-3) % keys.shape(-3) != 0 || queries.shape(-1) != keys.shape(-1))
    return {};
  const auto stream = mx::to_stream(s);
  if (stream.device.type != mx::Device::gpu) return {};
  // Only where MLX itself would execute the op-level fallback.
  if (!mx::fast::ScaledDotProductAttention::use_fallback(
          queries, keys, values, /*has_mask=*/true, /*has_arr_mask=*/false,
          /*do_causal=*/true, /*is_training=*/false, /*output_logsumexp=*/false,
          /*force_fused=*/false, stream))
    return {};
  static std::once_flag message;
  std::call_once(message, [] {
    std::fputs("[experimental] masked causal softmax fallback active\n", stderr);
  });
  // MLX 0.32.2 fast.cpp fallback, op for op, minus the full-tensor where.
  const int n_q_heads = queries.shape(-3);
  const int n_kv_heads = keys.shape(-3);
  auto q = mx::multiply(mx::array(scale, queries.dtype()), queries, s);
  const int n_repeats = n_q_heads / n_kv_heads;
  auto k = keys;
  auto v = values;
  if (n_repeats > 1) {
    q = mx::unflatten(q, 1, {n_kv_heads, n_repeats}, s);
    k = mx::expand_dims(k, 2, s);
    v = mx::expand_dims(v, 2, s);
  }
  const int q_rows = queries.shape(-2);
  const int keys_len = keys.shape(-2);
  auto scores = mx::matmul(q, mx::swapaxes(k, -1, -2, s), s);
  scores = experimental_masked_causal_softmax(scores, keys_len - q_rows, s);
  auto out = mx::matmul(scores, v, s);
  if (n_repeats > 1) out = mx::flatten(out, 1, 2, s);
  return out;
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
std::optional<mlx::core::array> experimental_masked_causal_sdpa(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    float, mlx::core::StreamOrDevice) {
  if (prefill_masked_softmax_requested())
    throw std::runtime_error("Masked softmax experiment unavailable: Metal C++ headers missing");
  return {};
}
}  // namespace cxx_mlx
#endif
