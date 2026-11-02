#include "cxx_mlx_shim/quantization.h"
#include "cxx_mlx_shim/prefill_qmm_mtile.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>

#include "mlx/transforms.h"

namespace cxx_mlx {

using helpers::decode_stream_or_device;
using helpers::dtype_from_repr;
using helpers::opt_arr;
using helpers::opt_dtype;
using helpers::opt_i;

namespace {
// Upstream MLX does not expose the fork's batch/product-isolation entry
// points. Keep their contract in the bridge using ordinary upstream QMM:
// each batch owns its product shape, and product-stable calls use one row.
// This is the correctness fallback; qualified native IronMLX kernels retain
// their own dispatch. No changes to the pinned upstream MLX are required.
mlx::core::array isolated_qmm(
    const mlx::core::array& x, const mlx::core::array& w,
    const mlx::core::array& scales, const std::optional<mlx::core::array>& biases,
    bool transpose, std::optional<int> group_size, std::optional<int> bits,
    const std::string& mode, bool product_stable, mlx::core::StreamOrDevice s) {
  namespace mx = mlx::core;
  auto ordinary = [&](const mx::array& input, const mx::array& weight,
                      const mx::array& scale, const std::optional<mx::array>& bias) {
    return mx::quantized_matmul(input, weight, scale, bias, transpose,
                               group_size, bits, mode, s);
  };
  if (x.ndim() < 2 || w.ndim() < 2 || scales.ndim() < 2 ||
      (biases && biases->ndim() < 2)) {
    return ordinary(x, w, scales, biases); // upstream operand validation
  }
  if (x.ndim() == 2 && w.ndim() == 2) {
    if (!product_stable || x.shape(0) <= 1) return ordinary(x, w, scales, biases);
    std::vector<mx::array> rows;
    rows.reserve(x.shape(0));
    for (int row = 0; row < x.shape(0); ++row) {
      rows.push_back(ordinary(mx::slice(x, {row, 0}, {row + 1, x.shape(1)}, s),
                              w, scales, biases));
    }
    return mx::concatenate(rows, 0, s);
  }
  const int rank = std::max(x.ndim(), w.ndim());
  mx::Shape batch(rank - 2, 1);
  for (const auto* a : {&x, &w}) {
    for (int axis = 0; axis < a->ndim() - 2; ++axis) {
      int& dim = batch[rank - a->ndim() + axis];
      const int incoming = a->shape(axis);
      if (dim != 1 && incoming != 1 && dim != incoming)
        throw std::invalid_argument("isolated QMM batch dimensions do not broadcast");
      dim = dim == 1 ? incoming : dim;
    }
  }
  int batches = 1;
  for (int dim : batch) batches *= dim;
  if (batches == 0) return ordinary(x, w, scales, biases);
  auto expand = [&](const mx::array& a) {
    auto shape = batch;
    shape.push_back(a.shape(-2));
    shape.push_back(a.shape(-1));
    return mx::reshape(mx::broadcast_to(a, shape, s),
                       {batches, a.shape(-2), a.shape(-1)}, s);
  };
  const auto bx = expand(x), bw = expand(w), bs = expand(scales);
  const auto bb = biases ? std::optional<mx::array>(expand(*biases)) : std::nullopt;
  auto row = [&](const mx::array& a, int i) {
    return mx::squeeze(mx::slice(a, {i, 0, 0}, {i + 1, a.shape(1), a.shape(2)}, s), 0, s);
  };
  std::vector<mx::array> outputs;
  outputs.reserve(batches);
  for (int i = 0; i < batches; ++i) {
    outputs.push_back(isolated_qmm(row(bx, i), row(bw, i), row(bs, i),
        bb ? std::optional<mx::array>(row(*bb, i)) : std::nullopt,
        transpose, group_size, bits, mode, product_stable, s));
  }
  auto shape = batch;
  shape.push_back(outputs.front().shape(-2));
  shape.push_back(outputs.front().shape(-1));
  return mx::reshape(mx::stack(outputs, s), shape, s);
}
} // namespace

// ===== QuantizeResult =====

QuantizeResult::QuantizeResult(std::vector<mlx::core::array> data)
    : arrays_(std::move(data)), taken_(arrays_.size(), false) {}

std::unique_ptr<MlxArray> QuantizeResult::take_at(size_t idx) {
  if (idx >= arrays_.size()) {
    throw std::runtime_error("QuantizeResult::take_at: idx out of range");
  }
  if (taken_[idx]) {
    throw std::runtime_error("QuantizeResult::take_at: already taken at idx");
  }
  taken_[idx] = true;
  return std::make_unique<MlxArray>(std::move(arrays_[idx]));
}

size_t quantize_result_count(const QuantizeResult& r) { return r.count(); }

std::unique_ptr<MlxArray> quantize_result_take_at(QuantizeResult& r, size_t idx) {
  return r.take_at(idx);
}

// ===== quantize =====

std::unique_ptr<QuantizeResult> quantize(
    const MlxArray& w,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  auto result = mlx::core::quantize(
      w,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale),
      target);
  return std::make_unique<QuantizeResult>(std::move(result));
}

// ===== dequantize =====

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale,
    bool has_dtype, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::dequantize(
      w, scales, opt_arr(biases),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale),
      opt_dtype(has_dtype, dtype_repr),
      target));
}

// ===== quantized_matmul =====

std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  if (auto candidate = experimental_prefill_qmm_mtile(
          x, w, scales, opt_arr(biases), transpose,
          opt_i(has_group_size, group_size), opt_i(has_bits, bits), std::string(mode), target)) {
    return std::make_unique<MlxArray>(std::move(*candidate));
  }
  return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      target));
}

std::unique_ptr<MlxArray> quantized_matmul_batch_isolated(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(isolated_qmm(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      false, target));
}

std::unique_ptr<MlxArray> quantized_matmul_product_stable(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(isolated_qmm(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      true, target));
}

std::unique_ptr<MlxArray> quantized_matmul_product_stable_affine8_wide(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      isolated_qmm(
          x, w, scales, opt_arr(biases),
          transpose,
          opt_i(has_group_size, group_size),
          opt_i(has_bits, bits),
          std::string(mode),
          true, target));
}

rust::Vec<double> quantized_matmul_bench_ms(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    size_t runs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  auto maybe_biases = opt_arr(biases);
  auto maybe_group_size = opt_i(has_group_size, group_size);
  auto maybe_bits = opt_i(has_bits, bits);
  auto mode_string = std::string(mode);

  rust::Vec<double> timings;
  timings.reserve(runs);
  for (size_t i = 0; i < runs; ++i) {
    auto started = std::chrono::steady_clock::now();
    auto y = mlx::core::quantized_matmul(
        x, w, scales, maybe_biases,
        transpose,
        maybe_group_size,
        maybe_bits,
        mode_string,
        target);
    mlx::core::eval(std::vector<mlx::core::array>{y});
    mlx::core::synchronize();
    auto elapsed = std::chrono::steady_clock::now() - started;
    timings.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
  }
  return timings;
}

// ===== qqmm =====

std::unique_ptr<MlxArray> qqmm(
    const MlxArray& x, const MlxArray& w,
    const MlxArray* w_scales,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale_x,
    const MlxArray* global_scale_w,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::qqmm(
      x, w, opt_arr(w_scales),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale_x),
      opt_arr(global_scale_w),
      target));
}

// ===== gather_qmm =====

std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    bool sorted_indices,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::gather_qmm(
      x, w, scales, opt_arr(biases),
      opt_arr(lhs_indices), opt_arr(rhs_indices),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      std::nullopt,
      sorted_indices,
      target));
}

std::unique_ptr<MlxArray> from_fp8(
    const MlxArray& x, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::from_fp8(x, dtype_from_repr(dtype_repr), target));
}

std::unique_ptr<MlxArray> to_fp8(
    const MlxArray& x,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::to_fp8(x, target));
}

}  // namespace cxx_mlx
