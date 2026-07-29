#include "cxx_mlx_shim/quantization.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <chrono>
#include <stdexcept>

#include "mlx/transforms.h"

namespace cxx_mlx {

using helpers::decode_stream_or_device;
using helpers::dtype_from_repr;
using helpers::opt_arr;
using helpers::opt_dtype;
using helpers::opt_i;

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
  return std::make_unique<MlxArray>(mlx::core::quantized_matmul_batch_isolated(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      target));
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
  return std::make_unique<MlxArray>(mlx::core::quantized_matmul_product_stable(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      target));
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
