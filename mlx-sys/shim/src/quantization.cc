#include "cxx_mlx_shim/quantization.h"

#include <stdexcept>

namespace cxx_mlx {

namespace {

// pointer → optional<array>。array 拷贝廉价（refcount on array_desc_）。
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<int> opt_i(bool has, int32_t v) {
  return has ? std::optional<int>(v) : std::nullopt;
}

inline std::optional<mlx::core::Dtype> opt_dtype(bool has, uint8_t v) {
  // Dtype 在 MLX 中是含 size 的 struct，但 Val 枚举值定义了所有 dtype。
  // 用 size_of(Dtype) 反推不直接，只能依赖默认 Dtype 构造器。所幸 dequantize
  // 的 dtype 参数 MLX 内部按枚举 dispatch，传 Dtype{Val, size} 即可。
  if (!has) return std::nullopt;
  // 重建对应 Val 的 Dtype，size 从 default 实例查（MLX dtype.h 定义了 inline constexpr
  // 实例如 mlx::core::float32 等）。这里走简化路径：手动 case 所有 Val 值。
  switch (static_cast<mlx::core::Dtype::Val>(v)) {
    case mlx::core::Dtype::Val::bool_:    return mlx::core::bool_;
    case mlx::core::Dtype::Val::uint8:    return mlx::core::uint8;
    case mlx::core::Dtype::Val::uint16:   return mlx::core::uint16;
    case mlx::core::Dtype::Val::uint32:   return mlx::core::uint32;
    case mlx::core::Dtype::Val::uint64:   return mlx::core::uint64;
    case mlx::core::Dtype::Val::int8:     return mlx::core::int8;
    case mlx::core::Dtype::Val::int16:    return mlx::core::int16;
    case mlx::core::Dtype::Val::int32:    return mlx::core::int32;
    case mlx::core::Dtype::Val::int64:    return mlx::core::int64;
    case mlx::core::Dtype::Val::float16:  return mlx::core::float16;
    case mlx::core::Dtype::Val::float32:  return mlx::core::float32;
    case mlx::core::Dtype::Val::float64:  return mlx::core::float64;
    case mlx::core::Dtype::Val::bfloat16: return mlx::core::bfloat16;
    case mlx::core::Dtype::Val::complex64:return mlx::core::complex64;
  }
  throw std::runtime_error("unknown Dtype::Val");
}

}  // namespace

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
    const MlxArray* global_scale) {
  auto result = mlx::core::quantize(
      w,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale));
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
    bool has_dtype, uint8_t dtype_repr) {
  return std::make_unique<MlxArray>(mlx::core::dequantize(
      w, scales, opt_arr(biases),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale),
      opt_dtype(has_dtype, dtype_repr)));
}

// ===== quantized_matmul =====

std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode) {
  return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
      x, w, scales, opt_arr(biases),
      transpose,
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode)));
}

// ===== qqmm =====

std::unique_ptr<MlxArray> qqmm(
    const MlxArray& x, const MlxArray& w,
    const MlxArray* w_scales,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale_x,
    const MlxArray* global_scale_w) {
  return std::make_unique<MlxArray>(mlx::core::qqmm(
      x, w, opt_arr(w_scales),
      opt_i(has_group_size, group_size),
      opt_i(has_bits, bits),
      std::string(mode),
      opt_arr(global_scale_x),
      opt_arr(global_scale_w)));
}

}  // namespace cxx_mlx
