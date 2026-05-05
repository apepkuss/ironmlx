#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlx/array.h"
#include "mlx/dtype.h"

namespace cxx_mlx::helpers {

// pointer → optional<array>。array 拷贝廉价（refcount on array_desc_）。
inline std::optional<mlx::core::array> opt_arr(
    const mlx::core::array* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<int> opt_i(bool has, int32_t v) {
  return has ? std::optional<int>(v) : std::nullopt;
}

// Dtype::Val → Dtype。MLX 14 个 dtype 全覆盖；未知值抛 runtime_error。
inline mlx::core::Dtype dtype_from_repr(uint8_t v) {
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

inline std::optional<mlx::core::Dtype> opt_dtype(bool has, uint8_t v) {
  if (!has) return std::nullopt;
  return dtype_from_repr(v);
}

}  // namespace cxx_mlx::helpers
