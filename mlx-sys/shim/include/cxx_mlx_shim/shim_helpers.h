#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

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

// === P5.7 StreamOrDevice decoding ===
//
// 4-arg encoding from Rust:
//   has_target=false                       -> std::monostate (use MLX default)
//   has_target=true,  is_device_only=true  -> Device only
//   has_target=true,  is_device_only=false -> Stream(idx, Device(device_type, 0))
//   ... with stream_index < 0              -> ThreadLocalStream(-idx - 1, Device)
//
// device_type: 0=cpu, 1=gpu (matches mlx::core::Device::DeviceType).

inline mlx::core::Device decode_device(uint8_t device_type) {
  switch (device_type) {
    case 0: return mlx::core::Device(mlx::core::Device::DeviceType::cpu, 0);
    case 1: return mlx::core::Device(mlx::core::Device::DeviceType::gpu, 0);
    default:
      throw std::runtime_error("decode_device: unknown DeviceType");
  }
}

inline mlx::core::StreamOrDevice decode_stream_or_device(
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index) {
  if (!has_target) {
    return std::monostate{};
  }
  auto dev = decode_device(device_type);
  if (is_device_only) {
    return dev;
  }
  if (stream_index < 0) {
    return mlx::core::ThreadLocalStream((-stream_index) - 1, dev);
  }
  return mlx::core::Stream(stream_index, dev);
}

}  // namespace cxx_mlx::helpers
