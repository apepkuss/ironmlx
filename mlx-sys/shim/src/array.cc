#include "cxx_mlx_shim/array.h"

#include <stdexcept>
#include <string>

#include "mlx/dtype.h"
#include "mlx/ops.h"

// Endpoint static_asserts on mlx::core::Dtype::Val. If MLX inserts a new
// dtype at any position, at least one endpoint shifts and we fail fast at
// the C++ build step before the Rust Dtype mirror has a chance to drift.
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::bool_) == 0,
              "Dtype::Val::bool_ ordinal changed; update Dtype enum in mlx/src/dtype.rs");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::float32) == 10,
              "Dtype::Val::float32 ordinal changed; update FLOAT32 in sys_smoke.rs and Dtype enum");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::complex64) == 13,
              "Dtype::Val::complex64 ordinal changed; update Dtype enum in mlx/src/dtype.rs");

namespace cxx_mlx {

namespace {

mlx::core::Dtype dtype_from_u8(uint8_t v) {
  using V = mlx::core::Dtype::Val;
  switch (static_cast<V>(v)) {
    case V::bool_: return mlx::core::bool_;
    case V::uint8: return mlx::core::uint8;
    case V::uint16: return mlx::core::uint16;
    case V::uint32: return mlx::core::uint32;
    case V::uint64: return mlx::core::uint64;
    case V::int8: return mlx::core::int8;
    case V::int16: return mlx::core::int16;
    case V::int32: return mlx::core::int32;
    case V::int64: return mlx::core::int64;
    case V::float16: return mlx::core::float16;
    case V::float32: return mlx::core::float32;
    case V::float64: return mlx::core::float64;
    case V::bfloat16: return mlx::core::bfloat16;
    case V::complex64: return mlx::core::complex64;
    default:
      throw std::invalid_argument(
          "cxx_mlx: unknown Dtype::Val value: " + std::to_string(static_cast<int>(v)));
  }
}

}  // namespace

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::zeros(s, dtype_from_u8(dtype)));
}

rust::Vec<int32_t> array_shape(const MlxArray& a) {
  rust::Vec<int32_t> out;
  out.reserve(a.ndim());
  for (auto v : a.shape()) {
    out.push_back(v);
  }
  return out;
}

size_t array_ndim(const MlxArray& a) {
  return a.ndim();
}

size_t array_size(const MlxArray& a) {
  return a.size();
}

uint8_t array_dtype(const MlxArray& a) {
  return static_cast<uint8_t>(a.dtype().val());
}

}  // namespace cxx_mlx
