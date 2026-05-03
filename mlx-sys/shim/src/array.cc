#include "cxx_mlx_shim/array.h"

#include <stdexcept>

#include "mlx/dtype.h"
#include "mlx/ops.h"

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
  }
  throw std::invalid_argument("cxx_mlx: unknown Dtype::Val value");
}

}  // namespace

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::zeros(s, dtype_from_u8(dtype)));
}

rust::Vec<int32_t> array_shape(const MlxArray& a) {
  rust::Vec<int32_t> out;
  for (auto v : a.shape()) {
    out.push_back(v);
  }
  return out;
}

}  // namespace cxx_mlx
