#include "cxx_mlx_shim/array.h"

#include <cstring>
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

template <typename CppT>
std::unique_ptr<MlxArray> array_from_typed(
    const CppT* data,
    rust::Slice<const int32_t> shape,
    mlx::core::Dtype dtype) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::array(data, std::move(s), dtype));
}

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

std::unique_ptr<MlxArray> array_clone(const MlxArray& a) {
  // mlx::core::array's copy constructor shares the internal shared_ptr<ArrayDesc>;
  // this is cheap (atomic refcount++) and does not copy tensor data.
  return std::make_unique<MlxArray>(a);
}

bool array_is_available(const MlxArray& a) {
  // NB: mlx::core::array::is_available() is a const method that internally
  // mutates state via shared_ptr<ArrayDesc> (calls detach_event() and
  // set_status() on the available transition). This is safe under our
  // !Sync contract — only single-thread access to a given Array is allowed.
  return a.is_available();
}

std::unique_ptr<MlxArray> array_from_bool(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape) {
  // mlx stores bool as 1 byte; reinterpret uint8_t bridge to bool elements.
  return array_from_typed<bool>(reinterpret_cast<const bool*>(data.data()), shape, mlx::core::bool_);
}
std::unique_ptr<MlxArray> array_from_u8(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint8_t>(data.data(), shape, mlx::core::uint8);
}
std::unique_ptr<MlxArray> array_from_i8(rust::Slice<const int8_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int8_t>(data.data(), shape, mlx::core::int8);
}
std::unique_ptr<MlxArray> array_from_i16(rust::Slice<const int16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int16_t>(data.data(), shape, mlx::core::int16);
}
std::unique_ptr<MlxArray> array_from_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int32_t>(data.data(), shape, mlx::core::int32);
}
std::unique_ptr<MlxArray> array_from_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<int64_t>(data.data(), shape, mlx::core::int64);
}
std::unique_ptr<MlxArray> array_from_f16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  // half::f16 has the same memory layout as mlx::core::float16_t (both 2-byte POD, IEEE 754 binary16).
  return array_from_typed<mlx::core::float16_t>(
      reinterpret_cast<const mlx::core::float16_t*>(data.data()),
      shape, mlx::core::float16);
}
std::unique_ptr<MlxArray> array_from_bf16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<mlx::core::bfloat16_t>(
      reinterpret_cast<const mlx::core::bfloat16_t*>(data.data()),
      shape, mlx::core::bfloat16);
}
std::unique_ptr<MlxArray> array_from_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<float>(data.data(), shape, mlx::core::float32);
}
std::unique_ptr<MlxArray> array_from_f64(rust::Slice<const double> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<double>(data.data(), shape, mlx::core::float64);
}

bool array_item_bool(const MlxArray& a) { return a.item<bool>(); }
uint8_t array_item_u8(const MlxArray& a) { return a.item<uint8_t>(); }
int8_t array_item_i8(const MlxArray& a) { return a.item<int8_t>(); }
int16_t array_item_i16(const MlxArray& a) { return a.item<int16_t>(); }
int32_t array_item_i32(const MlxArray& a) { return a.item<int32_t>(); }
int64_t array_item_i64(const MlxArray& a) { return a.item<int64_t>(); }
uint16_t array_item_f16(const MlxArray& a) {
  // Read out as mlx::core::float16_t and reinterpret to raw uint16_t.
  auto v = a.item<mlx::core::float16_t>();
  uint16_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
uint16_t array_item_bf16(const MlxArray& a) {
  auto v = a.item<mlx::core::bfloat16_t>();
  uint16_t out;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
float array_item_f32(const MlxArray& a) { return a.item<float>(); }
double array_item_f64(const MlxArray& a) { return a.item<double>(); }

}  // namespace cxx_mlx
