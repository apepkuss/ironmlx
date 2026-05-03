#include "cxx_mlx_shim/array.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

// Endpoint static_asserts on mlx::core::Dtype::Val. If MLX inserts a new
// dtype at any position, at least one endpoint shifts and we fail fast at
// the C++ build step before the Rust Dtype mirror has a chance to drift.
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::bool_) == 0,
              "Dtype::Val::bool_ ordinal changed; update Dtype enum in mlx/src/dtype.rs");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::float32) == 10,
              "Dtype::Val::float32 ordinal changed; update FLOAT32 in sys_smoke.rs and Dtype enum");
static_assert(static_cast<uint8_t>(mlx::core::Dtype::Val::complex64) == 13,
              "Dtype::Val::complex64 ordinal changed; update Dtype enum in mlx/src/dtype.rs");

// Half-type width guards. Rust's element.rs has matching static asserts for
// half::f16 / half::bf16 size_of == 2. The reinterpret_cast bridges (e.g.
// `reinterpret_cast<mlx::core::float16_t*>(uint16_t*)` in array_from_f16)
// silently misbehave if MLX adds padding or grows these types. Catch it here.
static_assert(sizeof(mlx::core::float16_t) == 2,
              "MLX float16_t no longer 2 bytes; reinterpret_cast bridges in this shim are unsound");
static_assert(sizeof(mlx::core::bfloat16_t) == 2,
              "MLX bfloat16_t no longer 2 bytes; reinterpret_cast bridges in this shim are unsound");

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

template <typename CppT, typename WireT = CppT>
rust::Vec<WireT> array_to_vec_typed(const MlxArray& a) {
  // Flatten to a 1-D contiguous array so that data<T>() iterates elements in
  // logical (row-major) order regardless of strides. This handles transpose,
  // broadcast_to, and any other view ops that leave the backing buffer
  // non-contiguous. The flatten+eval is cheap when already contiguous (MLX
  // detects that and avoids a copy).
  mlx::core::array flat = mlx::core::flatten(a);
  mlx::core::eval(flat);
  rust::Vec<WireT> out;
  out.reserve(flat.size());
  const CppT* ptr = flat.data<CppT>();
  for (size_t i = 0; i < flat.size(); ++i) {
    if constexpr (std::is_same_v<CppT, WireT>) {
      out.push_back(ptr[i]);
    } else {
      WireT bits;
      std::memcpy(&bits, &ptr[i], sizeof(bits));
      out.push_back(bits);
    }
  }
  return out;
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

rust::Vec<uint8_t> array_to_vec_bool(const MlxArray& a) {
  // mlx stores bool as 1-byte; reinterpret to uint8_t for the wire.
  return array_to_vec_typed<bool, uint8_t>(a);
}
rust::Vec<uint8_t> array_to_vec_u8(const MlxArray& a)   { return array_to_vec_typed<uint8_t>(a); }
rust::Vec<int8_t> array_to_vec_i8(const MlxArray& a)    { return array_to_vec_typed<int8_t>(a); }
rust::Vec<int16_t> array_to_vec_i16(const MlxArray& a)  { return array_to_vec_typed<int16_t>(a); }
rust::Vec<int32_t> array_to_vec_i32(const MlxArray& a)  { return array_to_vec_typed<int32_t>(a); }
rust::Vec<int64_t> array_to_vec_i64(const MlxArray& a)  { return array_to_vec_typed<int64_t>(a); }
rust::Vec<uint16_t> array_to_vec_f16(const MlxArray& a) {
  return array_to_vec_typed<mlx::core::float16_t, uint16_t>(a);
}
rust::Vec<uint16_t> array_to_vec_bf16(const MlxArray& a) {
  return array_to_vec_typed<mlx::core::bfloat16_t, uint16_t>(a);
}
rust::Vec<float> array_to_vec_f32(const MlxArray& a)    { return array_to_vec_typed<float>(a); }
rust::Vec<double> array_to_vec_f64(const MlxArray& a)   { return array_to_vec_typed<double>(a); }

// === P1b1 binary element-wise ops ===

std::unique_ptr<MlxArray> array_add(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::add(a, b));
}
std::unique_ptr<MlxArray> array_subtract(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::subtract(a, b));
}
std::unique_ptr<MlxArray> array_multiply(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::multiply(a, b));
}
std::unique_ptr<MlxArray> array_divide(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::divide(a, b));
}

// === P1b1 unary element-wise ops ===

std::unique_ptr<MlxArray> array_negative(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::negative(a));
}
std::unique_ptr<MlxArray> array_exp(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::exp(a));
}
std::unique_ptr<MlxArray> array_log(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::log(a));
}
std::unique_ptr<MlxArray> array_sqrt(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::sqrt(a));
}
std::unique_ptr<MlxArray> array_tanh(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::tanh(a));
}
std::unique_ptr<MlxArray> array_sigmoid(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::sigmoid(a));
}
std::unique_ptr<MlxArray> array_square(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::square(a));
}
std::unique_ptr<MlxArray> array_rsqrt(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::rsqrt(a));
}
std::unique_ptr<MlxArray> array_erf(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::erf(a));
}
std::unique_ptr<MlxArray> array_reciprocal(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::reciprocal(a));
}

// === P1b2a reductions ===

std::unique_ptr<MlxArray> array_sum_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::sum(a, keepdims));
}
std::unique_ptr<MlxArray> array_sum_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::sum(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_sum_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::sum(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_mean_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::mean(a, keepdims));
}
std::unique_ptr<MlxArray> array_mean_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::mean(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_mean_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::mean(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_max_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::max(a, keepdims));
}
std::unique_ptr<MlxArray> array_max_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::max(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_max_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::max(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_min_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::min(a, keepdims));
}
std::unique_ptr<MlxArray> array_min_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::min(a, axis, keepdims));
}
std::unique_ptr<MlxArray> array_min_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::min(a, axes_vec, keepdims));
}

std::unique_ptr<MlxArray> array_argmax_all(const MlxArray& a, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::argmax(a, keepdims));
}
std::unique_ptr<MlxArray> array_argmax_axis(const MlxArray& a, int32_t axis, bool keepdims) {
  return std::make_unique<MlxArray>(mlx::core::argmax(a, axis, keepdims));
}

// === P1b2a shape ops ===

std::unique_ptr<MlxArray> array_reshape(const MlxArray& a, rust::Slice<const int32_t> shape) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::reshape(a, std::move(s)));
}

std::unique_ptr<MlxArray> array_transpose(const MlxArray& a) {
  return std::make_unique<MlxArray>(mlx::core::transpose(a));
}

std::unique_ptr<MlxArray> array_transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes) {
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::transpose(a, std::move(axes_vec)));
}

std::unique_ptr<MlxArray> array_broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape) {
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::broadcast_to(a, s));
}

std::unique_ptr<MlxArray> array_concatenate(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);  // copy ctor — refcount-shared, cheap
  }
  return std::make_unique<MlxArray>(mlx::core::concatenate(std::move(vec), axis));
}

std::unique_ptr<MlxArray> array_stack(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);
  }
  return std::make_unique<MlxArray>(mlx::core::stack(vec, axis));
}

std::unique_ptr<MlxArrayVec> array_split_n(const MlxArray& a, int32_t num_splits, int32_t axis) {
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, num_splits, axis));
}

std::unique_ptr<MlxArrayVec> array_split_at(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis) {
  mlx::core::Shape idx(indices.begin(), indices.end());
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, idx, axis));
}

size_t split_result_len(const MlxArrayVec& v) {
  return v.size();
}

std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i) {
  return std::make_unique<MlxArray>(v.at(i));  // copy ctor — refcount-shared
}

// === P1b2a matmul ===

std::unique_ptr<MlxArray> array_matmul(const MlxArray& a, const MlxArray& b) {
  return std::make_unique<MlxArray>(mlx::core::matmul(a, b));
}

}  // namespace cxx_mlx
