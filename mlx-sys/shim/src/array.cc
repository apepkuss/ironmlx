#include "cxx_mlx_shim/array.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

#include "cxx_mlx_shim/shim_helpers.h"

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
  // Use contiguous() to ensure a flat, stride-1 copy before reading raw data.
  // flatten() alone is insufficient for arrays with non-unit memory strides
  // (e.g. slice with stride > 1): it reshapes to 1-D but preserves the
  // underlying strides, causing ptr[i] to read the wrong elements.
  // contiguous() forces a materialized copy with unit element stride.
  // It is a no-op (zero extra copy) when the array is already contiguous.
  mlx::core::array flat = mlx::core::contiguous(mlx::core::flatten(a));
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

// === P1b1 binary element-wise ops (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> array_add(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::add(a, b, target));
}
std::unique_ptr<MlxArray> array_subtract(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::subtract(a, b, target));
}
std::unique_ptr<MlxArray> array_multiply(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::multiply(a, b, target));
}
std::unique_ptr<MlxArray> array_divide(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::divide(a, b, target));
}

// === P1b1 unary element-wise ops ===

std::unique_ptr<MlxArray> array_negative(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::negative(a, target));
}
std::unique_ptr<MlxArray> array_exp(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::exp(a, target));
}
std::unique_ptr<MlxArray> array_log(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::log(a, target));
}
std::unique_ptr<MlxArray> array_sqrt(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sqrt(a, target));
}
std::unique_ptr<MlxArray> array_tanh(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::tanh(a, target));
}
std::unique_ptr<MlxArray> array_sigmoid(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sigmoid(a, target));
}
std::unique_ptr<MlxArray> array_square(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::square(a, target));
}
std::unique_ptr<MlxArray> array_rsqrt(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::rsqrt(a, target));
}
std::unique_ptr<MlxArray> array_erf(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::erf(a, target));
}
std::unique_ptr<MlxArray> array_reciprocal(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::reciprocal(a, target));
}

// === P1b2a reductions (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> array_sum_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sum(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_sum_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sum(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_sum_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::sum(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_mean_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::mean(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_mean_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::mean(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_mean_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::mean(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_max_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::max(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_max_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::max(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_max_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::max(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_min_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::min(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_min_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::min(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_min_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::min(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_argmax_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argmax(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_argmax_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argmax(a, axis, keepdims, target));
}

// === P5.6 reduction completions (argmin / all / any / prod / logsumexp) ===

std::unique_ptr<MlxArray> array_argmin_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argmin(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_argmin_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argmin(a, axis, keepdims, target));
}

std::unique_ptr<MlxArray> array_all_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::all(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_all_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::all(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_all_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::all(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_any_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::any(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_any_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::any(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_any_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::any(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_prod_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::prod(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_prod_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::prod(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_prod_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::prod(a, axes_vec, keepdims, target));
}

std::unique_ptr<MlxArray> array_logsumexp_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::logsumexp(a, keepdims, target));
}
std::unique_ptr<MlxArray> array_logsumexp_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::logsumexp(a, axis, keepdims, target));
}
std::unique_ptr<MlxArray> array_logsumexp_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::logsumexp(a, axes_vec, keepdims, target));
}

// === P1b2a shape ops (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> array_reshape(
    const MlxArray& a, rust::Slice<const int32_t> shape,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::reshape(a, std::move(s), target));
}

std::unique_ptr<MlxArray> array_transpose(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::transpose(a, target));
}

std::unique_ptr<MlxArray> array_transpose_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> axes_vec(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::transpose(a, std::move(axes_vec), target));
}

std::unique_ptr<MlxArray> array_broadcast_to(
    const MlxArray& a, rust::Slice<const int32_t> shape,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::broadcast_to(a, s, target));
}

std::unique_ptr<MlxArray> array_concatenate(
    rust::Slice<const MlxArray* const> arrays, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);  // copy ctor — refcount-shared, cheap
  }
  return std::make_unique<MlxArray>(mlx::core::concatenate(std::move(vec), axis, target));
}

std::unique_ptr<MlxArray> array_stack(
    rust::Slice<const MlxArray* const> arrays, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<MlxArray> vec;
  vec.reserve(arrays.size());
  for (size_t i = 0; i < arrays.size(); ++i) {
    vec.push_back(*arrays[i]);
  }
  return std::make_unique<MlxArray>(mlx::core::stack(vec, axis, target));
}

std::unique_ptr<MlxArrayVec> array_split_n(
    const MlxArray& a, int32_t num_splits, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, num_splits, axis, target));
}

std::unique_ptr<MlxArrayVec> array_split_at(
    const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape idx(indices.begin(), indices.end());
  return std::make_unique<MlxArrayVec>(mlx::core::split(a, idx, axis, target));
}

size_t split_result_len(const MlxArrayVec& v) {
  return v.size();
}

std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i) {
  return std::make_unique<MlxArray>(v.at(i));  // copy ctor — refcount-shared
}

// === P1b2a matmul (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> array_matmul(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::matmul(a, b, target));
}

// === P1b2b dtype extension implementations ===

std::unique_ptr<MlxArray> array_from_u16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint16_t>(data.data(), shape, mlx::core::uint16);
}
std::unique_ptr<MlxArray> array_from_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint32_t>(data.data(), shape, mlx::core::uint32);
}
std::unique_ptr<MlxArray> array_from_u64(rust::Slice<const uint64_t> data, rust::Slice<const int32_t> shape) {
  return array_from_typed<uint64_t>(data.data(), shape, mlx::core::uint64);
}

uint16_t array_item_u16(const MlxArray& a) { return a.item<uint16_t>(); }
uint32_t array_item_u32(const MlxArray& a) { return a.item<uint32_t>(); }
uint64_t array_item_u64(const MlxArray& a) { return a.item<uint64_t>(); }

rust::Vec<uint16_t> array_to_vec_u16(const MlxArray& a) { return array_to_vec_typed<uint16_t>(a); }
rust::Vec<uint32_t> array_to_vec_u32(const MlxArray& a) { return array_to_vec_typed<uint32_t>(a); }
rust::Vec<uint64_t> array_to_vec_u64(const MlxArray& a) { return array_to_vec_typed<uint64_t>(a); }

// === P1b2b indexing implementations (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> array_where(
    const MlxArray& cond, const MlxArray& x, const MlxArray& y,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::where(cond, x, y, target));
}

std::unique_ptr<MlxArray> array_take(
    const MlxArray& a, const MlxArray& indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::take(a, indices, axis, target));
}

std::unique_ptr<MlxArray> array_take_along_axis(
    const MlxArray& a, const MlxArray& indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::take_along_axis(a, indices, axis, target));
}

std::unique_ptr<MlxArray> array_slice_strided(
    const MlxArray& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape s_start(start.begin(), start.end());
  mlx::core::Shape s_stop(stop.begin(), stop.end());
  mlx::core::Shape s_strides(strides.begin(), strides.end());
  return std::make_unique<MlxArray>(
      mlx::core::slice(a, std::move(s_start), std::move(s_stop), std::move(s_strides), target));
}

std::unique_ptr<MlxArray> array_gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<MlxArray> idx_vec;
  idx_vec.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    idx_vec.push_back(*indices[i]);  // copy ctor — refcount-shared, cheap
  }
  std::vector<int> axes_vec(axes.begin(), axes.end());
  mlx::core::Shape ss(slice_sizes.begin(), slice_sizes.end());
  return std::make_unique<MlxArray>(mlx::core::gather(a, idx_vec, axes_vec, ss, target));
}

// === P5 ops extensions (P5.7: + StreamOrDevice 4-arg encoding) ===

std::unique_ptr<MlxArray> tensordot_axis(
    const MlxArray& a, const MlxArray& b, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::tensordot(a, b, axis, target));
}

std::unique_ptr<MlxArray> tensordot_axes(
    const MlxArray& a, const MlxArray& b,
    rust::Slice<const int32_t> axes_a,
    rust::Slice<const int32_t> axes_b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  std::vector<int> va(axes_a.begin(), axes_a.end());
  std::vector<int> vb(axes_b.begin(), axes_b.end());
  return std::make_unique<MlxArray>(mlx::core::tensordot(a, b, va, vb, target));
}

std::unique_ptr<MlxArray> outer(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::outer(a, b, target));
}

std::unique_ptr<MlxArray> inner(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::inner(a, b, target));
}

std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::addmm(c, a, b, alpha, beta, target));
}

std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a, const MlxArray& b, int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::block_masked_mm(
      a, b, block_size,
      helpers::opt_arr(mask_out),
      helpers::opt_arr(mask_lhs),
      helpers::opt_arr(mask_rhs),
      target));
}

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a, const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::gather_mm(
      a, b,
      helpers::opt_arr(lhs_indices),
      helpers::opt_arr(rhs_indices),
      sorted_indices,
      target));
}

std::unique_ptr<MlxArray> segmented_mm(
    const MlxArray& a, const MlxArray& b, const MlxArray& segments,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::segmented_mm(a, b, segments, target));
}

// === P5.5 comparison + element-wise binary ===

std::unique_ptr<MlxArray> equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::equal(a, b, target));
}
std::unique_ptr<MlxArray> not_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::not_equal(a, b, target));
}
std::unique_ptr<MlxArray> less(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::less(a, b, target));
}
std::unique_ptr<MlxArray> less_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::less_equal(a, b, target));
}
std::unique_ptr<MlxArray> greater(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::greater(a, b, target));
}
std::unique_ptr<MlxArray> greater_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::greater_equal(a, b, target));
}
std::unique_ptr<MlxArray> maximum(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::maximum(a, b, target));
}
std::unique_ptr<MlxArray> minimum(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::minimum(a, b, target));
}

// === P5.5 clip ===

std::unique_ptr<MlxArray> clip(
    const MlxArray& a,
    const MlxArray* a_min,
    const MlxArray* a_max,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::clip(
      a, helpers::opt_arr(a_min), helpers::opt_arr(a_max), target));
}

// === P5.5 softmax ===

std::unique_ptr<MlxArray> softmax(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool precise,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  if (axes.empty()) {
    // Empty axes -> reduce over all axes (last axis behavior in MLX is the
    // single-axis form; the no-axes overload reduces over the full tensor
    // making each element 1/N. We want IntoAxes::All to mean "all axes",
    // which matches MLX's vector<int>{0,1,...,ndim-1} form).
    std::vector<int> all_axes;
    all_axes.reserve(a.ndim());
    for (int i = 0; i < static_cast<int>(a.ndim()); ++i) {
      all_axes.push_back(i);
    }
    return std::make_unique<MlxArray>(mlx::core::softmax(a, all_axes, precise, target));
  }
  std::vector<int> ax(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::softmax(a, ax, precise, target));
}

// === P5.5 sort family ===

std::unique_ptr<MlxArray> sort(
    const MlxArray& a, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sort(a, axis, target));
}

std::unique_ptr<MlxArray> argsort(
    const MlxArray& a, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argsort(a, axis, target));
}

std::unique_ptr<MlxArray> partition(
    const MlxArray& a, int32_t kth, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::partition(a, kth, axis, target));
}

std::unique_ptr<MlxArray> argpartition(
    const MlxArray& a, int32_t kth, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::argpartition(a, kth, axis, target));
}

std::unique_ptr<MlxArray> topk(
    const MlxArray& a, int32_t k, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::topk(a, k, axis, target));
}

// === P5.5 astype ===

std::unique_ptr<MlxArray> astype(
    const MlxArray& a, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::astype(a, t, target));
}

// === P5.5 array constructors ===

std::unique_ptr<MlxArray> arange(
    double start, double stop, double step, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::arange(start, stop, step, t, target));
}

std::unique_ptr<MlxArray> linspace(
    double start, double stop, int32_t num, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::linspace(start, stop, num, t, target));
}

std::unique_ptr<MlxArray> ones(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  mlx::core::Shape s(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::ones(s, t, target));
}

std::unique_ptr<MlxArray> ones_like(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::ones_like(a, target));
}

std::unique_ptr<MlxArray> zeros_like(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::zeros_like(a, target));
}

std::unique_ptr<MlxArray> full(
    rust::Slice<const int32_t> shape, const MlxArray& vals, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  mlx::core::Shape s(shape.begin(), shape.end());
  // mlx::core::full takes `array vals` by value (copy ctor — refcount-shared).
  return std::make_unique<MlxArray>(mlx::core::full(std::move(s), vals, t, target));
}

std::unique_ptr<MlxArray> full_like(
    const MlxArray& a, const MlxArray& vals,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::full_like(a, vals, target));
}

std::unique_ptr<MlxArray> eye(
    int32_t n, int32_t m, int32_t k, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::eye(n, m, k, t, target));
}

std::unique_ptr<MlxArray> identity(
    int32_t n, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::identity(n, t, target));
}

std::unique_ptr<MlxArray> tri(
    int32_t n, int32_t m, int32_t k, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto t = cxx_mlx::helpers::dtype_from_repr(dtype_repr);
  return std::make_unique<MlxArray>(mlx::core::tri(n, m, k, t, target));
}

std::unique_ptr<MlxArray> tril(
    const MlxArray& x, int32_t k,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  // mlx::core::tril takes `array x` by value (copy ctor — refcount-shared).
  return std::make_unique<MlxArray>(mlx::core::tril(x, k, target));
}

std::unique_ptr<MlxArray> triu(
    const MlxArray& x, int32_t k,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::triu(x, k, target));
}

// === P5.5 expand_dims / squeeze ===

std::unique_ptr<MlxArray> expand_dims(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  // expand_dims requires at least one axis. Empty input is illegal at this
  // boundary; we still forward it so MLX raises a uniform error message.
  std::vector<int> ax(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::expand_dims(a, ax, target));
}

std::unique_ptr<MlxArray> squeeze(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  if (axes.empty()) {
    // Empty axes -> "squeeze every size-1 dim" (no-axis overload). Matches
    // `IntoAxes::All` semantics in the safe layer.
    return std::make_unique<MlxArray>(mlx::core::squeeze(a, target));
  }
  std::vector<int> ax(axes.begin(), axes.end());
  return std::make_unique<MlxArray>(mlx::core::squeeze(a, ax, target));
}

// === P5.6 一元补完 ===

std::unique_ptr<MlxArray> abs(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::abs(a, target));
}
std::unique_ptr<MlxArray> sign(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sign(a, target));
}
std::unique_ptr<MlxArray> floor(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::floor(a, target));
}
std::unique_ptr<MlxArray> ceil(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::ceil(a, target));
}
std::unique_ptr<MlxArray> round(
    const MlxArray& a, int32_t decimals,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::round(a, decimals, target));
}
std::unique_ptr<MlxArray> sin(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::sin(a, target));
}
std::unique_ptr<MlxArray> cos(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::cos(a, target));
}
std::unique_ptr<MlxArray> tan(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::tan(a, target));
}
std::unique_ptr<MlxArray> expm1(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::expm1(a, target));
}

// === P5.6 数值卫生 + logical_not ===

std::unique_ptr<MlxArray> isnan(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::isnan(a, target));
}
std::unique_ptr<MlxArray> isinf(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::isinf(a, target));
}
std::unique_ptr<MlxArray> isfinite(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::isfinite(a, target));
}
std::unique_ptr<MlxArray> logical_not(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::logical_not(a, target));
}
std::unique_ptr<MlxArray> nan_to_num(
    const MlxArray& a, float nan,
    bool has_posinf, float posinf,
    bool has_neginf, float neginf,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  auto pos = has_posinf ? std::optional<float>(posinf) : std::nullopt;
  auto neg = has_neginf ? std::optional<float>(neginf) : std::nullopt;
  return std::make_unique<MlxArray>(mlx::core::nan_to_num(a, nan, pos, neg, target));
}

// === P5.6 二元补完 (power/logaddexp/remainder) ===

std::unique_ptr<MlxArray> power(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::power(a, b, target));
}
std::unique_ptr<MlxArray> logaddexp(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::logaddexp(a, b, target));
}
std::unique_ptr<MlxArray> remainder(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::remainder(a, b, target));
}

}  // namespace cxx_mlx
