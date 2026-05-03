#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;
using MlxArrayVec = std::vector<mlx::core::array>;

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype);
rust::Vec<int32_t> array_shape(const MlxArray& a);
size_t array_ndim(const MlxArray& a);
size_t array_size(const MlxArray& a);
uint8_t array_dtype(const MlxArray& a);
std::unique_ptr<MlxArray> array_clone(const MlxArray& a);
bool array_is_available(const MlxArray& a);

// from_slice family — one per Element dtype. Slice element type matches
// MLX dtype size; bool bridges through uint8_t (cxx limitation),
// f16/bf16 bridge through uint16_t with reinterpret_cast.

std::unique_ptr<MlxArray> array_from_bool(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u8(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i8(rust::Slice<const int8_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i16(rust::Slice<const int16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_bf16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_f64(rust::Slice<const double> data, rust::Slice<const int32_t> shape);

// item family — extract the single scalar value. Caller must ensure size()==1
// and dtype matches; the shim does eval implicitly (mlx::array::item triggers it).

bool array_item_bool(const MlxArray& a);
uint8_t array_item_u8(const MlxArray& a);
int8_t array_item_i8(const MlxArray& a);
int16_t array_item_i16(const MlxArray& a);
int32_t array_item_i32(const MlxArray& a);
int64_t array_item_i64(const MlxArray& a);
uint16_t array_item_f16(const MlxArray& a);   // raw bits of half::f16
uint16_t array_item_bf16(const MlxArray& a);  // raw bits of half::bf16
float array_item_f32(const MlxArray& a);
double array_item_f64(const MlxArray& a);

// to_vec family — copy all elements out as a rust::Vec. Triggers eval.

rust::Vec<uint8_t> array_to_vec_bool(const MlxArray& a);   // 1 byte per bool
rust::Vec<uint8_t> array_to_vec_u8(const MlxArray& a);
rust::Vec<int8_t> array_to_vec_i8(const MlxArray& a);
rust::Vec<int16_t> array_to_vec_i16(const MlxArray& a);
rust::Vec<int32_t> array_to_vec_i32(const MlxArray& a);
rust::Vec<int64_t> array_to_vec_i64(const MlxArray& a);
rust::Vec<uint16_t> array_to_vec_f16(const MlxArray& a);   // raw bits of half::f16
rust::Vec<uint16_t> array_to_vec_bf16(const MlxArray& a);  // raw bits of half::bf16
rust::Vec<float> array_to_vec_f32(const MlxArray& a);
rust::Vec<double> array_to_vec_f64(const MlxArray& a);

// Binary element-wise ops (broadcasting handled by MLX after Rust-side
// shape validation in mlx::broadcast::broadcast_shape).
std::unique_ptr<MlxArray> array_add(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_subtract(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_multiply(const MlxArray& a, const MlxArray& b);
std::unique_ptr<MlxArray> array_divide(const MlxArray& a, const MlxArray& b);

// Unary element-wise ops.
std::unique_ptr<MlxArray> array_negative(const MlxArray& a);
std::unique_ptr<MlxArray> array_exp(const MlxArray& a);
std::unique_ptr<MlxArray> array_log(const MlxArray& a);
std::unique_ptr<MlxArray> array_sqrt(const MlxArray& a);
std::unique_ptr<MlxArray> array_tanh(const MlxArray& a);
std::unique_ptr<MlxArray> array_sigmoid(const MlxArray& a);
std::unique_ptr<MlxArray> array_square(const MlxArray& a);
std::unique_ptr<MlxArray> array_rsqrt(const MlxArray& a);
std::unique_ptr<MlxArray> array_erf(const MlxArray& a);
std::unique_ptr<MlxArray> array_reciprocal(const MlxArray& a);

// === P1b2a reductions (5 ops × 3 forms = 15) ===

std::unique_ptr<MlxArray> array_sum_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_sum_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_mean_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_mean_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_mean_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_max_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_max_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_max_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

std::unique_ptr<MlxArray> array_min_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_min_axis(const MlxArray& a, int32_t axis, bool keepdims);
std::unique_ptr<MlxArray> array_min_axes(const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims);

// argmax: only single-axis variant in MLX. We expose array_argmax_all via
// flatten-then-argmax for symmetry.
std::unique_ptr<MlxArray> array_argmax_all(const MlxArray& a, bool keepdims);
std::unique_ptr<MlxArray> array_argmax_axis(const MlxArray& a, int32_t axis, bool keepdims);

// === P1b2a shape ops ===

std::unique_ptr<MlxArray> array_reshape(const MlxArray& a, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_transpose(const MlxArray& a);
std::unique_ptr<MlxArray> array_transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes);
std::unique_ptr<MlxArray> array_broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape);

// Concatenate/stack accept raw pointer slices because cxx 1.0 doesn't bridge
// &[&MlxArray] directly. Caller (Rust safe layer) builds the pointer slice.
std::unique_ptr<MlxArray> array_concatenate(rust::Slice<const MlxArray* const> arrays, int32_t axis);
std::unique_ptr<MlxArray> array_stack(rust::Slice<const MlxArray* const> arrays, int32_t axis);

// Split returns std::vector<array> wrapped in MlxArrayVec opaque holder.
std::unique_ptr<MlxArrayVec> array_split_n(const MlxArray& a, int32_t num_splits, int32_t axis);
std::unique_ptr<MlxArrayVec> array_split_at(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis);

// MlxArrayVec accessors.
size_t split_result_len(const MlxArrayVec& v);
std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i);

// === P1b2a matmul ===

std::unique_ptr<MlxArray> array_matmul(const MlxArray& a, const MlxArray& b);

}  // namespace cxx_mlx
