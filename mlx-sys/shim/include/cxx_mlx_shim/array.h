#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

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

}  // namespace cxx_mlx
