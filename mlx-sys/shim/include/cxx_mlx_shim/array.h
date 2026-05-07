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
// Stream-targeted variant of `array_zeros`. The 4 trailing params encode
// `mlx::core::StreamOrDevice` per `helpers::decode_stream_or_device` (P5.7).
std::unique_ptr<MlxArray> array_zeros_on(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
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
//
// 4 trailing stream params encode `mlx::core::StreamOrDevice` per
// `helpers::decode_stream_or_device` (P5.7).
std::unique_ptr<MlxArray> array_add(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_subtract(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_multiply(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_divide(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// Unary element-wise ops.
std::unique_ptr<MlxArray> array_negative(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_exp(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_log(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_sqrt(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_tanh(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_sigmoid(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_square(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_rsqrt(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_erf(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_reciprocal(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P1b2a reductions (5 ops × 3 forms = 15) ===
// 4 trailing stream params encode StreamOrDevice (P5.7).

std::unique_ptr<MlxArray> array_sum_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_sum_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_sum_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_mean_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_mean_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_mean_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_max_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_max_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_max_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_min_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_min_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_min_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// argmax: only single-axis variant in MLX. We expose array_argmax_all via
// flatten-then-argmax for symmetry.
std::unique_ptr<MlxArray> array_argmax_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_argmax_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 reduction completions (argmin / all / any / prod / logsumexp) ===
// argmin: 2 forms (no multi-axis); others: 3 forms each.

std::unique_ptr<MlxArray> array_argmin_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_argmin_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_all_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_all_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_all_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_any_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_any_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_any_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_prod_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_prod_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_prod_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_logsumexp_all(
    const MlxArray& a, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_logsumexp_axis(
    const MlxArray& a, int32_t axis, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_logsumexp_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool keepdims,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P1b2a shape ops (P5.7: + 4 trailing stream params) ===

std::unique_ptr<MlxArray> array_reshape(
    const MlxArray& a, rust::Slice<const int32_t> shape,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_transpose(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_transpose_axes(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_broadcast_to(
    const MlxArray& a, rust::Slice<const int32_t> shape,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// Concatenate/stack accept raw pointer slices because cxx 1.0 doesn't bridge
// &[&MlxArray] directly. Caller (Rust safe layer) builds the pointer slice.
std::unique_ptr<MlxArray> array_concatenate(
    rust::Slice<const MlxArray* const> arrays, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_stack(
    rust::Slice<const MlxArray* const> arrays, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// Split returns std::vector<array> wrapped in MlxArrayVec opaque holder.
std::unique_ptr<MlxArrayVec> array_split_n(
    const MlxArray& a, int32_t num_splits, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArrayVec> array_split_at(
    const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// MlxArrayVec accessors.
size_t split_result_len(const MlxArrayVec& v);
std::unique_ptr<MlxArray> split_result_at(const MlxArrayVec& v, size_t i);

// === P1b2a matmul (P5.7: + 4 trailing stream params) ===

std::unique_ptr<MlxArray> array_matmul(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P1b2b dtype extension: u16/u32/u64 ===

std::unique_ptr<MlxArray> array_from_u16(rust::Slice<const uint16_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape);
std::unique_ptr<MlxArray> array_from_u64(rust::Slice<const uint64_t> data, rust::Slice<const int32_t> shape);

uint16_t array_item_u16(const MlxArray& a);
uint32_t array_item_u32(const MlxArray& a);
uint64_t array_item_u64(const MlxArray& a);

rust::Vec<uint16_t> array_to_vec_u16(const MlxArray& a);
rust::Vec<uint32_t> array_to_vec_u32(const MlxArray& a);
rust::Vec<uint64_t> array_to_vec_u64(const MlxArray& a);

// === P1b2b indexing ops (P5.7: + 4 trailing stream params) ===

std::unique_ptr<MlxArray> array_where(
    const MlxArray& cond, const MlxArray& x, const MlxArray& y,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_take(
    const MlxArray& a, const MlxArray& indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> array_take_along_axis(
    const MlxArray& a, const MlxArray& indices, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_slice_strided(
    const MlxArray& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> array_gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5 ops extensions: matmul family (P5.7: + 4 trailing stream params) ===

std::unique_ptr<MlxArray> tensordot_axis(
    const MlxArray& a, const MlxArray& b, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> tensordot_axes(
    const MlxArray& a, const MlxArray& b,
    rust::Slice<const int32_t> axes_a,
    rust::Slice<const int32_t> axes_b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> outer(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> inner(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> addmm(
    const MlxArray& c, const MlxArray& a, const MlxArray& b,
    float alpha, float beta,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a, const MlxArray& b, int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a, const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

std::unique_ptr<MlxArray> segmented_mm(
    const MlxArray& a, const MlxArray& b, const MlxArray& segments,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 comparison + element-wise binary ===
std::unique_ptr<MlxArray> equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> not_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> less(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> less_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> greater(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> greater_equal(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> maximum(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> minimum(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 clip (3-input element-wise with optional bounds) ===
// `a_min` / `a_max` are nullable: nullptr means "no bound on that side".
std::unique_ptr<MlxArray> clip(
    const MlxArray& a,
    const MlxArray* a_min,
    const MlxArray* a_max,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 softmax (multi-axis dispatch) ===
// `axes` is empty -> last-axis default (mlx::core::softmax(a, precise, s)).
// `axes` non-empty -> multi-axis form (mlx::core::softmax(a, vector<int>, precise, s)).
std::unique_ptr<MlxArray> softmax(
    const MlxArray& a, rust::Slice<const int32_t> axes, bool precise,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 sort family (sort/argsort/partition/argpartition/topk) ===
std::unique_ptr<MlxArray> sort(
    const MlxArray& a, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> argsort(
    const MlxArray& a, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> partition(
    const MlxArray& a, int32_t kth, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> argpartition(
    const MlxArray& a, int32_t kth, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> topk(
    const MlxArray& a, int32_t k, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 astype (dtype conversion) ===
// `dtype_repr` is `Dtype::as_u8()` from Rust; decoded by `helpers::dtype_from_repr`.
std::unique_ptr<MlxArray> astype(
    const MlxArray& a, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 array constructors ===
std::unique_ptr<MlxArray> arange(
    double start, double stop, double step, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> linspace(
    double start, double stop, int32_t num, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> ones(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> ones_like(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> zeros_like(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> full(
    rust::Slice<const int32_t> shape, const MlxArray& vals, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> full_like(
    const MlxArray& a, const MlxArray& vals,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> eye(
    int32_t n, int32_t m, int32_t k, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> identity(
    int32_t n, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> tri(
    int32_t n, int32_t m, int32_t k, uint8_t dtype_repr,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> tril(
    const MlxArray& x, int32_t k,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> triu(
    const MlxArray& x, int32_t k,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 一元补完 (abs/sign/floor/ceil/round/sin/cos/tan/expm1) ===
// `round` carries an extra `decimals` parameter (forwarded to MLX's overload
// `mlx::core::round(a, decimals, s)`). The other 8 follow the standard
// unary signature.
std::unique_ptr<MlxArray> abs(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> sign(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> floor(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> ceil(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> round(
    const MlxArray& a, int32_t decimals,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> sin(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> cos(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> tan(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> expm1(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 数值卫生 + logical_not (isnan/isinf/isfinite/nan_to_num/logical_not) ===
// `nan_to_num` carries 3 scalar params; `posinf`/`neginf` are optional and
// encoded as `(has_*, value)` pairs (parallels P4 random's loc/scale
// pattern). The shim rebuilds the Option<float> on the C++ side.
std::unique_ptr<MlxArray> isnan(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> isinf(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> isfinite(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> logical_not(
    const MlxArray& a,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> nan_to_num(
    const MlxArray& a, float nan,
    bool has_posinf, float posinf,
    bool has_neginf, float neginf,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 二元补完 (power/logaddexp/remainder) ===
// All three are element-wise broadcast binary ops; broadcasting is handled
// MLX-side. Return dtype follows MLX promotion rules.
std::unique_ptr<MlxArray> power(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> logaddexp(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> remainder(
    const MlxArray& a, const MlxArray& b,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.5 expand_dims / squeeze (axis-driven shape ops) ===
// `axes` slice is forwarded as `std::vector<int>`. For `squeeze`, an empty
// slice falls through to the no-axis MLX overload (squeeze every size-1 dim).
// For `expand_dims`, an empty slice is illegal and MLX will throw.
std::unique_ptr<MlxArray> expand_dims(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> squeeze(
    const MlxArray& a, rust::Slice<const int32_t> axes,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 累积归约 (cumsum/cumprod) ===
// Scan along `axis`. `reverse=true` flips scan direction; `inclusive=true`
// includes the element at the index in the running aggregate.
std::unique_ptr<MlxArray> cumsum(
    const MlxArray& a, int32_t axis, bool reverse, bool inclusive,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> cumprod(
    const MlxArray& a, int32_t axis, bool reverse, bool inclusive,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

// === P5.6 shape 补完 (flatten/repeat) ===
// `flatten` collapses dims `[start_axis, end_axis]` (inclusive, neg indices
// allowed) into one. `repeat` tiles `a` `repeats` times along `axis`.
std::unique_ptr<MlxArray> flatten(
    const MlxArray& a, int32_t start_axis, int32_t end_axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
std::unique_ptr<MlxArray> repeat(
    const MlxArray& a, int32_t repeats, int32_t axis,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);

}  // namespace cxx_mlx
