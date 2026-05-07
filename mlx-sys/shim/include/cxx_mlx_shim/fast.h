#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/fast.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// rms_norm: weight=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray* weight,
    float eps,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

// layer_norm: weight=nullptr → no weight, bias=nullptr → no bias
std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,
    const MlxArray* bias,
    float eps,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

// rope (int offset)
//   has_base=false → std::nullopt (MLX 内部回落到默认 base 处理逻辑)
//   freqs=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    int32_t offset,
    const MlxArray* freqs,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

// rope (array offset) — 同上 base/freqs 处理；offset 改为引用 array
std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    const MlxArray& offset,
    const MlxArray* freqs,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

// scaled_dot_product_attention
//   mask_mode: rust::Str → std::string  ("" 等价 MLX 默认值)
//   mask_arr=nullptr → std::nullopt
//   sinks=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& queries,
    const MlxArray& keys,
    const MlxArray& values,
    float scale,
    rust::Str mask_mode,
    const MlxArray* mask_arr,
    const MlxArray* sinks,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

// === P3a metal_kernel ===

// Opaque types crossing cxx (declared here, defined inline because they hold
// non-cxx-friendly types: std::function and std::vector<Shape>).
struct MetalKernelInner {
  mlx::core::fast::CustomKernelFunction fn;
};

struct ShapesVec {
  std::vector<mlx::core::Shape> shapes;
};

// === ShapesVec API ===
std::unique_ptr<ShapesVec> shapes_vec_new();
void shapes_vec_push(ShapesVec& v, rust::Slice<const int32_t> shape);
size_t shapes_vec_count(const ShapesVec& v);

}  // namespace cxx_mlx
