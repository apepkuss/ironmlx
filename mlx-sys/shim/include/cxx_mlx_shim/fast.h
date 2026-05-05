#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// rms_norm: weight=nullptr → std::nullopt
std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray* weight,
    float eps);

// layer_norm: weight=nullptr → no weight, bias=nullptr → no bias
std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,
    const MlxArray* bias,
    float eps);

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
    const MlxArray* freqs);

// rope (array offset) — 同上 base/freqs 处理；offset 改为引用 array
std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    const MlxArray& offset,
    const MlxArray* freqs);

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
    const MlxArray* sinks);

}  // namespace cxx_mlx
