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

}  // namespace cxx_mlx
