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

}  // namespace cxx_mlx
