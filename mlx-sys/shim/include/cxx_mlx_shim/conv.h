#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

std::unique_ptr<MlxArray> ops_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index);

}  // namespace cxx_mlx
