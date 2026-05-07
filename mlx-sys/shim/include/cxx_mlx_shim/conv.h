#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// 1D convolution: mlx::core::conv1d.
//   input layout:  [N, L, C_in]
//   weight layout: [C_out, K, C_in / groups]    (note: NOT PyTorch [C_out, C_in/groups, K])
//   output:        [N, L_out, C_out] where L_out = (L + 2*padding - dilation*(K-1) - 1)/stride + 1
//   For depthwise: groups = C_in = C_out.
//   Stream encoding: 4 trailing args (has_target, is_device_only, device_type, stream_index)
//   per P5.7 — same convention as the array bridge.
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
