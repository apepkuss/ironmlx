#pragma once
#include "mlx/array.h"
#include <memory>
#include <cstdint>
namespace cxx_mlx {
using MlxArray = mlx::core::array;
std::unique_ptr<MlxArray> ops_real_fft(const MlxArray& input, int32_t n, int32_t axis, bool inverse, uint8_t norm,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
}
