#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

std::unique_ptr<MlxArray> array_zeros(rust::Slice<const int32_t> shape, uint8_t dtype);
rust::Vec<int32_t> array_shape(const MlxArray& a);

}  // namespace cxx_mlx
