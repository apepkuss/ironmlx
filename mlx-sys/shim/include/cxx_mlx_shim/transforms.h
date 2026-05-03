#pragma once

#include "rust/cxx.h"

namespace mlx::core {
class array;
}

namespace cxx_mlx {

using MlxArray = mlx::core::array;

void eval_one(const MlxArray& a);

}  // namespace cxx_mlx
