#pragma once

#include "rust/cxx.h"

namespace mlx::core {
class array;
}

namespace cxx_mlx {

using MlxArray = mlx::core::array;

void eval_one(const MlxArray& a);

// Wait for an already-scheduled array's event to fire. Cross-thread safe
// (MLX Events are MTLSharedEvent-backed and waitable from any thread),
// unlike Stream-based synchronization which is tied to per-thread TLS
// state in the MLX scheduler.
void array_wait(const MlxArray& a);

}  // namespace cxx_mlx
