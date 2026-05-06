#pragma once

#include <cstdint>
#include <memory>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== Global controls =====
void disable_compile();
void enable_compile();
// mode: 0=Disabled, 1=NoSimplify, 2=NoFuse, 3=Enabled.
// Out-of-range values are silently ignored (Rust enum guards the domain).
void set_compile_mode(uint8_t mode) noexcept;

} // namespace cxx_mlx
