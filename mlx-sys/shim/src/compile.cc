#include "cxx_mlx_shim/compile.h"

namespace cxx_mlx {

// ===== Global controls =====

void disable_compile() {
  mlx::core::disable_compile();
}

void enable_compile() {
  mlx::core::enable_compile();
}

void set_compile_mode(uint8_t mode) noexcept {
  using mlx::core::CompileMode;
  switch (mode) {
    case 0:
      mlx::core::set_compile_mode(CompileMode::disabled);
      break;
    case 1:
      mlx::core::set_compile_mode(CompileMode::no_simplify);
      break;
    case 2:
      mlx::core::set_compile_mode(CompileMode::no_fuse);
      break;
    case 3:
      mlx::core::set_compile_mode(CompileMode::enabled);
      break;
    default:
      break;
  }
}

} // namespace cxx_mlx
