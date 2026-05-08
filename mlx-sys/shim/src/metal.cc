#include "cxx_mlx_shim/metal.h"

#include <string>

#include "mlx/backend/metal/metal.h"

namespace cxx_mlx {

void start_capture(rust::Str path) {
  mlx::core::metal::start_capture(std::string(path));
}

void stop_capture() {
  mlx::core::metal::stop_capture();
}

}  // namespace cxx_mlx
