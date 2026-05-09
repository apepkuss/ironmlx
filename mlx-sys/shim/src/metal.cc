#include "cxx_mlx_shim/metal.h"

#include <stdexcept>
#include <string>
#include <variant>

#include "mlx/backend/metal/metal.h"

namespace cxx_mlx {

void start_capture(rust::Str path) {
  mlx::core::metal::start_capture(std::string(path));
}

void stop_capture() {
  mlx::core::metal::stop_capture();
}

rust::String device_architecture() {
  const auto& info = mlx::core::metal::device_info();
  auto it = info.find("architecture");
  if (it == info.end()) {
    throw std::runtime_error(
        "mlx::core::metal::device_info() has no 'architecture' entry");
  }
  if (const auto* s = std::get_if<std::string>(&it->second)) {
    return rust::String(*s);
  }
  throw std::runtime_error(
      "mlx::core::metal::device_info()['architecture'] is not a std::string");
}

}  // namespace cxx_mlx
