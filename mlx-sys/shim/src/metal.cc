#include "cxx_mlx_shim/metal.h"

#include <stdexcept>
#include <string>
#include <variant>

#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"

namespace cxx_mlx {

void start_capture(rust::Str path) {
  mlx::core::metal::start_capture(std::string(path));
}

void stop_capture() {
  mlx::core::metal::stop_capture();
}

rust::String device_architecture() {
  // mlx upstream relocated metal::device_info() to the device-agnostic
  // mlx::core::device_info(const Device&) entry point in mlx/device.h.
  // The legacy metal::device_info() symbol is no longer exported by
  // installed libmlx.a, so we route through the public API instead.
  const auto& info = mlx::core::device_info(
      mlx::core::Device(mlx::core::Device::gpu));
  auto it = info.find("architecture");
  if (it == info.end()) {
    throw std::runtime_error(
        "mlx::core::device_info(gpu) has no 'architecture' entry");
  }
  if (const auto* s = std::get_if<std::string>(&it->second)) {
    return rust::String(*s);
  }
  throw std::runtime_error(
      "mlx::core::device_info(gpu)['architecture'] is not a std::string");
}

}  // namespace cxx_mlx
