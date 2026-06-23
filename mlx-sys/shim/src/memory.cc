#include "cxx_mlx_shim/memory.h"

#include <stdexcept>
#include <string>
#include <variant>

#include "mlx/device.h"
#include "mlx/memory.h"

namespace cxx_mlx {

namespace {

const std::variant<std::string, std::size_t>& device_info_value(
    const std::string& key) {
  const auto& info = mlx::core::device_info(
      mlx::core::Device(mlx::core::Device::gpu));
  auto it = info.find(key);
  if (it == info.end()) {
    throw std::runtime_error("mlx::core::device_info(gpu) has no '" + key +
                             "' entry");
  }
  return it->second;
}

std::size_t device_info_size(const std::string& key) {
  const auto& value = device_info_value(key);
  if (const auto* size = std::get_if<std::size_t>(&value)) {
    return *size;
  }
  throw std::runtime_error("mlx::core::device_info(gpu)['" + key +
                           "'] is not a size_t");
}

rust::String device_info_string(const std::string& key) {
  const auto& value = device_info_value(key);
  if (const auto* text = std::get_if<std::string>(&value)) {
    return rust::String(*text);
  }
  throw std::runtime_error("mlx::core::device_info(gpu)['" + key +
                           "'] is not a string");
}

}  // namespace

std::size_t get_active_memory() {
  return mlx::core::get_active_memory();
}

std::size_t get_cache_memory() {
  return mlx::core::get_cache_memory();
}

std::size_t get_peak_memory() {
  return mlx::core::get_peak_memory();
}

std::size_t get_memory_limit() {
  return mlx::core::get_memory_limit();
}

std::size_t get_memory_size() {
  return device_info_size("memory_size");
}

std::size_t get_max_recommended_memory() {
  return device_info_size("max_recommended_working_set_size");
}

rust::String get_device_name() {
  return device_info_string("device_name");
}

}  // namespace cxx_mlx
