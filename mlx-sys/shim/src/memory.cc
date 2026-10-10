#include "cxx_mlx_shim/memory.h"

#include <stdexcept>
#include <string>
#include <variant>

#include "mlx/device.h"
#include "mlx/memory.h"

#if __has_include(<Metal/Metal.hpp>)
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/resident.h"
#endif

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

std::size_t set_cache_limit(std::size_t limit) {
  return mlx::core::set_cache_limit(limit);
}

std::size_t set_wired_limit(std::size_t limit) {
  return mlx::core::set_wired_limit(limit);
}

#if __has_include(<Metal/Metal.hpp>)
namespace {

using mlx::core::metal::ResidencySets;

// MLX keeps its residency sets private. Explicit instantiation is exempt from
// access checks, so this reaches them without changing MLX.
auto& residency_sets_of(ResidencySets& sets);
std::mutex& residency_mutex_of(ResidencySets& sets);

template <auto Sets, auto Mutex>
struct ResidencyAccess {
  friend auto& residency_sets_of(ResidencySets& sets) {
    return sets.*Sets;
  }
  friend std::mutex& residency_mutex_of(ResidencySets& sets) {
    return sets.*Mutex;
  }
};
template struct ResidencyAccess<&ResidencySets::sets_, &ResidencySets::mtx_>;

// macOS drops the residency of a set shortly after the GPU goes idle, even
// with a standing request, so the next request re-wires every weight it
// touches. Renewing the request keeps the wired allocations resident.
class ResidencyRefresher {
 public:
  ResidencyRefresher(ResidencySets& sets, std::chrono::milliseconds interval)
      : thread_([this, &sets, interval] {
          std::unique_lock<std::mutex> lock(mtx_);
          while (!cv_.wait_for(lock, interval, [this] { return stop_; })) {
            auto pool = mlx::core::metal::new_scoped_memory_pool();
            std::lock_guard<std::mutex> sets_lock(residency_mutex_of(sets));
            for (auto& set : residency_sets_of(sets)) {
              if (set.size != 0) {
                set.set->requestResidency();
              }
            }
          }
        }) {}

  ~ResidencyRefresher() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_{false};
  std::thread thread_;
};

}  // namespace

void start_residency_refresh(std::uint32_t interval_ms) {
  if (interval_ms == 0) {
    throw std::invalid_argument("residency refresh interval must be positive");
  }
  auto& sets =
      mlx::core::metal::device(mlx::core::Device(mlx::core::Device::gpu))
          .residency_sets();
  // Constructed after the device, so it stops before the device is destroyed.
  static ResidencyRefresher refresher(
      sets, std::chrono::milliseconds(interval_ms));
}
#else
void start_residency_refresh(std::uint32_t) {
  throw std::runtime_error("residency refresh requires Metal");
}
#endif

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
