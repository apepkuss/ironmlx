#pragma once

#include <cstddef>
#include <cstdint>

#include "rust/cxx.h"
#include "cxx_mlx_shim/array.h"

namespace cxx_mlx {

struct ScopedResidencyStats;
class ScopedResidency {
 public:
  struct Impl;
  explicit ScopedResidency(std::unique_ptr<Impl> impl);
  ~ScopedResidency();
  std::unique_ptr<Impl> impl;
};
std::unique_ptr<ScopedResidency> make_scoped_residency(
    rust::Slice<const MlxArray* const> arrays, std::size_t budget_bytes);
std::size_t scoped_residency_bytes(const ScopedResidency& lease);
ScopedResidencyStats scoped_residency_stats();
void set_scoped_residency_enabled(bool enabled);
bool scoped_residency_bind(const ScopedResidency& lease, bool has_target,
                          bool is_device_only, std::uint8_t device_type,
                          std::int32_t stream_index);
bool scoped_residency_external_wired(bool enabled);

std::size_t get_active_memory();
std::size_t get_cache_memory();
std::size_t get_peak_memory();
std::size_t get_memory_limit();
std::size_t set_cache_limit(std::size_t limit);
std::size_t set_wired_limit(std::size_t limit);
void start_residency_refresh(std::uint32_t interval_ms);
std::size_t get_memory_size();
std::size_t get_max_recommended_memory();
rust::String get_device_name();

}  // namespace cxx_mlx
