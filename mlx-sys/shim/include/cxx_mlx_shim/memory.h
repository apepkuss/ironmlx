#pragma once

#include <cstddef>

#include "rust/cxx.h"

namespace cxx_mlx {

std::size_t get_active_memory();
std::size_t get_cache_memory();
std::size_t get_peak_memory();
std::size_t get_memory_limit();
std::size_t set_cache_limit(std::size_t limit);
std::size_t get_memory_size();
std::size_t get_max_recommended_memory();
rust::String get_device_name();

}  // namespace cxx_mlx
