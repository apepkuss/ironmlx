#pragma once

#include "rust/cxx.h"

namespace cxx_mlx {

// === Metal capture (debug/profiling) ===
//
// Wraps `mlx::core::metal::start_capture(path) / stop_capture()` from
// /Volumes/Dev/mlx/mlx/backend/metal/metal.h. Used by ironmlx-p8a-stage4
// to produce Xcode-readable .gputrace bundles for per-kernel profiling.
//
// `start_capture(path)` opens an Xcode-compatible .gputrace file at `path`
// and starts capturing every Metal command submitted on the default device.
// `stop_capture()` finalizes the capture. Both throw `std::runtime_error`
// on Metal driver / capture-manager failure (e.g. missing Xcode entitlement,
// path not writable, capture already running).

void start_capture(rust::Str path);
void stop_capture();

}  // namespace cxx_mlx
