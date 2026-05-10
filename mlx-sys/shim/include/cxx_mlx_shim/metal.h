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

// === Device architecture query (tile-lookup tables) ===
//
// Returns the Metal device's architecture name as reported by
// `MTLDevice.architecture.name` (e.g. "apple_g13s" for the M1 Pro 16-core
// GPU, "apple_g15p" for M3 Pro). Wraps the public
// `mlx::core::metal::device_info()` map's "architecture" entry so a Rust
// caller can pick a per-arch tile without re-implementing the Metal device
// query.
//
// Throws `std::runtime_error` if the Metal backend isn't available, or if
// the entry's variant unexpectedly isn't a string (defensive — current
// MLX always stores it as `std::string`).

rust::String device_architecture();

}  // namespace cxx_mlx
