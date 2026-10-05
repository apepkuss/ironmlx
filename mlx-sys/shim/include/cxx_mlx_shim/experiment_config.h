#pragma once

#include <string>

#include "rust/cxx.h"

// Process-wide configuration of the optional prefill kernels (QMM M-tile,
// D256 NAX attention, masked causal softmax). The Rust device profile sets
// these once at startup, before the first prefill; unset, each falls back to
// its legacy IRONMLX_EXPERIMENTAL_* environment variable. A setter returns
// false once the value has been read (too late to change it).
namespace cxx_mlx {

bool set_prefill_qmm_mtile_library(rust::Str path);
bool set_prefill_d256_nax_library(rust::Str path);
bool set_prefill_masked_softmax(bool enabled);

const std::string& prefill_qmm_mtile_library();
const std::string& prefill_d256_nax_library();
bool prefill_masked_softmax_requested();

}  // namespace cxx_mlx
