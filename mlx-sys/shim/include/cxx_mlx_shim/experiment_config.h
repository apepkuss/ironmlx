#pragma once

#include <cstdint>
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
// BM128 QMM for every affine4/g64 prefill chunk (at least 256 rows, 64-column
// multiple outputs), not only the tested dense 27B MLP shapes.
bool set_prefill_qmm_mtile_any_shape(bool enabled);
// Fewest keys at which a causal chunk of any GQA layout takes the D256 NAX
// attention (0: only the tested dense 27B layout does).
bool set_prefill_d256_nax_min_keys(int32_t keys);
// Fewest keys at which a query block of at most 8 rows takes the split-K
// D256 NAX attention (0: never).
bool set_decode_d256_nax_min_keys(int32_t keys);

const std::string& prefill_qmm_mtile_library();
const std::string& prefill_d256_nax_library();
bool prefill_masked_softmax_requested();
bool prefill_qmm_mtile_any_shape();
int32_t prefill_d256_nax_min_keys();
int32_t decode_d256_nax_min_keys();
// The split-K minimum key count if that attention can run (library set,
// NAX available), else 0. Resolves both settings.
int32_t decode_d256_nax_active_min_keys();

}  // namespace cxx_mlx
