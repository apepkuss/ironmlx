#pragma once

#include <optional>
#include "mlx/array.h"
#include "mlx/utils.h"

namespace cxx_mlx {
// Default-off operator experiment. Returns no candidate for unsupported input.
// The explicitly supplied separately compiled metallib is NOT a replacement
// for the installed MLX library. No new API/ABI is taken from another MLX build.
std::optional<mlx::core::array> experimental_prefill_d256_nax(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, float scale, mlx::core::StreamOrDevice target);
// Split-K variant for query blocks of at most 8 rows over long caches
// (decode and speculative verify); `causal` aligns the rows to the cache end.
std::optional<mlx::core::array> experimental_decode_d256_nax(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, float scale, bool causal,
    mlx::core::StreamOrDevice target);
}  // namespace cxx_mlx
