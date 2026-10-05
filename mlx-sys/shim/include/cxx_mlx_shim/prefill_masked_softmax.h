#pragma once

#include <optional>
#include "mlx/array.h"
#include "mlx/utils.h"

namespace cxx_mlx {
// Default-off (IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX=1). For causal,
// unmasked, sink-free BF16 attention that MLX itself routes to its op-level
// fallback with more than 128 query rows, rebuild that fallback op for op but
// fuse the causal `where` into a precise softmax that reproduces MLX 0.32.2
// softmax arithmetic and may update the score buffer in place.
std::optional<mlx::core::array> experimental_masked_causal_sdpa(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, float scale, mlx::core::StreamOrDevice target);
}  // namespace cxx_mlx
