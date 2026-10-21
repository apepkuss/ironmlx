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
// Split-K D256 NAX attention of a B1 flat draft tree: each row computes the
// row of a linear verify of its own root-to-node path (see the shader).
std::optional<mlx::core::array> experimental_tree_d256_nax(
    const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
    const mlx::core::array& tk, const mlx::core::array& tv, const mlx::core::array& lims,
    float scale, int32_t gqa, int32_t keys, int32_t tail_from,
    mlx::core::StreamOrDevice target);
// Keys per split-K block of a decode / verify call over `keys` keys.
int32_t decode_d256_nax_block_keys(int32_t keys);
// Split-K NAX attention of a B1 head-dim-128 query block (at most 16 rows)
// where query row r sees keys [window_start + r, keys) (DFlash2 drafter).
std::optional<mlx::core::array> experimental_window_attention_nax(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, float scale, int32_t window_start,
    mlx::core::StreamOrDevice target);
}  // namespace cxx_mlx
