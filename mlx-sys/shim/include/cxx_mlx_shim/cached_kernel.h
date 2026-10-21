#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace cxx_mlx {

// Per-kernel cache of the custom-kernel signatures MLX has generated: the
// kernel source and the compiled pipeline are reused, so a dispatch of a
// known signature skips MLX's per-call source generation, source hashing
// and library / pipeline lookups. The kernel that runs is the one MLX built
// for that signature, so results are unchanged.
struct CachedKernelTable;

std::shared_ptr<CachedKernelTable> cached_kernel_table_new();

// Outputs of a dispatch whose signature `key` is cached; nullopt otherwise.
std::optional<std::vector<mlx::core::array>> cached_kernel_dispatch(
    CachedKernelTable& table,
    const std::vector<int64_t>& key,
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::Shape>& output_shapes,
    const std::vector<mlx::core::Dtype>& output_dtypes,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    std::optional<float> init_value,
    mlx::core::StreamOrDevice target);

// Remember the signature of `outputs` (fresh outputs of MLX's own dispatch).
void cached_kernel_remember(
    CachedKernelTable& table,
    const std::vector<int64_t>& key,
    const std::vector<mlx::core::array>& outputs);

}  // namespace cxx_mlx
