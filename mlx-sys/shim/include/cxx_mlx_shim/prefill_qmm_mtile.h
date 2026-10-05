#pragma once
#include <optional>
#include <string>
#include "mlx/array.h"
#include "mlx/utils.h"

namespace cxx_mlx {
// Default-off, separately compiled BF16 affine4 NAX QMM experiment.
std::optional<mlx::core::array> experimental_prefill_qmm_mtile(
    const mlx::core::array& x, const mlx::core::array& w,
    const mlx::core::array& scales, const std::optional<mlx::core::array>& biases,
    bool transpose, std::optional<int> group_size, std::optional<int> bits,
    const std::string& mode, mlx::core::StreamOrDevice target);
}  // namespace cxx_mlx
