#include "cxx_mlx_shim/fast.h"

#include <optional>
#include <string>

#include "mlx/fast.h"

namespace cxx_mlx {

namespace {

// pointer → optional<array>. mlx::array copy is cheap (refcount on array_desc_).
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

[[maybe_unused]] inline std::optional<float> opt_f(bool has, float v) {
  return has ? std::optional<float>(v) : std::nullopt;
}

}  // namespace

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x, const MlxArray* weight, float eps) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::rms_norm(x, opt_arr(weight), eps));
}

std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x, const MlxArray* weight, const MlxArray* bias, float eps) {
  return std::make_unique<MlxArray>(
      mlx::core::fast::layer_norm(x, opt_arr(weight), opt_arr(bias), eps));
}

}  // namespace cxx_mlx
