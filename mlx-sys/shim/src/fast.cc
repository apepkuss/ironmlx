#include "cxx_mlx_shim/fast.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <optional>
#include <string>

#include "mlx/fast.h"

namespace cxx_mlx {

namespace {

// pointer → optional<array>. mlx::array copy is cheap (refcount on array_desc_).
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<float> opt_f(bool has, float v) {
  return has ? std::optional<float>(v) : std::nullopt;
}

}  // namespace

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x, const MlxArray* weight, float eps,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rms_norm(x, opt_arr(weight), eps, target));
}

std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x, const MlxArray* weight, const MlxArray* bias, float eps,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::layer_norm(x, opt_arr(weight), opt_arr(bias), eps, target));
}

std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, int32_t offset,
    const MlxArray* freqs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs), target));
}

std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, const MlxArray& offset,
    const MlxArray* freqs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs), target));
}

std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& queries, const MlxArray& keys, const MlxArray& values,
    float scale, rust::Str mask_mode,
    const MlxArray* mask_arr, const MlxArray* sinks,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::scaled_dot_product_attention(
          queries, keys, values, scale,
          std::string(mask_mode),
          opt_arr(mask_arr),
          opt_arr(sinks),
          target));
}

}  // namespace cxx_mlx
