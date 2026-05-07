#include "cxx_mlx_shim/conv.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include "mlx/ops.h"

namespace cxx_mlx {

std::unique_ptr<MlxArray> ops_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    bool has_target,
    bool is_device_only,
    uint8_t device_type,
    int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::conv1d(input, weight, stride, padding, dilation, groups, target));
}

}  // namespace cxx_mlx
