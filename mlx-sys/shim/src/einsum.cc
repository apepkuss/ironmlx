#include "cxx_mlx_shim/einsum.h"
#include "cxx_mlx_shim/shim_helpers.h"
#include "mlx/einsum.h"
namespace cxx_mlx {
std::unique_ptr<MlxArray> ops_einsum(rust::Str equation, const ArrayVec& operands, bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::einsum(std::string(equation), operands.inner, target));
}
}
