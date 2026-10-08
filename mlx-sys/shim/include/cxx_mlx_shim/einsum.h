#pragma once
#include "cxx_mlx_shim/compile.h"
namespace cxx_mlx {
std::unique_ptr<MlxArray> ops_einsum(rust::Str equation, const ArrayVec& operands, bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index);
}
