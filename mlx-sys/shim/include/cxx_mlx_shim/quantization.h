#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== QuantizeResult (opaque) =====
// MLX 的 quantize 返回 std::vector<array>，cxx 不支持 Vec<UniquePtr<T>>。
// 包装为 opaque 类，提供 count() + take_at(idx) 接口。take_at 用 taken_
// bitmap 防止重复取（与 P2c take_by_name 单次性消费契约一致）。
class QuantizeResult {
 public:
  explicit QuantizeResult(std::vector<mlx::core::array> data);
  size_t count() const { return arrays_.size(); }
  std::unique_ptr<MlxArray> take_at(size_t idx);

 private:
  std::vector<mlx::core::array> arrays_;
  std::vector<bool> taken_;
};

size_t quantize_result_count(const QuantizeResult& r);
std::unique_ptr<MlxArray> quantize_result_take_at(QuantizeResult& r, size_t idx);

// ===== 量化函数 =====
// 可选参数编码:
//   Option<int>   → (bool has_value, int32_t value)
//   Option<Dtype> → (bool has_dtype, uint8_t dtype_repr)
//   Option<&Array>→ const MlxArray* (nullptr = None)
//   &str mode     → rust::Str

std::unique_ptr<QuantizeResult> quantize(
    const MlxArray& w,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale);

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    bool has_group_size, int32_t group_size,
    bool has_bits, int32_t bits,
    rust::Str mode,
    const MlxArray* global_scale,
    bool has_dtype, uint8_t dtype_repr);

}  // namespace cxx_mlx
