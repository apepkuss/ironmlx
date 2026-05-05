#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "mlx/array.h"
#include "mlx/random.h"
#include "rust/cxx.h"

namespace cxx_mlx {

using MlxArray = mlx::core::array;

// ===== KeyPair (opaque) =====
// MLX 的 split(key) 返回 std::pair<array, array>；cxx 不支持 pair。
// 包装为 opaque 类，提供 take_first() + take_second() 接口。
// 各自的 taken_ bool 防重取（与 P3 QuantizeResult 单次性消费契约一致）。
class KeyPair {
 public:
  KeyPair(mlx::core::array first, mlx::core::array second);
  std::unique_ptr<MlxArray> take_first();
  std::unique_ptr<MlxArray> take_second();

 private:
  mlx::core::array first_;
  mlx::core::array second_;
  bool first_taken_ = false;
  bool second_taken_ = false;
};

std::unique_ptr<MlxArray> key_pair_take_first(KeyPair& p);
std::unique_ptr<MlxArray> key_pair_take_second(KeyPair& p);

// ===== State management =====
std::unique_ptr<MlxArray> key(uint64_t seed);
void seed(uint64_t seed);
std::unique_ptr<KeyPair> split(const MlxArray& key);
std::unique_ptr<MlxArray> split_n(const MlxArray& key, int32_t num);

}  // namespace cxx_mlx
