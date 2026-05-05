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

// ===== Basic distributions =====

std::unique_ptr<MlxArray> bits(
    rust::Slice<const int32_t> shape, int32_t width,
    const MlxArray* key);

std::unique_ptr<MlxArray> uniform(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> uniform_default(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> normal(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* loc, const MlxArray* scale,
    const MlxArray* key);

std::unique_ptr<MlxArray> randint(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

// ===== Discrete distributions =====

std::unique_ptr<MlxArray> bernoulli(
    const MlxArray& p, rust::Slice<const int32_t> shape,
    const MlxArray* key);

std::unique_ptr<MlxArray> bernoulli_default(
    const MlxArray& p,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical(
    const MlxArray& logits, int32_t axis,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical_n(
    const MlxArray& logits, int32_t axis, int32_t num_samples,
    const MlxArray* key);

std::unique_ptr<MlxArray> categorical_shaped(
    const MlxArray& logits, int32_t axis,
    rust::Slice<const int32_t> shape,
    const MlxArray* key);

// ===== Special distributions =====

std::unique_ptr<MlxArray> truncated_normal(
    const MlxArray& lower, const MlxArray& upper,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> truncated_normal_default(
    const MlxArray& lower, const MlxArray& upper,
    uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> gumbel(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

std::unique_ptr<MlxArray> laplace(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    float loc, float scale,
    const MlxArray* key);

std::unique_ptr<MlxArray> multivariate_normal(
    const MlxArray& mean, const MlxArray& cov,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key);

}  // namespace cxx_mlx
