#include "cxx_mlx_shim/random.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <stdexcept>
#include <vector>

namespace cxx_mlx {

// ===== KeyPair =====

KeyPair::KeyPair(mlx::core::array first, mlx::core::array second)
    : first_(std::move(first)), second_(std::move(second)) {}

std::unique_ptr<MlxArray> KeyPair::take_first() {
  if (first_taken_) {
    throw std::runtime_error("KeyPair::take_first: already taken");
  }
  first_taken_ = true;
  return std::make_unique<MlxArray>(std::move(first_));
}

std::unique_ptr<MlxArray> KeyPair::take_second() {
  if (second_taken_) {
    throw std::runtime_error("KeyPair::take_second: already taken");
  }
  second_taken_ = true;
  return std::make_unique<MlxArray>(std::move(second_));
}

std::unique_ptr<MlxArray> key_pair_take_first(KeyPair& p) {
  return p.take_first();
}

std::unique_ptr<MlxArray> key_pair_take_second(KeyPair& p) {
  return p.take_second();
}

// ===== State =====

std::unique_ptr<MlxArray> key(uint64_t seed) {
  return std::make_unique<MlxArray>(mlx::core::random::key(seed));
}

void seed(uint64_t seed) {
  mlx::core::random::seed(seed);
}

std::unique_ptr<KeyPair> split(const MlxArray& key) {
  auto p = mlx::core::random::split(key);
  return std::make_unique<KeyPair>(std::move(p.first), std::move(p.second));
}

std::unique_ptr<MlxArray> split_n(const MlxArray& key, int32_t num) {
  return std::make_unique<MlxArray>(mlx::core::random::split(key, num));
}

// ===== Basic distributions =====

std::unique_ptr<MlxArray> bits(
    rust::Slice<const int32_t> shape, int32_t width,
    const MlxArray* key) {
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::bits(
      shape_vec, width, helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> uniform(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> uniform_default(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      shape_vec, helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> normal(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* loc, const MlxArray* scale,
    const MlxArray* key) {
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::normal(
      shape_vec, helpers::dtype_from_repr(dtype_repr),
      helpers::opt_arr(loc), helpers::opt_arr(scale),
      helpers::opt_arr(key)));
}

std::unique_ptr<MlxArray> randint(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key) {
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::randint(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key)));
}

}  // namespace cxx_mlx
