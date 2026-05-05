#include "cxx_mlx_shim/random.h"
#include "cxx_mlx_shim/shim_helpers.h"

#include <stdexcept>

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

}  // namespace cxx_mlx
