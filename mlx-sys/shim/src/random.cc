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
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::bits(
      shape_vec, width, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> uniform(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> uniform_default(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::uniform(
      shape_vec, helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> normal(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* loc, const MlxArray* scale,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::normal(
      shape_vec, helpers::dtype_from_repr(dtype_repr),
      helpers::opt_arr(loc), helpers::opt_arr(scale),
      helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> randint(
    const MlxArray& low, const MlxArray& high,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::randint(
      low, high, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

// ===== Discrete distributions =====

std::unique_ptr<MlxArray> bernoulli(
    const MlxArray& p, rust::Slice<const int32_t> shape,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::bernoulli(
      p, shape_vec, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> bernoulli_default(
    const MlxArray& p,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::bernoulli(
      p, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> categorical(
    const MlxArray& logits, int32_t axis,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> categorical_n(
    const MlxArray& logits, int32_t axis, int32_t num_samples,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, num_samples, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> categorical_shaped(
    const MlxArray& logits, int32_t axis,
    rust::Slice<const int32_t> shape,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::categorical(
      logits, axis, shape_vec, helpers::opt_arr(key), target));
}

// ===== Special distributions =====

std::unique_ptr<MlxArray> truncated_normal(
    const MlxArray& lower, const MlxArray& upper,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
      lower, upper, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> truncated_normal_default(
    const MlxArray& lower, const MlxArray& upper,
    uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
      lower, upper,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> gumbel(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::gumbel(
      shape_vec, helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> laplace(
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    float loc, float scale,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::laplace(
      shape_vec, helpers::dtype_from_repr(dtype_repr), loc, scale,
      helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> multivariate_normal(
    const MlxArray& mean, const MlxArray& cov,
    rust::Slice<const int32_t> shape, uint8_t dtype_repr,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  mlx::core::Shape shape_vec(shape.begin(), shape.end());
  return std::make_unique<MlxArray>(mlx::core::random::multivariate_normal(
      mean, cov, shape_vec,
      helpers::dtype_from_repr(dtype_repr), helpers::opt_arr(key), target));
}

// ===== Permutation =====

std::unique_ptr<MlxArray> permutation(
    const MlxArray& x, int32_t axis,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::permutation(
      x, axis, helpers::opt_arr(key), target));
}

std::unique_ptr<MlxArray> permutation_arange(
    int32_t n,
    const MlxArray* key,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(mlx::core::random::permutation(
      n, helpers::opt_arr(key), target));
}

}  // namespace cxx_mlx
