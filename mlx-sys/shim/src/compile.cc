#include "cxx_mlx_shim/compile.h"

#include <stdexcept>
#include <utility>

namespace cxx_mlx {

// ===== Global controls =====

void disable_compile() {
  mlx::core::disable_compile();
}

void enable_compile() {
  mlx::core::enable_compile();
}

void set_compile_mode(uint8_t mode) noexcept {
  using mlx::core::CompileMode;
  switch (mode) {
    case 0:
      mlx::core::set_compile_mode(CompileMode::disabled);
      break;
    case 1:
      mlx::core::set_compile_mode(CompileMode::no_simplify);
      break;
    case 2:
      mlx::core::set_compile_mode(CompileMode::no_fuse);
      break;
    case 3:
      mlx::core::set_compile_mode(CompileMode::enabled);
      break;
    default:
      break;
  }
}

// ===== ArrayVec =====

std::unique_ptr<ArrayVec> array_vec_new() {
  return std::make_unique<ArrayVec>();
}

size_t array_vec_count(const ArrayVec& v) {
  return v.inner.size();
}

std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i) {
  if (i >= v.inner.size()) {
    throw std::out_of_range("array_vec_get_at: index out of range");
  }
  // Copy ctor of mlx::core::array shares the underlying buffer cheaply.
  return std::make_unique<MlxArray>(v.inner[i]);
}

std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i) {
  if (i >= v.inner.size()) {
    throw std::out_of_range("array_vec_take_at: index out of range");
  }
  auto out = std::make_unique<MlxArray>(std::move(v.inner[i]));
  v.inner.erase(v.inner.begin() + static_cast<std::ptrdiff_t>(i));
  return out;
}

void array_vec_push(ArrayVec& v, const MlxArray& a) {
  v.inner.push_back(a);
}

} // namespace cxx_mlx
