#include "cxx_mlx_shim/compile.h"

#include <stdexcept>
#include <utility>

// `compile_clear_cache` is declared in MLX's internal-but-installed
// `compile_impl.h` (namespace `mlx::core::detail`), not the public
// `compile.h`. Include it here, scoped to the .cc to keep the public
// shim header clean.
#include "mlx/compile_impl.h"

// Pull in the cxx-generated header so the CompileCallback type and its
// `invoke` method are fully defined for the lambda below.
#include "mlx-sys/src/bridge/compile.rs.h"

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

// ===== CompiledFn =====

std::unique_ptr<CompiledFn> compile_with_callback(
    rust::Box<CompileCallback> cb, bool shapeless) {
  // std::function requires CopyConstructible; rust::Box is move-only.
  // shared_ptr lets the lambda satisfy the requirement.
  auto shared_cb =
      std::make_shared<rust::Box<CompileCallback>>(std::move(cb));

  auto traced = mlx::core::compile(
      [shared_cb](const std::vector<mlx::core::array>& inputs)
          -> std::vector<mlx::core::array> {
        // Wrap inputs into an ArrayVec the Rust callback can read.
        auto in_vec = std::make_unique<ArrayVec>();
        in_vec->inner = inputs; // copy ctor: cheap refcount per element.

        // Invoke Rust. cxx generates `invoke` returning UniquePtr<ArrayVec>;
        // a Rust Err (including panics caught on the Rust side) surfaces
        // here as a thrown rust::Error from cxx; MLX trace propagates it out.
        auto out_vec = (*shared_cb)->invoke(*in_vec);
        return std::move(out_vec->inner);
      },
      shapeless);

  auto out = std::make_unique<CompiledFn>();
  out->fn = std::move(traced);
  return out;
}

std::unique_ptr<ArrayVec> compiled_fn_invoke(
    const CompiledFn& cf, const ArrayVec& inputs) {
  auto outputs = cf.fn(inputs.inner);
  auto v = std::make_unique<ArrayVec>();
  v->inner = std::move(outputs);
  return v;
}

// === P5.7 compile cache control ===

void compile_clear_cache() {
  mlx::core::detail::compile_clear_cache();
}

} // namespace cxx_mlx
