#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "rust/cxx.h"

namespace cxx_mlx {

// Forward decl: cxx-generated for the extern "Rust" CompileCallback type
// (see mlx-sys/src/bridge/compile.rs). The full definition lives in
// mlx-sys/src/bridge/compile.rs.h, included only by compile.cc.
struct CompileCallback;

using MlxArray = mlx::core::array;

// ===== Global controls =====
void disable_compile();
void enable_compile();
// mode: 0=Disabled, 1=NoSimplify, 2=NoFuse, 3=Enabled.
// Out-of-range values are silently ignored (Rust enum guards the domain).
void set_compile_mode(uint8_t mode) noexcept;

// ===== ArrayVec (bidirectional opaque carrier) =====
//
// cxx 1.0 does not support `Vec<UniquePtr<T>>`, so we wrap
// `std::vector<mlx::core::array>` in an opaque struct and expose
// scalar accessors. Used for both C++→Rust (compile callback inputs)
// and Rust→C++ (callback outputs, compiled-fn invoke).
struct ArrayVec {
  std::vector<mlx::core::array> inner;
};

std::unique_ptr<ArrayVec> array_vec_new();
size_t array_vec_count(const ArrayVec& v);

// Returns a clone (shares storage with the element via MLX refcount).
// Throws std::out_of_range if i >= count.
std::unique_ptr<MlxArray> array_vec_get_at(const ArrayVec& v, size_t i);

// Moves element i out and erases it; subsequent elements shift down.
// Throws std::out_of_range if i >= count.
std::unique_ptr<MlxArray> array_vec_take_at(ArrayVec& v, size_t i);

// Appends a copy (cheap MLX refcount).
void array_vec_push(ArrayVec& v, const MlxArray& a);

// ===== CompiledFn (RAII handle for std::function) =====
struct CompiledFn {
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      fn;
};

// Build a CompiledFn from a Rust callback. The callback is wrapped in a
// shared_ptr<rust::Box<CompileCallback>> so the lambda passed to
// mlx::core::compile is CopyConstructible (rust::Box is move-only and
// std::function requires CopyConstructible).
std::unique_ptr<CompiledFn> compile_with_callback(
    rust::Box<CompileCallback> cb, bool shapeless);

// Replay the compiled graph on the given inputs.
std::unique_ptr<ArrayVec> compiled_fn_invoke(
    const CompiledFn& cf, const ArrayVec& inputs);

} // namespace cxx_mlx
