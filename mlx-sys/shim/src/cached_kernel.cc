#include "cxx_mlx_shim/cached_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

struct CachedKernelState {
  std::string name;
  std::string source;
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  bool ensure_row_contiguous = true;
  int compile_options = 0;
  std::string lib_name;
  std::once_flag resolved;
  MTL::ComputePipelineState* pipeline = nullptr;
};

// MLX's CustomKernel::eval_gpu (73ad5df2) with the pipeline resolved once.
class CachedCustomKernel final : public mx::Primitive {
 public:
  CachedCustomKernel(mx::Stream stream, std::shared_ptr<CachedKernelState> state,
                     std::tuple<int, int, int> grid, std::tuple<int, int, int> threadgroup,
                     std::optional<float> init_value)
      : Primitive(stream), state_(std::move(state)), grid_(grid), threadgroup_(threadgroup),
        init_value_(init_value) {}
  const char* name() const override { return "CachedCustomKernel"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Custom kernels only run on GPU.");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    for (auto& out : outputs) {
      if (init_value_) {
        copies.emplace_back(init_value_.value(), out.dtype());
        mx::fill_gpu(copies.back(), out, s);
      } else {
        out.set_data(mx::allocator::malloc(out.nbytes()));
      }
    }
    auto& st = *state_;
    auto check_input = [&copies, &s, &st](const mx::array& x) -> const mx::array {
      if (!st.ensure_row_contiguous || x.flags().row_contiguous) {
        return x;
      }
      copies.push_back(mx::array(x.shape(), x.dtype(), nullptr, {}));
      mx::copy_gpu(x, copies.back(), mx::CopyType::General, s);
      return copies.back();
    };
    std::vector<mx::array> checked;
    checked.reserve(inputs.size());
    for (const auto& in : inputs) {
      checked.push_back(check_input(in));
    }
    auto& d = mx::metal::device(s.device);
    std::call_once(st.resolved, [&] {
      auto lib = d.get_library(st.lib_name, mx::CompileOptions(st.compile_options),
                               [&st] { return mx::metal::utils() + st.source; });
      st.pipeline = d.get_kernel(st.name, lib);
    });
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(st.pipeline);
    int index = 0;
    for (size_t i = 0; i < checked.size(); i++) {
      const auto& in = checked[i];
      const auto& info = st.shape_infos[i];
      enc.set_input_array(in, index++);
      if (in.ndim() > 0) {
        int ndim = in.ndim();
        if (std::get<0>(info)) enc.set_vector_bytes(in.shape(), ndim, index++);
        if (std::get<1>(info)) enc.set_vector_bytes(in.strides(), ndim, index++);
        if (std::get<2>(info)) enc.set_bytes(ndim, index++);
      }
    }
    for (auto& out : outputs) {
      enc.set_output_array(out, index++);
    }
    const auto [tx, ty, tz] = threadgroup_;
    if (size_t(tx) * ty * tz > st.pipeline->maxTotalThreadsPerThreadgroup()) {
      std::ostringstream msg;
      msg << "Thread group size (" << tx * ty * tz << ") is greater than "
          << " the maximum allowed threads per threadgroup ("
          << st.pipeline->maxTotalThreadsPerThreadgroup() << ").";
      throw std::invalid_argument(msg.str());
    }
    const auto [gx, gy, gz] = grid_;
    enc.dispatch_threads(MTL::Size(gx, gy, gz),
                         MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz)));
    enc.add_temporaries(std::move(copies));
  }

 private:
  std::shared_ptr<CachedKernelState> state_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> threadgroup_;
  std::optional<float> init_value_;
};
}  // namespace

struct CachedKernelTable {
  std::mutex mu;
  std::map<std::vector<int64_t>, std::shared_ptr<CachedKernelState>> entries;
};

std::shared_ptr<CachedKernelTable> cached_kernel_table_new() {
  static const bool disabled = std::getenv("IRONMLX_DISABLE_CACHED_KERNELS") != nullptr;
  if (disabled) return nullptr;
  return std::make_shared<CachedKernelTable>();
}

std::optional<std::vector<mx::array>> cached_kernel_dispatch(
    CachedKernelTable& table, const std::vector<int64_t>& key,
    const std::vector<mx::array>& inputs, const std::vector<mx::Shape>& output_shapes,
    const std::vector<mx::Dtype>& output_dtypes, std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup, std::optional<float> init_value,
    mx::StreamOrDevice target) {
  std::shared_ptr<CachedKernelState> state;
  {
    std::lock_guard<std::mutex> lock(table.mu);
    auto it = table.entries.find(key);
    if (it == table.entries.end()) return std::nullopt;
    state = it->second;
  }
  if (state->shape_infos.size() != inputs.size()) return std::nullopt;
  auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu) return std::nullopt;
  return mx::array::make_arrays(
      std::vector<mx::Shape>(output_shapes), std::vector<mx::Dtype>(output_dtypes),
      std::make_shared<CachedCustomKernel>(stream, std::move(state), grid, threadgroup,
                                           init_value),
      inputs);
}

void cached_kernel_remember(CachedKernelTable& table, const std::vector<int64_t>& key,
                            const std::vector<mx::array>& outputs) {
  if (outputs.empty() || !outputs[0].has_primitive()) return;
  auto* kernel = dynamic_cast<mx::fast::CustomKernel*>(&outputs[0].primitive());
  if (kernel == nullptr) return;
  auto [name, source, grid, threadgroup, shape_infos, ensure, init, scalars, precompiled,
        shared_memory, compile_options] = kernel->state();
  (void)grid;
  (void)threadgroup;
  (void)init;
  if (!scalars.empty() || precompiled || shared_memory != 0) return;
  auto state = std::make_shared<CachedKernelState>();
  // MLX's library name for this kernel, so its compiled library is reused.
  char hash[32];
  std::snprintf(hash, sizeof(hash), "%zx", std::hash<std::string>{}(source));
  state->lib_name = name + "_" + hash + "_" + std::to_string(compile_options);
  state->name = std::move(name);
  state->source = std::move(source);
  state->shape_infos = std::move(shape_infos);
  state->ensure_row_contiguous = ensure;
  state->compile_options = compile_options;
  std::lock_guard<std::mutex> lock(table.mu);
  table.entries.emplace(key, std::move(state));
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
struct CachedKernelTable {};
std::shared_ptr<CachedKernelTable> cached_kernel_table_new() { return nullptr; }
std::optional<std::vector<mlx::core::array>> cached_kernel_dispatch(
    CachedKernelTable&, const std::vector<int64_t>&, const std::vector<mlx::core::array>&,
    const std::vector<mlx::core::Shape>&, const std::vector<mlx::core::Dtype>&,
    std::tuple<int, int, int>, std::tuple<int, int, int>, std::optional<float>,
    mlx::core::StreamOrDevice) {
  return std::nullopt;
}
void cached_kernel_remember(CachedKernelTable&, const std::vector<int64_t>&,
                            const std::vector<mlx::core::array>&) {}
}  // namespace cxx_mlx
#endif
