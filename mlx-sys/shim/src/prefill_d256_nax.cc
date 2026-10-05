#include "cxx_mlx_shim/prefill_d256_nax.h"
#include "cxx_mlx_shim/experiment_config.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

// Host dispatch follows Apple's MLX attention ABI (MIT, Copyright Apple Inc.).
// Deliberately bounded to the tested Qwen 27B B1 full-attention morphology.
class ExperimentalPrefillD256NAX final : public mx::UnaryPrimitive {
 public:
  ExperimentalPrefillD256NAX(mx::Stream stream, std::string library, float scale)
      : UnaryPrimitive(stream), library_(std::move(library)), scale_(scale) {}
  const char* name() const override { return "ExperimentalPrefillD256NAX"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Experimental D256 NAX requires GPU");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto lib = device.get_library("ironmlx_experimental_prefill_d256", library_);
    const int qlen = q.shape(2), klen = k.shape(2);
    const bool aq = true, ak = klen % 32 == 0;
    const bool mask = false, causal = true, sinks = false;
    mx::metal::MTLFCList constants = {
        {&aq, MTL::DataTypeBool, 200}, {&ak, MTL::DataTypeBool, 201},
        {&mask, MTL::DataTypeBool, 300}, {&causal, MTL::DataTypeBool, 301},
        {&sinks, MTL::DataTypeBool, 302}};
    const auto hash = std::string("ironmlx_probe_dsplit_bf16_alignedQ_") +
                      (ak ? "alignedK" : "raggedK");
    auto kernel = device.get_kernel("ironmlx_probe_dsplit_bf16", lib, hash, constants);
    if (kernel->maxTotalThreadsPerThreadgroup() < 256)
      throw std::runtime_error("Experimental D256 NAX tile is unsupported");
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    mlx::steel::AttnParams params{
        1, 24, 256, qlen, klen, 6, scale_,
        qlen / 64, (klen + 31) / 32, qlen / 64, klen / 32,
        0, klen % 32, klen - qlen,
        {q.strides(0), q.strides(1), q.strides(2)},
        {k.strides(0), k.strides(1), k.strides(2)},
        {v.strides(0), v.strides(1), v.strides(2)},
        {out.strides(0), out.strides(1), out.strides(2)}};
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(MTL::Size(params.NQ, 24, 1), MTL::Size(32, 4, 2));
    static std::once_flag message;
    std::call_once(message, [] {
      std::fputs("[experimental] B1 2048-row BF16 D256 NAX attention active\n", stderr);
    });
  }
 private:
  std::string library_;
  float scale_;
};
}  // namespace

std::optional<mlx::core::array> experimental_prefill_d256_nax(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, mx::StreamOrDevice target) {
  const std::string& library = prefill_d256_nax_library();
  if (library.empty()) return {};
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      q.shape(0) != 1 || k.shape(0) != 1 || v.shape(0) != 1 ||
      q.shape(1) != 24 || k.shape(1) != 4 || v.shape(1) != 4 ||
      q.shape(2) != 2048 || k.shape(2) < 2048 || k.shape(2) > 40960 ||
      q.shape(3) != 256 || k.shape(3) != 256 || v.shape() != k.shape() ||
      q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
      q.strides(3) != 1 || k.strides(3) != 1 || v.strides(3) != 1 ||
      !std::isfinite(scale) || scale != 0.0625f) return {};
  for (const auto* a : {&q, &k, &v}) {
    for (auto stride : a->strides()) if (stride <= 0) return {};
  }
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  return mx::array(q.shape(), q.dtype(),
      std::make_shared<ExperimentalPrefillD256NAX>(stream, library, scale), {q, k, v});
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
std::optional<mlx::core::array> experimental_prefill_d256_nax(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    float, mlx::core::StreamOrDevice) {
  if (!prefill_d256_nax_library().empty())
    throw std::runtime_error("D256 NAX experiment unavailable: Metal C++ headers missing at build time");
  return {};
}
}  // namespace cxx_mlx
#endif
