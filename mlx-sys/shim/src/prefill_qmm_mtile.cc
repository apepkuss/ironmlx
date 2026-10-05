#include "cxx_mlx_shim/prefill_qmm_mtile.h"
#include "cxx_mlx_shim/experiment_config.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

class ExperimentalPrefillQMMMtile final : public mx::UnaryPrimitive {
 public:
  ExperimentalPrefillQMMMtile(mx::Stream stream, std::string library)
      : UnaryPrimitive(stream), library_(std::move(library)) {}
  const char* name() const override { return "ExperimentalPrefillQMMMtile"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Experimental QMM M-tile requires GPU");
  }
  void eval_gpu(const std::vector<mx::array>& in, mx::array& out) override {
    const bool down = in[0].shape(-1) == 17408;
    // Lazy views acquire their final strides/flags during evaluation. Recheck
    // the realized layouts, rather than trusting construction-time metadata.
    for (const auto& a : in) {
      if (!a.flags().row_contiguous || a.strides(-1) != 1 ||
          a.strides(-2) != a.shape(-1)) {
        mx::QuantizedMatmul native(stream(), 64, 4, mx::QuantizationMode::Affine,
                                   true, false, false);
        native.eval_gpu(in, out);
        static std::once_flag fallback_witness[2];
        std::call_once(fallback_witness[down ? 1 : 0], [down] {
          std::fprintf(stderr, "[experimental] QMM M-tile %s realized-layout native fallback\n",
                       down ? "down" : "gate-up");
        });
        return;
      }
    }
    constexpr int m = 2048, bm = 128;
    const int n = down ? 5120 : 34816;
    const int k = down ? 17408 : 5120;
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(stream().device);
    auto lib = d.get_library("ironmlx_experimental_prefill_qmm_mtile", library_);
    auto kernel = d.get_kernel("ironmlx_qmm_bm128_aligned", lib);
    // GPU validation may add private instrumentation threadgroup storage.
    // The pinned non-instrumented operator reports 9216 bytes; admission here
    // must use actual device limits instead of rejecting validation pipelines.
    if (kernel->maxTotalThreadsPerThreadgroup() < 256 ||
        kernel->staticThreadgroupMemoryLength() > d.mtl_device()->maxThreadgroupMemoryLength())
      throw std::runtime_error("Experimental QMM M-tile exceeds device resource limits");
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(in[1], 0);
    enc.set_input_array(in[2], 1);
    enc.set_input_array(in[3], 2);
    enc.set_input_array(in[0], 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(k, 5);
    enc.set_bytes(n, 6);
    enc.set_bytes(m, 7);
    enc.dispatch_threadgroups(MTL::Size(n/64, m/bm, 1), MTL::Size(32, 2, 4));
    static std::once_flag witness[2];
    std::call_once(witness[down ? 1 : 0], [kernel, down] {
      std::fprintf(stderr,
          "[experimental] B1 M2048 BF16 affine4 %s QMM BM128 active "
          "(static_threadgroup_bytes=%lu, max_threads=%lu)\n",
          down ? "down" : "gate-up",
          static_cast<unsigned long>(kernel->staticThreadgroupMemoryLength()),
          static_cast<unsigned long>(kernel->maxTotalThreadsPerThreadgroup()));
    });
  }
 private:
  std::string library_;
};
}  // namespace

std::optional<mlx::core::array> experimental_prefill_qmm_mtile(
    const mx::array& x, const mx::array& w, const mx::array& scales,
    const std::optional<mx::array>& biases, bool transpose,
    std::optional<int> group_size, std::optional<int> bits,
    const std::string& mode, mx::StreamOrDevice target) {
  const std::string& library = prefill_qmm_mtile_library();
  if (library.empty() || !transpose || group_size != 64 || bits != 4 ||
      mode != "affine" || !biases) return {};
  const bool gate = (x.ndim() == 2 && x.shape() == mx::Shape{2048, 5120}) ||
                    (x.ndim() == 3 && x.shape() == mx::Shape{1, 2048, 5120});
  const bool down = (x.ndim() == 2 && x.shape() == mx::Shape{2048, 17408}) ||
                    (x.ndim() == 3 && x.shape() == mx::Shape{1, 2048, 17408});
  const int n = down ? 5120 : 34816;
  const int k = down ? 17408 : 5120;
  if ((!gate && !down) ||
      x.dtype() != mx::bfloat16 || w.dtype() != mx::uint32 ||
      scales.dtype() != mx::bfloat16 || biases->dtype() != mx::bfloat16 ||
      w.shape() != mx::Shape{n, k / 8} || scales.shape() != mx::Shape{n, k / 64} ||
      biases->shape() != scales.shape()) return {};
  for (const auto* a : {&x, &w, &scales, &*biases})
    if (!a->flags().row_contiguous) return {};
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  auto shape = x.shape();
  shape.back() = n;
  return mx::array(std::move(shape), mx::bfloat16,
      std::make_shared<ExperimentalPrefillQMMMtile>(stream, library), {x, w, scales, *biases});
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
std::optional<mlx::core::array> experimental_prefill_qmm_mtile(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    const std::optional<mlx::core::array>&, bool, std::optional<int>,
    std::optional<int>, const std::string&, mlx::core::StreamOrDevice) {
  if (!prefill_qmm_mtile_library().empty())
    throw std::runtime_error("QMM M-tile experiment requires Metal C++ headers");
  return {};
}
}  // namespace cxx_mlx
#endif
