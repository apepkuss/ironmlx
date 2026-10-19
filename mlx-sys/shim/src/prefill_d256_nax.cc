#include "cxx_mlx_shim/prefill_d256_nax.h"
#include "cxx_mlx_shim/experiment_config.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/primitives.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

// Host dispatch follows Apple's MLX attention ABI (MIT, Copyright Apple Inc.).
// Geometry (batch, heads, GQA factor, rows, keys) comes from the arrays; the
// admission rules below bound which inputs reach it.
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
    const int batch = q.shape(0), heads = q.shape(1), qlen = q.shape(2), klen = k.shape(2);
    const int gqa = heads / k.shape(1);
    // Tile of the library kernel: BQ 32 query rows (two row groups), BK 32.
    constexpr int wm = 2, bq = 16 * wm;
    const bool aq = qlen % bq == 0, ak = klen % 32 == 0;
    const bool mask = false, causal = true, sinks = false;
    mx::metal::MTLFCList constants = {
        {&aq, MTL::DataTypeBool, 200}, {&ak, MTL::DataTypeBool, 201},
        {&mask, MTL::DataTypeBool, 300}, {&causal, MTL::DataTypeBool, 301},
        {&sinks, MTL::DataTypeBool, 302}};
    const auto hash = std::string("ironmlx_probe_dsplit_bf16_") +
                      (aq ? "alignedQ_" : "raggedQ_") + (ak ? "alignedK" : "raggedK");
    auto kernel = device.get_kernel("ironmlx_probe_dsplit_bf16", lib, hash, constants);
    if (kernel->maxTotalThreadsPerThreadgroup() < unsigned(64 * wm))
      throw std::runtime_error("Experimental D256 NAX tile is unsupported");
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    mlx::steel::AttnParams params{
        batch, heads, 256, qlen, klen, gqa, scale_,
        (qlen + bq - 1) / bq, (klen + 31) / 32, qlen / bq, klen / 32,
        qlen % bq, klen % 32, klen - qlen,
        {q.strides(0), q.strides(1), q.strides(2)},
        {k.strides(0), k.strides(1), k.strides(2)},
        {v.strides(0), v.strides(1), v.strides(2)},
        {out.strides(0), out.strides(1), out.strides(2)}};
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(MTL::Size(params.NQ, heads, batch), MTL::Size(32, wm, 2));
    static std::once_flag message;
    std::call_once(message, [] {
      std::fputs("[experimental] BF16 D256 NAX prefill attention active\n", stderr);
    });
  }
 private:
  std::string library_;
  float scale_;
};

struct SplitKParams {
  int N, R, rows, KB, NB, kv_heads, causal;
  float scale2;
  int64_t Q_batch;
  int64_t K_strides[3];
  int64_t V_strides[3];
};

// Keys per split-K block. A row's result must not depend on how many rows
// share its call, so the block size follows the key count only across the
// verify grouping's boundaries (no query group straddles 16384 keys, and a
// single decode row sees its own key count).
int decode_block_keys(int keys) {
  return keys < 16384 ? 512 : 1024;
}

class ExperimentalDecodeD256NAX final : public mx::UnaryPrimitive {
 public:
  ExperimentalDecodeD256NAX(mx::Stream stream, std::string library, float scale, bool causal)
      : UnaryPrimitive(stream), library_(std::move(library)), scale_(scale), causal_(causal) {}
  const char* name() const override { return "ExperimentalDecodeD256NAX"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Experimental D256 NAX requires GPU");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto& encoder = mx::metal::get_command_encoder(stream());
    mx::array q = inputs[0];
    if (!q.flags().row_contiguous) {
      q = mx::contiguous_copy_gpu(q, stream());
      encoder.add_temporary(q);
    }
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto lib = device.get_library("ironmlx_experimental_prefill_d256", library_);
    const int batch = q.shape(0), heads = q.shape(1), qlen = q.shape(2), klen = k.shape(2);
    const int kv_heads = k.shape(1);
    const int rows = heads / kv_heads * qlen;
    const int block = decode_block_keys(klen);
    const int nb = (klen + block - 1) / block;
    const int wm = rows <= 16 ? 1 : rows <= 32 ? 2 : 4;
    SplitKParams params{klen, qlen, rows, block, nb, kv_heads, causal_ ? 1 : 0,
                        scale_ * 1.44269504089f, q.strides(0),
                        {k.strides(0), k.strides(1), k.strides(2)},
                        {v.strides(0), v.strides(1), v.strides(2)}};
    const size_t parts = size_t(batch) * kv_heads * nb * rows;
    mx::array opart({int(parts), 256}, mx::float32, nullptr, {});
    mx::array mpart({int(parts)}, mx::float32, nullptr, {});
    mx::array spart({int(parts)}, mx::float32, nullptr, {});
    opart.set_data(mx::allocator::malloc(opart.nbytes()));
    mpart.set_data(mx::allocator::malloc(mpart.nbytes()));
    spart.set_data(mx::allocator::malloc(spart.nbytes()));
    encoder.add_temporary(opart);
    encoder.add_temporary(mpart);
    encoder.add_temporary(spart);

    const auto pass1 = std::string("ironmlx_splitk_d256_pass1_wm") + std::to_string(wm);
    auto kernel1 = device.get_kernel(pass1, lib);
    if (kernel1->maxTotalThreadsPerThreadgroup() < unsigned(64 * wm))
      throw std::runtime_error("Experimental D256 NAX split-K tile is unsupported");
    encoder.set_compute_pipeline_state(kernel1);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(opart, 3);
    encoder.set_output_array(mpart, 4);
    encoder.set_output_array(spart, 5);
    encoder.set_bytes(params, 6);
    encoder.dispatch_threadgroups(MTL::Size(nb, kv_heads, batch), MTL::Size(32, wm, 2));

    auto kernel2 = device.get_kernel("ironmlx_splitk_d256_pass2", lib);
    encoder.set_compute_pipeline_state(kernel2);
    encoder.set_input_array(opart, 0);
    encoder.set_input_array(mpart, 1);
    encoder.set_input_array(spart, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(MTL::Size(rows, kv_heads, batch), MTL::Size(256, 1, 1));
    static std::once_flag message;
    std::call_once(message, [] {
      std::fputs("[experimental] BF16 D256 NAX split-K decode attention active\n", stderr);
    });
  }
 private:
  std::string library_;
  float scale_;
  bool causal_;
};
}  // namespace

std::optional<mlx::core::array> experimental_prefill_d256_nax(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, mx::StreamOrDevice target) {
  const std::string& library = prefill_d256_nax_library();
  if (library.empty()) return {};
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      q.shape(3) != 256 || k.shape(3) != 256 || v.shape() != k.shape() ||
      q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
      q.strides(3) != 1 || k.strides(3) != 1 || v.strides(3) != 1 ||
      !std::isfinite(scale) || scale != 0.0625f) return {};
  // Tested dense 27B B1 rule: 24 query / 4 KV heads, 2048-row chunks.
  const bool dense_b1 = q.shape(0) == 1 && k.shape(0) == 1 &&
      q.shape(1) == 24 && k.shape(1) == 4 &&
      q.shape(2) == 2048 && k.shape(2) >= 2048 && k.shape(2) <= 40960;
  // Long causal chunks of any GQA layout once the keys reach the configured
  // minimum: the chunk ends at the last key (causal offset keys - rows).
  const int32_t min_keys = prefill_d256_nax_min_keys();
  const bool long_chunk = min_keys > 0 && q.shape(0) == k.shape(0) &&
      k.shape(1) > 0 && q.shape(1) % k.shape(1) == 0 &&
      q.shape(2) >= 64 && k.shape(2) >= q.shape(2) && k.shape(2) >= min_keys;
  if (!dense_b1 && !long_chunk) return {};
  for (const auto* a : {&q, &k, &v}) {
    for (auto stride : a->strides()) if (stride <= 0) return {};
  }
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  return mx::array(q.shape(), q.dtype(),
      std::make_shared<ExperimentalPrefillD256NAX>(stream, library, scale), {q, k, v});
}

int32_t decode_d256_nax_active_min_keys() {
  const int32_t min_keys = decode_d256_nax_min_keys();
  if (min_keys <= 0 || prefill_d256_nax_library().empty()) return 0;
  return mx::metal::is_nax_available() ? min_keys : 0;
}

std::optional<mlx::core::array> experimental_decode_d256_nax(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, bool causal, mx::StreamOrDevice target) {
  const int32_t min_keys = decode_d256_nax_min_keys();
  if (min_keys <= 0) return {};
  const std::string& library = prefill_d256_nax_library();
  if (library.empty()) return {};
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
      q.shape(3) != 256 || k.shape(3) != 256 || v.shape() != k.shape() ||
      q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
      q.strides(3) != 1 || k.strides(3) != 1 || v.strides(3) != 1 ||
      !std::isfinite(scale)) return {};
  // Short query blocks over long caches: every row of a causal block sees at
  // least min_keys keys, and the GQA heads x rows of a KV head fit 64 rows.
  if (q.shape(0) != k.shape(0) || k.shape(1) <= 0 || q.shape(1) % k.shape(1) != 0 ||
      q.shape(2) < 1 || q.shape(2) > 8 || q.shape(1) / k.shape(1) * q.shape(2) > 64 ||
      k.shape(2) < q.shape(2) || k.shape(2) - q.shape(2) + 1 < min_keys) return {};
  for (const auto* a : {&k, &v}) {
    for (auto stride : a->strides()) if (stride <= 0) return {};
    if (a->strides(2) > INT32_MAX / 64) return {};
  }
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  return mx::array(q.shape(), q.dtype(),
      std::make_shared<ExperimentalDecodeD256NAX>(stream, library, scale, causal), {q, k, v});
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
std::optional<mlx::core::array> experimental_decode_d256_nax(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    float, bool, mlx::core::StreamOrDevice) {
  return {};
}
int32_t decode_d256_nax_active_min_keys() { return 0; }
}  // namespace cxx_mlx
#endif
