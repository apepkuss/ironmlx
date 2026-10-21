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
  int N, R, rows, KB, NB, kv_heads, causal, parts, lo, tail_from, TL;
  float scale2;
  int64_t Q_batch;
  int64_t K_strides[3];
  int64_t V_strides[3];
};

// Keys per split-K block. A row's result must not depend on how many rows
// share its call, so the block size follows the key count only across the
// verify grouping's boundaries (no query group spans two block sizes, and a
// single decode row sees its own key count). Past 64K keys the block is the
// smallest one keeping a row within 40 blocks: the second pass merges a
// row's blocks serially and their partials grow with the count, while larger
// blocks leave fewer threadgroups. Serial verify-shape calls (M5 Max, 8 rows)
// were fastest there at every measured length from 69632 to 262144 keys
// (1024-key blocks: 389 GB/s at 256K keys; 8192: 436 GB/s).
int decode_block_keys(int keys) {
  if (keys < 16384) return 512;
  if (keys <= 65536) return 1024;
  int block = 2048;
  while ((keys + block - 1) / block > 40) block *= 2;
  return block;
}

// No window start: every row sees keys from 0.
constexpr int kNoWindowStart = -(1 << 30);

// Split-K NAX attention of a short query block (head dim 128 or 256). `lo`
// is the sliding-window start of query row 0 (row r sees keys >= lo + r);
// `block` fixes the keys per split-K block (0: by key count, as decode).
class ExperimentalDecodeD256NAX final : public mx::UnaryPrimitive {
 public:
  ExperimentalDecodeD256NAX(mx::Stream stream, std::string library, float scale, bool causal,
                            int lo = kNoWindowStart, int block = 0)
      : UnaryPrimitive(stream), library_(std::move(library)), scale_(scale), causal_(causal),
        lo_(lo), block_(block) {}
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
    const int bd = q.shape(3);
    const int block = block_ > 0 ? block_ : decode_block_keys(klen);
    const int nb = (klen + block - 1) / block;
    // 16 rows per threadgroup: wider blocks run as row parts, side by side
    // over the same keys (less synchronization per key than wider tiles).
    const int wm = 1;
    const int row_parts = (rows + 16 * wm - 1) / (16 * wm);
    SplitKParams params{klen, qlen, rows, block, nb, kv_heads, causal_ ? 1 : 0, row_parts, lo_, 0, 0,
                        scale_ * 1.44269504089f, q.strides(0),
                        {k.strides(0), k.strides(1), k.strides(2)},
                        {v.strides(0), v.strides(1), v.strides(2)}};
    const size_t parts = size_t(batch) * kv_heads * nb * rows;
    mx::array opart({int(parts), bd}, mx::float32, nullptr, {});
    mx::array mpart({int(parts)}, mx::float32, nullptr, {});
    mx::array spart({int(parts)}, mx::float32, nullptr, {});
    opart.set_data(mx::allocator::malloc(opart.nbytes()));
    mpart.set_data(mx::allocator::malloc(mpart.nbytes()));
    spart.set_data(mx::allocator::malloc(spart.nbytes()));
    encoder.add_temporary(opart);
    encoder.add_temporary(mpart);
    encoder.add_temporary(spart);

    const auto prefix = std::string("ironmlx_splitk_d") + std::to_string(bd);
    const auto pass1 = prefix + "_pass1_wm" + std::to_string(wm);
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
    // Tree-only inputs (unused by this instantiation).
    encoder.set_input_array(q, 7);
    encoder.set_input_array(q, 8);
    encoder.set_input_array(q, 9);
    encoder.dispatch_threadgroups(MTL::Size(nb * row_parts, kv_heads, batch), MTL::Size(32, wm, 2));

    auto kernel2 = device.get_kernel(prefix + "_pass2", lib);
    encoder.set_compute_pipeline_state(kernel2);
    encoder.set_input_array(opart, 0);
    encoder.set_input_array(mpart, 1);
    encoder.set_input_array(spart, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.set_input_array(spart, 5);
    encoder.dispatch_threadgroups(MTL::Size(rows, kv_heads, batch), MTL::Size(bd, 1, 1));
    if (bd == 256) {
      static std::once_flag message;
      std::call_once(message, [] {
        std::fputs("[experimental] BF16 D256 NAX split-K decode attention active\n", stderr);
      });
    }
  }
 private:
  std::string library_;
  float scale_;
  bool causal_;
  int lo_;
  int block_;
};

// Split-K D256 NAX attention of a B1 flat draft tree (see the TREE kernels):
// inputs q [1, Hkv, slots * gqa, 256] node-major (two slots per row part, both
// on one root-to-leaf path), k / v the cache, tk / tv [1, Hkv, parts, TL, 256]
// each part's tail from `tail_from`, lims [slots] int32 per-slot key limits.
class ExperimentalTreeD256NAX final : public mx::UnaryPrimitive {
 public:
  ExperimentalTreeD256NAX(mx::Stream stream, std::string library, float scale, int gqa,
                          int keys, int block, int tail_from, int tail_len)
      : UnaryPrimitive(stream), library_(std::move(library)), scale_(scale), gqa_(gqa),
        keys_(keys), block_(block), tail_from_(tail_from), tail_len_(tail_len) {}
  const char* name() const override { return "ExperimentalTreeD256NAX"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Experimental D256 NAX tree attention requires GPU");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto& encoder = mx::metal::get_command_encoder(stream());
    std::vector<mx::array> ins = inputs;
    for (int i : {0, 3, 4, 5}) {
      if (!ins[i].flags().row_contiguous) {
        ins[i] = mx::contiguous_copy_gpu(ins[i], stream());
        encoder.add_temporary(ins[i]);
      }
    }
    const auto& q = ins[0];
    const auto& k = ins[1];
    const auto& v = ins[2];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto lib = device.get_library("ironmlx_experimental_prefill_d256", library_);
    const int kv_heads = k.shape(1);
    const int rows = q.shape(2);
    const int parts = rows / 16;
    const int nb = (keys_ + block_ - 1) / block_;
    SplitKParams params{keys_, gqa_, rows, block_, nb, kv_heads, 0, parts, -(1 << 30),
                        tail_from_, tail_len_, scale_ * 1.44269504089f, q.strides(0),
                        {k.strides(0), k.strides(1), k.strides(2)},
                        {v.strides(0), v.strides(1), v.strides(2)}};
    const size_t count = size_t(kv_heads) * nb * rows;
    mx::array opart({int(count), 256}, mx::float32, nullptr, {});
    mx::array mpart({int(count)}, mx::float32, nullptr, {});
    mx::array spart({int(count)}, mx::float32, nullptr, {});
    opart.set_data(mx::allocator::malloc(opart.nbytes()));
    mpart.set_data(mx::allocator::malloc(mpart.nbytes()));
    spart.set_data(mx::allocator::malloc(spart.nbytes()));
    encoder.add_temporary(opart);
    encoder.add_temporary(mpart);
    encoder.add_temporary(spart);
    auto kernel1 = device.get_kernel("ironmlx_splitk_d256_tree_pass1_wm1", lib);
    encoder.set_compute_pipeline_state(kernel1);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(opart, 3);
    encoder.set_output_array(mpart, 4);
    encoder.set_output_array(spart, 5);
    encoder.set_bytes(params, 6);
    encoder.set_input_array(ins[3], 7);
    encoder.set_input_array(ins[4], 8);
    encoder.set_input_array(ins[5], 9);
    encoder.dispatch_threadgroups(MTL::Size(nb * parts, kv_heads, 1), MTL::Size(32, 1, 2));
    auto kernel2 = device.get_kernel("ironmlx_splitk_d256_tree_pass2", lib);
    encoder.set_compute_pipeline_state(kernel2);
    encoder.set_input_array(opart, 0);
    encoder.set_input_array(mpart, 1);
    encoder.set_input_array(spart, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(params, 4);
    encoder.set_input_array(ins[5], 5);
    encoder.dispatch_threadgroups(MTL::Size(rows, kv_heads, 1), MTL::Size(256, 1, 1));
  }
 private:
  std::string library_;
  float scale_;
  int gqa_, keys_, block_, tail_from_, tail_len_;
};
}  // namespace

std::optional<mlx::core::array> experimental_tree_d256_nax(
    const mx::array& q, const mx::array& k, const mx::array& v, const mx::array& tk,
    const mx::array& tv, const mx::array& lims, float scale, int32_t gqa, int32_t keys,
    int32_t tail_from, mx::StreamOrDevice target) {
  if (decode_d256_nax_min_keys() <= 0) return {};
  const std::string& library = prefill_d256_nax_library();
  if (library.empty()) return {};
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || tk.ndim() != 5 || tv.ndim() != 5 ||
      q.shape(0) != 1 || k.shape(0) != 1 || q.shape(3) != 256 || k.shape(3) != 256 ||
      v.shape() != k.shape() || q.shape(1) != k.shape(1) || gqa <= 0 || q.shape(2) % 16 != 0 ||
      (q.shape(2) / gqa) * gqa != q.shape(2) || tk.shape() != tv.shape() ||
      tk.shape(1) != k.shape(1) || tk.shape(2) != q.shape(2) / 16 || tk.shape(4) != 256 ||
      lims.ndim() != 1 || lims.shape(0) != q.shape(2) / gqa || lims.dtype() != mx::int32 ||
      q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
      tk.dtype() != mx::bfloat16 || tv.dtype() != mx::bfloat16 || k.strides(3) != 1 ||
      v.strides(3) != 1 || tail_from % 32 != 0 || tail_from < 0 || keys > k.shape(2) ||
      keys - tail_from > tk.shape(3) || !std::isfinite(scale)) return {};
  for (const auto* a : {&k, &v}) {
    for (auto stride : a->strides()) if (stride <= 0) return {};
    if (a->strides(2) > INT32_MAX / 64) return {};
  }
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  return mx::array(q.shape(), q.dtype(),
      std::make_shared<ExperimentalTreeD256NAX>(
          stream, library, scale, gqa, keys, decode_block_keys(keys), tail_from,
          int(tk.shape(3))),
      {q, k, v, tk, tv, lims});
}

int32_t decode_d256_nax_block_keys(int32_t keys) { return decode_block_keys(keys); }

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

std::optional<mlx::core::array> experimental_window_attention_nax(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, int32_t window_start, mx::StreamOrDevice target) {
  const std::string& library = prefill_d256_nax_library();
  if (library.empty()) return {};
  // DFlash2 drafter blocks: B1, head dim 128, every query row sees every key
  // from its window start (row 0 at window_start, row r at window_start + r).
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || q.shape(0) != 1 ||
      k.shape(0) != 1 || q.shape(3) != 128 || k.shape(3) != 128 || v.shape() != k.shape() ||
      q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
      q.strides(3) != 1 || k.strides(3) != 1 || v.strides(3) != 1 || !std::isfinite(scale) ||
      k.shape(1) <= 0 || q.shape(1) % k.shape(1) != 0 || q.shape(2) < 1 || q.shape(2) > 16 ||
      q.shape(1) / k.shape(1) * q.shape(2) > 64 || k.shape(2) < q.shape(2) ||
      window_start >= k.shape(2) - q.shape(2) + 1) return {};
  for (const auto* a : {&k, &v}) {
    for (auto stride : a->strides()) if (stride <= 0) return {};
    if (a->strides(2) > INT32_MAX / 64) return {};
  }
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu || !mx::metal::is_nax_available()) return {};
  return mx::array(q.shape(), q.dtype(),
      std::make_shared<ExperimentalDecodeD256NAX>(
          stream, library, scale, false, std::max(window_start, kNoWindowStart), 256),
      {q, k, v});
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
std::optional<mlx::core::array> experimental_tree_d256_nax(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&, float,
    int32_t, int32_t, int32_t, mlx::core::StreamOrDevice) {
  return {};
}
// Same rule as the Metal build's decode_block_keys.
int32_t decode_d256_nax_block_keys(int32_t keys) {
  if (keys < 16384) return 512;
  if (keys <= 65536) return 1024;
  int32_t block = 2048;
  while ((keys + block - 1) / block > 40) block *= 2;
  return block;
}
std::optional<mlx::core::array> experimental_window_attention_nax(
    const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
    float, int32_t, mlx::core::StreamOrDevice) {
  return {};
}
}  // namespace cxx_mlx
#endif
