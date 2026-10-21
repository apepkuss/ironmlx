#include "cxx_mlx_shim/fast.h"
#include "cxx_mlx_shim/shim_helpers.h"
#include "cxx_mlx_shim/cached_kernel.h"
#include "cxx_mlx_shim/prefill_d256_nax.h"
#include "cxx_mlx_shim/prefill_masked_softmax.h"

#include <optional>
#include <stdexcept>
#include <string>

#include "mlx/fast.h"

// Pull in the cxxbridge-generated header so TemplateArgC is fully defined
// for this translation unit (mirrors how compile.cc includes compile.rs.h).
#include "mlx-sys/src/bridge/fast.rs.h"

namespace cxx_mlx {

namespace {

// pointer → optional<array>. mlx::array copy is cheap (refcount on array_desc_).
inline std::optional<mlx::core::array> opt_arr(const MlxArray* p) {
  return p ? std::optional<mlx::core::array>(*p) : std::nullopt;
}

inline std::optional<float> opt_f(bool has, float v) {
  return has ? std::optional<float>(v) : std::nullopt;
}

}  // namespace

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x, const MlxArray* weight, float eps,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rms_norm(x, opt_arr(weight), eps, target));
}

std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x, const MlxArray* weight, const MlxArray* bias, float eps,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::layer_norm(x, opt_arr(weight), opt_arr(bias), eps, target));
}

std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, int32_t offset,
    const MlxArray* freqs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs), target));
}

std::unique_ptr<MlxArray> fast_rope_with_array_offset(
    const MlxArray& x, int32_t dims, bool traditional,
    bool has_base, float base, float scale, const MlxArray& offset,
    const MlxArray* freqs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  return std::make_unique<MlxArray>(
      mlx::core::fast::rope(
          x, dims, traditional, opt_f(has_base, base), scale, offset,
          opt_arr(freqs), target));
}

std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& queries, const MlxArray& keys, const MlxArray& values,
    float scale, rust::Str mask_mode,
    const MlxArray* mask_arr, const MlxArray* sinks,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  if ((mask_mode == "causal" || mask_mode == "") && mask_arr == nullptr && sinks == nullptr) {
    if (auto candidate = experimental_decode_d256_nax(
            queries, keys, values, scale, mask_mode == "causal", target)) {
      return std::make_unique<MlxArray>(std::move(*candidate));
    }
  }
  if (mask_mode == "causal" && mask_arr == nullptr && sinks == nullptr) {
    if (auto candidate = experimental_prefill_d256_nax(queries, keys, values, scale, target)) {
      return std::make_unique<MlxArray>(std::move(*candidate));
    }
    if (auto candidate = experimental_masked_causal_sdpa(queries, keys, values, scale, target)) {
      return std::make_unique<MlxArray>(std::move(*candidate));
    }
  }
  return std::make_unique<MlxArray>(
      mlx::core::fast::scaled_dot_product_attention(
          queries, keys, values, scale,
          std::string(mask_mode),
          opt_arr(mask_arr),
          opt_arr(sinks),
          false,
          target));
}

std::unique_ptr<MlxArray> fast_window_attention(
    const MlxArray& queries, const MlxArray& keys, const MlxArray& values,
    float scale, int32_t window_start,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  if (auto candidate = experimental_window_attention_nax(
          queries, keys, values, scale, window_start, target)) {
    return std::make_unique<MlxArray>(std::move(*candidate));
  }
  return nullptr;
}

std::unique_ptr<MlxArray> fast_tree_attention(
    const MlxArray& queries, const MlxArray& keys, const MlxArray& values,
    const MlxArray& tail_keys, const MlxArray& tail_values, const MlxArray& limits,
    float scale, int32_t gqa, int32_t key_count, int32_t tail_from,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_target, is_device_only, device_type, stream_index);
  if (auto candidate = experimental_tree_d256_nax(
          queries, keys, values, tail_keys, tail_values, limits, scale, gqa, key_count,
          tail_from, target)) {
    return std::make_unique<MlxArray>(std::move(*candidate));
  }
  return nullptr;
}

int32_t fast_decode_block_keys(int32_t keys) { return decode_d256_nax_block_keys(keys); }

// === P3a ShapesVec ===

std::unique_ptr<ShapesVec> shapes_vec_new() {
  return std::make_unique<ShapesVec>();
}

void shapes_vec_push(ShapesVec& v, rust::Slice<const int32_t> shape) {
  v.shapes.emplace_back(shape.begin(), shape.end());
}

size_t shapes_vec_count(const ShapesVec& v) {
  return v.shapes.size();
}

// === P3a metal_kernel_build ===

std::unique_ptr<MetalKernelInner> metal_kernel_build(
    rust::Str name,
    rust::Slice<const rust::String> input_names,
    rust::Slice<const rust::String> output_names,
    rust::Str source,
    rust::Str header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  std::vector<std::string> in_names;
  in_names.reserve(input_names.size());
  for (const auto& s : input_names) {
    in_names.emplace_back(s);
  }
  std::vector<std::string> out_names;
  out_names.reserve(output_names.size());
  for (const auto& s : output_names) {
    out_names.emplace_back(s);
  }
  auto kernel = mlx::core::fast::metal_kernel(
      std::string(name),
      in_names,
      out_names,
      std::string(source),
      std::string(header),
      ensure_row_contiguous,
      atomic_outputs);
  auto inner = std::make_unique<MetalKernelInner>();
  inner->fn = std::move(kernel);
  inner->cache = cached_kernel_table_new();
  return inner;
}

// === P3a metal_kernel_dispatch ===

std::unique_ptr<ArrayVec> metal_kernel_dispatch(
    const MetalKernelInner& kernel,
    const ArrayVec& inputs,
    const ShapesVec& output_shapes,
    rust::Slice<const uint8_t> output_dtypes,
    int32_t gx, int32_t gy, int32_t gz,
    int32_t tx, int32_t ty, int32_t tz,
    rust::Slice<const TemplateArgC> template_args,
    bool has_init, float init_value,
    bool verbose,
    bool has_stream, bool dev_only, uint8_t dev_type, int32_t stream_idx) {
  // 1. inputs vector copy from ArrayVec.inner (refcount share, cheap)
  std::vector<mlx::core::array> ins(inputs.inner.begin(), inputs.inner.end());

  // 2. output dtypes
  std::vector<mlx::core::Dtype> out_dtypes;
  out_dtypes.reserve(output_dtypes.size());
  for (auto repr : output_dtypes) {
    out_dtypes.push_back(cxx_mlx::helpers::dtype_from_repr(repr));
  }

  // 3. template args: convert TemplateArgC to mlx variant
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> tmpl;
  tmpl.reserve(template_args.size());
  for (const auto& t : template_args) {
    std::string n(t.name);
    if (t.kind == 0) {
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{static_cast<int>(t.int_val)});
    } else if (t.kind == 1) {
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{t.bool_val});
    } else if (t.kind == 2) {
      auto dt = cxx_mlx::helpers::dtype_from_repr(t.dtype_val);
      tmpl.emplace_back(std::move(n), mlx::core::fast::TemplateArg{dt});
    } else {
      throw std::runtime_error("metal_kernel_dispatch: unknown TemplateArgC kind");
    }
  }

  // 4. init_value
  std::optional<float> init = has_init ? std::optional<float>(init_value) : std::nullopt;

  // 5. stream
  auto target = cxx_mlx::helpers::decode_stream_or_device(
      has_stream, dev_only, dev_type, stream_idx);

  // 6. invoke kernel: a cached signature skips MLX's per-call source
  // generation and pipeline lookup (the same kernel runs).
  std::vector<int64_t> key;
  if (kernel.cache && !verbose) {
    key.reserve(4 + 3 * template_args.size() + 3 * ins.size() + out_dtypes.size());
    key.push_back(init.has_value());
    for (const auto& t : template_args) {
      key.push_back(static_cast<int64_t>(
          std::hash<std::string_view>{}(std::string_view(t.name.data(), t.name.size()))));
      key.push_back(t.kind);
      key.push_back(t.kind == 0 ? t.int_val : t.kind == 1 ? t.bool_val : t.dtype_val);
    }
    key.push_back(-1);
    for (const auto& a : ins) {
      key.push_back(static_cast<int64_t>(a.dtype().val()) * 4 + (a.ndim() == 0 ? 1 : 0) +
                    (a.size() < 8 ? 2 : 0));
    }
    key.push_back(-2);
    for (const auto& dt : out_dtypes) {
      key.push_back(static_cast<int64_t>(dt.val()));
    }
    if (auto cached = cached_kernel_dispatch(
            *kernel.cache, key, ins, output_shapes.shapes, out_dtypes,
            std::make_tuple(gx, gy, gz), std::make_tuple(tx, ty, tz), init, target)) {
      auto out_vec = std::make_unique<ArrayVec>();
      out_vec->inner = std::move(*cached);
      return out_vec;
    }
  }
  auto outs = kernel.fn(
      ins,
      output_shapes.shapes,
      out_dtypes,
      std::make_tuple(gx, gy, gz),
      std::make_tuple(tx, ty, tz),
      tmpl,
      init,
      verbose,
      target);
  if (kernel.cache && !verbose) {
    cached_kernel_remember(*kernel.cache, key, outs);
  }

  // 7. wrap into ArrayVec (field name is `inner`, not `arrays`)
  auto out_vec = std::make_unique<ArrayVec>();
  out_vec->inner = std::move(outs);
  return out_vec;
}

}  // namespace cxx_mlx
