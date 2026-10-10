// Native kernels of the qualified Qwen3.6 MoE affine4 DFlash2 target: the
// row-stable affine4 projection (see ironmlx-lm nn::rs4_qmm for the
// arithmetic), the GatedDeltaNet gates and q/k norms, the MoE router, the
// product-stable qmv row grouping and the DFlash2 greedy selector walk.
// Native primitives so a dispatch costs one pipeline lookup: MLX's generic
// custom kernels rebuild and hash their source on every call.
#include "cxx_mlx_shim/quantization.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<Metal/Metal.hpp>)
#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/primitives.h"
#include "cxx_mlx_shim/shim_helpers.h"

namespace cxx_mlx {
namespace {
namespace mx = mlx::core;

// Operand checks of the entry points below. Every one runs before an entry
// point reads a dimension, divides by one or builds a graph, so malformed
// operands throw (an `Err` through the Rust bindings) instead of asserting or
// reaching a kernel that would index out of bounds.
void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

bool fits_i32(int64_t v) {
  return v >= 0 && v <= INT32_MAX;
}

constexpr const char* kSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr int G = K / 64;

inline float bf_lo(uint w) { return as_type<float>(w << 16); }
inline float bf_hi(uint w) { return as_type<float>(w & 0xFFFF0000u); }
inline float bf_at(uint4 v, int e) { return (e % 2) ? bf_hi(v[e / 2]) : bf_lo(v[e / 2]); }

// One row: lane (row group, slice j); every group's weights are loaded
// before any arithmetic.
[[kernel]] void rs4_row(
    const device bfloat* x [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LG = 32 / KS;
  constexpr int GL = G / KS;
  constexpr int GP = PF ? GL : 1;
  const int j = int(lane) % KS;
  const int n0 = (int(tgp.x) * SGS + int(sgi)) * (LG * R) + (int(lane) / KS) * R;
  const device uint4* w4 = (const device uint4*)w;
  const device uint4* x4 = (const device uint4*)x;
  float c[R][MR][A];
  uint4 wlo[GP][R], whi[GP][R];
  float sc[GP][R], bi[GP][R];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; m++) {
      #pragma clang loop unroll(full)
      for (int a = 0; a < A; a++) c[r][m][a] = 0.0f;
    }
  }
  #pragma clang loop unroll(full)
  for (int i = 0; i < GL; i++) {
    const int g = j + i * KS;
    if (!PF || i == 0) {
      #pragma clang loop unroll(full)
      for (int ii = 0; ii < GP; ii++) {
        const int gg = PF ? j + ii * KS : g;
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {
          const size_t n = size_t(n0 + r);
          wlo[ii][r] = w4[n * (K / 32) + gg * 2];
          whi[ii][r] = w4[n * (K / 32) + gg * 2 + 1];
          sc[ii][r] = float(scales[n * G + gg]);
          bi[ii][r] = float(biases[n * G + gg]);
        }
      }
    }
    const int ip = PF ? i : 0;
    uint4 xv[MR][8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; m++) {
      #pragma clang loop unroll(full)
      for (int kk = 0; kk < 8; kk++) xv[m][kk] = x4[size_t(m) * (K / 8) + g * 8 + kk];
    }
    #pragma clang loop unroll(full)
    for (int s = 0; s < 8; s++) {
      const float p = as_type<float>(uint(127 - 4 * s) << 23);
      const uint mask = 0xFu << (4 * s);
      float xs[MR][8];
      #pragma clang loop unroll(full)
      for (int m = 0; m < MR; m++) {
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < 8; kk++) xs[m][kk] = bf_at(xv[m][kk], s);
      }
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) {
        const float scs = sc[ip][r] * p;
        const uint wr[8] = {wlo[ip][r].x, wlo[ip][r].y, wlo[ip][r].z, wlo[ip][r].w,
                            whi[ip][r].x, whi[ip][r].y, whi[ip][r].z, whi[ip][r].w};
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < 8; kk++) {
          const float wq = fma(scs, float(wr[kk] & mask), bi[ip][r]);
          #pragma clang loop unroll(full)
          for (int m = 0; m < MR; m++) c[r][m][s % A] = fma(wq, xs[m][kk], c[r][m][s % A]);
        }
      }
    }
  }
  const int base = int(lane) - j;
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; m++) {
      float cs = c[r][m][0];
      #pragma clang loop unroll(full)
      for (int a = 1; a < A; a++) cs += c[r][m][a];
      float v = simd_shuffle(cs, ushort(base));
      #pragma clang loop unroll(full)
      for (int i = 1; i < KS; i++) v += simd_shuffle(cs, ushort(base + i));
      if (j == 0) y[size_t(m) * N + n0 + r] = bfloat(v);
    }
  }
}

// Two to four rows, x staged once per threadgroup as BF16 in step order
// ([g][s][m][kk]); lane (row group, slice j) evaluates R output rows.
[[kernel]] void rs4_rowx(
    const device bfloat* x [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LG = 32 / KS;
  constexpr int GL = G / KS;
  const int j = int(lane) % KS;
  const int tid = int(sgi) * 32 + int(lane);
  const int n0 = (int(tgp.x) * SGS + int(sgi)) * (LG * R) + (int(lane) / KS) * R;
  const device uint4* w4 = (const device uint4*)w;
  threadgroup bfloat xt[K * MR];
  // xt[((g * 8 + s) * MR + m) * 8 + kk] = x[m][64 g + 8 kk + s]
  for (int idx = tid; idx < MR * K; idx += SGS * 32) {
    const int m = idx / K, k = idx % K;
    const int g = k / 64, kk = (k % 64) / 8, s = k % 8;
    xt[((g * 8 + s) * MR + m) * 8 + kk] = x[size_t(m) * K + k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float c[R][MR][A];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; m++) {
      #pragma clang loop unroll(full)
      for (int a = 0; a < A; a++) c[r][m][a] = 0.0f;
    }
  }
  for (int i = 0; i < GL; i++) {
    const int g = j + i * KS;
    uint wr[R][8];
    float sc[R], bi[R];
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      const size_t n = size_t(n0 + r);
      const uint4 lo = w4[n * (K / 32) + g * 2];
      const uint4 hi = w4[n * (K / 32) + g * 2 + 1];
      wr[r][0] = lo.x; wr[r][1] = lo.y; wr[r][2] = lo.z; wr[r][3] = lo.w;
      wr[r][4] = hi.x; wr[r][5] = hi.y; wr[r][6] = hi.z; wr[r][7] = hi.w;
      sc[r] = float(scales[n * G + g]);
      bi[r] = float(biases[n * G + g]);
    }
    #pragma clang loop unroll(full)
    for (int s = 0; s < 8; s++) {
      const float p = as_type<float>(uint(127 - 4 * s) << 23);
      const uint mask = 0xFu << (4 * s);
      float xs[MR][8];
      #pragma clang loop unroll(full)
      for (int m = 0; m < MR; m++) {
        const uint4 v = *(const threadgroup uint4*)(xt + ((g * 8 + s) * MR + m) * 8);
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < 8; kk++) xs[m][kk] = bf_at(v, kk);
      }
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) {
        const float scs = sc[r] * p;
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < 8; kk++) {
          const float wq = fma(scs, float(wr[r][kk] & mask), bi[r]);
          #pragma clang loop unroll(full)
          for (int m = 0; m < MR; m++) c[r][m][s % A] = fma(wq, xs[m][kk], c[r][m][s % A]);
        }
      }
    }
  }
  const int base = int(lane) - j;
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    #pragma clang loop unroll(full)
    for (int m = 0; m < MR; m++) {
      float cs = c[r][m][0];
      #pragma clang loop unroll(full)
      for (int a = 1; a < A; a++) cs += c[r][m][a];
      float v = simd_shuffle(cs, ushort(base));
      #pragma clang loop unroll(full)
      for (int ii = 1; ii < KS; ii++) v += simd_shuffle(cs, ushort(base + ii));
      if (j == 0) y[size_t(m) * N + n0 + r] = bfloat(v);
    }
  }
}

// Up to eight rows padded to one 8x8x8 MMA tile: out^T = W x^T. Simdgroup j
// evaluates slice j; slices are summed in order through threadgroup memory.
[[kernel]] void rs4_mma(
    const device bfloat* x [[buffer(0)]],
    const device uint* w [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant int& M [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int GL = G / KS;
  const int j = int(sgi);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int nb = int(tgp.x) * 8;
  const int n = nb + fm;
  const device uint2* w2 = (const device uint2*)w;
  const bool v0 = fn < M;
  const bool v1 = fn + 1 < M;
  const device uint4* x0 = (const device uint4*)(x + size_t(min(fn, M - 1)) * K) + fm;
  const device uint4* x1 = (const device uint4*)(x + size_t(min(fn + 1, M - 1)) * K) + fm;
  simdgroup_matrix<float, 8, 8> C[A];
  for (int a = 0; a < A; a++) C[a] = simdgroup_matrix<float, 8, 8>(0.0f);
  uint2 wv[GL];
  float sc[GL], bi[GL];
  uint4 xa[GL], xb[GL];
  #pragma clang loop unroll(full)
  for (int i = 0; i < GL; i++) {
    const int g = j + i * KS;
    wv[i] = w2[size_t(n) * (K / 16) + g * 4 + fn / 2];
    sc[i] = float(scales[size_t(n) * G + g]);
    bi[i] = float(biases[size_t(n) * G + g]);
    xa[i] = v0 ? x0[g * 8] : uint4(0);
    xb[i] = v1 ? x1[g * 8] : uint4(0);
  }
  #pragma clang loop unroll(full)
  for (int i = 0; i < GL; i++) {
    #pragma clang loop unroll(full)
    for (int s = 0; s < 8; s++) {
      simdgroup_matrix<float, 8, 8> B;
      B.thread_elements()[0] = bf_at(xa[i], s);
      B.thread_elements()[1] = bf_at(xb[i], s);
      const float p = as_type<float>(uint(127 - 4 * s) << 23);
      const uint mask = 0xFu << (4 * s);
      const float scs = sc[i] * p;
      simdgroup_matrix<float, 8, 8> Aw;
      Aw.thread_elements()[0] = fma(scs, float(wv[i].x & mask), bi[i]);
      Aw.thread_elements()[1] = fma(scs, float(wv[i].y & mask), bi[i]);
      simdgroup_multiply_accumulate(C[s % A], Aw, B, C[s % A]);
    }
  }
  for (int a = 1; a < A; a++) {
    C[0].thread_elements()[0] += C[a].thread_elements()[0];
    C[0].thread_elements()[1] += C[a].thread_elements()[1];
  }
  threadgroup float red[KS * 64];
  simdgroup_store(C[0], red + j * 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = j * 32 + int(lane); e < 64; e += KS * 32) {
    float v = red[e];
    for (int i = 1; i < KS; i++) v += red[i * 64 + e];
    const int m = e % 8;
    if (m < M) y[size_t(m) * N + nb + e / 8] = bfloat(v);
  }
}
)METAL";

class RowStableAffine4 final : public mx::UnaryPrimitive {
 public:
  RowStableAffine4(mx::Stream stream, int kind, int m, int n, int k, int ks, int r, int sgs)
      : UnaryPrimitive(stream), kind_(kind), m_(m), n_(n), k_(k), ks_(ks), r_(r), sgs_(sgs) {}
  const char* name() const override { return "RowStableAffine4"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("RowStableAffine4 requires the GPU");
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const RowStableAffine4&>(other);
    return kind_ == o.kind_ && m_ == o.m_ && n_ == o.n_ && k_ == o.k_ && ks_ == o.ks_ &&
           r_ == o.r_ && sgs_ == o.sgs_;
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    auto contiguous = [&](const mx::array& a) -> mx::array {
      if (a.flags().row_contiguous) return a;
      copies.push_back(mx::array(a.shape(), a.dtype(), nullptr, {}));
      mx::copy_gpu(a, copies.back(), mx::CopyType::General, s);
      return copies.back();
    };
    const mx::array x = contiguous(inputs[0]);
    const mx::array w = contiguous(inputs[1]);
    const mx::array sc = contiguous(inputs[2]);
    const mx::array bi = contiguous(inputs[3]);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    // Kinds: 0 one row, 2 MMA tile, 5 rows staged once per threadgroup.
    static const char* kNames[] = {"rs4_row", "", "rs4_mma", "", "", "rs4_rowx"};
    const int mr = kind_ == 5 ? m_ : 1;
    std::string lib_name = "ironmlx_rs4_v5_k" + std::to_string(k_) + "_n" + std::to_string(n_) +
                           "_ks" + std::to_string(ks_) + "_r" + std::to_string(r_) + "_mr" +
                           std::to_string(mr) + "_sg" + std::to_string(sgs_);
    auto& d = mx::metal::device(s.device);
    auto lib = d.get_library(lib_name, [&] {
      return "#define K " + std::to_string(k_) + "\n#define N " + std::to_string(n_) +
             "\n#define KS " + std::to_string(ks_) + "\n#define R " + std::to_string(r_) +
             "\n#define MR " + std::to_string(mr) + "\n#define SGS " + std::to_string(sgs_) +
             "\n#define PF 1\n#define A 4" +
             "\n" + kSource;
    });
    auto kernel = d.get_kernel(kNames[kind_], lib);
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_input_array(sc, 2);
    enc.set_input_array(bi, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(m_, 5);
    if (kind_ == 0 || kind_ == 5) {
      enc.dispatch_threadgroups(MTL::Size(n_ / (sgs_ * (32 / ks_) * r_), 1, 1),
                                MTL::Size(32, sgs_, 1));
    } else {
      enc.dispatch_threadgroups(MTL::Size(n_ / 8, 1, 1), MTL::Size(32, ks_, 1));
    }
    enc.add_temporaries(std::move(copies));
  }

 private:
  int kind_, m_, n_, k_, ks_, r_, sgs_;
};

// The GatedDeltaNet gate chain of the ordinary op-by-op path, one MLX op
// after another, each rounding to the dtype MLX produces at that step:
//   x = a + dt_bias (bf16); sp = x > 20 ? x : logaddexp(0, x) (bf16);
//   g = exp(-exp(float(a_log)) * float(sp)) (float32); beta = sigmoid(b) (bf16).
// LogAddExp and Sigmoid are MLX's definitions written out with the precise
// exp/log of MLX's prebuilt kernels: in a runtime-compiled library the BF16
// math overloads of MLX's preamble resolve to the fast variants.
constexpr const char* kExactBf16MathSource = R"METAL(
inline bfloat16_t exp_bf16(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(x)));
}
inline bfloat16_t log1p_bf16(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }
  return bfloat16_t(x * (metal::precise::log(xp1) / (xp1 - 1.0f)));
}
inline bfloat16_t logaddexp_bf16(bfloat16_t x, bfloat16_t y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<bfloat16_t>::quiet_NaN();
  }
  constexpr bfloat16_t inf = metal::numeric_limits<bfloat16_t>::infinity();
  bfloat16_t maxval = metal::max(x, y);
  bfloat16_t minval = metal::min(x, y);
  return (minval == -inf || maxval == inf) ? maxval
                                           : (maxval + log1p_bf16(exp_bf16(minval - maxval)));
}
inline bfloat16_t sigmoid_bf16(bfloat16_t x) {
  auto y = 1 / (1 + exp_bf16(static_cast<bfloat16_t>(metal::precise::fabs(static_cast<float>(x)))));
  return (x < 0) ? y : 1 - y;
}
)METAL";

constexpr const char* kGdnGatesSource = R"METAL(
[[kernel]] void gdn_gates(
    const device bfloat16_t* a [[buffer(0)]],
    const device bfloat16_t* b [[buffer(1)]],
    const device bfloat16_t* a_log [[buffer(2)]],
    const device bfloat16_t* dt_bias [[buffer(3)]],
    device float* g [[buffer(4)]],
    device bfloat16_t* beta [[buffer(5)]],
    constant int& total [[buffer(6)]],
    constant int& heads [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if (int(i) >= total) return;
  const int h = int(i) % heads;
  const bfloat16_t x = Add()(a[i], dt_bias[h]);
  const bfloat16_t safe = logaddexp_bf16(bfloat16_t(0), x);
  const bool cond = Greater()(static_cast<float>(x), 20.0f);
  const bfloat16_t sp = cond ? x : safe;
  const float neg = Negative()(Exp()(static_cast<float>(a_log[h])));
  g[i] = Exp()(Multiply()(neg, static_cast<float>(sp)));
  beta[i] = static_cast<bfloat16_t>(sigmoid_bf16(b[i]));
}
)METAL";

class GdnGates final : public mx::Primitive {
 public:
  explicit GdnGates(mx::Stream stream) : Primitive(stream) {}
  const char* name() const override { return "IronGdnGates"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("IronGdnGates requires the GPU");
  }
  bool is_equivalent(const mx::Primitive&) const override { return true; }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    auto contiguous = [&](const mx::array& a) -> mx::array {
      if (a.flags().row_contiguous) return a;
      copies.push_back(mx::array(a.shape(), a.dtype(), nullptr, {}));
      mx::copy_gpu(a, copies.back(), mx::CopyType::General, s);
      return copies.back();
    };
    const mx::array a = contiguous(inputs[0]);
    const mx::array b = contiguous(inputs[1]);
    const mx::array a_log = contiguous(inputs[2]);
    const mx::array dt = contiguous(inputs[3]);
    for (auto& out : outputs) out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(s.device);
    static MTL::ComputePipelineState* kernel = nullptr;
    if (!kernel) {
      auto lib = d.get_library("ironmlx_gdn_gates_exact_v2", [] {
        return std::string(mx::metal::utils()) + mx::metal::unary_ops() + mx::metal::binary_ops() +
               kExactBf16MathSource + kGdnGatesSource;
      });
      kernel = d.get_kernel("gdn_gates", lib);
    }
    const int total = int(a.size());
    const int heads = int(a_log.size());
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(a, 0);
    enc.set_input_array(b, 1);
    enc.set_input_array(a_log, 2);
    enc.set_input_array(dt, 3);
    enc.set_output_array(outputs[0], 4);
    enc.set_output_array(outputs[1], 5);
    enc.set_bytes(total, 6);
    enc.set_bytes(heads, 7);
    enc.dispatch_threadgroups(MTL::Size((total + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
    enc.add_temporaries(std::move(copies));
  }
};
}  // namespace

namespace {
namespace mx = mlx::core;

// The MoE router of the ordinary op-by-op path in one dispatch, bit for bit:
// - MLX's precise softmax (`softmax_single_row`: 64 threads x 4 reads for 256
//   experts, float max/sum with the same simd and threadgroup reductions,
//   fast::exp, output rounded to bf16);
// - argpartition (MLX sorts: a stable ascending sort) keeping the last K, i.e.
//   the K largest by (probability, expert id), in ascending order;
// - their sum in bf16, added one by one in that order (MLX's small-row sum),
//   and (when `norm`) each divided by it in bf16.
constexpr const char* kRouterTopkSource = R"METAL(
[[kernel]] void router_topk(
    const device bfloat16_t* logits [[buffer(0)]],
    device bfloat16_t* scores [[buffer(1)]],
    device uint* inds [[buffer(2)]],
    constant int& norm [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  constexpr int SIMD_SIZE = 32;
  const int lid = _lid;
  threadgroup float local_max[SIMD_SIZE];
  threadgroup float local_normalizer[SIMD_SIZE];
  threadgroup float probs[E];
  threadgroup float sel_val[K];
  threadgroup uint sel_idx[K];
  float ld[N_READS];
  const device bfloat16_t* in = logits + gid * size_t(E) + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    ld[i] = float(in[i]);
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<float>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    float exp_x = fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  for (int i = 0; i < N_READS; i++) {
    probs[lid * N_READS + i] = float(bfloat16_t(ld[i] * normalizer));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Rank of each element in the descending (probability, id) order.
  for (int i = 0; i < N_READS; i++) {
    const int idx = lid * N_READS + i;
    const float p = probs[idx];
    int rank = 0;
    for (int j = 0; j < E; j++) {
      const float q = probs[j];
      rank += (q > p || (q == p && j > idx)) ? 1 : 0;
    }
    if (rank < K) {
      sel_val[K - 1 - rank] = p;
      sel_idx[K - 1 - rank] = uint(idx);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    bfloat16_t total = bfloat16_t(0);
    for (int j = 0; j < K; j++) {
      total = bfloat16_t(sel_val[j]) + total;
    }
    for (int j = 0; j < K; j++) {
      const bfloat16_t v = bfloat16_t(sel_val[j]);
      scores[gid * K + j] = norm ? Divide()(v, total) : v;
      inds[gid * K + j] = sel_idx[j];
    }
  }
}
)METAL";

class RouterTopk final : public mx::Primitive {
 public:
  RouterTopk(mx::Stream stream, int e, int k, bool norm) : Primitive(stream), e_(e), k_(k), norm_(norm) {}
  const char* name() const override { return "IronRouterTopk"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("IronRouterTopk requires the GPU");
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const RouterTopk&>(other);
    return e_ == o.e_ && k_ == o.k_ && norm_ == o.norm_;
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    mx::array logits = inputs[0];
    if (!logits.flags().row_contiguous) {
      copies.push_back(mx::array(logits.shape(), logits.dtype(), nullptr, {}));
      mx::copy_gpu(logits, copies.back(), mx::CopyType::General, s);
      logits = copies.back();
    }
    for (auto& out : outputs) out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(s.device);
    const std::string lib_name =
        "ironmlx_router_topk_exact_v1_e" + std::to_string(e_) + "_k" + std::to_string(k_);
    auto lib = d.get_library(lib_name, [&] {
      return std::string(mx::metal::utils()) + mx::metal::binary_ops() + "#define E " +
             std::to_string(e_) + "\n#define K " + std::to_string(k_) + "\n" + kRouterTopkSource;
    });
    auto kernel = d.get_kernel("router_topk", lib);
    const int rows = int(logits.size() / e_);
    const int norm = norm_ ? 1 : 0;
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(logits, 0);
    enc.set_output_array(outputs[0], 1);
    enc.set_output_array(outputs[1], 2);
    enc.set_bytes(norm, 3);
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(e_ / 4, 1, 1));
    enc.add_temporaries(std::move(copies));
  }

 private:
  int e_, k_;
  bool norm_;
};
}  // namespace

std::unique_ptr<MlxArrayVec> router_topk_fused(
    const MlxArray& logits, int32_t k, bool norm,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  require(logits.ndim() == 2 && logits.dtype() == mx::bfloat16 && logits.shape(0) >= 1 &&
              logits.shape(1) >= 128 && logits.shape(1) % 128 == 0 && logits.shape(1) <= 4096 &&
              fits_i32(int64_t(logits.size())) && k >= 1 && k <= 32 && k <= logits.shape(1),
          "router_topk_fused: unsupported operands");
  const int rows = logits.shape(0), e = logits.shape(1);
  auto outs = mx::array::make_arrays({mx::Shape{rows, k}, mx::Shape{rows, k}}, {mx::bfloat16, mx::uint32},
                                     std::make_shared<RouterTopk>(stream, e, k, norm), {logits});
  return std::make_unique<MlxArrayVec>(std::move(outs));
}

namespace {
namespace mx = mlx::core;

// MLX's own product-stable affine4 kernel (qmv_fast arithmetic per input
// row: each simdgroup computes one input row against four output rows), with
// up to five input rows per threadgroup instead of MLX's two. MLX's dispatch
// pads an odd row count to an even one (the padded simdgroups repeat the last
// row), so 3 and 5 rows cost as much as 4 and 6; here the rows are split into
// the fewest equal threadgroups. Every output bit is the kernel's own,
// whatever the row grouping.
class QmvFastWide final : public mx::UnaryPrimitive {
 public:
  QmvFastWide(mx::Stream stream, int max_nv) : UnaryPrimitive(stream), max_nv_(max_nv) {}
  const char* name() const override { return "IronQmvFastWide"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("IronQmvFastWide requires the GPU");
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    return max_nv_ == static_cast<const QmvFastWide&>(other).max_nv_;
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    auto contiguous = [&](const mx::array& a) -> mx::array {
      if (a.flags().row_contiguous) return a;
      copies.push_back(mx::array(a.shape(), a.dtype(), nullptr, {}));
      mx::copy_gpu(a, copies.back(), mx::CopyType::General, s);
      return copies.back();
    };
    const mx::array x = contiguous(inputs[0]);
    const mx::array w = contiguous(inputs[1]);
    const mx::array sc = contiguous(inputs[2]);
    const mx::array bi = contiguous(inputs[3]);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const int K = x.shape(-1), N = w.shape(0), M = int(x.size() / K);
    const int tiles = (M + max_nv_ - 1) / max_nv_;
    const int nv = (M + tiles - 1) / tiles;
    auto& d = mx::metal::device(s.device);
    auto kernel = d.get_kernel(
        "affine_qmv_fast_wide_bfloat16_t_gs_64_b_4_nv_" + std::to_string(nv) + "_kl_32_batch_0");
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(w, 0);
    enc.set_input_array(sc, 1);
    enc.set_input_array(bi, 2);
    enc.set_input_array(x, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(K, 5);
    enc.set_bytes(N, 6);
    enc.set_bytes(M, 7);
    enc.dispatch_threadgroups(MTL::Size(tiles, (N + 3) / 4, 1), MTL::Size(32, nv, 1));
    enc.add_temporaries(std::move(copies));
  }

 private:
  int max_nv_;
};

}  // namespace

std::unique_ptr<MlxArray> qmv_fast_wide(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales, const MlxArray& biases,
    int32_t max_nv, bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  require(x.ndim() >= 1 && w.ndim() == 2 && scales.ndim() == 2 && biases.ndim() == 2 &&
              x.dtype() == mx::bfloat16 && w.dtype() == mx::uint32 &&
              scales.dtype() == mx::bfloat16 && biases.dtype() == mx::bfloat16,
          "qmv_fast_wide: unsupported operands");
  const int64_t K = x.shape(-1), N = w.shape(0);
  require(K >= 512 && K % 512 == 0 && N >= 8 && N % 8 == 0 && int64_t(w.shape(1)) * 8 == K &&
              scales.shape(0) == N && scales.shape(1) == K / 64 && biases.shape() == scales.shape(),
          "qmv_fast_wide: unsupported operands");
  const int64_t M = int64_t(x.size()) / K;
  // MLX instantiates the kernel for 2..=5 rows per threadgroup; one row would
  // need the missing nv=1 variant.
  require(max_nv >= 2 && max_nv <= 5 && M >= 2 && fits_i32(int64_t(x.size())) &&
              fits_i32(M * N),
          "qmv_fast_wide: unsupported operands");
  auto shape = x.shape();
  shape.back() = N;
  return std::make_unique<MlxArray>(mx::array(shape, mx::bfloat16,
      std::make_shared<QmvFastWide>(stream, max_nv), {x, w, scales, biases}));
}

namespace {
namespace mx = mlx::core;

// q/k RMS norms of the ordinary op-by-op path for 128-wide heads, bit for
// bit: MLX's `rms_single_row` (one simdgroup, 4 reads per lane, float sum of
// squares, precise rsqrt, output rounded to bf16 and multiplied by a unit
// weight), then the float32 product with the per-tensor scale.
constexpr const char* kGlueSource = R"METAL(
[[kernel]] void qk_norm(
    const device bfloat16_t* src [[buffer(0)]],
    device float* qo [[buffer(1)]],
    device float* ko [[buffer(2)]],
    constant int& heads [[buffer(3)]],
    constant int& c [[buffer(4)]],
    constant int& q_off [[buffer(5)]],
    constant int& k_off [[buffer(6)]],
    constant float& qscale [[buffer(7)]],
    constant float& kscale [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& axis_size [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr int N_READS = 4;
  const int row = int(tg.x), head = int(tg.y);
  const bool is_q = tg.z == 0;
  const device bfloat16_t* x =
      src + size_t(row) * c + (is_q ? q_off : k_off) + head * int(axis_size) + lid * N_READS;
  float acc = 0;
  float thread_x[N_READS];
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] = x[i];
    acc += thread_x[i] * thread_x[i];
  }
  // One simdgroup: MLX's second reduction adds only zeros to this sum.
  acc = simd_sum(acc);
  const float inv_mean = metal::precise::rsqrt(acc / axis_size + eps);
  const bfloat16_t w = bfloat16_t(1);
  const float scale = is_q ? qscale : kscale;
  device float* out = (is_q ? qo : ko) + (size_t(row) * heads + head) * axis_size + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    const bfloat16_t normed = w * static_cast<bfloat16_t>(thread_x[i] * inv_mean);
    out[i] = static_cast<float>(normed) * scale;
  }
}
)METAL";

MTL::ComputePipelineState* glue_kernel(mx::metal::Device& d, const char* name) {
  auto lib = d.get_library("ironmlx_qk_norm_exact_v1", [] {
    return std::string(mx::metal::utils()) + kGlueSource;
  });
  return d.get_kernel(name, lib);
}

class QkNorm final : public mx::Primitive {
 public:
  QkNorm(mx::Stream stream, int heads, int d, int q_off, int k_off, float qscale, float kscale, float eps)
      : Primitive(stream), heads_(heads), d_(d), q_off_(q_off), k_off_(k_off), qscale_(qscale),
        kscale_(kscale), eps_(eps) {}
  const char* name() const override { return "IronQkNorm"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("IronQkNorm requires the GPU");
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const QkNorm&>(other);
    return heads_ == o.heads_ && d_ == o.d_ && q_off_ == o.q_off_ && k_off_ == o.k_off_ &&
           qscale_ == o.qscale_ && kscale_ == o.kscale_ && eps_ == o.eps_;
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    mx::array src = inputs[0];
    if (!src.flags().row_contiguous) {
      copies.push_back(mx::array(src.shape(), src.dtype(), nullptr, {}));
      mx::copy_gpu(src, copies.back(), mx::CopyType::General, s);
      src = copies.back();
    }
    for (auto& o : outputs) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& d = mx::metal::device(s.device);
    static MTL::ComputePipelineState* kernel = nullptr;
    if (!kernel) kernel = glue_kernel(d, "qk_norm");
    const int c = src.shape(-1);
    const int rows = int(src.size() / c);
    const uint32_t axis_size = uint32_t(d_);
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(src, 0);
    enc.set_output_array(outputs[0], 1);
    enc.set_output_array(outputs[1], 2);
    enc.set_bytes(heads_, 3);
    enc.set_bytes(c, 4);
    enc.set_bytes(q_off_, 5);
    enc.set_bytes(k_off_, 6);
    enc.set_bytes(qscale_, 7);
    enc.set_bytes(kscale_, 8);
    enc.set_bytes(eps_, 9);
    enc.set_bytes(axis_size, 10);
    enc.dispatch_threadgroups(MTL::Size(rows, heads_, 2), MTL::Size(32, 1, 1));
    enc.add_temporaries(std::move(copies));
  }

 private:
  int heads_, d_, q_off_, k_off_;
  float qscale_, kscale_, eps_;
};
}  // namespace

std::unique_ptr<MlxArrayVec> qk_norm_fused(
    const MlxArray& src, int32_t heads, int32_t d, int32_t q_off, int32_t k_off, float qscale,
    float kscale, float eps, bool has_target, bool is_device_only, uint8_t device_type,
    int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  require(src.ndim() >= 1 && src.dtype() == mx::bfloat16, "qk_norm_fused: unsupported operands");
  const int64_t c = src.shape(-1);
  const int64_t span = int64_t(heads) * d;
  require(c >= 1 && src.size() >= 1 && fits_i32(int64_t(src.size())) && heads >= 1 && d == 128 &&
              q_off >= 0 && k_off >= 0 && int64_t(q_off) + span <= c &&
              int64_t(k_off) + span <= c,
          "qk_norm_fused: unsupported operands");
  const int rows = int(int64_t(src.size()) / c);
  mx::Shape shape{rows, heads, d};
  auto outs = mx::array::make_arrays({shape, shape}, {mx::float32, mx::float32},
      std::make_shared<QkNorm>(stream, heads, d, q_off, k_off, qscale, kscale, eps), {src});
  return std::make_unique<MlxArrayVec>(std::move(outs));
}

std::unique_ptr<MlxArrayVec> gdn_gates_fused(
    const MlxArray& a, const MlxArray& b, const MlxArray& a_log, const MlxArray& dt_bias,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  require(a.ndim() >= 1 && a.dtype() == mx::bfloat16 && b.dtype() == mx::bfloat16 &&
              a_log.dtype() == mx::bfloat16 && dt_bias.dtype() == mx::bfloat16 &&
              a.shape() == b.shape() && a.size() >= 1 && fits_i32(int64_t(a.size())) &&
              a_log.size() >= 1 && a_log.size() == dt_bias.size() &&
              int64_t(a.shape(-1)) == int64_t(a_log.size()),
          "gdn_gates_fused: unsupported operands");
  auto outs = mx::array::make_arrays({a.shape(), a.shape()}, {mx::float32, mx::bfloat16},
                                     std::make_shared<GdnGates>(stream), {a, b, a_log, dt_bias});
  return std::make_unique<MlxArrayVec>(std::move(outs));
}

std::unique_ptr<MlxArray> row_stable_affine4_matmul(
    const MlxArray& x, const MlxArray& w, const MlxArray& scales, const MlxArray& biases,
    int32_t kind, int32_t ks, int32_t r, int32_t sgs,
    bool has_target, bool is_device_only, uint8_t device_type, int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  if (stream.device.type != mx::Device::gpu)
    throw std::invalid_argument("row_stable_affine4_matmul requires a GPU stream");
  require(x.ndim() == 2 && w.ndim() == 2 && scales.ndim() == 2 && biases.ndim() == 2 &&
              x.dtype() == mx::bfloat16 && w.dtype() == mx::uint32 &&
              scales.dtype() == mx::bfloat16 && biases.dtype() == mx::bfloat16,
          "row_stable_affine4_matmul: unsupported operands");
  const int64_t m = x.shape(0), k = x.shape(1), n = w.shape(0);
  require(m >= 1 && m <= 8 && k >= 512 && k % 512 == 0 && n >= 32 && n % 32 == 0 &&
              int64_t(w.shape(1)) * 8 == k && scales.shape(0) == n && scales.shape(1) == k / 64 &&
              biases.shape() == scales.shape() && fits_i32(m * n),
          "row_stable_affine4_matmul: unsupported shape");
  // ks slices divide the 32 lanes and the k/64 groups; r and sgs size the
  // one-row and staged-row threadgroups, which must tile n exactly.
  require(ks >= 1 && ks <= 32 && 32 % ks == 0 && (k / 64) % ks == 0 && r >= 1 && r <= 4 &&
              sgs >= 1 && sgs <= 32,
          "row_stable_affine4_matmul: unsupported shape");
  const int64_t row_tile = int64_t(sgs) * (32 / ks) * r;
  require((kind == 0 && m == 1 && n % row_tile == 0) ||
              (kind == 5 && m >= 2 && m <= 4 && m * k * 2 <= 16384 && n % row_tile == 0) ||
              (kind == 2 && r == 1 && sgs == 1),
          "row_stable_affine4_matmul: unsupported shape");
  return std::make_unique<MlxArray>(mx::array(
      mx::Shape{int(m), int(n)}, mx::bfloat16,
      std::make_shared<RowStableAffine4>(stream, kind, int(m), int(n), int(k), ks, r, sgs),
      {x, w, scales, biases}));
}

namespace {
namespace mx = mlx::core;

// Greedy DFlash2 selector walk (ironmlx-lm models::dflash2::selector): one
// threadgroup per batch row, one simdgroup per candidate. For each position
// every simdgroup scores its candidate edge (predecessor code * projected
// hidden * successor code over the selector rank, BF16 products as in the
// op-by-op walk), thread 0 takes the first maximum and publishes the next
// predecessor.
constexpr const char* kSelectorSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
[[kernel]] void dflash2_selector_walk(
    const device uint* candidates [[buffer(0)]],
    const device bfloat* unary [[buffer(1)]],
    const device bfloat* hidden [[buffer(2)]],
    const device uint* anchor [[buffer(3)]],
    const device bfloat* predecessor_codebook [[buffer(4)]],
    const device bfloat* successor_codebook [[buffer(5)]],
    device uint* path [[buffer(6)]],
    constant int& L [[buffer(7)]],
    constant uint& V [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint predecessor_shared;
  threadgroup float scores[K];
  if (simd == 0 && lane == 0) predecessor_shared = anchor[row];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 0; p < L; p++) {
    const uint pred = predecessor_shared;
    const size_t base = (size_t(row) * L + p) * K;
    const uint cand = candidates[base + simd];
    const device bfloat* pc = predecessor_codebook + size_t(pred) * D;
    const device bfloat* sc = successor_codebook + size_t(cand) * D;
    const device bfloat* hd = hidden + (size_t(row) * L + p) * D;
    float acc = 0.0f;
    // Token ids are data: an id outside the codebooks reads nothing.
    const int d_end = (pred < V && cand < V) ? D : 0;
    for (int d = int(lane); d < d_end; d += 32) {
      bfloat prod = bfloat(float(pc[d]) * float(hd[d]));
      bfloat term = bfloat(float(prod) * float(sc[d]));
      acc += float(term);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      bfloat edge = bfloat(acc);
      scores[simd] = float(bfloat(float(unary[base + simd]) + float(edge)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
      int best = 0;
      float best_score = scores[0];
      for (int k = 1; k < K; k++) {
        if (scores[k] > best_score) {
          best_score = scores[k];
          best = k;
        }
      }
      const uint chosen = candidates[base + best];
      path[size_t(row) * L + p] = chosen;
      predecessor_shared = chosen;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)METAL";

class DFlash2SelectorWalk final : public mx::UnaryPrimitive {
 public:
  DFlash2SelectorWalk(mx::Stream stream, int k, int d) : UnaryPrimitive(stream), k_(k), d_(d) {}
  const char* name() const override { return "IronDFlash2SelectorWalk"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("IronDFlash2SelectorWalk requires the GPU");
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const DFlash2SelectorWalk&>(other);
    return k_ == o.k_ && d_ == o.d_;
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    auto& s = stream();
    std::vector<mx::array> copies;
    std::vector<mx::array> in;
    for (const auto& a : inputs) {
      if (a.flags().row_contiguous) {
        in.push_back(a);
      } else {
        copies.push_back(mx::array(a.shape(), a.dtype(), nullptr, {}));
        mx::copy_gpu(a, copies.back(), mx::CopyType::General, s);
        in.push_back(copies.back());
      }
    }
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const int batch = in[0].shape(0), length = in[0].shape(1);
    auto& d = mx::metal::device(s.device);
    const std::string lib_name =
        "ironmlx_dflash2_selector_v2_k" + std::to_string(k_) + "_d" + std::to_string(d_);
    auto lib = d.get_library(lib_name, [&] {
      return "#define K " + std::to_string(k_) + "\n#define D " + std::to_string(d_) + "\n" +
             kSelectorSource;
    });
    auto kernel = d.get_kernel("dflash2_selector_walk", lib);
    auto& enc = mx::metal::get_command_encoder(s);
    enc.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 6; i++) enc.set_input_array(in[i], i);
    enc.set_output_array(out, 6);
    enc.set_bytes(length, 7);
    const uint32_t vocab = uint32_t(in[4].shape(0));
    enc.set_bytes(vocab, 8);
    enc.dispatch_threadgroups(MTL::Size(batch, 1, 1), MTL::Size(32 * k_, 1, 1));
    enc.add_temporaries(std::move(copies));
  }

 private:
  int k_, d_;
};
}  // namespace

std::unique_ptr<MlxArray> dflash2_selector_walk(
    const MlxArray& candidates, const MlxArray& unary, const MlxArray& hidden,
    const MlxArray& anchor, const MlxArray& predecessor_codebook,
    const MlxArray& successor_codebook, bool has_target, bool is_device_only, uint8_t device_type,
    int32_t stream_index) {
  auto target = helpers::decode_stream_or_device(has_target, is_device_only, device_type, stream_index);
  const auto stream = mx::to_stream(target);
  require(candidates.ndim() == 3 && unary.shape() == candidates.shape() && hidden.ndim() == 3 &&
              predecessor_codebook.ndim() == 2 &&
              successor_codebook.shape() == predecessor_codebook.shape() &&
              candidates.dtype() == mx::uint32 && anchor.dtype() == mx::uint32 &&
              unary.dtype() == mx::bfloat16 && hidden.dtype() == mx::bfloat16 &&
              predecessor_codebook.dtype() == mx::bfloat16 &&
              successor_codebook.dtype() == mx::bfloat16,
          "dflash2_selector_walk: unsupported operands");
  require(candidates.shape(0) >= 1 && candidates.shape(2) >= 1 && candidates.shape(2) <= 32 &&
              hidden.shape(0) == candidates.shape(0) && hidden.shape(1) == candidates.shape(1) &&
              hidden.shape(2) >= 1 && predecessor_codebook.shape(0) >= 1 &&
              predecessor_codebook.shape(1) == hidden.shape(2) &&
              anchor.size() == size_t(candidates.shape(0)) &&
              fits_i32(int64_t(candidates.size())) && fits_i32(int64_t(hidden.size())),
          "dflash2_selector_walk: unsupported operands");
  return std::make_unique<MlxArray>(mx::array(
      mx::Shape{candidates.shape(0), candidates.shape(1)}, mx::uint32,
      std::make_shared<DFlash2SelectorWalk>(stream, candidates.shape(2), hidden.shape(2)),
      {candidates, unary, hidden, anchor, predecessor_codebook, successor_codebook}));
}
}  // namespace cxx_mlx
#else
namespace cxx_mlx {
std::unique_ptr<MlxArray> row_stable_affine4_matmul(
    const MlxArray&, const MlxArray&, const MlxArray&, const MlxArray&, int32_t, int32_t,
    int32_t, int32_t, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("row_stable_affine4_matmul requires Metal C++ headers");
}
std::unique_ptr<MlxArrayVec> qk_norm_fused(const MlxArray&, int32_t, int32_t, int32_t, int32_t,
    float, float, float, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("qk_norm_fused requires Metal C++ headers");
}
std::unique_ptr<MlxArray> qmv_fast_wide(const MlxArray&, const MlxArray&, const MlxArray&,
    const MlxArray&, int32_t, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("qmv_fast_wide requires Metal C++ headers");
}
std::unique_ptr<MlxArrayVec> router_topk_fused(const MlxArray&, int32_t, bool, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("router_topk_fused requires Metal C++ headers");
}
std::unique_ptr<MlxArrayVec> gdn_gates_fused(
    const MlxArray&, const MlxArray&, const MlxArray&, const MlxArray&, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("gdn_gates_fused requires Metal C++ headers");
}
std::unique_ptr<MlxArray> dflash2_selector_walk(const MlxArray&, const MlxArray&, const MlxArray&,
    const MlxArray&, const MlxArray&, const MlxArray&, bool, bool, uint8_t, int32_t) {
  throw std::runtime_error("dflash2_selector_walk requires Metal C++ headers");
}
}  // namespace cxx_mlx
#endif
