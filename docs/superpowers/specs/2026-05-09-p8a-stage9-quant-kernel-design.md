# P8a Stage 9 — Self-Quant Matmul Metal Kernel Design

**Goal**: ironmlx 自研 4-bit quantized matmul Metal kernel + device-aware/quant-aware tile selection 框架，作为 stage 8 prefill 优化收尾后的下一步 kernel-level 优化。

**Background**: 见 `memory/project_p8a_stage8_findings.md` + `memory/feedback_device_aware_tile.md`。

Stage 8 已完成的关键事实：

- ironmlx vs omlx 4-bit prefill 已基本对齐（PP=2048: 281 vs 285，jitter 内）
- ironmlx vs llama.cpp Q4_K_M 4-bit prefill 落后 30-40%
- bf16 ablate 已确认：llama.cpp 4-bit 优势**全部**来自"量化路径专属优化集合"（quant matmul kernel + chunking 加速耦合）；bf16 路径上 mlx 阵营反超 llama.cpp +15%
- 深度调研已确认：llama.cpp 优势核心来自 tile 配置（64×128 vs mlx 32×32），dispatch overhead 在 M=2048 上是 mlx 的 1/8；但**llama.cpp 跟 mlx 都硬编码 tile**，没做 device-aware
- ironmlx decode 已全面领先（vs omlx +2-3%, vs llama.cpp +50%）

Stage 9 的核心论点：**ironmlx 在 4-bit 量化路径上自研 device-aware + quant-aware Metal kernel 是真正的差异化机会**——llama.cpp 和 mlx 都没做这个。

---

## 1. Scope

| 维度 | Stage 9 范围 | Stage 9 不做（deferred） |
|---|---|---|
| Quantization scheme | MLX 4-bit affine（group_size=64, bits=4） | Q4_K_M / Q5_K_M / Q8_0 / bf16 / fp16 → stage 11+ |
| Phase | Prefill only（qmm_t 大矩阵路径） | Decode (qmv vector kernel + split-K) → stage 10/12 |
| Device 范围 | M1 Pro 一行 lookup + 全部 fallback | M1 Max / M1 Ultra / M3+ → stage 10 |
| Replace 策略 | Opt-in via env var `IRONMLX_USE_SELF_QMM=1` | 完全替换默认路径 → stage 10/11 |
| Tile candidates | 3 个 hardcoded variants | 7+ candidates / first-run profiling → stage 10 |
| Loading | 仅 HF safetensors 目录（现有 Loader） | GGUF 加载支持 → stage 11+ |

---

## 2. Architecture

```
                       IRONMLX_USE_SELF_QMM=1?
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
          YES (新路径)                       NO (默认路径，跟 stage 8 commit 一致)
              │                                   │
              ▼                                   ▼
   ┌──────────────────────┐         mlx::quantization::
   │   nn::self_qmm       │         quantized_matmul_on()
   │  ┌────────────────┐  │         (现有，stage 8 commit 状态)
   │  │ lookup_tile    │  │
   │  │ (device, M, N) │  │
   │  │   ──> tile     │  │
   │  └────────────────┘  │
   │  ┌────────────────┐  │
   │  │ kernel cache   │  │
   │  │ (PSO per tile) │  │
   │  └────────────────┘  │
   │  ┌────────────────┐  │
   │  │ metal_kernel   │  │
   │  │ dispatch       │  │
   │  └────────────────┘  │
   └──────────────────────┘
              │
              ▼
       mx::Array [B, S, N]
```

**核心设计原则**：

- **Opt-in 路径与默认路径完全独立**。env var 关闭时 ironmlx Linear / GatedDeltaNet 走 mlx 原路径，跟 stage 8 commits 完全一致；env var 启用时走自研模块。无中间状态、无 fallback 切换。
- **三层模块化**：(1) lookup 决策（device + shape → tile），(2) kernel builder 缓存（mx::fast::metal_kernel + thread_local PSO 缓存），(3) Metal 源码（templated by function constants）。每层单独可测、单独可替换。
- **跨芯片 fallback 安全**：lookup 表只填 M1 Pro 一行，其他芯片 fallback 到 default tile (64,64,32)；保证在任何芯片上都不会 hard fail。
- **预先验证（stage 9 第一步）**：echo kernel（输入直接输出）插入 forward path，用 iron-bench 确认 fusion barrier 不引入 measurable 退化。验证通过后才真正实现 quant kernel。

---

## 3. 代码组织

```
ironmlx/src/nn/self_qmm/
├── mod.rs              入口 + env var 检测 + dispatch (qmm_t_on 函数)
├── lookup.rs           (device, M, N, K, bits, group_size) → (BM, BN, BK) 查表
├── kernel.rs           MetalKernel builder + thread_local PSO 缓存
└── metal/
    └── qmm_t.metal     MSL 源码：unpack + dequant + tile MMA + store
                        (BM, BN, BK 通过 function constants 注入)

ironmlx/src/nn/linear.rs    (修改)
  - forward_on(): if env enabled call self_qmm::qmm_t_on, else fallthrough

ironmlx/src/nn/gated_delta_net.rs  (修改)
  - in_proj_qkvz / in_proj_ba 调用点同上 dispatch

ironmlx-bench-kernel/   (新 crate, workspace member)
├── Cargo.toml
├── src/main.rs          CLI: --M --N --K --BM --BN --BK --bits --group-size --runs
└── src/lib.rs           benchmark runner + GPU 计时
```

---

## 4. Component Specifications

### 4.1 `nn::self_qmm::lookup`

输入：`(device_arch: &str, M: i32, N: i32, K: i32, bits: i32, group_size: i32)`

输出：`(BM: i32, BN: i32, BK: i32)`

逻辑：

```
match device_arch {
    "apple_g13s" | "apple_g13d" => {  // M1 Pro / M1 Pro Max GPU
        // M1 Pro 行：sweep 后填，初期占位 (64, 128, 32)
        match (M, N, K, bits, group_size) {
            // 后续 stage 9 step 7 sweep 后填具体 tile
            _ => (64, 128, 32),  // 占位 default
        }
    }
    _ => (64, 64, 32),  // 其他芯片 fallback 到最保守 tile
}
```

注意：`device_arch` 字符串从 `mlx::Device::get_architecture()` 获取（参考 mlx 内部用法 [quantized.cpp:84-126](/Volumes/Dev/mlx/mlx/backend/metal/quantized.cpp#L84)）。

### 4.2 `nn::self_qmm::kernel`

职责：管理 `mx::fast::metal_kernel` 的 builder + PSO 缓存。每个 (BM, BN, BK) 组合对应一个 PSO，第一次调用编译，后续从缓存读。

接口：

```rust
pub struct QmmKernelCache {
    cache: Mutex<HashMap<(i32, i32, i32), MetalKernel>>,
    source: &'static str,  // 静态 MSL 源码
}

impl QmmKernelCache {
    pub fn get_or_build(&self, bm: i32, bn: i32, bk: i32) -> Result<&MetalKernel>;
}

// thread_local 实例（参考 stage 8 删 mx::compile wrapper 时的 thread_local 解释）
thread_local! { static QMM_CACHE: QmmKernelCache = ...; }
```

注意：MLX command encoder 是 thread_local，所以 PSO 缓存也用 thread_local 存放避免跨线程问题。

### 4.3 `nn::self_qmm::metal::qmm_t.metal`

MSL 源码模板（伪代码 outline）：

```metal
// Function constants — 编译时注入 BM/BN/BK 值
constant int BM [[function_constant(0)]];
constant int BN [[function_constant(1)]];
constant int BK [[function_constant(2)]];

// Inputs (auto-injected shape buffers by mx::fast::metal_kernel)
//   x:        [B, S, K]   bf16 input
//   weights:  [N, K] packed uint32 (4-bit, group_size=64)
//   scales:   [N, K/group_size] bf16
//   biases:   [N, K/group_size] bf16
//
// Output:
//   out:      [B, S, N]   bf16

kernel void qmm_t(
    device const T* x,
    device const uint32_t* w,
    device const T* scales,
    device const T* biases,
    device T* out,
    constant int& M,
    constant int& N,
    constant int& K,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // 1. allocate threadgroup memory (Xs[BM x BK], Ws[BN x BK])
    // 2. K loop: for k=0; k<K; k+=BK:
    //    a. cooperative load Xs from x (BM × BK tile)
    //    b. unpack + dequant weights into Ws (BN × BK tile, group-aware)
    //    c. simdgroup MMA accumulate into result registers
    // 3. cooperative store result registers to out
}
```

实现细节（待 step 2 时补全）：

- Threadgroup memory 用量估算：BM × BK × 2 (bf16) + BN × BK × 2 = 例如 64×32×2 + 128×32×2 = 12KB（在 M1 Pro 32KB 限制内）
- SIMD-group MMA：使用 Metal `simdgroup_matrix_storage` API（8×8 bf16 MMA）
- Dequant：thread-level inline dequant，输出寄存器再写 shmem（参考 llama.cpp 模式 [ggml-metal.metal:681-697](/Volumes/CodeHub/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L681)）

### 4.4 `nn::self_qmm::mod` (入口)

```rust
pub fn qmm_t_on(
    x: &Array,
    weights: &Array,
    scales: &Array,
    biases: Option<&Array>,
    bits: i32,
    group_size: i32,
    target: impl Into<StreamOrDevice>,
) -> Result<Array>;
```

行为：

1. 从 `mlx::Device` 获取 architecture 字符串
2. 计算 (M, N, K) from input shapes
3. lookup → (BM, BN, BK)
4. 从 thread_local 缓存取 / 编译 PSO
5. dispatch with grid_dims = (ceil(N/BN), ceil(M/BM), 1), threadgroup_dims = (32, num_simdgroups_per_tg, 1)
6. 返回 mx::Array

env var 检测：在 caller (linear.rs / gated_delta_net.rs) 端做，self_qmm 模块本身始终是直接调用接口。

### 4.5 `Linear::forward_on`（修改）

```rust
pub fn forward_on(&self, x: &Array, target: impl Into<StreamOrDevice>) -> Result<Array> {
    let target = target.into();
    if env_self_qmm_enabled() && self.has_quant_meta() {
        // 新路径：self-quant kernel
        self_qmm::qmm_t_on(x, &self.weight, &self.scales, self.biases.as_ref(),
                            self.bits, self.group_size, target)
    } else {
        // 默认路径：mlx::quantized_matmul_on (stage 8 现状，不动)
        mlx::quantization::quantized_matmul_on(/* ... */)
    }
}
```

`env_self_qmm_enabled()` 是一个 `OnceLock<bool>` cached 函数，只在第一次调用时读 env var。

---

## 5. Data Flow

```
ironmlx Linear::forward_on(x: bf16 [B, S, K]) called
    ├── env IRONMLX_USE_SELF_QMM != 1
    │     └── mlx::quantized_matmul_on (stage 8 path) → return
    │
    └── env IRONMLX_USE_SELF_QMM == 1:
            self_qmm::qmm_t_on(x, weights, scales, biases, bits=4, group_size=64)
                 ├── M = B × S; N = output_dim; K = hidden_dim
                 ├── tile = lookup_tile(device.architecture(), M, N, K, 4, 64)
                 │        ├── M1 Pro 命中: 用 sweep 数据填的 entry
                 │        └── 其他 device: fallback (64, 64, 32)
                 ├── pso = QMM_CACHE.get_or_build(BM, BN, BK)
                 │        ├── 第一次: mx::fast::metal_kernel
                 │        │            .source(qmm_t.metal)
                 │        │            .template_int("BM", BM)
                 │        │            .template_int("BN", BN)
                 │        │            .template_int("BK", BK)
                 │        │            .build()
                 │        └── 后续: cached
                 ├── grid_dims = (ceil(N/BN), ceil(M/BM), 1)
                 ├── tg_dims = (32, num_simdgroups_per_tg, 1)
                 ├── pso.dispatch(...)
                 └── return mx::Array [B, S, N]
```

---

## 6. Error Handling

| 场景 | 处理 |
|---|---|
| 未知 device (lookup miss) | fallback 到 default tile (64, 64, 32)，warn log 一次（`OnceLock` 防重复警告） |
| PSO 编译失败 | panic with metal compile error。env var 启用是显式选择，应当显式失败便于调试 |
| K 不是 BK 倍数 | kernel 内部用 `min(BK, K - k_offset)` partial loop 处理 |
| M, N 不是 BM, BN 倍数 | 边界 threadgroup 内部 bound check 安全 store（`if (row < M && col < N)`） |
| 输入 dtype 不是 bf16 / weights 不是 4-bit | panic — env var 启用是显式选择 |
| group_size != 64 | 暂不支持，panic（stage 9 锁定 group=64） |

---

## 7. Testing 策略

### Level 1: 数值正确性（unit + integration）

- Small shape unit test: M=4, K=32, N=8（in `nn/self_qmm/mod.rs` mod tests）
  - 自研 kernel output vs mlx::quantized_matmul output: max abs diff < 0.5
- P4 fixture integration test: 复用 `tests/p4_qwen35_logits_match.rs` 框架，用 `IRONMLX_USE_SELF_QMM=1` 跑同一个 Qwen3.5-4B fixture
  - 对 last-position logits: max abs diff < 0.5
  - top-1 argmax: 跟 mlx-lm reference 完全一致
- 3 个 tile variants 都需通过同一组测试（不只默认 tile）

### Level 2: Micro-bench (`ironmlx-bench-kernel`)

- Sweep 3 tile variants × Qwen3.5 FFN shape (M=2048, K=2560, N=9216) × 5 runs median
- 输出：median wall-clock per kernel call (ms) + GFLOP/s
- 对照 baseline: mlx::quantized_matmul on same shape
- 用于填 lookup table M1 Pro 行（step 7）

### Level 3: End-to-end (`iron-bench`)

- IRONMLX_USE_SELF_QMM=1: PP=2048 prefill PP > 281 (mlx baseline)
  - 理想 stretch goal: > 332 (llama.cpp Q4_K_M chunk OFF)
  - 极致 stretch goal: > 390 (llama.cpp Q4_K_M chunk ON)
- 默认路径 (env 关): 端到端跟 stage 8 commit `811dd36` 数据无差异（确保 opt-in 完全隔离）
- 文本 sanity test: 同 prompt "Say hi in 5 words." 输出文本跟 mlx default 一致（"Hello there, friend." or 类似）

---

## 8. Stage 9 Task Outline

| Step | 任务 | 单 commit 大小 | 决策门 |
|---|---|---|---|
| **1** | Fusion barrier 预先验证（echo kernel 插入 forward path，iron-bench ON vs OFF） | 小 | > 5% 退化 → 重选机制 |
| **2** | Self-qmm Metal source（写死 (64,64,32) 起步）+ 数值正确性 unit test | 中 | atol < 0.5 不通过 → debug kernel |
| **3** | 推广到 3 个 tile variants（function constants 参数化）+ 数值测试 3 个 variants | 中 | 同 step 2 |
| **4** | Device + shape lookup 表（M1 Pro 占位行 + fallback） | 小 | — |
| **5** | 集成到 Linear / GatedDeltaNet（env var dispatch）+ 默认路径 regression test | 小 | 默认路径 regression > 1% → debug |
| **6** | `ironmlx-bench-kernel` binary（新 crate） | 中 | — |
| **7** | M1 Pro tile sweep + 填 lookup 表 M1 Pro 行 | 小 | — |
| **8** | End-to-end iron-bench 验证 + 三方对比（vs omlx, llama.cpp） | 小 | env=1 时 PP < 281 → debug 或回 step 2-3 |
| **9** | 文档（README env var）+ memory 更新 stage 9 完成状态 | 小 | — |

---

## 9. Acceptance Criteria

- ✅ `IRONMLX_USE_SELF_QMM=1`: PP=2048 prefill PP > **281 tok/s**（mlx baseline）
- 🎯 理想 stretch goal: > 332（llama.cpp chunk OFF）
- 🌟 极致 stretch goal: > 390（llama.cpp chunk ON）
- ✅ env var 关闭: 端到端跟 stage 8 commit `811dd36` 数据**无差异**（默认路径完全隔离，regression < 1%）
- ✅ 数值: atol < 0.5 vs mlx output, top-1 argmax 一致
- ✅ 文本 sanity: 同 prompt 输出文本跟 mlx default 一致

---

## 10. 风险 + Mitigations

| 风险 | 概率 | 影响 | Mitigation |
|---|---|---|---|
| fusion barrier 真退化 | 低 | 高 | step 1 预先验证暴露；重选 kernel 注入机制（fork mlx / cxx 桥接 metal） |
| 3 个 tile 都跑不过 mlx baseline | 中 | 中 | stage 10 加 candidate（128×64 / 128×256 等）+ dequant 优化 |
| 数值偏差超过 atol | 低 | 高 | step 2 unit test 早期暴露；debug kernel（reduction 顺序 / mask 边界） |
| PSO 编译开销影响首次请求延迟 | 中 | 低 | stage 10 加 server 启动 warmup |
| M1 Pro 最优 tile 不在 3 candidates 里 | 中 | 中 | stage 10 加候选，stage 9 接受次优 |

---

## 11. 后续 Stage（informational）

- **Stage 10** — 扩 device 分级 + 加 candidates + first-run profiling + warmup
- **Stage 11** — 扩 quant scheme（Q4_K_M / Q5_K_M / Q8_0 / bf16）+ GGUF loader
- **Stage 12** — Decode kernel 自研（vector qmv + split-K）

---

## 12. 设计原则记录（来自 brainstorming）

- **不照搬 llama.cpp 或 mlx**——它们都硬编码 tile size，是 stage 9 要超越的设计错误，不是 to align 的目标
- **device-aware + quant-aware tile selection** 是 ironmlx 跨 Apple Silicon 产品线的差异化设计点
- 数值正确性以 mlx-lm 输出作为参照（top-1 argmax + atol），但**实现路径独立**
- 测量用 ironmlx-bench-kernel + iron-bench 多维度数据，不只盯单一指标

参见 `memory/feedback_design_philosophy.md` + `memory/feedback_device_aware_tile.md`。
