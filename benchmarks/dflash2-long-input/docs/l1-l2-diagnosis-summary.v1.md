# 长输入诊断摘要：L2 prefill 分块筛查与 L1 矩阵乘可行性

2026-10-07。只做诊断：没有开发新内核，没有修改生产代码或默认配置，没有提交。

**基线：**
- 逐项基线：提交 8e6d01bc，二进制 8d3a42dc；
- 整轮初始基线：9f9c6580 / 4fc49e92，本次未使用。

**协议：**
- `docs/protocol-l2-chunk-screen.md`，含一条运行前补充；
- `docs/protocol-l1-qmm-feasibility.md`，含修订 1。

原始数据在 `reports/dflash2-long-input/results/`，分析结果在 `evidence/`。

## L2：prefill 分块大小 —— 收益不足（实际明显变差）

**路径诊断（`l2-path-v1`，带诊断，只用于核对路径）：**
- **2048：** 15 块 2048 加 1 块 12 token。BM128 gate-up、BM128 down、D256 NAX 注意力都启用。
- **4096：** 7 块 4096 加 1 块 2060。三条特化路径都退出，注意力改走 `masked causal softmax fallback`。这条回退按 [24 头, 分块行数, 约 30K key] 生成完整的分数矩阵。
- **8192：** 3 块 8192 加 1 块 6156。路径与 4096 相同。

**性能测量（`l2-perf-v1`）：**
- 无诊断，6 轮拉丁方，共 18 个会话，全部有效；
- 每种配置每个任务取 6 轮中位数，再对两个任务取几何平均；
- 区间为配对轮次 bootstrap 的 95% 区间。

| 对 2048 的比值 | 4096 | 8192 |
| --- | --- | --- |
| TTFT | 1.127 [1.109, 1.153] | 1.175 [1.160, 1.224] |
| E2E | 1.101 [1.085, 1.127] | 1.158 [1.143, 1.205] |
| Decode 速度 | 0.979 [0.949, 0.986] | 0.952 [0.916, 0.962] |
| 生命周期峰值内存 | 1.283（+6.8 GiB） | 1.652（+15.7 GiB） |
| MLX 峰值 | 1.358 | 1.744 |

- 两个任务的 TTFT 和 E2E 都是 6/6 轮更慢；Decode 也是 6/6 轮更低。
- 绝对时间有持续负载漂移：2048 的 TTFT 从首个会话的约 46 s 升到后几轮约 54 s，三种配置同步变慢。比值都在同一轮内配对计算，不受这个漂移影响。

**正确性（`l2-correctness-v1`）：**
- code-1 在三种分块下 token ID 完全一致（466 个）。
- knowledge-1 在 4096 和 8192 下 prompt 相同，但生成到第 57 个 token 时与 2048 分叉，分别生成 285 和 321 个 token（2048 为 320），都正常结束。
- 结论：换分块大小会改变数值路径。

**解读：**
- 变慢同时来自两方面：分块固定开销的变化，以及退出 BM128 和 NAX 特化路径、改走生成完整分数矩阵的回退注意力。两者无法分开，因此不能说明“减少分块固定开销”有无收益。
- 现有数据只能说明：按当前代码直接把默认分块调大，TTFT、E2E、内存和输出一致性都会变差。

**建议：** 不实施，保持 2048。如果以后要重新评估更大分块，前提是 4096 以上也有对应的注意力和 QMM 特化路径，那属于清单外的新内核方向，本轮不展开。

## L1：prefill 矩阵乘可实现优化空间

**范围：**
- 根据 L2 结果，只测 M=2048；
- 使用目标模型的真实权重，按 DFlash2 通道的方式拼接，覆盖 6 个投影；
- 激活为 bf16，权重为 affine4、group 64，`transpose=true`，与生产 FFI 一致。

**kernel 身份（`l1-identity-v1`）：** 每个进程只跑一个变体，在 MLX GPU capture 中读取该进程创建的 pipeline。

| 形状 | production 模式 qmm | default 模式 qmm | bf16 |
| --- | --- | --- | --- |
| mlp_gate_up | ironmlx BM128 库，有 BM128 标记，未创建 MLX QMM pipeline | MLX `affine_qmm_t_nax…bm64_bn64_bk64` | 稠密 GEMM，无 QMM pipeline |
| mlp_down | 同上 | 同上 | split-K 稠密 GEMM（`steel_gemm_splitk_accum`） |
| gdn_in_proj | MLX NAX QMM（N=16480 不是 64 的倍数，用 `alN_false` 变体） | 同一个 kernel | 稠密 GEMM |
| gdn_out_proj、attn_qkv、attn_o_proj | MLX NAX QMM | 同一个 kernel | 稠密 GEMM |

只有两个 MLP 形状在两种模式下走不同 kernel。其余 4 个形状两种模式是同一路径，它们之间的差异只是进程间噪声。

**计时方法与失败记录：**
- **单次墙钟（`l1-wall-v1`）：** 每次调用前空闲 30 ms，测到 7–22 ms，主要是降频和同步开销，与 kernel 本身的耗时不成比例。只用于首次调用成本和数值比较。
- **Metal System Trace（`l1-gpu-v1`）：** 每次调用的 GPU 区间只有约 0.6–1.7 ms，折算吞吐不可信。原定的 GPU 时间方法判为证据不足，数据保留。
- **修订 1，正式指标（`l1-chain-v1`）：** 连续提交 8 次、一次 eval，总时间除以 8，作为“连续执行每次耗时”，不是 GPU 时间戳。6 个进程，模式顺序 P D D P P D，每个变体计时 15 次。

**连续执行每次耗时（各进程中位数的中位数；配对比值来自 3 对相邻进程）：**

| 形状 | 30K 调用次数 | 生产 QMM | MLX 默认 QMM | BF16（同一权重反量化） | P/D 配对比值 |
| --- | --- | --- | --- | --- | --- |
| mlp_gate_up | 960 | 14.46 ms（50.5 TF，BM128） | 15.35 ms（47.6 TF） | 12.96 ms（56.3 TF） | 0.942、0.943、0.942 |
| mlp_down | 960 | 9.13 ms（40.0 TF，BM128） | 7.90 ms（46.2 TF） | 16.46 ms（22.2 TF，split-K） | 1.148、1.127、1.162 |
| gdn_in_proj | 720 | 7.49 ms | 7.01 ms（同一 kernel） | 6.34 ms | 1.096、1.002、1.073（噪声） |
| gdn_out_proj | 720 | 3.02 ms | 2.93 ms（同一 kernel） | 2.36 ms | 0.964、0.990、1.036 |
| attn_qkv | 240 | 6.82 ms | 6.99 ms（同一 kernel） | 5.51 ms | 0.994、0.992、0.969 |
| attn_o_proj | 240 | 3.09 ms | 3.00 ms（同一 kernel） | 2.36 ms | 1.030、0.990、1.004 |

**数值：**
- 6 个形状里，生产 QMM 与 MLX 默认 QMM 的输出位都相同，BM128 的两个形状也一样。
- BF16 matmul 只在 mlp_down 上输出位不同（split-K 累加顺序不同），但相对 F32 参考的最大误差和平均误差与 QMM 相同。
- 所有变体都没有非有限值。

**首次调用和准备成本（`l1-wall-v1`）：**
- QMM 首次调用 10–36 ms，含 pipeline 编译。
- BF16 需要先反量化：每个形状 5–24 ms，额外显存为 N×K×2 字节，例如 gate_up 为 356 MB。全模型换成 BF16 权重大约要多 40 GB，不具备实际可行性，只作参考。

**按 30K prefill 估算的上限（乐观：算子单独测，未计其他算子与它的相互影响）：**
- 当前生产路径 6 个投影合计约 32.6 s，占 L2 无诊断 2048 TTFT 中位数 53.2 s 的 61%。
- **可直接实现的路径：** 只有 mlp_down 改用 MLX 默认 QMM（输出位相同）。可省 1.23 ms × 960 ≈ 1.18 s，约为 TTFT 的 2.2%；按冷态 TTFT 约 43 s 计约为 2.7%。3 对进程方向一致。
- **BF16 稠密参考的差距：** 合计约 3.24 s，约 6.1%，其中 gate_up 1.44 s、gdn_in_proj 0.83 s。要拿到这部分，需要让 QMM 接近稠密 GEMM 的效率，属于新内核方向。BF16 不是 QMM 的严格上限，这个差距也不是可直接获得的收益。

**限制：**
- 微基准运行时间短，GPU 温度比 30K prefill 持续满载时低；比例估算是在不同的热状态下做的。
- “76%”屏障占比和“39 TFLOPS”估算没有用于这里的任何判断。

**结论：**
- **mlp_down 改走 MLX 默认 QMM：收益不足。** 上限 2.2% 低于预先定的 3% 门槛。不过差异稳定、可解释、输出位相同、改动很小（只让 BM128 限于 gate-up 形状）。是否作为低风险小项单独验收，由 Boss 决定。若实施，建议门槛：
  - 30K 两任务 TTFT 配对比值点估计 ≤ 0.985，且区间上界 < 1.00；
  - token ID 全部一致；
  - 短输入 TTFT ≤ 1.03；
  - B4 吞吐不低于 0.99；
  - 内存比值 ≤ 1.01。
- **为 QMM 开发新内核以接近 BF16 稠密效率：证据不足。** 上限只有 6.1% 且偏乐观。现有 BM128 在 gate_up 上也只达到稠密参考的约 90%，没有证据表明有可实现的路径能拿到其中大部分。不建议在本轮进入实现。
- **gate-up 的 BM128：** 比 MLX 默认快 5.8%，配对一致，应保留。

## 文件清单

**`benchmarks/dflash2-long-input/`：**
- 协议：`docs/protocol-l2-chunk-screen.md`、`docs/protocol-l1-qmm-feasibility.md`；
- 摘要：本文件；
- 脚本：`scripts/run_l2.py`、`analyze_l2.py`、`run_l1.py`、`analyze_l1.py`；
- 工具源码：`tools/prefill_qmm_feasibility.rs`（chain 版，二进制 5cd23fe3），`tools/prefill_qmm_feasibility.pre-chain.rs`（identity、wall、gpu 三次运行使用的版本，源码哈希 f5f2d8f2，与运行记录一致；对应二进制 50a47c6c 已被覆盖，未保留）。工具在 `mlx/examples/` 下构建：`cargo build --release -p mlx --example prefill_qmm_feasibility`。
- 分析结果：`evidence/` 下 L2 的 path、correctness、perf，L1 的 wall、gpu、chain。

**`reports/dflash2-long-input/results/`：**
- L2：`l2-path-v1`、`l2-correctness-v1`、`l2-perf-v1`；
- L1：`l1-identity-v1`（GPU capture 共 11 GB）、`l1-wall-v1`、`l1-gpu-v1`、`l1-chain-v1`；
- 诊断，不计入结论：`l1-smoke-v1`、`l1-chain-probe-v1`。
