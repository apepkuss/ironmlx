# B4 吞吐优化轮次 —— 测量协议

2026-10-06 写定，第 1 项正式测量之前。正式测量与诊断分开；本协议之后的样本数、顺序、门槛不改，不补跑到通过。

## 整轮固定项

- **基线：** dev 9f9c6580 构建的 `ironmlx-round-baseline-4fc49e92`（`reports/dflash2-b4-throughput/binaries/*.identity.json`）。
- **构建：** 所有二进制用 `MLX_DIR=/Users/xin/.local/mlx cargo build --release -p ironmlx`，MLX 0.32.2 fork 73ad5df2。
- **模型：** target `mlx-community--Qwen3.8-27B-4bit/3e6447f0…`，draft `z-lab--Qwen3.8-27B-DFlash2/50307d4c…`。
- **服务配置：** 产品默认值（M5 profile 自动启用），`--max-sequences 4`、`--max-cache-cap 8192`，不开前缀缓存，端口 18490。环境与 App 相同，正式计时不开任何诊断。见 `scripts/common.py`。
- **工作负载：**
  - **B4：** 沿用 b1-b4-concurrency-three-app-v1 的批次组合（code-1/2/3、knowledge-1；knowledge-2/3、code-1/2；code-3、knowledge-1/2/3）和固定预热批。四个请求同时发出，greedy，关闭思考，max_tokens 4096。四个请求的回答长度相差较大（约 450–1800 token）。
  - **B1 短输入：** code-1、knowledge-1、code-3，逐个发出。
  - **B1 长输入：** 30K code-1，max_tokens 1024。

## 每项的正式验收（`scripts/run_formal.py`、`scripts/analyze_formal.py`）

**对照：** A 为上一个已验收版本（第 1 项的 A 是整轮基线），B 为候选。

**B4 会话（每轮每臂各 1 个）：**
- 新进程，先跑预热批，再跑 3 个批次组合；
- 组合的起始位置按轮次轮换；
- 然后在同一进程里依次发 B1 短输入 3 个。

**轮次：**
- 8 个配对轮次，顺序 AB BA BA AB AB BA BA AB；
- 会话间隔 10 s，批次间隔 2 s。

**B1 长输入会话：**
- 另开 4 个配对轮次，顺序 AB BA BA AB；
- 每个会话是新进程，`--max-sequences 4`，容量 40960；
- 先预热一个短请求，再发 30K code-1 一次。

**统计：**
- 每个批次组合，取 8 轮的中位数，算 B/A 比值；三个组合的比值取几何平均。
- 95% 区间用配对轮次 bootstrap（重抽轮次，10000 次，种子 20261006）。

**通过门槛（全部满足才保留）：**

| 指标 | 条件 |
| --- | --- |
| B4 吞吐（输出 token ÷ 首个发出到最后完成） | 点估计 ≥ 1.03，且区间下界 > 1.00 |
| B4 整批完成时间 | 点估计 ≤ 0.97，且区间上界 < 1.00 |
| B4 逐请求 TTFT（12 个请求位置） | 中位数比值点估计 ≤ 1.05 |
| B1 短输入 Decode（3 任务） | 区间下界 ≥ 0.97 |
| B1 短输入 TTFT、E2E | 点估计 ≤ 1.03 |
| B1 长输入 TTFT | 点估计 ≤ 1.03 |
| 内存：B4 会话生命周期峰值中位数 | B/A ≤ 1.02 |
| 正确性 | 见下 |

**判定：**
- 全部满足：结论为“保留”。
- B4 吞吐有改善但某项非退化未过：结论为“修改”，写明原因，修改后按同一协议重新测量。
- 吞吐门槛未过：结论为“放弃”。
- 任何结论都保留原始数据和失败记录。

**正确性（`scripts/run_correctness.py`，单独会话，开 token-id 诊断）：**
- 每臂一个会话：预热批，3 个 B4 批次组合，B1 短输入 3 个，30K code-1 一次（容量 40960）。
- 逐请求比较生成的 token ID 和结束原因，要求 B 与 A 完全相同。
- 报告草稿接受率的差异，不设门槛。

## 整轮验收

计划内各项完成后，用同一协议比较最终版本与整轮基线，并冻结版本。之后才做四应用竞品对比；此前不做竞品测量。

## 第 1 项：ragged 窗口草稿合批

**二进制：**
- A：`ironmlx-round-baseline-4fc49e92`（整轮基线）。
- B：`ironmlx-item1b-dev-8d3a42dc`，源码见同名 `.source.diff`。

**改动内容：**
- ragged 线性窗口的草稿改为一次合批提案。投影、MLP、lm_head 和选择器对所有行展平计算，权重只读一遍；各行的上下文投影、RoPE、草稿缓存和注意力仍按单行路径执行。
- 附带三个默认关闭的诊断：步骤日志、ragged 草稿对照、scatter 屏障。

**诊断依据（不作验收）：**
- `diag-*-v1`、`item1*-v1`：4 行窗口草稿耗时 32.2 → 16.3 ms，接受率 0.371 → 0.368。
- 诊断运行的输出文本与基线一致。
- 批隔离版本（`ironmlx-item1-dev-85489e24`）草稿与逐行逐位一致，但草稿耗时只降到 26.9 ms，未进入正式测量。

**正式运行：** `results/item1-correctness-v1` 和 `results/item1-formal-v1`，各只运行一次。

## 第 2 项：verify 效率

2026-10-07 写定，在第 2 项正式运行之前。门槛、轮次、顺序和正确性检查与第 1 项相同（本文件“每项的正式验收”一节）。

**2a，合并 ragged verify 的逐行注意力：放弃。**
- 依据 `results/item2-attn-stages-v1`（二进制 `ironmlx-item2-diag-4de81a2c`）：逐行注意力每个窗口约 6 ms，从 2 行到 4 行几乎不变。
- 这个数字含每层两次计时同步，即使全部去掉也只占整批时间的 6.0%。
- 实际可省的部分预计只有 1–2%，低于 3% 的保留门槛。

**2b，ragged 窗口减小草稿预算：进入正式测量。**
- 依据：
  - 接受长度分布（item1b 屏障诊断）：4 行窗口中，接受 0 个的占 28%，接受满 7 个的只占 16%。
  - 预算诊断 `results/item2b-budget-{none,k3,k4,k5}-v1`（二进制 `ironmlx-item2b-diag-83e22bf5`）中 k=3 最好。这些组依次运行，没有交错，所以只用来选候选，不作结论。
- A：`ironmlx-item1b-dev-8d3a42dc`（第 1 项保留版本）。
- B：`ironmlx-item2b-dev-f6a40e20`。ragged 线性窗口（2 行及以上）的草稿预算为 min(3, 行预算)，单行窗口不变；另附两个默认关闭的诊断（ragged 注意力计时、ragged 预算覆盖）。
- 正式运行：`results/item2b-correctness-v1` 和 `results/item2b-formal-v1`，各只运行一次。
