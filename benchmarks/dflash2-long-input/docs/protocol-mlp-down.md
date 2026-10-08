# mlp_down 路径调整：独立验收协议

2026-10-07 定稿，在正确性会话和正式运行之前。文末的两项已由 Boss 确认。定稿之后，样本数、顺序、门槛、统计方法和停止条件都不再修改。

本项是对一个低成本路径调整的独立验收。L1 的 3% 可行性筛选仍记为“未通过”，不改判。

## 改动与身份

**改动：**
- `mlx-sys/shim/src/prefill_qmm_mtile.cc` 的资格判断只接受 gate-up 形状，down 形状（[2048, 17408] × [5120]）回到 MLX 原生 QMM；
- gate-up、其他投影、分块大小、模型、调度、缓存和其余配置都不变；
- 补丁见 `evidence/mlp-down-candidate.patch`。

**二进制（均由 `MLX_DIR=/Users/xin/.local/mlx cargo build --release` 在 HEAD 8e6d01bc 上构建）：**
- **A，已验收的 B4 版本：** `reports/dflash2-long-input/binaries/ironmlx-long-baseline-8d3a42dc`，sha256 8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65。
- **B，候选：** `reports/dflash2-long-input/binaries/mlp-down/ironmlx-mlp-down-candidate-ae9a9ef7`，sha256 ae9a9ef71c89d04350225c62c4a0aade00df0ddc853d27509933acdb45445a85。
- **可复现性：** 撤回补丁重新构建得到逐字节相同的 8d3a42dc，再打回补丁得到逐字节相同的 ae9a9ef7。构建日志在 `binaries/mlp-down/`。
- **整轮初始基线** 9f9c6580 / 4fc49e92 只作历史追踪，不参与本项 A/B。

**检查（`binaries/mlp-down/checks/`）：**
- `cargo fmt`、`cargo +nightly fmt --all -- --check`、`cargo +nightly clippy --all-features --workspace -- -D warnings` 均 exit 0；
- `cargo build --release` exit 0；
- 真实权重测试 `mlx/tests/prefill_qmm_down.rs`（ignored，用真实分片运行）通过：gate-up 仍走 BM128，down 在各种布局下都与原生 QMM 逐位一致，且不再出现 BM128 标记。
- 首次运行因分片路径未展开而失败，与改动无关，日志保留为 `prefill_qmm_down-test.failed-bad-shard-path.log`。

## 固定项

- **fixture、有效性定义、计时、内存采样：** 沿用 `scripts/common.py`，与 B4 阶段相同。有效请求要求：HTTP 200、无错误、收到 `[DONE]`、`finish_reason=stop`、生成 token 数大于 1 且小于上限。
- **模型快照：**
  - target `mlx-community--Qwen3.8-27B-4bit/3e6447f0…`；
  - draft `z-lab--Qwen3.8-27B-DFlash2/50307d4c…`。
- **其他固定项：** greedy（temperature 0、top_p 1），关闭思考，产品默认值（M5 profile 自动启用），App 环境，端口 18490，不开前缀缓存。
- **正式运行：** 关闭全部诊断和追踪，包括 token-id 记录。

## 测量安排（`scripts/run_mlp_down.py`）

**第 0 步，正确性（`--kind correctness`，先于性能运行）：**
- 每臂各跑 1 个 30K 会话和 1 个短输入会话，配置和请求与正式运行相同，顺序 A、B、A、B；
- 开 `IRONMLX_DIAGNOSTIC_DFLASH2_TOKEN_IDS`，覆盖全部请求：30K 的预热、code-1、knowledge-1；短输入的预热批、3 个批次组合、B1 短输入 3 个；
- 按“prompt token 序列 + 出现次序”配对，逐请求比较 prompt token、生成 token 和结束状态；
- 任何不一致都先判正确性失败，不进入性能运行。

**第 1 部分，30K：**
- 每个会话是新进程，`--max-sequences 1`、`--max-cache-cap 40960`、`--prefill-chunk-size 2048`；
- 先做短预热（max_tokens 64），然后按本轮的任务顺序发 30K code-1 和 knowledge-1，各一次，max_tokens 1024，间隔 5 s；
- 12 个配对轮次，臂的顺序为 AB BA BA AB AB BA BA AB AB BA BA AB（AB、BA 各 6 次）；
- 任务顺序：第 0、1、4、5、8、9 轮先 code-1，第 2、3、6、7、10、11 轮先 knowledge-1。AB 轮和 BA 轮中，两种任务顺序各占 3 轮；同一轮的两臂使用相同的任务顺序。

**第 2 部分，短输入（与 B4 阶段相同）：**
- 每个会话是新进程，`--max-sequences 4`、容量 8192；
- 先跑预热批，再跑 3 个批次组合（起始位置按轮次轮换），然后在同一进程里依次发 B1 短输入 code-1、knowledge-1、code-3；
- 6 个配对轮次，顺序 AB BA BA AB AB BA。

**顺序与间隔：** 先跑完第 1 部分，再跑第 2 部分；会话间隔 10 s。

## 统计（`scripts/analyze_mlp_down.py`）

- **比值方向：** 一律为“候选 / 基线”（B/A）。
- **点估计：** 每个任务（或批次组合）取各轮中位数，算中位数之比。
- **区间：** 配对轮次 bootstrap，有放回重抽轮次，10000 次，种子 20261008。
- **报告方式：** 每个任务分别判定；几何平均只作报告，不代替逐任务门槛。
- **内存：** 按部分取会话生命周期峰值中位数之比。
- **漂移：** 列出每个会话的时间顺序和绝对的 TTFT、E2E、Decode。

## 验收门槛（全部满足才通过）

| 指标 | 条件 |
| --- | --- |
| 30K code-1 TTFT | 点估计 ≤ 0.985，且区间上界 < 1.00 |
| 30K knowledge-1 TTFT | 点估计 ≤ 0.985，且区间上界 < 1.00 |
| 30K code-1、knowledge-1 的 Decode 吞吐（分别） | 比值点估计 ≥ 0.99；CI95 只报告，不设门槛 |
| 30K code-1、knowledge-1 的 E2E 耗时（分别） | 比值点估计 ≤ 1.00；CI95 只报告，不设门槛 |
| token ID（正确性会话） | 全部请求与基线一致（prompt token、生成 token、结束状态）；不一致先判正确性失败 |
| 正式请求的输出（正式运行） | 每个请求的输出文本哈希与同轮基线请求一致 |
| B1 短输入 TTFT（3 个任务分别） | 点估计 ≤ 1.03 |
| B1 短输入 E2E（3 个任务分别） | 点估计 ≤ 1.03（B4 阶段已批准的约束） |
| B1 短输入 Decode（3 个任务分别） | 区间下界 ≥ 0.97（B4 阶段已批准的约束） |
| B4 吞吐（3 个批次组合分别） | 点估计 ≥ 0.99 |
| 进程峰值内存（30K 会话、B4 会话分别） | 生命周期峰值中位数之比 ≤ 1.01 |

## 停止与无效规则

- 不重试，不追加轮次，不删样本。
- 出现以下情况立即停止剩余运行，保留全部证据：
  - 任何无效会话（服务启动失败、无效响应）；
  - 二进制在运行中被改变；
  - 正确性会话中首次出现 token 分叉；
  - 正式运行中首次出现输出哈希不一致。
- 因无效会话或基础设施故障停止、导致计划轮次不全时，记为“验收未完成”，不重跑。
- 因 token 分叉、输出哈希不一致或其他明确失败停止时，记为“未通过”。
- 小样本结果再好，也不提前宣布通过。

## 结论处理

- **通过：** 保留最小实现和完整验收材料，汇报实际收益和全部门槛，等待后续指令。
- **未通过或收益不足：** 保存补丁和证据，撤回候选的生产代码，确认恢复到已验收的 B4 状态（重新构建并核对 8d3a42dc），不继续调参或扩展。
- 本项不提交、不推送、不合并，不清理原始证据。

## Boss 确认的两项（2026-10-07）

**1. 30K 的 Decode 和 E2E：**
- code-1 和 knowledge-1 分别验收，比值为“候选 / 基线”；
- Decode 吞吐比值 ≥ 0.99，允许点估计最多退化 1%；
- E2E 耗时比值 ≤ 1.00，点估计不得退化；
- 两项都报告配对 CI95，但不设区间门槛；
- 使用与 TTFT 相同的完整配对轮次和统计方法，不跨任务平均，不追加轮次。

**2. token ID：**
- 正式运行全部关闭诊断；
- 另跑正确性会话比对 token ID（见第 0 步）；
- 正式请求比对输出文本哈希。依据：同一二进制在 `l2-perf-v1` 的 6 个 2048 会话和 `item1-formal-v1` 的 8 个会话中，每个任务的输出哈希都只有一个。
