# B4 阶段验收汇总

2026-10-07。Boss 的决策：
- 保留第 1 项；
- 放弃 2b 候选；
- 2a 本轮不做，第 2 项不保留实现；
- 不追加专项确认，不补跑。

## 最终保留内容

**本轮唯一保留的优化：第 1 项，ragged 窗口草稿合批。**
- 草稿模型新增 `propose_greedy_ragged_on`：投影、MLP、lm_head 和选择器对所有行展平计算，权重只读一遍；每行的上下文投影、RoPE、草稿缓存和注意力仍走单行路径。
- 另保留三个默认关闭的诊断：步骤日志、草稿逐行对照、scatter 屏障。

**改动的文件（基于 dev 9f9c6580，提交 d27a9306，未推送）：**
- `ironmlx-lm/src/models/dflash2/{attention,layer,model}.rs`
- `ironmlx-runtime/src/core/{dflash2.rs,dflash2_actor.rs,mod.rs}`
- `ironmlx-runtime/src/core/dflash2_step_diagnostic.rs`（新文件）

**已撤回第 2 项的全部代码：**
- ragged 草稿预算常量 `RAGGED_LINEAR_DRAFT_BUDGET`；
- 预算覆盖开关 `IRONMLX_DIAGNOSTIC_DFLASH2_RAGGED_BUDGET`；
- 2a 的 ragged 注意力计时诊断，包括 `take_ragged_attention_us` 和 `StageClock::note`。

撤回前的工作区补丁保存为 `reports/dflash2-b4-throughput/item2-abandoned-worktree-before-restore.diff`，与 2b 候选 f6a40e20 的冻结源码逐字节相同。

## 源码与构建身份（`evidence/final-source-identity.json`）

- **最终源码：** 与第 1 项冻结源码 `ironmlx-item1b-dev-8d3a42dc.source.diff` 只差一行，即默认关闭诊断里的 clippy 修正 `checks.is_multiple_of(100)`。全部源文件已与重建的 8d3a42dc 源码逐字节核对。最终 diff 保存为 `reports/dflash2-b4-throughput/final-source.diff`。
- **构建：** `MLX_DIR=/Users/xin/.local/mlx cargo build --release`（workspace）。
- **产物：** `ironmlx` 的 sha256 为 8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65，与正式测量的 B 二进制逐字节相同。

## 检查（`checks/final/`，最终源码）

- `cargo fmt` 已执行。
- `cargo +nightly fmt --all -- --check`：exit 0。
- `cargo +nightly clippy --all-features --workspace -- -D warnings`：exit 0。
- `cargo build --release`：exit 0，无警告。
- lib 测试：ironmlx-lm 615 项，ironmlx-runtime 543 项，全部通过。

## 阶段验收结果（复用已有证据，不重新测量）

第 1 项正式测量的 A 就是整轮基线 4fc49e92，B 的二进制与最终产物逐字节相同，所以“第 1 项对上一个已验收版本”和“最终版本对整轮基线”是同一组比较。这里不重新测量，也不以整体验收为由重复测 TTFT。

| 指标 | 最终版本 / 整轮基线 [95% 区间] | 证据 | 门槛 |
| --- | --- | --- | --- |
| B4 吞吐 | **1.105** [1.088, 1.114] | `item1-formal-v1` | 满足 |
| B4 整批完成时间 | **0.905** [0.898, 0.919] | 同上 | 满足 |
| B4 逐请求 TTFT | 0.937 [0.842, 1.121] | 同上 | 满足 |
| B4 逐请求 E2E（报告项） | 0.868 [0.861, 0.883] | 同上 | — |
| B1 短输入 Decode / E2E | 1.000 / 1.000 | 同上 | 满足 |
| B1 短输入 TTFT | 正式 8 轮：1.039 [0.937, 1.200]，**未通过（保留）**；专项确认 24 轮：**0.986 [0.950, 1.043]，通过** | `item1-formal-v1`、`b1-ttft-confirmation-v1` | 按 Boss 授权的专项确认判为通过 |
| B1 长输入 TTFT | 1.002 | `item1-formal-v1` | 满足 |
| 内存峰值 | 1.004 | 同上 | 满足 |
| token ID 正确性 | 21/21 一致 | `item1-correctness-v1` | 满足 |

**参考：** 专项确认的同一批会话中，B4 吞吐 1.114 [1.107, 1.117]，整批完成时间 0.898。

**限制：**
- B1 短输入 TTFT 的专项区间上界为 1.043，点估计通过并不等于在统计上排除了 3% 的退化。
- 只测了 Qwen3.8-27B-4bit 加 DFlash2 草稿，以及本轮固定的 B4 工作负载。

**收益口径：** 只计入第 1 项的 B4 收益，吞吐约 +10.5%。已放弃的 2b 在正式测量中的 +3.2% 不计入。

## 未计入的事项

- **2b 未通过。**
  - 依据：B4 吞吐 +3.2%，但 B1 短输入 TTFT 增加 20.5%，区间 [1.058, 1.285]；knowledge-1 和 code-3 都有 7/8 轮候选更慢。
  - 只能判定这个候选未通过，不能断定缩短预算就是退化的原因。
- **两次 TTFT 异常列为独立问题。** 第 1 项和 2b 的正式测量中，B1 短输入 TTFT 都偏慢，后续诊断都没有复现。本轮不扩展诊断，也不把查清根因作为收尾前提。

## 下一步

1. **长输入优化：** 先核对当前耗时和瓶颈，固定有限的优化清单，再逐项实施和验收。
2. **竞品对比：** 全部计划内优化完成并冻结版本后，才做四应用对比，对象为 IronMLX、oMLX 0.7.0 正式版、Splash 1.3.0、TensorFold 0.6.6。
   - 先做短 prompt 4 轮，覆盖 B1 和 B4，性能指标不变。
   - 完成后汇报，并询问 Boss 是否立即做 32K 4 轮；未经确认不启动。
