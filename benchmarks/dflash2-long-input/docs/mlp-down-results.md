# mlp_down 路径调整：验收结果 —— 未通过，已撤回

2026-10-07。协议：`docs/protocol-mlp-down.md`（sha256 9c62a1f8…，在正确性会话和正式运行之前定稿）。分析结果：`evidence/mlp-down-formal-v1-analysis.json`。

**本项的身份：**
- A：已验收的 B4 版本，二进制 8d3a42dc；
- B：候选，二进制 ae9a9ef7，补丁见 `evidence/mlp-down-candidate.patch`。

L1 的 3% 可行性筛选仍记为“未通过”，不改判。

## 运行完整性

- **正确性（`results/mlp-down-correctness-v1`）：** 每臂 1 个 30K 会话和 1 个短输入会话，开 token-id 记录。22 个请求的 prompt token、生成 token 和结束状态全部一致（30K 3 个，短输入 19 个）。
- **正式运行（`results/mlp-down-formal-v1`）：** 关闭全部诊断。30K 12 个配对轮次和短输入 6 个配对轮次全部完成，无无效会话，没有触发停止。138 个正式请求的输出文本哈希与同轮基线全部一致。
- **路径核对：** A 的服务日志有 `down QMM BM128 active`，B 没有，只有 gate-up 的标记。候选确实让 down 走了 MLX 原生 QMM。
- **冒烟检查 `results/mlp-down-smoke-v1`：** 只用于检查脚本，不计入结论。

## 结果（候选 / 基线；各轮中位数之比，配对轮次 bootstrap 95% 区间）

| 指标 | 比值 [CI95] | 门槛 | 判定 |
| --- | --- | --- | --- |
| 30K code-1 TTFT | **1.010** [1.006, 1.013] | ≤ 0.985 且上界 < 1.00 | **未通过** |
| 30K knowledge-1 TTFT | **1.008** [1.001, 1.013] | ≤ 0.985 且上界 < 1.00 | **未通过** |
| 30K code-1 E2E | 1.009 [1.006, 1.011] | ≤ 1.00 | **未通过** |
| 30K knowledge-1 E2E | 1.004 [1.001, 1.009] | ≤ 1.00 | **未通过** |
| 30K code-1 Decode | 0.999 [0.995, 1.002] | ≥ 0.99 | 通过 |
| 30K knowledge-1 Decode | 0.993 [0.983, 1.005] | ≥ 0.99 | 通过 |
| 30K 进程峰值内存 | 0.999 | ≤ 1.01 | 通过 |
| B1 code-1 TTFT / E2E / Decode 下界 | 1.003 / 1.005 / 0.990 | ≤ 1.03 / ≤ 1.03 / ≥ 0.97 | 通过 |
| B1 knowledge-1 TTFT | **1.075** [0.848, 1.363] | ≤ 1.03 | **未通过** |
| B1 knowledge-1 E2E / Decode 下界 | 1.000 / 0.999 | | 通过 |
| B1 code-3 TTFT / E2E / Decode 下界 | 0.997 / 1.003 / 0.986 | | 通过 |
| B4 吞吐（3 个批次组合） | 1.000、1.000、1.000 | ≥ 0.99 | 通过 |
| B4 会话进程峰值内存 | 1.003 | ≤ 1.01 | 通过 |
| token ID（正确性会话） | 22/22 一致 | 全部一致 | 通过 |
| 正式请求输出哈希 | 138/138 一致 | 全部一致 | 通过 |

**绝对值（中位数）：**
- 30K TTFT：code-1 51.94 → 52.45 s，knowledge-1 52.06 → 52.49 s。
- B4 吞吐：120.5 / 109.4 / 126.8 tok/s，两臂基本相同。

**时间顺序和漂移：**
- 第 0 轮的 A 是全程第一个会话（code-1 44.9 s），之后的会话稳定在 51–53 s。
- 逐轮配对比值中，code-1 候选 12/12 轮更慢：第 0 轮 1.142，其余 11 轮 1.004–1.014。knowledge-1 候选 11/12 轮更慢：第 0 轮 1.032，其余 1.000–1.014。
- 去掉第 0 轮，变慢方向仍然一致。所以结论不来自个别轮次。

**B1 knowledge-1 的短输入 TTFT：**
- 6 轮的 A、B 值分别在 0.109–0.146 s 和 0.109–0.164 s 之间，区间 [0.85, 1.36] 很宽。
- 短输入的 prompt 不会形成 2048 行的分块，不会进入被修改的路径。这一项按门槛记为未通过，原因没有确认，不改判。

## 结论：未通过，已撤回

- 预期的 TTFT 改善没有出现：两个 30K 任务的 TTFT 反而变慢约 1%，区间整体在 1.00 以上；E2E 也随之变差。
- 微基准中 down 形状上 MLX 原生 QMM 快约 14%（`l1-chain-v1`），但在真实服务的 30K prefill 中没有转化为收益，反而略慢。原因没有证据，不做推断；这个对照本身再次说明，算子加速不能代替端到端收益。

## 撤回与恢复

- 撤回前的工作区补丁保存为 `reports/dflash2-long-input/binaries/mlp-down/candidate-at-revert.source.diff`，与 `candidate.source.diff` 逐字节相同。
- `mlx-sys/shim/src/prefill_qmm_mtile.cc` 已恢复到 HEAD 8e6d01bc，工作区不再有生产代码改动。
- 用 `MLX_DIR=/Users/xin/.local/mlx cargo build --release` 重新构建，`target/release/ironmlx` 为 8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65，与已验收的 B4 二进制逐字节相同（日志：`binaries/mlp-down/revert-rebuild.build.log`）。
- 候选二进制 `binaries/mlp-down/ironmlx-mlp-down-candidate-ae9a9ef7` 和全部原始证据都已保留。
