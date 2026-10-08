# 长输入阶段证据索引与冻结记录

2026-10-07 冻结。完整的文件级 SHA256 见 `benchmarks/dflash2-long-input/MANIFEST.sha256`，覆盖本目录和 `reports/dflash2-long-input/` 下的全部文件，在本索引写完后生成，不含清单文件自身。大型原始产物只登记哈希，没有复制。

## 结论总表

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| B4 第 1 项：ragged 窗口草稿合批 | **保留**（已提交 d27a9306、8e6d01bc） | `benchmarks/dflash2-b4-throughput/docs/b4-stage-acceptance.md` |
| B4 第 2 项：2b 候选 | 放弃 | 同上 |
| B4 第 2 项：2a | 本轮不做 | 同上 |
| 缓存复用 | 本轮不做 | 同上 |
| 长输入 L2：调大 prefill 分块 | 收益不足（明显变差） | `docs/l1-l2-diagnosis-summary.md` |
| 长输入 L1：3% 可行性筛选 | 未通过，不改判 | 同上，第 2 版 |
| 长输入 L1：新 QMM 内核方向 | 证据不足，未实施 | 同上 |
| mlp_down 改走 MLX 默认 QMM（独立验收） | **未通过，已撤回** | `docs/mlp-down-results.md`、`docs/mlp-down-closeout.md` |

**最终状态：**
- 本轮只保留 B4 草稿合批，长输入阶段没有新增保留项；
- 内部有限优化清单已经结束，不再追加候选；
- 竞品对比没有启动。

## 身份

**源码：** 分支 `perf/dflash2-b4-throughput`，HEAD 8e6d01bc55582aa28b16943a6a7ea854e5d80b12。冻结时已跟踪文件与 HEAD 完全一致，没有 mlp_down 候选残留。

**二进制：**

| 角色 | 文件 | SHA256 |
| --- | --- | --- |
| 当前 release 产物（冻结状态） | `target/release/ironmlx` | 8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65 |
| 逐项基线：已验收 B4 版本 | `reports/dflash2-long-input/binaries/ironmlx-long-baseline-8d3a42dc`（身份见同名 `.identity.json`） | 同上 |
| 整轮初始基线（只作历史追踪） | `reports/dflash2-b4-throughput/binaries/ironmlx-round-baseline-4fc49e92` | 4fc49e9223f62468cebae3837ed7c1867d4047821a29405903502a41e9f5b487 |
| mlp_down 候选（已撤回，二进制保留） | `reports/dflash2-long-input/binaries/mlp-down/ironmlx-mlp-down-candidate-ae9a9ef7` | ae9a9ef71c89d04350225c62c4a0aade00df0ddc853d27509933acdb45445a85 |
| L1 工具 chain 版 | `reports/dflash2-long-input/binaries/prefill-qmm-feasibility-5cd23fe3`，另一份在 `binaries/l1-tool/` | 5cd23fe3c3a948483b1756f8d65a99458de26cfc9ecbe52bcc83a384c6ee4a1e |
| L1 工具 pre-chain 版 | **已被覆盖，未保留** | 运行记录中为 50a47c6c37ba30408b6f3a29ae15e2692fb7e7818fb960415dab3880a5f0ab3f |

**构建：** `MLX_DIR=/Users/xin/.local/mlx cargo build --release`，MLX 0.32.2 fork 73ad5df2。

**模型：**
- target `mlx-community--Qwen3.8-27B-4bit/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`；
- draft `z-lab--Qwen3.8-27B-DFlash2/50307d4c4cde6860d4eee73e2547cd786fe8e8a4`。

## 文件位置

**协议（运行时版本，未追溯修改）：**
- `docs/protocol-l2-chunk-screen.md`（含运行前补充）；
- `docs/protocol-l1-qmm-feasibility.md`（含修订 1）；
- `docs/protocol-mlp-down.md`（9c62a1f8…）。

**摘要与勘误：**
- `docs/l1-l2-diagnosis-summary.md`（第 2 版），第 1 版保留为 `l1-l2-diagnosis-summary.v1.md`；
- `docs/mlp-down-results.md`；
- `docs/mlp-down-closeout.md`（勘误与收尾说明）；
- 本索引。

**脚本：**
- `scripts/common.py`、`diagnose_long.py`；
- `run_l2.py`、`analyze_l2.py`；
- `run_l1.py`、`analyze_l1.py`（第 2 版），第 1 版保留为 `analyze_l1.v1.py`；
- `run_mlp_down.py`、`analyze_mlp_down.py`。

**工具源码：** `tools/prefill_qmm_feasibility.rs`（chain 版，源码 c2577252…），`tools/prefill_qmm_feasibility.pre-chain.rs`（源码 f5f2d8f2…）。

**分析结果（`evidence/`）：**
- 长输入耗时核对：`long-timing-check-v1.json`；
- L2：`l2-path-v1-analysis.json`、`l2-correctness-v1-analysis.json`、`l2-perf-v1-analysis.json`；
- L1：`l1-wall-v1-analysis.json`、`l1-gpu-v1-analysis.json`、`l1-chain-v1-analysis.json`（第 1 版，含已更正的 3.05% 口径）、`l1-chain-v1-analysis-v2.json`、`l1-tool-identity.json`；
- mlp_down：`mlp-down-candidate.patch`、`mlp-down-formal-v1-analysis.json`、`mlp-down-formal-v1-output-check.json`。

**原始结果（`reports/dflash2-long-input/results/`，git 忽略；每个运行目录下的 `run.json` 是主记录）：**

| 运行 | 性质 | run.json SHA256（前 16 位） |
| --- | --- | --- |
| long-plain-v1、long-phases-v1、long-stages-v1 | 长输入耗时核对（诊断） | d48d19f3…、8b83ad51…、fa3b4818… |
| l2-path-v1 | L2 路径诊断 | fd417bd60474a639 |
| l2-correctness-v1 | L2 正确性 | 9d6fe28889947298 |
| l2-perf-v1 | L2 性能（18 个会话） | b852df2816e539fd |
| l1-identity-v1 | kernel 身份捕获（含 24 个 `.gputrace`，共 11 GB） | 99307ca266de8cbf |
| l1-wall-v1 | 单次墙钟（受降频影响，只用于首次调用和数值） | 03ae77def065e09b |
| l1-gpu-v1 | Metal System Trace（GPU 区间不可信，判为证据不足） | b28fd53c7fbb5eac |
| l1-chain-v1 | 连续执行每次耗时（修订 1 的正式指标） | 76efe03c46b1fbcf |
| l1-smoke-v1、l1-chain-probe-v1 | 冒烟与探测（诊断，没有 run.json） | 见 MANIFEST |
| mlp-down-smoke-v1 | 脚本冒烟，不计入结论 | a846efe9b5e6b461 |
| mlp-down-correctness-v1 | token-id 正确性，22 对，只有一轮 AB | 74dbb4dfb42c4276 |
| mlp-down-formal-v1 | 正式运行（36 个会话） | d070988ab5ad7ee4 |

**构建与检查日志（`reports/dflash2-long-input/binaries/mlp-down/`）：**
- **构建：**
  - `candidate.build.log`：候选构建，`cargo build --release` exit 0；
  - `baseline-rebuild.build.log`、`candidate-rebuild.build.log`：可复现性核对，分别得到 8d3a42dc 和 ae9a9ef7；
  - `revert-rebuild.build.log`：撤回后重建，得到 8d3a42dc。
- **检查（`checks/`）：**
  - `fmt.log` 和 `fmt-check.log` 均为空，exit 0；
  - `clippy.log`，exit 0；
  - `prefill_qmm_down-test.log`，通过；
  - 失败记录 `prefill_qmm_down-test.failed-bad-shard-path.log`。
- **补丁：** `candidate.source.diff` 与 `candidate-at-revert.source.diff` 逐字节相同，sha256 efa9c352…。

## 已知缺口和限制

- **L1 pre-chain 工具二进制（50a47c6c）已被覆盖。** identity、wall、gpu 三组运行用的是它。保留了源码快照，源码哈希与运行记录一致，但无法逐字节复现原二进制。
- **l1-smoke-v1 的源码和二进制都没有记录**（诊断，不计入结论）。
- **l1-chain-probe-v1 的二进制哈希当时没有记录**，推定为 5cd23fe3：它与 chain 正式运行之间没有重新构建。
- **mlp_down 正确性会话只有一轮 AB，没有原定的 BA 轮。** 偏差出在协议本身，详见 `docs/mlp-down-closeout.md`。
- **mlp_down 正式运行没有开 token-id 诊断。** 138 对请求只比对了输出哈希、finish_reason 和生成 token 数。
- **mlp_down B1 knowledge-1 短输入 TTFT 为 1.075，按门槛记为未通过，原因没有确认。**
- **L2 绝对时间有持续负载漂移。** 交错配对降低并平衡了漂移的影响，但不代表完全不受影响。
- **L1 结论只覆盖第 0 层和第 3 层的权重，以及随机激活。**
- **原始结果只在本机**（`reports/` 被 git 忽略），没有备份副本。

## 未提交范围

只有未跟踪目录 `benchmarks/dflash2-long-input/`。`reports/dflash2-long-input/` 被 git 忽略。本轮没有提交、推送或合并。
