# DFlash2 合并准备

2026-10-08。此目录是合并审查入口，不改写既有验收结论。

## 集成范围

- 准备分支：`test/dflash2-merge-readiness`。
- dev 基线：`223bfd3f9428b254f52e7a73625e66860f3d1885`；创建后再次 fetch 核对未变化。
- 功能来源：`perf/dflash2-b4-throughput`，最终提交 `7669ac046045d0195537d755da6f23775bab047e`。
- 顺序应用五个提交的累积内容：`d27a9306e`、`8e6d01bc5`、`183ccea1f`、`5273c0b67`、`7669ac046`。无冲突，保留最新 dev 的 EmbeddingGemma2 和文档调整。
- 本次新增：真实模型 B4 tensor 输出尾部回归用例、Chat Completions 提前 EOF 错误语义的中英文说明，以及此证据索引。
- 不含已撤回的草稿预算候选，不实施长尾或 commit/rollback 新优化。没有合并、推送 dev，也没有创建 PR。

## 证据与结论边界

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| ragged 草稿合批 | 原冻结负载下 B4 吞吐比 1.105，CI95 [1.088, 1.114]；token ID 21/21 一致 | [原验收](../dflash2-b4-throughput/docs/b4-stage-acceptance.md) |
| B1 非退化 | 首次 8 轮 TTFT 未通过；授权 24 轮专项确认点估计 0.986，通过原门槛；CI95 上界 1.043，不能称为统计上排除 3% 退化 | 同上，首次失败保留 |
| 长输入候选 | 已放弃，不计入收益 | [长输入证据](../dflash2-long-input/docs/evidence-index.md) |
| KV 临时容量与提前 EOF | 保留容量修复、API 错误修复、变异检查及首次失败日志 | [修复验证](../four-app-b1-fixed256-preflight/docs/fix-verification.md) |
| 修复后固定 256 服务测试 | 旧冻结二进制 8271e413 的 IronMLX 72/72 请求合格，其中包括 12 个 B4 会话；不是本次集成二进制的性能测量 | [报告快照](snapshots/fixed256-report.md) |
| tensor 批路径输出尾部 | 本次定向真实模型测试补齐；与服务端默认 ragged/tree 验收区分 | 下节 |
| 草稿预算重审 | B4 吞吐比 1.010，未达到 1.03 门槛，已撤回 | [结果快照](snapshots/draft-budget-results.md) |
| 长尾解码 | 有限诊断未找到足以支持实施的明确开销，不实施；不声称已证明与 B1 等价 | [诊断快照](snapshots/tail-diagnosis.md) |
| commit/rollback | 本轮不继续投入；存在未使用输出等可精简工作，但收益未知，不能断言低于 3% | [评估快照](snapshots/commit-rollback-review.md) |

原报告中“张量批未覆盖”是当时的准确记录，原文不修改；本次只补充下述指定配置的覆盖。旧库测试的两个间歇性失败、早期无效预检和通用 tree 变异未触发失败等负面记录仍有效，不因后续通过而删除。

## 本地集成验证

在准备工作区执行，使用 `/Users/xin/.local/mlx/mlx-env.sh` 的本地 NAX-enabled MLX。构建日志包含上游 C++ 头文件警告；Rust Clippy 的 `-D warnings` 检查通过，不能将构建描述为完全无警告。

必做检查：`cargo fmt`、`cargo +nightly fmt --all -- --check`、`cargo +nightly clippy --all-features --workspace -- -D warnings`、`cargo build --release`。日志位于本地 `reports/dflash2-merge-readiness/`；定向测试结果见 `checks.txt`。

新增用例 `qwen38_tensor_batch_reaches_output_tail_and_matches_b1` 的条件：

- Qwen3.8-27B-4bit 快照 `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`；DFlash2 快照 `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`，draft 运行时 4-bit。
- M5 profile、block 8、greedy、4 个相同短输入、max_new_tokens 256、空停止 token 列表；显式关闭 tree，以测试 tensor 批路径。
- 与 scheduler B1 参考逐 token 比较，全部 4 行必须生成 256 tokens，仅末尾返回 `length`，之后无额外输出。
- 必须实际执行 B4 tensor 窗口及输出尾部窗口；检查持久化 tensor cache 存在，且真实 Full-attention KV 容量为 `4 + 256 + 8 = 268`。
- 这是 core 定向测试，不是 API 默认调度性能测试，也不扩大为采样、多 prompt、其他模型或硬件的验收。

重现命令（先设置上述两个快照目录）：

```sh
source /Users/xin/.local/mlx/mlx-env.sh
cargo test --locked --release -p ironmlx-runtime --lib qwen38_tensor_batch_reaches_output_tail_and_matches_b1 -- --ignored --nocapture --test-threads=1
cargo test --locked --release -p ironmlx --lib missing_terminal_event -- --test-threads=1
```

GitHub CI 应检查本准备分支提交的实际 SHA，要求 Documentation quality 与 Rust, Swift, MLX, and App Bundle 两个作业通过。原 dev 的绿色 CI 不替代这次集成检查；此静态报告不预先宣称远端 CI 通过，最终状态以该 SHA 的运行记录为准。

## 后续材料保留方式

`snapshots/` 保存五份报告的逐字节副本，SHA256 见 `MANIFEST.sha256`。快照内的相对路径仍以原目录为基准；它们是归档报告，不是可独立重跑的数据包。原始数据、脚本、旧版报告、完整清单继续保留在来源工作区，本次没有移动、删除或修改它们。

| 快照 | 来源工作区内路径 | 来源工作区 |
| --- | --- | --- |
| `fixed256-report.md` | `benchmarks/four-app-short-fixed256/docs/report.md` | `/Users/xin/workspace/perf-dflash2-b4-throughput` |
| `fixed256-evidence-index.md` | `benchmarks/four-app-short-fixed256/docs/evidence-index.md` | 同上 |
| `draft-budget-results.md` | `benchmarks/dflash2-ragged-draft-budget/docs/results.md` | `/Users/xin/workspace/perf-dflash2-ragged-draft-budget` |
| `tail-diagnosis.md` | `benchmarks/dflash2-tail-decoding/docs/diagnosis.md` | `/Users/xin/workspace/perf-dflash2-tail-decoding` |
| `commit-rollback-review.md` | `benchmarks/dflash2-commit-rollback-review/README.md` | `/Users/xin/workspace/perf-dflash2-b4-throughput` |

2026-10-08 本次重新运行各来源 `MANIFEST.sha256`：固定 256 为 219/219、草稿预算为 97/97、长尾为 25/25、commit/rollback 为 6/6，共 347 项全部通过。原始材料未完整纳入 Git；删除这些来源工作区前需另行安排证据保全。
