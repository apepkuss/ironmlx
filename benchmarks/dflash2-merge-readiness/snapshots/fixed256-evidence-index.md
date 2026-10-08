# 四应用短 prompt 固定 256 对比：证据索引

2026-10-08。全部文件的 SHA256 见本目录 `MANIFEST.sha256`，覆盖本目录和 `reports/four-app-short-fixed256/`，在本索引写完后生成，不含清单自身。2026-10-08 修订后重新生成（219 项）；修订前的 217 项清单原样保留为 `MANIFEST.pre-revision.sha256`，差异见 `docs/revision-notes.md`。原始结果、日志和二进制都在 git 忽略的 `reports/` 下。

## 文件

**协议与报告：**
- `docs/protocol.md`：定稿版，sha256 ce0cf0fa…，正式运行 `formal-v1` 登记的就是这一版；
- `docs/report.md`：结果报告（2026-10-08 修订了解释文字，数值未变）；
- `docs/revision-notes.md`：修订说明，列出改了哪些表述、依据和身份关系。

**Splash 内存诊断：**
- `docs/splash-memory-protocol.md`；
- `docs/splash-memory-results.md`。

**脚本：**
- `scripts/run_fixed256.py`：正式运行和预检的采集脚本。复用 `benchmarks/four-app-short-v1/scripts/` 中的 `run.py`、`plan.py`，以及冻结的客户端、采样器和清理函数；这些脚本的哈希记录在各运行的 `run.json` 中。
- `scripts/analyze_fixed256.py`：分析脚本；
- `scripts/splash_memory_diag.py`：Splash 内存诊断脚本。

**分析结果：** `evidence/formal-v1-analysis.json`。

## 原始数据（`reports/four-app-short-fixed256/results/`）

| 标签 | 性质 | 状态 |
| --- | --- | --- |
| `splash-memory-diag-v1` | Splash 内存归属诊断，不计分 | 完整 |
| `preflight-b4-v1` | B4 预检 | 在 oMLX 的 S1 停止。原因是检查规则沿用了旧的 `finish=stop` 要求，作为失败记录保留 |
| `preflight-b4-v2` | B4 预检：12 个会话、48 个目标请求；含 A/B/C/D 的最小内存核对 | 完整，全部通过 |
| `formal-v1` | 正式四轮：64 个会话、288 个计分请求 | 完整，全部有效 |

**每次运行包含：**
- `run.json`：身份冻结、启动配置、逐步骤的请求（含完整 SSE 和输出）、原生状态快照、缓存与推测证据、内存窗口和阶段峰值；
- 每个会话的 `.server.log` 和 `.footprint.jsonl`（进程树时间序列，目标间隔 50 ms；`formal-v1` 实际相邻间隔中位数 55.0 ms、最大 170.8 ms，见报告第 1 节）；
- `telemetry.jsonl`；
- 控制台日志 `<标签>.console.log`。

**TensorFold 1.0.2 核对：** `reports/four-app-short-fixed256/tensorfold-1.0.2-check/`，包括 release 身份、发行包哈希与包内校验、能力清单、带 `--drafter` 启动被拒的命令和输出，以及源码事实摘录。

## 复用的外部证据

- **B1 固定 256 预检：** `benchmarks/four-app-b1-fixed256-preflight/` 与 `reports/four-app-b1-fixed256-preflight/`。
  - B、C、D 用 `fixed256-v1`，A 用 `fixed256-fix-v2`；
  - 身份和配置一致性见协议的“预检结果与定稿”一节。
- **IronMLX 候选：** `reports/four-app-b1-fixed256-preflight/fix/binaries/ironmlx-fixed256-fix-8271e413`（sha256 8271e413…），产品修复提交 5273c0b6。

## 缺口

- Splash 诊断中 `footprint --unmapped` 需要 root，没有取得；引擎 resident 比 footprint 少约 1.9 GB 的原因没有完全解释。
- IronMLX 的张量批路径没有覆盖。
- oMLX 的 adaptive 配置值在实际批量 DFlash 路径中被忽略（服务日志 `Batched DFlash ignores dflash_verify_mode`，源码 `engine_pool.py` 第 3641–3703 行），这不表示没有 target 验证或推测解码；Splash 不公开逐轮的验证宽度。
- 没有做回答质量审阅。
- IronMLX B4 吞吐三组整体随轮次下降，其中 S1、S3 单调下降，首尾降幅约 10%–13%，原因未明，没有开展诊断实验。
- 两个原因不明的间歇性库测试失败仍未解决。
