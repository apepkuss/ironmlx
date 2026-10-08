# 固定 256 token 预检：证据索引

2026-10-08。全部文件的 SHA256 见 `benchmarks/four-app-b1-fixed256-preflight/MANIFEST.sha256`，覆盖本目录和 `reports/four-app-b1-fixed256-preflight/`（git 忽略的本地产物）；清单在本索引写完后生成，不含清单文件自身。二进制、模型和原始结果都不进入 Git。

## 结论

| 项目 | 状态 |
| --- | --- |
| 四应用 B1 固定 256 预检（`fixed256-v1`） | 流程跑完。oMLX、Splash、TensorFold 的六个任务都固定 256 合格；IronMLX 六个目标请求全部在服务端失败（`KVCache cap … exceeded`），客户端未收到错误 |
| IronMLX 修复（提交 1） | 尾部 KV 容量与 OpenAI 失败传播已修复；针对性测试和变异检查都符合预期 |
| IronMLX 修复后验证（`fixed256-fix-v2`，最终候选） | 六个目标请求都是 256/256/length，固定 256 合格；默认探针返回 stop，ignore_eos 探针返回 length（原生 256） |
| 四应用固定 256 正式比较 | 未开展 |

## 身份

| 对象 | 位置 | SHA256 |
| --- | --- | --- |
| 已验收的 B4 二进制（修复前，保留） | `reports/dflash2-long-input/binaries/ironmlx-long-baseline-8d3a42dc` | 8d3a42dc8ee05cc805791a0f6acaf907f48a71bc5500f92b46630fa9a726cc65 |
| 早期修复候选（`fixed256-fix-v1`） | `reports/four-app-b1-fixed256-preflight/fix/binaries/ironmlx-fixed256-fix-84b82b06` | 84b82b0651ed36aa296590fc544a92ad961c20a1db715f842318805b67e77443 |
| **最终修复候选**（对应提交 1 的源码；`fixed256-fix-v2`） | `reports/four-app-b1-fixed256-preflight/fix/binaries/ironmlx-fixed256-fix-8271e413` | 8271e41393de7ff4ba7d7497b87f69239b539201899bf3fef9d0987f009b0ef6 |
| 最终候选的源码补丁 | 同目录 `ironmlx-fixed256-fix-8271e413.source.diff` | f29ff2d7eb3999c78727be3caf86da6695166062627e25010db08865fcc5f25b |

**竞品和模型：**
- oMLX 0.7.0、Splash 1.3.0（`--kv-format bf16`）、TensorFold 0.6.6；
- target 快照 3e6447f0，drafter 快照 50307d4c。
- 身份核对复用 `benchmarks/four-app-short-v1/scripts/run.py` 的 `freeze`，记录在各运行的 `run.json` 中。

## 文件

**协议：** `docs/protocol.md`，含修订 2（IronMLX 修复验证）和修订 3（判定规则拆分、最终候选）。

**报告：**
- `docs/results.md`：四应用预检结果，含更正；初版保留为 `results.v1.md`；
- `docs/fix-verification.md`：根因、修复、测试和验证。

**脚本（同一运行使用的版本各自保留）：**
- `scripts/run_preflight.py`：revision 3，用于 `fixed256-fix-v2`；
- `scripts/run_preflight.r2.py`：revision 2，用于 `fixed256-fix-v1`，哈希 07a986ab…；
- `scripts/run_preflight.v1.py`：初版，用于 `fixed256-v1`，哈希 70868192…。

**判读证据：**
- `evidence/strict-reassessment-r3.json`：三次运行按 revision 3 的离线判读；
- `evidence/strict-reassessment.json`：按 revision 2 的判读。

**原始结果（`reports/four-app-b1-fixed256-preflight/results/`）：**
- `fixed256-v1/`：四应用；原始 SSE、服务端日志、footprint 采样都保留，未改动；
- `fixed256-fix-v1/`：IronMLX，早期候选；
- `fixed256-fix-v2/`：IronMLX，最终候选；
- 各运行的 `.console.log`。

**测试证据（`reports/four-app-b1-fixed256-preflight/fix/tests/`）：** 每项有 `.command.txt`、`.output.log`、`.exit_code` 三个文件。
- 单元测试、OpenAI 测试；
- 两组 OpenAI 变异检查；
- 真实模型尾部测试和它的变异检查；
- 首次未覆盖 M5 flat-tree 的负面结果（最小化重跑）。

**检查日志：**
- 最终源码：`fix/checks-final/`，fmt、nightly fmt check、clippy、release build、库测试首次运行和重跑；
- 早期候选：`fix/checks/`。

## 缺口

- 首次真实模型测试（未安装 M5 profile）和它的变异检查，当时没有保存完整输出，只能最小化重跑补证。
- 最终源码的库测试首次运行时有两个间歇性失败（`mtp_prefill_vl_uses_paged_ssd_prefix_cache_on_exact_hit`、`rejects_a_second_backend_and_releases_on_drop`）。它们所在的文件没有改动，重跑通过，原因未调查。
- 修复后没有做性能和内存测量，也没有在 B4 批量路径上用真实请求验证容量改动。
- 四应用固定 256 正式比较尚未进行。
