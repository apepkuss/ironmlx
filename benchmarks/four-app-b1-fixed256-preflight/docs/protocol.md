# 四应用 B1 固定输出 256 token —— 可行性预检协议

2026-10-08 写定，在运行之前。只判断可行性，不做性能排名。不修改产品代码，不重新构建，不提交。

## 对象和配置

- **应用：** 与 `four-app-short-v1` 相同的冻结对象：IronMLX 8d3a42dc（HEAD 183ccea1）、oMLX 0.7.0、Splash 1.3.0、TensorFold 0.6.6。
- **模型：** target 快照 3e6447f0、drafter 快照 50307d4c。
- **身份核对：** 复用该套件 `run.py` 的 `freeze` 和 `verify`。
- **启动配置：** 与 `four-app-short-v1` 相同（请求容量 4、上下文 8192），只有一处改动：Splash 加上 `--kv-format bf16`。
- **采样：** 都用 greedy（temperature 0、top_p 1），都关闭思考。

**KV 格式核对：**

| 应用 | 设置 | 核对方式 |
| --- | --- | --- |
| IronMLX | `--kv-quant` 默认为 none，KV 按 bf16 激活存储 | 服务参数说明和源码 |
| oMLX | model settings 中 `turboquant_kv_enabled` 默认为 False，本 runtime 没有开启 | 源码，以及 runtime 文件 |
| Splash | 显式传 `--kv-format bf16`；服务会把它传给引擎子进程 | 就绪时记录进程树命令行，确认引擎参数里有 `--kv-format bf16` |
| TensorFold | Mac lanes 路径上没有 KV 量化；`--kv-dtype` 只对 CUDA 生效 | 源码 |

## 固定长度机制

**源码核对的结果：**
- IronMLX、Splash、TensorFold 的服务端都解析请求中的 `ignore_eos`；
- oMLX 0.7.0 的请求模型没有 `ignore_eos`，也没有等效的最小长度参数，未知字段会被静默丢弃。

**决定：** 四个应用不是全部支持，所以都不启用 `ignore_eos`，统一用 `max_tokens=256`，再实测既定任务是否真的达到上限。

## 请求

- 应用顺序 A、B、C、D，串行运行，只用端口 18480；每个应用新开一个会话，会话之间间隔 10 s。
- **预热：** 冻结客户端的两个预热请求，请求参数不变（max_tokens 4096，自然结束）。
- **目标请求：** 六个短输入任务（fixture 138468e1），按 code-1 到 knowledge-3 的顺序各发一次，max_tokens 256。四个应用共 24 个。
- **探针：** 最后再发一个“只回答 OK”，max_tokens 256，单独判断模型是否会提前 EOS。
- **请求体：** 与冻结客户端相同（流式、单条 user 消息、关闭思考；oMLX 和 Splash 传 `reasoning_effort none`），只改 max_tokens。
- 请求间隔 1 s。不重试，不追加；失败照实记录，运行继续。

## 记录与判读

**逐请求记录：**
- 原生输出 token 数（usage.completion_tokens）；
- 统一 tokenizer 的重计数；
- finish_reason；
- `cached_tokens`；
- 错误；
- 原始 SSE 行和完整输出；
- 推测解码的原生计数，读取方式与上一轮相同。

**判读规则：**
- **预期情况：** 达到上限，即原生 256 个 token 且 finish_reason=length。
- **必须如实列出的情况：** 提前结束、超出上限、原生计数和统一计数之间的差异。
- **零缓存命中：** 所有目标请求和探针都必须为零命中。
- **结论的范围：** “这六个任务在这些应用上都能写到 256 个 token”，不能推广为“任意输入都能被强制写到 256 个 token”。没有 `ignore_eos` 时，只要模型提前输出 EOS，`max_tokens` 就无法保证长度。探针就是用来观察这种情况的。

**结果目录：** `reports/four-app-b1-fixed256-preflight/results/<label>`，与旧结果分开。

## 修订 2（2026-10-08，IronMLX 修复后的针对性验证，在 `fixed256-fix-v1` 之前写入）

**背景：** `fixed256-v1` 中，IronMLX 的六个目标请求都在服务端失败了，日志为 `KVCache cap … exceeded`。客户端只收到 usage 和 `[DONE]`，没有错误，也没有 finish_reason。原始响应和日志保留不动。

**判定规则（脚本 revision 2，`scripts/run_preflight.py`；原脚本保留为 `run_preflight.v1.py`）：**
- 一个请求有效，必须同时满足：HTTP 200、无错误、收到 `[DONE]`，并且 finish_reason 是 stop 或 length；
- 只有 HTTP 200 和 `[DONE]` 不算成功；
- 每个请求都列出异常：缺少终止原因、提前结束、超出上限、原生计数与统一计数不一致。

**本次验证（`fixed256-fix-v1`）：**
- 只跑 IronMLX，使用修复后的候选二进制 `reports/four-app-b1-fixed256-preflight/fix/binaries/ironmlx-fixed256-fix-84b82b06`；
- 模型、tree 配置（M5 profile，block 8，tree 15）、max_tokens=256、预热和六个任务都与 `fixed256-v1` 相同；
- 另外加一个探针：`ignore_eos=true` 的“只回答 OK”，max_tokens 256，没有其他停止条件。

**预期：**
- 六个目标请求都是原生 256 个 token，finish_reason=length，服务端没有错误；
- 默认探针允许提前 EOS 并返回 stop；
- ignore_eos 探针应生成满 256 个 token 并返回 length。

其他三个应用不重跑。

## 修订 3（2026-10-08，在 `fixed256-fix-v2` 之前写入）

**脚本 revision 3（`scripts/run_preflight.py`；revision 2 保留为 `run_preflight.r2.py`）分别记录三项：**
- `protocol_valid`：HTTP 200、无错误、收到 `[DONE]`、无 reasoning，finish_reason 为 stop 或 length；
- `fixed256_qualified`（只针对目标请求）：在 protocol_valid 的基础上，finish_reason 为 length，原生和统一 tokenizer 计数都是 256。只有满足这一项的请求才进入固定长度计分；
- `probe_expectation_met`：
  - 默认探针：protocol_valid，finish_reason 为 stop，允许提前结束；
  - ignore_eos 探针：protocol_valid，finish_reason 为 length，原生计数 256。统一计数可以更低，因为被忽略的 EOS 之后生成的特殊 token 不出现在可见文本中。

**最终候选：** 修复 + 回归测试的最终源码构建出 `fix/binaries/ironmlx-fixed256-fix-8271e413`（sha256 8271e41393de7ff4ba7d7497b87f69239b539201899bf3fef9d0987f009b0ef6）。

**`fixed256-fix-v2`：** 用最终候选和 revision 3 脚本重跑 IronMLX 针对性验证，条件与 `fixed256-fix-v1` 相同。`fixed256-fix-v1`（候选 84b82b06）的结果保留。
