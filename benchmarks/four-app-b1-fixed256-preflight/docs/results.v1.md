# 四应用 B1 固定输出 256 token —— 可行性预检结果

2026-10-08。运行 `reports/four-app-b1-fixed256-preflight/results/fixed256-v1`：完整，无失败。每个应用 2 个预热、6 个目标请求、1 个探针，共 24 个目标请求。所有请求都没有 HTTP 或流错误，全部收到 `[DONE]`，计分请求全部零缓存命中。没有使用 `ignore_eos`，因为 oMLX 0.7.0 不支持。

## 逐请求结果（max_tokens 256）

表中格式为“原生输出 token / 统一 tokenizer 重计数 / finish_reason”。

| 任务 | IronMLX | oMLX | Splash | TensorFold |
| --- | --- | --- | --- | --- |
| code-1 | **244 / 244 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| code-2 | **251 / 251 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| code-3 | **246 / 246 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| knowledge-1 | **242 / 242 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| knowledge-2 | **243 / 243 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| knowledge-3 | **243 / 243 / 无** | 256 / 256 / length | 256 / 256 / length | 256 / 256 / length |
| 探针“只回答 OK” | 2 / 1 / stop | 1 / 1 / stop | 2 / 1 / stop | 2 / 1 / stop |

**IronMLX：**
- 6 个任务都在 242–251 个 token 处提前结束，没有达到 256。
- 流中没有任何 `finish_reason`，但有 `[DONE]`；usage 记录的是实际写出的 token 数。
- usage 不带 `prompt_tokens_details`，所以 `cached_tokens` 无法从 usage 读取，零命中改由 `/healthz` 核实：前缀缓存关闭，命中 0。
- 原因没有定位。DFlash2 在剩余预算不足一个推测窗口时有专门处理，可能与此有关，需要另行诊断。本轮不改代码。

**计数差异：**
- 写到上限的请求，原生计数和统一计数都是 256，没有差异。
- 探针和预热中，IronMLX、Splash、TensorFold 的原生计数比统一计数多 1，看起来是把结束 token 也计入了。oMLX 没有这个差异。

**探针：** 四个应用都在 1–2 个 token 后自然结束（stop）。这说明没有 `ignore_eos` 时，`max_tokens` 不能把任意输入强制写到 256 个 token。

**结论范围：** 这六个任务在 oMLX、Splash、TensorFold 上都能写到 256 个 token，但这不等于任意输入都能。IronMLX 在这六个任务上就写不到 256。

## 实际生效的配置

| 项目 | IronMLX | oMLX | Splash | TensorFold |
| --- | --- | --- | --- | --- |
| KV 格式 | `--kv-quant` 默认 none，按 bf16 存储（来自参数说明和源码） | 没有开启 TurboQuant KV，按 bf16 存储（来自源码和 runtime 文件） | 服务和引擎进程都带 `--kv-format bf16`（来自运行时进程命令行） | Mac lanes 路径上没有 KV 量化（来自源码） |
| 固定长度机制 | 支持 `ignore_eos`（源码） | 不支持 | 支持 `ignore_eos`（源码） | 支持 `ignore_eos`（源码） |
| 推测解码证据 | 每个请求的 drafted 和 accepted 都增加 | 每个请求都有 `MTP[n]` 日志行 | drafted 和 accepted 都增加 | 每个请求的 speculative rounds、drafted、accepted 都大于 0 |
| 缓存命中 | `/healthz` 前缀命中 0 | 0 | 0 | 0 |

只有 Splash 的 KV 格式在运行时直接确认过，其余三个应用依据源码和配置推断。
