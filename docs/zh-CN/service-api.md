# 服务与管理 API

[English](../service-api.md) · [API 参考](api-reference.md)

本文说明 App、EnginePool 与常规 CLI 服务的公共访问约定、健康检查、模型发现和已列出的管理接口。各类推理请求与响应由对应专题定义，可从 [API 参考](api-reference.md)进入。

## 服务地址、认证与约定

App 默认地址为 `http://127.0.0.1:9068`，独立 CLI 默认端口为 8080，以实际配置为准。本机回环访问无需 API Key；要求非空 Key 的 SDK 可以使用 `local`。

LAN 使用 `https://<selected-ip>:<port>`，所有路由（包括健康检查）都必须带 `Authorization: Bearer <API-Key>`，并信任 App 导出的 CA。JSON 请求使用 `Content-Type: application/json`；图片编辑使用 multipart 数据。配置与资源限制见[安全边界](security-boundary.md)。

独立的 `ironmlx serve-systemone` 使用单独的端口与认证规则：`/v1/*` 在本机也要求 API Key，`/healthz` 无需凭据。见 [System One API](laya-systemone-api.md#可选的独立-cli)。LAN 凭据缺失或无效返回 401；错误格式由对应端点定义。

## 服务与管理端点

| 端点 | 用途 |
| --- | --- |
| [`GET /health`](#健康与模型列表) | HTTP 服务响应 |
| [`GET /healthz`](#健康与模型列表) | 运行状态 |
| [`GET /v1/models`](#健康与模型列表) | 模型发现 |
| [`POST /v1/models/{model_id}/load`](engine-pool.md) | 加载模型 |
| [`POST /v1/models/{model_id}/unload`](engine-pool.md) | 卸载模型 |
| [`GET /admin/api/models/loaded`](#app-决策运行时指标) | 已加载模型与指标 |
| [`GET /admin/api/log-level`](#日志级别管理仅本机) | 查询日志级别 |
| [`POST /admin/api/log-level`](#日志级别管理仅本机) | 修改日志级别 |

## 健康与模型列表

### GET /health

无请求体。成功返回 HTTP 200 和纯文本 `ok`，仅表示 HTTP 服务可响应。

### GET /healthz

无请求体。成功返回 HTTP 200 JSON。公共字段为 `status`、`degraded_reasons`、`version` 与 `mode`；模型、调度、内存和模型池详情随服务模式变化。客户端应允许可选观测字段缺失，具体请求能否执行仍以推理接口返回为准。

`/healthz.memory.free_ram_bytes` 是操作系统报告的原始空闲页，仅用于观测；
`available_ram_bytes` 使用与进程内存 governor 相同的可回收内存口径。内存健康
状态由 `process_governor.pressure_level` 决定，而不是固定的 raw-free 阈值。
`degraded_reasons` 会列出队列、KV 缓存、内存压力、遥测或后端背压等具体原因。

### GET /v1/models

无请求体。成功返回 HTTP 200 JSON，包含 `object:"list"` 和模型对象数组 `data`。App 与 EnginePool 还提供用于决策模型发现的 TypeSafe `models` 数组，见 [System One](laya-systemone-api.md)。已注册模型可以在加载前列出。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `data[].id` | string | 请求使用的公开模型 ID。 |
| `data[].object` | string | model |
| `data[].created` | integer | Unix 时间戳；模型池条目当前为 0。 |
| `data[].owned_by` | string | ironmlx |
| `data[].load_policy` | string, optional | 模型池 preload/lazy/disabled 加载策略。 |
| `data[].state` | string, optional | 模型池状态，如 unloaded/loading/loaded/draining/failed/missing/disabled。 |
| `data[].context_window` | integer, optional | 有效总上下文容量。 |
| `data[].max_output_tokens` | integer, optional | 扣除输入 token 前的输出上限。 |

能够可靠确定容量时，causal 模型条目还会返回以下 IronMLX 扩展字段：

- `context_window`：有效总 token 容量，取模型上下文容量与部署配置 `max_cache_cap`（App 的 **MAX CONTEXT TOKENS**）中的较小值。
- `max_output_tokens`：输出 token 上限，包含 reasoning、答案正文和工具调用。当前 causal 推理没有独立的输出硬上限，因此其值等于 `context_window`。具体请求的输出预算仍须满足 `context_window - input_tokens`，其中输入包含聊天模板和多模态输入 token。这**不是** App 的 **MAX OUTPUT TOKENS** 设置值：该设置仅在 Responses 请求未指定输出预算时提供默认值。

例如，模型上下文容量为 262144，部署配置 `max_cache_cap=65536`，则返回
`context_window:65536`、`max_output_tokens:65536`，即使默认输出预算为 256。
客户端必须扣除输入占用后再选择输出预算；声明的上限不保证非空输入能获得这么多输出，
也不保证答案完整。请求仍受准入与内存检查约束。

已加载模型使用实际准入容量；未加载的 causal 模型使用 checkpoint 中明确的容量元数据
与已注册的调度器配置，无需加载权重。容量缺失、无效或不受支持时省略字段，不返回零，
也不从生成默认值推断。Audio 与 DiffusionGemma 暂不返回这两个字段。
App DFlash2 discovery 返回已加载 target 的有效容量，而非 draft 模型容量。
这两个字段是可选扩展，客户端应保留字段缺失时的回退逻辑。

## 管理接口

### App 决策运行时指标

`GET /admin/api/models/loaded` · 仅 App。HTTP 200 返回已加载模型对象数组。下表描述 runtime_kind:decision 条目的 decision_metrics 子对象，其他运行时类型不返回该子对象。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `window_seconds` | 整数 | 近期性能窗口秒数，当前为 `60`。 |
| `completed_requests` | 整数 | 当前模型实例加载以来成功完成的请求数。 |
| `failed_requests` | 整数 | 当前模型实例加载以来失败的请求数。 |
| `recent_completed_requests` | 整数 | 当前近期窗口中参与统计的成功请求数。 |
| `latency_ms_p50` | 数值或 `null` | 近期窗口内端到端延迟的中位数，单位为毫秒。 |
| `input_tokens_per_second` | 数值或 `null` | 近期窗口内各请求输入 Token 速率的中位数。 |
| `questions_per_second` | 数值或 `null` | 近期窗口内各请求问题处理速率的中位数。 |
| `last_request_unix_ms` | 整数或 `null` | 最近一次成功或失败请求的完成时间，使用 Unix epoch 毫秒。 |

近期窗口内没有成功样本时，相应性能字段为 `null`。累计值只覆盖当前模型加载周期；
卸载后重新加载模型或重启后端会重新计数。

### App 向量运行时指标

`runtime_kind:embedding` 的已加载模型通过 `GET /admin/api/models/loaded` 和
App 的 `/healthz` 模型条目返回 `embedding_metrics`，包含向量请求计数、近期延迟、
输入 token 吞吐与向量吞吐。字段含义、统计窗口和重置行为见
[Embedding API](text-embeddings.md#运行状态)。

### 日志级别管理（仅本机）

`GET /admin/api/log-level` 成功返回 HTTP 200 JSON，含 level、revision 与 process_id；CLI 使用自定义 RUST_LOG 过滤器时 level 为 null。示例：

```json
{
  "level": "INFO",
  "revision": 0,
  "process_id": 12345
}
```

`POST /admin/api/log-level` 接受以下 JSON 字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `level` | string | 是 | — | ALL/DEBUG/INFO/WARNING/ERROR；TRACE/WARN 为旧别名。 |
| `expected_revision` | integer | 是 | — | 最近一次 GET 返回的无符号 revision。 |
| `expected_process_id` | integer | 是 | — | 最近一次 GET 返回的 process_id。 |

```json
{
  "level": "DEBUG",
  "expected_revision": 0,
  "expected_process_id": 12345
}
```

使用 GET 返回的实际值。成功返回 HTTP 200 和更新后的快照，revision 增加 1。进程/revision 过期返回 409 与纯文本 log_control_changed；控制器不可用返回 503 与纯文本 log_control_unavailable。JSON 或字段类型非法时由 JSON 提取器返回 4xx，不采用推理接口错误格式。

两个路由仅挂载在回环监听端，即使携带有效 LAN Key 也不可访问。App 设置应用与回退语义见[从源码构建](building-from-source.md#app-日志设置应用语义)。

## 相关参考

- [开发者指南](developer-guide.md#api-接入)：按能力选择 API，完成首次请求。
- [文本与视觉 API](text-vision-api.md)：Responses、Chat Completions 和 Messages。
- [Embedding API](text-embeddings.md)：文本、图片、音频与组合向量。
- [语音合成 API](audio-speech-api.md)：音频输出与声音管理。
- [图片生成 API](image-generation-api.md)：文生图与单图条件编辑。
- [System One API](laya-systemone-api.md)：选择、评分与真假概率。
