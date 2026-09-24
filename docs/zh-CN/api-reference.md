# HTTP API 协议参考

[English](../api-reference.md) · [快速开始](api.md)

App 默认在 `http://127.0.0.1:9068` 提供服务。直接运行 CLI 时默认端口为 8080，
应以实际启动参数或 Dashboard 显示的 endpoint 为准。

快速定位：[Responses](#openai-responses-api) · [Chat Completions](#openai-chat-completions) · [Messages](#anthropic-messages) · [图片](#图片输入) · [日志接口](#日志级别管理仅本机)

## 后端单实例约束

同一 macOS 用户只能运行一个 `ironmlx serve` 后端，不同 App、CLI 参数或监听端口
也不能绕过该约束。后端会在初始化 MLX、加载 metallib 或模型之前，对
`~/.ironmlx/run/backend.lock` 获取非阻塞独占文件锁，并持有锁文件描述符直到进程
退出。正常退出、崩溃或 `SIGKILL` 都由系统自动释放锁；锁文件本身可以保留，不应作为
进程是否存活的判断依据。

第二个后端会立即退出，并在标准错误输出稳定错误码
`ironmlx_instance_already_running`。IronMLX App 会停止自动恢复循环，并提示用户先
退出已有实例。

## 健康与模型列表

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` 只表示 HTTP 进程可响应；`/healthz` 返回包含产品版本、模型、调度器、
缓存和内存状态的 JSON 快照。

App daemon 与 EnginePool 的 `GET /v1/models` 返回 OpenAI-compatible 模型列表；
`data[]` 至少包含 `id`、`object:"model"`、`created` 和 `owned_by`，并附带 IronMLX
模型加载策略与运行状态字段。列表来自当前可服务的注册模型，因此 OMP 等客户端可通过
OpenAI models-list discovery 自动发现已经注册但尚未加载或已经加载的模型。

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

`/healthz.memory.free_ram_bytes` 是操作系统报告的原始空闲页，仅用于观测；
`available_ram_bytes` 使用与进程内存 governor 相同的可回收内存口径。内存健康
状态由 `process_governor.pressure_level` 决定，而不是固定的 raw-free 阈值。
`degraded_reasons` 会列出队列、KV 缓存、内存压力、遥测或后端背压等具体原因。

## 错误契约

Chat Completions 与 Responses 的非流式错误使用 OpenAI 风格信封：

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_json"
  }
}
```

Anthropic Messages 的非流式错误使用 Anthropic 风格信封，并返回与响应体
`request_id` 相同的 `request-id` header：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "...",
    "code": "invalid_json"
  },
  "request_id": "req_..."
}
```

`error.code` 是 IronMLX 的稳定机器可读错误码；Messages 响应中的该字段属于
IronMLX 扩展。客户端应按 HTTP status 和 `error.type` 判断错误类别，使用
`error.code` 区分同一类别的具体原因。

| HTTP status | 稳定 `error.code` | 语义 | `Retry-After` |
|---:|---|---|---|
| 400 | `invalid_json` 及各字段/约束错误码 | JSON、字段、采样或输出约束不合法 | 无 |
| 413 | `request_body_too_large` | HTTP request body 超过 32 MiB | 无 |
| 413 | `request_token_capacity_exceeded` | 输入 token 与请求输出预算超过模型上下文容量；`error.details` 提供容量明细 | 无 |
| 503 | `scheduler_queue_full`、`scheduler_unavailable`、`scheduler_reply_lost` | 调度器暂时不可用 | `5` 秒 |
| 503 | `memory_budget_exceeded`、`memory_pressure`、`prefill_peak_unsafe`、`vision_prefill_peak_unsafe`、`cold_materialization_unsafe`、`prefix_store_backpressure` | 内存 governor 或存储背压暂时拒绝请求 | `5` 秒 |
| 503 | `engine_unavailable`、`diffusion_lane_overloaded` | 模型引擎或 DiffusionGemma lane 暂时不可用 | `5` 秒 |
| 500 | `generation_error` 及内部任务错误码 | 非预期服务端错误 | 无 |

所有可重试 503 都返回 JSON 和 `Retry-After: 5`。IronMLX 的 Messages 本地契约使用
HTTP 503 + `overloaded_error` 表达暂时过载；413 使用 `request_too_large`，并通过
上述两个稳定 code 区分传输体上限与模型上下文容量上限。

### 运行拓扑一致性

公开推理 API 的 transport 契约不随服务器启动方式变化。普通 causal 服务、
DFlash2 actor、Gemma4 drafter、DiffusionGemma、EnginePool 和 App daemon 共用同一请求提取、
模型无关字段校验、协议错误渲染和 SSE header 构造路径。

该一致性只约束 HTTP transport、协议错误和模型分派语义，不表示所有模型架构拥有
相同推理能力。DiffusionGemma 的 sampling、MTP、KV cache 和 PromptLookup 限制仍按
其 capability 描述明确拒绝。`/v1/models` 在 EnginePool、App daemon 及 App 的 DFlash2 discovery 路径公开；
`/admin/api/models/*` 只在 App daemon 拓扑公开。

### SSE 断连与取消契约

Chat Completions、Responses 和 Anthropic Messages 的流式请求在 SSE 响应开始后
支持客户端断连取消。HTTP response body 被丢弃时，transport 会立即发布协议无关的
断连信号；各协议的流式编码器停止消费生成事件，也不会在已观测到断连后继续构造
协议终止事件。

| 生成路径 | 取消生效点 | 释放内容 |
|---|---|---|
| Scheduler（包括 MTP/辅助 drafter） | 当前模型 forward 结束后的下一次安全调度边界 | 活跃请求、调度槽、KV cache 与内存预算 |
| DFlash2 actor | 当前 target/draft forward 结束后的下一次安全事件边界 | 请求级 target/draft cache、活动槽与内存预算 |
| 直接 `GenerationStream` | 当前 token forward 结束后的下一次 token 边界 | 生成状态与直接请求内存预留 |
| DiffusionGemma | 当前 block-diffusion 步骤结束后的下一次事件边界 | generation lane 与请求状态 |

取消不会强行中断正在执行的 Metal forward；这是为了避免在设备工作未完成时破坏模型
和 KV 状态。因此，从 TCP 断开到资源归还可能包含一个当前 forward/扩散步骤的尾延迟。
本契约只承诺已经开始返回 SSE 的流式请求；v0.1 不承诺非流式 HTTP 请求在客户端断开
后取消底层生成。

## DFlash2 Server

`ironmlx serve --dflash2-model-dir ...` 为固定的 Qwen3.8 target/draft 组合启动独立
DFlash2 actor。该路径支持 Chat Completions、Responses 和 Anthropic Messages 的
同步与 SSE 文本请求，也支持 Greedy、精确 sampling 及
`--max-sequences N`（`N >= 1`）请求级并发。

DFlash2 的启动参数、sampling、并发语义、隔离限制、`/healthz` 字段
见 [`dflash2-server-api.md`](dflash2-server-api.md)。App 启用 DFlash2 时会从普通 daemon
受控切换为独立 actor，并继续通过 `/v1/models` 公开唯一 target 的稳定模型 ID；动态
`/admin/api/models/*` 仍只属于普通 App daemon。

## OpenAI Responses API

`POST /v1/responses` 是推荐给本地 Agent 客户端的新接口。IronMLX 实现无状态
Responses 工作流：客户端发送完整 typed item 历史，服务执行本地模型推理，但不
持久化 response、conversation，也不执行任何工具。

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your-model-id",
    "instructions": "回答要简洁。",
    "input": "用一句话介绍 Metal。",
    "store": false,
    "max_output_tokens": 128,
    "stream": false
  }'
```

流式响应使用原生 Responses SSE 生命周期，包括 `response.created`、typed output
item、文本或函数参数 delta，以及终止事件 `response.completed`、
`response.incomplete` 或 `response.failed`；不会发送 Chat Completions 的
`[DONE]` 标记。

### Responses reasoning

具备精确原生 reasoning 模板契约的模型可通过 `reasoning.effort` 开启或关闭思考
通道。IronMLX 将模型原生 `<think>` 或 Gemma `thought` channel 解码为独立的
Responses `reasoning` item；流式响应使用 `response.reasoning_text.delta` 和
`response.reasoning_text.done`，不会把 reasoning 混入 `output_text`。

```json
{
  "model": "your-model-id",
  "input": "先分析，再简洁回答。",
  "reasoning": {"effort": "high", "summary": "none"},
  "store": false
}
```

Qwen3.8 原生模板支持三档 reasoning effort：Responses 的 `minimal`/`low` 映射为
`low`，`medium` 映射为 `medium`，`high`/`xhigh`/`max` 映射为 `xhigh`；`none`
表示关闭。未实现分档模板的其他模型只把非 `none` effort 作为原生 reasoning 开关。
具体推理长度仍由 checkpoint 决定。Responses 请求省略 `reasoning`、将其设为 `null`
或未提供 `reasoning.effort` 时，IronMLX 均按 `effort=none` 处理并在响应中回显该有效值；
客户端必须显式提供非 `none` effort 才会开启 reasoning。

对于使用精确受支持模板的 causal Qwen3.5/3.6/3.8 与 Gemma4/Unified，开启 reasoning
时会自动从总输出预算中预留 `min(floor(max_output_tokens / 4), 1024)` 个 token
给正文或工具调用，并另留原生通道标记和 UTF-8 边界所需空间，其余作为 reasoning
预算。达到该预算时若仍在思考，解码会约束后续 token，完成模型原生结束标记
（`</think>` 或 `<channel|>`），然后继续生成。结束标记会进入实际模型上下文，
并计入原来的总输出上限；模型提前自然结束思考以及 Gemma 直接回答的路径均保留。
Chat Completions 和 Messages 也根据各自的 `max_tokens` 总预算应用相同策略，
客户端无需修改。

这项策略与 JSON/工具调用约束、请求独立的推测解码分叉与回滚共同生效。关闭
reasoning、模板无法识别、其他 reasoning 方言、DiffusionGemma，以及不足以容纳
通道标记和正数 reasoning/正文预算的小额请求，保留原行为。首版采用自动策略，
不新增 App 配置或请求字段，不提高客户端明确指定的总预算，也不自动重试请求。
预留空间不保证答案完整或正确；达到总上限后，仍按原有规则报告 incomplete
及条目级截断状态。完成 reasoning 结束标记仅表示通道关闭，不代表推理质量保证。
强制切换后，模型仍可能在正文通道继续输出分析式文字，尤其在总预算很小时；
不能据此认定任务已完成。该策略会禁止在正文中重复原生通道标记或重新打开
reasoning 通道。

无状态历史回灌接受 `reasoning` item 中的明文 `reasoning_text`，并将它传给下一轮
原生模板。IronMLX 不生成 OpenAI 托管的 `encrypted_content`；只有 encrypted
content、没有明文 reasoning 的历史无法在本地重放，会返回 400。

为兼容历史重放，没有后续 assistant 正文或 function call 的孤立明文 reasoning 会被跳过，
其余历史及正常配对的 reasoning 保留。旧历史没有 `status` 时也适用；这不代表能够
准确判断旧记录是否因输出上限截断。无法解读的加密历史即使孤立也仍被拒绝，空的
`reasoning_text` 不能用来绕过这项校验。

新增 reasoning 条目以 `status:"in_progress"` 开始，并在 `response.output_item.done`、
最终响应及非流式输出中标记为 `completed` 或 `incomplete`。只有原生 reasoning 通道
仍未关闭时达到输出上限，才将该条目标记为 `incomplete`。已消费结束标记，或已转入正文/
工具调用的 reasoning 不会因为后续内容截断而被误标。客户端应保留
`response.output_item.done` 中的完整条目供重放。这项兼容处理解决历史校验错误，
不会恢复被截断的答案，也不保证答案完整。

`reasoning.summary:"auto"` 仍可作为上游客户端的自动能力请求，但当前模型没有
独立 summary 生成通道，因此不会把完整 reasoning 截断或改写成 summary。原生
reasoning 支持以 [supported-models.md](supported-models.md) 的矩阵和精确模板检测
为准。

### Responses function tools

Responses 使用顶层 function tool 形状：

```json
{
  "model": "your-model-id",
  "input": "东京天气如何？",
  "tools": [{
    "type": "function",
    "name": "get_weather",
    "description": "查询城市天气",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    },
    "strict": true
  }],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "store": false
}
```

模型产生 `function_call` item 后，客户端负责执行函数，并在下一次请求的完整
`input` 历史中追加原调用和同一 `call_id` 的 `function_call_output`。支持文本、
严格图片 `data:` URL message item、function call/output、同步和 SSE，以及现有全部
模型工具 dialect 和约束选项。

#### Hermes Agent

Hermes Agent 应使用无状态 Responses transport。配置方法和验证命令见
[Hermes Agent 集成指南](hermes-agent.md)。Hermes 执行客户端工具，IronMLX 只负责
推理和生成结构化调用。

#### oh-my-pi

oh-my-pi 应使用 `openai-responses` provider。配置方法和验证命令见
[oh-my-pi 集成指南](oh-my-pi.md)。OMP 执行客户端工具，IronMLX 只负责推理和
生成结构化调用。

#### DeepSeek Harness

dsh 应通过 `llm-pi-ai` 使用 `openai-responses` 路由。配置方法和验证命令见
[DeepSeek Harness 集成指南](dsh.md)。dsh 执行客户端工具，IronMLX 只负责推理和
生成结构化调用。

### Responses structured outputs

`text.format` 支持 JSON mode 和受 Schema 约束的 Structured Outputs。JSON mode 使用：

```json
{"text":{"format":{"type":"json_object"}}}
```

Schema 模式使用：

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "weather_answer",
      "description": "结构化天气回答",
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "days": {"type": "integer"}
        },
        "required": ["city", "days"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

输出仍是 Responses 的 `message` / `output_text` item；客户端将其中的文本解析为
JSON。IronMLX 在 token 采样前应用 grammar mask，并在生成结束后再次验证完整 JSON。
支持的 Schema 子集为 `object`、`array`、`string`、`number`、`integer`、`boolean`、
`null`、nullable type 数组、`properties`、`required`、`items`、`enum`、`const`、
`anyOf`、`minItems`、`maxItems`、`minLength`、`maxLength`、`minimum`、`maximum`、
`exclusiveMinimum` 和 `exclusiveMaximum`。非 strict 工具的嵌套 object 还支持
`additionalProperties:true` 或以受支持 Schema 约束动态属性值；顶层工具参数对象仍
必须封闭。Schema 最大深度为 8。不支持的关键字会在生成前返回 400，不会静默弱化。
`strict:true` 要求每层 object 都设置 `additionalProperties:false`，并把所有
properties 列入 `required`。

当请求同时包含 function tools 和 `text.format` 时：

- `tool_choice:"none"`：只允许结构化 JSON 最终回答。
- `tool_choice:"required"` 或指定函数：只允许工具调用。
- `tool_choice:"auto"`：允许原生工具调用，或符合 Schema 的 JSON 最终回答。

该联合约束适用于当前全部工具 dialect、普通生成、Scheduler、推测解码和
DiffusionGemma canvas 解码路径。

本地无状态边界：

- `store` 只能省略或设为 `false`；不支持 `store:true`。
- 不支持 `previous_response_id`、conversation、background response 或 response
  retrieve/delete/cancel API。
- 工具支持客户端执行的顶层 `type:"function"`，以及 Codex 使用的客户端
  `type:"namespace"` 函数组。namespace 会编译为有界 dispatcher，并在
  `function_call`/历史回灌时恢复公开的 `namespace`、函数名和参数；IronMLX
  仍不执行工具。
- 不支持托管 Web/File Search、托管 MCP、Code Interpreter 或 custom/freeform
  工具。namespace 子函数的指定 `tool_choice` 暂不支持；应使用 `auto` 或
  `required`。超出约束 Schema 子集的动态参数只允许用于 `strict:false` 的
  namespace 子函数，并使用 JSON 参数信封；`strict:true` 不会降级。
- 支持无状态明文 reasoning typed item 及历史回灌；不持久化 reasoning，也不生成
  `encrypted_content`。
- 不支持 reasoning summary、refusal typed item、OpenAI file ID、音频输入/输出、
  图片输出或图片形式的 function output；这些能力不会以普通 `output_text` 伪装。
- sampling 公开字段仅为 `temperature`（有限数且位于 `[0, 2]`）和 `top_p`
  （有限数且位于 `(0, 1]`）。`top_k`、`repetition_penalty` 不是 Responses
  标准字段，发送后会返回 400；其他未知字段同样不会被静默忽略。

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:9068/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your-model-id",
    "messages": [{"role": "user", "content": "用一句话介绍 Metal。"}],
    "max_tokens": 128,
    "temperature": 0.2,
    "stream": false
  }'
```

流式响应将 `stream` 设为 `true`；如需最终 usage chunk，可同时传入
`"stream_options":{"include_usage":true}`。

Chat Completions 对顶层请求、message、content part、`image_url` payload 和
`stream_options` 使用严格字段契约；未在本节公开的字段会返回 400，不会被静默
忽略。sampling 公开字段仅为 `temperature`（有限数且位于 `[0, 2]`）和 `top_p`
（有限数且位于 `(0, 1]`）。`top_k` 与 `repetition_penalty` 不属于公开的 Chat
Completions 字段。

Qwen3.8 原生模板额外支持顶层 `reasoning_effort`，有效值为 `low`、`medium`、
`xhigh`（默认）。`chat_template_kwargs.enable_thinking=false` 可关闭思考，
`chat_template_kwargs.preserve_thinking=false` 可不保留旧 assistant 消息中的
`reasoning_content`。需要独立 reasoning 输出 item/block 时，应使用 Responses 或
Anthropic Messages。

### Structured Outputs

Chat Completions 通过标准 `response_format` 支持 JSON mode 和受 Schema 约束的
Structured Outputs：

```json
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "返回东京的天气。"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather_answer",
      "description": "结构化天气回答",
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "days": {"type": "integer"}
        },
        "required": ["city", "days"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

JSON mode 使用 `{"response_format":{"type":"json_object"}}`。Chat 的
`json_schema` 定义位于 `response_format.json_schema`；不要使用 Responses API
扁平的 `text.format` 形状。支持的 Schema 子集与上文 Responses Structured
Outputs 相同，不支持的 schema 或字段形状会在生成前返回 400。

`response_format` 可与 function tools 同时使用：`tool_choice:"auto"` 允许工具调用
或符合 Schema 的 JSON 最终回答；`none` 只允许 JSON 最终回答；`required` 或指定
函数时只允许工具调用。三种约束共用同一编译路径，适用于同步、SSE、Scheduler、
MTP/辅助 drafter 和 DiffusionGemma。若因 token 上限以 `finish_reason:"length"`
结束，JSON 可能不完整，客户端应按截断结果处理。

### Function tools

具有受支持原生工具模板的 Qwen 3.5/3.6/3.8、Gemma 4、DiffusionGemma、GLM、
Llama 和 MiniCPM 模型，可通过 Chat Completions 的 `tools` 字段请求客户端
函数调用。具体模型与模板要求见[支持模型矩阵](supported-models.md)：

```json
{
  "model": "your-qwen-model-id",
  "messages": [{"role": "user", "content": "东京天气如何？"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "stream": false
}
```

服务只生成结构化 `tool_calls`，不会执行函数。客户端执行后，应把原 assistant
消息及每个结果按 `role: "tool"`、`tool_call_id` 追加到 `messages`，再发起下一次
请求。同步响应中的 `function.arguments` 是 JSON 字符串；SSE 使用稳定的
`tool_calls[].index`/`id`，参数可跨多个 delta，结束原因是 `tool_calls`。

Chat tools 当前边界：

- `tool_choice` 支持 `auto`、`none`、`required`，以及通过
  `{"type":"function","function":{"name":"..."}}` 指定函数。
- `parallel_tool_calls` 默认为 `true`；设为 `false` 时约束当前 assistant turn
  最多生成一个调用。Llama 3.1/3.2 原生协议始终只支持单调用。
- 只支持 `type: "function"`。`strict: true` 支持约束解码所覆盖的 JSON Schema
  子集；对象 schema 必须递归设置 `additionalProperties: false`，且所有属性都
  必须列入 `required`。不支持的 schema 关键字会在生成前返回 400。
- 不支持旧 `functions` / `function_call` 字段；Responses 客户端应使用上文独立的
  `/v1/responses` typed-item 协议。
- 当前支持经过精确模板契约检测的 Qwen 3.5/3.6/3.8、Gemma 4/Gemma 4 Unified、
  DiffusionGemma、GLM-4 MoE Lite、Llama 3.1/3.2、MiniCPM-V 4.6 和 MiniCPM5
  原生工具 dialect；其他模板收到 `tools` 时会在生成前返回 400。
- 工具结果必须引用此前尚未完成的 assistant tool call；孤立、重复或缺失 ID
  会在生成前返回 400。

## Anthropic Messages

```bash
curl http://127.0.0.1:9068/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your-model-id",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "stream": false
  }'
```

Messages 对请求和嵌套 content block 使用严格字段契约。sampling 公开字段为
`temperature`（有限数且位于 `[0, 1]`）、`top_p`（有限数且位于 `(0, 1]`）和
正整数 `top_k`。`repetition_penalty` 不是 Anthropic Messages 字段，发送后会返回
400；其他未知字段同样不会被静默忽略。

### Anthropic Structured Outputs

Messages 通过 Anthropic 当前正式协议 `output_config.format` 支持受 JSON Schema
约束的最终文本输出：

```json
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "提取姓名和备注。"}],
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "notes": {"type": "string"}
        },
        "required": ["name"],
        "additionalProperties": false
      }
    }
  },
  "max_tokens": 128
}
```

符合 Schema 的 JSON 位于普通 `text` content block 中；同步与 SSE 使用相同的
token 级约束。`output_config.format` 可与客户端 tools 同时使用：`auto` 允许工具
调用或结构化最终回答，`none` 只允许结构化最终回答，`any` 和指定 `tool` 只允许
工具调用。Schema 子集与本文件前述 Structured Outputs 约束一致，不支持的类型、
关键字或字段形状会在生成前返回 400。

若达到 token 上限，响应以 `stop_reason: "max_tokens"` 结束，此时 JSON 可能不完整。
Structured Outputs 不允许以最后一条 assistant message 进行 prefill。仅支持正式的
`output_config.format`；已废弃的顶层 `output_format` 不在兼容范围内。
`output_config.format` 可以与已启用的 `thinking` 同时使用。IronMLX 对原生输出
section 进行组合约束：thinking section 保持自由生成，最终 text section 才启用
JSON Schema grammar。该语义同样覆盖同步、SSE、Scheduler、MTP/辅助 drafter 和
DiffusionGemma。客户端 tools 可与两者组合；工具调用使用自己的参数 grammar，
最终直接回答使用 `output_config.format` grammar。若本轮先返回 `tool_use`，客户端
回灌 `tool_result` 后的下一轮会重新执行相同的 thinking/最终 JSON section 约束。

### Anthropic extended/adaptive thinking

具备精确原生 reasoning 模板契约的模型支持 Anthropic `thinking`：

```json
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "请仔细分析后回答。"}],
  "thinking": {"type": "adaptive", "display": "summarized"},
  "output_config": {"effort": "high"},
  "max_tokens": 4096
}
```

支持 `disabled`、`enabled` 和 `adaptive`。手动模式要求 `budget_tokens >= 1024` 且
小于 `max_tokens`；adaptive 模式可通过 `output_config.effort` 接收 `low`、
`medium`、`high`、`xhigh` 或 `max`。Qwen3.8 将其映射到原生 `low`、`medium`、
`xhigh` 三档；其他当前本地模板只使用 `enable_thinking` 布尔开关。两者都没有
Claude 服务端的分级预算控制器，因此 `budget_tokens` 与 `effort` 会被严格校验并
控制原生模板，但不会被描述为已经校准的 token 预算或质量档位。总生成硬上限仍为
`max_tokens`，具体 thinking 长度由 checkpoint 决定。

同步响应将原生 reasoning 放入位于 `text`/`tool_use` 之前的 `thinking` content
block；流式响应依次发出 `thinking_delta`、`signature_delta` 和
`content_block_stop`。`usage.output_tokens_details.thinking_tokens` 提供本地原生
reasoning token 计数。历史回灌接受一个位于 assistant 可见内容之前的
`thinking` block，并校验 IronMLX 生成的本地完整性 signature；修改后的 block
返回 400。

当前不支持 `display: "omitted"`、`redacted_thinking`、多个或交错 thinking block，
因为本地模型没有 Claude 的加密隐藏思考通道，也没有可保持 block 顺序的
interleaved-thinking 模板契约。这些形状会明确返回 400。来自 Claude 服务的签名
不能作为 IronMLX 本地历史直接回灌。

### Anthropic client tools

`/v1/messages` 支持 Anthropic 原生客户端工具协议，并与 Chat Completions、Responses
复用同一套模型工具模板、历史关联校验和 token 级约束解码：

```json
{
  "model": "your-model-id",
  "system": "回答要简洁。",
  "messages": [{"role": "user", "content": "东京天气如何？"}],
  "tools": [{
    "name": "get_weather",
    "description": "查询城市天气",
    "input_schema": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    },
    "strict": true
  }],
  "tool_choice": {"type": "auto"},
  "max_tokens": 128,
  "stream": false
}
```

模型选择调用工具时，同步响应使用原生 `tool_use` content block，并以
`stop_reason: "tool_use"` 结束：

```json
{
  "type": "message",
  "role": "assistant",
  "content": [{
    "type": "tool_use",
    "id": "call_...",
    "name": "get_weather",
    "input": {"city": "东京"}
  }],
  "stop_reason": "tool_use"
}
```

IronMLX 只生成调用信息，不执行函数、Shell、MCP、HTTP API 或其他外部工具。
客户端执行工具后，必须在下一次请求中原样回灌 assistant `tool_use`，并在紧随的
user message 中用同一 ID 提交 `tool_result`：

```json
{
  "messages": [
    {"role": "user", "content": "东京天气如何？"},
    {"role": "assistant", "content": [{
      "type": "tool_use",
      "id": "toolu_123",
      "name": "get_weather",
      "input": {"city": "东京"}
    }]},
    {"role": "user", "content": [{
      "type": "tool_result",
      "tool_use_id": "toolu_123",
      "content": "晴，26°C",
      "is_error": false
    }]}
  ],
  "tools": [{
    "name": "get_weather",
    "input_schema": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    }
  }]
}
```

工具协议边界：

- `tool_choice` 支持 `auto`、`any`、指定 `tool` 和 `none`；前三种支持
  `disable_parallel_tool_use`。`any` 要求至少一个调用，指定 `tool` 要求调用该工具，
  禁用并行后一个 assistant turn 最多一个调用。
- 一个 assistant message 可同时包含文本和一个或多个 `tool_use`；一个 user message
  可回传多个 `tool_result`，随后继续附带文本或图片。调用 ID 必须非空、唯一且完整
  配对；孤立、重复、遗漏或顺序错误会在生成前返回 400。
- SSE 使用原生 `message_start` / `content_block_*` / `message_delta` /
  `message_stop` 生命周期。工具块以 `tool_use` 开始，参数通过一个或多个
  `input_json_delta.partial_json` 增量发送，客户端应拼接后再解析 JSON。
- `input_schema` 与 `strict` 使用上文 Structured Outputs 所述的受支持 Schema 子集；
  不支持的类型、关键字或模型模板会在生成前明确返回 400，不会静默降级为文本。
- 支持范围是客户端定义的函数工具。Anthropic 托管工具、服务器工具、MCP、
  computer use、Web Search、Code Execution 等不属于本地推理服务能力。

## 兼容性与验证

逐字段支持范围及 SDK 验证方法见 [API 兼容矩阵](api-compatibility-matrix.md)。协议验证不等于具体模型的生成质量验收。

## 图片输入

OpenAI 兼容接口只接受 JPEG、PNG 或 WebP 的严格 `data:` URL；不会抓取远程
HTTP/HTTPS URL。示例：

```json
{
  "model": "your-vlm-id",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "max_tokens": 128
}
```

请求体与图片数量、大小、尺寸均有资源上限，详见[安全边界](security-boundary.md)。

## LAN 模式

LAN 模式使用 `https://<selected-ip>:<port>`，所有路由（包括 health）都必须带：

```text
Authorization: Bearer <API-Key>
```

客户端必须信任 App 导出的本地 CA，不能关闭 TLS 证书校验。

## 日志级别管理（仅本机）

`GET /admin/api/log-level` 返回 `level`、`revision`、`process_id`。独立 CLI 的自定义 `RUST_LOG` 过滤器以空 `level` 表示，直到被规范级别替换；App 要求初始级别为规范值。

`POST /admin/api/log-level` 接收：

```json
{"level":"DEBUG","expected_revision":0,"expected_process_id":12345}
```

成功后返回新状态。版本号或进程 ID 过期返回 409；无效 JSON 和不支持的级别被拒绝；运行时控制不可用返回 503。接口仅挂在回环监听端，LAN 监听端不提供该路由，即使携带有效 LAN Key 也不能访问。

### App 日志设置应用语义

App 先读取后端级别，携带进程 ID 和 revision 更新过滤器，再仅保存 `log_level` 并调整自身过滤器。
后端停止时不发送请求，保存值在下次启动时应用；启动、恢复、停止或其他设置应用过程中拒绝变更。
响应丢失或保存失败时查询实际状态，在前提仍匹配时恢复旧级别；进程或 revision 改变、回退未确认时不会显示为成功。
旧配置 TRACE/WARN 兼容为 ALL/WARNING；第三方依赖日志最多为 WARNING，选择 ERROR 时为 ERROR。
App 启动 helper 时传入 `IRONMLX_LOG_LEVEL`，独立 CLI 仍可使用 `RUST_LOG`。

### 原生工具模板参考

MiniCPM-V 4.6 与 MiniCPM5 使用不同 XML 工具协议；MiniCPM5 对包含 `<`、`&` 或换行的字符串参数使用 CDATA。
Gemma 内部将动态 object 投影为确定性键值条目，响应时恢复原对象；公开 Schema 和参数形状保持不变。
