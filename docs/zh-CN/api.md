# HTTP API

App 默认在 `http://127.0.0.1:9068` 提供服务。直接运行 CLI 时默认端口为 8080，
应以实际启动参数或 Dashboard 显示的 endpoint 为准。

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

| 行为 | 普通服务 | DFlash2 actor | Gemma4 drafter | DiffusionGemma | EnginePool | App daemon |
|---|---|---|---|---|---|---|
| Chat/Responses/Messages 严格 JSON 与模型无关字段校验 | 相同 | 相同 | 相同 | 相同 | 相同 | 相同 |
| OpenAI/Anthropic 错误 envelope、413/503 与 `Retry-After` | 相同 | 相同 | 相同 | 相同 | 相同 | 相同 |
| SSE transport headers | `text/event-stream` + `no-cache` | 相同 | 相同 | 相同 | 相同 | 相同 |
| typed request 进入模型实现 | 直接 | 独立 actor | 直接 | 直接进入 block-diffusion lane | 解析模型后直接分派 | 解析模型后直接分派 |
| 模型选择 | 启动时固定 | target + draft 启动时固定 | 启动时固定 | 启动时固定 | request `model` 或唯一/default model | request `model` 或唯一/default model |

该一致性只约束 HTTP transport、协议错误和模型分派语义，不表示所有模型架构拥有
相同推理能力。DiffusionGemma 的 sampling、MTP、KV cache 和 PromptLookup 限制仍按
其 capability 描述明确拒绝。`/v1/models` 只在 EnginePool 和 App daemon 拓扑公开；
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

DFlash2 的启动参数、sampling、并发语义、隔离限制、`/healthz` 字段与最终验收结果
见 [`dflash2-server-api.md`](../dflash2-server-api.md)。App 启用 DFlash2 时会从普通 daemon
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

无状态历史回灌接受 `reasoning` item 中的明文 `reasoning_text`，并将它传给下一轮
原生模板。IronMLX 不生成 OpenAI 托管的 `encrypted_content`；只有 encrypted
content、没有明文 reasoning 的历史无法在本地重放，会返回 400。

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

## API contract 与官方 SDK 门禁

完整的 v0.1 逐字段、错误、拓扑和 SDK 版本兼容矩阵见
[API 兼容矩阵](../api-compatibility-matrix.md)。本文继续作为各协议的使用说明和示例；
矩阵是发布承诺边界，若两者出现冲突，以矩阵中明确的“支持/受限/拒绝”定义为准。

CI 固定执行服务端 contract 测试和官方 Python SDK 黑盒测试。SDK 测试通过真实的
loopback HTTP/SSE 连接发送请求，不调用 SDK 内部模型构造器，也不以 `curl` 形状的
JSON 代替 SDK 解析。

| 协议 | 固定客户端 | 自动验收范围 |
|---|---|---|
| Chat Completions | OpenAI Python SDK `2.48.0` | 同步、SSE、function tools、Structured Outputs 请求、usage、400 错误 |
| Responses | OpenAI Python SDK `2.48.0` | 同步 typed output/reasoning、SSE typed events、function tools、Structured Outputs 请求、413/503 与 `Retry-After` |
| Anthropic Messages | Anthropic Python SDK `0.121.0` | 同步、原生 SSE、tool use、Structured Outputs + adaptive thinking 请求、400/413/503、`request-id` 与 `Retry-After` |

三套由固定 SDK 实际生成的复杂请求保存在
`ironmlx/tests/fixtures/api_contract_sdk/`。Rust 测试会把相同字节送入生产
`ApiJson` extractor 和共享的预调度校验，因此 SDK fixture、公开 DTO 与模型无关
字段契约不能独立漂移。官方 SDK 测试服务器返回按当前公开契约固定的 JSON/SSE
fixture，用于验证客户端 typed object、stream event 和异常类型解析；生产 serializer
则由同一 CI 门禁中的 Rust 服务端测试独立验证。

本地执行：

```bash
python3 -m venv /tmp/ironmlx-api-contract-sdk
/tmp/ironmlx-api-contract-sdk/bin/python -m pip install \
  --requirement scripts/api-contract-sdk/requirements.txt
/tmp/ironmlx-api-contract-sdk/bin/python \
  scripts/api-contract-sdk/contract.py --fixture

cargo test --locked --all-features -p ironmlx --lib core::server::
```

该门禁不加载 checkpoint，因此证明的是协议、transport 和官方客户端解析兼容性，
不证明特定模型的生成质量、工具选择准确率或 Structured Outputs 成功率。真实模型
HTTP/SDK smoke 仍作为发布候选的独立验收层；不能用 fixture 结果替代。

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
