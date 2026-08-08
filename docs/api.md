# HTTP API

App 默认在 `http://127.0.0.1:9068` 提供服务。直接运行 CLI 时默认端口为 8080，
应以实际启动参数或 Dashboard 显示的 endpoint 为准。

## 健康与模型列表

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` 只表示 HTTP 进程可响应；`/healthz` 返回包含产品版本、模型、调度器、
缓存和内存状态的 JSON 快照。

`/healthz.memory.free_ram_bytes` 是操作系统报告的原始空闲页，仅用于观测；
`available_ram_bytes` 使用与进程内存 governor 相同的可回收内存口径。内存健康
状态由 `process_governor.pressure_level` 决定，而不是固定的 raw-free 阈值。
`degraded_reasons` 会列出队列、KV 缓存、内存压力、遥测或后端背压等具体原因。

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

当前本地模板只提供 reasoning 开关，不提供可校准的分级预算，因此
`minimal`、`low`、`medium`、`high`、`xhigh` 和 `max` 都表示启用模型原生
reasoning；`none` 表示关闭。具体推理长度由 checkpoint 决定。未显式指定 effort
时使用模型模板默认值。

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
`null`、nullable type 数组、`properties`、`required`、
`additionalProperties:false`、`items`、`enum`、`const` 和 `anyOf`。不支持的关键字
会在生成前返回 400，不会静默弱化。`strict:true` 要求每层 object 都设置
`additionalProperties:false`，并把所有 properties 列入 `required`。

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

具有受支持原生工具模板的 Qwen 3.5/3.6、Gemma 4、DiffusionGemma、GLM、
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
- 当前支持经过精确模板契约检测的 Qwen 3.5/3.6、Gemma 4/Gemma 4 Unified、
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
