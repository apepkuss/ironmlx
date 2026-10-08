# 文本与视觉 API

[English](../text-vision-api.md) · [API 参考](api-reference.md) · [服务与管理 API](service-api.md)

本文定义 OpenAI Responses、Chat Completions 和 Anthropic Messages 的请求、响应、流式事件、工具及结构化输出。视觉能力指图片输入与理解；图片生成和编辑见[图片生成 API](image-generation-api.md)。服务地址、认证、健康检查、模型发现和管理接口见[服务与管理 API](service-api.md)。

## 阅读约定

“—”表示省略。未指定采样参数时使用已加载模型的默认值，并非上游托管服务的固定默认值。只有 Responses 会先采用模型配置的输出预算，再回退到 256；不会提高显式指定的预算。标为片段的 JSON 示例需合并到包含必填字段的完整请求中。

## API 兼容性与限制

IronMLX 提供兼容 OpenAI Responses、Chat Completions 和 Anthropic Messages 的接口，支持的字段与限制以下方说明为准。IronMLX 不执行客户端工具，不提供托管工具、对话存储或后台响应任务。

推理请求中的未知字段或不支持的字段形状返回 400；不支持的能力不会被静默忽略。

固定 CLI 服务在启动时选择模型；App 与 EnginePool 按请求 `model` 或明确的默认模型分派。图片、思考和工具能力取决于模型与原生模板，见[支持的模型](supported-models.md)。[DFlash2 配置](dflash2-server-api.md)说明其纯文本服务与采样限制。

## 端点索引

| 端点 | 用途 / 可用范围 |
| --- | --- |
| [`POST /v1/responses`](#openai-responses-api) | 支持 Responses 的文本/视觉模型 |
| [`POST /v1/chat/completions`](#openai-chat-completions) | 支持 Chat 的文本/视觉模型 |
| [`POST /v1/messages`](#anthropic-messages) | 支持 Messages 的文本/视觉模型 |

## OpenAI Responses API

`POST /v1/responses` · `application/json`

### Responses 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | string | 否 | 服务默认模型 | 使用 /v1/models 返回的 ID；没有明确默认模型时必须指定。 |
| `input` | string / array | 是 | — | 文本，或 message、reasoning、function_call、function_call_output 类型化历史。 |
| `instructions` | string | 否 | — | 本次请求的 system/developer 指令。 |
| `tools` | array | 否 | `[]` | 客户端 function 工具或受支持的 namespace 分组。 |
| `tool_choice` | string / object | 否 | `auto` | auto、none、required 或指定函数；不能强制指定 namespace 子函数。 |
| `parallel_tool_calls` | boolean | 否 | `true` | false 限制每轮一次调用；显式指定时需提供 tools。 |
| `text` | object | 否 | `{"format":{"type":"text"}}` | text.format 选择 text、json_object 或 json_schema。 |
| `stream` | boolean | 否 | `false` | true 返回类型化 SSE 事件。 |
| `stream_options` | object | 否 | — | 必须为对象；不启用 Chat 风格的 usage chunk。 |
| `max_output_tokens` | integer | 否 | 模型默认值或 256 | 正整数；包含思考与工具调用的总输出预算，不能超过可用上下文。 |
| `temperature` | number | 否 | 模型采样默认值 | 有限数，范围 [0,2]。 |
| `top_p` | number | 否 | 模型采样默认值 | 有限数，范围 (0,1]。 |
| `reasoning` | object | 否 | `effort: none` | effort：none/minimal/low/medium/high/xhigh/max；summary：none/auto。 |
| `store` | boolean | 否 | `false` | 仅支持 false；不存储响应。 |
| `background` | boolean | 否 | `false` | 仅支持 false；不提供后台任务。 |
| `include` | array of strings | 否 | `[]` | 仅接受 reasoning.encrypted_content；不生成加密内容。 |
| `prompt_cache_key` | string | 否 | — | 1–256 字节；仅校验，不提供托管平台缓存语义。 |
| `client_metadata` | object | 否 | — | 校验对象形状；不提供托管平台语义。 |
| `metadata` | object | 否 | — | 校验对象形状；不是持久化对话元数据。 |
| `service_tier` | string | 否 | `default` | auto/default 为前台；flex 在普通因果与 DFlash2 服务中使用后台优先级。见[请求优先级](#请求优先级)。 |
| `truncation` | string | 否 | `disabled` | 仅支持 disabled；不自动截断输入。 |

Responses 省略 max_output_tokens 时，先采用模型配置的输出预算，再回退到 256。

### 请求示例

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "input": "Hello", "store": false, "max_output_tokens": 128, "stream": false}'
```

### 响应

成功返回 HTTP 200 JSON。示例 ID、时间戳和 token 数为示意值。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id / object / created_at` | string / string / integer | 响应 ID、response 对象类型与 Unix 时间戳。 |
| `status` | string | 成功最终响应为 completed 或 incomplete；流式处理中为 in_progress。 |
| `output` | array | message、reasoning 或 function_call 类型化条目；按 type 解析。 |
| `output[].content` | array, conditional | 消息中的 output_text，或思考中的 reasoning_text 块。 |
| `output[].arguments / call_id / name` | string, conditional | 函数调用的 JSON 参数字符串、关联 ID 与公开名称；namespace 可选。 |
| `usage` | object or null | input_tokens、output_tokens、total_tokens 及详情对象；最终响应提供用量。 |
| `incomplete_details` | object or null | 达到输出上限时 reason 为 max_output_tokens。 |
| `error` | object or null | 成功为 null；流式失败由 response.failed 描述。 |
| `model` | string | 实际模型 ID。 |
| `store / background / previous_response_id` | boolean / boolean / null | false / false / null：无服务端响应状态。 |
| `reasoning / text / tools / tool_choice` | object / object / array / string or object | 有效思考、格式与工具配置。 |
| `max_output_tokens / parallel_tool_calls` | integer / boolean | 有效输出预算与并行调用设置。 |
| `temperature / top_p` | number or null | 显式请求值；省略时为 null。 |
| `instructions` | string or null | 请求中的指令。 |
| `service_tier / truncation` | string | 有效 tier 为 default 或 flex；truncation 为 disabled。 |

```json
{
  "id": "resp_example",
  "object": "response",
  "created_at": 0,
  "status": "completed",
  "background": false,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": 128,
  "model": "your-model-id",
  "output": [
    {
      "type": "message",
      "id": "msg_example",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "annotations": [],
          "logprobs": [],
          "text": "Hello!"
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": "none",
    "summary": null
  },
  "service_tier": "default",
  "store": false,
  "temperature": null,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": null,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 8,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 2,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 10
  }
}
```

### 流式响应

设置 `stream:true`。SSE 帧包含 `event:` 名称，JSON `data` 含 `type` 与递增的 `sequence_number`。生命周期为 `response.created` → 条目/内容事件及 delta → `response.completed`、`response.incomplete` 或 `response.failed`。没有 `[DONE]` 标记。最终成功事件包含响应对象与用量。

以下为文本事件节选，省略其他生命周期事件：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"content_index":0,"delta":"Hello!","item_id":"msg_example","logprobs":[],"output_index":0}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":4,"content_index":0,"item_id":"msg_example","logprobs":[],"output_index":0,"text":"Hello!"}
```

工具参数使用 `response.function_call_arguments.delta` / `.done`；拼接 delta 后再解析 JSON。思考使用 `response.reasoning_text.delta` / `.done`。保留 `response.output_item.done` 的完整条目用于历史重放。

### Responses reasoning

`reasoning.effort` 接受 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`；省略/null reasoning 或省略 effort 时为 `none`。Qwen3.8 将 minimal/low 映射为 low，medium 映射为 medium，high/xhigh/max 映射为 xhigh；其他受支持模板将非 none 档位视为开启开关。这些值不是经过校准的思考 token 预算。`summary` 接受 none/auto，但不生成独立摘要。

原生思考以独立 `reasoning` 条目和明文 `reasoning_text` 返回。只有加密内容的历史返回 400。没有后续 assistant 正文或调用的孤立明文 reasoning 会在历史重放时跳过；正常配对条目及其 status 应完整保留。

思考条目从 `in_progress` 开始，以 `completed` 或 `incomplete` 结束；只有仍打开的思考通道达到上限，才将该条目标为 incomplete。正文后续截断可使整体响应 incomplete，但已完成的思考仍为 completed。共享预算策略见[输出预算](#输出预算)。

```json
{
  "model": "your-model-id",
  "input": "先分析，再简洁回答。",
  "reasoning": {"effort": "high", "summary": "none"},
  "store": false
}
```

### Responses function tools

Responses 使用顶层 function tool 形状；后续历史请求应保留对应工具定义：

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

请求字段片段，需合并到包含必填字段的完整请求中：

```json
{"text":{"format":{"type":"json_object"}}}
```

Schema 模式使用：

请求字段片段，需合并到包含必填字段的完整请求中：

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
当请求同时包含 function tools 和 `text.format` 时：

- `tool_choice:"none"`：只允许结构化 JSON 最终回答。
- `tool_choice:"required"` 或指定函数：只允许工具调用。
- `tool_choice:"auto"`：允许原生工具调用，或符合 Schema 的 JSON 最终回答。

该联合约束适用于当前全部工具 dialect、普通生成、Scheduler、推测解码和
DiffusionGemma canvas 解码路径。

JSON Schema 子集见[通用约束](#json-schema)。

### 限制

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

`POST /v1/chat/completions` · `application/json`

### Chat 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | string | 否 | 服务默认模型 | 使用 /v1/models 返回的 ID；没有明确默认模型时必须指定。 |
| `messages` | array | 是 | — | 消息历史，包括受支持的内容块与工具结果。 |
| `tools` | array | 否 | — | 仅接受 type:function；见下方工具说明。 |
| `tool_choice` | string / object | 否 | 提供 tools 时为 auto | auto/none/required 或指定函数对象。 |
| `parallel_tool_calls` | boolean | 否 | `true` | false 限制每轮一次调用；显式指定时需提供 tools。 |
| `response_format` | object | 否 | `{"type":"text"}` | text、json_object 或包含嵌套 json_schema 的 json_schema。 |
| `stream` | boolean | 否 | `false` | true 返回 Chat completion SSE chunks。 |
| `stream_options` | object | 否 | — | 仅接受 include_usage:boolean，默认 false。 |
| `max_tokens` | integer | 否 | `256` | 包含思考与工具调用的总输出预算，受上下文容量约束。 |
| `service_tier` | string | 否 | `default` | auto/default 为前台；flex 在普通因果与 DFlash2 服务中使用后台优先级。见[请求优先级](#请求优先级)。 |
| `temperature` | number | 否 | 模型采样默认值 | 有限数，范围 [0,2]。 |
| `top_p` | number | 否 | 模型采样默认值 | 有限数，范围 (0,1]。 |
| `seed` | integer | 否 | — | IronMLX 扩展：无符号请求种子。 |
| `ignore_eos` | boolean | 否 | `false` | IronMLX 扩展：忽略 EOS，继续到 max_tokens，用于受控测量。 |
| `reasoning_effort` | string | 否 | 原生模板默认值 | Qwen3.8 的 low/medium/xhigh；要求匹配模板。 |
| `chat_template_kwargs` | object | 否 | — | IronMLX 扩展：enable_thinking 等受支持模板参数。 |

### 请求示例

```bash
curl http://127.0.0.1:9068/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

### 响应

成功返回 HTTP 200 JSON。示例 ID、时间戳和 token 数为示意值。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id / object / created / model` | string / string / integer / string | 响应 ID、chat.completion 类型、Unix 时间戳与模型 ID。 |
| `choices` | array | 生成结果数组；每项含 index、message 与 finish_reason。 |
| `choices[].message` | object | role:assistant、content:string 或 null，以及可选 tool_calls。 |
| `choices[].message.tool_calls` | array, optional | 每项含 id、type:function 与 function.name/arguments；arguments 为 JSON 字符串。 |
| `choices[].finish_reason` | string | stop、length 或 tool_calls；length 可导致 JSON 或文本截断。 |
| `usage` | object | prompt_tokens、completion_tokens 与 total_tokens。 |

```json
{
  "id": "chatcmpl_example",
  "object": "chat.completion",
  "created": 0,
  "model": "your-model-id",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 2,
    "total_tokens": 10
  }
}
```

### 流式响应

设置 `stream:true`。每个 data 帧为 `chat.completion.chunk`，`choices[].delta` 提供角色、文本或工具调用片段，随后返回终止原因和 `data: [DONE]`。文本流示例：

```text
data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{"content":"Hello!"}}]}

data: {"id":"chatcmpl_example","object":"chat.completion.chunk","created":0,"model":"your-model-id","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

设置 `stream_options.include_usage:true` 时，在 `[DONE]` 前额外返回含 `choices:[]` 与 `usage` 的最终用量 chunk。工具参数字符串可跨多次 delta，按 index/id 关联，拼接后再解析 JSON。

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

工具调用历史必须提供匹配且非空的 `tools` 数组。

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

### 限制

仅接受 function 工具与受支持原生模板。拒绝旧 functions/function_call、top_k 与 repetition_penalty。Qwen3.8 reasoning_effort 接受 low/medium/xhigh，原生默认 xhigh；chat_template_kwargs.enable_thinking:false 关闭思考，preserve_thinking:false 不回传历史 reasoning_content。需要独立思考条目/块时使用 Responses/Messages。

## Anthropic Messages

`POST /v1/messages` · `application/json`

### Messages 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | string | 否 | 服务默认模型 | 使用 /v1/models 返回的 ID；没有明确默认模型时必须指定。 |
| `messages` | array | 是 | — | user/assistant 历史，可包含文本、base64 图片、工具与本地 thinking 块。 |
| `system` | string / array | 否 | — | system 文本或受支持的文本块。 |
| `tools` | array | 否 | — | 包含 name 与 input_schema 的客户端工具。 |
| `tool_choice` | object | 否 | 提供 tools 时为 auto | type:auto/any/tool/none；受支持选择可设置 disable_parallel_tool_use。 |
| `output_config` | object | 否 | — | format 使用 type:json_schema 与 schema；effort 要求启用 enabled/adaptive thinking。 |
| `thinking` | object | 否 | 关闭 | type:disabled/enabled/adaptive；enabled 要求 budget_tokens >=1024 且 <max_tokens。 |
| `max_tokens` | integer | 否 | `256` | 包含思考与工具调用的总输出预算，受上下文容量约束。 |
| `stream` | boolean | 否 | `false` | true 返回原生 Messages SSE 事件。 |
| `temperature` | number | 否 | 模型采样默认值 | 有限数，范围 [0,1]。 |
| `top_p` | number | 否 | 模型采样默认值 | 有限数，范围 (0,1]。 |
| `top_k` | integer | 否 | 模型采样默认值 | 正整数。 |

### 请求示例

```bash
curl http://127.0.0.1:9068/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

### 响应

成功返回 HTTP 200 JSON。示例 ID、时间戳和 token 数为示意值。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id / type / role / model` | string | 消息 ID、message 类型、assistant 角色与模型 ID。 |
| `content` | array | text、thinking 或 tool_use 类型化块；保留原顺序。 |
| `content[].thinking / signature` | string, conditional | 本地思考文本与完整性签名；重放时保持两者不变。 |
| `content[].id / name / input` | string / string / object, conditional | tool_use ID、工具名称与已解析的 JSON 参数。 |
| `stop_reason` | string or null | end_turn、max_tokens 或 tool_use；流式未结束时为 null。 |
| `stop_sequence` | null | 未提供受支持的 stop-sequence 请求字段。 |
| `usage` | object | input_tokens、output_tokens 与可选 output_tokens_details.thinking_tokens。 |

```json
{
  "id": "msg_example",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello!"
    }
  ],
  "model": "your-model-id",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 8,
    "output_tokens": 2
  }
}
```

### 流式响应

设置 `stream:true`。原生事件顺序为 message_start → content_block_start/delta/stop 分组 → message_delta → message_stop。内容 index 标识块；工具参数使用 input_json_delta.partial_json。思考块使用 thinking_delta 与 signature_delta。以下为完整纯文本生命周期示例：

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_example","type":"message","role":"assistant","content":[],"model":"your-model-id","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":8,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
```

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

响应字段片段：

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

### 限制

不支持旧 output_format、repetition_penalty、托管/服务器工具、MCP、computer use、Web Search 或 Code Execution。思考历史必须使用本地生成的签名，不能重放 Claude 托管签名、redacted/omitted/交错 thinking。

## 通用约束

### 请求优先级

Chat Completions 与 Responses 接受 `service_tier:auto/default`，使用前台优先级。
`flex` 在普通因果与 DFlash2 服务中以后台优先级运行，可被前台请求抢占；
其他 tier 值返回 400。Flex 选择调度优先级，不创建后台响应任务。
Responses 响应中的有效 tier 为 `default` 或 `flex`。

### 图片输入

图片输入仅接受 JPEG、PNG 或 WebP 的 base64 数据，不抓取远程 HTTP/HTTPS URL，也不接受 OpenAI file ID。三套协议的嵌套形状不同：

| 协议 | 位置 | 图片内容块 |
| --- | --- | --- |
| Chat | `messages[].content[]` | `{"type":"image_url","image_url":{"url":"data:image/png;base64,<Base64>"}}` |
| Responses | `input[].content[]` | `{"type":"input_image","image_url":"data:image/png;base64,<Base64>"}` |
| Messages | `messages[].content[]` | `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<Base64>"}}` |

将 `<Base64>` 替换为实际图片数据，并与文本块一起放入消息的内容数组。图片要求支持视觉输入的模型；数量、字节与像素限制见[安全边界](security-boundary.md)。

### JSON Schema

支持类型：object、array、string、number、integer、boolean、null 和可空类型数组。支持关键字：properties、required、items、enum、const、anyOf、minItems/maxItems、minLength/maxLength、minimum/maximum、exclusiveMinimum/exclusiveMaximum。最大嵌套深度为 8。

strict:true 要求每层 object 设置 additionalProperties:false，且所有属性列入 required。顶层工具参数对象必须封闭。非 strict 工具可在嵌套对象使用 additionalProperties:true 或受支持的值 Schema。不支持的关键字在生成前返回 400，strict 约束不静默降级。

### 输出预算

输入（包括模板与图片）加请求输出预算必须符合有效上下文容量，超限返回 413 request_token_capacity_exceeded。max_output_tokens/max_tokens 包含思考、正文与工具调用。

对于具有精确原生思考模板的 causal Qwen3.5/3.6/3.8 和 Gemma4/Unified，启用思考时会为正文/工具调用预留 min(floor(总输出预算/4),1024) token，另保留原生通道标记空间，其余用于思考。三套协议采用相同策略，不提高总预算，也不自动重试。关闭/不支持思考、DiffusionGemma 或不足以容纳通道标记的小额预算保留原行为。

思考通道已完成不代表答案完整。解析完整约束 JSON 前应检查 Responses status/incomplete_details、Chat finish_reason:length 或 Messages stop_reason:max_tokens。

### 流式与取消

流式响应使用 text/event-stream。网络 chunk 不等同于 SSE 帧，工具 JSON 必须拼接 delta 后解析。开始流式前的错误使用普通 HTTP 错误格式；HTTP 200 后发生的错误通过协议流式事件报告。

对于 Chat Completions，如果内部生成流在没有终止事件时关闭，请求按失败处理，不会被报告为正常停止。文本和工具流式响应会发送错误，不伪造成功的终止原因、最终用量 chunk 或 `[DONE]`；非流式请求返回 HTTP 错误。已经收到的部分内容不代表请求成功完成。

客户端断连会在下一安全执行边界取消流式生成并释放请求资源，不会中断正在执行的 Metal 操作。断连后不伪造终止事件。不保证非流式请求断连时取消已运行的生成。

## 错误契约

请求体上限为 32 MiB，超限返回 413 `request_body_too_large`。

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
| 400 | `invalid_json`、`invalid_request`、`invalid_sampling_parameters`、`invalid_response_format`、`invalid_tools` 等 | JSON、字段、采样或输出约束不合法 | 无 |
| 401 | `auth_invalid` | LAN Key 缺失或无效 | 无 |
| 404 | `model_not_found` | 未知或被禁用的模型 | 无 |
| 413 | `request_body_too_large` | HTTP request body 超过 32 MiB | 无 |
| 413 | `request_token_capacity_exceeded` | 输入 token 与请求输出预算超过模型上下文容量；`error.details` 提供容量明细 | 无 |
| 503 | `scheduler_queue_full`、`scheduler_unavailable`、`scheduler_reply_lost` | 调度器暂时不可用 | `5` 秒 |
| 503 | `memory_budget_exceeded`、`memory_pressure`、`prefill_peak_unsafe`、`vision_prefill_peak_unsafe`、`cold_materialization_unsafe`、`prefix_store_backpressure` | 内存 governor 或存储背压暂时拒绝请求 | `5` 秒 |
| 503 | `engine_unavailable`、`diffusion_lane_overloaded` | 模型引擎或 DiffusionGemma lane 暂时不可用 | `5` 秒 |
| 500 | `generation_error` 及内部任务错误码 | 非预期服务端错误 | 无 |

所有可重试 503 都返回 JSON 和 `Retry-After: 5`。IronMLX 的 Messages 本地契约使用
HTTP 503 + `overloaded_error` 表达暂时过载；413 使用 `request_too_large`，并通过
上述两个稳定 code 区分传输体上限与模型上下文容量上限。

上述格式适用于本页三套文本与视觉协议的推理及 LAN 认证错误；语音、图片生成和 System One 的错误见各自参考。本机管理端点的成功与错误格式见[服务与管理 API](service-api.md#管理接口)。

## 相关参考

- [服务与管理 API](service-api.md)：公共访问约定、健康检查、模型发现及管理接口。
- [语音合成 API](audio-speech-api.md)：WAV/PCM、声音管理与参考录音。
- [图片生成 API](image-generation-api.md)：文生图与单图条件编辑；不提供蒙版局部重绘、图片变体或 Anthropic 图片生成接口。
- [System One API](laya-systemone-api.md)：类型化决策请求与模型发现。
- [SDK 兼容验证](building-from-source.md#sdk-兼容验证)：协议与客户端 SDK 的检查方法。
- Agent 配置：[Hermes Agent](hermes-agent.md)、[oh-my-pi](oh-my-pi.md)、[DeepSeek Harness](dsh.md)。
