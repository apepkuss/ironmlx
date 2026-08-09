# IronMLX v0.1 API 兼容矩阵

状态基线：`dev@c82659b`。本文是 IronMLX v0.1 的公开协议承诺，适用于：

- `POST /v1/chat/completions`（OpenAI Chat Completions 子集）；
- `POST /v1/responses`（OpenAI Responses 无状态子集）；
- `POST /v1/messages`（Anthropic Messages 子集）。

本文中的“支持”表示字段有明确的解析、验证和响应语义，并由服务端 contract
测试覆盖；“受限”表示只接受文档列出的子集或依赖模型 capability；“拒绝”表示
收到后返回 400，不能静默忽略或降级。

## 符号与证据

| 符号 | 含义 |
|---|---|
| ✅ | 支持并纳入 v0.1 承诺 |
| ◐ | 受限支持：仅文档列出的形状、模型或取值有效 |
| ❌ | 明确拒绝，返回 400 |
| — | 该协议没有此字段或语义 |

发布门禁由两层组成：

1. Rust 服务端 contract suite：`cargo test --locked --all-features -p ironmlx --lib core::server::`，覆盖请求提取、字段校验、错误渲染、主要拓扑和 SSE 断连释放；
2. 固定版本官方 SDK 黑盒 suite：`scripts/api-contract-sdk/contract.py --fixture`，使用 OpenAI Python SDK `2.48.0` 和 Anthropic Python SDK `0.121.0`，通过真实 loopback HTTP/SSE 验证客户端解析。

这两层不证明特定 checkpoint 的回答质量、工具选择准确率或模型生成的 JSON
质量；真实模型 HTTP/SDK smoke 是独立发布验收层。

## 能力矩阵

| 能力 | Chat Completions | Responses | Anthropic Messages | v0.1 边界 |
|---|---:|---:|---:|---|
| 同步文本响应 | ✅ | ✅ | ✅ | 使用各自协议的原生响应 envelope |
| SSE 文本响应 | ✅ | ✅ | ✅ | Responses 使用 typed events；Chat 使用 `[DONE]`；Messages 使用原生 event lifecycle |
| 客户端 function tools | ◐ | ◐ | ◐ | 依赖模型原生 tool template 和受支持 JSON Schema；IronMLX 只生成调用，不执行函数 |
| Responses namespace tools | — | ◐ | — | 仅 Responses；namespace 编译为有界内部 dispatcher，仍由客户端执行实际函数 |
| 最终答案 Structured Outputs | ✅ | ✅ | ✅ | Chat=`response_format`；Responses=`text.format`；Messages=`output_config.format` |
| tools 与最终 JSON 组合 | ✅ | ✅ | ✅ | `auto` 可在工具调用和结构化最终答案间选择；强制 tool choice 时只允许工具调用 |
| 原生 typed reasoning/thinking | — | ◐ | ◐ | Responses 使用 `reasoning` item；Messages 使用 `thinking` block；依赖精确模型模板契约 |
| thinking + Structured Outputs | — | — | ◐ | Messages 先自由生成 thinking section，再对最终 text section 启用 JSON grammar |
| 完整历史回灌 | ◐ | ✅ | ✅ | Chat 回灌 assistant/tool history；Responses 回灌 typed input item；Messages 回灌 signed thinking/tool history |
| 图片输入 | ◐ | ◐ | ◐ | Chat 严格 `data:` URL；Responses 仅支持文档列出的 input image 形状；Messages 仅 base64 |
| 远程图片 URL | ❌ | ❌ | ❌ | 不抓取 HTTP/HTTPS 图片 URL |
| `temperature` | ✅ `[0,2]` | ✅ `[0,2]` | ✅ `[0,1]` | 必须为有限数；非法取值返回 400 |
| `top_p` | ✅ `(0,1]` | ✅ `(0,1]` | ✅ `(0,1]` | 必须为有限数；非法取值返回 400 |
| `top_k` | ❌ | ❌ | ✅ 正整数 | 仅 Anthropic Messages 是公开 sampling 字段 |
| `repetition_penalty` | ❌ | ❌ | ❌ | 不是三套协议的公开字段 |
| 服务端工具执行、Shell、MCP、HTTP | ❌ | ❌ | ❌ | 服务只生成结构化调用，不执行外部动作 |
| response/conversation 持久化 | ❌ | ❌ | ❌ | Responses 仅无状态请求；`store:true` 等返回 400 |

## 顶层请求字段矩阵

未列出的顶层字段均因 `deny_unknown_fields` 或等价契约校验返回 400。字段的
“受限”不是静默忽略：取值或组合不符合下表时同样返回 400。

### Chat Completions

| 字段 | 状态 | 接受形状/语义 |
|---|---:|---|
| `model` | ◐ | 可省略，使用服务默认模型；指定值必须能由当前拓扑解析 |
| `messages` | ✅ | 文本、严格 content parts、assistant/tool history |
| `tools` | ◐ | `type:function`；模型模板和 JSON Schema 子集受限 |
| `tool_choice` | ◐ | `auto`、`none`、`required` 或指定 function |
| `parallel_tool_calls` | ◐ | 需存在 tools；`false` 限制当前 turn 最多一个调用 |
| `response_format` | ✅ | `text`、`json_object`、`json_schema`；Schema 必须符合支持子集 |
| `stream` | ✅ | 同步或 SSE |
| `stream_options` | ◐ | 仅 `include_usage` |
| `max_tokens` | ✅ | 输出预算；受模型上下文容量约束 |
| `temperature` / `top_p` | ✅ | 取值范围见能力矩阵 |
| `seed` | ◐ | IronMLX 扩展；用于请求级随机种子，不属于跨服务兼容承诺 |
| `ignore_eos` | ◐ | IronMLX 扩展；用于受控长度/基准测试，不属于 OpenAI 标准字段 |
| `chat_template_kwargs` | ◐ | IronMLX 扩展；只允许模型模板公开的 kwargs，不能替代协议字段 |
| `functions` / `function_call` | ❌ | 已废弃字段，明确要求改用 `tools` |
| `top_k` / `repetition_penalty` | ❌ | 不属于 Chat 公开 sampling 契约 |

### Responses

| 字段 | 状态 | 接受形状/语义 |
|---|---:|---|
| `model` / `instructions` / `input` | ✅ | 无状态 typed input；`input` 为文本或支持的 typed item 历史 |
| `tools` | ◐ | function 与 namespace 子集；工具参数 Schema 受限 |
| `tool_choice` / `parallel_tool_calls` | ◐ | `auto`、`none`、`required` 和 function 选择；namespace 不支持指定子函数 |
| `text` | ✅ | `format=json_object` 或受支持的 `json_schema` |
| `stream` / `stream_options` | ◐ | 原生 Responses SSE；`stream_options` 仅接受对象形状 |
| `max_output_tokens` | ✅ | 输出预算；受模型上下文容量约束 |
| `temperature` / `top_p` | ✅ | 取值范围见能力矩阵 |
| `reasoning` | ◐ | `effort` 与 `summary` 仅支持文档列出的本地语义；输出为明文 reasoning item |
| `store` | ◐ | 只能省略或设为 `false`；`true` 返回 400 |
| `previous_response_id` / `conversation` | ❌ | 本地服务不提供服务端 response/conversation 存储 |
| `background` | ◐ | 仅 `false`；`true` 返回 400，不启动后台任务 |
| `include` | ◐ | 仅接受 `reasoning.encrypted_content` 请求形状；IronMLX 不生成 encrypted content |
| `prompt_cache_key` / `client_metadata` / `metadata` | ◐ | 仅执行结构和长度校验，不承诺 OpenAI 托管平台语义 |
| `service_tier` | ◐ | 仅 `auto` 或 `default`；不提供 OpenAI 托管 tier |
| `truncation` | ◐ | 仅 `disabled` |
| `top_k` / `repetition_penalty` | ❌ | 不属于 Responses 公开 sampling 契约 |

### Anthropic Messages

| 字段 | 状态 | 接受形状/语义 |
|---|---:|---|
| `model` / `messages` | ✅ | 文本、base64 图片、tool_use/tool_result 和 signed thinking history |
| `system` | ✅ | 文本或文本 block |
| `tools` | ◐ | 客户端 function tools；`input_schema` 使用受支持 Schema 子集 |
| `tool_choice` | ◐ | `auto`、`any`、指定 `tool`、`none`；支持并行调用开关 |
| `output_config.format` | ✅ | 仅受支持的 JSON Schema format |
| `output_config.effort` | ◐ | 需同时启用 `thinking`；表示本地模板开关，不是已校准的 Claude 预算档位 |
| `thinking` | ◐ | `disabled`、`enabled`、`adaptive`；`budget_tokens` 和 `display` 有严格限制 |
| `max_tokens` / `stream` | ✅ | 输出预算及同步/SSE 选择 |
| `temperature` / `top_p` / `top_k` | ✅ | 取值范围见能力矩阵 |
| `repetition_penalty` | ❌ | 不是 Anthropic Messages 公开字段 |
| `display:omitted` / `redacted_thinking` | ❌ | 本地没有 Claude 加密隐藏思考通道 |

## 错误矩阵

三套协议共享 HTTP status 和稳定机器码，但错误 body 使用协议原生 envelope。

| 场景 | HTTP | 稳定 code | Chat / Responses body | Messages body | `Retry-After` |
|---|---:|---|---|---|---:|
| JSON 无法解析 | 400 | `invalid_json` | OpenAI `error` envelope | Anthropic `type:error` envelope | — |
| 未知字段或 JSON 形状非法 | 400 | `invalid_json` | OpenAI envelope，`error.code=invalid_json` | Anthropic envelope，`error.code=invalid_json` | — |
| 已识别字段组合或 sampling 非法 | 400 | `invalid_request`、`invalid_sampling_parameters` 等 | OpenAI envelope，`error.code` 可定位原因 | Anthropic envelope，`error.code` 为 IronMLX 扩展 | — |
| Structured Outputs / tools / thinking Schema 不支持 | 400 | `invalid_response_format`、`invalid_tools` 或 `invalid_request` | OpenAI envelope | Anthropic envelope | — |
| HTTP body 超过 32 MiB | 413 | `request_body_too_large` | OpenAI envelope | Anthropic envelope，`error.type=request_too_large` | — |
| token 与输出预算超过上下文容量 | 413 | `request_token_capacity_exceeded` | OpenAI envelope，含容量 details | Anthropic envelope，`error.type=request_too_large` | — |
| 调度队列/引擎暂时不可用 | 503 | `scheduler_queue_full`、`scheduler_unavailable`、`engine_unavailable` 等 | OpenAI envelope | Anthropic envelope，过载使用 `overloaded_error` | `5` |
| 内存 governor / prefix store 背压 | 503 | `memory_budget_exceeded` 等 | OpenAI envelope | Anthropic envelope | `5` |
| 非预期生成错误 | 500 | `generation_error` 或内部错误码 | OpenAI envelope | Anthropic envelope | — |
| SSE 客户端断连 | — | — | 停止生成事件，不发送终止事件 | 同左 | — |

`request-id` 由 Messages 响应 body 与 `request-id` header 共同提供；OpenAI 响应
遵循 SDK 可解析的 OpenAI error envelope。客户端应先按 HTTP status 和协议 envelope
分类，再使用 `error.code` 选择重试、修正请求或报告内部错误。

## 拓扑矩阵

| 契约 | 普通服务 | Gemma4 drafter | DiffusionGemma | EnginePool | App daemon |
|---|---:|---:|---:|---:|---:|
| 请求字段/模型无关校验 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 协议错误 envelope | ✅ | ✅ | ✅ | ✅ | ✅ |
| 413 body/token 区分 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 503 JSON + `Retry-After:5` | ✅ | ✅ | ✅ | ✅ | ✅ |
| SSE `text/event-stream` + `no-cache` | ✅ | ✅ | ✅ | ✅ | ✅ |
| TCP 断连后驱逐与预算释放 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 模型选择 | 启动时固定 | 启动时固定 | 启动时固定 | request model 或默认模型 | request model 或默认模型 |
| 模型能力完全相同 | ❌ | ❌ | ❌ | ❌ | ❌ |

最后一行是有意保留的限制：拓扑统一的是 transport、协议和生命周期契约，不是
模型架构、sampling、MTP、KV cache 或工具模板能力。具体模型能力仍以
[`docs/supported-models.md`](supported-models.md) 为准。

## SDK 与版本矩阵

| SDK | 固定版本 | 覆盖 | 当前不承诺 |
|---|---:|---|---|
| OpenAI Python | `2.48.0` | Chat 同步/SSE、Responses typed output/reasoning、function tools、Structured Outputs、400/413/503 | OpenAI 平台存储、托管工具、background、conversation、encrypted reasoning |
| Anthropic Python | `0.121.0` | Messages 同步/SSE、tool_use、tool_result、Structured Outputs + adaptive thinking、400/413/503 | Anthropic 托管工具、MCP、computer use、Web Search、Code Execution、Claude 加密 thinking |

SDK contract 使用 deterministic fixture server；它验证官方客户端对公开 wire
shape 的解析，不代表某个模型一定会选择工具或生成符合 Schema 的答案。真实模型
验收必须另行记录模型 ID、模板、量化、采样参数、响应模式和结果。

## 发布规则

本矩阵是 v0.1 的公开承诺边界：

1. 新增字段必须先进入 DTO 严格解析、协议 contract tests 和 SDK 兼容验证，再更新矩阵；
2. 不支持字段必须返回 400，不能静默忽略或退化到默认采样；
3. 任何错误 status、envelope、稳定 code 或 `Retry-After` 改动，都必须同步更新错误矩阵和 SDK contract；
4. 真实模型能力、性能和质量结论不能仅凭 fixture/SDK contract 宣称；
5. IronMLX 扩展（例如 Chat 的 `seed`、`ignore_eos`、`chat_template_kwargs`）必须使用独立命名并标注为扩展，不得伪装成 OpenAI 或 Anthropic 标准字段。
