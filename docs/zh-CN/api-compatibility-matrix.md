# IronMLX 0.2.0 API 兼容矩阵

[English](../api-compatibility-matrix.md)

本文面向客户端集成者，描述 0.2.0 的公开协议范围。使用发布 tag 对应的文档；候选包的精确源码提交以 Bundle 元数据为准，不以持续变化的分支作为发布基线。

使用示例见 [API 快速开始](api.md)，嵌套字段、工具、思考和 Schema 规则见[协议参考](api-reference.md)。模型相关条件见[支持模型](supported-models.md)。

## 通用规则

三套协议均支持同步文本、SSE、受支持的 Structured Outputs，以及依赖原生模板的客户端工具调用。
Responses 使用 typed events，Chat 以 `[DONE]` 结束，Messages 使用原生事件生命周期。
Responses reasoning item 和 Messages thinking block 依赖精确模板，Chat 不提供同形状的 typed reasoning。Messages 可组合 thinking 与最终 JSON 约束。

图片输入仅接受受支持的 base64 形状，不接受远程 URL。服务不执行外部工具、不提供托管工具或对话存储。
下表列出接受字段及条件，未列出字段或不符合条件的形状返回 400，不静默忽略。

## Chat Completions

| 字段 | 接受范围与限制 |
| --- | --- |
| `model` | 可省略，由当前服务或默认模型解析 |
| `messages` | 文本、受支持内容块及 assistant/tool 历史 |
| `tools / tool_choice / parallel_tool_calls` | function 工具；auto/none/required/指定函数；false 限制每轮一次调用，需存在 tools |
| `response_format` | text、json_object 或受支持的 json_schema |
| `stream / stream_options` | 同步/SSE；stream_options 仅接受 include_usage |
| `max_tokens` | 输出预算受模型上下文容量约束 |
| `temperature / top_p` | 有限数，范围分别为 [0,2] / (0,1] |
| `reasoning_effort` | Qwen3.8 原生 low/medium/xhigh 档位，要求匹配模板 |
| `seed / ignore_eos / chat_template_kwargs` | IronMLX 扩展：请求种子、受控长度生成及模板公开参数 |
| `functions / function_call / top_k / repetition_penalty` | 拒绝；旧函数字段应改用 tools |

## Responses

| 字段 | 接受范围与限制 |
| --- | --- |
| `model / instructions / input` | 无状态文本或受支持 typed 历史 |
| `tools` | function 和 namespace 子集，Schema 受限 |
| `tool_choice / parallel_tool_calls` | auto/none/required/指定函数；不能强制指定 namespace 子函数 |
| `text` | json_object 或受支持 json_schema 格式 |
| `stream / stream_options` | typed SSE；stream_options 必须为对象 |
| `max_output_tokens` | 输出预算受上下文容量约束 |
| `temperature / top_p` | 有限数，范围分别为 [0,2] / (0,1] |
| `reasoning` | 本地 effort/summary 语义；缺省 effort 为 none；明文输出 |
| `store / background` | 仅 false 或省略；true 被拒绝 |
| `previous_response_id / conversation` | 拒绝；无服务端历史存储 |
| `include` | 仅接受 reasoning.encrypted_content 请求形状；不生成加密内容 |
| `prompt_cache_key / client_metadata / metadata` | 仅校验结构和长度，不提供托管平台语义 |
| `service_tier / truncation` | tier 仅 auto/default；truncation 仅 disabled |
| `top_k / repetition_penalty` | 拒绝 |

## Anthropic Messages

| 字段 | 接受范围与限制 |
| --- | --- |
| `model / messages / system` | 文本、base64 图片、工具历史和签名 thinking；system 接受文本或文本块 |
| `tools / tool_choice` | 客户端函数；auto/any/指定 tool/none 及受支持的并行开关 |
| `output_config.format` | 仅受支持的 JSON Schema 格式 |
| `output_config.effort / thinking` | 要求兼容思考模板；disabled/enabled/adaptive，严格校验预算和 display；不代表校准后的 Claude 预算 |
| `max_tokens / stream` | 输出预算及同步/SSE 选择 |
| `temperature / top_p / top_k` | 有限数 [0,1] / (0,1]；top_k 为正整数 |
| `repetition_penalty / output_format` | 拒绝；格式应使用 output_config.format |
| `display:omitted / redacted_thinking` | 拒绝；无加密隐藏思考通道 |

## 语音合成

`POST /v1/audio/speech` 接受模型、输入文本、WAV/PCM 输出策略，以及一种参考来源：Base64 `ref_audio` 或托管的 `voice` ID，两者必须二选一。`voice` 字符串遵循 OpenAI-compatible 客户端惯例；IronMLX 将其解析为经过验证的本地参考音频。`GET /v1/audio/voices` 是供支持动态声音列表的客户端使用的 IronMLX 扩展。声音管理与参考录音试听接口见[语音 API](audio-speech-api.md)。

## 错误与重试

Chat/Responses 使用 OpenAI 错误信封；Messages 使用 Anthropic 信封，响应体 request_id 与 request-id header 对应。error.code 是 IronMLX 的机器可读扩展。

| HTTP | 场景 | code 示例 |
| ---: | --- | --- |
| 400 | JSON 或字段形状非法 | `invalid_json` |
| 400 | 采样或约束非法 | `invalid_request / invalid_sampling_parameters` |
| 400 | Schema、tools 或 thinking 不支持 | `invalid_response_format / invalid_tools / invalid_request` |
| 413 | 请求体超过 32 MiB | `request_body_too_large` |
| 413 | 输入与输出预算超过上下文 | `request_token_capacity_exceeded` |
| 503 | 队列、引擎、内存或存储背压 | `scheduler_queue_full / engine_unavailable / memory_budget_exceeded` |
| 500 | 非预期生成失败 | `generation_error` |

可重试 503 返回 JSON 和 `Retry-After: 5`；Messages 过载使用 `overloaded_error`，413 使用 `request_too_large`。完整错误码和断连释放边界见[协议参考](api-reference.md)。

## 运行方式与能力边界

普通服务、DFlash2、Gemma4 drafter、DiffusionGemma、EnginePool 和 App daemon 共用请求校验、协议错误、SSE headers 及流式断连释放契约，模型能力并不因此相同。
固定服务在启动时选择模型；EnginePool/App daemon 按请求 model 或默认模型分派。App DFlash2 保留只读模型发现，不提供动态模型管理。
DFlash2 的采样、并发及互斥组合见 [DFlash2 说明](dflash2-server-api.md)。

## SDK 兼容验证

| SDK | 固定版本 | 覆盖范围 |
| --- | --- | --- |
| OpenAI Python | `2.48.0` | Chat / Responses, SSE, tools, Structured Outputs, reasoning, 400/413/503 |
| Anthropic Python | `0.121.0` | Messages, SSE, tools, Structured Outputs + thinking, 400/413/503 |

固定 SDK 通过真实 loopback HTTP/SSE 访问 fixture server，验证客户端解析；Rust 测试独立验证生产请求和响应契约。两者不加载模型，不证明回答质量、工具选择准确率或性能。
在仓库根目录执行：

```bash
python3 -m venv /tmp/ironmlx-api-contract-sdk
/tmp/ironmlx-api-contract-sdk/bin/python -m pip install -r scripts/api-contract-sdk/requirements.txt
/tmp/ironmlx-api-contract-sdk/bin/python scripts/api-contract-sdk/contract.py --fixture
cargo test --locked --all-features -p ironmlx --lib server::
```

## 维护要求

新增字段需同步严格解析、协议测试、SDK 验证和两版文档。错误状态、信封、code 和 Retry-After 的变更也必须同步验证。扩展字段应明确标注，不能声称属于上游标准。真实模型结论需单独记录模型 revision、模板、量化、采样、响应模式和结果。
