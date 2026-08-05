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
- 不支持旧 `functions` / `function_call` 字段，也不实现 `/v1/responses`。
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
