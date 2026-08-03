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
