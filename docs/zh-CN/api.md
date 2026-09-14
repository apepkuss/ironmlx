# HTTP API 快速开始

[English](../api.md)

本文面向客户端集成者。先启动 App 并加载模型，将示例中的 `your-model-id` 替换为实际模型 ID。
App 默认地址为 `http://127.0.0.1:9068`；独立 CLI 默认端口为 8080，以实际设置为准。

## 检查服务与模型

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` 仅表示 HTTP 服务可响应；`/healthz` 提供运行状态。App 的模型列表包括已注册但未加载的模型。
同一 macOS 用户只能运行一个后端，启动独立 CLI 前应退出已有 App 后端。

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:9068/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

## OpenAI Responses

```bash
curl http://127.0.0.1:9068/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "input": "Hello", "store": false, "max_output_tokens": 128, "stream": false}'
```

Responses 为无状态接口，每次请求必须发送完整历史；不保存对话，也不执行工具。

## Anthropic Messages

```bash
curl http://127.0.0.1:9068/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

## 流式、工具和结构化输出

将 `stream` 设为 `true` 可使用 SSE。各协议的事件和终止标记不同，应使用对应客户端解析。
工具调用需要兼容模型；工具由客户端执行，再将结果送回后续请求。
Structured Outputs 的字段分别为 Chat 的 `response_format`、Responses 的 `text.format`、Messages 的 `output_config.format`。
字段形状、JSON Schema 子集、思考内容和历史回传规则见[协议参考](api-reference.md)。

## 图片、LAN 与错误处理

图片仅接受 JPEG/PNG/WebP base64，不抓取远程 URL。LAN 使用 HTTPS 和 Bearer API Key，客户端需信任导出的 CA。
端口、认证及图片容量要求见[安全边界](security-boundary.md)。日志级别管理接口仅供本机调用，不对 LAN 开放。

无效请求通常返回 400；请求体或上下文容量超限返回 413；可重试的过载返回 503 和 `Retry-After: 5`。
按 HTTP 状态及协议错误类型分类，再读取 `error.code`；完整规则见[API 兼容矩阵](api-compatibility-matrix.md)。

Agent 配置可直接使用 [Hermes Agent](hermes-agent.md) 或 [oh-my-pi](oh-my-pi.md) 指南。
