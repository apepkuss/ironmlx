# HTTP API quick start

[简体中文](zh-CN/api.md)

For client integrators: start the App, load a model and replace `your-model-id` with its actual ID.
The default App endpoint is `http://127.0.0.1:9068`; direct CLI serving defaults to port 8080. Use the actual configured endpoint.

## Check the service and models

```bash
curl http://127.0.0.1:9068/health
curl http://127.0.0.1:9068/healthz
curl http://127.0.0.1:9068/v1/models
```

`/health` only indicates HTTP responsiveness; `/healthz` reports runtime state. The App model list includes registered models that are not loaded.
Only one backend may run per macOS user; exit the existing App backend before starting a separate CLI server.

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

Responses is stateless: send complete history with every request. The service does not store conversations or execute tools.

## Anthropic Messages

```bash
curl http://127.0.0.1:9068/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "your-model-id", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128, "stream": false}'
```

## Streaming, tools and structured outputs

Set `stream` to `true` for SSE. Events and termination markers differ by protocol; use the corresponding client parser.
Tools require a compatible model. The client executes calls and sends results in the next request.
Structured Outputs use `response_format` in Chat, `text.format` in Responses and `output_config.format` in Messages.
See the [protocol reference](api-reference.md) for field shapes, the JSON Schema subset, reasoning and history replay.

## Images, LAN and errors

Images accept JPEG/PNG/WebP base64 only; remote URLs are not fetched. LAN uses HTTPS and a Bearer API key, and clients must trust the exported CA.
See [Security boundaries](security-boundary.md) for authentication and image limits. Log-level management is loopback-only and unavailable over LAN.

Invalid requests generally return 400, body/context limits return 413, and retryable overload returns 503 with `Retry-After: 5`.
Classify by HTTP status and protocol error type, then inspect `error.code`. See the [API compatibility matrix](api-compatibility-matrix.md).

For Agent configuration, use the [Hermes Agent](hermes-agent.md), [oh-my-pi](oh-my-pi.md), or [DeepSeek Harness](dsh.md) guides.
