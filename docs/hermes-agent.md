# Hermes Agent 集成

Hermes Agent 可通过 Responses API 将 IronMLX App 用作推理服务，无需修改 Hermes
Agent 源码。开始前，请先启动 IronMLX App 并加载需要使用的模型。

## 配置

编辑 `~/.hermes/config.yaml`：

```yaml
model:
  default: "mlx-community/Qwen3.5-2B-4bit"
  provider: "custom:ironmlx-responses"
  context_length: 262144

providers:
  ironmlx-responses:
    api: "http://127.0.0.1:9068/v1"
    transport: "codex_responses"
    discover_models: false

terminal:
  env_type: "local"
  cwd: "/absolute/path/to/agent-workspace"
```

- `model.default` 必须与 IronMLX 中加载的模型 ID 一致。
- `model.context_length` 应填写该模型在 IronMLX 中的实际上下文上限；Hermes
  Agent v0.20.0 要求至少 64000。
- `transport` 必须为 `codex_responses`。
- 仅在使用 terminal 工具时需要设置 `terminal.cwd`。
- 本机默认配置不需要 API Key。

## 验证

```bash
curl -fsS http://127.0.0.1:9068/healthz
hermes -z "Reply with exactly IRONMLX_OK"
hermes -z -t terminal "Use the terminal tool exactly once to run pwd, then report its output."
```

Hermes 负责执行 terminal、MCP 等工具，并将工具结果回传给 IronMLX；IronMLX 只负责
推理和生成结构化工具调用。
