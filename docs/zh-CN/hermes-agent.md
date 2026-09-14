# Hermes Agent 集成

[English](../hermes-agent.md)

Hermes Agent 可通过 Responses API 将 IronMLX App 用作推理服务，无需修改 Hermes
Agent 源码。开始前，请先启动 IronMLX App 并加载需要使用的模型。

## 使用 Dashboard 向导

在 Dashboard 的 **Agent** 页选择 Hermes Agent，按当前 endpoint 和模型生成配置，再复制到客户端指定文件。向导不代替客户端安装。
向导标注适用于 Hermes Agent v0.20.0 及以上；模型实际上下文上限需满足至少 64000 tokens。
[前往官方配置指南](https://hermes-agent.nousresearch.com/docs/zh-Hans/integrations/providers#%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%87%AA%E6%89%98%E7%AE%A1-llm-%E6%8F%90%E4%BE%9B%E5%95%86)。以下保留手动配置方法。

## 手动配置

创建专用 profile，避免影响 Hermes 默认配置和会话：

```bash
hermes profile create ironmlx
```

编辑 `~/.hermes/profiles/ironmlx/config.yaml`：

```yaml
model:
  default: "mlx-community/Qwen3.5-2B-4bit"
  provider: "custom:ironmlx-responses"
  context_length: 64000

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

如需直接使用默认 profile，也可将同一配置写入 `~/.hermes/config.yaml`；这会改变
Hermes 默认 profile 使用的 Provider。

## 验证

```bash
curl -fsS http://127.0.0.1:9068/healthz
hermes --profile ironmlx -z "Reply with exactly IRONMLX_OK"
hermes --profile ironmlx -z -t terminal "Use the terminal tool exactly once to run pwd, then report its output."
```

- TUI：`hermes --profile ironmlx --tui`
- Desktop：选择 `ironmlx` profile 后新建会话。

若使用默认 profile，请将命令中的 `--profile ironmlx` 改为 `--profile default`。

Hermes 负责执行 terminal、MCP 等工具，并将工具结果回传给 IronMLX；IronMLX 只负责
推理和生成结构化工具调用。

## 判断是否成功

文本检查应返回 `IRONMLX_OK`；工具检查应实际执行一次 `pwd` 并返回工作目录。仅收到模型描述命令的文本不算工具调用成功。
失败时先核对 endpoint、模型 ID 和模板支持，再查看[故障排查](troubleshooting.md)。
