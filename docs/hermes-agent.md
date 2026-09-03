# Hermes Agent integration

Hermes Agent can use the IronMLX App as an inference provider through the
Responses API without changing Hermes source. Start IronMLX and load the model
you intend to use first.

## Recommended configuration

Create a dedicated profile:

```bash
hermes profile create ironmlx
```

Edit `~/.hermes/profiles/ironmlx/config.yaml`:

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

The model ID must match the model loaded in IronMLX. Set the actual context
limit; Hermes Agent v0.20.0 requires at least 64000. Use `codex_responses` and
set `terminal.cwd` only when using the terminal tool. Local default mode needs
no API key.

## Verify

```bash
curl -fsS http://127.0.0.1:9068/healthz
hermes --profile ironmlx -z "Reply with exactly IRONMLX_OK"
hermes --profile ironmlx -z -t terminal "Use the terminal tool exactly once to run pwd, then report its output."
```

Hermes executes terminal and MCP tools and returns their results to IronMLX;
IronMLX only performs inference and emits structured tool calls.
