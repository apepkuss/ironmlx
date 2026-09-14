# Hermes Agent integration

[简体中文](zh-CN/hermes-agent.md)

Hermes Agent can use the IronMLX App as an inference provider through the
Responses API without changing Hermes source. Start IronMLX and load the model
you intend to use first.

## Use the Dashboard guide

Select Hermes Agent on Dashboard **Agent**, generate configuration from the current endpoint/model, then copy it into the client configuration file. The guide does not install the client.
The guide targets Hermes Agent v0.20.0 and later, with an actual model context limit of at least 64000 tokens.
See the [official configuration guide](https://hermes-agent.nousresearch.com/docs/integrations/providers#custom--self-hosted-llm-providers). Manual steps follow.

## Manual configuration

Create a dedicated profile:

```bash
hermes profile create ironmlx
```

Edit `~/.hermes/profiles/ironmlx/config.yaml`:

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

For the default profile, use `~/.hermes/config.yaml` and `--profile default`; this changes the default provider. Use `hermes --profile ironmlx --tui` for TUI or select the profile in Desktop before creating a session.

## Confirm the result

The text check should return `IRONMLX_OK`. The tool check should execute `pwd` once and return the working directory; a model merely describing the command is not a successful tool call.
If it fails, check the endpoint, model ID and template support, then see [Troubleshooting](troubleshooting.md).
