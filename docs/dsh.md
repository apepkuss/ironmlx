# DeepSeek Harness (dsh) integration

[简体中文](zh-CN/dsh.md)

DeepSeek Harness can use IronMLX as an inference provider through the Responses API. This guide was verified with dsh `master` at `ddefc45fbc` and a local Qwen3.8-27B-4bit + DFlash2 service. Other dsh revisions and models may behave differently. Start IronMLX and load the intended model first; install dsh separately following its own project instructions.

## Use the Dashboard guide

On the IronMLX Dashboard **Agent** page, select **DeepSeek Harness**. The guide uses the current local endpoint and loaded model, reads `context_window` and `max_output_tokens` from `/v1/models` when available, and generates a dsh overlay. Review the per-turn output budget before copying: the guide initially uses at most 4096 tokens, not the service's full output capability. If capacity information is unavailable, enter the model's actual limits manually.

Save the overlay as `$DSH_HOME/ironmlx.patch.yml` (normally `~/.dsh/ironmlx.patch.yml`). Pass it with `--patch` only on runs that should use IronMLX; do not replace an existing home-level `cordis.patch.yml`.
If `$DSH_HOME/settings.yaml` already selects a default model or contains `llm-pi-ai` overrides, update or remove conflicting entries there: dsh applies user settings after the overlay's base configuration.

## Manual configuration

The following is an example; replace `MODEL_ID` and both capacities with values for the model actually served by IronMLX. `maxTokens` is dsh's default per-request output budget as well as its configured model output limit; reasoning and visible text share it. Keep it within the server's reported `max_output_tokens` and the available context. Use the **Off** option in the Dashboard guide, or change `reasoningEfforts` to `false` and omit `reasoning: high`, for a non-reasoning model.

```yaml
- id: agent-default-model
  config:
    provider: ironmlx-local
    model: "MODEL_ID"

- id: llm-pi-ai
  config:
    providers:
      ironmlx-local:
        displayName: IronMLX
        apiKeyEnv: IRONMLX_API_KEY
        api: openai-responses
        baseURL: "http://127.0.0.1:9068/v1"
        cacheRetention: none
        reasoning: high
        models:
          - id: "MODEL_ID"
            name: "MODEL_ID"
            contextWindow: 16384
            maxTokens: 4096
            reasoningEfforts:
              off:
              high: high
```

`IRONMLX_API_KEY=local` is a placeholder that satisfies dsh's credential lookup for the local keyless endpoint. It is not an IronMLX LAN API key. This guide targets the local loopback listener, not authenticated LAN access.

## Verify

Run from a project directory after saving the overlay. If `DSH_HOME` is customized, the commands use that directory automatically.

```bash
curl -fsS http://127.0.0.1:9068/healthz
export IRONMLX_API_KEY=local
dsh --profile headless --patch "${DSH_HOME:-$HOME/.dsh}/ironmlx.patch.yml" --json \
  "Reply with exactly IRONMLX_OK"

probe_file="$(mktemp)"
printf 'IRONMLX_DSH_TOOL_OK\n' > "$probe_file"
dsh --profile headless --patch "${DSH_HOME:-$HOME/.dsh}/ironmlx.patch.yml" --json \
  "Use the read tool to read $probe_file, then reply with its complete contents."
```

The first run should emit a completed turn with `IRONMLX_OK`. The second should include `tool_call` and `tool_result` events and return `IRONMLX_DSH_TOOL_OK`; text merely describing a tool call is not enough. dsh executes local tools and sends their results back; IronMLX performs inference and emits structured tool calls. A turn that ends at the output-token limit can still lack a complete answer; raise the budget or simplify the task when that happens. If either check fails, verify the endpoint, exact model ID, capacities, and the model's reasoning support, then see [Troubleshooting](troubleshooting.md).
