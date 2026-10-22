# DeepSeek Harness (dsh) integration

[简体中文](zh-CN/dsh.md)

DeepSeek Harness can use IronMLX as an inference provider through the Responses API. This guide was verified with dsh `master` at `ddefc45fbc` and a local Qwen3.8-27B-4bit + DFlash2 service. Other dsh revisions and models may behave differently. Start IronMLX and load the intended model first; install dsh separately following its own project instructions.

Choose the configuration path for the DSH client you use:

- [DSH Desktop App](#dsh-desktop-app-configuration): create a provider and select models through the GUI.
- [DSH CLI](#dsh-cli-configuration): generate or write an overlay, then launch with `--patch`.

## DSH Desktop App configuration

DSH Desktop here means the community application from [dshdesktop.com](https://dshdesktop.com/en/) (macOS bundle ID `io.dsh.desktop`), rather than DeepSeek's official desktop app.

Community **DSH Desktop** has its own Dashboard card. **DeepSeek Harness** retains the CLI guide; its official desktop entry is marked **Not supported yet** and has no configuration action.

### One-click setup

In the IronMLX App, open **Dashboard → Agent → DSH Desktop**, select a model, and click **One-click setup**. Installation checks run before model checks. Missing community installations, versions below `0.12.1`, and unverifiable versions stop setup without creating or changing DSH configuration, and show the community installation link. Click again after installation or upgrading to repeat detection.

`0.12.1` is the adapter's initial configuration-format baseline. Quit DSH Desktop and load a conversation LLM in IronMLX local mode before applying configuration. The adapter merges the provider into `~/Library/Application Support/dsh-desktop/harness/profiles/web/cordis.patch.yml` and a local placeholder credential reference into `.credentials.yaml` in the same Harness home. If legacy `settings.yaml` needs migration, launch the community app once to finish migration first.

Existing files are backed up beside the originals as `*.ironmlx-backup-<UUID>` before modification. Other providers and settings are retained; YAML formatting and comments may be normalized, with original text retained in backups. Reapplying unchanged configuration does not create more backups. The output budget is capped at 4096 tokens and bounded by advertised model capacity.

Configuration saved is separate from connection verified: open community DSH Desktop, start a new chat, select **IronMLX / your model**, and run the verification below. Native Dashboard tests cover the uninstalled machine; configuration writes are tested in temporary fixtures. End-to-end verification with the installed community application is pending.

### Manual setup

The Dashboard keeps manual instructions collapsed by default for custom endpoints, LAN access, or troubleshooting. Connection verification is shown separately and applies to both setup methods.

DSH Desktop stores this provider through its GUI. It does not load the CLI overlay, and you do not need to change `DSH_HOME`.

### Get connection details

On the IronMLX Dashboard **Agent** page, select **DSH Desktop**. The guide provides the connection fields for the current local endpoint and the GUI setup steps.

### Create a custom provider

1. Start IronMLX and load the conversation LLM you intend to use with the agent.
2. Open **DSH Desktop → Settings → Models → Add custom provider**.
3. Fill in the custom provider form:

   | Field | Value |
   | --- | --- |
   | Provider ID | `ironmlx-local` |
   | Display name | `IronMLX` |
   | API address | `http://127.0.0.1:9068/v1` |
   | API protocol | `openai-responses` |
   | API key | `local` |

4. Select **Fetch available models**.
5. Select **Deselect all**, then choose only an Agent-capable conversation LLM. IronMLX's model catalog can also contain TTS, image-generation, vision, and decision models; do not add those as DSH chat models.
6. Select **Add selected** and expand the model settings. Keep the discovered **Context window**. Review **Max output tokens**: IronMLX publishes the model parameter page's `default_max_output_tokens` when configured, otherwise it publishes a conservative default of `4096` (or less for a small context window). You may raise it, for example to `8192` or `16384`, when the workload needs longer answers and the model has enough context. Reasoning tokens and visible output share this budget, so larger values can increase latency and memory pressure.
7. Select **Create provider**.
8. Start a new chat and select **IronMLX / your model**.

### Verify the connection

To verify the Desktop path, start a new chat with the IronMLX model and ask it to reply with exactly `IRONMLX_OK`. A normal response confirms provider discovery and a Responses request. Tool execution still happens inside DSH Desktop; IronMLX only performs inference and returns structured tool calls.

### Common configuration problems

`local` is only a placeholder for the keyless loopback endpoint. For authenticated LAN access, use the actual IronMLX LAN address and API key instead. If model discovery fails, confirm that IronMLX is running, the address includes `/v1`, and the selected protocol is `openai-responses`.

## DSH CLI configuration

Use an overlay generated by the Dashboard, or write one using the example below. After saving it, launch with `--patch` and verify your requests.

### Generate configuration with the Dashboard

On the IronMLX Dashboard **Agent** page, select **DeepSeek Harness → DSH CLI**.

The CLI guide uses the currently loaded model, reads `context_window` and `max_output_tokens` from `/v1/models` when available, and generates a dsh overlay. Review the per-turn output budget before copying: the guide initially uses at most 4096 tokens, not the service's full output capability. If capacity information is unavailable, enter the model's actual limits manually.

### Write configuration manually

The following is an example; replace `MODEL_ID` and both capacities with values for the model actually served by IronMLX. `maxTokens` is dsh's default per-request output budget as well as its configured model output limit; reasoning and visible text share it. Treat the server's reported `max_output_tokens` as a conservative starting value when no independent model limit is known, and keep the chosen value within the available context. Use the **Off** option in the Dashboard guide, or change `reasoningEfforts` to `false` and omit `reasoning: high`, for a non-reasoning model.

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

### Save the configuration

Save the overlay as `$DSH_HOME/ironmlx.patch.yml` (normally `~/.dsh/ironmlx.patch.yml`). Pass it with `--patch` only on runs that should use IronMLX; do not replace an existing home-level `cordis.patch.yml`.
If `$DSH_HOME/settings.yaml` already selects a default model or contains `llm-pi-ai` overrides, update or remove conflicting entries there: dsh applies user settings after the overlay's base configuration.

`IRONMLX_API_KEY=local` is a placeholder that satisfies dsh's credential lookup for the local keyless endpoint. It is not an IronMLX LAN API key. This guide targets the local loopback listener, not authenticated LAN access.

### Launch and verify

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

The first run should emit a completed turn with `IRONMLX_OK`. The second should include `tool_call` and `tool_result` events and return `IRONMLX_DSH_TOOL_OK`; text merely describing a tool call is not enough. dsh executes local tools and sends their results back; IronMLX performs inference and emits structured tool calls.

### Common configuration problems

A turn that ends at the output-token limit can still lack a complete answer; raise the budget or simplify the task when that happens. If either check fails, verify the endpoint, exact model ID, capacities, and the model's reasoning support, then see [Troubleshooting](troubleshooting.md).
