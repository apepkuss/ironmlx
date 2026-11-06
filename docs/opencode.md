# OpenCode integration

[简体中文](zh-CN/opencode.md)

OpenCode 2.2.5 or later can use IronMLX through its native Chat Completions
runtime, `@opencode/ai/providers/openai-compatible`. Requests go to
`/v1/chat/completions`. No OpenCode patch or experimental native-runtime flag
is needed.
This integration also adds prefix-pattern constraints required by OpenCode’s
default `subagent` tool (`^ses`); use an IronMLX build containing this change.

## One-click setup

1. Install [OpenCode CLI or Desktop](https://opencode.ai/v2/docs/), start IronMLX in local
   mode, and load a conversation model that supports tool calling.
2. Open **Dashboard → Agent → OpenCode**, select the loaded model, and click
   **One-click setup**.
3. Start a new OpenCode session in your project and choose IronMLX with
   `/models`, or use the explicit model selector in the commands below.

Desktop-only installations are supported: the App checks Desktop's bundled
runtime without requiring a separate shell command. In Desktop, connect to the
local OpenCode server, open your project and select the configured IronMLX model
in a new session. A remote OpenCode server needs its own provider configuration.

The App checks the runtime version and live model inventory before writing. It
registers the `ironmlx` provider with the local endpoint, the selected model's
actual context capacity, and an output budget capped at 4096 tokens. Vision
input is advertised only when the loaded model supports it. Thinking defaults
to off. For templates detected as supporting native reasoning efforts, setup
adds model variants and the `reasoning_content` compatibility mapping. In Desktop,
select the IronMLX model, then choose its variant beside the model selector:

| Variant | Request parameters |
| --- | --- |
| Default / None | `chat_template_kwargs.enable_thinking=false`; no `reasoning_effort` |
| Low | `reasoning_effort=low`, `enable_thinking=true`, `preserve_thinking=true` |
| Medium | `reasoning_effort=medium`, `enable_thinking=true`, `preserve_thinking=true` |
| Xhigh | `reasoning_effort=xhigh`, `enable_thinking=true`, `preserve_thinking=true` |

These are Qwen3.8's native template efforts, not fixed token budgets. `high` is
not a supported Qwen3.8 effort. Unknown templates receive no inferred effort
controls. Thinking and the final answer share the 4096-token output limit.
Chat Completions streams thoughts in `delta.reasoning_content` and keeps final
text in `delta.content`; subsequent assistant messages replay `reasoning_content`.
Reapplying setup restores these managed variants and keeps the default off.

For exclusive DFlash2 serving, setup verifies the configured target against live
health and reads capacity from `/v1/models`; daemon-only admin routes are not
required. This path advertises text input.

Setup keeps the default model, other providers, other registered model
definitions, and unrelated JSONC text and comments. The `ironmlx` provider's
transport settings and selected model definition are managed by this adapter:
obsolete provider headers, body overrides, catalog inheritance, and selected
model overrides are cleared when configuring native Chat.

The default directory is `~/.config/opencode`. Existing `opencode.jsonc` takes
precedence over `opencode.json`; otherwise the existing JSON file is used, or
a new JSONC file is created. If both files contain an `ironmlx` provider, both
managed entries are updated so lower-priority transport settings cannot leak
into the selected configuration. Existing files are backed up beside their
originals as `*.ironmlx-backup-<UUID>`. Reapplying identical settings creates
no additional backups or writes.

The App honors absolute `XDG_CONFIG_HOME` and `OPENCODE_CONFIG_DIR` values
visible in its environment. For explicit `OPENCODE_CONFIG`, inline
`OPENCODE_CONFIG_CONTENT`, or project overrides, use manual configuration in
the corresponding location. A legacy `provider.ironmlx` entry must first be
converted to the V2 `providers` format below. Setup does not install or launch
OpenCode, change terminal environment variables, or override provider policies.

## Manual configuration

Merge this provider into the configuration file OpenCode uses. Replace
`your-model-id`, endpoint, context capacity, and output budget with the actual
values from IronMLX; 32768 and 4096 below are illustrative. The Dashboard
generates a copyable configuration after live capacities become available.

```json
{
  "providers": {
    "ironmlx": {
      "name": "IronMLX",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:9068/v1",
        "apiKey": "",
        "provider": "ironmlx"
      },
      "body": { "chat_template_kwargs": { "enable_thinking": false } },
      "models": {
        "your-model-id": {
          "name": "your-model-id",
          "capabilities": {
            "tools": true,
            "input": ["text"],
            "output": ["text"]
          },
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

This adapter uses an explicit model list; it does not discover IronMLX models
automatically. Configure additional loaded models from the Dashboard as needed.
See the [OpenCode provider guide](https://opencode.ai/v2/docs/providers/).

## Start and verify

Replace the directory and model ID:

```bash
cd /absolute/path/to/project
opencode models
opencode run --model 'ironmlx/your-model-id' 'Reply with exactly IRONMLX_OK'
opencode run --model 'ironmlx/your-model-id' \
  'Use the shell tool exactly once to run pwd, then report its output.'
opencode .
```

The text check should return `IRONMLX_OK`. The tool check must execute `shell`
and return its result; describing a command does not verify tool calling.
OpenCode executes tools and applies permissions locally. IronMLX supplies
inference and structured tool calls. In the TUI, use `/models` to select the
registered IronMLX model; setup leaves your previous default unchanged.

If the provider is absent, check the effective config directory, project
overrides, and provider policies. If a request exceeds context capacity,
reduce the conversation/output budget or adjust the model's actual capacity:
OpenCode 2.2.5 does not classify IronMLX's capacity error as a context-overflow
signal for automatic recovery. JSON Schema, model-template, and modality
limits remain those of the [Chat API](text-vision-api.md#openai-chat-completions).

## Recorded verification

On 2026-11-06, continuous acceptance passed with the branch's real macOS App:
Dashboard's one-click button wrote the configuration, and the official OpenCode
2.2.5 CLI used that unchanged file to connect directly to the App-managed local
Qwen3.5-4B 4bit model. The text check returned `IRONMLX_OK`; one successful `read`
call returned a randomly generated file marker that appeared in the final answer.
The backend request counter increased from 0 to 1 to 3. No inventory fixture or
forwarding proxy was used in this continuous check. The isolated test profile
advertised the live model's 262144-token context and 4096-token output limit.

This verifies the Dashboard-to-CLI functional path using a local ad-hoc App build;
it does not qualify release packaging, the interactive TUI, long-context
compaction, reasoning-on, or vision. Earlier fixture and captured-transport
checks are preserved alongside identities, results, and coverage boundaries in the
[acceptance evidence](validation/opencode-chat-acceptance.json).

On 2026-11-07, a separate Desktop check closed two gaps in that earlier CLI-only
acceptance: Desktop runtime discovery and exclusive DFlash2 model discovery.
With only OpenCode Desktop 2.2.5 installed, the real Dashboard button generated
the normal global configuration for Qwen3.8-27B 4bit with DFlash2. Without editing
that file, Desktop displayed and selected IronMLX and returned
`IRONMLX_DESKTOP_OK`. The exported session identifies provider `ironmlx` and the
expected model. This checks Desktop configuration, selection and text inference;
Desktop tool execution and Web UI interaction remain outside the recorded scope.


On 2026-11-07, native reasoning acceptance passed in the installed OpenCode
Desktop 2.2.5 against the real App-managed Qwen3.8-27B 4bit + DFlash2 model.
The real Dashboard button generated the global provider and variants; no manual
provider edits or endpoint proxy were used. Seven user requests (two per native
effort plus None) and four automatic title requests completed with HTTP 200,
`finish_reason=stop`, and `[DONE]`.

| Variant | First / second turn thought deltas | History replay |
| --- | --- | --- |
| Low | 101 / 60 | Exact `reasoning_content` match |
| Medium | 293 / 49 | Exact `reasoning_content` match |
| Xhigh | 1141 / 489 | Exact `reasoning_content` match |
| None | 0 | Thinking disabled, no `reasoning_effort` |

Every effort's actual wire parameters matched the generated variant. The three
second-turn requests replayed the complete first-turn thoughts exactly, and the
exported Desktop sessions matched both streamed thoughts and final answers.
Desktop displayed separate Thought entries; the Medium thought was expanded and
Xhigh showed `Thinking 2s` while generation was still active, then its final text.
The observational HTTP hooks cloned requests/responses without replacing them,
filtered only the synthetic test marker, and were removed and observed unloaded.
Delta counts above describe these different test prompts, not comparative speed
or effort calibration.

The first attempt exposed a backend gap: Chat Completions discarded the native
output decoder configuration, leaking thoughts and `</think>` into content.
The fix now passes that configuration to text and tool decoders and represents
reasoning separately in streaming and unary responses. The initial three model
requests remain recorded as failed diagnostic evidence, separate from the eleven
passing requests. Regression checks passed: 11 targeted Swift tests, 55 Chat
Completions tests with all features, formatting, workspace Clippy and release
build. This extends reasoning coverage; it does not add live Desktop tool-call,
vision, long-context, release-package, TUI or Web UI qualification.
