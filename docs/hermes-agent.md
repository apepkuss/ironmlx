# Hermes Agent integration

[简体中文](zh-CN/hermes-agent.md)

Hermes Agent can use the IronMLX App as an inference provider through the
Responses API without changing Hermes source. Start IronMLX and load the model
you intend to use first.

## Use the Dashboard guide

Select Hermes Agent on Dashboard **Agent** and use **One-click setup**, or expand **Manual configuration** to copy the settings. The App does not install Hermes.
The guide targets Hermes Agent v0.20.0 and later, with an actual model context limit of at least 64000 tokens.
See the [official configuration guide](https://hermes-agent.nousresearch.com/docs/integrations/providers#custom--self-hosted-llm-providers). Manual steps follow.

## One-click setup

Select a loaded conversation model in **Dashboard → Agent → Hermes Agent** and click **One-click setup**. The App checks for Hermes CLI or an official Desktop installation containing a local agent runtime, version `0.20.0` or later. Missing, outdated, or unverifiable installations stop before any configuration is written and show an installation link. A source checkout alone or a remote-only Desktop client does not count as a local installation. A bootstrap Desktop shell must finish installing its CLI/runtime first.

Setup requires IronMLX local mode and at least 64,000 tokens of actual model context capacity. It writes the selected model, actual capacity, local endpoint, and `codex_responses` transport to the dedicated `~/.hermes/profiles/ironmlx/config.yaml`. Hermes recognizes the configuration file as a profile identity and creates other runtime directories when launched. The default profile, active-profile selection, credentials, terminal workspace, and other providers are retained. Custom `HERMES_HOME` directories or data-directory suffixes use manual configuration.

Existing configuration is backed up alongside the original as `config.yaml.ironmlx-backup-<UUID>` before modification. Configuration and backups use owner-only permissions; YAML formatting/comments may be normalized. Repeating unchanged setup does not rewrite the file or add backups. Deleted profiles and symbolic links within the profiles path stop automatic configuration; restore the profile in Hermes or configure manually.

Restart open Hermes sessions after setup. In Desktop, click the **I** profile button at the bottom of the left sidebar (hover to confirm **ironmlx**), then create a new chat. The house button switches to **default**. Selecting a profile in Settings only chooses which configuration to edit; it does not switch the chat profile. With CLI installed, use the verification commands below. Manual configuration is collapsed by default and verification remains visible. A Desktop-only installation can be verified directly in Desktop without installing a separate CLI.

## Manual configuration

One-click setup and manual configuration both use the dedicated `ironmlx` profile.

With CLI installed, create the dedicated profile only if it does not already exist. Skip this step after one-click setup:

```bash
hermes profile create ironmlx
```

Merge these fields into `~/.hermes/profiles/ironmlx/config.yaml`, retaining other settings:

```yaml
model:
  default: "mlx-community/Qwen3.5-2B-4bit"
  provider: "custom:ironmlx-responses"
  context_length: 64000
  base_url: "http://127.0.0.1:9068/v1"
  api_mode: "codex_responses"

providers:
  ironmlx-responses:
    api: "http://127.0.0.1:9068/v1"
    base_url: "http://127.0.0.1:9068/v1"
    transport: "codex_responses"
    api_mode: "codex_responses"
    discover_models: false
    enabled: true
```

The model ID must match the model loaded in IronMLX. Set the actual context
limit; Hermes Agent v0.20.0 requires at least 64000. Use `codex_responses` and
optionally set `terminal.cwd` to an existing project directory when choosing a terminal workspace. The configuration above retains the current workspace. Local default mode needs
no API key.

## Verify

```bash
curl -fsS http://127.0.0.1:9068/healthz
hermes --profile ironmlx -z "Reply with exactly IRONMLX_OK"
hermes --profile ironmlx -z -t terminal "Use the terminal tool exactly once to run pwd, then report its output."
```

Hermes executes terminal and MCP tools and returns their results to IronMLX;
IronMLX only performs inference and emits structured tool calls.

Use `hermes --profile ironmlx --tui` for TUI or select the ironmlx profile in Desktop before creating a session.

## Confirm the result

The text check should return `IRONMLX_OK`. The tool check should execute `pwd` once and return the working directory; a model merely describing the command is not a successful tool call.
If it fails, check the endpoint, model ID and template support, then see [Troubleshooting](troubleshooting.md).

## Advanced: use the default profile

To configure the default profile explicitly, merge the same fields into `~/.hermes/config.yaml` and replace `--profile ironmlx` with `--profile default` in the commands above. This changes the provider used by the default profile. In Desktop, select the house button at the bottom of the sidebar and start a new chat. Dashboard setup always uses the dedicated `ironmlx` profile.
