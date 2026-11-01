# oh-my-pi integration

[简体中文](zh-CN/oh-my-pi.md)

oh-my-pi (OMP) can use the IronMLX App as an inference provider through the
Responses API. Start IronMLX and make sure the desired model is available.

## Use the Dashboard guide

Select oh-my-pi on Dashboard **Agent** and use **One-click setup**, or expand **Manual configuration** to copy the provider settings. The App does not install the client.
This example targets OMP with `openai-responses` and models-list discovery support. Check installed-version options after upgrading.
See the [official configuration guide](https://omp.sh/docs/custom-models). Manual steps follow.

## One-click setup

In the IronMLX App, open **Dashboard → Agent → oh-my-pi**, select a loaded conversation model, and click **One-click setup**. The adapter requires installed oh-my-pi `18.1.16` or later. Missing, outdated, or unverifiable installations stop without writing configuration and show an installation link. Discovery checks inherited PATH, common installation directories, and login-shell PATH.

Setup targets the default `~/.omp/agent` directory and requires IronMLX local mode. It uses `models.yml`, falling back to `models.yaml` only when the former is absent. Launch OMP to migrate legacy `models.json` first. Use manual configuration for named profiles, CLI overrides, or custom directories selected by shell environment variables.

The adapter merges the `ironmlx` provider and sets the selected model's current context capacity and a default output budget of at most 4096 tokens, with configurable reasoning disabled. Other providers and model settings are retained. Existing files are backed up alongside the originals as `*.ironmlx-backup-<UUID>` before modification. YAML formatting and comments may be normalized; the original text is retained in backups. Reapplying unchanged configuration does not rewrite files or create extra backups. Setup does not change default model roles or launch OMP sessions.

Restart open OMP sessions after setup, then use the commands below to refresh models and verify the connection. Manual configuration is collapsed by default; verification commands remain visible.

## Manual configuration

Edit `~/.omp/agent/models.yml`:

```yaml
providers:
  ironmlx:
    baseUrl: "http://127.0.0.1:9068/v1"
    auth: none
    api: openai-responses
    discovery:
      type: openai-models-list
```

Refresh and inspect available models:

```bash
omp models refresh ironmlx
omp models ironmlx
```

## Start and verify

Replace the model selector and project directory with your values:

```bash
omp --cwd /absolute/path/to/project \
  --model ironmlx/mlx-community/Qwen3.5-2B-4bit

omp --cwd /absolute/path/to/project \
  --model ironmlx/mlx-community/Qwen3.5-2B-4bit \
  -p "Reply with exactly IRONMLX_OK"

omp --cwd /absolute/path/to/project \
  --model ironmlx/mlx-community/Qwen3.5-2B-4bit \
  -p --auto-approve "Use the bash tool exactly once to run pwd, then report its output."
```

OMP executes bash and other client-side tools in `--cwd` and returns results to
IronMLX. IronMLX only performs inference and emits structured tool calls.

## A local request returns 502 when using a proxy

If OMP reports `502 status code (no body)` while IronMLX is running, retry after adding loopback addresses to the current terminal's proxy exclusions:

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
```

Exit OMP and start it again from the same terminal with the intended `--model` selector. If the request now succeeds, the proxy path was affecting the local connection. These commands preserve existing exclusions and apply only to the current terminal and its child processes. They do not change shell startup files or disable proxies for remote providers.

One-click setup configures `models.yml`; it does not change terminal proxy variables. OMP exposes proxy exclusions through [`NO_PROXY` / `no_proxy`](https://omp.sh/docs/env), rather than a per-provider field in `models.yml`.

## Confirm the result

The text check should return `IRONMLX_OK`. The tool check should execute `pwd` once and return the working directory; a model merely describing the command is not a successful tool call.
If it fails, check the endpoint, model ID and template support, then see [Troubleshooting](troubleshooting.md).
