# oh-my-pi integration

[简体中文](zh-CN/oh-my-pi.md)

oh-my-pi (OMP) can use the IronMLX App as an inference provider through the
Responses API. Start IronMLX and make sure the desired model is available.

## Use the Dashboard guide

Select oh-my-pi on Dashboard **Agent**, generate configuration from the current endpoint/model, then copy it into the client configuration file. The guide does not install the client.
This example targets OMP with `openai-responses` and models-list discovery support. Check installed-version options after upgrading.
See the [official configuration guide](https://omp.sh/docs/custom-models). Manual steps follow.

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
omp models refresh
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

## Confirm the result

The text check should return `IRONMLX_OK`. The tool check should execute `pwd` once and return the working directory; a model merely describing the command is not a successful tool call.
If it fails, check the endpoint, model ID and template support, then see [Troubleshooting](troubleshooting.md).
