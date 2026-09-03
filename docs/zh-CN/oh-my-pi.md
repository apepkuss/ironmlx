# oh-my-pi 集成

oh-my-pi（OMP）可通过 Responses API 将 IronMLX App 用作推理服务。开始前，请先
启动 IronMLX App，并确保需要使用的模型可用。

## 配置

编辑 `~/.omp/agent/models.yml`：

```yaml
providers:
  ironmlx:
    baseUrl: "http://127.0.0.1:9068/v1"
    auth: none
    api: openai-responses
    discovery:
      type: openai-models-list
```

刷新并查看 IronMLX 模型：

```bash
omp models refresh
omp models ironmlx
```

## 启动与验证

将模型选择器和项目目录替换为实际值：

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

OMP 在 `--cwd` 指定的目录执行 bash 等客户端工具，并将结果回传给 IronMLX；
IronMLX 只负责推理和生成结构化工具调用。
