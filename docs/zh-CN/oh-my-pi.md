# oh-my-pi 集成

[English](../oh-my-pi.md)

oh-my-pi（OMP）可通过 Responses API 将 IronMLX App 用作推理服务。开始前，请先
启动 IronMLX App，并确保需要使用的模型可用。

## 使用 Dashboard 向导

在 Dashboard 的 **Agent** 页选择 oh-my-pi，按当前 endpoint 和模型生成配置，再复制到客户端指定文件。向导不代替客户端安装。
此示例针对支持 `openai-responses` 和 models-list discovery 的 OMP；升级后配置选项以所安装版本为准。
[前往官方配置指南](https://omp.sh/docs/custom-models)。以下保留手动配置方法。

## 手动配置

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

## 判断是否成功

文本检查应返回 `IRONMLX_OK`；工具检查应实际执行一次 `pwd` 并返回工作目录。仅收到模型描述命令的文本不算工具调用成功。
失败时先核对 endpoint、模型 ID 和模板支持，再查看[故障排查](troubleshooting.md)。
