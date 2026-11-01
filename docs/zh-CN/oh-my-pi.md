# oh-my-pi 集成

[English](../oh-my-pi.md)

oh-my-pi（OMP）可通过 Responses API 将 IronMLX App 用作推理服务。开始前，请先
启动 IronMLX App，并确保需要使用的模型可用。

## 使用 Dashboard 向导

在 Dashboard 的 **Agent** 页选择 oh-my-pi，点击 **一键配置**，或展开 **手动配置** 复制 Provider 设置。App 不代替客户端安装。
此示例针对支持 `openai-responses` 和 models-list discovery 的 OMP；升级后配置选项以所安装版本为准。
[前往官方配置指南](https://omp.sh/docs/custom-models)。以下保留手动配置方法。

## 一键配置

在 IronMLX App 中打开 **Dashboard → Agent → oh-my-pi**，选择已加载的对话模型并点击 **一键配置**。适配器要求本机安装 oh-my-pi `18.1.16` 或更高版本；未安装、版本过低或版本无法确认时会停止，并提供安装链接，不写入配置。App 会检查常见安装目录、继承的 PATH 和登录 Shell 的 PATH。

一键配置面向默认配置目录 `~/.omp/agent`，要求 IronMLX 处于本地模式。它使用 `models.yml`；仅当该文件不存在时才使用 `models.yaml`。旧版 `models.json` 需先由 OMP 启动完成迁移。命名 profile、CLI 参数和 Shell 环境指定的自定义配置目录请使用手动配置。

适配器合并 `ironmlx` Provider，并为所选模型设置当前上下文容量和最多 4096 token 的默认输出预算，关闭可配置思考。其他 Provider 和模型设置会保留。修改前在原文件旁备份为 `*.ironmlx-backup-<UUID>`；YAML 排版和注释可能被规范化，原文保留在备份中。重复应用相同配置不会重写文件或生成额外备份。不会改动默认模型角色或启动 OMP 会话。

配置后重新启动已打开的 OMP 会话，使用下方命令刷新模型并验证连接。手动配置信息默认折叠，验证命令始终可见。

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
omp models refresh ironmlx
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

## 使用代理时，本机请求返回 502

如果 IronMLX 正常运行，但 OMP 报 `502 status code (no body)`，可先在当前终端将本机地址加入代理排除项：

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
```

退出 OMP，再从同一终端使用目标 `--model` 选择器重新启动。如果请求恢复正常，说明代理路径影响了本机连接。这些命令保留已有排除项，只影响当前终端及其子进程，不修改 Shell 启动文件，也不关闭远程 Provider 的代理。

一键配置负责写入 `models.yml`，不改动终端的代理变量。OMP 通过 [`NO_PROXY` / `no_proxy`](https://omp.sh/docs/env) 设置代理排除项，`models.yml` 没有对应的 Provider 字段。

## 判断是否成功

文本检查应返回 `IRONMLX_OK`；工具检查应实际执行一次 `pwd` 并返回工作目录。仅收到模型描述命令的文本不算工具调用成功。
失败时先核对 endpoint、模型 ID 和模板支持，再查看[故障排查](troubleshooting.md)。
