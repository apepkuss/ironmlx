# Hermes Agent 集成

[English](../hermes-agent.md)

Hermes Agent 可通过 Responses API 将 IronMLX App 用作推理服务，无需修改 Hermes
Agent 源码。开始前，请先启动 IronMLX App 并加载需要使用的模型。

## 使用 Dashboard 向导

在 Dashboard 的 **Agent** 页选择 Hermes Agent，点击 **一键配置**，或展开 **手动配置** 复制设置。App 不代替用户安装 Hermes。
向导标注适用于 Hermes Agent v0.20.0 及以上；模型实际上下文上限需满足至少 64000 tokens。
[前往官方配置指南](https://hermes-agent.nousresearch.com/docs/zh-Hans/integrations/providers#%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E8%87%AA%E6%89%98%E7%AE%A1-llm-%E6%8F%90%E4%BE%9B%E5%95%86)。以下保留手动配置方法。

## 一键配置

在 **Dashboard → Agent → Hermes Agent** 中选择已加载的对话模型，点击 **一键配置**。App 首先检测 Hermes CLI，或内含本地 Agent 运行时的官方 Desktop，要求版本 `0.20.0` 或以上。未安装、版本过低或无法确认时，在写入任何配置之前停止，并显示安装链接。仅有源码仓库或仅连接远程服务的 Desktop 不算本地安装；引导安装型 Desktop 需先完成 CLI/运行时安装。

配置要求 IronMLX 处于本地模式，且模型实际上下文容量至少为 64,000 token。所选模型、实际容量、本机 endpoint 和 `codex_responses` 传输方式会写入专用的 `~/.hermes/profiles/ironmlx/config.yaml`。Hermes 将配置文件识别为 Profile 身份，并在启动时创建其他运行目录。默认 Profile、当前 Profile 选择、凭据、终端工作目录和其他 Provider 会保留。自定义 `HERMES_HOME` 或数据目录后缀请使用手动配置。

修改前将已有配置备份为同目录的 `config.yaml.ironmlx-backup-<UUID>`。配置和备份仅当前用户可读写，YAML 排版和注释可能被规范化。重复应用相同配置不会重写文件或增加备份。已删除的 Profile 或 Profile 路径内的符号链接会停止自动配置，请先在 Hermes 中恢复 Profile，或使用手动配置。

配置后重新启动已打开的 Hermes 会话。在 Desktop 中选择 **ironmlx**，再创建新对话；已安装 CLI 时可使用下方验证命令。手动配置默认折叠，验证命令始终可见。仅安装 Desktop 时，可直接在 Desktop 验证，无需单独安装 CLI。

## 手动配置

一键配置和手动配置均使用专用的 `ironmlx` Profile。

已安装 CLI 时，仅在专用 profile 尚不存在时创建；若已完成一键配置，可跳过此步：

```bash
hermes profile create ironmlx
```

将以下字段合并到 `~/.hermes/profiles/ironmlx/config.yaml`，保留其他设置：

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

- `model.default` 必须与 IronMLX 中加载的模型 ID 一致。
- `model.context_length` 应填写该模型在 IronMLX 中的实际上下文上限；Hermes
  Agent v0.20.0 要求至少 64000。
- `transport` 必须为 `codex_responses`。
- 如需指定 terminal 工作目录，可将 `terminal.cwd` 设置为已存在的项目目录；上述配置保留当前工作目录。
- 本机默认配置不需要 API Key。

## 验证

```bash
curl -fsS http://127.0.0.1:9068/healthz
hermes --profile ironmlx -z "Reply with exactly IRONMLX_OK"
hermes --profile ironmlx -z -t terminal "Use the terminal tool exactly once to run pwd, then report its output."
```

- TUI：`hermes --profile ironmlx --tui`
- Desktop：在左侧边栏底部点击 **I** 按钮（悬停确认名称为 `ironmlx`），再新建会话。房子按钮用于切换到 `default`。设置页顶部的 profile 选择只改变编辑对象，不切换聊天使用的 profile。

Hermes 负责执行 terminal、MCP 等工具，并将工具结果回传给 IronMLX；IronMLX 只负责
推理和生成结构化工具调用。

## 判断是否成功

文本检查应返回 `IRONMLX_OK`；工具检查应实际执行一次 `pwd` 并返回工作目录。仅收到模型描述命令的文本不算工具调用成功。
失败时先核对 endpoint、模型 ID 和模板支持，再查看[故障排查](troubleshooting.md)。

## 高级用法：使用默认 Profile

如需明确配置默认 Profile，可将同样的字段合并到 `~/.hermes/config.yaml`，并将上述命令中的 `--profile ironmlx` 改为 `--profile default`。这会改变默认 Profile 使用的 Provider。在 Desktop 侧边栏底部选择房子按钮，再新建会话。Dashboard 配置始终使用专用 `ironmlx` Profile。
