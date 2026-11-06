# OpenCode 集成

[English](../opencode.md)

OpenCode 2.2.5 或更新版本可通过原生 Chat Completions 运行时
`@opencode/ai/providers/openai-compatible` 使用 IronMLX，请求发送到
`/v1/chat/completions`。无需修改 OpenCode，也无需启用实验性 native 标志。
本集成同时补充了默认 `subagent` 工具所需的 `^ses` 前缀正则约束支持；
请使用包含本集成改动的 IronMLX 构建。

## 一键配置

1. 安装 [OpenCode CLI 或 Desktop](https://opencode.ai/v2/docs/)，启动本地模式的 IronMLX，
   加载支持工具调用的对话模型。
2. 打开 **Dashboard → Agent → OpenCode**，选择已加载的模型，点击**一键配置**。
3. 在项目目录中新建 OpenCode 会话，通过 `/models` 选择 IronMLX，或使用下方命令指定模型。

仅安装 Desktop 也可使用一键配置：App 会检查 Desktop 自带的运行时，无需单独安装
终端命令。在 Desktop 中连接本地 OpenCode 服务，打开项目，在新会话中选择已配置的
IronMLX 模型。远程 OpenCode 服务需要在其所在机器上配置 Provider。

App 会先检查运行时版本和实时模型列表，再写入配置。配置包含独立的 `ironmlx`
Provider、本地 endpoint、所选模型的实际上下文容量，以及最高 4096 token 的输出预算。
仅当已加载模型支持视觉时，才声明图像输入能力。默认关闭思考；当模型模板明确声明
原生思考强度时，一键配置会生成模型档位和 `reasoning_content` 兼容映射。
在 Desktop 选择 IronMLX 模型后，使用模型选择器旁的档位菜单切换：

| 档位 | 请求参数 |
| --- | --- |
| Default / None | `chat_template_kwargs.enable_thinking=false`，不发送 `reasoning_effort` |
| Low | `reasoning_effort=low`、`enable_thinking=true`、`preserve_thinking=true` |
| Medium | `reasoning_effort=medium`、`enable_thinking=true`、`preserve_thinking=true` |
| Xhigh | `reasoning_effort=xhigh`、`enable_thinking=true`、`preserve_thinking=true` |

这些是 Qwen3.8 模板的原生强度，不代表固定 token 预算；Qwen3.8 不支持 `high`。
未知模板不会生成推测档位。思考和最终答案共用 4096 token 输出上限。
Chat Completions 通过 `delta.reasoning_content` 流式输出思考，最终正文使用
`delta.content`；后续 assistant 消息通过 `reasoning_content` 回放历史思考。
再次一键配置会恢复这些受管理的档位，默认仍关闭思考。

DFlash2 独占服务模式下，一键配置通过实时健康状态核对当前目标模型，并从
`/v1/models` 读取容量，不要求 daemon 专用管理接口。此路径声明文本输入能力。

配置保留默认模型、其他 Provider、其他已注册模型，以及无关的 JSONC 文本和注释。
适配器管理 `ironmlx` Provider 的传输设置和当前所选模型定义；切换到原生 Chat 时，
会清除该 Provider 旧的请求头、body 覆盖、目录继承和所选模型覆盖项。

默认配置目录为 `~/.config/opencode`。已有的 `opencode.jsonc` 优先于
`opencode.json`；否则使用已有 JSON 文件，或新建 JSONC 文件。两份文件都包含
`ironmlx` Provider 时，会更新两处受管理的配置，避免低优先级的旧传输设置残留。
现有文件在修改前会备份到同目录，名称为 `*.ironmlx-backup-<UUID>`。
重复应用相同设置不会再次写入或创建备份。

App 支持其环境中可见、采用绝对路径的 `XDG_CONFIG_HOME` 和 `OPENCODE_CONFIG_DIR`。
如使用显式 `OPENCODE_CONFIG`、内联 `OPENCODE_CONFIG_CONTENT` 或项目覆盖配置，
请将手动配置合并到对应位置。旧格式的 `provider.ironmlx` 需先转换为下方 V2
`providers` 格式。一键配置不安装或启动 OpenCode，不修改终端环境变量，也不覆盖
用户的 Provider 权限策略。

## 手动配置

将以下 Provider 合并到 OpenCode 实际使用的配置中。替换 `your-model-id`、endpoint、
上下文容量和输出预算；示例中的 32768 和 4096 仅用于说明。Dashboard 获取实际容量后，
会生成可复制的配置。

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

此适配器使用显式模型列表，不会自动发现 IronMLX 模型。需要其他模型时，在
Dashboard 中选择对应的已加载模型，再次配置。另见
[OpenCode Provider 指南](https://opencode.ai/v2/docs/providers/)。

## 启动与验证

替换项目目录和模型 ID：

```bash
cd /absolute/path/to/project
opencode models
opencode run --model 'ironmlx/your-model-id' 'Reply with exactly IRONMLX_OK'
opencode run --model 'ironmlx/your-model-id' \
  'Use the shell tool exactly once to run pwd, then report its output.'
opencode .
```

文本检查应返回 `IRONMLX_OK`。工具检查应真正执行一次 `shell` 并返回结果，
模型只描述命令不算验证通过。OpenCode 在本地执行工具并应用权限设置；IronMLX
提供推理和结构化工具调用。TUI 中可通过 `/models` 选择已注册的 IronMLX 模型，
原来的默认模型保持不变。

如果 Provider 未出现，请检查实际配置目录、项目覆盖和 Provider 权限策略。
如果请求超过上下文容量，请缩减对话或输出预算，或调整模型实际容量：OpenCode
2.2.5 尚未将 IronMLX 的容量错误识别为自动恢复所需的 context-overflow。
JSON Schema、模型模板和输入类型限制仍以
[Chat API](text-vision-api.md#openai-chat-completions) 为准。

## 已记录的验证

2026-11-06，分支构建的真实 macOS App 已通过连续验收：在 Dashboard 点击一键配置，
正式版 OpenCode 2.2.5 CLI 原样读取生成的文件，直接连接该 App 管理的本地
Qwen3.5-4B 4bit 模型。文本检查返回 `IRONMLX_OK`；一次成功的 `read` 调用读取
文件中的随机标记，标记随后出现在最终回答中。后台请求计数为 `0 → 1 → 3`。
本轮连续验收没有使用模型列表 fixture 或转发代理，隔离配置使用模型实际的
262144-token 上下文容量和 4096-token 输出上限。

本轮验证覆盖本地 ad-hoc App 的 Dashboard 到 CLI 功能链路；发布包、交互式 TUI、
长上下文压缩、开启思考和视觉请求不在此次验收范围。此前的 fixture 和传输捕获
检查仍保留。版本身份、检查结果、已复现问题和验证边界见
[验收证据](../validation/opencode-chat-acceptance.json)。

2026-11-07 的单独 Desktop 验收补齐了此前 CLI 验收遗漏的 Desktop 运行时识别和
DFlash2 独占模型识别。仅安装 OpenCode Desktop 2.2.5 时，在真实 Dashboard
点击一键配置，为 Qwen3.8-27B 4bit + DFlash2 生成正常的全局配置。未手动修改
该文件，Desktop 即显示并选中了 IronMLX，文本请求返回 `IRONMLX_DESKTOP_OK`。
导出的会话记录确认 Provider 为 `ironmlx`，模型与配置一致。此项覆盖 Desktop
配置、模型选择和文本推理；Desktop 工具执行及 Web 界面交互仍未纳入验收。


2026-11-07，安装的 OpenCode Desktop 2.2.5 已连接真实 App 管理的
Qwen3.8-27B 4bit + DFlash2 完成原生思考验收。真实 Dashboard 按钮生成全局
Provider 和档位，未手动修改 Provider，也未改用转发代理。7 次用户请求
（三个原生档位各两轮，加 None 对照）及 4 次自动标题请求均以 HTTP 200、
`finish_reason=stop` 和 `[DONE]` 正常结束。

| 档位 | 第一轮 / 第二轮思考增量数 | 历史回放 |
| --- | --- | --- |
| Low | 101 / 60 | `reasoning_content` 逐字一致 |
| Medium | 293 / 49 | `reasoning_content` 逐字一致 |
| Xhigh | 1141 / 489 | `reasoning_content` 逐字一致 |
| None | 0 | 关闭思考，不发送 `reasoning_effort` |

各档位的实际线上参数均与生成的配置一致。三个第二轮请求完整回放第一轮思考，
Desktop 导出会话中的思考和最终答案也与流内容一致。Desktop 显示独立 Thought；
已展开 Medium 的思考文本，并观察到 Xhigh 生成中的 `Thinking 2s`，随后显示正文。
临时 HTTP 观察钩子仅克隆请求和响应，未替换它们，且只匹配合成测试标记；
验收后已移除并确认卸载。上表增量数来自不同测试题，不代表性能或强度标定比较。

首次验收发现后端缺口：Chat Completions 丢弃了原生输出解码配置，导致思考及
`</think>` 混入正文。现已将该配置传入文本和工具解码器，在流式及非流式响应中
分别表示思考内容。最初 3 次模型请求保留为失败诊断证据，与修复后通过的 11 次
分开记录。回归检查通过：11 项针对性 Swift 测试、55 项全 features Chat
Completions 测试、格式检查、workspace Clippy 和 Release 构建。此次补齐思考
覆盖，未新增真实 Desktop 工具调用、视觉、长上下文、发布包、TUI 或 Web 界面验收。
