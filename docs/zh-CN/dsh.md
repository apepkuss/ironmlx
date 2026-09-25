# DeepSeek Harness（dsh）集成

[English](../dsh.md)

DeepSeek Harness 可通过 Responses API 将 IronMLX 用作推理服务。本指南已使用 dsh `master` 的 `ddefc45fbc` 与本地 Qwen3.8-27B-4bit + DFlash2 组合验证；其他 dsh 版本和模型的行为可能不同。请先启动 IronMLX 并加载目标模型；dsh 需按其项目说明单独安装。

## 使用 Dashboard 向导

在 IronMLX Dashboard 的 **Agent** 页选择 **DeepSeek Harness**，再选择 **DSH CLI** 或 **DSH Desktop**。两种路径都会使用当前本机 endpoint：CLI 路径生成 overlay，Desktop 路径提供创建自定义提供方所需的准确字段和 GUI 步骤。

### DSH CLI

CLI 向导根据当前已加载模型生成 dsh overlay，并在可用时从 `/v1/models` 读取 `context_window`、`max_output_tokens`。复制前请检查单轮输出预算：向导初始值不超过 4096 tokens，不会直接采用服务端公布的完整输出能力。若容量信息不可用，请手动填写模型实际的上限。

将 overlay 保存为 `$DSH_HOME/ironmlx.patch.yml`（通常为 `~/.dsh/ironmlx.patch.yml`）。仅在需要使用 IronMLX 的运行中传入 `--patch`；不要覆盖现有的 home 级 `cordis.patch.yml`。
如果 `$DSH_HOME/settings.yaml` 已指定默认模型或包含 `llm-pi-ai` 覆盖项，请在该文件中更新或移除冲突的配置：dsh 会在 overlay 的基础配置之后应用用户设置。

## DSH Desktop GUI 配置

DSH Desktop 通过 GUI 保存提供方配置，不会自动加载 CLI overlay，也不需要修改 `DSH_HOME`。

1. 启动 IronMLX，并加载准备供 Agent 使用的对话 LLM。
2. 打开 **DSH Desktop → 设置 → 模型 → 添加自定义提供方**。
3. 在“自定义提供方”窗口中填写：

   | 配置项 | 填写值 |
   | --- | --- |
   | Provider ID | `ironmlx-local` |
   | 显示名称 | `IronMLX` |
   | API 地址 | `http://127.0.0.1:9068/v1` |
   | API 协议 | `openai-responses` |
   | API 密钥 | `local` |

4. 点击 **获取可用模型**。
5. 点击 **取消全选**，然后只选择具备 Agent 能力的对话 LLM。IronMLX 的模型目录还可能包含 TTS、图像生成、视觉和 decision model；不要将这些模型添加为 DSH 对话模型。
6. 点击 **添加所选** 并展开模型设置。保留自动发现的 **上下文窗口**，并检查 **最大输出 token 数**：如果模型参数页配置了 `default_max_output_tokens`，IronMLX 会公布该值；否则公布保守默认值 `4096`（小上下文窗口会进一步降低）。任务确实需要更长回答且上下文空间充足时，可以改为 `8192` 或 `16384`。reasoning token 与可见正文共同消耗该预算，数值越大也可能带来越高的延迟和内存压力。
7. 点击 **创建提供方**。
8. 新建会话，选择 **IronMLX / 你的模型**。

`local` 仅是本机回环免鉴权接口的占位值。使用需要鉴权的 LAN 接口时，应改填实际的 IronMLX LAN 地址和 API Key。若无法获取模型，请确认 IronMLX 正在运行、API 地址包含 `/v1`，并且 API 协议为 `openai-responses`。

验证 Desktop 路径时，新建会话并选择 IronMLX 模型，要求模型只回复 `IRONMLX_OK`。正常返回即可确认提供方发现和 Responses 请求可用。工具仍由 DSH Desktop 在本机执行；IronMLX 只负责推理和返回结构化工具调用。

## 手动配置 DSH CLI

以下是示例，请将 `MODEL_ID` 和两个容量值换成 IronMLX 实际提供的模型信息。`maxTokens` 既是 dsh 声明的模型输出上限，也会成为默认单轮请求预算；reasoning 与可见正文共同消耗它。如果模型没有已知的独立输出上限，应将服务端公布的 `max_output_tokens` 视为保守起始值，并确保调整后的值不超过可用上下文空间。非推理模型应在向导中选“关闭”，或手动将 `reasoningEfforts` 设为 `false` 并删除 `reasoning: high`。

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

`IRONMLX_API_KEY=local` 只是满足 dsh 对本地免鉴权 endpoint 的凭据查询的占位值，不是 IronMLX 的 LAN API Key。本指南仅针对本机回环接口，不覆盖需要鉴权的 LAN 访问。

## 验证 DSH CLI

保存 overlay 后，在项目目录执行以下命令。如果自定义了 `DSH_HOME`，命令会自动使用该目录。

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

第一轮应以 completed 结束并返回 `IRONMLX_OK`。第二轮应出现 `tool_call`、`tool_result` 事件，最终返回 `IRONMLX_DSH_TOOL_OK`；模型只是描述工具调用不算通过。dsh 在本机执行工具并回传结果，IronMLX 负责推理和生成结构化工具调用。若单轮输出 token 用尽，仍可能没有完整答案；此时应增加预算或简化任务。失败时先核对 endpoint、准确的模型 ID、容量和模型推理能力，再查看[故障排查](troubleshooting.md)。
