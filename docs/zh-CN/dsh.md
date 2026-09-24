# DeepSeek Harness（dsh）集成

[English](../dsh.md)

DeepSeek Harness 可通过 Responses API 将 IronMLX 用作推理服务。本指南已使用 dsh `master` 的 `ddefc45fbc` 与本地 Qwen3.8-27B-4bit + DFlash2 组合验证；其他 dsh 版本和模型的行为可能不同。请先启动 IronMLX 并加载目标模型；dsh 需按其项目说明单独安装。

## 使用 Dashboard 向导

在 IronMLX Dashboard 的 **Agent** 页选择 **DeepSeek Harness**。向导根据当前本地 endpoint 和已加载模型生成 dsh overlay，并在可用时从 `/v1/models` 读取 `context_window`、`max_output_tokens`。复制前请检查单轮输出预算：向导初始值不超过 4096 tokens，不会直接采用服务端公布的完整输出能力。若容量信息不可用，请手动填写模型实际的上限。

将 overlay 保存为 `$DSH_HOME/ironmlx.patch.yml`（通常为 `~/.dsh/ironmlx.patch.yml`）。仅在需要使用 IronMLX 的运行中传入 `--patch`；不要覆盖现有的 home 级 `cordis.patch.yml`。
如果 `$DSH_HOME/settings.yaml` 已指定默认模型或包含 `llm-pi-ai` 覆盖项，请在该文件中更新或移除冲突的配置：dsh 会在 overlay 的基础配置之后应用用户设置。

## 手动配置

以下是示例，请将 `MODEL_ID` 和两个容量值换成 IronMLX 实际提供的模型信息。`maxTokens` 既是 dsh 声明的模型输出上限，也会成为默认单轮请求预算；reasoning 与可见正文共同消耗它。该值不得超过服务端公布的 `max_output_tokens`，还要受可用上下文空间约束。非推理模型应在向导中选“关闭”，或手动将 `reasoningEfforts` 设为 `false` 并删除 `reasoning: high`。

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

## 验证

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
