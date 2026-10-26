# DeepSeek Harness（dsh）集成

[English](../dsh.md)

DeepSeek Harness 可通过 Responses API 将 IronMLX 用作推理服务。本指南已使用 dsh `master` 的 `ddefc45fbc` 与本地 Qwen3.8-27B-4bit + DFlash2 组合验证；其他 dsh 版本和模型的行为可能不同。请先启动 IronMLX 并加载目标模型；dsh 可通过安装器的可选组件安装，也可按其项目说明自行安装。

根据你使用的 DSH 客户端，选择对应的配置方式：

- [官方 DeepSeek Harness 一键配置](#官方-deepseek-harness-一键配置)：分别配置官方 CLI 和 Desktop。
- [DSH Desktop App](#dsh-desktop-app-配置)：通过图形界面创建提供方并选择模型。
- [DSH CLI](#dsh-cli-配置)：生成或编写 overlay，通过 `--patch` 启动。

`IronMLX.pkg` 安装器提供默认勾选的官方 DeepSeek Harness Desktop 可选组件；使用其他 Agent 客户端时可以取消。之后自行安装 Harness，仍可使用现有一键配置。加载对话模型并在 Agent 页面配置官方桌面版后，从 IronMLX 菜单栏图标选择 **与 DSH 对话**，悬浮提示为 **打开 DeepSeek Harness**。入口会打开或唤起官方窗口；缺安装、模型、服务或配置时提供与界面语言一致的引导，不自动改写配置。

## 官方 DeepSeek Harness 一键配置

在 **Dashboard → Agent → DeepSeek Harness** 中选择应用（默认 **官方桌面版**，也可切换为 **DSH CLI**），选择已加载的对话模型，然后点击 **一键配置**。适配器依据官方仓库 `687f282e86` 的 `0.2.1-rc.2` 配置格式实现；这是最低支持版本，稳定版及后续 RC 可用，开发／测试／alpha 构建不能通过检测。官方 CLI 安装说明见 [DeepSeek 官方仓库](https://github.com/deepseek-ai/deepseek-harness)，桌面版从 [DeepSeek 官方下载页](https://www.deepseek.com/zh/download/) 安装。

CLI 需提供可直接执行的 `dsh` 命令，例如按官方 npm 包安装：`npm install -g @deepseek-ai/dsh`。仅存在 `npx` 缓存不视为已安装 CLI。

安装和版本检查先于模型、后端及配置写入。仅安装社区 DSH Desktop 时，官方 CLI 和 Desktop 都提示未安装并停止，不创建 `~/.dsh`。官方 Desktop 通过 bundle ID `com.deepseek.dsh` 识别；社区版的 `io.dsh.desktop` 不能通过。CLI 要求可找到官方 `@deepseek-ai/dsh` 包的 `dsh` 入口，或官方 Desktop 提供的命令链接；社区应用内携带的同名 runtime 不算官方 CLI。

| 应用 | 一键配置位置 | 使用方式 |
| --- | --- | --- |
| 官方 CLI | `~/.dsh/profiles/web/cordis.patch.yml`、`~/.dsh/profiles/headless/cordis.patch.yml` | `dsh web` 或 `dsh --profile headless "Hi"` |
| 官方 Desktop | `~/.dsh/profiles/desktop/cordis.patch.yml` | 重新打开官方应用，以 IronMLX 开始新会话 |
| 社区 DSH Desktop | `~/Library/Application Support/dsh-desktop/harness/profiles/web/cordis.patch.yml` | 使用独立 DSH Desktop 卡片 |

官方两个入口共享 `~/.dsh/.credentials.yaml`，使用独立引用 `IRONMLX_HARNESS_API_KEY` 保存本机占位值 `local`。一键配置只适用于本机模式，采用当前模型公布的 context window，输出预算不超过 `4096` tokens，使用 SSE，默认不声明 reasoning 档位。它合并已有 Provider 和模型，选择 IronMLX 作为对应 profile 的新会话默认模型；其他 profile、已有会话和社区配置不变。修改前备份现有文件，文件权限设为 `0600`；不更改的文件不重写、不重复备份，写入失败时尝试恢复已修改文件。YAML 注释和格式可能规范化，原始内容保留在备份中。

仅支持默认 `~/.dsh`。检测到进程或 CLI 登录 Shell 的自定义 `DSH_HOME` 时停止；不会跟随符号链接写入社区目录。请先关闭官方 Desktop 与运行中的 `dsh`。检测到 profile 锁、旧 `settings.yaml`、home 级 `cordis.patch.yml` 对模型／Provider／凭据存储的覆盖，或不能安全合并的 YAML 时停止，改用手动配置或先处理冲突。首次启动官方应用负责初始化 profile 的 package manifest，一键配置不会代为安装插件。

手动配置默认折叠。下文 CLI 的独立 `ironmlx.patch.yml` 是按次使用的高级替代方式；一键配置不需要 `--patch`。官方 Desktop 的手动方式是 **设置 → 模型 → 自定义模型 API**，填写 `ironmlx-local`、`OpenAI Responses`、IronMLX `/v1` 地址和 `local`，获取并添加模型后在新会话选择它。

已验证仅安装社区版时的官方未安装提示与配置隔离；安装官方 Desktop `0.2.1-rc.2` 后，已核对签名发布包标识 `com.deepseek.dsh` 并通过真实安装检测。成功写入、旧版本、合并、备份、重复执行和阻止条件通过临时目录验证。官方 Desktop `0.2.1-rc.2` 已通过同一原生配置服务的真实写入验证，确认备份权限、默认模型、原有设置和社区版配置均符合预期。重新打开官方 Desktop 后，已确认默认使用 IronMLX 的 Qwen3.5-4B-MLX-4bit，并发送“Hi”收到正常回复。官方 CLI 的真实聊天验证仍待安装；配置成功提示本身不代表已完成聊天验证。

## DSH Desktop App 配置

这里的 DSH Desktop 是从 [dshdesktop.com](https://dshdesktop.com/en/) 下载的社区版（macOS bundle ID：`io.dsh.desktop`），不是 DeepSeek 官方桌面应用。

社区版 **DSH Desktop** 与官方 **DeepSeek Harness** 使用独立 Dashboard 卡片和安装检测。官方卡片可分别配置 CLI 与官方 Desktop。

### 一键配置

在 IronMLX App 内打开 **Dashboard → Agent → DSH Desktop**，选择模型后点击 **一键配置**。安装检查先于模型检查：未安装社区版、版本低于 `0.12.1` 或无法识别版本时，向导提示安装／升级并停止，不创建或修改任何 DSH 配置。安装后可再次点击，重新检测。

`0.12.1` 是此适配器采用的首个配置格式基线。检测通过后，请先退出 DSH Desktop，并在 IronMLX 本机模式中加载对话 LLM。适配器将 IronMLX provider 合并到 `~/Library/Application Support/dsh-desktop/harness/profiles/web/cordis.patch.yml`，并在同一 Harness home 的 `.credentials.yaml` 中保存本机占位凭据引用。旧版 `settings.yaml` 尚待迁移时，会提示先启动一次社区版完成迁移。

已有文件在修改前保存为同目录下的 `*.ironmlx-backup-<UUID>`；保留其他 provider 和设置，YAML 格式及注释可能被规范化，原始文本可从备份恢复。重复应用相同配置不重复生成备份。单轮输出预算最多采用 `4096` tokens，并受模型公布容量限制。

“配置已保存”只表示写入完成。打开社区版 DSH Desktop，新建会话并选择 **IronMLX / 你的模型**，按下文验证连接。本机未安装时的提示已通过原生 Dashboard 测试；已安装路径使用临时配置验证，真实社区版应用的端到端验证仍待安装后完成。

### 手动配置

Dashboard 中的手动说明默认折叠，需要自定义地址、LAN 接入或排障时可展开；“验证连接”作为一键配置和手动配置的共用步骤单独展示。

DSH Desktop 通过 GUI 保存提供方配置，不会自动加载 CLI overlay，也不需要修改 `DSH_HOME`。

### 获取连接信息

在 IronMLX Dashboard 的 **Agent** 页选择 **DSH Desktop**。向导会提供当前本机 endpoint 对应的连接字段和 GUI 操作步骤。

### 创建自定义提供方

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

### 验证连接

验证 Desktop 路径时，新建会话并选择 IronMLX 模型，要求模型只回复 `IRONMLX_OK`。正常返回即可确认提供方发现和 Responses 请求可用。工具仍由 DSH Desktop 在本机执行；IronMLX 只负责推理和返回结构化工具调用。

### 常见配置问题

`local` 仅是本机回环免鉴权接口的占位值。使用需要鉴权的 LAN 接口时，应改填实际的 IronMLX LAN 地址和 API Key。若无法获取模型，请确认 IronMLX 正在运行、API 地址包含 `/v1`，并且 API 协议为 `openai-responses`。

## DSH CLI 配置

CLI 优先使用上述一键配置。需要独立 overlay 时，可展开 Dashboard 手动配置或使用下面的示例，通过 `--patch` 启动并验证请求。

### 使用 Dashboard 生成配置

在 IronMLX Dashboard 的 **Agent** 页选择 **DeepSeek Harness → DSH CLI**。

CLI 向导根据当前已加载模型生成 dsh overlay，并在可用时从 `/v1/models` 读取 `context_window`、`max_output_tokens`。复制前请检查单轮输出预算：向导初始值不超过 4096 tokens，不会直接采用服务端公布的完整输出能力。若容量信息不可用，请手动填写模型实际的上限。

### 手动编写配置

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

### 保存配置

将 overlay 保存为 `$DSH_HOME/ironmlx.patch.yml`（通常为 `~/.dsh/ironmlx.patch.yml`）。仅在需要使用 IronMLX 的运行中传入 `--patch`；不要覆盖现有的 home 级 `cordis.patch.yml`。
如果 `$DSH_HOME/settings.yaml` 已指定默认模型或包含 `llm-pi-ai` 覆盖项，请在该文件中更新或移除冲突的配置：dsh 会在 overlay 的基础配置之后应用用户设置。

`IRONMLX_API_KEY=local` 只是满足 dsh 对本地免鉴权 endpoint 的凭据查询的占位值，不是 IronMLX 的 LAN API Key。本指南仅针对本机回环接口，不覆盖需要鉴权的 LAN 访问。

### 启动与验证

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

第一轮应以 completed 结束并返回 `IRONMLX_OK`。第二轮应出现 `tool_call`、`tool_result` 事件，最终返回 `IRONMLX_DSH_TOOL_OK`；模型只是描述工具调用不算通过。dsh 在本机执行工具并回传结果，IronMLX 负责推理和生成结构化工具调用。

### 常见配置问题

若单轮输出 token 用尽，仍可能没有完整答案；此时应增加预算或简化任务。失败时先核对 endpoint、准确的模型 ID、容量和模型推理能力，再查看[故障排查](troubleshooting.md)。
