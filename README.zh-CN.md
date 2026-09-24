# IronMLX

[English](README.md)

IronMLX 是面向 Apple Silicon 的本地大语言模型推理 App 与服务运行时。它将
Rust 推理引擎、MLX/Metal 运行时、模型管理 Dashboard，以及 OpenAI/Anthropic
兼容 HTTP API 打包为一个自包含的 macOS App。

![IronMLX Dashboard 中文界面总览](docs/images/dashboard-overview-zh-CN.png)

截图展示了本地 Dashboard、运行中的服务、已加载模型以及 DFlash2 运行状态。
运行时指标会因模型和硬件而变化。

当前正式版本：[0.1.0](https://github.com/apepkuss/ironmlx/releases/tag/v0.1.0)

## 系统要求

- Apple Silicon（arm64）；不支持 Intel Mac；
- macOS 26.4 或更高版本；

## 核心能力

- 支持从 Hugging Face 和 ModelScope 发现模型，并提供不可变快照下载、
  断点续传与完整性校验；
- 支持多种模型架构，支持范围持续扩展，详见
  [支持模型矩阵](docs/zh-CN/supported-models.md)；
- 提供标准化、兼容主流客户端的 API，包括 OpenAI 兼容的
  `/v1/chat/completions`、`/v1/responses` 和 Anthropic `/v1/messages`，
  支持客户端函数工具调用协议；
- 支持同步响应、SSE 流式输出、Structured Outputs 和 reasoning；
- 提供模型发现、健康检查与运行状态查询 API，包括 `/healthz` 和
  `/v1/models`；
- 支持 Hermes Agent、oh-my-pi 等 Agent Harness，支持范围持续扩展，详见
  [Hermes Agent 集成文档](docs/zh-CN/hermes-agent.md) 和
  [oh-my-pi 集成文档](docs/zh-CN/oh-my-pi.md)；
- 面向高吞吐推理，支持连续批处理、分页 KV/前缀缓存与 Prompt Lookup；
- 支持 MTP 和 DFlash2 等投机解码路径，具体取决于模型兼容性；
- 支持多模型加载、卸载、固定、TTL 与内存保护；
- 支持 LLM/VLM 和多模态推理，具体能力取决于模型；
- 本地脱敏诊断信息导出，不包含 prompt、凭据且不上传网络；
- 提供本地数据、隐私与模型授权边界说明；
- 默认仅监听 loopback；可选 LAN 模式使用 HTTPS 与 API Key。

## 安装与首次运行

从 [GitHub Release](https://github.com/apepkuss/ironmlx/releases/tag/v0.1.0)
下载当前 Apple Silicon 正式版。使用 DMG 时，打开 DMG，将 `IronMLX.app` 拖到
`Applications` 文件夹图标，推出磁盘映像，再从 Finder 的“应用程序”中启动
IronMLX；使用 ZIP 时，先解压，再打开解压出的 App。然后在 Dashboard 中选择并加载
兼容的模型。

App 默认监听 `http://127.0.0.1:9068`。如需发送第一条 API 请求，请参阅
[HTTP API 快速开始](docs/zh-CN/api.md)。

需要从源码构建和运行 IronMLX 的开发者，请参阅[开发者指南](docs/zh-CN/developer-guide.md)。

## 文档

- [用户指南](docs/zh-CN/user-guide.md)：安装、模型使用、API 客户端、Agent 集成、隐私和故障排查；
- [开发者指南](docs/zh-CN/developer-guide.md)：源码构建、测试、贡献、架构和发布验收。
- [支持模型矩阵](docs/zh-CN/supported-models.md)：当前版本已记录的模型架构和具体版本。

## 许可证

IronMLX 原创源代码采用 Apache License 2.0，详见 [LICENSE](LICENSE) 和
[NOTICE](NOTICE)。第三方依赖和随 App 打包的资产仍分别受其自身许可证约束，
完整清单见 `THIRD_PARTY_NOTICES.md` 和 `THIRD_PARTY_LICENSES/`。模型权重不由
IronMLX 授权或再分发，用户自行承担上游模型仓库条款责任。
