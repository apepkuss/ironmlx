**{{RELEASE_STATUS_EN}} This build is for {{PURPOSE_EN}} Requires Apple Silicon and macOS {{MIN_MACOS}} or later.**

**[Download DMG]({{DMG_URL}})** · [ZIP alternative]({{ZIP_URL}})

Model weights are downloaded separately. Memory requirements depend on the model, quantization and context length.

### Highlights

- Local text and supported image inference across Qwen, Gemma, GLM, Llama and MiniCPM families, plus DiffusionGemma.
- Import existing local model directories into the IronMLX-managed library with copy verification while preserving the source directory.
- OpenAI Chat Completions / Responses and Anthropic Messages compatible APIs, streaming, model-dependent tool calling and structured outputs.
- Agent application integrations configured from the Dashboard.
- IndexTTS 2.5 speech synthesis with WAV/PCM output and managed local voice profiles.
- Laya decision-model serving through the TypeSafe-compatible System One API.
- Resumable model downloads with integrity checks, model loading controls, logs and local diagnostic exports.
- Concurrent inference, prompt caching and memory protection, with MTP or DFlash2 acceleration on compatible models.

Compatibility depends on the specific architecture, weights and runtime mode. Check the [supported model matrix]({{SUPPORTED_MODELS_URL}}) and [known issues]({{KNOWN_ISSUES_URL}}). {{UPDATE_CHANNELS_EN}}

[Full release notes]({{RELEASE_NOTES_URL}}) · [Report an issue]({{ISSUES_URL}}) · [Full changelog]({{CHANGELOG_URL}})

<details>
<summary>简体中文</summary>

**{{RELEASE_STATUS_ZH}}此版本用于{{PURPOSE_ZH}}，需要 Apple Silicon 和 macOS {{MIN_MACOS}} 或更高版本。**

**[下载 DMG 安装包]({{DMG_URL}})** · [ZIP 备选]({{ZIP_URL}})

模型权重需要另行下载；内存需求取决于模型、量化方式和上下文长度。

### 主要能力

- 支持 Qwen、Gemma、GLM、Llama、MiniCPM 系列及 DiffusionGemma 的兼容架构，提供文本及受支持的图片推理。
- 将本机已有模型目录复制并校验至 IronMLX 托管模型库，同时保留原始目录。
- 兼容 OpenAI Chat Completions / Responses 和 Anthropic Messages API，支持流式输出，以及依赖模型能力的工具调用和结构化输出。
- 在 Dashboard 中配置和管理 Agent 应用集成。
- IndexTTS 2.5 语音合成，支持 WAV/PCM 输出和本地托管语音档案。
- 通过兼容 TypeSafe 的 System One API 提供 Laya 决策模型服务。
- 模型断点续传、完整性校验、加载管理、日志及本地诊断导出。
- 并发推理、提示词缓存与内存保护；兼容模型可使用 MTP 或 DFlash2 加速。

具体兼容性取决于架构、权重和运行模式，请查看[支持模型矩阵]({{SUPPORTED_MODELS_ZH_URL}})及[已知问题]({{KNOWN_ISSUES_ZH_URL}})。{{UPDATE_CHANNELS_ZH}}

[完整发布说明]({{RELEASE_NOTES_ZH_URL}}) · [反馈问题]({{ISSUES_URL}})

</details>
