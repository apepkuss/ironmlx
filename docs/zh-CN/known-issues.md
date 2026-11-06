# 已知问题与限制 — 0.3.0

[English](../known-issues.md)

本页记录使用限制，不记录发布测试的完成状态。本发布线的变更见 [0.3.0 发布说明](release-notes/0.3.0.md)。

## 模型与 API 限制

- 图片请求接受 JPEG/PNG/WebP base64 内容，不接受远程图片 URL。
- 文本、图片与音频向量仅支持 EmbeddingGemma 2 BF16 和 affine4，当前不支持视频向量，详见[Embedding API](text-embeddings.md)。识别其他 embedding 和 reranker 元数据不代表运行时受支持。语音合成仅限文档中的 IndexTTS 2.5 配置及其独立音频运行时。
- 语音识别仅支持文档中的 Qwen3-ASR 检查点；时间戳／字幕需要 Qwen3 ForcedAligner，说话人标注需要 Nemotron 3。实时会话支持转写、可选 VAD 与说话人标签，不提供翻译、语音输出或降噪。Dashboard 不提供录音／转写客户端。限制与扩展见[音频 API](audio-transcriptions-api.md)；识别其他 ASR 元数据不代表运行时受支持。
- 决策服务仅限文档中的 Laya 模型及 System One／Decisions 契约，Decisions 输入仅支持文本；不是通用分类或 embedding 运行时。
- DiffusionGemma 不支持 KV cache、Prompt Lookup 或 MTP，可用采样参数也较少。
- MTP 和辅助 drafter 取决于具体 Qwen/Gemma 模型及兼容辅助模型。DFlash2 有独立的组合限制，详见[支持模型](supported-models.md)。
- 跨请求 Prompt Lookup 只适用于同一受信任域，不能作为多租户隔离机制。

## 平台与帮助

需要 Apple Silicon 和 macOS 26.4 或更高版本。本地源码构建使用 ad-hoc 签名，与签名发布的安装包不同。
启动、下载或连接失败时，请先阅读[故障排查](troubleshooting.md)。
