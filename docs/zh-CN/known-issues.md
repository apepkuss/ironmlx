# 已知问题与限制 — 0.1.0

[English](../known-issues.md)

本页记录使用限制，不记录发布测试的完成状态。特定版本的变更见[发布说明](release-notes/0.1.0.md)。

## 模型与 API 限制

- 图片请求接受 JPEG/PNG/WebP base64 内容，不接受远程图片 URL。
- 模型列表可以识别 embedding、reranker、ASR 和 TTS 元数据，但生成后端仅加载[支持模型](supported-models.md)中列出的 LLM/VLM 模型族。
- DiffusionGemma 不支持 KV cache、Prompt Lookup 或 MTP，可用采样参数也较少。
- MTP 和辅助 drafter 取决于具体 Qwen/Gemma 模型及兼容辅助模型。DFlash2 有独立的组合限制，详见[支持模型](supported-models.md)。
- 跨请求 Prompt Lookup 只适用于同一受信任域，不能作为多租户隔离机制。

## 平台与帮助

需要 Apple Silicon 和 macOS 26.4 或更高版本。本地源码构建使用 ad-hoc 签名，与签名发布的安装包不同。
启动、下载或连接失败时，请先阅读[故障排查](troubleshooting.md)。
