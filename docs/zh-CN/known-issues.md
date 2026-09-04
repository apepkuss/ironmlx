# Known Issues — 0.1.0

- 仅支持 Apple Silicon arm64 和 macOS 26.2+；没有 Intel/旧系统兼容路径。
- 当前源码构建仅为 ad-hoc 签名且未经 Apple 公证，不是正式可分发安装包。
- 第三方 Notices、许可证原文与工程清单已生成，但尚未完成法律复核与 CycloneDX
  SBOM 批准；public binary 发布继续由 P0-8B 门禁阻止。
- HTTP 图片输入只接受 JPEG/PNG/WebP base64；远程 URL 有意禁用。
- App 模型列表会识别部分 embedding、reranker、ASR、TTS 元数据，但服务端只加载
  [支持模型矩阵](supported-models.md)中的 LLM/VLM 生成架构。
- DiffusionGemma 走独立 block-diffusion 路径，不支持 KV cache、Prompt Lookup
  或 MTP，采样参数也少于 causal 模型。
- MTP/辅助 drafter 只支持 Qwen 与 Gemma4；GLM、Llama、MiniCPM-V 与
  DiffusionGemma 不支持该模式。
- 跨请求 Prompt Lookup 只能用于同一受信任域，不能作为多租户隔离机制。
- GitHub-hosted runner 与 fixture 测试不能替代最低目标机器上的真实模型验收。
