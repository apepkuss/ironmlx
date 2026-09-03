# IronMLX

[English](README.md)

IronMLX 是面向 Apple Silicon 的本地大语言模型推理 App 与服务运行时。它将
Rust 推理引擎、MLX/Metal 运行时、模型管理 Dashboard，以及 OpenAI/Anthropic
兼容 HTTP API 打包为一个自包含的 macOS App。

当前产品版本：**0.1.0**

## 系统要求

- Apple Silicon（arm64）；不支持 Intel Mac；
- macOS 26.2 或更高版本；
- 本地构建需要完整 Xcode、Rust 1.94 和项目锁定的 MLX 源码版本。

## 核心能力

- 本地模型搜索、不可变快照下载、断点续传与完整性校验；
- 多模型加载、卸载、固定、TTL 与内存保护；
- OpenAI `/v1/chat/completions`、`/v1/responses` 和 Anthropic `/v1/messages`，均支持客户端函数工具调用协议；
- 流式输出、连续批处理、分页 KV/前缀缓存、MTP 与 Prompt Lookup；
- 文本及受控 base64 图片输入；
- 默认仅监听 loopback；可选 LAN 模式使用 HTTPS 与 API Key。

## 快速开始

当前仓库已生成与 macOS arm64 Release 产物对应的第三方依赖清单、Notices 和
许可证文本，但在完成 P0-8B 法律复核、SBOM 与明确授权前，公开二进制分发仍被
发布门禁阻止。开发者可以从源码构建并在本机验证：

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
scripts/checkout-release-mlx.sh /tmp/ironmlx-mlx-source
MLX_SRC=/tmp/ironmlx-mlx-source scripts/build-app-bundle.sh
open dist/IronMLX.app
```

构建产物位于 `dist/IronMLX.app`。详细步骤见
[安装与构建](docs/zh-CN/installation.md)。

## 文档

- [支持模型矩阵](docs/zh-CN/supported-models.md)
- [模型权利边界](docs/zh-CN/model-license-boundary.md)
- [API 示例](docs/zh-CN/api.md)
- [Hermes Agent 集成](docs/zh-CN/hermes-agent.md)
- [oh-my-pi 集成](docs/zh-CN/oh-my-pi.md)
- [隐私与网络边界](docs/zh-CN/privacy.md)
- [数据位置与卸载](docs/zh-CN/storage-and-uninstall.md)
- [故障排查](docs/zh-CN/troubleshooting.md)
- [Known Issues](docs/zh-CN/known-issues.md)
- [0.1.0 候选发布说明](docs/zh-CN/release-notes/0.1.0.md)
- [版本与发布流程](docs/zh-CN/versioning-and-releases.md)
- [第三方依赖与许可证材料](docs/zh-CN/third-party-materials.md)
- [安全边界](docs/zh-CN/security-boundary.md)
- [诊断信息导出](docs/zh-CN/diagnostic-bundle.md)
- [开发预览发布](docs/zh-CN/development-preview-release.md)

## 开发验证

```bash
scripts/verify-version-consistency.sh
cargo fmt --all -- --check
cargo +nightly fmt --all -- --check
cargo +nightly clippy --locked --all-features --workspace -- -D warnings
cargo build --locked --release
swift test --package-path ironmlx-app --configuration release --no-parallel
```

App Bundle 的静态验证使用：

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
```
