<p align="center">
  <img src="ironmlx-app/Packaging/AppIcon-1024.png" width="120" alt="IronMLX App icon">
</p>

<h1 align="center">IronMLX</h1>

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Built_with-Rust-B7410E?logo=rust" alt="Built with Rust"></a>
  <img src="https://img.shields.io/badge/Platform-Apple_Silicon-000000?logo=apple" alt="Apple Silicon">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <strong>Private AI on Apple Silicon.</strong> Power your Agents with local AI models.
</p>

<p align="center">
  <a href="#install-and-run">Installation</a> ·
  <a href="https://apepkuss.github.io/ironmlx/docs/supported-models.html">Supported Models</a> ·
  <a href="https://apepkuss.github.io/ironmlx/docs/api-reference.html">API Reference</a> ·
  <a href="https://apepkuss.github.io/ironmlx/docs/">Documentation</a> ·
  <a href="https://apepkuss.github.io/ironmlx/">Website</a>
</p>

<p align="center">
  <span lang="en">🇺🇸 English</span> |
  <a href="README.zh-CN.md" lang="zh-Hans">🇨🇳 中文</a>
</p>

## Requirements

- Apple Silicon (`arm64`); Intel Macs are not supported;
- macOS 26.4 or later;

## Capabilities

- **GUI Dashboard** — Manage models, monitor service status, and adjust settings through a visual interface.

- **Agent integrations** — Connect [Hermes Agent](docs/hermes-agent.md), [oh-my-pi](docs/oh-my-pi.md), and [DSH CLI / Desktop](docs/dsh.md).

- **Model capabilities** — [Text and vision](docs/text-vision-api.md), [text, image and audio embeddings](docs/text-embeddings.md), [speech synthesis](docs/audio-speech-api.md), [image generation](docs/image-generation-api.md), and [decision inference](docs/laya-systemone-api.md).

- **Model management** — Download models from Hugging Face or ModelScope, [import local models](docs/user-guide.md#import-a-model-already-on-this-mac), and manage multiple models.

- **API compatibility** — [OpenAI- and Anthropic-compatible APIs](docs/api-reference.md) with streaming, tool-call protocols, and Structured Outputs.

- **Inference optimization** — Continuous batching and KV caching, with [MTP, DFlash2, or Assistant](docs/supported-models.md) acceleration for compatible models.

- **Privacy and diagnostics** — Local access by default, optional [LAN authentication](docs/security-boundary.md), and [local redacted diagnostic exports](docs/diagnostic-bundle.md).

## Install and run

1. **Download** — Get the Apple Silicon DMG or ZIP from [GitHub Releases](https://github.com/apepkuss/ironmlx/releases).
2. **Install and launch** — For a DMG, drag `IronMLX.app` to `Applications`, wait for the copy to finish, eject the disk image, and launch the App from Finder. For a ZIP, extract it and open the App.
3. **Load a model** — In Dashboard, download or import a [compatible model](docs/supported-models.md), then load it.
4. **Connect an Agent or API client (optional)** — Follow [Agent configuration](docs/user-guide.md#agent-configuration) or the [API quick start](docs/developer-guide.md#api-quick-start). The App endpoint defaults to `http://127.0.0.1:9068`.

> [!TIP]
> For complete setup instructions, see the [User guide](docs/user-guide.md#install-and-start).

## License

- **Source code** — IronMLX original source code is licensed under [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.

- **Third-party components** — Dependencies and bundled assets retain their own licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md) and [license texts](THIRD_PARTY_LICENSES/).

- **Model weights** — Model weights are not licensed or redistributed by IronMLX; follow the license and usage terms of their upstream repositories.
