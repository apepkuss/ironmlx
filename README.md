# IronMLX

[简体中文](README.zh-CN.md)

IronMLX is a local large-language-model inference App and service runtime for
Apple Silicon. It packages a Rust inference engine, MLX/Metal runtime, model
management Dashboard, and OpenAI/Anthropic-compatible HTTP APIs into a
self-contained macOS App.

![IronMLX Dashboard overview](docs/images/dashboard-overview.png)

The screenshot shows the local Dashboard with a running server, a loaded model,
and the DFlash2 runtime status. Runtime metrics vary by model and hardware.

Latest public release: **0.1.0-rc.1**

This is a release candidate, not a stable release. The product version base is
**0.1.0**.

## Requirements

- Apple Silicon (`arm64`); Intel Macs are not supported;
- macOS 26.4 or later;

## Capabilities

- Local model search, immutable-snapshot downloads, resume, and integrity checks;
- Multi-model loading, unloading, pinning, TTL, and memory protection;
- OpenAI `/v1/chat/completions` and `/v1/responses`, plus Anthropic
  `/v1/messages`, with client-side function-call protocols;
- Streaming, continuous batching, paged KV/prefix cache, MTP, and Prompt Lookup;
- Qwen3.8 reasoning/tools, matching MTP, and isolated DFlash2 text execution;
- Text and controlled base64 image input;
- Local redacted diagnostic export with no prompt, credential, or network upload;
- Loopback by default, with optional LAN mode using HTTPS and API keys.

## Install and run

Download the current Apple Silicon release candidate from the
[GitHub Release](https://github.com/apepkuss/ironmlx/releases/tag/v0.1.0-rc.1),
open the DMG or ZIP, and launch `IronMLX.app`. Then use the Dashboard to select
and load a compatible model.

For API clients, the App listens on `http://127.0.0.1:9068` by default. See the
[HTTP API quick start](docs/api.md) for a first request.

Developers who need to build and run IronMLX from source should follow the
[Developer Guide](docs/developer-guide.md).

## Documentation

- [User Guide](docs/user-guide.md) — installation, model use, API clients,
  agent integrations, privacy, and troubleshooting.
- [Developer Guide](docs/developer-guide.md) — source builds, tests,
  contributions, architecture, and release validation.
- [Supported model matrix](docs/supported-models.md) — architectures and
  concrete versions recorded for this release candidate.

## License

IronMLX original source code is licensed under the Apache License, Version 2.0;
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party dependencies and
bundled assets remain under their respective licenses, as listed in
`THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/`. Model weights are not
licensed or redistributed by IronMLX; users are responsible for the terms of
the upstream model repository.
