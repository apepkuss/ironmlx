# Known issues and limitations — 0.2.0

[简体中文](zh-CN/known-issues.md)

This page lists usage limitations. It is not a record of completed or pending release tests.
For changes in this release line, see the [0.2.0 release notes](release-notes/0.2.0.md).

## Model and API limitations

- Image requests accept JPEG/PNG/WebP base64 content, not remote image URLs.
- Embedding, reranker and ASR metadata recognition does not imply runtime support. Speech synthesis is limited to the documented IndexTTS 2.5 profile and its separate audio runtime.
- Decision serving is limited to the documented Laya model and System One contract. It is not a generic classifier or embedding runtime.
- DiffusionGemma does not support KV cache, Prompt Lookup or MTP and has a smaller sampling-parameter set.
- MTP and auxiliary drafter support depends on the Qwen or Gemma model and a compatible auxiliary model. DFlash2 has a separate set of combination limits; see [Supported models](supported-models.md).
- Cross-request Prompt Lookup assumes one trusted domain; it is not a multi-tenant isolation mechanism.

## Platform and help

Apple Silicon and macOS 26.4 or later are required. Local source builds use ad-hoc signing and differ from signed release installers.
For startup, download or connection failures, follow [Troubleshooting](troubleshooting.md).
