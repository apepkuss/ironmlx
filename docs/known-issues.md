# Known issues — 0.1.0

- Only Apple Silicon arm64 and macOS 26.2+ are supported; there is no Intel or
  older-system compatibility path.
- Source builds use an ad-hoc signature and are not notarized; they are not
  formal distribution installers.
- Third-party notices, license texts, and the engineering inventory exist, but
  legal review and the CycloneDX SBOM are incomplete; public binary release
  remains blocked by the P0-8B gate.
- HTTP image input accepts JPEG/PNG/WebP base64 only; remote URLs are disabled.
- The App may identify embedding, reranker, ASR, or TTS metadata, but the server
  loads only the LLM/VLM generation architectures in the [supported-models matrix](supported-models.md).
- DiffusionGemma uses an independent block-diffusion path and does not support
  KV cache, Prompt Lookup, or MTP; it also exposes fewer sampling parameters.
- MTP/auxiliary drafter support is limited to Qwen and Gemma4.
- Cross-request Prompt Lookup is for one trusted domain only, not multi-tenant
  isolation.
- GitHub-hosted runners and fixture tests do not replace real-model acceptance
  on the minimum target machine.
