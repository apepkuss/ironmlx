# IronMLX App Bundle 0.1.0

The release build produces one self-contained `IronMLX.app` for Apple Silicon `arm64` on
macOS 26.4 or newer. It does not support Intel or earlier macOS releases.

## App responsibility

The App manages local models and configures and monitors the inference service.
Upstream applications submit task inputs through the service APIs and consume the
results. Dashboard controls are for model and service management; task-specific
upload, inference and result workflows belong in upstream applications.

## Release build

The release builder requires a clean MLX checkout pinned to commit
`73ad5df20cb30be4192e5c4d0ae8130674773427`. By default it uses the sibling
checkout at `../iron-rivals/mlx`; set `MLX_SRC` to another clean checkout of
that exact commit when needed.

```bash
scripts/build-app-bundle.sh
```

The command rebuilds MLX, `mlx.metallib`, the Rust helpers, and the Swift App
executable in an isolated build directory. It does not copy from an existing
`target/debug`, `target/release`, SwiftPM product directory, or external MLX
installation. The resulting bundle is `dist/IronMLX.app`.

The assembled layout is:

```text
IronMLX.app/Contents/
├── Info.plist
├── MacOS/IronMLX
├── Helpers/
│   ├── ironmlx
│   └── iron-bench
└── Resources/
    ├── mlx.metallib
    ├── dashboard2.html
    ├── AppIcon.icns
    ├── menubar-icon.png
    ├── menubar-icon@2x.png
    ├── logo.png
    ├── sidebar-logo@2x.png
    └── Legal/
        ├── THIRD_PARTY_NOTICES.md
        ├── third-party-inventory.json
        └── THIRD_PARTY_LICENSES/
```

Ordinary local builds are ad-hoc signed. RC and stable distribution use the
signing, notarization and archive checks described in the
[release pipeline](../docs/stable-release-pipeline.md).

The bundled helper shares the same per-user backend lock as direct
`ironmlx serve` CLI launches. Only one IronMLX backend may run for a macOS
user, regardless of App instance or port. When another backend owns the lock,
the App reports the conflict before creating its Dashboard or menu-bar UI,
does not enter automatic helper recovery, and exits after the alert is dismissed.
The App always starts the model-manager helper first, even when no local model
is configured; after the helper is healthy, the UI is shown and confirmed models
continue restoring asynchronously.

## Downloading incompatible models

Downloads stop before weights are transferred when metadata preflight rejects the
model. Explicit architecture or quantization-format incompatibility offers
**Download anyway** between **View supported models** and **Delete task**.
This choice permits file storage only: disk checks, required metadata, provider
file identities, checksums, and publication verification remain mandatory.
The choice survives pause/resume and restoration of paused tasks after restart.
Completed snapshots are marked **Downloaded · Cannot run in this version** and
cannot be loaded or selected as the default model. Model Management still allows
file access, integrity verification, and deletion. Missing files, malformed
metadata, and integrity failures cannot be overridden.

Nemotron 3 Diarization BF16 runs through the native Rust/MLX audio runtime.
The App identifies it as Diarization, manages its lifecycle, and promotes verified
file-only snapshots after native metadata preflight without re-downloading weights.
Upstream clients use `POST /v1/audio/diarizations` for independent JSON/RTTM or
uploaded-file SSE results. `response_format=diarized_json` on transcription combines
Qwen3-ASR, ForcedAligner and Nemotron, with explicit ambiguous-overlap annotations.
No recording or transcription UI is added. See the
[recognition and diarization API](../docs/audio-transcriptions-api.md).

Qwen3-ASR 0.6B/1.7B 8-bit checkpoints run through the native Rust/MLX ASR
engine. The App manages their download, loading and service status. Upstream
applications submit audio through `POST /v1/audio/transcriptions`; the Dashboard
does not include a transcription client. The supported offline profile accepts
WAV, FLAC or MP3 recordings with automatic long-audio segmentation.
Realtime PCM transcription is available over `/v1/realtime`, with optional CPU
server VAD configured per API session. The backend detects speech boundaries and
automatically commits turns; the App adds no recording or transcription controls.
Each ASR model's parameter page exposes maximum audio duration and maximum
output tokens per segment. Both Qwen3-ASR sizes default to 1,200 seconds and
1,024 tokens, using the official chunk/evaluation policy as a reference.
Values persist independently by model ID and apply to new requests and realtime
sessions, without reloading weights. The App restores them on backend startup.
Qwen3-ForcedAligner checkpoints whose top-level `model_type` is `qwen3_asr`
are identified by the nested `thinker_config.model_type` and serve the independent
`POST /v1/audio/alignments` task. Timestamped transcription uses this checkpoint
as a fixed auxiliary model. It has no autoregressive output-token setting and
cannot become the default chat model.

Previously downloaded file-only snapshots are promoted after native preflight,
without downloading their weights again. ASR tokenizers may use `tokenizer.json`
or split `vocab.json`/`merges.txt`, with `tokenizer_config.json` and
`preprocessor_config.json`. See the [transcription API](../docs/audio-transcriptions-api.md)
for CLI usage, OpenAI-compatible requests and limits.

Hugging Face transfers automatically switch to macOS URLSession after a Rust
transport connection failure. The switch retains the identity-bound committed
prefix, and subsequent retries resume with URLSession. Both transports require
the pinned revision, expected size, and SHA-256 verification. Network and TLS
failures show recovery guidance without exposing signed download URLs.

## Static verification

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
```

This gate validates bundle contents, `arm64`, macOS 26.4 minimum versions,
system-only dynamic dependencies, the metallib AIR target, absence of symlinks,
absence of developer-home/Cargo fallback paths, exact bundled legal materials,
and the ad-hoc signature.

Real release acceptance additionally requires launching this exact artifact in
an environment without Rust, Xcode, an external MLX installation, or `MLX_*` /
`DYLD_LIBRARY_PATH`; then loading a real model and completing real inference.
The same artifact must be copied from the M5 Max builder to an M1 Pro running
macOS 26.4 or newer without rebuilding or replacing any bundled file.
