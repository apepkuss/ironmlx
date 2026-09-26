# IronMLX Developer Guide

[简体中文](zh-CN/developer-guide.md)

This guide organizes source development and release work for IronMLX. User
installation and API usage belong in the [User Guide](user-guide.md).

## Build from source

Start with [Building from source](building-from-source.md). It documents the
supported Apple Silicon platform, pinned MLX checkout, Rust and Xcode
requirements, self-contained Release App build, runtime environment, and local
serving smoke test.

## Understand the project

- `ironmlx-core` and `ironmlx-lm` contain model and tensor logic.
- `ironmlx-runtime` owns execution, scheduling, lifecycle, and resources.
- `ironmlx` provides the HTTP API and CLI layer.
- `ironmlx-app` provides the macOS App and Dashboard.

When changing a model family or protocol, trace the complete path from the App
or HTTP request through runtime scheduling to model execution and response
streaming.

## Verify changes

Run the checks relevant to the change, then use the full workspace gates before
opening a pull request:

```bash
cargo fmt --all -- --check
cargo +nightly fmt --all -- --check
cargo +nightly clippy --locked --all-features --workspace -- -D warnings
cargo build --locked --release
cargo test --locked --all-features --workspace -- --test-threads=1
swift test --package-path ironmlx-app --configuration release --no-parallel
```

For a built App Bundle, also run:

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
scripts/verify-model-distribution-boundary.sh dist/IronMLX.app
```

Fixture, ignored, or source-only checks do not replace real model, protocol,
streaming, and App runtime validation when those paths are affected.

## Extend IronMLX

- Model work: review [Supported models](supported-models.md), the relevant
  model loader, native template, tokenizer, weight layout, and quantization
  metadata.
- API work: review [API reference](api-reference.md) and the [API compatibility
  matrix](api-compatibility-matrix.md), including typed streaming events and
  client-side tool execution boundaries.
- Runtime work: review [Engine pool](engine-pool.md), [Scheduler profile](scheduler-profile-v5.md),
  and the applicable cache or acceleration guide.

## Contribute and release

- [Contributing](../CONTRIBUTING.md)
- [Support](../SUPPORT.md)
- [Security reporting](../SECURITY.md)
- [Versioning and releases](versioning-and-releases.md)
- [Stable release pipeline](stable-release-pipeline.md)
- [0.2.0 release notes](release-notes/0.2.0.md)
- [0.1.0 release notes](release-notes/0.1.0.md)

Release validation must distinguish source tests, static Bundle checks, signed
and notarized artifacts, Gatekeeper checks, and public distribution.
