# Building from source

[简体中文](zh-CN/building-from-source.md)

## Supported platform

IronMLX 0.2.0 supports Apple Silicon (`arm64`) and macOS 26.4 or later. Intel
Macs and older macOS versions are outside the supported range.

## Distribution status

RC/stable workflows support Developer ID signing, notarization and stapled tickets. Ordinary local source builds remain ad-hoc and are not equivalent to signed distribution artifacts. Available downloads are defined by actual public GitHub Releases; enabling the distribution gate does not itself publish a release.

## Model rights boundary

IronMLX can search for and download models, but it does not own or relicense
model rights. Before use, consult the upstream model page for its license,
gated-access terms, use restrictions, and redistribution rules. App, DMG, and
ZIP artifacts do not contain model weights. See [Model rights boundary](model-license-boundary.md)
for the full statement.

## Build the App from source

The build host needs full Xcode, CMake, Rust 1.94, `cargo-about 0.9.1`, and the
macOS 26.4 SDK/Metal toolchain. The following commands check out the pinned MLX
commit and build a self-contained Release App:

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
scripts/checkout-release-mlx.sh /tmp/ironmlx-mlx-source
MLX_SRC=/tmp/ironmlx-mlx-source scripts/build-app-bundle.sh
```

The builder rejects a dirty MLX checkout or a checkout at the wrong commit.
After a successful build:

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
open dist/IronMLX.app
```

Local builds use an ad-hoc signature and are not notarized; they are not formal
distribution installers. Do not bypass macOS security controls to run an
untrusted build.

## CLI development build

For backend-only work, prepare a pinned MLX Release install with NAX Metal
kernels enabled, then set `MLX_DIR` and `MLX_METAL_PATH`:

```bash
export MLX_DIR=/path/to/validated/mlx-install
export MLX_METAL_PATH="$MLX_DIR/lib"
cargo build --release --bin ironmlx --bin iron-bench
target/release/ironmlx --version
```

## Build and install the MLX dependency

The MLX C++ dependency must be built as a static arm64 library with Metal
kernels enabled. The repository helper performs this build, installs the
libraries, and writes a sourceable environment file:

```bash
MLX_SRC=/path/to/mlx-source \
MLX_PREFIX="$HOME/.local/mlx" \
scripts/setup-mlx.sh
source "$HOME/.local/mlx/mlx-env.sh"
```

For a manual build, use the same deployment target and static-library options:

```bash
MLX_SRC=/path/to/mlx-source
MLX_PREFIX="$HOME/.local/mlx"

cd "$MLX_SRC"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_METAL_JIT=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.4 \
  -DCMAKE_INSTALL_PREFIX="$MLX_PREFIX"
cmake --build build --parallel "$(sysctl -n hw.ncpu)"
cmake --install build
```

MLX's install step does not export its private GGUF transitive library. Copy
it into the install prefix so that the `mlx` crate's GGUF tests and any GGUF
consumer can link successfully:

```bash
cp "$MLX_SRC/build/mlx/io/libgguflib.a" "$MLX_PREFIX/lib/"
```

The production `ironmlx` binary does not use GGUF weights, but omitting this
library causes GGUF-related tests to fail with undefined `_gguf_*` symbols.
`mlx-sys/build.rs` links every `lib*.a` in `MLX_DIR/lib`, so no additional
linker flags are needed.

## MLX build and runtime environment

Each shell, CI job, and tool invocation that builds or runs IronMLX must set
the MLX paths explicitly:

```bash
export MLX_ROOT="$HOME/.local/mlx"
export MLX_DIR="$MLX_ROOT"
export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export MACOSX_DEPLOYMENT_TARGET=26.4
export CMAKE_OSX_DEPLOYMENT_TARGET=26.4
```

Although MLX is statically linked, `mlx.metallib` is loaded at runtime and
must be present under `MLX_METAL_PATH`.

## MLX installation sanity check

Before building IronMLX, verify the headers, static libraries, and Metal
kernel library are present:

```bash
test -f "$MLX_DIR/include/mlx/array.h"
test -f "$MLX_DIR/lib/libmlx.a"
test -f "$MLX_DIR/lib/libgguflib.a"
test -f "$MLX_DIR/lib/mlx.metallib"
```

## Backend tests and local serving

After sourcing the environment above, run the complete workspace test suite:

```bash
cargo build --release
cargo test --all-features --workspace
```

To run a local text-generation smoke test:

```bash
MODEL="$HOME/.ironmlx/models/<org>/<model>"
./target/release/ironmlx generate \
  --model "$MODEL" \
  --prompt "Describe mixture-of-experts architecture in one sentence." \
  --max-tokens 128 \
  --temperature 0 \
  --prefill-chunk-size 2048
```

To start the local server:

```bash
./target/release/ironmlx serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --prefill-chunk-size 2048 \
  --b-max 1 \
  --max-cache-cap 32768
```

### Single backend instance

One `ironmlx serve` backend is allowed per macOS user, regardless of arguments or port. Before MLX, metallib or model initialization, the process takes an exclusive nonblocking lock on `~/.ironmlx/run/backend.lock` until exit. Normal exit, crashes and SIGKILL release the lock; the file itself can remain and is not a liveness indicator.
A second instance exits with `ironmlx_instance_already_running`. The App stops its automatic recovery loop and asks the user to exit the existing instance.

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

Release validation must distinguish source tests, static Bundle checks, signed
and notarized artifacts, Gatekeeper checks, and public distribution.

### SDK compatibility checks

| SDK | Pinned version | Coverage |
| --- | --- | --- |
| OpenAI Python | `2.48.0` | Chat / Responses, SSE, tools, Structured Outputs, reasoning, 400/413/503 |
| Anthropic Python | `0.121.0` | Messages, SSE, tools, Structured Outputs + thinking, 400/413/503 |

Pinned SDKs access a fixture server over real loopback HTTP/SSE to check client parsing; Rust tests separately cover production request/response contracts. Neither loads a model or establishes response quality, tool selection accuracy or performance.
Run from the repository root:

```bash
python3 -m venv /tmp/ironmlx-api-contract-sdk
/tmp/ironmlx-api-contract-sdk/bin/python -m pip install -r scripts/api-contract-sdk/requirements.txt
/tmp/ironmlx-api-contract-sdk/bin/python scripts/api-contract-sdk/contract.py --fixture
cargo test --locked --all-features -p ironmlx --lib server::
```

## Runtime development notes

### App log-setting application

The App reads the backend snapshot, updates the filter with process/revision preconditions, saves only `log_level`, then updates its own filter. While stopped, it saves the setting for the next launch without an HTTP call; startup, recovery, shutdown or another settings transaction blocks changes.
After a lost response or save failure it queries actual state and restores the previous level only while preconditions still match. A changed process/revision or unconfirmed rollback is not displayed as success.
Legacy TRACE/WARN map to ALL/WARNING. Third-party dependency logs are capped at WARNING, or ERROR when selected. Helper launches receive `IRONMLX_LOG_LEVEL`; independent CLI use can retain `RUST_LOG`.

### Native tool templates

MiniCPM-V 4.6 and MiniCPM5 use distinct XML dialects. MiniCPM5 encodes string parameters containing <, & or newlines using CDATA. Gemma internally projects dynamic objects into deterministic key/value entries and restores original objects on output; public Schema and argument shapes stay unchanged.

### Runtime topology

Ordinary causal HTTP requests all use SchedulerActor, including long chunked-prefill,
multimodal, sampled and constrained requests; request shape no longer selects a
direct GenerationStream serving path. Public request priorities are defined in the
[text and vision API](text-vision-api.md#request-priority).

### Streaming resource release

Once SSE starts, dropping the HTTP response publishes a cancellation signal. Encoders stop consuming generation events and do not fabricate a terminal event after observing disconnect.

| Path | Cancellation boundary | Released state |
| --- | --- | --- |
| Scheduler, including ordinary causal, MTP and drafter serving | Next safe scheduling boundary after the current forward | Request, slot, KV cache and budget |
| DFlash2 | Next safe event boundary after target/draft forward | Per-request caches, slot and budget |
| DiffusionGemma | Next event boundary after the current diffusion step | Lane and request state |

Cancellation does not interrupt an in-flight Metal operation. Resource release can therefore include the remainder of that operation. Version 0.1 does not promise cancellation of underlying non-streaming generation when its client disconnects.

## MLX troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `MLX_DIR is not set` | The current shell did not export the build path. | Source `mlx-env.sh` or set the variables above. |
| `missing include/ or lib/` | `MLX_DIR` points to the MLX build tree instead of its install prefix. | Point it to `MLX_PREFIX`. |
| `Undefined symbols: _gguf_*` | `libgguflib.a` was not copied into the install prefix. | Repeat the copy step above or rerun `scripts/setup-mlx.sh`. |
| `Failed to load the default metallib` | `MLX_METAL_PATH` is unset or points to the wrong directory. | Set it to the directory containing `mlx.metallib`. |
