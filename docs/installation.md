# Installation and build

## Supported platform

IronMLX 0.1.0 supports Apple Silicon (`arm64`) and macOS 26.2 or later. Intel
Macs and older macOS versions are outside the supported range.

## Distribution status

Third-party inventories, notices, and license texts are generated and bundled
for engineering review. Public binary distribution remains blocked until the
P0-8B legal review, CycloneDX SBOM, and explicit distribution authorization are
complete. Until then, build from a trusted source checkout for local validation.

## Model rights boundary

IronMLX can search for and download models, but it does not own or relicense
model rights. Before use, consult the upstream model page for its license,
gated-access terms, use restrictions, and redistribution rules. App, DMG, and
ZIP artifacts do not contain model weights. See [Model rights boundary](model-license-boundary.md)
for the full statement.

## Build the App from source

The build host needs full Xcode, CMake, Rust 1.94, `cargo-about 0.9.1`, and the
macOS 26.2 SDK/Metal toolchain. The following commands check out the pinned MLX
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
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
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
export MACOSX_DEPLOYMENT_TARGET=26.2
export CMAKE_OSX_DEPLOYMENT_TARGET=26.2
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

## MLX troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `MLX_DIR is not set` | The current shell did not export the build path. | Source `mlx-env.sh` or set the variables above. |
| `missing include/ or lib/` | `MLX_DIR` points to the MLX build tree instead of its install prefix. | Point it to `MLX_PREFIX`. |
| `Undefined symbols: _gguf_*` | `libgguflib.a` was not copied into the install prefix. | Repeat the copy step above or rerun `scripts/setup-mlx.sh`. |
| `Failed to load the default metallib` | `MLX_METAL_PATH` is unset or points to the wrong directory. | Set it to the directory containing `mlx.metallib`. |
