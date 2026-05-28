# Qwen3.6 Performance Phase 6 MLX NAX Runtime Matrix

**Goal:** Determine whether the remaining Qwen3.6 GatedDeltaNet/qmm gap is caused by
MLX source version, ordinary Release optimization flags, or local MLX Metal runtime
build features.

**Artifact root:** `/tmp/ironmlx-qwen36-perf-mlx-matrix-latest`

## Controlled MLX Builds

Built four MLX runtime variants for the same ironmlx benchmark code:

| Runtime | Source / install | Build notes |
| --- | --- | --- |
| `v0.31.2` | `/tmp/ironmlx-mlx-matrix/install-v0.31.2` | Clean source worktree, `Release`, static, Metal |
| `84961223` | `/tmp/ironmlx-mlx-matrix/install-84961223` | Clean source worktree, `Release`, static, Metal |
| `2165dc08` | `/tmp/ironmlx-mlx-matrix/install-2165dc08` | Clean source worktree, `Release`, static, Metal |
| `local` | `/Users/xin/.local/mlx` | Existing installed MLX control |

All clean builds used:

- `CMAKE_BUILD_TYPE=Release`
- `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`
- `BUILD_SHARED_LIBS=OFF`
- `MLX_BUILD_METAL=ON`
- `MLX_METAL_JIT=OFF`

The clean builds were compiled with the active full Xcode toolchain:

- `xcode-select -p`: `/Applications/Xcode.app/Contents/Developer`
- macOS SDK: `26.5`
- Metal version macro: `400`

## Decisive Build Difference

The existing installed `/Users/xin/.local/mlx` was also a Release/O3 build, but its
MLX C++ flags include:

```text
-DMLX_METAL_NO_NAX
```

The clean `84961223` and `2165dc08` builds do not include that define. MLX enables
NAX kernels only when:

```text
MLX_METAL_VERSION >= 400 && MACOS_SDK_VERSION >= 26.2
```

This explains the local install gap better than a source-level MLX 0.32 qmm
regression. The local install was built with a toolchain/SDK path that disabled
NAX kernels, producing a smaller `mlx.metallib`:

| Runtime | `mlx.metallib` size | NAX status |
| --- | ---: | --- |
| local `/Users/xin/.local/mlx` | 120 MB | disabled (`MLX_METAL_NO_NAX`) |
| clean `84961223` | 150 MB | enabled |
| clean `2165dc08` | 150 MB | enabled |

## QLinear Matrix

Command shape:

```bash
MLX_DIR=<runtime> CARGO_TARGET_DIR=<isolated-target> \
  cargo run --release -p ironmlx --bin ironmlx-qlinear-bench -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --layer 0 --seq 521 --seq 1 --runs 50 --warmup-runs 10 \
  --include-cxx-qmm --out <artifact>
```

Selected `seq=521` p50:

| Runtime | qkvz C++ loop qmm | qkvz linear+slice | out direct qmm | out C++ loop qmm |
| --- | ---: | ---: | ---: | ---: |
| clean `v0.31.2` | 1.071 ms | 0.945 ms | 0.479 ms | 0.483 ms |
| clean `84961223` | 0.991 ms | 0.960 ms | 0.481 ms | 0.474 ms |
| clean `2165dc08` | 1.065 ms | 0.955 ms | 0.482 ms | 0.476 ms |
| local `/Users/xin/.local/mlx` | 2.300 ms | 2.297 ms | 0.966 ms | 0.980 ms |

The clean 0.32-era builds are in the same fast band as `v0.31.2`; the installed
local runtime is the outlier.

## GatedDeltaNet Matrix

Command shape:

```bash
MLX_DIR=<runtime> CARGO_TARGET_DIR=<isolated-target> \
  cargo run --release -p ironmlx --bin ironmlx-gdn-bench -- \
  --model /Users/xin/.ironmlx/models/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --layer 0 --seq 521 --runs 50 --warmup-runs 10 \
  --cache-mode all --out <artifact>
```

Selected `seq=521` p50:

| Runtime | no-cache | cache-out-only | cache-state-eval |
| --- | ---: | ---: | ---: |
| clean `v0.31.2` | 2.465 ms | 2.147 ms | 2.148 ms |
| clean `84961223` | 2.453 ms | 2.149 ms | 2.153 ms |
| clean `2165dc08` | 2.940 ms | 2.152 ms | 2.138 ms |
| local `/Users/xin/.local/mlx` | 4.174 ms | 3.996 ms | 4.000 ms |

For the production-relevant cache materialization path, the local non-NAX runtime is
about 1.86x slower than clean NAX-enabled MLX.

## Product Runtime Policy

1. Treat `MLX_METAL_NO_NAX` as a production performance gate for Qwen3.6 4-bit.
2. Rebuild `/Users/xin/.local/mlx` with full Xcode selected and SDK/Metal support
   satisfying the NAX condition.
3. Verify the installed MLX before product benchmarking:
   - `CMAKE_BUILD_TYPE=Release`
   - `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`
   - no `MLX_METAL_NO_NAX` in `CMakeFiles/mlx.dir/flags.make`
   - `mlx.metallib` size in the expected NAX-enabled range, currently about 150 MB
   - `ironmlx-qlinear-bench --include-cxx-qmm` p50 in the clean-build range
4. Do not pin Qwen3.6 production to MLX `v0.31.2` on this machine; clean
   `84961223` and `2165dc08` are fast when NAX is enabled.
5. Additional Rust release-profile tuning (`lto`, `codegen-units`, `panic`) remains
   a secondary optimization path. It is not the root cause of the qmm/GDN gap.

## Canonical Install Rebuild

Rebuilt `/Users/xin/.local/mlx` from current MLX `2165dc08` using the full Xcode
toolchain and a persistent installed Metal path:

```bash
cmake -S /Users/xin/workspace/iron-rivals/mlx \
  -B /Users/xin/workspace/iron-rivals/mlx/build-nax-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/Users/xin/.local/mlx \
  -DMLX_METAL_PATH=/Users/xin/.local/mlx/lib \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_GGUF=ON \
  -DMLX_BUILD_SAFETENSORS=ON \
  -DMLX_BUILD_PYTHON_STUBS=ON \
  -DMLX_METAL_DEBUG=OFF \
  -DMLX_METAL_JIT=OFF \
  -DMLX_USE_CCACHE=ON
cmake --build /Users/xin/workspace/iron-rivals/mlx/build-nax-release \
  --target install --parallel 10
```

The previous non-NAX install was backed up at:

```text
/tmp/ironmlx-mlx-matrix/local-mlx-non-nax-backup-20260528-221344
```

Post-install verification:

- no `MLX_METAL_NO_NAX` in
  `/Users/xin/workspace/iron-rivals/mlx/build-nax-release/CMakeFiles/mlx.dir/flags.make`
- `METAL_PATH="/Users/xin/.local/mlx/lib/mlx.metallib"`
- `/Users/xin/.local/mlx/lib/mlx.metallib` is 150 MB

Canonical post-rebuild `seq=521` p50:

| Runtime | qkvz C++ loop qmm | qkvz linear+slice | out direct qmm | GDN cache-out-only |
| --- | ---: | ---: | ---: | ---: |
| canonical NAX `/Users/xin/.local/mlx` | 1.328 ms | 0.960 ms | 0.477 ms | 2.162 ms |
| previous local non-NAX | 2.300 ms | 2.297 ms | 0.966 ms | 3.996 ms |

The rebuilt canonical install is now in the clean-build fast band for the full GDN
path.

## Next Tasks

- Qwen3.6 core/serve benchmarks were rerun after the canonical NAX MLX rebuild;
  see `2026-05-28-qwen36-performance-phase7-nax-e2e.md`.
- Use the qlinear/GDN benchmark probes as runtime regression checks for future MLX
  source or toolchain upgrades.
