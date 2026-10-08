# 从源码构建

[English](../building-from-source.md)

## 支持平台

IronMLX 0.2.0 仅支持 Apple Silicon arm64 与 macOS 26.4 或更高版本。
Intel Mac 和更早的 macOS 版本不在支持范围内。

## 当前分发状态

RC/稳定版工作流支持 Developer ID 签名、公证和票据附加。普通本地源码构建仍为 ad-hoc，不能等同于正式分发包。可下载版本以 GitHub Releases 实际公开结果为准；启用分发门禁不代表已公开发布。

## 模型权利边界

IronMLX 可以搜索和下载模型，但不拥有或重新授权模型权利。用户必须在上游模型
页面查阅许可证、gated access、用途和再分发限制，并自行确保使用合规。App、DMG
和 ZIP 不包含模型权重；完整边界说明见[模型权利边界](model-license-boundary.md)。

## 从源码构建 App

构建机需要完整 Xcode、CMake、Rust 1.94、`cargo-about 0.9.1`，以及可用的
macOS 26.4 SDK/Metal 工具链。以下命令会检出项目锁定的 MLX commit，并生成
自包含 Release App：

```bash
cargo install --locked --features cli --version 0.9.1 cargo-about
scripts/checkout-release-mlx.sh /tmp/ironmlx-mlx-source
MLX_SRC=/tmp/ironmlx-mlx-source scripts/build-app-bundle.sh
```

构建器会拒绝 dirty 或 commit 不匹配的 MLX checkout。成功后运行：

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
open dist/IronMLX.app
```

本地构建使用 ad-hoc 签名，未经 Developer ID 签名与 Apple 公证，不能作为正式
安装包对外分发。不要绕过 macOS 安全机制运行来源不明的构建。

## CLI 开发构建

若只调试后端，需要先准备项目锁定且启用 NAX Metal kernels 的 MLX Release
安装，然后导出 `MLX_DIR` 与 `MLX_METAL_PATH`：

```bash
export MLX_DIR=/path/to/validated/mlx-install
export MLX_METAL_PATH="$MLX_DIR/lib"
cargo build --release --bin ironmlx --bin iron-bench
target/release/ironmlx --version
```

## 构建并安装 MLX 依赖

MLX C++ 依赖必须以静态 arm64 库构建，并启用 Metal kernels。仓库提供的
辅助脚本会完成构建、安装、补齐传递库，并生成可 `source` 的环境文件：

```bash
MLX_SRC=/path/to/mlx-source \
MLX_PREFIX="$HOME/.local/mlx" \
scripts/setup-mlx.sh
source "$HOME/.local/mlx/mlx-env.sh"
```

如需手动构建，请使用相同的部署目标和静态库选项：

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

MLX 的安装步骤不会导出私有的 GGUF 传递库。必须将它复制到安装前缀，
这样 `mlx` crate 的 GGUF 测试及其他 GGUF 使用方才能正确链接：

```bash
cp "$MLX_SRC/build/mlx/io/libgguflib.a" "$MLX_PREFIX/lib/"
```

生产 `ironmlx` 二进制不使用 GGUF 权重，但缺少该库会导致 GGUF 相关测试
因 `_gguf_*` 符号未定义而失败。`mlx-sys/build.rs` 会自动链接
`MLX_DIR/lib` 下的所有 `lib*.a`，无需额外链接器参数。

## MLX 编译期与运行期环境

每个编译或运行 IronMLX 的 shell、CI 作业和工具调用都必须显式设置 MLX
路径：

```bash
export MLX_ROOT="$HOME/.local/mlx"
export MLX_DIR="$MLX_ROOT"
export MLX_METAL_PATH="$MLX_ROOT/lib"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export MACOSX_DEPLOYMENT_TARGET=26.4
export CMAKE_OSX_DEPLOYMENT_TARGET=26.4
```

虽然 MLX 以静态方式链接，但运行期仍会加载 `mlx.metallib`，因此
`MLX_METAL_PATH` 必须指向该文件所在目录。

## MLX 安装完整性检查

构建 IronMLX 前，请确认头文件、静态库和 Metal kernel 库都存在：

```bash
test -f "$MLX_DIR/include/mlx/array.h"
test -f "$MLX_DIR/lib/libmlx.a"
test -f "$MLX_DIR/lib/libgguflib.a"
test -f "$MLX_DIR/lib/mlx.metallib"
```

## 后端测试与本地服务

导出上述环境后，可运行完整 workspace 测试：

```bash
cargo build --release
cargo test --all-features --workspace
```

运行本地文本生成 smoke test：

```bash
MODEL="$HOME/.ironmlx/models/<org>/<model>"
./target/release/ironmlx generate \
  --model "$MODEL" \
  --prompt "请用一句话介绍 MoE 架构。" \
  --max-tokens 128 \
  --temperature 0 \
  --prefill-chunk-size 2048
```

启动本地服务：

```bash
./target/release/ironmlx serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --prefill-chunk-size 2048 \
  --b-max 1 \
  --max-cache-cap 32768
```

### 后端单实例约束

同一 macOS 用户只能运行一个 `ironmlx serve` 后端，不同 App、CLI 参数或监听端口
也不能绕过该约束。后端会在初始化 MLX、加载 metallib 或模型之前，对
`~/.ironmlx/run/backend.lock` 获取非阻塞独占文件锁，并持有锁文件描述符直到进程
退出。正常退出、崩溃或 `SIGKILL` 都由系统自动释放锁；锁文件本身可以保留，不应作为
进程是否存活的判断依据。

第二个后端会立即退出，并在标准错误输出稳定错误码
`ironmlx_instance_already_running`。IronMLX App 会停止自动恢复循环，并提示用户先
退出已有实例。

## 验证修改

先运行与改动相关的检查；提交 Pull Request 前运行完整 workspace 门禁：

```bash
cargo fmt --all -- --check
cargo +nightly fmt --all -- --check
cargo +nightly clippy --locked --all-features --workspace -- -D warnings
cargo build --locked --release
cargo test --locked --all-features --workspace -- --test-threads=1
swift test --package-path ironmlx-app --configuration release --no-parallel
```

构建 App Bundle 后还应运行：

```bash
scripts/verify-app-bundle.sh dist/IronMLX.app
scripts/verify-model-distribution-boundary.sh dist/IronMLX.app
```

如果改动影响真实模型、协议、流式输出或 App 运行时，fixture、ignored test 或仅源码检查不能替代真实验收。

发布验收必须分别记录源码测试、Bundle 静态检查、签名与公证产物、Gatekeeper 检查和公开分发状态。

### SDK 兼容验证

| SDK | 固定版本 | 覆盖范围 |
| --- | --- | --- |
| OpenAI Python | `2.48.0` | Chat / Responses, SSE, tools, Structured Outputs, reasoning, 400/413/503 |
| Anthropic Python | `0.121.0` | Messages, SSE, tools, Structured Outputs + thinking, 400/413/503 |

固定 SDK 通过真实 loopback HTTP/SSE 访问 fixture server，验证客户端解析；Rust 测试独立验证生产请求和响应契约。两者不加载模型，不证明回答质量、工具选择准确率或性能。
在仓库根目录执行：

```bash
python3 -m venv /tmp/ironmlx-api-contract-sdk
/tmp/ironmlx-api-contract-sdk/bin/python -m pip install -r scripts/api-contract-sdk/requirements.txt
/tmp/ironmlx-api-contract-sdk/bin/python scripts/api-contract-sdk/contract.py --fixture
cargo test --locked --all-features -p ironmlx --lib server::
```

## 运行时开发说明

### App 日志设置应用语义

App 先读取后端级别，携带进程 ID 和 revision 更新过滤器，再仅保存 `log_level` 并调整自身过滤器。
后端停止时不发送请求，保存值在下次启动时应用；启动、恢复、停止或其他设置应用过程中拒绝变更。
响应丢失或保存失败时查询实际状态，在前提仍匹配时恢复旧级别；进程或 revision 改变、回退未确认时不会显示为成功。
旧配置 TRACE/WARN 兼容为 ALL/WARNING；第三方依赖日志最多为 WARNING，选择 ERROR 时为 ERROR。
App 启动 helper 时传入 `IRONMLX_LOG_LEVEL`，独立 CLI 仍可使用 `RUST_LOG`。

### 原生工具模板

MiniCPM-V 4.6 与 MiniCPM5 使用不同 XML 工具协议；MiniCPM5 对包含 <、& 或换行的字符串参数使用 CDATA。Gemma 内部将动态 object 投影为确定性键值条目，响应时恢复原对象；公开 Schema 和参数形状保持不变。

### 运行拓扑

普通因果模型的全部 HTTP 请求统一使用 SchedulerActor，包括长上下文 chunked-prefill、
多模态、sampling 与约束解码请求；请求形态不再选择直接 GenerationStream 服务路径。
公开请求优先级见
[文本与视觉 API](text-vision-api.md#请求优先级)。

### 流式资源释放

Chat Completions、Responses 和 Anthropic Messages 的流式请求在 SSE 响应开始后
支持客户端断连取消。HTTP response body 被丢弃时，transport 会立即发布协议无关的
断连信号；各协议的流式编码器停止消费生成事件，也不会在已观测到断连后继续构造
协议终止事件。

| 生成路径 | 取消生效点 | 释放内容 |
|---|---|---|
| Scheduler（包括普通因果、MTP 与辅助 drafter 服务） | 当前模型 forward 结束后的下一次安全调度边界 | 活跃请求、调度槽、KV cache 与内存预算 |
| DFlash2 actor | 当前 target/draft forward 结束后的下一次安全事件边界 | 请求级 target/draft cache、活动槽与内存预算 |
| DiffusionGemma | 当前 block-diffusion 步骤结束后的下一次事件边界 | generation lane 与请求状态 |

取消不会强行中断正在执行的 Metal forward；这是为了避免在设备工作未完成时破坏模型
和 KV 状态。因此，从 TCP 断开到资源归还可能包含一个当前 forward/扩散步骤的尾延迟。
本契约只承诺已经开始返回 SSE 的流式请求；v0.1 不承诺非流式 HTTP 请求在客户端断开
后取消底层生成。

## MLX 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `MLX_DIR is not set` | 当前 shell 未导出构建路径。 | `source mlx-env.sh`，或设置上面的环境变量。 |
| `missing include/ or lib/` | `MLX_DIR` 指向 MLX build 目录，而非安装前缀。 | 将其指向 `MLX_PREFIX`。 |
| `Undefined symbols: _gguf_*` | `libgguflib.a` 未复制到安装前缀。 | 重复上面的复制步骤，或重新运行 `scripts/setup-mlx.sh`。 |
| `Failed to load the default metallib` | `MLX_METAL_PATH` 未设置或目录错误。 | 将其设置为包含 `mlx.metallib` 的目录。 |
