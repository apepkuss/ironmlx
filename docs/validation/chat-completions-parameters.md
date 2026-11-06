# Chat Completions 参数兼容验证

日期：2026-11-06（Asia/Tokyo）。基线为本地 `dev` 的 `04455928a`，工作分支为 `fix/chat-completions-parameters`，独立 worktree 为 `/Users/xin/workspace/fix-chat-completions-parameters`。

## 实现范围

- Chat 输出预算依次采用 `max_completion_tokens`、`max_tokens`、所选模型的 `default_max_output_tokens`、256。前两项为省略或 null 时才继续回退；两个显式值都须为正整数。优先级是 IronMLX 的兼容契约。
- 模型默认值复用 App 的 **MAX OUTPUT TOKENS**，在 App／CLI 多模型服务解析目标模型后应用。固定单模型 CLI 没有该模型默认配置时回退到 256。请求预算不修改 **MAX CONTEXT TOKENS**，输入加输出超限仍返回 413。
- `frequency_penalty` 与 `presence_penalty` 只来自当前请求，要求 [-2,2] 内有限数；省略、null、零均不启用，不新增持久模型参数或前端控件。DiffusionGemma 不具备此能力，非零值明确返回 400。
- DFlash2 实测发现原贪心惩罚路径直接以 F32 读取 BF16 logits。频次／出现惩罚现在仅根据词表大小统计历史 token，不读取 logits；重复惩罚在主机读取前显式转换为 F32。新增 F32/F16/BF16 回归，并扩展 DFlash2 确定性验证覆盖两个惩罚参数。
- 更新中英文 API 文档、五种界面语言的帮助说明、固定 SDK 请求 fixture。顺带校正文档对模型发现接口 `max_output_tokens` 的描述，使其符合已有代码的建议预算规则（配置值或 4096，受上下文一半限制），没有修改模型发现实现。

## 环境与模型

主机 Apple M5 Max、128 GiB，macOS 26.4。Rust 1.94.0；nightly 为 1.97.0-nightly（507271bc1，2026-05-17）。MLX 安装目录 `/Users/xin/workspace/ironmlx-backend/.build/app-bundle/mlx-install`，头文件版本 0.32.2；`libmlx.a` SHA-256 为 `c2c33bb73d8e707e7f23997a4010ec402236a638b31616d17662f9d3b895a8d0`。

仅使用现有 `~/.ironmlx/models/huggingface` 模型，没有下载模型。大小为顶层 safetensors 文件总量：

| 来源 | 本地 snapshot revision | 大小 | 用途 |
| --- | --- | --- | --- |
| mlx-community/Llama-3.2-1B-Instruct-4bit | `08231374eeacb049a0eade7922910865b8fce912` | 0.648 GiB | 路径检查、模型路由与默认预算、结构化输出、真实 SDK |
| mlx-community/Qwen3.8-27B-4bit | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | 14.952 GiB | 普通、MTP、DFlash2 共用目标模型 |
| mlx-community/Qwen3.8-27B-MTP-4bit | `b643c01b6d3b094e325edb6ebd832e16c486c575` | 0.223 GiB | MTP 草稿，draft tokens=2 |
| z-lab/Qwen3.8-27B-DFlash2 | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | 3.584 GiB | DFlash2 草稿，运行时 4bit、block size=8 |

真实 HTTP 验证使用 `max_cache_cap=4096`、`max_sequences=2`、`prefill_chunk_size=0`、`m5_dflash2_profile=off`。固定服务的生成测试使用 `ignore_eos=true`，包括 temperature=0 与 temperature=0.7/seed=42；这用于确定预算终止行为，不是性能测试。模型默认值测试注册预算 12、20、未配置的三个 Llama 别名，覆盖显式模型和默认模型路由。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| `cargo fmt`、`cargo +nightly fmt --all -- --check`、`git diff --check` | 通过 |
| `cargo +nightly clippy --all-features --workspace -- -D warnings` | 通过 |
| `cargo build --release` | 通过 |
| `cargo test --release --all-features --workspace -- --test-threads=1` | 2468 通过，0 失败，285 按声明跳过 |
| CI 默认特性矩阵：core / image / lm / runtime，各自 `--no-default-features` | 分别 108 / 12 / 693 / 658 通过，0 失败 |
| `swift test --package-path ironmlx-app --no-parallel` | 562 通过 |
| 固定官方 SDK 契约 | OpenAI 2.48.0、Anthropic 0.121.0，15 个请求通过 |
| 真实模型 HTTP | Llama 的路由／默认值及固定服务通过；Qwen3.8-27B 普通／MTP／DFlash2 均通过 |
| 真实官方 SDK | OpenAI 2.48.0 严格响应校验，7 个请求通过（3 非流式、1 流式、2 个 400、1 个 413） |
| 真实结构化输出与截断／流式 | 通过，验证新预算优先于旧字段，并携带非零惩罚 |
| 文档结构与网站文档测试 | 通过；网站文档测试 14 项 |

真实 HTTP 检查同时验证两个字段的优先级、null／缺省、非法值 400、输出加输入超限 413、正负惩罚并发、零值／缺省请求不受前一次请求影响，以及 temperature=0.7 的随机采样。MTP／DFlash2 健康指标确认实际生成并接受草稿 token；这不代表每个请求均使用推测解码，也不构成加速或性能保证。

普通 Release 二进制 SHA-256：`3241ac63d59b924fe932687e930435f41b87b951c9afc4318f83ecb1f85838f9`。此前完整 CLI 模型矩阵使用 Cargo 集成测试构建的二进制（额外启用 metallib 测试初始化支持），SHA-256 为 `18dd20cbafb92c1ec7383d32cc0f8b42e22f2f817983d473a6e93b1c6532121c`。最终普通 Release 构建成功；重复启动该 CLI 时遇到另一个工作区正在运行的正式实例，按现有单实例锁规则退出，未停止或修改该实例。因此不声称普通 Release 二进制已经独立复跑整个模型矩阵。真实 SDK 检查通过后端库直接启动的隔离 HTTP 测试服务执行。

285 项跳过不计作通过；本次相关真实模型测试已单独显式运行。编译日志包含 MLX 头文件 unused-parameter 与测试辅助函数 dead-code 告警，未阻止检查；Rust Clippy 的 `-D warnings` 检查通过。

原始日志与结果保存在工作区 `.build/validation/chat-completions-parameters/`：`test-summary.json`、`source-manifest.json`、`binary.json`、`live-health.json`、`sdk-live.json` 与 `logs/`。普通 Release 副本保存在同目录的 `ironmlx`。这些构建／证据产物不加入 Git；本报告保留复现入口与检查结论。

## 复现入口

```bash
export MLX_DIR=/Users/xin/workspace/ironmlx-backend/.build/app-bundle/mlx-install
cargo fmt
cargo +nightly fmt --all -- --check
cargo +nightly clippy --all-features --workspace -- -D warnings
cargo build --release
IRONMLX_TOTAL_RAM_BYTES=68719476736 cargo test --release --all-features --workspace -- --test-threads=1
swift test --package-path ironmlx-app --no-parallel
```

Metal 测试前参照 `.github/workflows/ci.yml` 的 metallib 准备步骤，将 `MLX_DIR/lib/mlx.metallib` 放入各测试 crate 的查找位置；本次使用的临时链接已在收尾删除。

真实模型检查使用 `ironmlx/tests/chat_parameters_http.rs`：设置 `CHAT_PARAMETERS_MODEL` 为模型 snapshot 完整路径，`CHAT_PARAMETERS_BINARY` 可固定普通 Release 二进制，然后执行：

```bash
cargo test --release -p ironmlx --test chat_parameters_http -- --ignored --test-threads=1 --nocapture
```

为固定普通 Release 构建，先将 `target/release/ironmlx` 复制到 Cargo 输出目录以外，再将 `CHAT_PARAMETERS_BINARY` 指向副本；Cargo 集成测试自身会重新构建带测试支持的 CLI。该命令的模型默认值测试使用小模型。27B 只运行 `chat_fixed_serving` 筛选；设置 `CHAT_PARAMETERS_MTP_MODEL` 或 `CHAT_PARAMETERS_DFLASH2_MODEL` 分别验证推测解码，不能同时设置。真实结构化输出检查使用 `LLAMA32_MODEL` 和 `--test chat_structured_outputs_http -- --ignored --test-threads=1`。设置 `CHAT_PARAMETERS_SDK_PYTHON` 为安装了 `scripts/api-contract-sdk/requirements.txt` 的 Python 路径时，该测试还会在同一隔离 HTTP 服务上执行 `scripts/api-contract-sdk/live_chat.py`，覆盖 SDK 非流式、流式、null／默认值、双字段优先级以及 400／413。这个测试不经过 CLI 的全局单实例锁。

## 验证边界

M1 Max 没有实机复测；本次结论来自上述 M5 Max 配置。未做吞吐／延迟基准、签名打包、安装或发布。原有未知字段拒绝策略保留，只新增本报告列出的三个字段；DiffusionGemma 的非零惩罚属于明确不支持的能力。测试服务由测试夹具负责终止，不作为 App 退出／析构验收。本次改动在独立功能分支交付，保留 worktree 供审查。
