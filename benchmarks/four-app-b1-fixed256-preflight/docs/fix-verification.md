# IronMLX 固定 256 token 问题：根因、修复与验证

2026-10-08。工作目录 `/Users/xin/workspace/perf-dflash2-b4-throughput`，分支 `perf/dflash2-b4-throughput`，修复前的 HEAD 为 183ccea1。

**最终候选二进制：** `reports/four-app-b1-fixed256-preflight/fix/binaries/ironmlx-fixed256-fix-8271e413`
- sha256 8271e41393de7ff4ba7d7497b87f69239b539201899bf3fef9d0987f009b0ef6；
- 源码补丁为同名 `.source.diff`（sha256 f29ff2d7…）。

**早期候选：** 84b82b06，用于 `fixed256-fix-v1`，保留但不代表最终源码。它之后还补了工具路径的拆分和测试，二进制因此变化。

## 根因

### 1. DFlash2 输出尾部的 KV 容量不足

- target KV 缓存按“prompt + max_new_tokens”分配，floor 为 256。
- M5 profile 的 flat-tree（tf-v1）验证一次要临时写入“当前 token + 15 个树节点”，共 16 个位置。接近输出上限时，草稿块会按剩余预算截短，但树验证仍写满 16 个位置。
- 每个窗口开始时，已提交位置最多可达 prompt + max_new − 1，所以最后几个窗口会超出容量。
  - 例：`fixed256-v1` 的 code-1，容量 62 + 256 = 318，提交到 305 后需要写到 305 + 16 = 321。
  - 六个目标请求都因 `KVCache cap … exceeded` 失败。

### 2. OpenAI 接口吞掉了生成失败

- DFlash2 actor 失败时只记日志，然后丢掉事件发送端。
- `serve_via_scheduler_stream` 把“通道关闭但没收到 finish_reason”当作正常结束，照常发出 usage 和 `[DONE]`。
- 工具流式和工具非流式路径更进一步，把结束原因默认成 `"stop"`，等于伪造了结束原因。
- Anthropic、Responses 和普通非流式路径原本就会报错。

## 修复（提交 1：`fix: handle DFlash2 tail capacity and generation errors`）

**`ironmlx-runtime/src/core/dflash2.rs`**
- 新增两个函数：
  - `dflash2_verify_scratch_tokens = max(block_size, tree_max_nodes + 1)`；
  - `dflash2_target_cache_tokens = prompt + max_new_tokens + verify_scratch`。
- 单请求、批量 prefill、张量批、ragged 批四处容量计算都改用它；单请求路径改为带检查的转换。
- 逻辑上下文上限（`RequestTooLarge`）、剩余预算、提交长度和 finish_reason 的逻辑都不变。被拒绝的临时位置照旧回滚。

**`ironmlx-runtime/src/core/dflash2_actor.rs`**
- 两处准入显存预算改为按物理容量收取。

**`ironmlx/src/server/openai.rs`**
- 普通流式、工具流式、工具非流式三条路径：没收到终止事件就按生成失败处理。
  - 流式：在客户端仍连接时发送 `{"error":{"message":"scheduler stream ended before a terminal event","type":"internal_error"}}`，然后结束，不补 finish_reason、usage 或 `[DONE]`；
  - 非流式：返回 5xx。
- 客户端主动断开时直接结束。
- 准入之后的处理拆为四个独立函数，供测试直接注入提前关闭的事件通道：`stream_admitted`、`collect_admitted`、`stream_tools_admitted`、`collect_tools_admitted`。

**`ironmlx/src/server/responses.rs`、`responses_eof_tests.rs`**
- EOF 测试模块和它的 `state()` 改为 crate 内可见，只在测试编译时生效。

## 针对性测试

原始命令、完整输出和退出码都在 `reports/four-app-b1-fixed256-preflight/fix/tests/`，每项有 `<名称>.command.txt`、`.output.log`、`.exit_code` 三个文件。

| 名称 | 内容 | 退出码 / 结果 |
| --- | --- | --- |
| `unit-target-cache-capacity` | 容量单元测试：多种 block、tree、prompt、max_new 组合下，最坏提交位置加窗口宽度不超过容量；复现原失败算式 305 + 16 > 318 | 0，通过 |
| `openai-missing-terminal-event` | OpenAI 的 7 个测试：普通流式、工具流式、普通非流式、工具非流式在中途关闭时都报错，没有伪造 stop、length 或 `[DONE]`；普通流式、工具流式、工具非流式在 length 终止时正常收尾 | 0，7 个通过 |
| `mutation-openai-plain-stream-disabled` | 关闭普通流式的修复 | 101，只有普通流式失败测试失败，符合预期 |
| `mutation-openai-tool-paths-disabled` | 关闭工具流式和工具非流式的修复 | 101，两个工具失败测试都失败，符合预期 |
| `real-model-tail-m5` | 真实模型尾部测试：安装 M5 profile 并断言 tf-v1 生效；block 8、tree 15、max_new 256、无停止 token；要求 256 个 token、以 length 结束、rollback_count > 0、与 linear 路径逐个一致 | 0，通过 |
| `mutation-real-model-tail-m5-no-scratch` | 上一项加变异：容量中去掉验证临时空间 | 101，`KVCache cap 260 exceeded on row 0: offset 249 + new 16 = 265`，符合预期 |
| `negative-real-model-tail-generic` | 首次版本的测试（不安装 M5 profile，走通用 tree 路径），修复代码 | 0，通过 |
| `negative-mutation-real-model-tail-generic-no-scratch` | 首次版本加同样的变异 | 0，**仍然通过**，即负面结果：首次测试没有覆盖 M5 flat-tree 路径 |

**负面结果的来历：**
- 首次真实模型测试没有安装 M5 profile。那次运行和它的变异检查当时只在会话中显示了过滤后的几行，没有保存完整输出，无法恢复。
- 上表的两个 `negative-*` 是最小化重跑：临时去掉 M5 安装，运行后恢复源码，恢复后哈希与重跑前一致（298b46c1…）。
- 所有变异改动都已恢复；源码中没有 `0 * dflash2_verify`、`if false` 或 `unwrap_or("stop")` 残留。

## 检查（最终源码，`reports/four-app-b1-fixed256-preflight/fix/checks-final/`，每项附命令和退出码）

| 检查 | 退出码 |
| --- | --- |
| `cargo fmt` | 0 |
| `cargo +nightly fmt --all -- --check` | 0 |
| `cargo +nightly clippy --all-features --workspace -- -D warnings` | 0 |
| `cargo build --release` | 0，生成最终候选 8271e413 |
| `cargo test --release -p ironmlx-runtime --lib`（首次） | 101：543 通过，1 失败（`core::scheduler::tests::mtp_prefill_vl_uses_paged_ssd_prefix_cache_on_exact_hit`） |
| `cargo test --release -p ironmlx --lib`（首次） | 101：334 通过，1 失败（`cli::backend_instance_lock::tests::rejects_a_second_backend_and_releases_on_drop`，报 “lock after release”） |
| 两个失败用例单独重跑 | 都是 0 |
| `cargo test --release -p ironmlx-runtime --lib`（重跑） | 0：544 通过，8 ignored |
| `cargo test --release -p ironmlx --lib`（重跑） | 0：335 通过，2 ignored |

**两个失败用例：**
- 它们所在的文件（`scheduler.rs`、`backend_instance_lock.rs`）本次没有改动；上一轮完整运行时它们都通过了，这次单独重跑和完整重跑也都通过。
- 记为间歇性失败，首次失败的日志保留。原因没有调查。

早期候选 84b82b06 的检查日志在 `fix/checks/`。

## 针对性验证

条件与 `fixed256-v1` 相同：只跑 IronMLX；模型、M5 profile（block 8、tree 15）、max_tokens=256、预热和六个任务都不变；没有关闭 tree，也没有提高上限。

**判定标准（脚本 revision 3）：**
- 协议有效：HTTP 200、无错误、`[DONE]`、无 reasoning，finish_reason 为 stop 或 length；
- 固定 256 合格（只针对目标请求）：在协议有效的基础上，finish_reason 为 length，原生和统一计数都是 256；
- 两个探针各按自己的预期判断。

**`fixed256-fix-v2`（最终候选 8271e413，revision 3 脚本）：**

| 请求 | 原生 | 统一 tokenizer | finish_reason | 错误 | 缓存命中 | 协议有效 | 固定 256 合格 / 探针符合预期 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code-1 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| code-2 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| code-3 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| knowledge-1 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| knowledge-2 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| knowledge-3 | 256 | 256 | length | 无 | 0 | 是 | 合格 |
| 探针“只回答 OK”（默认） | 2 | 1 | stop | 无 | 0 | 是 | 符合（允许提前 EOS） |
| 探针“只回答 OK”，ignore_eos=true | 256 | 120 | length | 无 | 0 | 是 | 符合 |

**说明：**
- 服务端日志没有 ERROR 或 `exceeded`；前缀缓存关闭，`/healthz` 命中为 0。
- ignore_eos 探针忽略 EOS 后，模型反复生成 `<|im_end|>`、`<|im_start|>` 等特殊 token。原生 256 包含这些 token，可见文本重新分词为 120。这是预期行为，不是计数错误。
- `fixed256-fix-v1`（早期候选 84b82b06，revision 2 脚本）的结果相同：六个目标都是 256/256/length。
- 按 revision 3 离线判读三次运行（`evidence/strict-reassessment-r3.json`）：
  - `fixed256-v1`：IronMLX 协议有效 0/6、固定 256 合格 0/6；oMLX、Splash、TensorFold 都是 6/6；
  - `fixed256-fix-v1`、`fixed256-fix-v2`：IronMLX 都是 6/6，探针都符合预期；
  - revision 2 的判读保留为 `evidence/strict-reassessment.json`。

## 尚未覆盖的范围

- 修复改变了所有 DFlash2 请求的 KV 分配，每个请求多出验证窗口宽度（最多 16 个 token）。本次没有重跑 B1/B4 性能，也没有测内存。
- 修复后的四应用对比没有重跑，其他三个应用也没有重跑。
- ragged 批和张量批这两条批量路径上的容量改动，只经过编译、库测试和 clippy；B4 下的真实请求没有验证。
