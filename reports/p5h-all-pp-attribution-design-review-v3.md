# P5h All-PP Attribution Design Review v3

审查对象: `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md`  
分支: `ironmlx-p5h-perf`  
HEAD: `28a3737 docs(p5h): spec v3 — Codex review v2 fixes (5 items: 2 P1 + 2 P2 + 1 P3 + T0 split)`  
日期: 2026-05-21

## 结论

v3 已解决上一轮 review 的主要结构问题: GDN 不再复用旧 schema、coverage gate 改为 residual-based、GatedAttention 不再拆 fused SDPA、MoE taxonomy 与 top-8/sorted path 对齐、T0 也拆成 T0a/T0b hard gate。

但当前 spec 仍有几个会影响 P5h attribution 可执行性的残留问题。建议先修下面的 P1/P2，再进入 implementation plan。

## Findings

### [P1] Root span 结束在 "first SSE chunk write" 会把 prefill 排除在 root 之外

位置:
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:136`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:145`
- `ironmlx/src/core/server/openai.rs:546`
- `ironmlx/src/core/server/openai.rs:562`
- `ironmlx/src/core/server/scheduler_actor.rs:276`
- `ironmlx/src/core/server/scheduler_actor.rs:302`
- `iron-bench/src/client.rs:106`

spec 把 root 定义为 `server_request_recv_to_first_sse_write`, 结束点是 "first SSE chunk write"。但当前 OpenAI scheduler streaming path 会先发送一个 role chunk:

- `serve_via_scheduler_stream` 在收到 `AdmitReply` 后立即 spawn forwarder, forwarder 首先 `tx.send(role_chunk)`。
- scheduler actor 的 `AdmitReply` 在 admission 后发回, first-batch prefill 是之后才执行的。
- `iron-bench` 的 TTFT 不是 role chunk, 而是第一个非空 `delta.content`。

因此如果实现按 spec 字面含义在 first SSE chunk 结束 root, root 很可能在 role chunk 处结束, 而模型 prefill 和 first-token sampling 发生在 root 之外。这样 `model_prefill_forward` 作为 root 子 span 在结构上不成立, `client_transport_residual_us = iron_bench_ttft_us - server_root_inclusive_us` 也会变成一个混入 server-side prefill 的假 transport residual。

建议把 root 明确定义为:

`server_request_recv_to_first_content_sse_write`: 从 OpenAI/Anthropic request handler entry 到第一个非空 content SSE chunk 被发送到 body channel。

同时要求实现把 root context/request_id 从 handler 传入 scheduler forwarder task, 在 `event_rx.recv()` 得到 first token、detok 产生非空 content、并执行第一个 content `tx.send(...)` 时关闭 root。role chunk 可以作为 root 内的 `sse_write_role_chunk` 子 span, 但不能作为 root 的结束点。

### [P2] `[p5h-profile]` schema 要求 server emit `pp/run_id`, 但 spec 又把 client metadata propagation 排除在 P5h 外

位置:
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:123`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:124`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:125`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:134`
- `ironmlx/src/core/generate.rs:26`
- `ironmlx/src/core/server/openai.rs:370`
- `iron-bench/src/runner.rs:65`
- `iron-bench/src/runner.rs:80`
- `iron-bench/src/runner.rs:97`

§ 2.5a 规定每条 `[p5h-profile]` record 必须包含 `request_id/run_id/pp`, 其中 `pp` 是 iron-bench `--prompt-len`, `run_id` 是 warmup/measured run index。但同一段又说 iron-bench 到 ironmlx 的 request-id propagation out of scope。当前 server 侧 `GenerateRequest` 只有 tokenized prompt、max_new_tokens、sampler/cache/VL 字段, 没有 benchmark metadata; `iron-bench` 的 `pp` 和 `run_idx` 也只存在 client runner/report 侧。

这会让 T0a 实现者面临两种不兼容解释:

1. server 必须 emit `pp/run_id`, 但没有可靠来源。
2. aggregator 后处理补 `pp/run_id`, 但这违反 "每个 `[p5h-profile]` record 必须含 schema fields" 的文字要求。

建议二选一写死:

- 方案 A: P5h 增加 feature-gated benchmark trace metadata, iron-bench 发送 `x-ironmlx-p5h-pp` / `x-ironmlx-p5h-run-id` / `x-ironmlx-p5h-request-id`, server 只在 `p5h-profile` feature 下读取并写入 span。
- 方案 B: server span schema 只包含 `request_id`、server-computed `prompt_tokens/seq`、span tree 字段; T5 aggregator 按每个 sweep cell 的 log capture 边界和 client CSV 注入 `pp/run_idx`。这种方案需要把 § 2.5a 的必填字段改成 "aggregated record fields", 而不是 "server log line fields"。

### [P2] T5 task 仍保留旧的 trivial coverage formula

位置:
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:332`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:398`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:421`

§ 7.1 和 § 7.2 已正确使用 residual-based gate:

`coverage_pct = 1 - Σ unattributed_*.inclusive_us / root.inclusive_us`

但 T5 task 仍写:

`coverage_pct = Σ span.exclusive_us / root_span.inclusive_us ≥ 95%`

这正是 v2 已经修过的 tree identity 问题。由于 T5 是执行任务清单, 实现者很可能照 T5 bullet 写 aggregator/report, 让 coverage gate 重新退化成永远接近 100% 的无效门禁。

建议把 T5 bullet 改成与 § 7.1 完全一致, 并明确 `Σ span.exclusive_us / root` 只能作为 sum-to-root sanity check, 不能作为 coverage gate。

### [P3] Rust validation 文案只覆盖 T1-T4, 但 T0a/T0b 也会改 Rust

位置:
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:230`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:231`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:233`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:343`
- `docs/superpowers/specs/2026-05-20-ironmlx-p5h-all-pp-attribution-design.md:428`

T0a 明确包含 Cargo feature、Rust span tracker、GDN log emission 等 Rust 改动, T0b 也可能新增 ablation/profile mode。§ 7.2 gate 说 `T0a/T0b/T1-T5 each independently green`, 但 § 4 的 Rust 检测命令只写 "每个 T1-T4 instrumentation task 完成时"。

建议把 § 4 改为: 任一 task 触碰 Rust 代码时都必须执行 AGENTS/CLAUDE Rust 检测命令; 至少 T0a/T0b/T1-T4 必须独立 green, T5 若只改 Python/Markdown 则执行对应脚本测试和 markdown/link/schema 检查。

## 已确认修复项

- GDN 旧 `[p5g-profile]` 数据只作为 prior reference, 不进入 P5h coverage gate。
- GDN T0a rerun now uses new `[p5h-profile]` schema and UMA cold/warm pair。
- coverage gate 主体已从 naive sum medians 改为 explicit residual leaves。
- GatedAttention taxonomy 已把 q gate 纳入 `q_gate_k_v_proj`, 并正确承认 fused SDPA 内部不可拆。
- MoE taxonomy 已更新到 256 experts/top-8/sorted-routing path。
- T0 已拆成 T0a hard gate + T0b Phase D root-cause checkpoint。
