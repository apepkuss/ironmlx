# Scheduler/Autotune 设计研究与轻量落地

状态：本阶段轻量落地已完成。worktree `ironmlx-backend-scheduler-autotune`，branch `codex/scheduler-autotune`。

## 0. 结论摘要

本阶段不做自动改参，不改变默认运行行为。先落地一个显式 opt-in 的启动诊断能力：读取当前 CLI 参数、模型 `ModelMeta`、有效上下文上限、KV cache 预算和模型级 fresh-prefill batch limit，输出 scheduler/autotune 建议，作为后续离线校准和真正 `--autotune` 的基础。

原因：

- 当前 scheduler 的性能主要受 `b_max`、`prefill_chunk_size`、`admission_deadline_ms`、`admission_queue_max`、`max_cache_cap`、模型级 `fresh_prefill_batch_limit` 和机器内存共同影响。
- 这些参数在不同硬件、不同模型、不同 agent 长 prompt 负载下没有单一静态最优值。
- 直接自动覆盖用户参数风险较高；先诊断再建议，可以积累真实机器与真实负载数据，避免把一次机器上的经验固化成全局默认。

## 1. 三阶段方向

```mermaid
flowchart TD
    A["阶段 1: 诊断与推荐"] --> B["阶段 2: 离线校准"]
    B --> C["阶段 3: 显式运行时 autotune"]
    A --> D["不改变默认参数"]
    B --> E["生成机器/模型 profile"]
    C --> F["用户 opt-in 后应用 profile"]
```

### 阶段 1：诊断与推荐（本次轻量落地）

- 新增 `--scheduler-autotune-report`。
- 仅在启动时打印报告。
- 报告包含当前参数、KV 预算、模型 fresh-prefill 限制样本和建议。
- 不修改 `serve` 传入 scheduler 的任何参数。

### 阶段 2：离线校准（后续）

- 复用 `iron-bench` 或新增小型校准 runner。
- 对 PP/TG/C 矩阵采样 TTFT、ITL、E2E、吞吐与 queue wait。
- 产出机器/模型 profile，例如推荐 `b_max`、chunk 粒度、admission deadline。

### 阶段 3：显式运行时 autotune（后续）

- 用户传入 `--scheduler-autotune` 或指定 profile 后才应用。
- 应用前打印最终参数和理由。
- 保留手动参数优先级：用户显式指定的参数不被覆盖，除非 Boss 后续要求支持 override 模式。

## 2. 本次范围

本次只实现阶段 1：

- 新建纯函数模块，便于单元测试和后续离线校准复用。
- CLI 新增 opt-in flag。
- `server::serve` 在模型 meta 解析后打印报告。
- 新增中文研究文档和实施计划。

不做：

- 不自动调整 `b_max` / `prefill_chunk_size` / `admission_deadline_ms`。
- 不新增运行期动态控制回路。
- 不改变 `/healthz` JSON schema。
- 不把 GLM-4.7 的模型级经验硬编码为全模型默认。

## 3. 诊断输入

| 输入 | 来源 | 用途 |
|---|---|---|
| `b_max` | CLI | 并发槽数、预算计算、fresh-prefill limit 对比 |
| `prefill_chunk_size` | CLI | 判断长 prompt 是否可 chunked rolling |
| `admission_deadline_ms` | CLI | 判断 admission window 是否可能增加 TTFT |
| `admission_queue_max` | CLI | 判断队列容量和队列禁用 |
| `max_cache_cap` | CLI | 用户请求的单请求 cap |
| `effective_cap_max` | `min(max_cache_cap, model_max_context)` | 实际调度 cap |
| `ModelMeta` | `model.model_meta()` | KV bytes/token、模型上下文、权重估算 |
| total RAM | `system_total_ram_bytes()` | 机器资源诊断 |
| fresh-prefill batch limit | `M::fresh_prefill_batch_limit(pp, b_max)` | 模型级 prefill 策略观测 |

## 4. 推荐策略

### 内存预算

- 计算 `kv_bytes_per_token`。
- 计算 `reserved_kv_bytes = b_max * effective_cap_max * kv_bytes_per_token`。
- 计算 `available_budget_bytes = total_ram - model_weight_bytes - safety_margin`。
- 若 `reserved_kv_bytes > available_budget_bytes`，输出 warning，建议降低 `b_max` 或 `max_cache_cap`。
- 若接近预算上限，输出 warning，提醒长 prompt/并发场景容易触发 runtime admission gate。

### 并发与 agent 长 prompt

- `b_max == 1`：说明当前是单请求优化模式；agent 并发请求会排队，TTFT 可能上升。
- `prefill_chunk_size == 0`：说明禁用 chunking；长 prompt 会形成整段 prefill，占用 decode cadence。
- chunk 很小：提示可能增加 chunk/eval overhead。
- chunk 很大：提示 queued 请求 TTFT 可能变差。

### admission

- `admission_queue_max == 0`：提示队列禁用，饱和时会直接拒绝。
- `admission_deadline_ms` 过大：提示 admission window 可能拉长首批请求 TTFT。

### 模型级 fresh-prefill 限制

报告固定采样 PP=512/1024/2048/8192 下的 `fresh_prefill_batch_limit`。如果某些 PP 下 limit 小于 `b_max`，说明模型已参与降低 fresh-batch prefill 并发，后续调优应把它视为性能路径的一部分。

## 5. 后续研究问题

- agent 常见长 prompt 下，TTFT 与 ITL 哪个应作为主优化目标，需要按产品场景定义权重。
- chunk 粒度应按模型、prompt 长度、机器内存和 decode cadence 综合决定，不能只用固定阈值。
- 离线 profile 的保存格式、版本兼容和手动参数优先级需要单独设计。
- 真正运行时 autotune 是否允许根据 `/healthz` 和队列状态动态调整策略，需要先证明不会引入抖动。
