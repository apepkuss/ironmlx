# DFlash2 Server 与 CLI 支持

## 范围

DFlash2 当前通过独立执行路径提供三类入口：

- `ironmlx generate --dflash2-model-dir ...`：本地文本生成；
- `ironmlx serve --dflash2-model-dir ...`：固定 target 与 draft 的 HTTP 服务。
- IronMLX App：在模型管理中为兼容 target 选择 DFlash2 draft，并重启为独立 actor。

已验收的模型组合是：

| 角色 | Checkpoint |
|---|---|
| Target | `mlx-community/Qwen3.8-27B-4bit` 或 `mlx-community/Qwen3.8-27B-8bit` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` |

当前范围仅包含文本生成。DFlash2 draft 不能作为普通 base model 加载，也不能用于
图片或视频请求。Target 必须使用 affine 4-bit 或 affine 8-bit；其他量化格式和
非量化 target 不会被 App 标记为 DFlash2 兼容。

## 启动示例

```bash
ironmlx serve \
  --model /path/to/Qwen3.8-27B-8bit \
  --model-id mlx-community/Qwen3.8-27B-8bit \
  --dflash2-model-dir /path/to/Qwen3.8-27B-DFlash2 \
  --dflash2-block-size 4 \
  --dflash2-draft-bits 4 \
  --max-sequences 8 \
  --dflash2-tensor-batch-max-width 4 \
  --admission-queue-max 2 \
  --port 8080
```

`--dflash2-block-size` 接受 `2..=8`；`--dflash2-draft-bits` 接受 `0`、`4` 或
`8`，其中 `0` 表示保持 draft BF16。`--max-sequences` 必须大于零。`--model-id`
用于把稳定的公开模型 ID 与本地 target 路径分离；省略时沿用 `--model` 的值。

`--dflash2-tensor-batch-max-width` 是单个 DFlash2 tensor group 的安全上限，接受
正整数。省略时使用当前认证默认值 `4`；设为 `1` 会关闭跨请求 tensor batching，
但不关闭 actor 的请求级并发。实际宽度为
`min(max_sequences, tensor_batch_max_width, 当前就绪且执行形态兼容的请求数)`。
因此该参数只限制一次 tensor 操作合并的行数，不增加活动请求数，也不替代
`--max-sequences`。

## App 接入

App 的模型扫描器把 `DFlash2DraftModel` 识别为辅助 artifact，不把它列为可独立加载的
base model。只有同时满足后端 DFlash2 draft 配置约束，并与 target 的 hidden size、
intermediate size、vocab、上下文长度、target 层数、RMS epsilon 和 RoPE theta 一致的
draft，才会出现在 target 的 DFlash2 选择器中；不完整或不兼容的 artifact fail closed。

Dashboard 提供 DFlash2 开关、兼容 draft 选择、block size、运行时 draft 精度和
Tensor Batch 上限。Tensor Batch 上限留空时不下发显式参数，由后端使用认证默认值
`4`；填写正整数时作为高级安全护栏下发。App 帮助信息同步说明该值与全局
Max Sequences 的最小值关系。
DFlash2 与 MTP、Prompt Lookup 严格互斥。启用后 App 只保留一个 default target，重启
后端并以固定 target/draft actor 提供服务；禁用或修改 DFlash2 参数同样通过受控重启
生效。配置校验、启动或恢复失败时，App 恢复此前的配置与模型参数，再重启原执行路径。

App 的 DFlash2 模式继续公开 `GET /v1/models`，其中只包含稳定的 target ID，供 OpenAI
compatible 客户端发现。该模式不公开动态模型管理 API；Dashboard 与菜单栏从
`/healthz` 和持久化配置恢复 target/draft 状态，不把 draft 暴露为普通模型。

Dashboard 运行态展示 target/draft、block size、draft 精度、TPS、接受率、窗口数、
回滚数、精确残差修正数和峰值内存；`/healthz` 与诊断包同时记录 tensor batch 的
生效上限、实际最大宽度、窗口数、建组数和分歧拆分数。

## 执行与并发语义

DFlash2 server 使用独立 actor，不进入普通 Scheduler、MTP 或 Prompt Lookup
执行路径。每个活动请求独立持有 target cache、draft cache、sampler 和 PRNG 状态。

当 `--max-sequences=1` 时，actor 只推进一个请求；当值大于 1 时，actor 最多维护
对应数量的活动请求。执行 key 相同的就绪请求会在 Tensor Batch 上限内组成一个
`B=N` MLX tensor group；超出上限的请求组成后续 group，并由 actor 轮转推进。
不同约束、采样形态或 cache 状态的请求保持独立；组内接受长度分歧时拆回请求级 cache，
后续满足条件时可重新建组。`batch>1` 的实际收益仍取决于硬件、请求形态和接受率。

活动槽满后，请求进入 `--admission-queue-max` 控制的等待队列。活动槽和队列都满时，
服务返回 HTTP 503、稳定错误码 `scheduler_queue_full` 和 `Retry-After: 5`。流式客户
端断连后，请求在当前 forward 完成后的下一个安全边界释放 cache 与活动槽。

## Sampling

Greedy 和非 Greedy 请求都使用 DFlash2 verify。GreedyVerify 保持与普通 Q=1 解码
逐字节一致；SampledVerify 使用精确 speculative sampling，包括概率接受、拒绝后的
残差采样、bonus token 采样和请求级可复现 PRNG。

公开 sampling 字段仍遵循各协议契约：Chat Completions 与 Responses 接受
`temperature`、`top_p`，Anthropic Messages 额外接受 `top_k`。请求未指定字段时，
使用 checkpoint 的 generation defaults；已验收 Qwen3.8 配置默认提供
`top_k=20`。固定 seed 只承诺同一 IronMLX/MLX、checkpoint、配置和执行形态下的
可复现性，不是跨版本随机序列兼容承诺。

## HTTP 协议

DFlash2 actor 接入以下文本协议的同步与 SSE 路径：

| 协议 | 同步 | SSE | 终止语义 |
|---|:---:|:---:|---|
| `POST /v1/chat/completions` | 是 | 是 | Chat chunk + `[DONE]` |
| `POST /v1/responses` | 是 | 是 | Responses typed lifecycle |
| `POST /v1/messages` | 是 | 是 | Anthropic Messages lifecycle |

HTTP transport、严格字段校验、错误 envelope 与断连语义见
[`api.md`](api.md) 和 [`api-compatibility-matrix.md`](api-compatibility-matrix.md)。

## 隔离约束

当前 DFlash2 路径明确拒绝与以下功能组合：

- MTP；
- Prompt Lookup；
- KV quantization；
- paged/persistent prefix cache；
- active KV offload；
- Scheduler profile 与 Scheduler autotune report。

这些限制用于保持 DFlash2 路径的 cache 所有权、数值正确性与性能资格独立，不会静默
回退到其他 speculative 路径。

## `/healthz`

`GET /healthz` 的 `dflash2` 对象公开配置与累计指标。以下数值仅用于展示字段形状：

```json
{
  "dflash2": {
    "enabled": true,
    "block_size": 4,
    "draft_quantization_bits": 4,
    "requests": 3,
    "windows": 96,
    "drafted_tokens": 384,
    "accepted_draft_tokens": 256,
    "rollback_count": 31,
    "sampled_requests": 1,
    "exact_sampling_windows": 32,
    "exact_acceptance_draws": 128,
    "exact_residual_corrections": 9,
    "exact_bonus_samples": 18,
    "latest_generation_tps": 54.9,
    "latest_acceptance_rate": 0.68,
    "peak_memory_bytes": 21474836480
  }
}
```

`scheduler.b_max`、`scheduler.b_active` 和 `scheduler.b_queued` 分别表示 actor 配置的
活动上限、当前活动请求和排队请求。性能字段用于现场观测，不能脱离硬件、prompt、
接受率和采样配置作为通用性能承诺。

## 验收证据

P3、P3.5、P3.5.1、P3.5.2、P3.6 的最终正确性、性能、并发和三协议验收见
[`benchmarks/dflash2-final-validation/2026-08-23/summary.md`](benchmarks/dflash2-final-validation/2026-08-23/summary.md)。
当前 `dd37fde` 上 Qwen3.8-27B-4bit 的回归吞吐与精确性结果见
[`benchmarks/qwen38-affine4-dflash2/2026-08-29/summary.md`](benchmarks/qwen38-affine4-dflash2/2026-08-29/summary.md)。
P4 的 App matcher、启动参数、恢复、Dashboard 和诊断由 App 全量测试、Rust 严格门禁、
Release Bundle 校验及本机 target/draft HTTP smoke 共同验收。
