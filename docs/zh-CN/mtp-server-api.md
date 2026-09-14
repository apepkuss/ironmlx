# Qwen MTP 支持

[English](../mtp-server-api.md)

面向配置 Qwen MTP 的 CLI/API 集成者。需要兼容主模型和匹配的 MTP head；启动开关不代表每个请求都会使用 MTP。

## 入口

- `ironmlx generate --mtp-model-dir ...`：CLI 贪心生成。
- `ironmlx serve --mtp-model-dir ...`：启动时启用 MTP；OpenAI/Anthropic 请求不接受逐请求 MTP 参数。

## 模型范围

| 主模型 | 辅助模型要求 |
| --- | --- |
| Qwen3.5 / 3.6 / 3.8 Dense | 匹配的 Qwen MTP head |
| Qwen3.5 / 3.6 MoE | 匹配的 Qwen MoE MTP head |

兼容组合支持文本/VL CLI 和 OpenAI/Anthropic 服务，仍须满足下列请求条件。
本页描述 Qwen MTP 路径。同一个 `--mtp-model-dir` CLI 参数也接受兼容的 Gemma4 assistant drafter，但使用不同执行路径和采样规则；下文 Qwen 限制不应套用于 Gemma4，见[支持模型](supported-models.md)。

## 请求与缓存限制

- CLI 文本和视觉贪心请求均可使用 MTP；视觉请求在视觉 token 替换后，使用文本主干隐藏状态驱动 draft。
- 服务支持 `--b-max N`，N 至少为 1。仅符合调度器条件的贪心请求使用 MTP；非贪心等请求回退到普通调度路径，仍正常返回。
- `--paged-prefix-cache-dir` 可与 MTP 组合，重复的兼容文本/视觉请求同时恢复主模型和 draft 缓存。不带值时目录为 `~/.ironmlx/cache/paged_prefix_cache`。
- `--mtp-draft-tokens` 是启动配置，省略时按模型选择默认 draft 深度。

## `/healthz` 字段

启用 MTP 时包含：

```json
{
  "mtp": {
    "enabled": true,
    "draft_tokens": 2,
    "prefill_count": 7,
    "step_count": 42,
    "fallback_prefill_count": 1,
    "drafted_tokens": 84,
    "accepted_draft_tokens": 63
  }
}
```

`enabled` 表示启动时启用了 MTP head；`draft_tokens` 是配置的 draft 预算，禁用时为 null。
`prefill_count` 和 `step_count` 分别表示 MTP prefill 与 decode-step 调用数；`fallback_prefill_count` 表示活动 batch 不满足 MTP 条件、改用普通调度路径的 prefill 次数。
`drafted_tokens` 和 `accepted_draft_tokens` 用于观察主模型验证路径接受了多少 draft token。
禁用时字段形状不变：

```json
{
  "mtp": {
    "enabled": false,
    "draft_tokens": null,
    "prefill_count": 0,
    "step_count": 0,
    "fallback_prefill_count": 0,
    "drafted_tokens": 0,
    "accepted_draft_tokens": 0
  }
}
```

## 请求边界

不支持逐请求设置 `mtp_model_dir` 或 `mtp_draft_tokens`。使用启动参数选择执行路径，通过运行指标确认实际使用情况。
