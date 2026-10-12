# OpenAI Decisions API

[English](../openai-decisions-api.md) · [API 参考](api-reference.md) · [System One API](laya-systemone-api.md) · [服务与管理 API](service-api.md)

IronMLX 为本地 `aac6fef/laya-multilingual-mlx` 提供 `POST /v1/decisions`，官方
OpenAI SDK 的 `decisions.create` 可以直接调用 Laya 完成文本分类与评分。该端点与
[System One](laya-systemone-api.md) 共用同一个 Laya worker、模型池、lease、队列、
鉴权、生命周期和指标；`/v1/systemone` 的契约保持不变。

Laya 是纯文本编码器，返回 predicate、choice 和 score 答案，不处理图片，不产生
refusal，也不生成文本。该端点不提供 OpenAI 托管的 `gpt-6-luna` 及其行为。

## 可用范围

| 服务 | OpenAI SDK 的 base URL | 鉴权 |
| --- | --- | --- |
| IronMLX App | `http://127.0.0.1:9068/v1`（默认） | 本机回环无需密钥，SDK 可用 `api_key="local"`；局域网使用 App 显示的 HTTPS 地址和 Bearer API Key。 |
| CLI EnginePool（`--model-manifest`） | `http://127.0.0.1:<port>/v1` | 本机与局域网规则同 App。 |
| 独立 `ironmlx serve-systemone` | `http://127.0.0.1:8767/v1`（默认） | 所有 `/v1/*` 请求都需要 `Authorization: Bearer <IRONMLX_SYSTEMONE_API_KEY>`，本机回环也不例外。 |

在 App 中按[使用 App](laya-systemone-api.md#使用步骤)下载并加载模型。已注册但未加载的模型
会按共享模型池策略在首个请求时加载。独立服务在启动时加载模型，见其[独立 CLI 说明](laya-systemone-api.md#可选的独立-cli)。

## 快速开始

curl：

```bash
curl http://127.0.0.1:9068/v1/decisions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
    "model": "aac6fef/laya-multilingual-mlx",
    "input": "I was charged twice for my order. Please refund me today.",
    "questions": [
      {"type": "predicate", "name": "refund", "instructions": "Does the customer ask for a refund?"},
      {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
       "choices": [{"value": "billing", "description": "Payments, invoices, and refunds."},
                   {"value": "technical", "description": "Bugs and outages."},
                   {"value": "other", "description": "Anything else."}]},
      {"type": "score", "name": "urgency", "instructions": "How urgent is this?",
       "levels": [{"label": "low", "description": "Can wait."},
                  {"label": "medium"},
                  {"label": "high", "description": "Needs action today."}]}
    ]
  }'
```

Python（`openai>=3.26.0`；已用 `openai==3.26.0` 验证）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9068/v1", api_key="local")
decision = client.decisions.create(
    model="aac6fef/laya-multilingual-mlx",
    input="I was charged twice for my order. Please refund me today.",
    questions=[
        {"type": "predicate", "name": "refund",
         "instructions": "Does the customer ask for a refund?"},
        {"type": "choice", "name": "department",
         "instructions": "Which team should handle this?",
         "choices": [
             {"value": "billing", "description": "Payments, invoices, and refunds."},
             {"value": "technical", "description": "Bugs and outages."},
             {"value": "other", "description": "Anything else."},
         ]},
        {"type": "score", "name": "urgency", "instructions": "How urgent is this?",
         "levels": [
             {"label": "low", "description": "Can wait."},
             {"label": "medium"},
             {"label": "high", "description": "Needs action today."},
         ]},
    ],
)
for answer in decision.answers:
    if answer.type == "predicate":
        print(answer.name, answer.probability)
    elif answer.type == "choice":
        print(answer.name, answer.choice, answer.confidence)
    elif answer.type == "score":
        print(answer.name, answer.score, answer.confidence)
print(decision.usage.input_tokens)
```

局域网访问时，使用 App 的 HTTPS base URL（例如 `https://192.168.1.20:9068/v1`），
把它的 Bearer API Key 作为 `api_key`，并信任 App 导出的 CA。独立服务把
`IRONMLX_SYSTEMONE_API_KEY` 作为 `api_key`。

## 请求

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `model` | 是 | 必须是 `aac6fef/laya-multilingual-mlx`。其他 ID 返回 `404 model_not_found`；已注册的非决策模型返回 `400 model_task_mismatch`。 |
| `input` | 是 | 字符串，或 user 消息数组。所有问题针对同一份输入评估。 |
| `questions` | 是 | 非空数组，元素为 `predicate`、`choice` 或 `score` 问题。答案按相同顺序返回。 |
| `safety_identifier` | 否 | 不超过 128 个字符的字符串，或 `null`。为 SDK 兼容而校验并接受，不影响推理。 |

顶层、消息、内容片段、问题、选项或等级中的未知字段返回 `400 unknown_parameter`。
可选字符串字段（`name`、`description`、`safety_identifier`）传 `null` 视为未提供。

### 输入

- 字符串原样作为 Laya 状态使用，空字符串也合法。
- 数组中的每条消息须为 `role: "user"`，可带 `type: "message"`；`content` 为字符串，或
  非空的 `{"type": "input_text", "text": ...}` 片段数组。文本按消息和片段顺序拼接，
  中间以一个换行符（`\n`）分隔。
- `input_image`、`input_file`、`input_audio`、其他片段类型、非 user 角色、函数调用、
  item 引用及其他输入项返回 `400 unsupported_value`，`param` 指出具体位置；不会静默丢弃任何内容。

### 问题

每个问题包含 `type`、`instructions`（字符串）和可选的 `name`。

| 类型 | 额外字段 | Laya 映射 |
| --- | --- | --- |
| `predicate` | — | Laya 真假判断（`noul`），使用其默认的真/假描述。 |
| `choice` | `choices`：2–255 个 `{value, description?}` | 按请求顺序评分。 |
| `score` | `levels`：2–10 个 `{label, description?}`，由低到高 | 第 `i` 级渲染为 `level i: <label>: <description>`，无描述时为 `level i: <label>`。 |

choice 的值带类型：`value` 必须是字符串或布尔值，`true` 与 `"true"` 是两个不同的值。
同一问题中同一个值只能出现一次（`false` 与 `"false"` 可以同时出现）。每个选项向
模型渲染为 `<value>: <description>`，无描述时为 `<value>`；布尔值渲染为 `true` 或
`false`。允许空字符串值，但必须提供非空描述，此时只渲染描述。类型信息不会进入
提示词，因此文本和描述都相同的布尔值与字符串是两个不同的答案，但模型看到的文本相同。

`name` 可选，会原样回显在答案中；未命名的问题返回 `"name": null`。名称可以重复，
答案按位置对应，不会互相覆盖。

## 响应

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "answers": [
    {"type": "predicate", "name": "refund", "probability": 0.9959},
    {"type": "choice", "name": "department", "choice": "billing",
     "probabilities": [{"value": "billing", "probability": 0.9263},
                       {"value": "technical", "probability": 0.031},
                       {"value": "other", "probability": 0.0427}],
     "confidence": 0.7149},
    {"type": "score", "name": "urgency", "score": 1.7944,
     "probabilities": [{"value": 0, "label": "low", "probability": 0.0376},
                       {"value": 1, "label": "medium", "probability": 0.1304},
                       {"value": 2, "label": "high", "probability": 0.832}],
     "confidence": 0.5066}
  ],
  "usage": {
    "input_tokens": 148,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 148
  }
}
```

以上数值是快速开始中的请求在本地 checkpoint、默认设置下的实际输出。

- `predicate.probability` 是 Laya 判断条件成立的概率（即 System One 的 `noul`）。
- `choice.choice` 是概率最高的值，保留原始类型；概率相同时取最靠前的选项。
  `probabilities` 按请求顺序列出所有选项。
- `score.score` 是按概率加权的零基等级索引（`Σ i · p_i`），可能落在两级之间。
  `probabilities[].value` 是等级索引，`label` 是请求中的标签。
- `confidence` 沿用 Laya 的计算方式，与 System One 相同：1 减去选项概率的归一化熵，
  即 `1 − H(p) / ln(n)`，并限制在 `[0, 1]`。
- 概率、score 和 confidence 与 System One 一样保留四位小数。
- 不会返回 `refusal` 答案：Laya 没有拒答输出，IronMLX 也不会伪造。处理 `refusal`
  的 SDK 代码仍然兼容。

### usage

- `input_tokens` 是模型实际评估的编码器 token 数：每个问题的指令与选项，加上（可能被截断的）输入。
  输入按问题逐个计入。等价请求下与 System One 的 `usage.input_tokens` 相同。
- `output_tokens` 为 `0`，`total_tokens` 等于 `input_tokens`。
- `cached_tokens`、`cache_write_tokens` 和 `reasoning_tokens` 恒为 `0`。可选的问题前缀缓存
  只复用 CPU 分词结果，所有 token 仍由模型计算，因此不计为模型 token 缓存。

## 限制

Laya 的上下文、评分等级、token 预算与截断规则保持不变：问题与选项前缀以 256 个 token
为目标，每个选项最多保留 48 个 token（必要时更少），总上下文为 1,024 个 token。输入
文本从右侧截断。选项不会被丢弃；放不下时返回 `400 question_token_budget_exceeded`。
choice 问题接受 2–255 个选项（System One 仍为 1–255），score 问题接受 2–10 个等级。
通过数量校验不代表一定放得下：选项较多或较长时仍可能超出 token 预算。

## 错误

错误使用 OpenAI 格式：`{"error": {"message", "type", "param", "code"}}`。
`503` 响应带有 `Retry-After: 5`。

| 状态码 | `type` | `code` | 场景 |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | `invalid_json` | 请求体不是合法 JSON。 |
| 400 | `invalid_request_error` | `missing_required_parameter`、`invalid_type`、`invalid_value`、`unknown_parameter` | 请求格式错误；`param` 指出字段，例如 `questions[1].choices[0].value`。 |
| 400 | `invalid_request_error` | `unsupported_value` | 图片、文件、音频、非 user 角色、函数调用等不支持的输入。 |
| 400 | `invalid_request_error` | `question_token_budget_exceeded` | 问题选项超出 Laya 的 token 预算。 |
| 400 | `invalid_request_error` | `model_task_mismatch` | 模型已注册但不是决策模型。 |
| 401 | `authentication_error` | `auth_invalid` | 局域网或独立服务缺少或使用了无效的 Bearer Key。 |
| 404 | `invalid_request_error` | `model_not_found` | 模型未注册、已禁用，或不是独立服务所提供的模型。 |
| 500 | `server_error` | `decision_inference_failed` | 推理失败。 |
| 503 | `server_error` | `memory_budget_exceeded` | 进程内存管理拒绝了本次推理的内存预留，稍后重试。 |
| 503 | `server_error` | `model_load_failed` | 已注册模型加载失败。 |
| 503 | `server_error` | `decision_queue_full` | worker 队列（16 个请求）已满。 |
| 503 | `server_error` | `decision_worker_unavailable` | worker 已停止。 |
| 504 | `server_error` | `decision_timeout` | 请求未在 worker 超时时间内完成。 |

## 生命周期与指标

每个已加载模型的 Decisions 与 System One 请求共用同一队列和 worker。进行中的请求持有
模型 lease：卸载会等待其完成；按需加载的模型会在后续请求时重新加载。两种 API 都计入
`GET /v1/models` 中模型的 `request_count`，以及 `GET /admin/api/models/loaded` 中的
`decision_metrics`，其中 `questions_per_second` 按 Decisions 的问题数计算。请求校验失败时
在进入 worker 前即返回，不计为推理失败。见 [App 决策运行时指标](service-api.md#app-决策运行时指标)。

## SDK 兼容性检查

[`scripts/api-contract-sdk/decisions`](../../scripts/api-contract-sdk/decisions/contract.py)
单独锁定 `openai==3.26.0`，与 Chat Completions / Responses 使用的 `openai==2.48.0`
环境分开。它对已加载模型的实际服务发起调用，用 SDK 严格模型校验响应，在等价请求上与
`/v1/systemone` 比对数值，并检查并发与错误码：

```bash
python3 -m venv /tmp/ironmlx-decisions-sdk
/tmp/ironmlx-decisions-sdk/bin/python -m pip install -r scripts/api-contract-sdk/decisions/requirements.txt
/tmp/ironmlx-decisions-sdk/bin/python scripts/api-contract-sdk/decisions/contract.py --base-url http://127.0.0.1:9068/v1
```

独立服务或局域网加 `--api-key <key> --wrong-api-key local`，自签名局域网证书再加 `--insecure`。
