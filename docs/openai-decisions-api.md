# OpenAI Decisions API

[简体中文](zh-CN/openai-decisions-api.md) · [API reference](api-reference.md) · [System One API](laya-systemone-api.md) · [Service and management API](service-api.md)

IronMLX serves `POST /v1/decisions` for the local `aac6fef/laya-multilingual-mlx`
checkpoint, so the official OpenAI SDK's `decisions.create` can classify and score
text with Laya. The endpoint uses the same Laya worker, model pool, leases, queue,
authentication, lifecycle and metrics as [System One](laya-systemone-api.md);
`/v1/systemone` keeps its own contract unchanged.

Laya is a text-only encoder. It returns predicate, choice and score answers but
no images, refusals or generated text. OpenAI's hosted `gpt-6-luna` and its
behavior are not available through this endpoint.

## Availability

| Service | Base URL for the OpenAI SDK | Authentication |
| --- | --- | --- |
| IronMLX App | `http://127.0.0.1:9068/v1` (default) | None on loopback; SDKs may use `api_key="local"`. LAN uses the App's HTTPS endpoint and Bearer API key. |
| CLI EnginePool (`--model-manifest`) | `http://127.0.0.1:<port>/v1` | Same as the App for local and LAN listeners. |
| Standalone `ironmlx serve-systemone` | `http://127.0.0.1:8767/v1` (default) | `Authorization: Bearer <IRONMLX_SYSTEMONE_API_KEY>` on every `/v1/*` request, including loopback. |

In the App, download and load the model as described in [Use the App](laya-systemone-api.md#use-the-app).
A registered but unloaded model loads on the first request according to the shared pool policy.
The standalone service loads the model at startup; see its [CLI section](laya-systemone-api.md#optional-standalone-cli).

## Quick start

curl:

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

Python (`openai>=3.26.0`; verified with `openai==3.26.0`):

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

For LAN access, use the App's HTTPS base URL (for example
`https://192.168.1.20:9068/v1`) and its Bearer API key as `api_key`, and trust the
App-exported CA. For the standalone service, pass `IRONMLX_SYSTEMONE_API_KEY` as
`api_key`.

## Request

| Field | Required | Description |
| --- | --- | --- |
| `model` | Yes | Exactly `aac6fef/laya-multilingual-mlx`. Other IDs return `404 model_not_found`; a registered non-decision model returns `400 model_task_mismatch`. |
| `input` | Yes | A string, or an array of user messages. Every question is evaluated against the same input. |
| `questions` | Yes | A nonempty array of `predicate`, `choice` and `score` questions. Answers are returned in the same order. |
| `safety_identifier` | No | A string of at most 128 characters, or `null`. It is validated and accepted for SDK compatibility; it does not affect inference. |

Unknown top-level, message, part, question, choice or level fields return
`400 unknown_parameter`. Optional string fields (`name`, `description`,
`safety_identifier`) treat `null` as omitted.

### Input

- A string is used as Laya state exactly as sent, including an empty string.
- An array contains messages with `role: "user"`, optional `type: "message"`, and
  `content` as a string or a nonempty array of `{"type": "input_text", "text": ...}` parts.
  Text is concatenated in message and part order, separated by one newline (`\n`).
- `input_image`, `input_file`, `input_audio`, other part types, non-user roles,
  function calls, item references and other input item types return
  `400 unsupported_value` with the offending path in `param`. Nothing is silently dropped.

### Questions

Every question has `type`, `instructions` (string) and an optional `name`.

| Type | Extra fields | Laya mapping |
| --- | --- | --- |
| `predicate` | — | Laya true/false (`noul`) with its default true/false descriptions. |
| `choice` | `choices`: 2–255 `{value, description?}` | Options are scored in request order. |
| `score` | `levels`: 2–10 `{label, description?}`, lowest first | Level `i` is rendered as `level i: <label>: <description>`, or `level i: <label>` without a description. |

Choice values are typed: `value` must be a string or boolean, and `true` and
`"true"` are distinct values. A value may appear only once per question
(`false` and `"false"` may both appear). Each option is rendered for the model as
`<value>: <description>`, or `<value>` without a description; booleans render as
`true` or `false`. An empty-string value is allowed but must have a nonempty
description, which is then rendered alone. Typed identity never reaches the
prompt, so a boolean and a string with the same text and description are
distinct answers that the model sees as identical text.

`name` is optional and is echoed in the answer. Unnamed questions return
`"name": null`. Names may repeat; answers are matched by position, so no answer
overwrites another.

## Response

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

The numbers above are actual output of the quick-start request on the local
checkpoint with default settings.

- `predicate.probability` is Laya's probability that the condition is true
  (System One's `noul`).
- `choice.choice` is the most probable value with its original type. Ties pick
  the earliest option. `probabilities` lists every option in request order.
- `score.score` is the probability-weighted zero-based level index
  (`Σ i · p_i`), so it can fall between levels. `probabilities[].value` is the
  level index and `label` is the request label.
- `confidence` uses Laya's calculation, unchanged from System One: one minus the
  normalized entropy of the option probabilities, `1 − H(p) / ln(n)`, clamped to
  `[0, 1]`.
- Probabilities, scores and confidence are rounded to four decimal places, as in System One.
- No `refusal` answers are produced: Laya has no refusal output, and IronMLX does
  not synthesize one. SDK code that handles `refusal` remains compatible.

### Usage

- `input_tokens` is the number of encoder tokens actually evaluated: for each
  question, its instructions and options plus the (possibly truncated) input.
  The input is counted once per question. It equals System One's
  `usage.input_tokens` for equivalent requests.
- `output_tokens` is `0` and `total_tokens` equals `input_tokens`.
- `cached_tokens`, `cache_write_tokens` and `reasoning_tokens` are always `0`. The
  optional question prefix cache only reuses CPU tokenization; every token is
  still evaluated by the model, so it is not reported as cached model tokens.

## Limits

Laya's context, rubric, token budget and truncation rules apply unchanged:
the question/option prefix targets 256 tokens, each option is shortened to at
most 48 tokens (lower when needed), and the total context is 1,024 tokens. Input
text is truncated from the right. Options are never dropped; if they cannot fit,
the request returns `400 question_token_budget_exceeded`. Choice questions
accept 2–255 options (System One keeps 1–255) and score questions 2–10 levels.
Passing the count check does not guarantee a fit: many or long options can still
exceed the token budget.

## Errors

Errors use the OpenAI format: `{"error": {"message", "type", "param", "code"}}`.
`503` responses include `Retry-After: 5`.

| Status | `type` | `code` | When |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | `invalid_json` | Body is not valid JSON. |
| 400 | `invalid_request_error` | `missing_required_parameter`, `invalid_type`, `invalid_value`, `unknown_parameter` | Malformed request; `param` names the field, such as `questions[1].choices[0].value`. |
| 400 | `invalid_request_error` | `unsupported_value` | Images, files, audio, non-user roles, function calls or other unsupported input. |
| 400 | `invalid_request_error` | `question_token_budget_exceeded` | Question options cannot fit Laya's token budget. |
| 400 | `invalid_request_error` | `model_task_mismatch` | The model is registered but is not a decision model. |
| 401 | `authentication_error` | `auth_invalid` | Missing or invalid Bearer key on LAN or the standalone service. |
| 404 | `invalid_request_error` | `model_not_found` | The model is not registered, is disabled, or is not served by the standalone service. |
| 500 | `server_error` | `decision_inference_failed` | Inference failed. |
| 503 | `server_error` | `memory_budget_exceeded` | The process memory governor declined the inference reservation; retry later. |
| 503 | `server_error` | `model_load_failed` | The registered model could not be loaded. |
| 503 | `server_error` | `decision_queue_full` | The worker queue (16 requests) is full. |
| 503 | `server_error` | `decision_worker_unavailable` | The worker stopped. |
| 504 | `server_error` | `decision_timeout` | The request did not finish within the worker timeout. |

## Lifecycle and metrics

Decisions and System One requests share one queue and worker per loaded model.
An in-flight request holds a model lease: unloading waits until it finishes, and
a later request reloads a lazily loaded model. Both APIs contribute to the
model's `request_count` in `GET /v1/models` and to `decision_metrics` in
`GET /admin/api/models/loaded`, where `questions_per_second` counts Decisions
questions. Requests rejected during validation are answered before reaching the
worker and are not counted as failed inferences. See
[App decision runtime metrics](service-api.md#app-decision-runtime-metrics).

## SDK compatibility check

[`scripts/api-contract-sdk/decisions`](../scripts/api-contract-sdk/decisions/contract.py)
pins `openai==3.26.0` separately from the `openai==2.48.0` harness used for
Chat Completions and Responses. It calls a live server with the loaded model,
validates responses with the SDK's strict models, compares numbers with
`/v1/systemone` on an equivalent request, and checks concurrency and error codes:

```bash
python3 -m venv /tmp/ironmlx-decisions-sdk
/tmp/ironmlx-decisions-sdk/bin/python -m pip install -r scripts/api-contract-sdk/decisions/requirements.txt
/tmp/ironmlx-decisions-sdk/bin/python scripts/api-contract-sdk/decisions/contract.py --base-url http://127.0.0.1:9068/v1
```

Add `--api-key <key> --wrong-api-key local` for the standalone service or LAN,
and `--insecure` for a self-signed LAN certificate.
