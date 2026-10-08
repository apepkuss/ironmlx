# System One API

[简体中文](zh-CN/laya-systemone-api.md) · [API reference](api-reference.md) · [Service and management API](service-api.md)

IronMLX App manages `aac6fef/laya-multilingual-mlx` and serves its typed decision
API on the same port as the other models. `jev-latest` and other TypeSafe model
names are not aliases for this checkpoint.

Reference for Laya choice, score and true/false probability requests, followed by [App setup](#use-the-app) and [standalone CLI](#optional-standalone-cli) instructions. Shared App and EnginePool access conventions are defined in [Service and management API](service-api.md); standalone CLI authentication is defined on this page.

## Endpoints

- `POST /v1/systemone`: accepts the [request](#request) below and returns
  TypeSafe-shaped `model`, `answers`, and `usage` fields. See [Response](#response).
- `GET /v1/models`: returns `{"models":[{"name", "description", "release_date"}]}`
  with registered decision models in App mode, or the loaded Laya model in
  standalone mode. App mode also includes OpenAI `object`/`data` fields.
  `release_date` is the checkpoint repository's
  Hugging Face creation date (`2026-09-19`).

On authenticated listeners, missing or invalid Bearer credentials return `401`. Invalid JSON, unsupported
model IDs, malformed questions, and options exceeding the checkpoint's token
budget return `422`. Inference failures return `500`, and a stopped worker
returns `503`. The managed service also returns `503` if model loading fails or the queue is full, and `504` on a request timeout. The service does not claim Jev model behavior or its context
limits.

## Request

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "state": {"message": "发票被重复扣款，请退款。"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "refunds", "technical": "bugs"}
    },
    "refund": {
      "type": "noul",
      "instructions": "Is a refund requested?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["can wait", "soon", "today"]
    }
  }
}
```

- `model` identifies the actual loaded checkpoint. No Jev alias maps to Laya.
- `state` is a string, JSON object, or JSON array. Each question is evaluated
  against the same state. Question keys must be nonempty and are returned unchanged as answer keys.
- `choice.criteria` is an ordered map of 1–255 distinct, nonempty labels to optional
  descriptions (strings, objects, arrays, or `null`). The model scores the labels in their request order.
- `score.criteria` is an ordered array of 2–10 descriptions (strings, objects, or
  arrays), indexed from zero.
  Structured descriptions are rendered to JSON strings in the response legend,
  so `legend` always has the TypeSafe `map<string, string>` shape.
- `noul.criteria` may describe `true` and `false`; its value is P(true).
- `instructions` is required and may be a string, object, or array. The runtime
  renders structured values deterministically before tokenization.

### Input limits

`questions` must be nonempty. The question/option prefix targets 256 tokens.
Each option is shortened to at most 48 tokens, with a lower cap when needed.
Requests are rejected if the resulting options cannot fit the 1,024-token total
context. State is truncated from the right; options are never silently dropped.

## Response

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "answers": {
    "department": {
      "type": "choice", "choice": "billing", "confidence": 0.5635,
      "probabilities": {"billing": 0.91, "technical": 0.09}
    },
    "refund": {"type": "noul", "noul": 0.94},
    "urgency": {
      "type": "score", "score": 1.32, "confidence": 0.1108,
      "legend": {"0": "can wait", "1": "soon", "2": "today"},
      "probabilities": {"0": 0.12, "1": 0.44, "2": 0.44}
    }
  },
  "usage": {"input_tokens": 327, "output_tokens": 0}
}
```

Numbers above illustrate the shape, not measured model output. `input_tokens`
counts every encoded question-state sequence, including repeated state tokens;
`output_tokens` is zero because this model does not generate text. Choice and
score confidence use the reference runtime's normalized-entropy calculation;
score is the expected zero-based rubric index. Noul returns only its probability
in the TypeSafe-compatible response. The Laya action head and any diagnostic
outputs remain internal.

## Official SDK clients

For the App, point the official SDKs at `http://127.0.0.1:9068` and set their default model
to `aac6fef/laya-multilingual-mlx`.

Python (`typesafe-sdk==0.7.1`):

```python
from typesafe_sdk import Noul, TypeSafeClient

with TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:9068",
    model="aac6fef/laya-multilingual-mlx",
) as client:
    print([model.name for model in client.models.list().models])
    result = client.system_one(
        state="发票被重复扣款，请退款。",
        questions={"refund": Noul(instructions="Is a refund requested?")},
    )
    print(result.nouls["refund"].noul)
```

JavaScript (`@typesafe-ai/sdk@0.6.0`, Node.js 22):

```javascript
import { noul, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  apiKey: "local",
  baseURL: "http://127.0.0.1:9068",
  defaultModel: "aac6fef/laya-multilingual-mlx",
});
console.log((await client.models.list()).map((model) => model.name));
const result = await client.systemOne({
  state: "发票被重复扣款，请退款。",
  questions: { refund: noul("Is a refund requested?") },
});
console.log(result.answers.refund.noul);
```

## Use the App

1. In **Models → Model Download → Hugging Face**, search for
   `aac6fef/laya-multilingual-mlx` and download it.
2. In **Model Management**, click **Load** on its `DECISION` row. If a DFlash2
   target is running, unload that exclusive target first.
3. Open **Model Param Settings** using the row's gear button to view the model
   alias, `DECISION` type and read-only context size. The context size comes from
   the checkpoint configuration and includes instructions, options and state.
   Find the service address on **Status** and use the exact model ID.
4. Configure the external application to call `POST /v1/systemone`. The default
   local endpoint is `http://127.0.0.1:9068/v1/systemone`; official SDKs use
   `http://127.0.0.1:9068` as their base URL, without `/v1`.

Use the existing **Unload**, pin, default-model and restart controls. The App
restores its managed loaded models after restart. In-flight requests retain the
model until inference finishes; unload may briefly show a draining state. Lazy
loading and idle eviction follow the shared model pool policy.

Local access needs no API key. SDKs requiring a nonempty key may use `local`.
For another machine, configure LAN access in **Settings** and use the displayed
HTTPS endpoint and its Bearer API key. System One shares the App's network and
security settings. The model has no chat, streaming, MTP or KV-cache settings.

`GET /v1/models` keeps the OpenAI `object`/`data` fields and adds a TypeSafe
`models` array containing registered decision models. This allows both client
families to discover models on the same port.

## Model inference settings

Below the read-only basic information, the App provides:

| Setting | Default | Behavior |
| --- | --- | --- |
| Compute precision | FP16 | Choose FP16 or FP32. Weights are converted in memory for inference; the downloaded checkpoint stays FP16. FP32 increases memory use. |
| Question batch size | 16 | Process 1–256 questions in each forward pass. This is a per-request batch size, not concurrent API requests. Larger batches use more memory. |
| Question prefix cache | Off | Reuse CPU tokenization and question-prefix preparation, with up to 128 prefixes retained. State and decision results are not cached. |

The collapsed **Advanced Settings** section adds:

| Setting | Default | Behavior |
| --- | --- | --- |
| Compute device | Auto | Prefer an available GPU; CPU explicitly selects CPU inference. |
| Compile optimization | Off | Compile the forward graph. New input shapes incur compilation overhead; speed depends on the workload. |
| Sequence alignment | Off | Optional padding multiple from 1 to 1024 (suggested: 16), capped at the model context limit. Padding does not count as input usage. |
| Question / option budget | Read only | `head_max_len` read from the checkpoint; consumes part of the total context. |
| Calibration temperature | Read only | Effective per-task and option-count calibration from the checkpoint, clamped to 0.5–5.0 as in inference. This is not sampling temperature. |

Save persists the settings and reloads a loaded model after active requests finish.
Settings also apply to subsequent loads and App restart recovery. Context Size is
fixed by the checkpoint and cannot be edited.

The model-management load/register payload accepts a `decision` object:
`{"dtype":"float16","batch_size":16,"cache_prompts":false,"device":"auto","compile":false,"pad_to_multiple":null}`. Loaded-model
information reports the applied settings. These are model settings, not fields
in a System One inference request. The standalone CLI uses the defaults.

## Status and decision metrics

While a decision model is loaded, **Status** presents request activity using
the decision runtime rather than causal-model token-generation metrics:

- **Processing** means at least one decision request is active or queued.
- **Just completed** remains visible briefly after the most recent request
  finishes, so short decisions do not disappear between Dashboard refreshes.
- **Idle** means no request is active, queued or recently completed.

The decision performance section reports:

| Metric | Meaning |
| --- | --- |
| Completed | Successful requests since this model instance was loaded. |
| Errors | Failed requests since this model instance was loaded, including validation, queue, timeout and inference failures. |
| P50 latency | Median end-to-end latency of successful requests completed in the most recent 60-second window, in milliseconds. |
| Input throughput | Median per-request input-token rate in the same recent window, in tokens per second. |
| Question throughput | Median per-request question rate in the same recent window, in questions per second. |

The three recent-window metrics show `—` until a successful request supplies a
sample, and return to `—` when the 60-second window becomes empty. Completed and
error counts remain cumulative for the current loaded instance. Unloading and
reloading the model, or restarting the backend, starts a new metrics lifecycle.

App integrations can read the same data from
`GET /admin/api/models/loaded`. A loaded decision model has
`runtime_kind: "decision"` and a `decision_metrics` object. See the
[Service and management API](service-api.md#app-decision-runtime-metrics) for its
field contract.

## Optional standalone CLI

```sh
export IRONMLX_SYSTEMONE_API_KEY='replace-with-a-local-secret'
ironmlx --mlx-metallib /path/to/mlx.metallib serve-systemone \
  --model-dir /path/to/laya-multilingual-mlx --port 8767
```

For this standalone CLI, the default bind address is `127.0.0.1`. Every `/v1/*` request requires
`Authorization: Bearer <IRONMLX_SYSTEMONE_API_KEY>`, including model listing.
`GET /healthz` is available without credentials. The service keeps one model
instance and serializes inference on its dedicated worker thread. It has no
dependency on the LLM/VLM chat server's model loader.
