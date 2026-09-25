# Laya in IronMLX App and the System One API

[简体中文](zh-CN/laya-systemone-api.md)

IronMLX App manages `aac6fef/laya-multilingual-mlx` and serves its typed decision
API on the same port as the other models. `jev-latest` and other TypeSafe model
names are not aliases for this checkpoint.

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

## Endpoints

- `POST /v1/systemone`: accepts the [phase-one request contract](laya-phase-one-contract.md)
  and returns its TypeSafe-shaped `model`, `answers`, and `usage` fields.
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
