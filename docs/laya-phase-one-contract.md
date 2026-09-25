# Laya multilingual: phase-one decision contract

This document fixes the data boundary for native `aac6fef/laya-multilingual-mlx`
inference. Phase one exposes it to in-process callers and a validation harness.
The TypeSafe-compatible HTTP transport belongs to phase two; it must preserve
this boundary without changing model behavior.

The validated Hugging Face revision is
`f2b4faf51023039425946074e2cf1361d2db11d5`; its `model.safetensors`
SHA-256 is `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`.

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
  against the same state. Question keys are opaque identifiers, returned
  unchanged as answer keys.
- `choice.criteria` is an ordered map of 1–255 distinct labels to optional
  descriptions. The model scores the labels in their request order.
- `score.criteria` is an ordered array of 2–10 descriptions, indexed from zero.
  Structured descriptions are rendered to JSON strings in the response legend,
  so `legend` always has the TypeSafe `map<string, string>` shape.
- `noul.criteria` may describe `true` and `false`; its value is P(true).
- `instructions` is required and may be a string, object, or array. The runtime
  renders structured values deterministically before tokenization.
- Questions must be nonempty. The prompt builder targets a 256-token
  question/option prefix, shortening each option to at most 48 tokens and
  reducing that cap when needed, as in the reference runtime. A request is
  rejected if the resulting options cannot fit the 1,024-token total context.
  State is truncated from the right; options are never silently dropped.

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

## Acceptance boundary

The phase-one native API must parse the unmodified pinned checkpoint, preserve
tokenizer and question formatting, and match the reference `laya-mlx` runtime
on the same fixed input set for chosen options, score, noul, probability vectors,
and input-token counts. Test single and mixed question batches, structured
state/criteria, non-English text, empty/long state, and option-budget rejection.
Measure synchronized end-to-end latency and peak memory separately from the
reference project's published numbers. The App must identify, download, and
verify the artifact as a decision model. Native inference is available through
`ironmlx decide --model-dir <snapshot> --request <request.json>` in phase one;
the App-managed API and standalone service are documented in [Local System One API](laya-systemone-api.md).

The checked-in real-model fixture can be rerun with
`LAYA_MODEL_DIR=<snapshot> LAYA_METALLIB=<mlx.metallib> cargo test -p ironmlx-decision --test laya_reference_parity`.
The fixture now checks FP16 and FP32, batch sizes 1/2/16, and prefix caching
on/off. FP32 reference outputs were generated with the same upstream revision
`0a859518634112655cb97c745dbf04f5191aaf13` and checkpoint, using
`Agent(model_path, dtype="float32", batch_size=16)`.

On the development Apple Silicon host (2026-09-24), three serial cold-process
runs of the three-question example had median wall time 1.93 s and peak RSS
1,305 MB for `ironmlx decide`, versus 0.58 s and 1,240 MB for `laya-mlx`
under `/usr/bin/time -l`. This includes startup and model loading; it is not a
warm throughput comparison. Those measurements used the initial sequential
native path. The current native runtime batches questions (default 16); the
historical timings do not describe its current performance.

Phase two exposes `POST /v1/systemone` and TypeSafe-shaped `GET /v1/models`
on the App service port or an optional standalone address. Its official Python and JavaScript SDK usage is
recorded in [Local System One API](laya-systemone-api.md).
