# Laya reference validation

[简体中文](README.zh-CN.md)

Developer acceptance criteria and historical measurements for the native
`aac6fef/laya-multilingual-mlx` runtime. User-facing request and response fields
are maintained in the [System One API guide](../../../../docs/laya-systemone-api.md#request).

## Reference checkpoint

The validated Hugging Face revision is
`f2b4faf51023039425946074e2cf1361d2db11d5`; its `model.safetensors`
SHA-256 is `7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1`.

## Acceptance boundary

The native API must parse the unmodified pinned checkpoint, preserve
tokenizer and question formatting, and match the reference `laya-mlx` runtime
on the same fixed input set for chosen options, score, noul, probability vectors,
and input-token counts. Test single and mixed question batches, structured
state/criteria, non-English text, empty/long state, and option-budget rejection.
Measure synchronized end-to-end latency and peak memory separately from the
reference project's published numbers. The App must identify, download, and
verify the artifact as a decision model. Native inference is available through
`ironmlx decide --model-dir <snapshot> --request <request.json>`;
the App-managed API and standalone service are documented in [Local System One API](../../../../docs/laya-systemone-api.md).

From the repository root, the checked-in [real-model fixture test](../../laya_reference_parity.rs) can be rerun with
`LAYA_MODEL_DIR=<snapshot> LAYA_METALLIB=<mlx.metallib> cargo test -p ironmlx-decision --test laya_reference_parity`.
The fixture now checks FP16 and FP32, batch sizes 1/2/16, and prefix caching
on/off. FP32 reference outputs were generated with the same upstream revision
`0a859518634112655cb97c745dbf04f5191aaf13` and checkpoint, using
`Agent(model_path, dtype="float32", batch_size=16)`.

## Historical measurements

On the development Apple Silicon host (2026-09-24), three serial cold-process
runs of the three-question example had median wall time 1.93 s and peak RSS
1,305 MB for `ironmlx decide`, versus 0.58 s and 1,240 MB for `laya-mlx`
under `/usr/bin/time -l`. This includes startup and model loading; it is not a
warm throughput comparison. Those measurements used the initial sequential
native path. The current native runtime batches questions (default 16); the
historical timings do not describe its current performance.
