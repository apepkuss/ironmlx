# Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen36-moe-b4-short.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model; for the dense Qwen3.8-27B model, see [Qwen3.8 27B B4 concurrency performance comparison (short prompts)](qwen38-dense-b4-short.md). All requests use short prompts (40–83 input tokens) with a fixed 256-token output, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the short-prompt B4 concurrency comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen36-moe-b4-short -->

- **Fastest first token.** The median wait for the first words is 158 ms, first in all four rounds; TensorFold 0.6.6 follows at 174 ms.
- **Second for the slowest request's first token, close to Splash.** The slowest of the four requests shows its first words after 260 ms; Splash is 1.3% faster at 256 ms and leads in three of the four rounds.
- **Second in throughput.** Four requests together complete at 304.7 tok/s, 12.3% below Splash (347.5) and ahead of TensorFold 0.6.6 (289.0) and oMLX (195.7). Splash leads in all four rounds.
- **Lowest memory.** Peak memory while serving is 20.9 GiB, lowest in all four rounds; TensorFold 0.6.6 follows at 21.2 GiB.

### Throughput vs. time to first token

Batch throughput is the total output tokens of the four requests divided by the batch completion time (from the first request sent to the last request finished), and reflects the server's total capacity under concurrent load. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b4-short -->

> [!NOTE]
> TensorFold 1.0.5 is not included: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#qualified-models), Qwen3.6 is still under qualification for the native engine, and the native engine accepts an external DFlash2 draft model only for Qwen3.8-27B-class checkpoints ([source](https://github.com/ashhart/TensorFold/blob/v1.0.5/zig/src/native/metal.zig#L101-L105)).

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher batch throughput is better, and lower is better for the time and memory metrics; the best value in each of the first six rows is in bold. On the first six metrics, the ranking of the applications stayed the same in all four rounds, except for the slowest request's first token, where IronMLX was ahead of Splash in one of the four rounds.

| Metric | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Batch throughput (tok/s, higher is better)¹ | 304.7 (303.9–⁠305.3) | 195.7 (188.8–⁠201.5) | **347.5** (342.2–⁠352.5) | 289.0 (285.5–⁠295.5) |
| Median time to first token (ms) | **158** (155–⁠162) | 309 (288–⁠347) | 256 (233–⁠263) | 174 (166–⁠182) |
| Slowest request's first token (ms) | 260 (256–⁠267) | 458 (422–⁠479) | **256** (234–⁠263) | 303 (298–⁠333) |
| Median completion time (s) | 3.24 (3.23–⁠3.27) | 4.21 (4.07–⁠4.43) | **2.31** (2.28–⁠2.33) | 2.81 (2.78–⁠2.87) |
| Slowest request's completion (s)¹ | 3.36 (3.35–⁠3.37) | 5.23 (5.08–⁠5.42) | **2.95** (2.91–⁠2.99) | 3.54 (3.46–⁠3.59) |
| Peak memory while serving (GiB) | **20.9** (20.8–⁠20.9) | 23.7 (23.2–⁠24.1) | 21.5 (21.3–⁠21.5) | 21.2 (21.1–⁠21.3) |
| Per-request decode speed (tok/s, auxiliary) | 82.6 (81.8–⁠82.7) | 66.8 (63.0–⁠70.0) | 128.1 (127.2–⁠128.5) | 99.1 (97.7–⁠101.8) |

> [!NOTE]
> ¹ Batch throughput is derived from the batch completion time (1,024 tokens ÷ batch completion time), and the batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 1.3 ms), so these two rows reflect the same result and should not be counted as two separate advantages.

> [!NOTE]
> Peak memory while serving is measured during the scored batch. During model loading and warm-up, IronMLX peaks higher (about 21.8 GiB) than during serving; this page does not compare loading-time memory.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 12 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-short -->

Concurrent requests from the client do not guarantee batching inside the engine. According to each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0-rc.2:** uses the external DFlash2 draft model and verifies the requests together in ragged linear batches (no draft tree is used for this model). The four first tokens arrive one after another, about 60–75 ms apart, and the four requests then finish within about 0.17 s of each other. Under its default draft-budget policy for this model, about 93–95% of the batched steps advanced each request by one token without a draft. Outputs were byte-identical to the same prompts served one at a time.
- **oMLX 0.7.0:** runs batched speculative decoding for the four requests with the external DFlash2 draft model (its log names this batched path "Lightning MTP", activated first for two and then for four requests in every batch). Its output for the same prompt varied between batches.
- **Splash 1.3.0:** uses the external DFlash2 draft model; in each batch one request receives its first token first (about 0.17 s), and the other three receive theirs at the same moment (about 0.26 s).
- **TensorFold 0.6.6:** Python engine serving the requests in parallel lanes (`--parallel 4`); uses the external DFlash2 draft model and, on this model, also its built-in copy proposer. Every request accepted DFlash2 drafts.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 13, 2026 (12:26–14:58 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, then until macOS thermal pressure had read nominal for 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.2 | Official release `IronMLX.dmg` (sha256 `55a37763…`), tag `v0.2.0-rc.2` → `0ef3c55` | Bundled server and Metal runtime, started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none`; default M5 profile, no tuning switches |
| oMLX 0.7.0 | Official app | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.3.0 | Official bottle | `--kv-format bf16 --max-context 8192`, SSD cache off; the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 0.6.6 | Source tag (`cb2ebf05`) | `--drafter-bits 4 --lane-kernels on --parallel 4`, prompt cache and snapshots off |

Common settings for all applications: 4-bit draft model, BF16 KV cache, 8,192-token context, and capacity 4 where configurable; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (expert router gates 8-bit); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens) in three sets of four, so every batch has four different prompts.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four, then one scored batch of four released together (largest send spread 1.3 ms), then stops the server. Four rounds × four applications × three prompt sets = 48 sessions, in a balanced order in which each application has the same average position.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is 1,024 tokens divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch.
- **Validity:** 48 of 48 sessions and 192 of 192 scored requests passed every check, and both token counts were 256 for every request; no session in this run was retried or dropped.

> [!NOTE]
> Earlier runs: two runs were stopped because one oMLX request's output re-encoded to 255 tokens with the target tokenizer although oMLX reported 256. The counting rule was then revised for all applications (the server reports 256 tokens; the tokenizer recount may be 255 or 256), and the run was repeated from the start. A further complete run without the cool-down between sessions gave the same order on every metric.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. For example, TensorFold 1.0.5 was not run because its native engine has not yet qualified Qwen3.6, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Long prompts, other models, other concurrency levels, or real multi-user traffic.
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Between batches with identical work, throughput varied across the four rounds on the same prompt set by at most 2.0% for IronMLX, at most 3.0% for Splash, at most 6.4% for TensorFold 0.6.6, and at most 7.0% for oMLX. The order of the four applications did not change.
