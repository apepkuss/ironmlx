# Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen36-moe-b4-short.md)

> [!NOTE]
> Short-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B4 concurrency
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md)
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md)
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with four requests arriving together. All requests use short prompts (40–83 input tokens) with an output limit of 256 tokens, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the [B1 page](qwen36-moe-b1-short.md) (see the note below). TensorFold 1.1.6 does not support this model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows`:
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; the output is reproducible.

## Summary

Based on all data collected in the short-prompt B4 concurrency comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b4-short -->

- **First in throughput.** The throughput for four requests together is 427.0 tok/s, first in both rounds; 4.0% above Splash auto (410.5) and 14.4% above Splash all (373.4); oMLX reaches 243.8. By prompt set, both Splash settings matched or exceeded IronMLX on the first set, and IronMLX's lead comes from the second and third sets (see [Scope and limitations](#scope-and-limitations)).
- **Shortest median time to first token.** The median of the four requests' times to first token is 109 ms, first in both rounds; both Splash settings take about 150 ms and oMLX 331 ms.
- **Third in the slowest request's first token.** The slowest of the four requests receives its first output at 167 ms, 11.5% later than Splash all (150 ms) and 10.2% later than Splash auto (151 ms), in both rounds.
- **Completion: the whole batch finishes first, the median request later.** The slowest request completes at 2.40 s, the shortest in both rounds (Splash auto: 2.50 s); but the median completion time of the four requests is 2.05 s, 10.6% longer than Splash auto (1.86 s) and close to Splash all (2.04 s).
- **Lowest memory.** Peak memory while serving is 20.5 GiB, the lowest in both rounds; both Splash settings use about 21.5–21.6 GiB and oMLX 24.5 GiB.

### Throughput vs. time to first token

Batch throughput is the total number of tokens the four requests actually output divided by the batch completion time (from the earliest request send to the end of the last request), and reflects the server's total capacity under concurrent load. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b4-short -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first six rows is in bold, and the last two rows are auxiliary. On the first six metrics, the order of the configurations was the same in both rounds.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Batch throughput (tok/s, higher is better)¹ | **427.0** (423.8–⁠430.3) | 243.8 (241.0–⁠246.6) | 410.5 (401.3–⁠419.7) | 373.4 (372.3–⁠374.5) |
| Median time to first token (ms) | **109** (105–⁠112) | 331 (316–⁠346) | 151 (150–⁠152) | 149 (149–⁠149) |
| Slowest request's first token (ms) | 167 (164–⁠170) | 410 (387–⁠433) | 151 (150–⁠153) | **150** (150–⁠150) |
| Median completion time (s) | 2.05 (2.04–⁠2.07) | 3.79 (3.76–⁠3.82) | **1.86** (1.83–⁠1.88) | 2.04 (2.04–⁠2.05) |
| Slowest request's completion (s)¹ | **2.40** (2.38–⁠2.42) | 4.20 (4.15–⁠4.25) | 2.50 (2.44–⁠2.55) | 2.74 (2.73–⁠2.75) |
| Peak memory while serving (GiB) | **20.5** (20.4–⁠20.5) | 24.5 (24.5–⁠24.5) | 21.6 (21.6–⁠21.6) | 21.5 (21.5–⁠21.5) |
| Per-request decode speed (tok/s, auxiliary) | 131.0 (130.1–⁠131.9) | 73.1 (71.7–⁠74.6) | 155.9 (152.8–⁠159.0) | 136.3 (135.9–⁠136.7) |
| Decode concurrency (auxiliary)² | 3.37 (3.36–⁠3.37) | 3.51 (3.50–⁠3.52) | 2.84 (2.81–⁠2.88) | 2.95 (2.92–⁠2.97) |

> [!NOTE]
> ¹ Batch throughput is derived from the batch completion time (tokens actually output ÷ batch completion time; 1,024 tokens for every batch in this run), and the batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 1.3 ms), so these two rows reflect the same result and should not be counted as two separate advantages.

> [!NOTE]
> ² Decode concurrency was derived after the test from the client timestamps of each scored batch; it is not part of the test protocol written beforehand. It is the four requests' decode times added up ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`). 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> Peak memory while serving is measured during the scored batch. While loading the model and warming up, IronMLX peaks higher (about 22.1 GiB) than while serving; this page does not compare memory during loading.

### Splash's two settings

For what the two Splash settings mean, see the [B1 page](qwen36-moe-b1-short.md#splashs-two-settings). With four concurrent requests, Splash auto had 9.9% higher batch throughput than Splash all (410.5 against 373.4 tok/s) and a 9.1% shorter median completion time (1.86 against 2.04 s).

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-short -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX dev (c6ef14ec2):** the four requests receive their first tokens at about 48, 87, 125, and 167 ms; the requests being decoded are then verified together in ragged linear batches (no draft tree for this model). About 96% of all verification windows batched several requests, and the rest served a single remaining request; every window carried drafts. The four requests complete between about 1.8 and 2.4 s.
- **oMLX 0.7.2:** runs batched speculative decoding for the four requests (its log names this batched path "Lightning MTP"); first tokens arrive between about 177 and 410 ms, and its decode concurrency is the highest (3.51). During batching it calibrates batched against standard decoding, and individual requests switch to standard decoding for a stretch; in this run 3 requests finished on standard decoding after being parked.
- **Splash 1.4.1 auto and all:** in each batch one request receives its first token first (about 45 ms), and the other three receive theirs at the same moment (about 150 ms); the first request completes earliest, and the whole batch completes at about 2.5 s (default setting) and 2.7 s (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 28, 2026 (07:29–08:21 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none`; default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.4.1 auto | Official bottle | `--max-context 8192`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.4.1 all | As above | Splash 1.4.1 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.4.1 auto's engine process not to) |
| TensorFold 1.1.6 | — | Not included: according to its [1.1.6 documentation](https://github.com/ashhart/TensorFold/blob/v1.1.6/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`), which runs only on the older Python engine |

Common settings for all applications: 4-bit draft model, 8,192-token context, capacity 4 where configurable, and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All three applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens) in three sets of four, so every batch has four different prompts.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four, then one scored batch of four released together (largest send spread 1.3 ms), then stops the server. Two rounds × four configurations × three prompt sets = 24 sessions; the orders of the two rounds are ABCS and SCBA (reversed; A is IronMLX, B oMLX, C Splash auto, and S Splash all), so each configuration has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Decode concurrency was derived after the test from the same timestamps, without a rerun; see the notes under [Comparison by metric](#comparison-by-metric) for its definition.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.
- **Output reproducibility:** on the same prompt set across the two rounds, all 12 IronMLX requests were byte-identical; 2 requests of Splash all, 9 of Splash auto, and 11 of oMLX differed between the rounds. IronMLX's output also did not depend on the other requests in the batch (the same prompt gave the same output in batches of different composition), but on this model it is not identical to the output when served alone on the [B1 page](qwen36-moe-b1-short.md) (4 of 6 tasks differ): on this model IronMLX verifies batched requests with different arithmetic than a single request.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration for reproducible output); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- **The two completion measures point different ways.** IronMLX finishes the whole batch first (shortest slowest-request completion), but its median request completes later than with Splash auto; the two reflect "all four requests done" and "a typical request done", and should be read together.
- **The throughput lead depends on the prompt set.** On the first of the three prompt sets, Splash all (462.2 and 454.6 tok/s in the two rounds) and Splash auto (442.0 and 455.6) matched or exceeded IronMLX (442.0 and 448.0); IronMLX led both Splash settings on the second and third sets. With other prompts, the throughput order may differ.
- Long prompts, other models, other concurrency levels, or real multi-user traffic.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Batch-to-batch variation. Across the two rounds on the same prompt set, oMLX's batch throughput differed by up to 12.6% (first set: 219.9 to 251.5 tok/s), Splash auto by 4.4%, and IronMLX and Splash all by at most 1.6%.
