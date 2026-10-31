# Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen36-moe-b4-short.md)

> [!NOTE]
> This page compares concurrent requests with short prompts. For 30K-token prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-30k.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-30k.md). Short-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B4 concurrency
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with four requests arriving together. All requests use short prompts (40–83 input tokens) with an output limit of 256 tokens, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the [B1 page](qwen36-moe-b1-short.md) (see the note below). TensorFold 1.1.6 does not support this model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows`:
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; its output is reproducible for single requests but not fully under the concurrent load of this page (see [Workload and procedure](#workload-and-procedure)).

## Summary

Based on all data collected in the short-prompt B4 concurrency comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b4-short -->

- **First in throughput.** The throughput for four requests together is 428.9 tok/s, first in both rounds and on every prompt set; 3.5% above Splash auto (414.6) and 9.8% above Splash all (390.6); oMLX reaches 242.5.
- **Shortest median time to first token.** The median of the four requests' times to first token is 92 ms, first in both rounds; Splash auto takes 147 ms, Splash all 155 ms, and oMLX 348 ms.
- **Slowest request's first token: no clear order with Splash auto.** The slowest of the four requests receives its first output at 146 ms against 148 ms for Splash auto; each led in one round. Splash all takes 155 ms and oMLX 424 ms.
- **Completion: the whole batch finishes first, the median request later.** The slowest request completes at 2.39 s, the shortest in both rounds (Splash auto: 2.47 s); but the median completion time of the four requests is 1.97 s, 5.5% longer than Splash auto (1.87 s) and close to Splash all (2.01 s).
- **Lowest memory.** Peak memory while serving is 20.5 GiB, the lowest in both rounds; both Splash settings use about 21.5 GiB and oMLX 24.4 GiB.

### Throughput vs. time to first token

Batch throughput is the total number of tokens the four requests actually output divided by the batch completion time (from the earliest request send to the end of the last request), and reflects the server's total capacity under concurrent load. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b4-short -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first six rows is in bold, and the last two rows are auxiliary.

| Metric | IronMLX 0.2.0-rc.4 | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Batch throughput (tok/s, higher is better)¹ | **428.9** (427.3–⁠430.5) | 242.5 (234.5–⁠250.5) | 414.6 (406.4–⁠422.7) | 390.6 (384.3–⁠397.0) |
| Median time to first token (ms) | **92** (87–⁠97) | 348 (308–⁠388) | 147 (145–⁠150) | 155 (154–⁠155) |
| Slowest request's first token (ms) | **146** (142–⁠149) | 424 (401–⁠446) | 148 (145–⁠150) | 155 (154–⁠156) |
| Median completion time (s) | 1.97 (1.97–⁠1.98) | 3.88 (3.76–⁠3.99) | **1.87** (1.85–⁠1.89) | 2.01 (1.97–⁠2.05) |
| Slowest request's completion (s)¹ | **2.39** (2.38–⁠2.40) | 4.23 (4.09–⁠4.37) | 2.47 (2.42–⁠2.52) | 2.62 (2.58–⁠2.66) |
| Peak memory while serving (GiB) | **20.5** (20.4–⁠20.6) | 24.4 (24.3–⁠24.6) | 21.5 (21.5–⁠21.6) | 21.5 (21.5–⁠21.5) |
| Per-request decode speed (tok/s, auxiliary) | 137.2 (137.0–⁠137.4) | 71.8 (70.2–⁠73.3) | 151.5 (151.3–⁠151.6) | 143.4 (140.9–⁠145.9) |
| Decode concurrency (auxiliary)² | 3.35 (3.34–⁠3.35) | 3.46 (3.37–⁠3.54) | 2.91 (2.90–⁠2.93) | 2.94 (2.92–⁠2.96) |

The order was the same in both rounds for batch throughput, median time to first token, the slowest request's completion (IronMLX, Splash auto, Splash all, oMLX), memory (IronMLX, Splash all, Splash auto, oMLX), and per-request decode speed (Splash auto, Splash all, IronMLX, oMLX). For the slowest request's first token, IronMLX led in round 1 and Splash auto in round 2; for the median completion time, Splash auto led in both rounds, and IronMLX was second in round 1 and third, behind Splash all, in round 2.

> [!NOTE]
> ¹ Batch throughput is derived from the batch completion time (tokens actually output ÷ batch completion time; 1,024 tokens for every batch in this run), and the batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 1.3 ms), so these two rows reflect the same result and should not be counted as two separate advantages.

> [!NOTE]
> ² Decode concurrency was derived after the test from the client timestamps of each scored batch; it is not part of the test protocol written beforehand. It is the four requests' decode times added up ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`). 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> Peak memory while serving is measured during the scored batch. While loading the model and warming up, IronMLX peaks higher (about 22.1 GiB) than while serving; this page does not compare memory during loading.

### Splash's two settings

For what the two Splash settings mean, see the [B1 page](qwen36-moe-b1-short.md#splashs-two-settings). With four concurrent requests, Splash auto had 6.1% higher batch throughput than Splash all (414.6 against 390.6 tok/s) and a 7.1% shorter median completion time (1.87 against 2.01 s).

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-short -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0-rc.4:** the four requests receive their first tokens at about 41, 77, 109, and 147 ms; the requests being decoded are then verified together in ragged linear batches (no draft tree for this model). About 84% of the verification windows batched several requests (about 95% counted per request), and the rest served a single remaining request; every window carried drafts. The four requests complete between about 1.8 and 2.4 s.
- **oMLX 0.7.2:** runs batched speculative decoding for the four requests (its log names this batched path "Lightning MTP"); first tokens arrive between about 182 and 424 ms, and its decode concurrency is the highest (3.46). During batching it calibrates batched against standard decoding and parks the batch; in this run every batch was parked once, and 5 of the 24 requests finished on standard decoding after being parked.
- **Splash 1.4.1 auto and all:** in each batch one request receives its first token first (about 46 ms), and the other three receive theirs at the same moment (about 147 ms with the default setting, 155 ms with `--verify-rows all`); the first request completes earliest, and the whole batch completes at about 2.5 s (default setting) and 2.6 s (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 30, 2026 (17:04–17:56 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session of the formal run was the exception, waiting for the thermal condition only and starting about 60 s after the previous run's server stopped. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.4 | Release build (`cargo build --locked --release`) of commit `93e9b3a69`; server and MLX Metal runtime frozen together. The `v0.2.0-rc.4` tag adds only release metadata and documentation on top of this commit, so the measured code path is that of 0.2.0-rc.4 | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
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

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server (the client's recount with the target tokenizer, allowed to differ by up to 3 tokens, was 255 for one oMLX request and 257 for one Splash all request).
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens) in three sets of four, so every batch has four different prompts.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four, then one scored batch of four released together (largest send spread 1.3 ms), then stops the server. Two rounds × four configurations × three prompt sets = 24 sessions; the orders of the two rounds are ABCS and SCBA (reversed; A is IronMLX, B oMLX, C Splash auto, and S Splash all), so each configuration has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer; a request that ends by length counts 256) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Decode concurrency was derived after the test from the same timestamps, without a rerun; see the notes under [Comparison by metric](#comparison-by-metric) for its definition.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.
- **Output reproducibility:** on the same prompt set across the two rounds, all 12 IronMLX requests were byte-identical; 4 requests of Splash all, 5 of Splash auto, and 6 of oMLX differed between the rounds. IronMLX's output also did not depend on the other requests in the batch (the same prompt gave the same output in batches of different composition), but on this model it is not identical to the output when served alone on the [B1 page](qwen36-moe-b1-short.md) (4 of 6 tasks differ): on this model IronMLX verifies batched requests with different arithmetic than a single request.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration that verifies the whole draft block in every cycle); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a release candidate.** 0.2.0-rc.4 is a pre-release, not the final 0.2.0 release; the final release may perform differently.
- **The two completion measures point different ways.** IronMLX finishes the whole batch first (shortest slowest-request completion), but its median request completes later than with Splash auto; the two reflect "all four requests done" and "a typical request done", and should be read together.
- **Close results.** On the first prompt set, Splash all came within 0.4% of IronMLX's throughput in one round (463.1 against 465.0 tok/s); the slowest request's first token differs by 1.4% between IronMLX and Splash auto, and each led in one round; the median completion times of IronMLX and Splash all differ by 2.0%, and they swapped order between the rounds. None of these should be read as a clear advantage or disadvantage.
- Long prompts, other models, other concurrency levels, or real multi-user traffic. For concurrent requests with prompts of about 30K tokens, see [Qwen3.6 35B A3B B4 30K-prompt concurrency performance comparison](qwen36-moe-b4-30k.md).
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Batch-to-batch variation. Across the two rounds on the same prompt set, oMLX's batch throughput differed by up to 19.2% (first set: 223.3 to 266.2 tok/s), Splash all by 6.7% (first set: 433.9 to 463.1 tok/s), Splash auto by 4.0%, and IronMLX by 0.8%.
