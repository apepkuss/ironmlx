# Qwen3.6 35B A3B B4 30K-prompt concurrency performance comparison

[简体中文](../zh-CN/benchmarks/qwen36-moe-b4-30k.md)

> [!NOTE]
> This page compares concurrent requests with 30K-token prompts. For short prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B4 concurrency
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-30k.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with four requests arriving together. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to several users or agents handing long documents or parts of a codebase to the same local server at once; all four 30K inputs have to be read, so the last request waits about half a minute. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the [B1 page](qwen36-moe-b1-30k.md) (see the note below). TensorFold 1.1.6 does not support this model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows` (explained on the [short-prompt B1 page](qwen36-moe-b1-short.md#splashs-two-settings)):
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; its output is reproducible for single requests but was not under the concurrent load of this page (see [Splash's two settings](#splashs-two-settings)).

## Summary

Based on all data collected in the 30K-prompt B4 concurrency comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b4-30k -->

- **Shortest median time to first token.** The median of the four requests' times to first token is 17.2 s, first in both rounds, but by less than 1%: Splash auto takes 17.3 s and Splash all 17.3 s; oMLX takes 27.1 s.
- **Shortest median completion time.** The median of the four requests' completion times is 18.4 s, first in both rounds; Splash auto takes 21.3 s, 15.7% longer than IronMLX, Splash all 21.4 s, and oMLX 32.1 s.
- **Highest output speed while decoding.** During the time in which a request is decoding, IronMLX outputs 208.7 tokens per second, first in both rounds and about 2.6 times Splash auto's 81.4; oMLX reaches 78.1 and Splash all 77.8. Each IronMLX request completes 0.9–1.6 s after its first token.
- **Lowest memory.** Peak memory while serving is 22.0 GiB, the lowest in both rounds; both Splash settings use 22.6 GiB and oMLX 34.3 GiB.
- **Slowest prefill, so the batch finishes slightly later.** The four 30K inputs total 122,928 tokens; IronMLX's batch prefill speed is 4,228 tok/s, the slowest in both rounds, 2.8% below oMLX (4,352) and 0.6–0.8% below both Splash settings (4,252–4,264). At 30K, 84–95% of each application's batch time is spent reading the four inputs, so IronMLX's decode advantage barely shows in the batch results: its slowest request's first token (29.1 s) is the latest, its slowest request's completion (30.7 s) is 1.4% later than Splash auto's (30.3 s) and the same as Splash all's (30.7 s), and its batch throughput (33.3 tok/s) is 1.4% below Splash auto (33.8) and the same as Splash all (33.3).
- **Reproducible output.** Each IronMLX task's output was byte-identical in every batch and identical to the output when served alone on the [B1 page](qwen36-moe-b1-30k.md); the outputs of oMLX and both Splash settings changed from batch to batch.

### Prefill speed and time to first token

Batch prefill speed is the four requests' total input tokens divided by the time from the earliest request send to the last request's first token, which shows how fast the server reads long inputs; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long inputs and waiting in the queue. Lower right is better in the chart.

<!-- benchmark:scatter qwen36-moe-b4-30k -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first seven rows is in bold, and the last three rows are auxiliary.

| Metric | IronMLX 0.2.0-rc.4 | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Median time to first token (s) | **17.2** (17.2–⁠17.2) | 27.1 (27.0–⁠27.2) | 17.3 (17.3–⁠17.3) | 17.3 (17.3–⁠17.3) |
| Slowest request's first token (s) | 29.1 (28.7–⁠29.4) | **28.2** (28.2–⁠28.3) | 28.8 (28.7–⁠28.9) | 28.9 (28.7–⁠29.1) |
| Median completion time (s) | **18.4** (18.3–⁠18.6) | 32.1 (31.1–⁠33.2) | 21.3 (21.3–⁠21.3) | 21.4 (21.1–⁠21.7) |
| Slowest request's completion (s) | 30.7 (30.5–⁠30.9) | 34.0 (33.4–⁠34.6) | **30.3** (30.2–⁠30.4) | 30.7 (30.7–⁠30.8) |
| Peak memory while serving (GiB) | **22.0** (22.0–⁠22.0) | 34.3 (34.1–⁠34.5) | 22.6 (22.6–⁠22.6) | 22.6 (22.6–⁠22.6) |
| Batch prefill speed (input tok/s, higher is better)¹ | 4,228 (4,178–⁠4,279) | **4,352** (4,338–⁠4,366) | 4,264 (4,248–⁠4,280) | 4,252 (4,224–⁠4,280) |
| Output speed while decoding (output tok/s, higher is better)¹ | **208.7** (208.6–⁠208.9) | 78.1 (74.5–⁠81.7) | 81.4 (81.2–⁠81.7) | 77.8 (77.0–⁠78.7) |
| Decode concurrency (auxiliary)¹ | 1.00 (1.00–⁠1.00) | 1.58 (1.51–⁠1.66) | 1.00 (1.00–⁠1.00) | 1.00 (1.00–⁠1.00) |
| Batch throughput (output tok/s, auxiliary)² | 33.35 (33.17–⁠33.53) | 30.12 (29.61–⁠30.62) | 33.81 (33.69–⁠33.94) | 33.31 (33.24–⁠33.39) |
| Per-request decode speed (tok/s, auxiliary)³ | 223.5 (223.2–⁠223.8) | 47.5 (44.9–⁠50.0) | 87.7 (87.5–⁠87.9) | 84.0 (81.2–⁠86.8) |

The order was the same in both rounds for the slowest request's first token and batch prefill speed (oMLX, Splash auto, Splash all, IronMLX), per-request decode speed (IronMLX, Splash auto, Splash all, oMLX), and decode concurrency. IronMLX was first and oMLX last in both rounds for median time to first token, median completion time, memory, and output speed while decoding; in these metrics the two Splash settings swapped places between the rounds (and in output speed while decoding oMLX was second in round 2). For the slowest request's completion and batch throughput, Splash auto led in both rounds, and IronMLX and Splash all swapped places between the rounds.

> [!NOTE]
> ¹ These three metrics were derived after the test from the client timestamps of each scored batch; they are not part of the test protocol written beforehand. They are defined as follows, and all of them include transfer time and the generation of the first output token, so they are not the engines' internal timings:
>
> - Batch prefill speed: the four requests' input tokens (server counts, 122,928 in total) ÷ the time from the earliest request send to the last request's first output.
> - Output speed while decoding: the batch's tokens actually output ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`; the union of the four intervals). Other requests' inputs may be read during this time, so it is the effective decode speed under load.
> - Decode concurrency: the four requests' decode times added up ÷ that union. 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> ² Batch throughput is the batch's tokens actually output ÷ the batch completion time (from the earliest send to the last `[DONE]`). At 30K the batch time is mostly spent reading inputs, so this metric mainly reflects prefill speed and is shown for reference only. Every batch output 1,024 tokens. The batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 2.9 ms).

> [!NOTE]
> ³ Per-request decode speed depends on how each application schedules the four requests: a request that decodes on its own before or after the other requests are read runs close to single-request speed, while one that alternates with the other requests' prefill runs much slower.

> [!NOTE]
> Peak memory while serving is measured during the scored batch and includes the KV cache of four 30K inputs. Peaks while loading the model and warming up: about 22.1 GiB for IronMLX, about the same as while serving; about 23.4–23.9 GiB for oMLX and 21.2–21.3 GiB for both Splash settings. This page does not compare memory during loading.

### Splash's two settings

With four concurrent 30K requests, the two Splash settings have almost the same time to first token; Splash auto's batch throughput is 1.5% higher than Splash all's (33.81 against 33.31 tok/s), and its slowest request completes 1.5% sooner (30.3 against 30.7 s).

Unlike single requests, Splash all's output was not reproducible on this page either: the same prompt produced different outputs in different batches (3–4 per task), all of them different from the output when served alone on the [B1 page](qwen36-moe-b1-30k.md). This page does not establish why.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-30k -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0-rc.4:** reads the four inputs one after another, with first tokens at about 6.1, 13.5, 20.9, and 29.1 s; each request completes 0.9–1.6 s after its first token, and the completion log shows that every request was decoded on its own (batch width 1). The next request's first token arrives 6.4–6.6 s after the previous request completes, about the same as a single request's time to first token on the [B1 page](qwen36-moe-b1-30k.md) (6.46 s): while one request decodes, the next request's input is not read at the same time. All 2,500 verification windows carried drafts, as decided by the default window-cost policy.
- **oMLX 0.7.2:** reads the first request on its own, with its first token at about 6.7 s; the other three inputs are read together, with first tokens at about 26–28 s. These requests are then decoded together with batched speculative decoding (its log names this batched path "Lightning MTP"); during batching it calibrates batched against standard decoding, and 5 of the 24 requests finished with standard decoding after being parked. With a decode concurrency of 1.58, it is the only application in which requests decoded at the same time.
- **Splash 1.4.1 auto and all:** read the four inputs one after another, with first tokens at about 6.3, 13.6–13.7, 20.9–21.1, and 28.8–28.9 s; the previous request decodes while the next input is being read, and the next request's first token arrives only 3.0–4.5 s after the previous request completes. First tokens therefore come at about the same time as with IronMLX, but each request takes 1.5–4.9 s to decode, so per-request decode speed is lower.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 30, 2026 (17:56–18:59 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session of the formal run was the exception, waiting for the thermal condition only and starting about 60 s after the previous run's server stopped. Thermal pressure was sampled every second during each session and stayed nominal during every scored batch |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.4 | Release build (`cargo build --locked --release`) of commit `93e9b3a69`; server and MLX Metal runtime frozen together. The `v0.2.0-rc.4` tag adds only release metadata and documentation on top of this commit, so the measured code path is that of 0.2.0-rc.4 | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.4.1 auto | Official bottle | `--max-context 32768`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.4.1 all | As above | Splash 1.4.1 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.4.1 auto's engine process not to) |
| TensorFold 1.1.6 | — | Not included: according to its [1.1.6 documentation](https://github.com/ashhart/TensorFold/blob/v1.1.6/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`), which runs only on the older Python engine |

Common settings for all applications: 4-bit draft model, 32,768-token context, capacity 4 where configurable, and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy. Each server's effective context was verified as 32,768 from its own interface or settings after startup.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All three applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** the six 30K prompts of the B1 page (three coding, three knowledge), each 30,732 tokens with the chat template, in three sets of four, so every batch has four different prompts. Every request's server-side prompt token count was verified as 30,732; no input was truncated or dropped.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four (short prompts), then one scored batch of four released together (largest send spread 2.9 ms), then stops the server. Two rounds × four configurations × three prompt sets = 24 sessions; the orders of the two rounds are ABCS and SCBA (reversed; A is IronMLX, B oMLX, C Splash auto, and S Splash all), so each configuration has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer; a request that ends by length counts 256) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test from the same timestamps; see the notes under [Comparison by metric](#comparison-by-metric) for their definitions.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. The check of IronMLX's draft evidence allows windows without drafts up to 10% of a batch's windows (a rule introduced before the October 28 round, when the default window-cost policy produced a few such windows at 30K, and replayed on real records then); in this run every window carried drafts.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run all 96 recounts were exactly 256.
- **Output reproducibility:** each task appears in 4 batches (two rounds × the two prompt sets that contain it). IronMLX's 4 outputs of each task were byte-identical and identical to the output when served alone on the B1 page; Splash all produced 3–4 different outputs per task, oMLX 3–4, and Splash auto 4 different outputs for every task.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration that verifies the whole draft block in every cycle); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a release candidate.** 0.2.0-rc.4 is a pre-release, not the final 0.2.0 release; the final release may perform differently.
- **Results without a clear order.** IronMLX and Splash all differ by 0.1% in batch throughput and in the slowest request's completion, and swapped order between the rounds; IronMLX's lead in median time to first token over both Splash settings is under 1%; batch prefill speed differs by under 1% between IronMLX and both Splash settings. The two Splash settings swapped places between the rounds in several metrics.
- **Metrics derived after the test.** Batch prefill speed, output speed while decoding, and decode concurrency were defined after the October 28 round was complete, from client timestamps recorded during the test, without a rerun; they were not comparisons fixed beforehand.
- Other input lengths, other models, other concurrency levels, or real multi-user traffic. This page covers one long input of about 30K tokens only, with four requests arriving together.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the output changes of oMLX and Splash affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache.
- Batch-to-batch variation. On the same prompt set, oMLX's batch throughput differed by up to 7.0% between the two rounds (first set: 29.25 to 31.29 tok/s), IronMLX's by 1.1%, Splash auto's by 0.8%, and Splash all's by 0.5%.
