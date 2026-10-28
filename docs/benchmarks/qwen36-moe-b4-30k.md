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

- **Shortest median completion time.** The median of the four requests' completion times is 20.3 s, first in both rounds; Splash auto takes 21.3 s, 5.0% longer than IronMLX, Splash all 21.4 s, and oMLX 32.9 s.
- **Highest output speed while decoding.** During the time in which a request is decoding, IronMLX outputs 198.0 tokens per second, first in both rounds and about 2.4 times Splash auto's 83.0; Splash all reaches 77.9 and oMLX 75.1. Each IronMLX request completes 0.7–1.9 s after its first token.
- **Lowest memory.** Peak memory while serving is 22.1 GiB, the lowest in both rounds; both Splash settings use about 22.6 GiB and oMLX 34.2 GiB.
- **Third in median time to first token.** The median of the four requests' times to first token is 19.1 s, 10.0% longer than Splash auto (17.3 s) and 9.4% longer than Splash all (17.4 s) in both rounds; oMLX takes 27.2 s.
- **Slowest prefill, so the batch finishes later.** The four 30K inputs total 122,928 tokens; IronMLX's batch prefill speed is 3,786 tok/s, the slowest in both rounds, 12.8% below oMLX (4,340) and 10.7–10.8% below both Splash settings (about 4,240). At 30K, 83–95% of each application's batch time is spent reading the four inputs, so IronMLX's decode advantage does not show in the batch results: its slowest request's first token (32.5 s) is the latest, its slowest request's completion (34.0 s) ties with oMLX (34.1 s) for third and is 11.7% later than Splash auto's (30.4 s), and its batch throughput (30.1 tok/s) also ties with oMLX for third, 10.4% below Splash auto (33.7).
- **Reproducible output.** Each IronMLX task's output was byte-identical in every batch and identical to the output when served alone on the [B1 page](qwen36-moe-b1-30k.md); the outputs of oMLX and both Splash settings changed from batch to batch.

### Prefill speed and time to first token

Batch prefill speed is the four requests' total input tokens divided by the time from the earliest request send to the last request's first token, which shows how fast the server reads long inputs; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long inputs and waiting in the queue. Lower right is better in the chart.

<!-- benchmark:scatter qwen36-moe-b4-30k -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first seven rows is in bold, and the last three rows are auxiliary.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Median time to first token (s) | 19.1 (18.8–⁠19.3) | 27.2 (27.2–⁠27.3) | **17.3** (17.2–⁠17.4) | 17.4 (17.4–⁠17.5) |
| Slowest request's first token (s) | 32.5 (32.4–⁠32.6) | **28.3** (28.3–⁠28.4) | 29.0 (28.9–⁠29.1) | 29.0 (28.7–⁠29.2) |
| Median completion time (s) | **20.3** (20.3–⁠20.3) | 32.9 (32.4–⁠33.3) | 21.3 (21.1–⁠21.5) | 21.4 (21.3–⁠21.4) |
| Slowest request's completion (s) | 34.0 (33.9–⁠34.1) | 34.1 (33.6–⁠34.5) | **30.4** (30.3–⁠30.5) | 31.2 (30.9–⁠31.4) |
| Peak memory while serving (GiB) | **22.1** (22.1–⁠22.1) | 34.2 (34.0–⁠34.5) | 22.6 (22.6–⁠22.6) | 22.6 (22.5–⁠22.6) |
| Batch prefill speed (input tok/s, higher is better)¹ | 3,786 (3,775–⁠3,798) | **4,340** (4,329–⁠4,351) | 4,244 (4,231–⁠4,258) | 4,242 (4,207–⁠4,277) |
| Output speed while decoding (output tok/s, higher is better)¹ | **198.0** (197.0–⁠199.1) | 75.1 (71.5–⁠78.8) | 83.0 (79.0–⁠86.9) | 77.9 (76.9–⁠78.9) |
| Decode concurrency (auxiliary)¹ | 1.00 (1.00–⁠1.00) | 1.69 (1.67–⁠1.71) | 1.00 (1.00–⁠1.00) | 1.00 (1.00–⁠1.00) |
| Batch throughput (output tok/s, auxiliary)² | 30.14 (30.06–⁠30.21) | 30.06 (29.64–⁠30.48) | 33.65 (33.56–⁠33.75) | 32.87 (32.57–⁠33.18) |
| Per-request decode speed (tok/s, auxiliary)³ | 209.4 (209.3–⁠209.6) | 47.7 (45.7–⁠49.6) | 87.2 (85.4–⁠89.0) | 84.2 (82.4–⁠86.0) |

The order was the same in both rounds for median time to first token, output speed while decoding, and decode concurrency. IronMLX was first and oMLX last in both rounds for median completion time, memory, and per-request decode speed; oMLX was first and IronMLX last in both rounds for the slowest request's first token and batch prefill speed; in these metrics the two Splash settings swapped places between the rounds. For the slowest request's completion and batch throughput, Splash auto and Splash all led in both rounds, and IronMLX and oMLX swapped places between the rounds.

> [!NOTE]
> ¹ These three metrics were derived after the test from the client timestamps of each scored batch; they are not part of the test protocol written beforehand. They are defined as follows, and all of them include transfer time and the generation of the first output token, so they are not the engines' internal timings:
>
> - Batch prefill speed: the four requests' input tokens (server counts, 122,928 in total) ÷ the time from the earliest request send to the last request's first output.
> - Output speed while decoding: the batch's tokens actually output ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`; the union of the four intervals). Other requests' inputs may be read during this time, so it is the effective decode speed under load.
> - Decode concurrency: the four requests' decode times added up ÷ that union. 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> ² Batch throughput is the batch's tokens actually output ÷ the batch completion time (from the earliest send to the last `[DONE]`). At 30K the batch time is mostly spent reading inputs, so this metric mainly reflects prefill speed and is shown for reference only. Every batch output 1,024 tokens (one Splash auto request recounted as 255 on the client, within the allowed difference, and counts as 256). The batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 2.7 ms).

> [!NOTE]
> ³ Per-request decode speed depends on how each application schedules the four requests: a request that decodes on its own before or after the other requests are read runs close to single-request speed, while one that alternates with the other requests' prefill runs much slower.

> [!NOTE]
> Peak memory while serving is measured during the scored batch and includes the KV cache of four 30K inputs. Peaks while loading the model and warming up: about 22.1 GiB for IronMLX, about the same as while serving; about 23.4–23.9 GiB for oMLX and 21.2–21.3 GiB for both Splash settings. This page does not compare memory during loading.

### Splash's two settings

With four concurrent 30K requests, the two Splash settings have almost the same time to first token; Splash auto's batch throughput is 2.4% higher than Splash all's (33.65 against 32.87 tok/s), and its slowest request completes 2.3% sooner (30.4 against 31.2 s).

Unlike single requests, Splash all's output was not reproducible on this page either: the same prompt produced different outputs in different batches (2–4 per task), all of them different from the output when served alone on the [B1 page](qwen36-moe-b1-30k.md). This page does not establish why.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-30k -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX dev (c6ef14ec2):** reads the four inputs one after another, with first tokens at about 6.8, 15.0, 23.3, and 32.5 s; each request completes 0.7–1.9 s after its first token, and the completion log shows that every request was decoded on its own (batch width 1). The next request's first token arrives 7.1–7.6 s after the previous request completes, about the same as a single request's time to first token on the [B1 page](qwen36-moe-b1-30k.md) (7.13 s): while one request decodes, the next request's input is not read at the same time. Two verification windows per batch carried no drafts (12 in all, 0.5% of 2,372 windows), as decided by the default window-cost policy; every other window carried drafts.
- **oMLX 0.7.2:** reads the first request on its own, with its first token at about 6.7 s; the other three inputs are read together, with first tokens at 26–28 s. These requests are then decoded together with batched speculative decoding (its log names this batched path "Lightning MTP"); during batching it calibrates batched against standard decoding, and some requests finish with standard decoding after being parked. With a decode concurrency of 1.69, it is the only application in which requests decoded at the same time.
- **Splash 1.4.1 auto and all:** read the four inputs one after another, with first tokens at about 6.3, 13.6, 21.0, and 29.0 s; the previous request decodes while the next input is being read, and the next request's first token arrives only 1.8–4.6 s after the previous request completes. First tokens therefore come earlier than with IronMLX, but each request takes 1.5–6.6 s to decode, so per-request decode speed is lower.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 28, 2026 (11:15–12:19 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal during every scored batch |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
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
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four (short prompts), then one scored batch of four released together (largest send spread 2.7 ms), then stops the server. Two rounds × four configurations × three prompt sets = 24 sessions; the orders of the two rounds are ABCS and SCBA (reversed; A is IronMLX, B oMLX, C Splash auto, and S Splash all), so each configuration has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer; a request that ends by length counts 256) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test from the same timestamps; see the notes under [Comparison by metric](#comparison-by-metric) for their definitions.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. In the preflight before the formal run, a check carried over from the short-prompt test required every IronMLX verification window to carry drafts, while the default window-cost policy produces a few windows without drafts at 30K; before the formal run, the check was changed to allow windows without drafts up to 10% of a batch's windows and replayed on existing real records.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run only one Splash auto request (Knowledge 1) recounted as 255; all other recounts were exactly 256.
- **Output reproducibility:** each task appears in 4 batches (two rounds × the two prompt sets that contain it). IronMLX's 4 outputs of each task were byte-identical and identical to the output when served alone on the B1 page; Splash all produced 2–4 different outputs per task, oMLX 2–4, and Splash auto 4 different outputs for every task.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration that verifies the whole draft block in every cycle); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- **Results without a clear order.** IronMLX and oMLX differ by only 0.3% in batch throughput and in the slowest request's completion, and each led in one round. The two Splash settings swapped places between the rounds in several metrics.
- **Metrics derived after the test.** Batch prefill speed, output speed while decoding, and decode concurrency were defined after the test was complete, from client timestamps already recorded during the test, without a rerun; they were not comparisons fixed beforehand.
- Other input lengths, other models, other concurrency levels, or real multi-user traffic. This page covers one long input of about 30K tokens only, with four requests arriving together.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the output changes of oMLX and Splash affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache.
- Batch-to-batch variation. On the same prompt set, oMLX's batch throughput differed by up to 5.2% between the two rounds (first set: 29.44 to 30.98 tok/s), Splash all's by 1.9%, Splash auto's by 1.2%, and IronMLX's by 1.1%.
