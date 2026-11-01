# Qwen3.6 35B A3B B4 30K-prompt concurrency performance comparison

[简体中文](../zh-CN/benchmarks/qwen36-moe-b4-30k.md)

> [!NOTE]
> This page compares concurrent requests with 30K-token prompts. For short prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B4 concurrency
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-30k.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with four requests arriving together. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to several users or agents handing long documents or parts of a codebase to the same local server at once; all four 30K inputs have to be read, so the last request waits about half a minute. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the [B1 page](qwen36-moe-b1-30k.md) (see the note below). TensorFold 1.2.0 does not support the incoai DFlash2 draft model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows` (explained on the [short-prompt B1 page](qwen36-moe-b1-short.md#splashs-two-settings)):
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; its output is reproducible for single requests but was not under the concurrent load of this page (see [Splash's two settings](#splashs-two-settings)).

## Summary

Based on all data collected in the 30K-prompt B4 concurrency comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b4-30k -->

- **Shortest median completion time.** The median of the four requests' completion times is 18.7 s, first in both rounds; Splash auto takes 20.9 s, 11.7% longer than IronMLX, Splash all 21.8 s, and oMLX 32.2 s.
- **Highest output speed while decoding.** During the time in which a request is decoding, IronMLX outputs 206.3 tokens per second, first in both rounds and about 2.4 times Splash auto's 85.4; oMLX reaches 78.6 and Splash all 72.2. Each IronMLX request completes 1.1–1.7 s after its first token.
- **Lowest memory.** Peak memory while serving is 22.0 GiB, the lowest in both rounds; both Splash settings use 22.5 GiB and oMLX 29.4 GiB.
- **Third in median time to first token, by less than 1%.** The median of the four requests' times to first token is 17.6 s, third in both rounds and 0.6–0.7% longer than both Splash settings (17.5 s each); oMLX takes 27.6 s.
- **Slowest prefill, so the batch finishes later.** The four 30K inputs total 122,928 tokens; IronMLX's batch prefill speed is 4,140 tok/s, the slowest in both rounds, 3.4% below Splash auto (4,285), 2.5% below oMLX (4,246), and 1.4% below Splash all (4,197). At 30K, 86–95% of each application's batch time is spent reading the four inputs, so IronMLX's decode advantage barely shows in the batch results: its slowest request's first token (29.7 s) is the latest, its slowest request's completion (31.3 s) is 3.8% later than Splash auto's (30.2 s) and 1.6% later than Splash all's (30.8 s), and its batch throughput (32.7 tok/s) is 3.6% below Splash auto (33.9) and 1.6% below Splash all (33.2).
- **Reproducible output.** Each IronMLX task's output was byte-identical in every batch and identical to the output when served alone on the [B1 page](qwen36-moe-b1-30k.md); the outputs of oMLX and both Splash settings changed from batch to batch.

### Prefill speed and time to first token

Batch prefill speed is the four requests' total input tokens divided by the time from the earliest request send to the last request's first token, which shows how fast the server reads long inputs; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long inputs and waiting in the queue. Lower right is better in the chart.

<!-- benchmark:scatter qwen36-moe-b4-30k -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first seven rows is in bold, and the last three rows are auxiliary.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Median time to first token (s) | 17.6 (17.6–⁠17.6) | 27.6 (27.5–⁠27.6) | **17.5** (17.4–⁠17.6) | **17.5** (17.4–⁠17.5) |
| Slowest request's first token (s) | 29.7 (29.5–⁠29.9) | 29.0 (28.8–⁠29.1) | **28.7** (28.5–⁠28.8) | 29.3 (29.3–⁠29.3) |
| Median completion time (s) | **18.7** (18.6–⁠18.8) | 32.2 (31.6–⁠32.7) | 20.9 (20.6–⁠21.2) | 21.8 (21.7–⁠21.9) |
| Slowest request's completion (s) | 31.3 (31.3–⁠31.4) | 33.5 (33.2–⁠33.8) | **30.2** (30.1–⁠30.3) | 30.8 (30.8–⁠30.9) |
| Peak memory while serving (GiB) | **22.0** (22.0–⁠22.0) | 29.4 (29.3–⁠29.5) | 22.5 (22.5–⁠22.5) | 22.5 (22.5–⁠22.5) |
| Batch prefill speed (input tok/s, higher is better)¹ | 4,140 (4,110–⁠4,170) | 4,246 (4,229–⁠4,262) | **4,285** (4,262–⁠4,308) | 4,197 (4,197–⁠4,198) |
| Output speed while decoding (output tok/s, higher is better)¹ | **206.3** (206.1–⁠206.4) | 78.6 (78.6–⁠78.6) | 85.4 (81.7–⁠89.0) | 72.2 (71.1–⁠73.3) |
| Decode concurrency (auxiliary)¹ | 1.00 (1.00–⁠1.00) | 1.50 (1.49–⁠1.51) | 1.00 (1.00–⁠1.00) | 1.00 (1.00–⁠1.00) |
| Batch throughput (output tok/s, auxiliary)² | 32.67 (32.64–⁠32.70) | 30.54 (30.26–⁠30.82) | 33.90 (33.82–⁠33.98) | 33.20 (33.17–⁠33.22) |
| Per-request decode speed (tok/s, auxiliary)³ | 220.3 (219.7–⁠220.9) | 52.2 (52.2–⁠52.3) | 89.9 (88.5–⁠91.3) | 82.8 (82.7–⁠82.8) |

The order was the same in both rounds in every metric except median time to first token: the slowest request's first token and batch prefill speed (Splash auto, oMLX, Splash all, IronMLX); median completion time, memory, and per-request decode speed (IronMLX, Splash auto, Splash all, oMLX); output speed while decoding (IronMLX, Splash auto, oMLX, Splash all); the slowest request's completion and batch throughput (Splash auto, Splash all, IronMLX, oMLX); and decode concurrency. In median time to first token, IronMLX was third and oMLX fourth in both rounds, and the two Splash settings swapped places between the rounds.

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
> Peak memory while serving is measured during the scored batch and includes the KV cache of four 30K inputs. Peaks while loading the model and warming up: about 22.1 GiB for IronMLX, about the same as while serving; about 23.4–23.9 GiB for oMLX and 21.0–21.2 GiB for both Splash settings. This page does not compare memory during loading.

### Splash's two settings

With four concurrent 30K requests, the two Splash settings have almost the same median time to first token; Splash auto's batch throughput is 2.1% higher than Splash all's (33.90 against 33.20 tok/s), and its slowest request completes 2.1% sooner (30.2 against 30.8 s).

Unlike single requests, Splash all's output was not reproducible on this page either: the same prompt produced different outputs in different batches (2–4 per task), all of them different from the output when served alone on the [B1 page](qwen36-moe-b1-30k.md). This page does not establish why.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen36-moe-b4-30k -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0:** reads the four inputs one after another, with first tokens at about 6.2, 13.8, 21.4, and 29.7 s; each request completes 1.1–1.7 s after its first token, and the completion log shows that every request was decoded on its own (batch width 1). The next request's first token arrives 6.5–6.9 s after the previous request completes, about the same as or slightly longer than a single request's time to first token on the [B1 page](qwen36-moe-b1-30k.md) (6.47 s): while one request decodes, the next request's input is not read at the same time. All 2,500 verification windows carried drafts, as decided by the default window-cost policy.
- **oMLX 0.7.3rc1:** reads the first request on its own, with its first token at about 6.8 s; the other three inputs are read together, with first tokens at about 26–29 s. These requests are then decoded together with batched speculative decoding (its log names this batched path "Lightning MTP"); during batching it calibrates batched against standard decoding, and 8 of the 24 requests finished with standard decoding after being parked. With a decode concurrency of 1.50, it is the only application in which requests decoded at the same time.
- **Splash 1.5.0 auto and all:** read the four inputs one after another, with first tokens at about 6.3, 13.8, 21.1, and 28.7–29.3 s; the previous request decodes while the next input is being read, and the next request's first token arrives only 2.8–4.4 s after the previous request completes. First tokens therefore come at about the same time as with IronMLX, but each request takes 1.5–5.3 s to decode, so per-request decode speed is lower.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | November 1, 2026 (01:32–02:35 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session of the formal run was the exception, waiting for the thermal condition only and starting about 80 s after the previous run's server stopped. Thermal pressure was sampled every second during each session and stayed nominal during every scored batch |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.5.0 auto | Official bottle | `--max-context 32768`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.5.0 all | As above | Splash 1.5.0 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.5.0 auto's engine process not to) |
| TensorFold 1.2.0 | — | Not included: TensorFold 1.2.0 does not support the incoai DFlash2 draft model used by the other applications on this page. According to its [1.2.0 documentation](https://github.com/ashhart/TensorFold/blob/v1.2.0/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`); on Macs the model runs only on the Python engine frozen at 0.6.6, which drafts with the model's own MTP layer or with z-lab DFlash v1 |

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
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four (short prompts), then one scored batch of four released together (largest send spread 2.9 ms), then stops the server. Two rounds × four configurations × three prompt sets = 24 sessions (the protocol fixed two rounds before the test, with no rule for adding rounds); the orders of the two rounds are ABCS and SCBA (reversed; A is IronMLX, B oMLX, C Splash auto, and S Splash all), so each configuration has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer; a request that ends by length counts 256) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test from the same timestamps; see the notes under [Comparison by metric](#comparison-by-metric) for their definitions.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. The check of IronMLX's draft evidence allows windows without drafts up to 10% of a batch's windows (a rule introduced before the October 28 round, when the default window-cost policy produced a few such windows at 30K, and replayed on real records then); in this run every window carried drafts.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run all 96 recounts were exactly 256.
- **Output reproducibility:** each task appears in 4 batches (two rounds × the two prompt sets that contain it). IronMLX's 4 outputs of each task were byte-identical and identical to the output when served alone on the B1 page; Splash all produced 2–4 different outputs per task, oMLX 3–4, and Splash auto 4 different outputs for every task.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration that verifies the whole draft block in every cycle); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because it does not support the incoai DFlash2 draft model, which says nothing about its performance in other setups. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **Results without a clear order.** IronMLX's median time to first token is under 1% longer than both Splash settings' (0.6–0.7%); the two Splash settings differ by under 0.1% in median time to first token and swapped places between the rounds, and differ by 0.1% in peak memory while serving.
- **Metrics derived after the test.** Batch prefill speed, output speed while decoding, and decode concurrency were defined after the October 28 round was complete, from client timestamps recorded during the test, without a rerun; they were not comparisons fixed beforehand.
- Other input lengths, other models, other concurrency levels, or real multi-user traffic. This page covers one long input of about 30K tokens only, with four requests arriving together.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the output changes of oMLX and Splash affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache.
- Batch-to-batch variation. On the same prompt set, oMLX's batch throughput differed by up to 5.0% between the two rounds (first set: 29.71 to 31.21 tok/s), IronMLX's by 1.3%, Splash auto's by 1.9%, and Splash all's by 1.0%.
