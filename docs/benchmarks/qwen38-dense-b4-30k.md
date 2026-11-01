# Qwen3.8 27B B4 30K-prompt concurrency performance comparison

[简体中文](../zh-CN/benchmarks/qwen38-dense-b4-30k.md)

> [!NOTE]
> This page compares concurrent requests with 30K-token prompts. For short prompts, see [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md) and [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.8 27B B4 concurrency
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-30k.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with four requests arriving together. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to several users or agents handing long documents or parts of a codebase to the same local server at once; all four 30K inputs have to be read, so the waits run to minutes. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the 30K-prompt B4 concurrency comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b4-30k -->

- **Shortest median time to first token.** The median of the four requests' times to first token is 83.7 s, first in both rounds; Splash takes 86.9 s, 3.8% longer, and the two-round ranges of the two do not overlap. TensorFold takes 136.2 s and oMLX 159.7 s.
- **Shortest median completion time.** The median of the four requests' completion times is 87.2 s, first in both rounds and 9.5% shorter than Splash (96.3 s); oMLX takes 176.4 s and TensorFold 181.2 s.
- **Highest prefill speed.** The four 30K inputs total 122,928 tokens; IronMLX's batch prefill speed is 858 tok/s, first in both rounds and 3.9% higher than Splash's 826 tok/s; oMLX reaches 753 and TensorFold 619. At 30K, about 90–97% of each application's batch time is spent reading the four inputs, so prefill speed largely sets the wait. IronMLX is also first in both rounds for the slowest request's first token (143.3 s against Splash's 148.9 s) and completion (148.9 s against 153.9 s).
- **Highest output speed while decoding.** During the time in which a request is decoding, IronMLX outputs 67.6 tokens per second, first in both rounds and about twice Splash's 32.3; oMLX reaches 23.5 and TensorFold 7.2. Of IronMLX's 24 requests, 22 complete 2–8 s after their first token, while each Splash request takes about 5–17 s to decode.
- **Almost no batching at 30K.** IronMLX and Splash both have a decode concurrency of 1.00 (no two Splash requests ever decoded at the same time in its 6 batches, and only IronMLX's two batches of the third prompt set overlapped slightly, both at 1.02), TensorFold 1.05, and only oMLX 1.63 (it reads the last three requests together and decodes them together). The four inputs are read one after another: one request is read and decoded while the next one is still being read, so the requests rarely meet to decode together. For comparison, all four applications have a decode concurrency of 2.7–3.5 with short prompts (see the [short-prompt B4 page](qwen38-dense-b4-short.md)).
- **Lowest memory.** Peak memory while serving is 23.3 GiB, the lowest in both rounds; Splash uses 23.7 GiB, TensorFold 34.8 GiB, and oMLX 38.8 GiB.

### Prefill speed and time to first token

Batch prefill speed is the four requests' total input tokens divided by the time from the earliest request send to the last request's first token, which shows how fast the server reads long inputs; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long inputs and waiting in the queue. Lower right is better in the chart.

<!-- benchmark:scatter qwen38-dense-b4-30k -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first seven rows is in bold, and the last three rows are auxiliary.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Median time to first token (s) | **83.7** (83.4–⁠84.0) | 159.7 (157.1–⁠162.2) | 86.9 (86.7–⁠87.1) | 136.2 (136.0–⁠136.4) |
| Slowest request's first token (s) | **143.3** (143.3–⁠143.4) | 163.3 (161.3–⁠165.3) | 148.9 (148.5–⁠149.3) | 198.6 (198.5–⁠198.6) |
| Median completion time (s) | **87.2** (87.1–⁠87.3) | 176.4 (173.7–⁠179.0) | 96.3 (96.0–⁠96.6) | 181.2 (181.0–⁠181.4) |
| Slowest request's completion (s) | **148.9** (148.2–⁠149.5) | 180.5 (179.0–⁠182.0) | 153.9 (153.6–⁠154.1) | 206.8 (206.8–⁠206.9) |
| Peak memory while serving (GiB) | **23.3** (23.3–⁠23.3) | 38.8 (38.5–⁠39.0) | 23.7 (23.6–⁠23.8) | 34.8 (34.8–⁠34.8) |
| Batch prefill speed (input tok/s, higher is better)¹ | **858** (857–⁠858) | 753 (744–⁠762) | 826 (823–⁠828) | 619 (619–⁠619) |
| Output speed while decoding (output tok/s, higher is better)¹ | **67.6** (67.0–⁠68.3) | 23.5 (22.6–⁠24.4) | 32.3 (32.1–⁠32.4) | 7.2 (7.2–⁠7.2) |
| Decode concurrency (auxiliary)¹ | 1.00 (1.00–⁠1.00) | 1.63 (1.57–⁠1.69) | 1.00 (1.00–⁠1.00) | 1.05 (1.05–⁠1.05) |
| Batch throughput (output tok/s, auxiliary)² | 6.73 (6.70–⁠6.76) | 5.67 (5.63–⁠5.72) | 6.66 (6.65–⁠6.67) | 4.95 (4.95–⁠4.95) |
| Per-request decode speed (tok/s, auxiliary)³ | 82.0 (80.8–⁠83.3) | 15.1 (15.0–⁠15.2) | 35.3 (35.0–⁠35.6) | 5.7 (5.7–⁠5.7) |

For every metric in the table, the order of the four applications was the same in both rounds.

> [!NOTE]
> ¹ These three metrics were derived after the test from the client timestamps of each scored batch; they are not part of the test protocol written beforehand. They are defined as follows, and all of them include transfer time and the generation of the first output token, so they are not the engines' internal timings:
>
> - Batch prefill speed: the four requests' input tokens (server counts, 122,928 in total) ÷ the time from the earliest request send to the last request's first output.
> - Output speed while decoding: the batch's tokens actually output ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`; the union of the four intervals). Other requests' inputs may be read during this time, so it is the effective decode speed under load.
> - Decode concurrency: the four requests' decode times added up ÷ that union. 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> ² Batch throughput is the batch's tokens actually output ÷ the batch completion time (from the earliest send to the last `[DONE]`). At 30K the batch time is mostly spent reading inputs, so this metric mainly reflects prefill speed and is shown for reference only. IronMLX's Knowledge 3 ended naturally (235 tokens counted by the server, including the end token; 234 tokens of output text), so batches with Knowledge 3 output 1,002 tokens; every other batch output 1,024 tokens. The batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 2.9 ms).

> [!NOTE]
> ³ Per-request decode speed depends on how each application schedules the four requests: a request that decodes on its own while the others are still being read runs close to single-request speed, while one that alternates with the other requests' prefill runs much slower.

> [!NOTE]
> Peak memory while serving is measured during the scored batch and includes the KV cache of four 30K inputs. Peaks while loading the model and warming up differ: about 27.9–28.2 GiB for IronMLX, higher than its serving peak; about 19.5 GiB for Splash, 22.3–23.0 GiB for oMLX, and 34.3–34.7 GiB for TensorFold. This page does not compare memory during loading.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen38-dense-b4-30k -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0:** reads the four inputs one after another, with first tokens at about 31, 66, 102, and 143 s; most requests complete 2–8 s after their first token, but in the two batches of the third prompt set the second request takes about 38 s after its first token to complete, while the next input is being read. The completion log shows that in those two batches the second and third requests were decoded together (batch width 2) for 19 verification windows in one batch and 21 in the other; otherwise every request was decoded on its own (batch width 1), with a 15-node draft tree in each verification window. Outputs were byte-identical to the same prompts served one at a time on the B1 page.
- **oMLX 0.7.3rc1:** reads the first request on its own, with its first token at about 36 s; the other three inputs are read together, with first tokens at 157–164 s. The three requests are then decoded together with batched speculative decoding (its log names this batched path "Lightning MTP"); in every batch, requests switch to standard decoding for a stretch, and in 5 of the 6 batches the log shows the batched path parked for a while.
- **Splash 1.5.0:** reads the four inputs one after another, with first tokens at about 31, 68, 105, and 149 s; every decode batch had a width of 1, so it decoded one request at a time. By default, the prefill of long inputs hands part of the FFN computation to the Neural Engine. Some tasks' outputs differed from those served one at a time on the B1 page.
- **TensorFold 1.2.0:** native engine (`qwen27`) with four streams from `--parallel 4`. It reads the four inputs one after another, with first tokens at about 57, 113, 160, and 199 s; requests already read keep decoding while the later inputs are read, so per-request decode speed is only about 5.7 tok/s, and the last two requests complete close together (at about 204 and 207 s). Outputs were byte-identical to the same prompts served one at a time on the B1 page.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 31, 2026 (20:21–22:32 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session: it stayed nominal during every warm-up batch, and the prefill of four 30K inputs heated the Mac enough that pressure rose to a high level during every scored batch of every application |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (ANE `auto`, active for 30K inputs), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.5.0 | Official bottle | `--max-context 32768`; KV cache kept at its default int8; Neural Engine left on by default; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.2.0 | Official native release (sha256 `81fffaab…`), tag `v1.2.0` → `a388876` | `--drafter-bits 4 --parallel 4 --context 32768`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, 32,768-token context, capacity 4 where configurable, and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy. Each server's effective context was verified as 32,768 from its own interface or settings after startup.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run IronMLX's Knowledge 3 ended naturally at 235 tokens every time, and every other request ended by length at 256.
- **Prompts:** the six 30K prompts of the B1 page (three coding, three knowledge), each 30,732 tokens with the chat template, in three sets of four, so every batch has four different prompts. Every request's server-side prompt token count was verified as 30,732; no input was truncated or dropped.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four (short prompts), then one scored batch of four released together (largest send spread 2.9 ms), then stops the server. Two rounds × four applications × three prompt sets = 24 sessions; the application orders of the two rounds are ABCE and ECBA (reversed), so each application has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer; a request that ends by length counts 256) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test from the same timestamps; see the notes under [Comparison by metric](#comparison-by-metric) for their definitions.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was dropped. In a batch, oMLX may move a request from speculative to standard decoding to finish it, and that stretch writes no speculative-decoding end-of-request log line; the check, fixed before the formal run, accepts such a request as finished by standard decoding, while its end and token count are still checked against the stream. In this run, 7 requests in 5 of the 6 oMLX batches finished this way, and all of them completed normally on the client side.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run, 4 of TensorFold's Knowledge 3 requests recounted as 257 while the server counted 256 for each, and one of Splash's Knowledge 1 requests recounted as 255 while the server counted 256; IronMLX's Knowledge 3 was counted as 235 by the server, including the end token, with 234 tokens of output text.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **Small margin over Splash in prefill speed.** IronMLX's batch prefill speed is 3.9% higher than Splash's and its batch throughput 1.1% higher; IronMLX led in both rounds in both, but the margin is small and rests on two rounds on one machine.
- **Round-to-round variation.** On the same prompt set, IronMLX's batch throughput was 0.4–1.3% lower in the second round than in the first and oMLX's 0.6–1.7% higher, while Splash's changed by at most 0.9% and TensorFold's by at most 0.2%.
- **Metrics derived after the test.** Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test was complete, from client timestamps already recorded during the test, without a rerun; they are not part of the test protocol and were not comparisons fixed beforehand.
- **The ANE changes the output.** By default IronMLX uses the ANE in the prefill of 30K inputs, and its greedy outputs differ from those of the GPU alone; the natural end of Knowledge 3 occurs under the default setting (see the [B1 page](qwen38-dense-b1-30k.md)).
- Other input lengths, other models, other concurrency levels, or real multi-user traffic. This page covers one long input of about 30K tokens only, with four requests arriving together.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications.
- A hot machine. Every session started from nominal thermal pressure, but every scored batch ran under high thermal pressure; sustained high load over long periods is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache.
