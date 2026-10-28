# Qwen3.8 27B B4 30K-prompt concurrency performance comparison

[简体中文](../zh-CN/benchmarks/qwen38-dense-b4-30k.md)

> [!NOTE]
> This page compares concurrent requests with 30K-token prompts. For short prompts, see [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md) and [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.8 27B B4 concurrency
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-30k.md)
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-30k.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with four requests arriving together. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to several users or agents handing long documents or parts of a codebase to the same local server at once; all four 30K inputs have to be read, so the waits run to minutes. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the 30K-prompt B4 concurrency comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b4-30k -->

- **Shortest median time to first token.** The median of the four requests' times to first token is 84.7 s, first in both rounds; Splash takes 87.2 s, 2.9% longer, and the two-round ranges of the two overlap slightly. TensorFold takes 131.1 s and oMLX 160.4 s.
- **Shortest median completion time.** The median of the four requests' completion times is 88.4 s, first in both rounds and 8.2% shorter than Splash (96.3 s); oMLX takes 177.1 s and TensorFold 179.2 s.
- **Prefill speed level with Splash.** The four 30K inputs total 122,928 tokens; IronMLX's batch prefill speed is 843 tok/s and Splash's 821 tok/s, each leading in one round; oMLX reaches 751 and TensorFold 600. At 30K, about 89–97% of each application's batch time is spent reading the four inputs, so prefill speed largely sets the wait. The slowest request's first token (146.3 s against 149.7 s) and completion (151.8 s against 156.1 s) also split one round each between the two.
- **Highest output speed while decoding.** During the time in which a request is decoding, IronMLX outputs 67.4 tokens per second, first in both rounds and about twice Splash's 32.8; oMLX reaches 22.7 and TensorFold 7.0. In the timeline, each IronMLX request completes 2–7 s after its first token, little affected by the other requests being read, while each Splash request takes about 7–11 s to decode.
- **Almost no batching at 30K.** IronMLX and Splash both have a decode concurrency of 1.00 (no two requests ever decoded at the same time), TensorFold 1.05, and only oMLX 1.88 (it reads the last three requests together and decodes them together). The four inputs are read one after another: one request is read and decoded while the next one is still being read, so the requests never meet to decode together. For comparison, all four applications have a decode concurrency of 3.0–3.5 with short prompts (see the [short-prompt B4 page](qwen38-dense-b4-short.md)).
- **Lowest memory.** Peak memory while serving is 23.3 GiB, the lowest in both rounds; Splash uses 23.6 GiB, TensorFold 28.0 GiB, and oMLX 52.9 GiB.

### Prefill speed and time to first token

Batch prefill speed is the four requests' total input tokens divided by the time from the earliest request send to the last request's first token, which shows how fast the server reads long inputs; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long inputs and waiting in the queue. Lower right is better in the chart.

<!-- benchmark:scatter qwen38-dense-b4-30k -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher speed is better, and lower is better for time and memory; the best value in each of the first seven rows is in bold, and the last three rows are auxiliary.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Median time to first token (s) | **84.7** (83.3–⁠86.1) | 160.4 (155.3–⁠165.5) | 87.2 (85.9–⁠88.4) | 131.1 (130.8–⁠131.5) |
| Slowest request's first token (s) | **146.3** (138.2–⁠154.3) | 163.8 (160.2–⁠167.5) | 149.7 (147.8–⁠151.6) | 204.9 (204.7–⁠205.1) |
| Median completion time (s) | **88.4** (87.1–⁠89.6) | 177.1 (170.9–⁠183.3) | 96.3 (96.2–⁠96.3) | 179.2 (178.5–⁠179.9) |
| Slowest request's completion (s) | **151.8** (142.4–⁠161.2) | 182.0 (174.5–⁠189.6) | 156.1 (152.6–⁠159.6) | 211.7 (211.7–⁠211.7) |
| Peak memory while serving (GiB) | **23.3** (23.2–⁠23.3) | 52.9 (52.7–⁠53.0) | 23.6 (23.6–⁠23.6) | 28.0 (28.0–⁠28.0) |
| Batch prefill speed (input tok/s, higher is better)¹ | **843** (797–⁠889) | 751 (734–⁠768) | 821 (811–⁠831) | 600 (599–⁠601) |
| Output speed while decoding (output tok/s, higher is better)¹ | **67.4** (62.4–⁠72.3) | 22.7 (21.4–⁠24.0) | 32.8 (32.5–⁠33.1) | 7.0 (7.0–⁠7.1) |
| Decode concurrency (auxiliary)¹ | 1.00 (1.00–⁠1.00) | 1.88 (1.77–⁠2.00) | 1.00 (1.00–⁠1.00) | 1.05 (1.04–⁠1.05) |
| Batch throughput (output tok/s, auxiliary)² | 6.63 (6.21–⁠7.04) | 5.63 (5.40–⁠5.85) | 6.56 (6.42–⁠6.71) | 4.84 (4.84–⁠4.84) |
| Per-request decode speed (tok/s, auxiliary)³ | 83.1 (81.5–⁠84.7) | 15.4 (14.3–⁠16.6) | 37.6 (36.4–⁠38.8) | 5.6 (5.6–⁠5.6) |

The order was the same in both rounds for median time to first token, memory, output speed while decoding, and per-request decode speed; for median completion time, IronMLX and Splash led in both rounds. For batch prefill speed, the slowest request's first token and completion, and batch throughput, IronMLX and Splash swapped places between the rounds, and the rest of the order was the same in both rounds.

> [!NOTE]
> ¹ These three metrics were derived after the test from the client timestamps of each scored batch; they are not part of the test protocol written beforehand. They are defined as follows, and all of them include transfer time and the generation of the first output token, so they are not the engines' internal timings:
>
> - Batch prefill speed: the four requests' input tokens (server counts, 122,928 in total) ÷ the time from the earliest request send to the last request's first output.
> - Output speed while decoding: the batch's tokens actually output ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`; the union of the four intervals). Other requests' inputs may be read during this time, so it is the effective decode speed under load.
> - Decode concurrency: the four requests' decode times added up ÷ that union. 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout.

> [!NOTE]
> ² Batch throughput is the batch's tokens actually output ÷ the batch completion time (from the earliest send to the last `[DONE]`). At 30K the batch time is mostly spent reading inputs, so this metric mainly reflects prefill speed and is shown for reference only. IronMLX's Knowledge 3 ended naturally at token 235, so batches with Knowledge 3 output 1,002 tokens; in one oMLX batch, Knowledge 3 ended naturally at token 250; every other batch output 1,024 tokens. The batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 2.4 ms).

> [!NOTE]
> ³ Per-request decode speed depends on how each application schedules the four requests: a request that decodes on its own while the others are still being read runs close to single-request speed, while one that alternates with the other requests' prefill runs much slower.

> [!NOTE]
> Peak memory while serving is measured during the scored batch and includes the KV cache of four 30K inputs. Peaks while loading the model and warming up differ: about 27.9–28.2 GiB for IronMLX, higher than its serving peak; about 19.4 GiB for Splash, 22.3–22.9 GiB for oMLX, and 28.0–28.2 GiB for TensorFold. This page does not compare memory during loading.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen38-dense-b4-30k -->

Concurrent requests from the client do not guarantee batching inside the engine. According to the timeline and each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX dev (c6ef14ec2):** reads the four inputs one after another, with first tokens at about 31, 67, 103, and 146 s; each request completes 2–7 s after its first token. The completion log shows that every request was decoded on its own (batch width 1), with a 15-node draft tree in each verification window. Outputs were byte-identical to the same prompts served one at a time on the B1 page.
- **oMLX 0.7.2:** reads the first request on its own, with its first token at about 36 s; the other three inputs are read together, with first tokens at 156–164 s. The three requests are then decoded together with batched speculative decoding (its log names this batched path "Lightning MTP"), and individual requests switch to standard decoding for a stretch.
- **Splash 1.4.1:** reads the four inputs one after another, with first tokens at about 32, 69, 107, and 151 s; every decode batch had a width of 1, so it decoded one request at a time. By default, the prefill of long inputs hands part of the FFN computation to the Neural Engine. Some tasks' outputs differed from those served one at a time on the B1 page.
- **TensorFold 1.1.6:** native engine (`qwen27`) with four streams from `--parallel 4`. It reads the four inputs one after another, with first tokens at about 42, 104, 159, and 205 s; requests already read keep decoding while the later inputs are read, so per-request decode speed is only about 5.6 tok/s, and the last two requests complete almost together (at about 211 s). Outputs were byte-identical to the same prompts served one at a time on the B1 page.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 28, 2026 (00:36–02:46 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session: it stayed nominal during every warm-up batch, and the prefill of four 30K inputs heated the Mac enough that pressure rose to a high level during every scored batch of every application |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none`; default M5 profile (ANE `auto`, active for 30K inputs), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.4.1 | Official bottle | `--max-context 32768`; KV cache kept at its default int8; Neural Engine left on by default; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.1.6 | Official native release (sha256 `0cd208ba…`), tag `v1.1.6` → `eea9dc9` | `--drafter-bits 4 --parallel 4 --context 32768`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, 32,768-token context, capacity 4 where configurable, and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy. Each server's effective context was verified as 32,768 from its own interface or settings after startup.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run IronMLX's Knowledge 3 ended naturally at 235 tokens every time, oMLX's Knowledge 3 once at 250 tokens, and every other request ended by length at 256.
- **Prompts:** the six 30K prompts of the B1 page (three coding, three knowledge), each 30,732 tokens with the chat template, in three sets of four, so every batch has four different prompts. Every request's server-side prompt token count was verified as 30,732; no input was truncated or dropped.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four (short prompts), then one scored batch of four released together (largest send spread 2.4 ms), then stops the server. Two rounds × four applications × three prompt sets = 24 sessions; the application orders of the two rounds are ABCE and ECBA (reversed), so each application has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is the number of tokens actually output (counted on the client with the target tokenizer) divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Batch prefill speed, output speed while decoding, and decode concurrency were derived after the test from the same timestamps; see the notes under [Comparison by metric](#comparison-by-metric) for their definitions.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was dropped. The first formal run was stopped by a check at its fifth session: oMLX moved one request in a batch from speculative to standard decoding to finish it, and that stretch writes no speculative-decoding end-of-request log line, while the check required one for every request. On the client side, all four requests of that batch completed normally. After the check was corrected, all 24 sessions were rerun from the start; this page uses the rerun only.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded).

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- **No order between IronMLX and Splash in prefill speed.** Their batch prefill speed differs by 2.7% and their batch throughput by 1.0%, and each led in one round in both.
- **Round-to-round variation.** On the same prompt set, IronMLX's batch throughput was 9.8–12.3% lower in the second round than in the first (the round medians of its batch prefill speed are 889 and 797 tok/s), oMLX's 5.2–8.0% lower, and Splash's 4.3–5.4% lower, while TensorFold's changed by at most 2.3%. This page does not establish why IronMLX's second round was lower.
- **Metrics derived after the test.** Batch prefill speed, output speed while decoding, and decode concurrency were defined after the test was complete, from client timestamps already recorded during the test, without a rerun; they were not comparisons fixed beforehand.
- **The ANE changes the output.** By default IronMLX uses the ANE in the prefill of 30K inputs, and its greedy outputs differ from those of the GPU alone; the natural end of Knowledge 3 occurs under the default setting (see the [B1 page](qwen38-dense-b1-30k.md)).
- Other input lengths, other models, other concurrency levels, or real multi-user traffic. This page covers one long input of about 30K tokens only, with four requests arriving together.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications.
- A hot machine. Every session started from nominal thermal pressure, but every scored batch ran under high thermal pressure; sustained high load over long periods is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache.
