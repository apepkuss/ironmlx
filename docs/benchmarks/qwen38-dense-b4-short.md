# Qwen3.8 27B B4 concurrency performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen38-dense-b4-short.md)

> [!NOTE]
> Short-prompt performance comparison series:
>
> - Qwen3.8 27B B4 concurrency
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md)
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md)
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with four requests arriving together. All requests use short prompts (40–83 input tokens) with a fixed 256-token output, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the short-prompt B4 concurrency comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b4-short -->

- **First in throughput, by a small margin.** Four requests together complete at 195.5 tok/s, first in both rounds; 2.0% above Splash (191.7) and 20.2% above TensorFold (162.7).
- **Second in time to first token.** The median wait for the first words is 224 ms, second in both rounds; TensorFold is fastest in both rounds (190 ms).
- **Second in the slowest request's first token.** The slowest of the four requests shows its first words after 376 ms; TensorFold's after 329 ms.
- **Lowest memory.** Peak memory while serving is 19.4 GiB, the lowest in both rounds; Splash uses 20.1 GiB.

### Throughput vs. time to first token

Batch throughput is the total output tokens of the four requests divided by the batch completion time (from the first request sent to the last request finished), and reflects the server's total capacity under concurrent load. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen38-dense-b4-short -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher batch throughput is better, and lower is better for the time and memory metrics; the best value in each of the first six rows is in bold. On the first six metrics, the ranking of the applications stayed the same in both rounds.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Batch throughput (tok/s, higher is better)¹ | **195.5** (194.2–⁠196.8) | 103.4 (102.4–⁠104.4) | 191.7 (189.8–⁠193.6) | 162.7 (159.4–⁠166.0) |
| Median time to first token (ms) | 224 (215–⁠234) | 667 (660–⁠673) | 500 (499–⁠501) | **190** (189–⁠191) |
| Slowest request's first token (ms) | 376 (369–⁠383) | 822 (804–⁠839) | 500 (499–⁠502) | **329** (328–⁠330) |
| Median completion time (s) | 4.32 (4.30–⁠4.34) | 8.29 (8.07–⁠8.51) | **4.16** (4.13–⁠4.20) | 4.64 (4.62–⁠4.66) |
| Slowest request's completion (s)¹ | **5.24** (5.20–⁠5.27) | 9.90 (9.80–⁠10.00) | 5.34 (5.29–⁠5.39) | 6.30 (6.17–⁠6.43) |
| Peak memory while serving (GiB) | **19.4** (19.4–⁠19.4) | 22.8 (22.8–⁠22.8) | 20.1 (20.1–⁠20.1) | 22.5 (22.4–⁠22.6) |
| Per-request decode speed (tok/s, auxiliary) | 62.6 (61.7–⁠63.5) | 33.6 (32.9–⁠34.3) | 70.8 (70.6–⁠71.0) | 59.7 (59.7–⁠59.8) |
| Decode concurrency (auxiliary)² | 3.46 (3.46–⁠3.46) | 3.19 (3.16–⁠3.21) | 3.03 (3.00–⁠3.05) | 3.11 (3.11–⁠3.12) |

> [!NOTE]
> ¹ Batch throughput is derived from the batch completion time (1,024 tokens ÷ batch completion time), and the batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 1.5 ms), so these two rows reflect the same result and should not be counted as two separate advantages.

> [!NOTE]
> ² Decode concurrency was derived after the test from the client timestamps of each scored batch; it is not part of the test protocol written beforehand. It is the four requests' decode times added up ÷ the total time during which at least one request was decoding (each request counts as decoding from its first output to `[DONE]`). 1 means no two requests ever decoded at the same time; 4 means all four decoded together throughout. All four applications are at 3.0–3.5, so with short prompts the four requests are decoded mostly together; for the contrast with 30K-token prompts, see [Qwen3.8 27B B4 30K-prompt concurrency performance comparison](qwen38-dense-b4-30k.md).

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 6 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen38-dense-b4-short -->

Concurrent requests from the client do not guarantee batching inside the engine. According to each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX dev (c6ef14ec2):** uses the external DFlash2 draft model; whenever two or more requests can be batched, it verifies them together in ragged linear batches, and when only one request is left, that request is verified alone with a 15-node draft tree. Outputs were byte-identical to the same prompts served one at a time.
- **oMLX 0.7.2:** runs batched speculative decoding for the four requests with the external DFlash2 draft model (its log names this batched path "Lightning MTP"); during batching it calibrates batched against standard decoding, and individual requests switch to standard decoding for a stretch.
- **Splash 1.4.1:** uses the external DFlash2 draft model and decodes the four requests in shared batches of up to four rows; in each batch one request receives its first token first (about 0.14 s), and the other three receive theirs at the same moment (about 0.50 s). Outputs were byte-identical to the same prompts served one at a time.
- **TensorFold 1.1.6:** native engine (`qwen27`) with four streams from `--parallel 4`; the four requests share each verification round, and every request uses DFlash2 drafts. Outputs were byte-identical to the same prompts served one at a time.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 27, 2026 (12:08–13:03 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (ANE `auto`; short prompts do not use the ANE), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.4.1 | Official bottle | `--max-context 8192`; KV cache kept at its default int8; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every batch); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.1.6 | Official native release (sha256 `0cd208ba…`), tag `v1.1.6` → `eea9dc9` | `--drafter-bits 4 --parallel 4`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, 8,192-token context, capacity 4 where configurable, and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens) in three sets of four, so every batch has four different prompts.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four, then one scored batch of four released together (largest send spread 1.5 ms), then stops the server. Two rounds × four applications × three prompt sets = 24 sessions; the application orders of the two rounds are ABCE and ECBA (reversed), so each application has the same average position, and the order of the three prompt sets rotates by round.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is 1,024 tokens divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch. Decode concurrency was derived after the test from the same timestamps, without a rerun; see the notes under [Comparison by metric](#comparison-by-metric) for its definition.
- **Validity:** 24 of 24 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- Long prompts, other models, other concurrency levels, or real multi-user traffic. For concurrent requests with prompts of about 30K tokens, see [Qwen3.8 27B B4 30K-prompt concurrency performance comparison](qwen38-dense-b4-30k.md).
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- A small throughput gap. IronMLX's batch throughput is only 2.0% above Splash's; the two-round ranges do not overlap and the order held in both rounds, but the gap is close to the batch-to-batch variation.
- Batch-to-batch variation. Across the two rounds on the same prompt set, oMLX's throughput differed by up to about 8% (second set: 102 to 111 tok/s), TensorFold's by about 4%, and IronMLX's and Splash's by at most 2%. The order of the applications did not change.
