# Qwen3.8 27B B4 concurrency performance comparison

[简体中文](../zh-CN/benchmarks/qwen38-dense-b4.md)

The B4 concurrency comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the B4 concurrency comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b4 -->

- **Fastest first token.** The median wait for the first words is 251 ms, first in all four rounds; TensorFold 0.6.6 follows at 276 ms.
- **Fastest first token for the slowest request too.** The slowest of the four requests shows its first words after 430 ms, first in all four rounds; TensorFold 0.6.6 follows at 441 ms.
- **Third in throughput.** Four requests together complete at 146.8 tok/s, 4.2% below Splash (153.3) and 2.4% below TensorFold 0.6.6 (150.4). Splash leads in all four rounds.
- **Third in memory.** Peak memory while serving is 21.9 GiB; Splash uses the least (19.3 GiB).

### Throughput vs. time to first token

Batch throughput is the total output tokens of the four requests divided by the batch completion time (from the first request sent to the last request finished), and reflects the server's total capacity under concurrent load. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen38-dense-b4 -->

> [!NOTE]
> TensorFold 1.0.5: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#exact-decoding), the native engine does not process concurrent requests for Qwen3.8-27B.

### Comparison by metric

The table below lists the results of each application (TensorFold appears as two versions, 0.6.6 and 1.0.5) for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's three batches (three different prompt sets). Higher batch throughput is better, and lower is better for the time and memory metrics; the best value in each of the first six rows is in bold. On the first six metrics, the ranking of the applications stayed the same in all four rounds.

| Metric | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 | TensorFold 1.0.5¹ |
|---|---|---|---|---|---|
| Batch throughput (tok/s, higher is better)² | 146.8 (136.4–⁠148.8) | 114.3 (113.2–⁠116.2) | **153.3** (151.8–⁠153.7) | 150.4 (149.6–⁠150.8) | 82.0 (81.6–⁠82.0) |
| Median time to first token (ms) | **251** (245–⁠258) | 615 (593–⁠641) | 637 (617–⁠651) | 276 (274–⁠284) | 5,282 (4,620–⁠5,761) |
| Slowest request's first token (ms) | **430** (420–⁠435) | 791 (769–⁠806) | 638 (618–⁠651) | 441 (436–⁠467) | 10,195 (9,115–⁠10,573) |
| Median completion time (s) | 5.79 (5.69–⁠6.18) | 7.53 (7.37–⁠7.63) | **4.80** (4.75–⁠4.87) | 5.22 (5.18–⁠5.33) | 8.38 (7.84–⁠8.77) |
| Slowest request's completion (s)² | 6.98 (6.88–⁠7.51) | 8.96 (8.81–⁠9.04) | **6.68** (6.66–⁠6.75) | 6.81 (6.79–⁠6.84) | 12.49 (12.48–⁠12.54) |
| Peak memory while serving (GiB) | 21.9 (21.9–⁠22.0) | 22.9 (22.8–⁠22.9) | **19.3** (19.3–⁠19.3) | 21.3 (21.3–⁠21.3) | 41.1 (41.1–⁠41.1) |
| Per-request decode speed (tok/s, auxiliary) | 46.5 (43.8–⁠47.1) | 37.3 (37.0–⁠39.2) | 61.3 (60.9–⁠61.7) | 51.6 (50.9–⁠52.4) | 94.2 (93.8–⁠94.4) |

> [!NOTE]
> ¹ TensorFold 1.0.5: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#exact-decoding), the native engine does not process concurrent requests for Qwen3.8-27B. Its figures in the table are the result of the four requests being queued and served one at a time.

> [!NOTE]
> ² Batch throughput is derived from the batch completion time (1,024 tokens ÷ batch completion time), and the batch completion time differs from the slowest request's completion time only by the gap between the requests' send times (at most 1.5 ms), so these two rows reflect the same result and should not be counted as two separate advantages.

### Request response timeline

The chart below shows, for four requests sent at the same moment, when each request receives its first output and when it completes, to show how each application schedules the four requests under concurrent load. The horizontal axis is the time since the requests were sent; ticks mark when a request receives its first output, and dots mark when a request completes. In each batch, the four first-token times and the four completion times are sorted separately from earliest to latest, and each position shows the median of that position over 12 batches. Ticks and dots are therefore sorted independently, and a tick and a dot at the same position do not necessarily belong to the same request.

<!-- benchmark:timeline qwen38-dense-b4 -->

> [!NOTE]
> TensorFold 1.0.5: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#exact-decoding), the native engine does not process concurrent requests for Qwen3.8-27B.

Concurrent requests from the client do not guarantee batching inside the engine. According to each server's own logs and status pages, the four requests were handled as follows:

- **IronMLX 0.2.0-rc.2:** uses the external DFlash2 draft model; whenever two or more requests can be batched, it verifies them together in ragged linear batches, and when only one request is left, or a request is close to its output limit, that request is verified alone with a 15-node draft tree. Outputs were byte-identical to the same prompts served one at a time.
- **oMLX 0.7.0:** runs batched speculative decoding for the four requests with the external DFlash2 draft model (its log names this batched path "Lightning MTP").
- **Splash 1.3.0:** uses the external DFlash2 draft model and decodes the four requests in shared batches of up to four rows; in each batch one request receives its first token first (about 0.4 s), and the other three receive theirs at the same moment (about 0.64 s).
- **TensorFold 0.6.6:** Python engine; uses the external DFlash2 draft model, with up to four requests sharing each verification round.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 13, 2026 (04:04–07:25 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, then until macOS thermal pressure had read nominal for 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.2 | Official release `IronMLX.dmg` (sha256 `55a37763…`), tag `v0.2.0-rc.2` → `0ef3c55` | Bundled server and Metal runtime, started as the app starts it; `--dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none`; default M5 profile, no tuning switches |
| oMLX 0.7.0 | Official app | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.3.0 | Official bottle | `--kv-format bf16 --max-context 8192`, SSD cache off; the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 0.6.6 | Source tag (`cb2ebf05`) | `--drafter-bits 4 --lane-kernels on --parallel 4`, prompt cache and snapshots off |
| TensorFold 1.0.5 | Official native release (sha256 `0fa38f57…`), tag `v1.0.5` → `bec00ae` | `--drafter-bits 4 --parallel 4`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, BF16 KV cache, 8,192-token context, and capacity 4 where configurable; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All five applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens) in three sets of four, so every batch has four different prompts.
- **Sessions:** each session starts a fresh server, sends one warm-up batch of four, then one scored batch of four released together (largest send spread 1.4 ms), then stops the server. Four rounds × five applications × three prompt sets = 60 sessions, in a balanced order in which each application has the same average position.
- **Caching:** prefix and disk caches were off or verified unused for every batch; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first content; completion runs from the send to `[DONE]`, including any queueing; batch throughput is 1,024 tokens divided by the time from the first send to the last `[DONE]`. Memory is the process-tree physical footprint, sampled every 50 ms during the scored batch.
- **Validity:** 60 of 60 sessions and 240 of 240 scored requests passed every check; no session was retried or dropped.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. For example, the TensorFold 1.0.5 native engine does not process concurrent requests for this 27B model, which is how that version handles this model and says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Long prompts, other models, other concurrency levels, or real multi-user traffic.
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Between batches with identical work, IronMLX's throughput varied more than any other application's: about 10% across the four rounds on the same prompt set (for example 145 to 161 tok/s), against at most 6.5% for the others. The order of the five applications did not change.
