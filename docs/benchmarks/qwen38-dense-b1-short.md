# Qwen3.8 27B B1 single-request performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen38-dense-b1-short.md)

> [!NOTE]
> Short-prompt performance comparison series:
>
> - Qwen3.8 27B B1 single request
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md)
> - [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md)
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with one request at a time. All requests use short prompts (40–83 input tokens) with a fixed 256-token output, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the short-prompt B1 single-request comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b1-short -->

- **First in decode speed.** The median decode speed is 89.6 tok/s, first in both rounds; 14.6% above second-place Splash (78.2) and 21.0% above TensorFold (74.1).
- **Shortest completion time.** The median time from sending a request to its completion is 2.98 s, first in both rounds, against 3.48 s for Splash and 3.56 s for TensorFold.
- **Second in time to first token.** The median time to first token is 105 ms, second in both rounds; TensorFold is fastest in both rounds (81 ms).
- **Lowest memory.** Peak memory while serving is 17.9 GiB, the lowest in both rounds; Splash uses 19.3 GiB.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen38-dense-b1-short -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Time to first token (ms) | 105 (95–⁠116) | 193 (191–⁠195) | 146 (140–⁠152) | **81** (76–⁠85) |
| Decode speed (tok/s, higher is better) | **89.6** (89.3–⁠89.9) | 68.3 (67.8–⁠68.8) | 78.2 (78.1–⁠78.3) | 74.1 (74.0–⁠74.1) |
| Completion time (s) | **2.98** (2.97–⁠2.99) | 3.99 (3.97–⁠4.02) | 3.48 (3.48–⁠3.49) | 3.56 (3.56–⁠3.57) |
| Peak memory while serving (GiB) | **17.9** (17.9–⁠18.0) | 21.7 (21.7–⁠21.8) | 19.3 (19.2–⁠19.3) | 19.8 (19.8–⁠19.8) |

The order of the applications was identical in both rounds for all four metrics: IronMLX, Splash, TensorFold, oMLX for decode speed, completion time, and memory; TensorFold, IronMLX, Splash, oMLX for time to first token.

### Results by task

All six tasks produce the same output length (256 tokens), but how often each application's drafts are accepted depends on the content, so the tasks differ in speed. The tables below show the median of two rounds for each task; the best value in each row is in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Coding 1 (62) | **1.73** | 2.61 | 2.12 | 2.34 |
| Coding 2 (54) | 2.67 | 3.19 | 2.62 | **1.93** |
| Coding 3 (83) | **2.75** | 3.75 | 3.05 | 3.27 |
| Knowledge 1 (46) | **3.38** | 4.50 | 4.49 | 4.14 |
| Knowledge 2 (40) | **3.20** | 4.24 | 3.91 | 3.86 |
| Knowledge 3 (49) | **3.57** | 4.70 | 4.05 | 4.68 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Coding 1 (62) | **158.3** | 105.9 | 129.4 | 112.9 |
| Coding 2 (54) | 99.9 | 85.1 | 103.2 | **137.9** |
| Coding 3 (83) | **97.0** | 73.6 | 88.5 | 80.7 |
| Knowledge 1 (46) | **77.8** | 59.1 | 58.7 | 62.6 |
| Knowledge 2 (40) | **82.2** | 63.0 | 67.8 | 67.4 |
| Knowledge 3 (49) | **73.4** | 56.5 | 65.4 | 55.5 |

IronMLX completes fastest and decodes fastest on five tasks; TensorFold is fastest on Coding 2.

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request. According to each server's own logs and status pages:

- **IronMLX dev (c6ef14ec2):** each verification window uses a 15-node draft tree, verifies 16 rows at once, and has a fixed draft budget of 7; every window was a tree window.
- **oMLX 0.7.2:** the log records the speculative decoding acceptance for every request.
- **Splash 1.4.1:** the drafted and accepted counters on its status page grow with every request.
- **TensorFold 1.1.6:** native engine (`qwen27`); each round verifies 16 rows.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 27, 2026 (10:57–11:18 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (ANE `auto`; short prompts do not use the ANE), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.4.1 | Official bottle | `--max-context 8192`; KV cache kept at its default int8; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.1.6 | Official native release (sha256 `0cd208ba…`), tag `v1.1.6` → `eea9dc9` | `--drafter-bits 4 --parallel 4`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, 8,192-token context, capacity 4 where configurable (the client sends one request at a time), and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens), all different from each other and from the warm-up prompts.
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. Two rounds × four applications = 8 sessions, in the orders ABCE and ECBA (reversed), so each application has the same average position. A rule fixed before the test added two more rounds (orders BAEC and CEAB) right after, if in any metric two adjacent applications swapped order between the rounds or their two-round ranges overlapped; every order and range was separable, so no rounds were added.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the two rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 8 of 8 sessions and 48 of 48 scored requests passed every check; no session was retried or dropped. For IronMLX, Splash, and TensorFold, each task's output was byte-identical in both rounds; oMLX's output differed between the rounds on Knowledge 2 and Knowledge 3.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.8 27B B4 concurrency performance comparison (short prompts)](qwen38-dense-b4-short.md); for single requests with prompts of about 30K tokens, see [Qwen3.8 27B B1 30K-prompt performance comparison](qwen38-dense-b1-30k.md).
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Between requests with identical work, IronMLX's time to first token varied more than any other application's: across the two rounds of the same task it differed by up to 50 ms (Coding 1: 93 to 143 ms), against at most 32 ms for Splash and 16 ms for oMLX and TensorFold. Its decode speed differed by up to 5.7% for the same task, against at most 2.5% for the others. The order of the applications did not change.
