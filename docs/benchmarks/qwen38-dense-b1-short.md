# Qwen3.8 27B B1 single-request performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen38-dense-b1-short.md)

> [!NOTE]
> This page compares single requests with short prompts. For 30K-token prompts, see [Qwen3.8 27B B1 single request](qwen38-dense-b1-30k.md) and [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-30k.md). Short-prompt performance comparison series:
>
> - Qwen3.8 27B B1 single request
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with one request at a time. All requests use short prompts (40–83 input tokens) with an output limit of 256 tokens, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the short-prompt B1 single-request comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b1-short -->

- **First in decode speed.** The median decode speed is 89.4 tok/s, first in all four rounds; 11.3% above second-place Splash (80.3) and 17.7% above TensorFold (75.9).
- **Shortest completion time.** The median time from sending a request to its completion is 3.00 s, first in all four rounds, against 3.37 s for Splash and 3.51 s for TensorFold.
- **Second in time to first token.** The median time to first token is 107 ms, second behind TensorFold (102 ms); the two swapped first place between rounds, with IronMLX first in rounds 1 and 3 and TensorFold first in rounds 2 and 4.
- **Lowest memory.** Peak memory while serving is 18.0 GiB, the lowest in all four rounds; Splash uses 19.3 GiB.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen38-dense-b1-short -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Time to first token (ms) | 107 (100–⁠136) | 198 (193–⁠199) | 149 (149–⁠150) | **102** (102–⁠104) |
| Decode speed (tok/s, higher is better) | **89.4** (89.0–⁠90.0) | 68.9 (68.1–⁠69.1) | 80.3 (79.7–⁠82.2) | 75.9 (75.7–⁠76.1) |
| Completion time (s) | **3.00** (2.97–⁠3.02) | 3.97 (3.96–⁠4.02) | 3.37 (3.29–⁠3.38) | 3.51 (3.50–⁠3.52) |
| Peak memory while serving (GiB) | **18.0** (18.0–⁠18.0) | 21.7 (21.7–⁠21.9) | 19.3 (19.3–⁠19.3) | 26.3 (26.3–⁠26.3) |

For decode speed, completion time, and memory, the order of the applications was identical in all four rounds: IronMLX, Splash, TensorFold, oMLX for decode speed and completion time; IronMLX, Splash, oMLX, TensorFold for memory. For time to first token, Splash and oMLX were third and fourth in every round, while IronMLX and TensorFold swapped first place: IronMLX was first in rounds 1 and 3, TensorFold in rounds 2 and 4.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests. While loading the model and warming up, IronMLX peaks at about 28.2 GiB, well above its serving peak and above every other application's serving peak; oMLX peaks at about 22.0 GiB, Splash 19.4 GiB, and TensorFold 26.3 GiB. This page does not compare memory during loading.

### Results by task

All six tasks produce the same output length (256 tokens), but how often each application's drafts are accepted depends on the content, so the tasks differ in speed. The tables below show the median of four rounds for each task; the best value in each row is in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Coding 1 (62) | **1.73** | 2.63 | 2.17 | 2.26 |
| Coding 2 (54) | 2.62 | 3.17 | 2.61 | **1.91** |
| Coding 3 (83) | **2.79** | 3.72 | 3.12 | 3.18 |
| Knowledge 1 (46) | **3.34** | 4.47 | 4.70 | 4.09 |
| Knowledge 2 (40) | **3.21** | 4.22 | 3.61 | 3.84 |
| Knowledge 3 (49) | **3.56** | 4.52 | 4.69 | 4.49 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Coding 1 (62) | **157.5** | 105.0 | 127.6 | 118.7 |
| Coding 2 (54) | 102.7 | 85.8 | 103.5 | **141.1** |
| Coding 3 (83) | **96.8** | 74.5 | 86.7 | 83.6 |
| Knowledge 1 (46) | **78.8** | 59.6 | 56.1 | 64.0 |
| Knowledge 2 (40) | **82.6** | 63.3 | 74.0 | 68.3 |
| Knowledge 3 (49) | **74.0** | 58.9 | 56.2 | 58.1 |

IronMLX completes fastest and decodes fastest on five tasks; TensorFold is fastest on Coding 2.

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request. According to each server's own logs and status pages:

- **IronMLX 0.2.0:** each verification window uses a 15-node draft tree, verifies 16 rows at once, and has a fixed draft budget of 7; every window was a tree window.
- **oMLX 0.7.3rc1:** the log records the speculative decoding acceptance for every request.
- **Splash 1.5.0:** the drafted and accepted counters on its status page grow with every request.
- **TensorFold 1.2.0:** native engine (`qwen27`); the minimum number of rows verified in a round, which the server reports for every request, was 6 to 12 depending on the request.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 31, 2026 (14:21–15:02 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. The one exception is the first session of rounds 3–4 (oMLX): as the first session of the added run, only the thermal pressure condition applied to it, and it started about 82 seconds after the previous server stopped. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (ANE `auto`; short prompts do not use the ANE), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.5.0 | Official bottle | `--max-context 8192`; draft verification kept at its default `--verify-rows auto`; KV cache kept at its default int8; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.2.0 | Official native release (sha256 `81fffaab…`), tag `v1.2.0` → `a388876` | `--drafter-bits 4 --parallel 4 --context 8192`, prompt cache and snapshots off, default keep-warm |

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
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. The first two rounds ran in the orders ABCE and ECBA (reversed), so each application has the same average position. A rule fixed before the test added two more rounds (orders BAEC and CEAB) right after, if in any metric two adjacent applications swapped order between the rounds or their two-round ranges overlapped. In time to first token, TensorFold and IronMLX swapped order (IronMLX first in round 1, TensorFold in round 2) and their two-round ranges overlapped, so rounds 3 and 4 were added: four rounds × four applications = 16 sessions, with each application running once in each position.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. For IronMLX, Splash, and TensorFold, each task's output was byte-identical in all four rounds; oMLX's output differed between rounds on Knowledge 1, Knowledge 2, and Knowledge 3.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.8 27B B4 concurrency performance comparison (short prompts)](qwen38-dense-b4-short.md); for single requests with prompts of about 30K tokens, see [Qwen3.8 27B B1 30K-prompt performance comparison](qwen38-dense-b1-30k.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Between requests with identical work, IronMLX's time to first token varied more than any other application's: across the four rounds of the same task it differed by up to 76 ms (Coding 2: 91 to 167 ms), against at most 63 ms for Splash, 14 ms for oMLX, and 3 ms for TensorFold. Its decode speed differed by up to 3.0% for the same task, against at most 6.8% for oMLX, 4.6% for Splash, and 0.8% for TensorFold. The order of the applications did not change for decode speed, completion time, or memory; for time to first token, IronMLX's round medians ranged from 100 to 136 ms against 102 to 104 ms for TensorFold, so the two swapped first place between rounds.
