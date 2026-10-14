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

- **Tied with TensorFold 1.0.5 for second in time to first token.** The median time to first token is 123 ms against 131 ms for TensorFold 1.0.5, and each is second in two of the four rounds; TensorFold 0.6.6 is fastest in all four rounds (111 ms).
- **Third in decode speed.** The median decode speed is 73.9 tok/s, 9.2% below Splash (81.5) and 1.8% below TensorFold 1.0.5 (75.3); Splash leads in all four rounds. IronMLX decodes faster than oMLX and TensorFold 0.6.6 in all four rounds.
- **Third in completion time.** The median time from sending a request to its completion is 3.61 s, 6.5% longer than Splash (3.38 s) and 1.8% longer than TensorFold 1.0.5 (3.54 s); Splash leads in all four rounds.
- **Second in memory.** Peak memory while serving is 18.9 GiB; Splash uses the least (18.6 GiB), about 1.5% less than IronMLX. The order was the same in all four rounds.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen38-dense-b1-short -->

### Comparison by metric

The table below lists the results of each application (TensorFold appears as two versions, 0.6.6 and 1.0.5) for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 | TensorFold 1.0.5 |
|---|---|---|---|---|---|
| Time to first token (ms) | 123 (115–⁠129) | 199 (195–⁠201) | 228 (222–⁠228) | **111** (107–⁠112) | 131 (118–⁠140) |
| Decode speed (tok/s, higher is better) | 73.9 (73.9–⁠76.0) | 72.8 (71.8–⁠73.0) | **81.5** (80.7–⁠81.8) | 68.4 (67.5–⁠68.5) | 75.3 (74.5–⁠75.9) |
| Completion time (s) | 3.61 (3.51–⁠3.62) | 3.77 (3.76–⁠3.82) | **3.38** (3.37–⁠3.41) | 3.86 (3.86–⁠3.91) | 3.54 (3.50–⁠3.59) |
| Peak memory while serving (GiB) | 18.9 (18.9–⁠18.9) | 19.1 (19.0–⁠19.1) | **18.6** (18.6–⁠18.6) | 20.3 (20.2–⁠20.3) | 41.1 (41.1–⁠41.1) |

The order of the applications was mostly the same across the four rounds: memory was identical in all four; Splash had the fastest decoding and shortest completion time in all four rounds and oMLX and TensorFold 0.6.6 were fourth and fifth, with TensorFold 1.0.5 second in three rounds and IronMLX in one; TensorFold 0.6.6 had the fastest first token in all four rounds, while IronMLX and TensorFold 1.0.5 swapped second and third.

### Results by task

All six tasks produce the same output length (256 tokens), but how often each application's drafts are accepted depends on the content, so the tasks differ in speed. The tables below show the median of four rounds for each task; the best value in each row is in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 | TensorFold 1.0.5 |
|---|---|---|---|---|---|
| Coding 1 (62) | **2.09** | 2.47 | 2.13 | 2.41 | 2.41 |
| Coding 2 (54) | 3.07 | 3.04 | 2.30 | 3.19 | **1.98** |
| Coding 3 (83) | 3.36 | 3.54 | **3.16** | 3.71 | 3.48 |
| Knowledge 1 (46) | 4.25 | **4.01** | 4.79 | 5.17 | 4.74 |
| Knowledge 2 (40) | 3.84 | 4.10 | 3.61 | 4.02 | **3.60** |
| Knowledge 3 (49) | 4.38 | 4.70 | **4.18** | 4.76 | 4.79 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 | TensorFold 1.0.5 |
|---|---|---|---|---|---|
| Coding 1 (62) | 129.1 | 112.7 | **134.9** | 110.7 | 112.0 |
| Coding 2 (54) | 86.6 | 90.0 | 122.3 | 82.8 | **137.9** |
| Coding 3 (83) | 79.7 | 78.6 | **87.6** | 71.4 | 76.0 |
| Knowledge 1 (46) | 61.5 | **66.9** | 55.9 | 50.5 | 55.3 |
| Knowledge 2 (40) | 68.3 | 65.3 | **75.4** | 65.3 | 74.5 |
| Knowledge 3 (49) | 60.4 | 56.6 | **64.6** | 54.8 | 54.7 |

IronMLX completes fastest only on Coding 1; Splash is fastest on three tasks, TensorFold 1.0.5 on two, and oMLX on Knowledge 1.

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request. According to each server's own logs and status pages:

- **IronMLX 0.2.0-rc.2:** each verification window uses a 15-node draft tree, verifies 16 rows at once, and has a fixed draft budget of 7; every window was a tree window.
- **oMLX 0.7.0:** the log records the speculative decoding acceptance for every request.
- **Splash 1.3.0:** the drafted and accepted counters on its status page grow with every request.
- **TensorFold 0.6.6:** Python engine; on one task a small share of the drafts come from its text-copy proposer, and the rest from the DFlash2 draft model.
- **TensorFold 1.0.5:** native engine (`qwen27`); each round verifies 16 rows (15 draft nodes).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 14, 2026 (02:15–03:28 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, then until macOS thermal pressure had read nominal for 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.2 | Official release `IronMLX.dmg` (sha256 `55a37763…`), tag `v0.2.0-rc.2` → `0ef3c55` | Bundled server and Metal runtime, started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile, no tuning switches |
| oMLX 0.7.0 | Official app | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.3.0 | Official bottle | `--kv-format bf16 --max-context 8192`, SSD cache off; the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 0.6.6 | Source tag (`cb2ebf05`) | `--drafter-bits 4 --lane-kernels on --parallel 4`, prompt cache and snapshots off |
| TensorFold 1.0.5 | Official native release (sha256 `0fa38f57…`), tag `v1.0.5` → `bec00ae` | `--drafter-bits 4`, prompt cache and snapshots off, default keep-warm; this version serves one request at a time for this model, so `--parallel` was not set |

Common settings for all applications: 4-bit draft model, BF16 KV cache, 8,192-token context, and capacity 4 where configurable (the client sends one request at a time); each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All five applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens), all different from each other and from the warm-up prompts.
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. Four rounds × five applications = 20 sessions, in the orders ABCDE, EDCBA, BCDEA, and AEDCB, so each application has the same average position.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 20 of 20 sessions and 120 of 120 scored requests passed every check; no session was retried or dropped. For every application, each task's output was byte-identical in all four rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.8 27B B4 concurrency performance comparison (short prompts)](qwen38-dense-b4-short.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Close results. IronMLX and TensorFold 1.0.5 swapped second place in time to first token; four rounds cannot reliably separate them.
- Run-to-run variation. Between requests with identical work, IronMLX varied more than any other application: across the four rounds of the same task, its time to first token differed by up to 75 ms (Knowledge 1: 105 to 180 ms) and its decode speed by up to 7.9%, against at most 48 ms and 3.0% for the others.
