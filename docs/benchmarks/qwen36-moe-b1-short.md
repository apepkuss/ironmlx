# Qwen3.6 35B A3B B1 single-request performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen36-moe-b1-short.md)

> [!NOTE]
> Short-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B1 single request
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md)
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md)
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. All requests use short prompts (40–83 input tokens) with a fixed 256-token output, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the short-prompt B1 single-request comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen36-moe-b1-short -->

- **Tied with Splash for the shortest completion time.** The median time from sending a request to its completion is 1.225 s against 1.235 s for Splash, and each is first in two of the four rounds; IronMLX completes faster than oMLX (1.340 s) and TensorFold 0.6.6 (1.407 s) in all four rounds.
- **Lowest memory.** Peak memory while serving is 20.5 GiB, the lowest in all four rounds; Splash follows at 21.2 GiB.
- **Second in decode speed.** The median decode speed is 225.9 tok/s, 3.9% below Splash (235.1), which leads in all four rounds; IronMLX decodes faster than oMLX (221.4) and TensorFold 0.6.6 (195.4) in all four rounds.
- **Second in time to first token.** The median time to first token is 75 ms, 3.2% slower than TensorFold 0.6.6 (73 ms); TensorFold 0.6.6 is first in three rounds and IronMLX in one, and both are well ahead of Splash (98 ms) and oMLX (106 ms).

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-short -->

> [!NOTE]
> TensorFold 1.0.5 is not included: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#qualified-models), Qwen3.6 is still being qualified for the native engine, and the native engine accepts an external DFlash2 draft model only for Qwen3.8-27B-class checkpoints ([source](https://github.com/ashhart/TensorFold/blob/v1.0.5/zig/src/native/metal.zig#L101-L105)).

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Time to first token (ms) | 75 (74–⁠76) | 106 (99–⁠108) | 98 (96–⁠98) | **73** (72–⁠77) |
| Decode speed (tok/s, higher is better) | 225.9 (224.1–⁠226.3) | 221.4 (219.1–⁠221.7) | **235.1** (229.4–⁠238.6) | 195.4 (193.9–⁠196.1) |
| Completion time (s) | **1.225** (1.218–⁠1.230) | 1.340 (1.333–⁠1.343) | 1.235 (1.214–⁠1.260) | 1.407 (1.406–⁠1.420) |
| Peak memory while serving (GiB) | **20.5** (20.5–⁠20.5) | 21.4 (21.4–⁠21.5) | 21.2 (21.1–⁠21.2) | 21.3 (21.2–⁠21.3) |

The order for memory and decode speed was the same in all four rounds. IronMLX and Splash swapped first and second place in completion time, with oMLX and TensorFold 0.6.6 third and fourth in all four rounds; TensorFold 0.6.6 had the fastest first token in three rounds and IronMLX in one, with Splash and oMLX third and fourth in all four rounds.

> [!NOTE]
> Peak memory while serving is measured while the six scored requests run. While loading the model and warming up, IronMLX peaks higher (about 21.8 GiB) than while serving; this page does not compare memory during loading.

### Results by task

All six tasks produce the same output length (256 tokens), but how often each application's drafts are accepted depends on the content, so the tasks differ in speed. The tables below show the median of four rounds for each task; the best value in each row is in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Coding 1 (62) | 1.062 | 1.005 | **0.955** | 1.236 |
| Coding 2 (54) | 0.962 | 0.849 | **0.727** | 0.900 |
| Coding 3 (83) | 1.127 | 1.162 | **0.998** | 1.269 |
| Knowledge 1 (46) | **1.417** | 1.700 | 1.471 | 1.878 |
| Knowledge 2 (40) | **1.323** | 1.517 | 1.503 | 1.548 |
| Knowledge 3 (49) | **1.513** | 1.970 | 1.781 | 1.851 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Coding 1 (62) | 260.1 | 287.3 | **301.7** | 220.6 |
| Coding 2 (54) | 286.6 | 343.9 | **406.4** | 310.2 |
| Coding 3 (83) | 248.4 | 263.2 | **285.0** | 218.3 |
| Knowledge 1 (46) | **190.1** | 159.8 | 185.3 | 141.1 |
| Knowledge 2 (40) | **203.4** | 179.7 | 180.8 | 172.3 |
| Knowledge 3 (49) | **177.2** | 136.3 | 153.0 | 143.5 |

IronMLX completes and decodes fastest on all three knowledge tasks; Splash completes and decodes fastest on all three coding tasks, where IronMLX also decodes more slowly than oMLX.

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request. According to each server's own logs and status pages:

- **IronMLX 0.2.0-rc.2:** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy; a few requests include a small number of windows without drafts.
- **oMLX 0.7.0:** the log records the speculative decoding acceptance for every request.
- **Splash 1.3.0:** the drafted and accepted counters on its status page grow with every request.
- **TensorFold 0.6.6:** Python engine; on this model it also uses its built-in copy proposer, which supplies a small share of the drafts on some tasks, with the rest from the DFlash2 draft model.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 14, 2026 (04:08–05:00 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, then until macOS thermal pressure had read nominal for 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.2 | Official release `IronMLX.dmg` (sha256 `55a37763…`), tag `v0.2.0-rc.2` → `0ef3c55` | Bundled server and Metal runtime, started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile, no tuning switches |
| oMLX 0.7.0 | Official app | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.3.0 | Official bottle | `--kv-format bf16 --max-context 8192`, SSD cache off; the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 0.6.6 | Source tag (`cb2ebf05`) | `--drafter-bits 4 --lane-kernels on --parallel 4`, prompt cache and snapshots off |

Common settings for all applications: 4-bit draft model, BF16 KV cache, 8,192-token context, and capacity 4 where configurable (the client sends one request at a time); each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens, counted both by the server and by the target tokenizer.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens), all different from each other and from the warm-up prompts.
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. Four rounds × four applications = 16 sessions, in the orders ABCD, BCDA, CDAB, and DABC, so each application has the same average position.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. For every application, each task's output was byte-identical in all four rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. For example, TensorFold 1.0.5 is not included because its native engine has not yet qualified Qwen3.6, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)](qwen36-moe-b4-short.md); for single requests with prompts of about 30K tokens, see [Qwen3.6 35B A3B B1 30K-prompt performance comparison](qwen36-moe-b1-30k.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Close results. IronMLX and Splash each had the shorter completion time in two of the four rounds, and their four-round medians differ by only 0.8%; four rounds cannot reliably separate them.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 56 ms for Splash (Knowledge 3: 94 to 150 ms) and by at most 21 ms for the others; decode speed varied by up to 6.1% for IronMLX, 5.5% for Splash, and at most 2.9% for oMLX and TensorFold 0.6.6.
