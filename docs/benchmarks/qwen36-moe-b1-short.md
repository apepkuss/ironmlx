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

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. All requests use short prompts (40–83 input tokens) with an output limit of 256 tokens, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings (see the note below and [Splash's two settings](#splashs-two-settings)). TensorFold 1.1.6 does not support this model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows`:
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; the output is reproducible.

## Summary

Based on all data collected in the short-prompt B1 single-request comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b1-short -->

- **Lowest memory.** Peak memory while serving is 20.0 GiB, the lowest in all four rounds; both Splash settings use 21.1 GiB and oMLX 21.6 GiB.
- **Tied with Splash auto for first in decode speed.** The median decode speed is 249.7 tok/s against 253.6 tok/s for Splash auto, and each led in two of the four rounds; Splash auto ranged from 225.1 to 263.1 over the four rounds, IronMLX from 248.8 to 251.5. IronMLX decodes 3.0% faster than Splash all (242.4), faster in all four rounds; oMLX reaches 236.1.
- **Second in completion time.** The median time from sending a request to its completion is 1.090 s, 0.7% longer than Splash auto (1.082 s), which led in three of the four rounds, and 6.2% shorter than Splash all (1.162 s), shorter in all four rounds; oMLX takes 1.260 s.
- **Third in time to first token, by a few milliseconds.** The median time to first token is 50 ms, against 47 ms for Splash auto and 48 ms for Splash all; the four-round ranges of the three overlap. oMLX takes 102 ms.
- **Reproducible output.** For IronMLX and for Splash all, each task's output was byte-identical in all four rounds; Splash auto differed between rounds on five tasks and oMLX on four.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-short -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Time to first token (ms) | 50 (44–⁠52) | 102 (100–⁠103) | **47** (46–⁠51) | 48 (46–⁠49) |
| Decode speed (tok/s, higher is better) | 249.7 (248.8–⁠251.5) | 236.1 (232.5–⁠237.8) | **253.6** (225.1–⁠263.1) | 242.4 (234.9–⁠248.0) |
| Completion time (s) | 1.090 (1.080–⁠1.098) | 1.260 (1.238–⁠1.292) | **1.082** (1.059–⁠1.188) | 1.162 (1.117–⁠1.173) |
| Peak memory while serving (GiB) | **20.0** (20.0–⁠20.0) | 21.6 (21.5–⁠21.6) | 21.1 (21.0–⁠21.2) | 21.1 (21.0–⁠21.2) |

IronMLX used the least memory and oMLX the most in all four rounds. oMLX had the slowest time to first token in all four rounds; the order of the other three changed from round to round. For completion time, the first three rounds ran Splash auto, IronMLX, Splash all, and oMLX; in the fourth round Splash auto slowed down markedly (decode speed 225.1 tok/s) and IronMLX was first. In decode speed, IronMLX was first in rounds 1 and 4 and Splash auto in rounds 2 and 3.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests. While loading the model and warming up, IronMLX peaks higher (about 22.1 GiB) than while serving; this page does not compare memory during loading.

### Splash's two settings

On mixture-of-experts models, Splash 1.4.1 by default (`--verify-rows auto`) lets a verify budget learned from earlier cycles choose which rows of each draft block a decode cycle verifies; its documentation states that this is faster but lets greedy outputs differ from run to run. `--verify-rows all` verifies the whole draft block in every cycle and makes the output reproducible (on dense models the two settings are the same). This page lists both: the default shows out-of-the-box speed, and `--verify-rows all` shows the speed when output must be reproducible.

In this run, Splash auto decoded 4.6% faster than Splash all (253.6 against 242.4 tok/s) and completed 6.9% sooner (1.082 against 1.162 s), but varied more between rounds. oMLX 0.7.2 has a similar mechanism: its DFlash verify mode defaults to `adaptive`, which picks the number of drafts verified per cycle from measured cost and acceptance; oMLX runs with its default setting on this page. IronMLX has no such setting, and its output is always reproducible.

### Results by task

The six tasks have the same output length (256 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of four rounds for each task, with the best value in each row in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Coding 1 (62) | **0.819** | 0.939 | 0.846 | 0.894 |
| Coding 2 (54) | 0.761 | 0.861 | 0.718 | **0.699** |
| Coding 3 (83) | 0.969 | 1.092 | **0.907** | 0.936 |
| Knowledge 1 (46) | 1.518 | 1.923 | **1.348** | 1.430 |
| Knowledge 2 (40) | **1.215** | 1.444 | 1.275 | 1.381 |
| Knowledge 3 (49) | 1.470 | 1.601 | **1.420** | 1.691 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Coding 1 (62) | **332.3** | 308.0 | 319.4 | 302.3 |
| Coding 2 (54) | 362.5 | 337.2 | 381.3 | **391.2** |
| Coding 3 (83) | 280.5 | 282.2 | **299.6** | 294.7 |
| Knowledge 1 (46) | 173.3 | 139.7 | **195.4** | 187.2 |
| Knowledge 2 (40) | **218.1** | 189.5 | 207.2 | 190.6 |
| Knowledge 3 (49) | 179.3 | 169.9 | **185.7** | 154.9 |

IronMLX completes and decodes fastest on Coding 1 and Knowledge 2; Splash auto is fastest on Coding 3, Knowledge 1, and Knowledge 3; Splash all is fastest on Coding 2.

Every configuration uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX dev (c6ef14ec2):** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy; every window carried drafts.
- **oMLX 0.7.2:** the log records the speculative decoding acceptance for every request.
- **Splash 1.4.1 auto:** the drafted and accepted counters on its status page grow with every request; its startup log states that decode cycles verify the rows the verify budget chooses (`--verify-rows auto`).
- **Splash 1.4.1 all:** as above; its startup log states that decode cycles verify every row of the draft block (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 28, 2026 (05:55–06:32 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.4.1 auto | Official bottle | `--max-context 8192`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.4.1 all | As above | Splash 1.4.1 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.4.1 auto's engine process not to) |
| TensorFold 1.1.6 | — | Not included: according to its [1.1.6 documentation](https://github.com/ashhart/TensorFold/blob/v1.1.6/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`), which runs only on the older Python engine |

Common settings for all applications: 4-bit draft model, 8,192-token context, capacity 4 where configurable (the client sends one request at a time), and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All three applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** six short prompts (three coding, three knowledge; 40–83 input tokens), all different from each other and from the warm-up prompts.
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCS and SCBA, where A is IronMLX, B oMLX, C Splash auto, and S Splash all) and added two more rounds (BASC and CSAB) right after if, in any metric, two adjacent configurations swapped order between the rounds or their two-round ranges overlapped. Time to first token (the two Splash settings) and decode speed (Splash auto and IronMLX) swapped order over the first two rounds, so the rounds were added: four rounds × four configurations = 16 sessions, with the same average position for each configuration.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.
- **Output reproducibility:** for IronMLX and for Splash all, each task's output was byte-identical in all four rounds. Splash auto on Coding 1, Coding 3, and the three knowledge tasks, and oMLX on Coding 1 and the three knowledge tasks, produced outputs that were not identical across the four rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration for reproducible output); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- **Close results.** IronMLX and Splash auto each led in two of the four rounds in decode speed and differ by only 0.7% in completion time; time to first token differs by a few milliseconds among three configurations whose four-round ranges overlap. None of these should be read as a clear advantage or disadvantage.
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)](qwen36-moe-b4-short.md); for single requests with prompts of about 30K tokens, see [Qwen3.6 35B A3B B1 30K-prompt performance comparison](qwen36-moe-b1-30k.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the run-to-run output changes of Splash's and oMLX's default settings affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 80 ms for oMLX (Coding 3: 116 to 196 ms), 39 ms for Splash all, 26 ms for IronMLX, and 16 ms for Splash auto; decode speed varied by up to 32.4% for Splash auto (Coding 3: 241.4 to 319.6 tok/s), 23.2% for oMLX, 19.8% for Splash all, and 3.9% for IronMLX.
