# Qwen3.6 35B A3B B1 single-request performance comparison (short prompts)

[简体中文](../zh-CN/benchmarks/qwen36-moe-b1-short.md)

> [!NOTE]
> This page compares single requests with short prompts. For 30K-token prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-30k.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-30k.md). Short-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B1 single request
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. All requests use short prompts (40–83 input tokens) with an output limit of 256 tokens, and the conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings (see the note below and [Splash's two settings](#splashs-two-settings)). TensorFold 1.2.0 does not support the incoai DFlash2 draft model used by the other applications on this page and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows`:
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; the output is reproducible.

## Summary

Based on all data collected in the short-prompt B1 single-request comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b1-short -->

- **Fastest time to first token.** The median time to first token is 43 ms, against 46 ms for Splash auto and 48 ms for Splash all, first in both rounds; the two-round ranges of IronMLX (42–44 ms) and Splash auto (45–47 ms) do not overlap, but the margin is 3 ms. oMLX takes 102 ms.
- **Fastest decode.** The median decode speed is 267.4 tok/s, 7.0% faster than Splash auto (250.0) and 18.9% faster than Splash all (224.9), first in both rounds; IronMLX measured 267.4 in both rounds, while Splash auto's two rounds lie within 244.9–255.1. oMLX reaches 234.6.
- **Shortest completion time.** The median time from sending a request to its completion is 1.026 s, 6.6% shorter than Splash auto (1.099 s) and 16.9% shorter than Splash all (1.236 s), first in both rounds; oMLX takes 1.259 s.
- **Lowest memory.** Peak memory while serving is 20.0 GiB, the lowest in both rounds; both Splash settings use about 21.0 GiB and oMLX 21.6 GiB.
- **Reproducible output.** For IronMLX and for Splash all, each task's output was byte-identical in both rounds; Splash auto differed between rounds on five tasks and oMLX on four.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, and reflects the response latency users perceive. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-short -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of two rounds, with the minimum and maximum of the two rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Time to first token (ms) | **43** (42–⁠44) | 102 (97–⁠106) | 46 (45–⁠47) | 48 (47–⁠50) |
| Decode speed (tok/s, higher is better) | **267.4** (267.4–⁠267.4) | 234.6 (234.2–⁠235.1) | 250.0 (244.9–⁠255.1) | 224.9 (221.6–⁠228.3) |
| Completion time (s) | **1.026** (1.026–⁠1.027) | 1.259 (1.241–⁠1.277) | 1.099 (1.078–⁠1.121) | 1.236 (1.231–⁠1.241) |
| Peak memory while serving (GiB) | **20.0** (20.0–⁠20.0) | 21.6 (21.6–⁠21.6) | 21.0 (21.0–⁠21.0) | 21.0 (21.0–⁠21.0) |

IronMLX was first in every metric in both rounds, and every metric had the same order in both rounds. In time to first token and completion time the order was IronMLX, Splash auto, Splash all, oMLX; in decode speed Splash all fell behind oMLX, so the order was IronMLX, Splash auto, oMLX, Splash all. In memory the order was IronMLX, Splash all, Splash auto, oMLX, with the two Splash settings less than 0.1 GiB apart.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests. While loading the model and warming up, IronMLX peaks higher (about 22.1 GiB) than while serving; this page does not compare memory during loading.

### Splash's two settings

On mixture-of-experts models, Splash 1.5.0 by default (`--verify-rows auto`) lets a verify budget learned from earlier cycles choose which rows of each draft block a decode cycle verifies; its documentation states that this is faster but lets greedy outputs differ from run to run. `--verify-rows all` verifies the whole draft block in every cycle and makes the output reproducible (on dense models the two settings are the same). This page lists both: the default shows out-of-the-box speed, and `--verify-rows all` shows the speed when output must be reproducible.

In this run, Splash auto decoded 11.1% faster than Splash all (250.0 against 224.9 tok/s) and completed 11.0% sooner (1.099 against 1.236 s), but its decode speed and completion time varied more between the two rounds. oMLX 0.7.3rc1 has a similar mechanism: its DFlash verify mode defaults to `adaptive`, which picks the number of drafts verified per cycle from measured cost and acceptance; oMLX runs with its default setting on this page. IronMLX has no such setting, and its output is always reproducible.

### Results by task

The six tasks have the same output length (256 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of two rounds for each task, with the best value in each row in bold.

Completion time (s, lower is better):

| Task (input tokens) | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Coding 1 (62) | **0.774** | 0.969 | 0.839 | 0.922 |
| Coding 2 (54) | 0.747 | 0.866 | 0.723 | **0.711** |
| Coding 3 (83) | **0.871** | 1.066 | 0.924 | 0.992 |
| Knowledge 1 (46) | 1.311 | 1.736 | **1.303** | 1.507 |
| Knowledge 2 (40) | **1.182** | 1.452 | 1.282 | 1.485 |
| Knowledge 3 (49) | 1.419 | 1.609 | **1.396** | 1.657 |

Decode speed (tok/s, higher is better):

| Task (input tokens) | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Coding 1 (62) | **349.1** | 298.6 | 322.7 | 291.6 |
| Coding 2 (54) | 363.0 | 334.8 | 376.5 | **385.5** |
| Coding 3 (83) | **311.6** | 281.3 | 292.8 | 271.8 |
| Knowledge 1 (46) | 201.0 | 158.5 | **202.4** | 174.9 |
| Knowledge 2 (40) | **223.2** | 188.0 | 206.1 | 177.1 |
| Knowledge 3 (49) | 185.2 | 169.1 | **189.1** | 158.5 |

IronMLX completes and decodes fastest on Coding 1, Coding 3, and Knowledge 2; Splash auto is fastest on Knowledge 1 and Knowledge 3; Splash all is fastest on Coding 2.

Every configuration uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX 0.2.0:** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy; every window carried drafts.
- **oMLX 0.7.3rc1:** the log records the speculative decoding acceptance for every request.
- **Splash 1.5.0 auto:** the drafted and accepted counters on its status page grow with every request; its startup log states that decode cycles verify the rows the verify budget chooses (`--verify-rows auto`).
- **Splash 1.5.0 all:** as above; its startup log states that decode cycles verify every row of the draft block (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 31, 2026 (23:04–23:22 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session was the exception, waiting for the thermal condition only and starting about 80 s after the previous server stopped. Thermal pressure was sampled every second during each session and stayed nominal throughout |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 8192 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 8,192 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.5.0 auto | Official bottle | `--max-context 8192`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.5.0 all | As above | Splash 1.5.0 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.5.0 auto's engine process not to) |
| TensorFold 1.2.0 | — | Not included: TensorFold 1.2.0 does not support the incoai DFlash2 draft model used by the other applications on this page. According to its [1.2.0 documentation](https://github.com/ashhart/TensorFold/blob/v1.2.0/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`); on Macs the model runs only on the Python engine frozen at 0.6.6, which drafts with the model's own MTP layer or with z-lab DFlash v1 |

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
- **Sessions:** each session starts a fresh server, sends two warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCS and SCBA, where A is IronMLX, B oMLX, C Splash auto, and S Splash all) and would have added two more rounds (BASC and CSAB) right after if, in any metric, two adjacent configurations swapped order between the rounds or their two-round ranges overlapped. Neither happened in any metric, so the rule did not add rounds: two rounds × four configurations = 8 sessions, with the same average position for each configuration.
- **Caching:** prefix and disk caches were off or verified unused for every request; speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the two rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 8 of 8 sessions and 48 of 48 scored requests passed every check; no session was retried or dropped.
- **Output reproducibility:** for IronMLX and for Splash all, each task's output was byte-identical in both rounds. Splash auto on Coding 1, Coding 3, and the three knowledge tasks, and oMLX on Coding 3 and the three knowledge tasks, produced outputs that differed between the two rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration for reproducible output); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because it does not support the incoai DFlash2 draft model, which says nothing about its performance with other draft models or on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **Close results.** IronMLX's lead in time to first token over Splash auto is 3 ms, and Splash auto's lead over Splash all is 2 ms; the two Splash settings differ by less than 0.1 GiB in memory; in completion time Splash all is 1.8% ahead of oMLX, and their two-round ranges (1.231–1.241 s and 1.241–1.277 s) almost touch. None of these small differences should be read as a clear advantage or disadvantage.
- Long prompts, other models, concurrent requests, or real multi-user traffic. For concurrent requests, see [Qwen3.6 35B A3B B4 concurrency performance comparison (short prompts)](qwen36-moe-b4-short.md); for single requests with prompts of about 30K tokens, see [Qwen3.6 35B A3B B1 30K-prompt performance comparison](qwen36-moe-b1-30k.md).
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the run-to-run output changes of Splash's and oMLX's default settings affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Run-to-run variation. Across the two rounds of the same task, time to first token varied by up to 64 ms for oMLX (Coding 3: 127 to 191 ms), 5 ms for each Splash setting, and 3 ms for IronMLX; decode speed varied by up to 31.0% for oMLX (Knowledge 1: 137.2 to 179.7 tok/s), 6.6% for Splash all (Coding 3: 263.2 to 280.5 tok/s), 5.8% for Splash auto, and 0.2% for IronMLX.
