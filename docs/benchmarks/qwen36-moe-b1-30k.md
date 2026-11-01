# Qwen3.6 35B A3B B1 30K-prompt performance comparison

[简体中文](../zh-CN/benchmarks/qwen36-moe-b1-30k.md)

> [!NOTE]
> This page compares single requests with 30K-token prompts. For short prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B1 single request
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-30k.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to handing a long document, part of a codebase, or a long conversation to the model in one go, where most of the wait is the model reading the long input. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the short-prompt pages (see the note below). TensorFold 1.2.0 does not support the incoai DFlash2 draft model used by the other applications on this page and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows` (explained on the [short-prompt B1 page](qwen36-moe-b1-short.md#splashs-two-settings)):
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; the output is reproducible.

## Summary

Based on all data collected in the 30K-prompt B1 single-request comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b1-30k -->

- **Lowest memory.** Peak memory while serving is 22.0 GiB, the lowest in all four rounds; both Splash settings use 22.9 GiB and oMLX 23.4 GiB.
- **Second in decode speed.** The median decode speed is 189.3 tok/s, 3.9% below Splash auto (196.9), which led in all four rounds; IronMLX decodes 2.2% faster than Splash all (185.2) and 24.0% faster than oMLX (152.7), faster than both in all four rounds.
- **Third in time to first token.** The median time to first token is 6.47 s, 3.6% longer than Splash auto (6.25 s) and 3.7% longer than Splash all (6.24 s); oMLX takes 6.68 s. IronMLX was third in all four rounds, but only 3 ms ahead of oMLX in round 2. With 30K inputs, most of the time to first token is the prefill that reads the long input.
- **Third in completion time.** The median time from sending a request to its completion is 7.89 s, third in all four rounds: 3.3% longer than Splash auto (7.63 s) and 2.8% longer than Splash all (7.67 s), and 5.4% shorter than oMLX (8.34 s). IronMLX decodes faster than Splash all, but not enough to make up for the gap in prefill.
- **Reproducible output.** For IronMLX and for Splash all, each task's output was byte-identical in all four rounds; oMLX differed between rounds on five tasks and Splash auto on six.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long input. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-30k -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Time to first token (s) | 6.47 (6.46–⁠6.53) | 6.68 (6.48–⁠6.74) | 6.25 (6.23–⁠6.26) | **6.24** (6.22–⁠6.25) |
| Decode speed (tok/s, higher is better) | 189.3 (188.9–⁠189.5) | 152.7 (146.8–⁠157.1) | **196.9** (195.4–⁠209.5) | 185.2 (184.3–⁠185.3) |
| Completion time (s) | 7.89 (7.86–⁠7.96) | 8.34 (8.19–⁠8.36) | **7.63** (7.61–⁠7.66) | 7.67 (7.66–⁠7.69) |
| Peak memory while serving (GiB) | **22.0** (21.9–⁠22.0) | 23.4 (23.4–⁠23.4) | 22.9 (22.9–⁠22.9) | 22.9 (22.9–⁠22.9) |

In decode speed, the order was Splash auto, IronMLX, Splash all, oMLX in all four rounds. In completion time, IronMLX was third and oMLX fourth in all four rounds; Splash auto led in rounds 2–4 and Splash all in round 1. In time to first token, the two Splash settings were within 0.04 s of each other in every round (all led in rounds 1, 2, and 4, auto in round 3); IronMLX was third and oMLX fourth in all four rounds, though in round 2 oMLX's time dropped to 6.48 s, 3 ms behind IronMLX. IronMLX used the least memory and oMLX the most in all four rounds; the two Splash settings differ by less than 0.02 GiB in any round.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests and includes the KV cache of a 30K input. Peaks while loading the model and warming up: about 22.1 GiB for IronMLX, about the same as while serving; about 21.6 GiB for oMLX and 20.9 GiB for both Splash settings. This page does not compare memory during loading.

### Splash's two settings

With 30K inputs, the two Splash settings have almost the same time to first token (6.25 against 6.24 s); Splash auto decodes 6.3% faster than Splash all (196.9 against 185.2 tok/s) and completes 0.5% sooner (7.63 against 7.67 s). At 30K most of the completion time is prefill, so the difference in decode speed matters less for completion time.

### Results by task

The six tasks have the same input length (30,732 tokens) and the same output length (256 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of four rounds for each task, with the best value in each row in bold.

Time to first token (s, lower is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Coding 1 | 6.30 | 7.02 | **6.03** | 6.04 |
| Coding 2 | 6.27 | 6.33 | 6.12 | **6.11** |
| Coding 3 | 6.32 | 6.35 | 6.16 | **6.13** |
| Knowledge 1 | 6.64 | 6.58 | **6.31** | 6.34 |
| Knowledge 2 | 6.90 | 6.86 | **6.60** | 6.65 |
| Knowledge 3 | 6.93 | 6.86 | **6.69** | 6.71 |

Completion time (s, lower is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Coding 1 | 7.25 | 8.11 | 6.96 | **6.91** |
| Coding 2 | **6.99** | 7.27 | 7.22 | 7.26 |
| Coding 3 | 7.57 | 7.98 | 7.30 | **7.18** |
| Knowledge 1 | 8.23 | 8.46 | **8.00** | 8.09 |
| Knowledge 2 | 8.35 | 8.56 | **8.05** | 8.37 |
| Knowledge 3 | 8.75 | 8.88 | **8.56** | 8.81 |

Decode speed (tok/s, higher is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 auto | Splash 1.5.0 all |
|---|---|---|---|---|
| Coding 1 | 268.7 | 234.1 | 276.9 | **295.0** |
| Coding 2 | **356.7** | 272.1 | 235.0 | 222.8 |
| Coding 3 | 203.0 | 152.8 | 219.6 | **243.2** |
| Knowledge 1 | **159.7** | 131.9 | 152.1 | 145.8 |
| Knowledge 2 | 175.7 | 150.2 | **175.9** | 147.7 |
| Knowledge 3 | **140.2** | 126.6 | 136.1 | 121.4 |

IronMLX's time to first token is third on the three coding tasks (behind both Splash settings, ahead of oMLX) and fourth on the three knowledge tasks, 42–75 ms behind oMLX. Its completion time is the shortest on Coding 2, second on Knowledge 2 and Knowledge 3 (behind Splash auto, ahead of Splash all), and third on the other three tasks. In decode speed, IronMLX is clearly fastest on Coding 2 (51.8% above Splash auto), fastest on Knowledge 1 and Knowledge 3 (5.0% and 3.0% above Splash auto), 0.1% behind Splash auto on Knowledge 2, and clearly slower than both Splash settings on Coding 1 and Coding 3.

Every configuration uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX 0.2.0:** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy; all 2,500 verification windows carried drafts.
- **oMLX 0.7.3rc1:** the log records the speculative decoding acceptance for every request.
- **Splash 1.5.0 auto:** the drafted and accepted counters on its status page grow with every request; its startup log states that decode cycles verify the rows the verify budget chooses (`--verify-rows auto`).
- **Splash 1.5.0 all:** as above; its startup log states that decode cycles verify every row of the draft block (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 31 to November 1, 2026 (23:33–00:22 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session of each of the two formal runs was the exception, waiting for the thermal condition only and starting about 83 s and 143 s after the previous server stopped. Thermal pressure was sampled every second during each session and stayed nominal during every scored request |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.5.0 auto | Official bottle | `--max-context 32768`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.5.0 all | As above | Splash 1.5.0 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.5.0 auto's engine process not to) |
| TensorFold 1.2.0 | — | Not included: TensorFold 1.2.0 does not support the incoai DFlash2 draft model used by the other applications on this page. According to its [1.2.0 documentation](https://github.com/ashhart/TensorFold/blob/v1.2.0/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`); on Macs the model runs only on the Python engine frozen at 0.6.6, which drafts with the model's own MTP layer or with z-lab DFlash v1 |

Common settings for all applications: 4-bit draft model, 32,768-token context, capacity 4 where configurable (the client sends one request at a time), and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy. Each server's effective context was verified as 32,768 from its own interface or settings after startup.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All three applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** six 30K prompts (three coding, three knowledge). Each prompt is a block of synthetic project material related to the question, followed by the question; the user text is exactly 30,720 tokens with the target tokenizer, and 30,732 tokens with the chat template. The six prompts all differ, and any two requests share at most their first 8 tokens (template and heading).
- **Full input:** every request's server-side prompt token count was 30,732, matching the client's count with the target tokenizer; no input was truncated, compressed, or dropped. All applications received exactly the same text.
- **Sessions:** each session starts a fresh server, sends two short warm-up requests, then the six 30K scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCS and SCBA, where A is IronMLX, B oMLX, C Splash auto, and S Splash all) and added two more rounds (BASC and CSAB) right after if, in any metric, two adjacent configurations swapped order between the rounds or their two-round ranges overlapped. Over the first two rounds, the two-round ranges of IronMLX and oMLX overlapped in time to first token, and the two Splash settings swapped order in completion time, so the rounds were added: four rounds × four configurations = 16 sessions, with the same average position for each configuration.
- **Caching:** prefix and disk caches were off or verified unused for every request (zero cached tokens reported by the server and unchanged hit counters); speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not the same as the GPU prefill time alone; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens. In this run 94 of the 96 recounts were exactly 256; the other two (Splash auto on Coding 2 in rounds 1 and 2) were 255, within the allowed difference.
- **Output reproducibility:** for IronMLX and for Splash all, each task's output was byte-identical in all four rounds. oMLX on Coding 1, Coding 3, and the three knowledge tasks, and Splash auto on all six tasks, produced outputs that were not identical across the four rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration for reproducible output); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support the incoai DFlash2 draft model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **Close results.** The two Splash settings differ by 0.1% in time to first token and 0.5% in completion time, and in both metrics each led in some rounds; IronMLX and Splash all differ by 2.2% in decode speed; in round 2, IronMLX's time to first token was only 3 ms shorter than oMLX's. None of these small differences should be read as a clear advantage or disadvantage.
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page covers one long input of about 30K tokens only; for four 30K requests arriving together, see [Qwen3.6 35B A3B B4 30K-prompt concurrency performance comparison](qwen36-moe-b4-30k.md), and for short prompts, see the links at the top of this page.
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the run-to-run output changes of Splash's and oMLX's default settings affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a conversation reuses context it already has, the time to first token can be very different.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 714 ms for oMLX (Coding 1: 6.35 to 7.07 s), 177 ms for IronMLX (Coding 1), 113 ms for Splash all, and 94 ms for Splash auto; decode speed varied by up to 16.0% for oMLX (Coding 3: 144.3 to 167.4 tok/s), 14.3% for Splash auto (Coding 1: 258.2 to 295.0 tok/s), 1.5% for IronMLX, and 0.9% for Splash all.
