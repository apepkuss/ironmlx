# Qwen3.6 35B A3B B1 30K-prompt performance comparison

[简体中文](../zh-CN/benchmarks/qwen36-moe-b1-30k.md)

> [!NOTE]
> This page compares single requests with 30K-token prompts. For short prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B1 single request
> - [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-30k.md)
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-30k.md)
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-30k.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to handing a long document, part of a codebase, or a long conversation to the model in one go, where most of the wait is the model reading the long input. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, and IronMLX, with Splash tested in two settings as on the short-prompt pages (see the note below). TensorFold 1.1.6 does not support this model and is not included. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

> [!NOTE]
> Splash was tested in two settings on this page, named after the value of `--verify-rows` (explained on the [short-prompt B1 page](qwen36-moe-b1-short.md#splashs-two-settings)):
>
> - **Splash auto**: the default setting (`--verify-rows auto`). Faster, but on this model its greedy output may differ from run to run.
> - **Splash all**: `--verify-rows all`, which verifies the whole draft block in every cycle; the output is reproducible.

## Summary

Based on all data collected in the 30K-prompt B1 single-request comparison, IronMLX's overall performance is as follows (ranked among the four configurations):

<!-- benchmark:kpis qwen36-moe-b1-30k -->

- **Lowest memory.** Peak memory while serving is 22.1 GiB, the lowest in all four rounds; both Splash settings use 23.1 GiB and oMLX 23.4 GiB.
- **Second in decode speed.** The median decode speed is 184.3 tok/s, second in all four rounds: 5.0% below Splash auto (194.0), 4.3% above Splash all (176.8), and 21.0% above oMLX (152.4).
- **Longest time to first token.** The median time to first token is 7.13 s, the longest in all four rounds; Splash all (6.32 s) and Splash auto (6.34 s) tie for the shortest, 12.5–12.8% shorter than IronMLX, and oMLX takes 6.70 s. With 30K inputs, most of the time to first token is the prefill that reads the long input.
- **Longest completion time.** The median time from sending a request to its completion is 8.57 s, the longest in all four rounds; Splash auto is the shortest (7.71 s), and IronMLX takes 11.1% longer. Splash all takes 7.85 s and oMLX 8.35 s. IronMLX's lead in decode speed is not enough to make up for the gap in prefill.
- **Reproducible output.** For IronMLX and for Splash all, each task's output was byte-identical in all four rounds; oMLX differed between rounds on four tasks and Splash auto on six.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects reading the long input. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-30k -->

### Comparison by metric

The table below lists the results of each configuration for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Time to first token (s) | 7.13 (7.05–⁠7.26) | 6.70 (6.50–⁠6.83) | 6.34 (6.23–⁠6.38) | **6.32** (6.20–⁠6.40) |
| Decode speed (tok/s, higher is better) | 184.3 (184.0–⁠185.5) | 152.4 (145.9–⁠157.3) | **194.0** (192.2–⁠202.3) | 176.8 (176.0–⁠177.9) |
| Completion time (s) | 8.57 (8.52–⁠8.73) | 8.35 (8.17–⁠8.41) | **7.71** (7.61–⁠7.82) | 7.85 (7.70–⁠7.92) |
| Peak memory while serving (GiB) | **22.1** (22.1–⁠22.1) | 23.4 (23.3–⁠23.4) | 23.1 (23.0–⁠23.1) | 23.1 (23.1–⁠23.1) |

The order of decode speed and completion time was the same in all four rounds: Splash auto, IronMLX, Splash all, and oMLX for decode speed; Splash auto, Splash all, oMLX, and IronMLX for completion time. In time to first token, oMLX was third and IronMLX fourth in all four rounds, while of the two Splash settings auto led in rounds 1 and 2 and all in rounds 3 and 4. IronMLX used the least memory and oMLX the most in all four rounds; the two Splash settings differ by less than 0.01 GiB.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests and includes the KV cache of a 30K input. Peaks while loading the model and warming up: about 22.1 GiB for IronMLX, about the same as while serving; about 21.5 GiB for oMLX and 21.0 GiB for both Splash settings. This page does not compare memory during loading.

### Splash's two settings

With 30K inputs, the two Splash settings have almost the same time to first token (6.34 against 6.32 s); Splash auto decodes 9.7% faster than Splash all (194.0 against 176.8 tok/s) and completes 1.7% sooner (7.71 against 7.85 s). At 30K most of the completion time is prefill, so the difference in decode speed matters less for completion time.

### Results by task

The six tasks have the same input length (30,732 tokens) and the same output length (256 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of four rounds for each task, with the best value in each row in bold.

Time to first token (s, lower is better):

| Task | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Coding 1 | 6.77 | 7.06 | **6.04** | 6.07 |
| Coding 2 | 6.92 | 6.37 | **6.19** | 6.20 |
| Coding 3 | 6.90 | 6.42 | 6.25 | **6.24** |
| Knowledge 1 | 7.33 | 6.62 | 6.41 | **6.40** |
| Knowledge 2 | 7.49 | 6.87 | **6.68** | 6.76 |
| Knowledge 3 | 7.43 | 6.94 | **6.77** | 6.78 |

Completion time (s, lower is better):

| Task | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Coding 1 | 7.77 | 8.15 | **6.92** | 6.94 |
| Coding 2 | 7.64 | **7.32** | 7.35 | 7.51 |
| Coding 3 | 8.18 | 7.98 | **7.35** | 7.50 |
| Knowledge 1 | 8.98 | 8.61 | **8.05** | 8.17 |
| Knowledge 2 | 9.00 | 8.67 | **8.11** | 8.37 |
| Knowledge 3 | 9.30 | 9.03 | **8.65** | 8.97 |

Decode speed (tok/s, higher is better):

| Task | IronMLX dev (c6ef14ec2) | oMLX 0.7.2 | Splash 1.4.1 auto | Splash 1.4.1 all |
|---|---|---|---|---|
| Coding 1 | 249.7 | 233.7 | 297.3 | **301.5** |
| Coding 2 | **356.0** | 270.5 | 217.5 | 194.8 |
| Coding 3 | 199.7 | 165.0 | **231.1** | 201.3 |
| Knowledge 1 | **154.2** | 130.4 | 153.2 | 144.2 |
| Knowledge 2 | 169.1 | 142.3 | **174.2** | 158.8 |
| Knowledge 3 | **135.4** | 118.4 | 134.4 | 117.6 |

IronMLX has the longest time to first token on five tasks and is shorter than oMLX on Coding 1. Its completion time is the longest on five tasks and third on Coding 1. In decode speed, IronMLX is clearly fastest on Coding 2 (63.7% above Splash auto), within 1% of Splash auto on Knowledge 1 and Knowledge 3, and clearly slower than Splash auto on Coding 1 and Coding 3.

Every configuration uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX dev (c6ef14ec2):** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy; all 2,354 verification windows carried drafts.
- **oMLX 0.7.2:** the log records the speculative decoding acceptance for every request.
- **Splash 1.4.1 auto:** the drafted and accepted counters on its status page grow with every request; its startup log states that decode cycles verify the rows the verify budget chooses (`--verify-rows auto`).
- **Splash 1.4.1 all:** as above; its startup log states that decode cycles verify every row of the draft block (`--verify-rows all`).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 28, 2026 (10:25–11:15 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session and stayed nominal during every scored request |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX dev (c6ef14ec2) | Release build (`cargo build --locked --release`) of commit `c6ef14ec2` on the `dev` branch; server and MLX Metal runtime frozen together | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile (the ANE does not apply to this model), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings, DFlash verify mode left at its default `adaptive` |
| Splash 1.4.1 auto | Official bottle | `--max-context 32768`; KV cache kept at its default int8; `--verify-rows` left at its default `auto`; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| Splash 1.4.1 all | As above | Splash 1.4.1 auto plus `--verify-rows all`; after startup, the Splash engine process was verified to carry the option (and Splash 1.4.1 auto's engine process not to) |
| TensorFold 1.1.6 | — | Not included: according to its [1.1.6 documentation](https://github.com/ashhart/TensorFold/blob/v1.1.6/docs/recipes/qwen3.6-moe.md), the Metal native engine has no family for this model (`qwen3_5_moe`), which runs only on the older Python engine |

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
- **Sessions:** each session starts a fresh server, sends two short warm-up requests, then the six 30K scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCS and SCBA, where A is IronMLX, B oMLX, C Splash auto, and S Splash all) and added two more rounds (BASC and CSAB) right after if, in any metric, two adjacent configurations swapped order between the rounds or their two-round ranges overlapped. The two-round ranges of the two Splash settings overlapped in time to first token, so the rounds were added: four rounds × four configurations = 16 sessions, with the same average position for each configuration.
- **Caching:** prefix and disk caches were off or verified unused for every request (zero cached tokens reported by the server and unchanged hit counters); speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not the same as the GPU prefill time alone; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens. In this run all 96 recounts were exactly 256.
- **Output reproducibility:** for IronMLX and for Splash all, each task's output was byte-identical in all four rounds. oMLX on Coding 3 and the three knowledge tasks, and Splash auto on all six tasks, produced outputs that were not identical across the four rounds.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test (Splash all is an additional configuration for reproducible output); with other models, other workloads, or each application's recommended settings, the results and their order may differ. TensorFold is not included because its current release does not support this model, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a development build.** The IronMLX on this page is a build of the `dev` branch, not a release; a release may perform differently.
- **No order between the two Splash settings in time to first token.** They differ by 0.3%, and each led in two of the four rounds.
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page covers one long input of about 30K tokens only; for four 30K requests arriving together, see [Qwen3.6 35B A3B B4 30K-prompt concurrency performance comparison](qwen36-moe-b4-30k.md), and for short prompts, see the links at the top of this page.
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality, including whether the run-to-run output changes of Splash's and oMLX's default settings affect it.
- A warm machine. Every session on this page started from nominal thermal pressure; sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a conversation reuses context it already has, the time to first token can be very different.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 783 ms for oMLX (Coding 1), 386 ms for IronMLX, 294 ms for Splash all, and 186 ms for Splash auto; decode speed varied by up to 22.6% for oMLX (Knowledge 1: 111.0 to 136.1 tok/s), 15.6% for Splash auto, 7.8% for Splash all, and 3.7% for IronMLX.
