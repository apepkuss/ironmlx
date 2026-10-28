# Qwen3.6 35B A3B B1 30K-prompt performance comparison

[简体中文](../zh-CN/benchmarks/qwen36-moe-b1-30k.md)

> [!NOTE]
> This page compares single requests with 30K-token prompts. For short prompts, see [Qwen3.6 35B A3B B1 single request](qwen36-moe-b1-short.md) and [Qwen3.6 35B A3B B4 concurrency](qwen36-moe-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.6 35B A3B B1 single request
> - [Qwen3.8 27B B1 single request](qwen38-dense-b1-30k.md)
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-30k.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the Qwen3.6-35B-A3B mixture-of-experts model with one request at a time. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with a fixed 256-token output. This workload corresponds to handing the model a long document, part of a codebase, or a long conversation in one go, where most of the wait is the model reading the long input. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the 30K long-prompt B1 single-request comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen36-moe-b1-30k -->

- **Longest completion time.** The median time from sending a request to its completion is 14.20 s, the longest of the four applications in all four rounds; oMLX is the shortest (8.41 s), and IronMLX takes 68.9% longer. Splash takes 8.90 s and TensorFold 0.6.6 10.97 s.
- **Longest time to first token.** The median time to first token is 11.98 s, the longest in all four rounds and 77.3% longer than the fastest, oMLX (6.76 s). Splash takes 7.28 s and TensorFold 0.6.6 8.60 s. With a 30K input, most of the time to first token is the prefill that reads the long input: IronMLX's own log records prefill times of 11.0–12.2 s. This is also where most of the completion-time gap comes from.
- **Third in decode speed.** The median decode speed is 116.3 tok/s, 26.5% below Splash (158.2) and also below oMLX (146.2); IronMLX decodes faster than TensorFold 0.6.6 (110.8) in all four rounds.
- **Third in memory.** Peak memory while serving is 25.3 GiB, 8.1% more than oMLX (23.4 GiB) and slightly more than Splash (25.1 GiB) in all four rounds; TensorFold 0.6.6 uses 30.8 GiB.

### Decode speed vs. time to first token

Decode speed is the number of tokens a request generates per second once its output has started, and reflects how fast long replies are produced. Time to first token (TTFT) is the time from sending a request to receiving its first output; with long prompts it mainly reflects the time to read the long input. Points toward the lower right are better.

<!-- benchmark:scatter qwen36-moe-b1-30k -->

> [!NOTE]
> TensorFold 1.0.5 is not included: according to its [1.0.5 documentation](https://github.com/ashhart/TensorFold/blob/v1.0.5/README.md#qualified-models), Qwen3.6 is still being qualified for the native engine, and the native engine accepts an external DFlash2 draft model only for Qwen3.8-27B-class checkpoints ([source](https://github.com/ashhart/TensorFold/blob/v1.0.5/zig/src/native/metal.zig#L101-L105)).

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Time to first token (s) | 11.98 (11.89–⁠12.02) | **6.76** (6.70–⁠6.79) | 7.28 (7.27–⁠7.30) | 8.60 (8.59–⁠8.64) |
| Decode speed (tok/s, higher is better) | 116.3 (114.4–⁠116.8) | 146.2 (145.2–⁠146.5) | **158.2** (157.1–⁠158.7) | 110.8 (110.1–⁠111.3) |
| Completion time (s) | 14.20 (14.11–⁠14.27) | **8.41** (8.39–⁠8.42) | 8.90 (8.88–⁠8.91) | 10.97 (10.93–⁠11.00) |
| Peak memory while serving (GiB) | 25.3 (25.2–⁠25.4) | **23.4** (23.3–⁠23.4) | 25.1 (25.1–⁠25.1) | 30.8 (30.8–⁠30.8) |

The order was the same in all four rounds for all four metrics: oMLX, Splash, TensorFold 0.6.6, and IronMLX for both time to first token and completion time; Splash, oMLX, IronMLX, and TensorFold 0.6.6 for decode speed; and oMLX, Splash, IronMLX, and TensorFold 0.6.6 for memory.

> [!NOTE]
> Peak memory while serving is measured while the six scored requests run and includes the KV cache of the 30K input. After loading the model and warming up, the applications peak at about 20.8–23.0 GiB (IronMLX about 21.8 GiB); this page does not compare memory during loading.

### Results by task

All six tasks have the same input length (30,732 tokens) and output length (256 tokens), but how often each application's drafts are accepted depends on the content, so the tasks differ in speed. The tables below show the median of four rounds for each task; the best value in each row is in bold.

Time to first token (s, lower is better):

| Task | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Coding 1 | 11.13 | **7.10** | 7.13 | 8.11 |
| Coding 2 | 11.28 | **6.31** | 7.12 | 8.20 |
| Coding 3 | 11.93 | **6.37** | 7.17 | 8.49 |
| Knowledge 1 | 12.18 | **6.57** | 7.40 | 8.93 |
| Knowledge 2 | 12.02 | **6.95** | 7.74 | 8.83 |
| Knowledge 3 | 12.17 | **7.02** | 7.71 | 8.73 |

Completion time (s, lower is better):

| Task | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Coding 1 | 12.89 | 8.27 | **8.19** | 10.04 |
| Coding 2 | 13.09 | **7.09** | 8.38 | 9.46 |
| Coding 3 | 14.04 | **8.00** | 8.73 | 10.77 |
| Knowledge 1 | 14.47 | **8.54** | 9.07 | 11.57 |
| Knowledge 2 | 14.37 | **8.83** | 9.49 | 11.15 |
| Knowledge 3 | 14.64 | **9.47** | 9.85 | 11.92 |

Decode speed (tok/s, higher is better):

| Task | IronMLX 0.2.0-rc.2 | oMLX 0.7.0 | Splash 1.3.0 | TensorFold 0.6.6 |
|---|---|---|---|---|
| Coding 1 | 142.6 | 217.5 | **239.7** | 133.4 |
| Coding 2 | 141.6 | **327.9** | 203.7 | 203.1 |
| Coding 3 | 121.1 | 157.2 | **163.6** | 111.8 |
| Knowledge 1 | 111.3 | 129.2 | **152.8** | 96.7 |
| Knowledge 2 | 109.0 | 135.3 | **145.5** | 109.5 |
| Knowledge 3 | 101.9 | 104.3 | **118.7** | 79.8 |

IronMLX has the longest time to first token and completion time on all six tasks. oMLX has the shortest time to first token on all six tasks and the shortest completion time on five; Splash completes Coding 1 fastest. Splash decodes fastest on five tasks and oMLX on Coding 2; IronMLX decodes faster than TensorFold 0.6.6 on five tasks and slightly slower on Knowledge 2.

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request; every scored request had accepted drafts, and no application fell back to plain decoding for a whole request. According to each server's own logs and status pages:

- **IronMLX 0.2.0-rc.2:** linear draft verification (no draft tree for this model), with the number of drafts per window set automatically by its default window-cost policy. In the later requests of a session, some requests include many windows without drafts: the policy adjusts its budget to the acceptance seen so far, and every request still has accepted DFlash2 drafts. Each request's output was identical in all four rounds.
- **oMLX 0.7.0:** the log records the speculative decoding acceptance for every request; its long-context fallback was not triggered (`dflash_max_ctx` is not set).
- **Splash 1.3.0:** the drafted and accepted counters on its status page grow with every request.
- **TensorFold 0.6.6:** Python engine; on this model it also uses its built-in copy proposer, which supplies a few drafts (at most 15 tokens) on some requests, with the rest from the DFlash2 draft model.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Power | Connected to AC power for every session |
| Test date | October 14, 2026 (06:52–08:01 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, then until macOS thermal pressure had read nominal for 60 seconds in a row. Thermal pressure was sampled every second during each session: it stayed nominal throughout the oMLX, Splash, and TensorFold 0.6.6 sessions; in all four IronMLX sessions it rose during the last task (Knowledge 3) and was nominal during the other tasks |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.2 | Official release `IronMLX.dmg` (sha256 `55a37763…`), tag `v0.2.0-rc.2` → `0ef3c55` | Bundled server and Metal runtime, started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which uses the draft model's own block size of 8); default M5 profile, no tuning switches |
| oMLX 0.7.0 | Official app | `--no-cache --max-concurrent-requests 4`; context 32,768 (`max_context_window` and `max_context_window_policy`) and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.3.0 | Official bottle | `--kv-format bf16 --max-context 32768`, SSD cache off; the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 0.6.6 | Source tag (`cb2ebf05`) | `--drafter-bits 4 --lane-kernels on --parallel 4 --context 32768`, prompt cache and snapshots off |

Common settings for all applications: 4-bit draft model, BF16 KV cache, 32,768-token context, and capacity 4 where configurable (the client sends one request at a time); each application otherwise keeps its own default strategy. Each server's effective context was read back from its own interface after startup and was 32,768 in every case.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| Draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. Every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** six 30K prompts (three coding, three knowledge). Each consists of synthetic project material related to the question, followed by the question itself. The user text is exactly 30,720 tokens by the target tokenizer, and 30,732 tokens with the chat template. The six prompts all differ, and any two requests share at most their first 8 tokens (template and title).
- **Complete input:** every request's server-side prompt token count was 30,732, matching the client's count with the target tokenizer; no input was truncated, compressed, or dropped. All applications received exactly the same text.
- **Sessions:** each session starts a fresh server, sends two short-prompt warm-up requests, then the six 30K scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. Four rounds × four applications = 16 sessions, in the orders ABCD, BCDA, CDAB, and DABC, so each application has the same average position.
- **Caching:** prefix and disk caches were off or verified unused for every request (zero cache-hit tokens, hit counters unchanged); speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not pure GPU prefill time; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. For every application, each task's output was byte-identical in all four rounds.
- **Output counts:** recounting the full output with the target tokenizer gives exactly 256 for 95 requests. Splash's Coding 3 gives 255 in all four rounds: it generated 256 tokens, two adjacent ones of which merge into one when the text is re-encoded; the output itself is complete.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. For example, TensorFold 1.0.5 is not included because its native engine has not yet qualified Qwen3.6, which says nothing about its performance on other models. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page tests a single long-input size of about 30K tokens; for short-prompt single-request and concurrency results, see the series links at the top of this page.
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications, and every scored request generated 256 tokens.
- A warm machine. Every session started from nominal thermal pressure; during the sessions, thermal pressure rose for IronMLX in its last task and stayed nominal for the other applications. Sustained operation on a hot machine is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a multi-turn conversation reuses existing context, time to first token can be very different.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 241 ms for IronMLX, 188 ms for TensorFold 0.6.6, 144 ms for oMLX, and 59 ms for Splash; decode speed varied by up to 11.1% for IronMLX (Knowledge 3: 95.9 to 106.6 tok/s), 5.1% for Splash, 4.2% for TensorFold 0.6.6, and 1.3% for oMLX.
