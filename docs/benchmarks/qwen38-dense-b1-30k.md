# Qwen3.8 27B B1 30K-prompt performance comparison

[简体中文](../zh-CN/benchmarks/qwen38-dense-b1-30k.md)

> [!NOTE]
> This page compares single requests with 30K-token prompts. For short prompts, see [Qwen3.8 27B B1 single request](qwen38-dense-b1-short.md) and [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-short.md). 30K-prompt performance comparison series:
>
> - Qwen3.8 27B B1 single request
> - [Qwen3.8 27B B4 concurrency](qwen38-dense-b4-30k.md)

The B1 single-request comparison benchmarks IronMLX against the leading inference applications on Apple Silicon today. It has two purposes: to help developers identify IronMLX's strengths and weaknesses and set the direction for further optimization, and to show users how IronMLX actually performs so they can judge whether it fits their use case.

This page covers the dense Qwen3.8-27B model with one request at a time. Every scored request uses a long prompt of about 30K tokens (30,732 input tokens per request) with an output limit of 256 tokens. This workload corresponds to handing the model a long document, part of a codebase, or a long conversation in one go, where most of the wait is the model reading the long input. The conclusions on this page apply to this workload only. The local inference applications in this comparison are oMLX, Splash, TensorFold, and IronMLX. See [Applications and configuration](#applications-and-configuration) for details on each. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

Based on all data collected in the 30K-prompt B1 single-request comparison, IronMLX's overall performance is as follows:

<!-- benchmark:kpis qwen38-dense-b1-30k -->

- **Shortest time to first token, by a small margin.** The median time to first token is 34.19 s, first in all four rounds; Splash takes 34.40 s, only 0.6% longer, and the four-round ranges of the two overlap. oMLX takes 39.70 s and TensorFold 49.90 s. With a 30K input, most of the time to first token is the prefill that reads the long input.
- **Second in decode speed.** The median decode speed is 68.6 tok/s, second in all four rounds and 4.3% below Splash (71.6); oMLX reaches 62.4 and TensorFold 59.9.
- **Tied with Splash for first in completion time.** The median time from sending a request to its completion is 39.06 s, against 39.02 s for Splash; each led in two of the four rounds, and their ranges overlap. oMLX takes 44.67 s and TensorFold 54.35 s.
- **Lowest memory.** Peak memory while serving is 20.6 GiB, the lowest in all four rounds; TensorFold uses 21.4 GiB, oMLX 25.1 GiB, and Splash 25.9 GiB.

### Decode speed and time to first token

Decode speed is the number of tokens a request generates per second after its output starts, which shows how fast long replies are generated; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects the time to read the long input. Lower right is better in the chart.

<!-- benchmark:scatter qwen38-dense-b1-30k -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0-rc.3 | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Time to first token (s) | **34.19** (33.99–⁠34.35) | 39.70 (39.67–⁠39.77) | 34.40 (34.28–⁠34.66) | 49.90 (49.73–⁠50.37) |
| Decode speed (tok/s, higher is better) | 68.6 (68.5–⁠68.7) | 62.4 (60.9–⁠62.9) | **71.6** (71.4–⁠72.0) | 59.9 (59.4–⁠60.0) |
| Completion time (s) | 39.06 (38.91–⁠39.28) | 44.67 (44.64–⁠44.81) | **39.02** (38.91–⁠39.03) | 54.35 (54.24–⁠54.85) |
| Peak memory while serving (GiB) | **20.6** (20.6–⁠20.7) | 25.1 (25.1–⁠25.1) | 25.9 (25.9–⁠25.9) | 21.4 (21.4–⁠21.4) |

The order was identical in all four rounds for time to first token (IronMLX, Splash, oMLX, TensorFold), decode speed (Splash, IronMLX, oMLX, TensorFold), and memory (IronMLX, TensorFold, oMLX, Splash). For completion time, IronMLX led in rounds 1 and 3 and Splash in rounds 2 and 4, followed by oMLX and TensorFold in every round.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests and includes the KV cache of the 30K inputs. Peaks while loading the model and warming up differ: about 28.2 GiB for IronMLX, higher than its serving peak; about 22.0 GiB for oMLX, 19.4 GiB for Splash, and 21.3 GiB for TensorFold. This page does not compare memory during loading.

### Results by task

The six tasks have the same input length (30,732 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of four rounds for each task, with the best value in each row in bold.

Time to first token (s, lower is better):

| Task | IronMLX 0.2.0-rc.3 | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Coding 1 | **30.54** | 35.11 | 31.01 | 40.83 |
| Coding 2 | **32.51** | 37.32 | 33.23 | 46.24 |
| Coding 3 | **32.54** | 38.14 | 32.88 | 48.50 |
| Knowledge 1 | 35.82 | 41.27 | **35.60** | 51.29 |
| Knowledge 2 | 38.32 | 42.76 | **38.00** | 51.45 |
| Knowledge 3 | 38.59 | 43.10 | **38.29** | 51.42 |

Completion time (s, lower is better):

| Task | IronMLX 0.2.0-rc.3 | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Coding 1 | **32.89** | 38.37 | 33.44 | 43.71 |
| Coding 2 | **34.63** | 40.27 | 35.49 | 48.98 |
| Coding 3 | **35.64** | 41.62 | 35.81 | 52.15 |
| Knowledge 1 | 42.60 | 49.47 | **42.44** | 59.81 |
| Knowledge 2 | 42.95 | 47.75 | **42.55** | 56.56 |
| Knowledge 3 | **44.48**¹ | 50.69 | 45.19 | 59.63 |

Decode speed (tok/s, higher is better):

| Task | IronMLX 0.2.0-rc.3 | oMLX 0.7.2 | Splash 1.4.1 | TensorFold 1.1.6 |
|---|---|---|---|---|
| Coding 1 | **108.6** | 78.1 | 104.6 | 88.7 |
| Coding 2 | **120.5** | 87.1 | 112.8 | 93.1 |
| Coding 3 | 82.2 | 73.4 | **87.4** | 69.8 |
| Knowledge 1 | 37.6 | 31.1 | **37.9** | 29.9 |
| Knowledge 2 | 55.0 | 51.4 | **56.0** | 50.0 |
| Knowledge 3 | **39.4** | 33.5 | 36.9 | 31.2 |

> [!NOTE]
> ¹ On Knowledge 3, IronMLX ended naturally at token 235 (the model emitted its end-of-sequence token) in all four rounds, while every other application generated the full 256 tokens, so this task's completion time favors IronMLX. Knowledge 3 is already IronMLX's slowest task, so it does not move the median of the six tasks, and the overall completion time in the table above is unaffected.

IronMLX has the shortest time to first token on the three coding tasks and Splash on the three knowledge tasks, 0.2–0.7 s apart on each task. IronMLX decodes fastest on Coding 1, Coding 2, and Knowledge 3, and Splash on the other three. Every application's time to first token broadly grows from Coding 1 to Knowledge 3 (IronMLX: 30.5 s to 38.6 s; Splash's Coding 3 is slightly shorter than its Coding 2, and TensorFold's Knowledge 3 about equals its Knowledge 2); the six inputs are equally long, and thermal pressure rose over the same stretch of each session, as described under thermal control in [Hardware and software](#hardware-and-software).

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX 0.2.0-rc.3:** each verification window uses a 15-node draft tree, verifies 16 rows at once, and has a fixed draft budget of 7; every window was a tree window. The prefill of the 30K inputs hands part of the MLP computation to the ANE under the default ANE `auto` (the status page shows the ANE as active).
- **oMLX 0.7.2:** the log records the speculative decoding acceptance for every request.
- **Splash 1.4.1:** the drafted and accepted counters on its status page grow with every request; by default, the prefill of long inputs hands part of the FFN computation to the Neural Engine (recorded in its log).
- **TensorFold 1.1.6:** native engine (`qwen27`); each round verifies at least 16 rows (the server reports a minimum of 16 rows a round for every request).

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 27, 2026 (17:23–19:13 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session. The prefill of the 30K inputs heats the Mac noticeably: the scored requests are sent in the fixed order Coding 1 to Knowledge 3, and every application stayed nominal for the first one or two of them; pressure then rose to a high level from Coding 3 on for IronMLX and Splash and from Coding 2 on for TensorFold, while oMLX read moderate at Coding 2 and high from Coding 3 on |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0-rc.3 | Release build (`cargo build --locked --release`) of commit `c6ef14ec2`; server and MLX Metal runtime frozen together. The commits between it and the `v0.2.0-rc.3` tag (`2d5186bb0`) only remove experimental switches that were never enabled and update documentation, so the measured code path is that of 0.2.0-rc.3 | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which takes the draft model's own block size of 8); default M5 profile (ANE `auto`, active for 30K inputs), no tuning switches |
| oMLX 0.7.2 | Official app (`oMLX-0.7.2-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.4.1 | Official bottle | `--max-context 32768`; KV cache kept at its default int8; Neural Engine left on by default; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.1.6 | Official native release (sha256 `0cd208ba…`), tag `v1.1.6` → `eea9dc9` | `--drafter-bits 4 --parallel 4 --context 32768`, prompt cache and snapshots off, default keep-warm |

Common settings for all applications: 4-bit draft model, 32,768-token context, capacity 4 where configurable (the benchmark client sends one request at a time), and reusable cross-request prefix caching off; BF16 KV cache for all applications except Splash, which keeps its default int8; each application otherwise keeps its own default strategy. Each server's effective context was verified as 32,768 from its own interface or settings after startup.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> All four applications load the same model files; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos` (oMLX does not support it). A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run only IronMLX's Knowledge 3 ended naturally (235 tokens), and every other request ended by length at 256.
- **Prompts:** six 30K prompts (three coding, three knowledge). Each consists of synthetic project material related to the question, followed by the question itself, and is 30,732 tokens with the chat template. The six prompts all differ, and any two requests share at most their first 8 tokens (template and title).
- **Complete input:** every request's server-side prompt token count was 30,732, matching the client's count with the target tokenizer; no input was truncated, compressed, or dropped. All applications received exactly the same text.
- **Sessions:** each session starts a fresh server, sends two short-prompt warm-up requests, then the six 30K scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCE and ECBA) and added two more rounds (orders BAEC and CEAB) right after if, in any metric, two adjacent applications swapped order between the rounds or their two-round ranges overlapped. IronMLX and Splash swapped order in completion time over the first two rounds, so the rounds were added: four rounds × four applications = 16 sessions, with the same average position for each application.
- **Caching:** prefix and disk caches were off or verified unused for every request (zero cache-hit tokens, hit counters unchanged); speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not pure GPU prefill time; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. For IronMLX, Splash, and TensorFold, each task's output was byte-identical in all four rounds; oMLX's outputs were not identical across the four rounds on Knowledge 1 and Knowledge 3.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run, TensorFold's Knowledge 3 recounts to 257 and one of oMLX's Knowledge 1 outputs to 255, both with a server count of 256; IronMLX's Knowledge 3 has a server count of 235, which includes the end-of-sequence token, and an output text of 234 tokens.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **IronMLX is a release candidate.** 0.2.0-rc.3 is a pre-release, not the final 0.2.0 release; the final release may perform differently.
- **Small gaps in time to first token and completion time.** IronMLX and Splash differ by only 0.6% in time to first token and 0.1% in completion time, where each led in two of the four rounds; neither should be read as a clear advantage or disadvantage.
- **The ANE changes the output.** By default IronMLX uses the ANE in the prefill of 30K inputs, which differs slightly in numerics from the GPU alone, and greedy outputs differ accordingly. An offline cross-check (not scored): with the ANE off (`--ane off`), the same build generated the full 256 tokens on Knowledge 3, with a different text than under the default setting.
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page covers one long input of about 30K tokens only; for concurrent requests, see [Qwen3.8 27B B4 30K-prompt concurrency performance comparison](qwen38-dense-b4-30k.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications.
- A hot machine. Every session started from nominal thermal pressure, but the 30K inputs made most scored requests of every application run under high thermal pressure; sustained high load over long periods is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a multi-turn conversation reuses existing context, time to first token can be very different.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 954 ms for TensorFold, 849 ms for IronMLX, 715 ms for Splash, and 197 ms for oMLX; decode speed varied by up to 5.8% for oMLX, 3.6% for Splash, 2.4% for IronMLX, and 1.7% for TensorFold.
