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

- **Second in time to first token, close behind Splash.** The median time to first token is 35.84 s, first in round 1 and second in rounds 2–4; Splash takes 35.41 s, 1.2% shorter, and the four-round ranges of the two overlap. oMLX takes 41.28 s and TensorFold 52.00 s. With a 30K input, most of the time to first token is the prefill that reads the long input.
- **Second in decode speed, nearly tied with Splash.** The median decode speed is 66.5 tok/s, first in round 1 and second in rounds 2–4, only 0.1% below Splash (66.6); the four-round ranges of the two overlap. oMLX reaches 60.1 and TensorFold 57.8.
- **Second in completion time, close behind Splash.** The median time from sending a request to its completion is 41.05 s, against 40.59 s for Splash, 1.1% longer; IronMLX led in round 1 and Splash in rounds 2–4, and their ranges overlap. oMLX takes 46.13 s and TensorFold 56.57 s.
- **Lowest memory.** Peak memory while serving is 20.6 GiB, the lowest in all four rounds; oMLX uses 25.1 GiB, Splash 25.9 GiB, and TensorFold 27.9 GiB.

### Decode speed and time to first token

Decode speed is the number of tokens a request generates per second after its output starts, which shows how fast long replies are generated; time to first token (TTFT) is the time from sending a request to receiving its first output, which with long prompts mostly reflects the time to read the long input. Lower right is better in the chart.

<!-- benchmark:scatter qwen38-dense-b1-30k -->

### Comparison by metric

The table below lists the results of each application for each metric. Values are medians of four rounds, with the minimum and maximum of the four rounds in parentheses; each round's value is the median of that round's six tasks. Higher decode speed is better, and lower is better for the time and memory metrics; the best value in each row is in bold.

| Metric | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Time to first token (s) | 35.84 (35.82–⁠36.39) | 41.28 (41.08–⁠41.32) | **35.41** (35.27–⁠36.20) | 52.00 (51.93–⁠52.36) |
| Decode speed (tok/s, higher is better) | 66.5 (66.1–⁠66.7) | 60.1 (58.9–⁠60.7) | **66.6** (66.4–⁠67.1) | 57.8 (57.4–⁠58.1) |
| Completion time (s) | 41.05 (41.00–⁠41.38) | 46.13 (46.01–⁠46.33) | **40.59** (40.41–⁠41.29) | 56.57 (56.54–⁠57.13) |
| Peak memory while serving (GiB) | **20.6** (20.5–⁠21.0) | 25.1 (25.0–⁠25.1) | 25.9 (25.9–⁠25.9) | 27.9 (27.9–⁠27.9) |

The order was identical in all four rounds for memory (IronMLX, oMLX, Splash, TensorFold). For time to first token, decode speed, and completion time, IronMLX led in round 1 and Splash in rounds 2–4, followed by oMLX and TensorFold in every round.

> [!NOTE]
> Peak memory while serving is measured during the six scored requests and includes the KV cache of the 30K inputs. Peaks while loading the model and warming up differ: about 28.2 GiB for IronMLX, higher than its serving peak; about 21.9 GiB for oMLX, 19.3 GiB for Splash, and 27.8 GiB for TensorFold. This page does not compare memory during loading.

### Results by task

The six tasks have the same input length (30,732 tokens), but each application's draft acceptance varies with the content, so the tasks run at different speeds. The tables below show the median of four rounds for each task, with the best value in each row in bold.

Time to first token (s, lower is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Coding 1 | **30.62** | 35.12 | 31.09 | 40.99 |
| Coding 2 | **32.72** | 37.61 | 33.46 | 46.66 |
| Coding 3 | **33.50** | 39.31 | 33.59 | 50.49 |
| Knowledge 1 | 38.22 | 43.21 | **37.36** | 53.55 |
| Knowledge 2 | 40.62 | 44.18 | **39.89** | 53.69 |
| Knowledge 3 | 40.02 | 44.44 | **39.36** | 53.66 |

Completion time (s, lower is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Coding 1 | **32.98** | 38.35 | 33.74 | 43.90 |
| Coding 2 | **34.84** | 40.56 | 35.71 | 49.34 |
| Coding 3 | **36.66** | 42.90 | 36.78 | 54.25 |
| Knowledge 1 | 45.61 | 51.96 | **45.51** | 62.24 |
| Knowledge 2 | 45.47 | 49.39 | **44.67** | 58.96 |
| Knowledge 3 | **46.24**¹ | 52.01 | 46.43 | 62.04 |

Decode speed (tok/s, higher is better):

| Task | IronMLX 0.2.0 | oMLX 0.7.3rc1 | Splash 1.5.0 | TensorFold 1.2.0 |
|---|---|---|---|---|
| Coding 1 | **108.5** | 78.8 | 96.4 | 87.6 |
| Coding 2 | **120.4** | 86.5 | 112.9 | 94.4 |
| Coding 3 | **80.4** | 71.2 | 80.0 | 67.7 |
| Knowledge 1 | **34.5** | 29.2 | 31.4 | 29.3 |
| Knowledge 2 | 52.5 | 49.0 | **53.4** | 47.8 |
| Knowledge 3 | **37.8** | 33.7 | 36.1 | 30.6 |

> [!NOTE]
> ¹ On Knowledge 3, IronMLX ended naturally at token 235 (the model emitted its end-of-sequence token) in all four rounds, while every other application generated the full 256 tokens, so this task's completion time favors IronMLX. Knowledge 3 is already IronMLX's slowest task (its second slowest in round 4), so it does not move the median of the six tasks, and the overall completion time in the table above is unaffected.

IronMLX has the shortest time to first token on the three coding tasks and Splash on the three knowledge tasks, 0.1–0.9 s apart on each task. IronMLX decodes fastest on five tasks and Splash on Knowledge 2. IronMLX is nevertheless second in overall decode speed: each round's value is the median of the six tasks, which for both applications is the mean of Coding 3 and Knowledge 2, where IronMLX is 0.5 tok/s faster and 0.8 tok/s slower respectively. Every application's time to first token broadly grows from Coding 1 to Knowledge 3 (IronMLX: from 30.6 s on Coding 1 to 40.6 s on Knowledge 2; IronMLX's and Splash's Knowledge 3 are slightly shorter than their Knowledge 2, and TensorFold's three knowledge tasks are all 53.6–53.7 s); the six inputs are equally long, and thermal pressure rose over the same stretch of each session, as described under thermal control in [Hardware and software](#hardware-and-software).

Every application uses the external DFlash2 draft model for speculative decoding when serving a single request, and every scored request had drafts accepted. According to each server's own logs and status pages:

- **IronMLX 0.2.0:** each verification window uses a 15-node draft tree, verifies 16 rows at once, and has a fixed draft budget of 7; every window was a tree window. The prefill of the 30K inputs hands part of the MLP computation to the ANE under the default ANE `auto` (the status page shows the ANE as active).
- **oMLX 0.7.3rc1:** the log records the speculative decoding acceptance for every request.
- **Splash 1.5.0:** the drafted and accepted counters on its status page grow with every request; by default, the prefill of long inputs hands part of the FFN computation to the Neural Engine (recorded in its log).
- **TensorFold 1.2.0:** native engine (`qwen27`); the minimum number of rows a round that the server reports for each request is 6–12 depending on the task, the same for each task in all four rounds.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | October 31, 2026 (15:29–17:22 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row. Thermal pressure was sampled every second during each session. The prefill of the 30K inputs heats the Mac noticeably: the scored requests are sent in the fixed order Coding 1 to Knowledge 3, and every application stayed nominal for the first one or two of them; pressure then rose to a high level from Coding 3 on for IronMLX and Splash and from Coding 2 on for TensorFold, while oMLX read moderate at Coding 2 and high from Coding 3 on |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 32768 --kv-quant none` (block size 8 is the same as leaving it unset, which takes the draft model's own block size of 8); default M5 profile (ANE `auto`, active for 30K inputs), no tuning switches |
| oMLX 0.7.3rc1 | Official app (`oMLX-0.7.3rc1-macos26-27.dmg`) | `--no-cache --max-concurrent-requests 4`; context 32,768 and DFlash2 draft quantized to 4 bits (group 64) in its settings |
| Splash 1.5.0 | Official bottle | `--max-context 32768`; KV cache kept at its default int8; Neural Engine left on by default; the default 32 GiB non-persistent disk KV cache stays on (every session is a fresh process, and the cache hit counters were verified unchanged for every request); the draft model is always quantized to 4 bits at load (built-in Splash behavior, no setting) |
| TensorFold 1.2.0 | Official native release (sha256 `81fffaab…`), tag `v1.2.0` → `a388876` | `--drafter-bits 4 --parallel 4 --context 32768`, prompt cache and snapshots off, default keep-warm |

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
- **Sessions:** each session starts a fresh server, sends two short-prompt warm-up requests, then the six 30K scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. A rule fixed before the test ran two rounds first (orders ABCE and ECBA) and added two more rounds (orders BAEC and CEAB) right after if, in any metric, two adjacent applications swapped order between the rounds or their two-round ranges overlapped. IronMLX and Splash swapped order in time to first token, decode speed, and completion time over the first two rounds, so the rounds were added: four rounds × four applications = 16 sessions, with the same average position for each application.
- **Caching:** prefix and disk caches were off or verified unused for every request (zero cache-hit tokens, hit counters unchanged); speculative decoding was verified active from each server's counters or logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not pure GPU prefill time; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the four rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them).
- **Validity:** 16 of 16 sessions and 96 of 96 scored requests passed every check; no session was retried or dropped. For IronMLX, Splash, and TensorFold, each task's output was byte-identical in all four rounds; oMLX's outputs were not identical across the four rounds on Knowledge 1 and Knowledge 3.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens (adjacent tokens may be split differently when the text is re-encoded). In this run, TensorFold's Knowledge 3 recounts to 257 and Splash's Knowledge 1 to 255 in all four rounds, and two of oMLX's four Knowledge 1 outputs recount to 255, all with a server count of 256; IronMLX's Knowledge 3 has a server count of 235, which includes the end-of-sequence token, and an output text of 234 tokens.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this comparison or need to be kept in mind:

- **Not an overall verdict on any application.** These results reflect the listed versions on this model, this workload, and this Mac only; they do not show that any application is better or worse overall. Apart from the common settings listed in Method, each application ran its default strategy without tuning for this test; with other models, other workloads, or each application's recommended settings, the results and their order may differ. If you think an application's configuration is unfair, please let us know through [GitHub Issues](https://github.com/apepkuss/ironmlx/issues).
- **Small gaps to Splash in time to first token, decode speed, and completion time.** IronMLX trails Splash by 1.2% in time to first token, 0.1% in decode speed, and 1.1% in completion time; Splash led in three of the four rounds on each, but the four-round ranges overlap, so none of these gaps should be read as a clear advantage or disadvantage.
- **The ANE changes the output.** By default IronMLX uses the ANE in the prefill of 30K inputs, which differs slightly in numerics from the GPU alone, and greedy outputs differ accordingly.
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page covers one long input of about 30K tokens only; for concurrent requests, see [Qwen3.8 27B B4 30K-prompt concurrency performance comparison](qwen38-dense-b4-30k.md).
- Tail latency or statistical significance: four rounds on one machine are not enough for either.
- Answer quality. This test compares speed and memory only and does not evaluate answer quality. Outputs are not identical across applications.
- A hot machine. Every session started from nominal thermal pressure, but the 30K inputs made most scored requests of every application run under high thermal pressure; sustained high load over long periods is not covered.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a multi-turn conversation reuses existing context, time to first token can be very different.
- Run-to-run variation. Across the four rounds of the same task, time to first token varied by up to 1,345 ms for TensorFold, 1,277 ms for Splash, 901 ms for IronMLX, and 312 ms for oMLX; decode speed varied by up to 12.8% for oMLX, 4.7% for Splash, 4.0% for TensorFold, and 2.5% for IronMLX.
