# IronMLX long-context performance (128K and 256K prompts)

[简体中文](../zh-CN/benchmarks/ironmlx-long-context.md)

> [!NOTE]
> This page measures IronMLX alone and does not compare it with other applications. For comparisons with 30K-token prompts, see [Qwen3.8 27B B1 30K-prompt performance comparison](qwen38-dense-b1-30k.md) and [Qwen3.6 35B A3B B1 30K-prompt performance comparison](qwen36-moe-b1-30k.md).

This page shows how IronMLX 0.2.0 performs when a single request carries a prompt of about 128K or 256K tokens, and how its time to first token, decode speed, and memory change as the prompt grows. It covers the Qwen3.8-27B dense model and the Qwen3.6-35B-A3B mixture-of-experts (MoE) model, with one request at a time and an output limit of 256 tokens. The 128K requests have 131,084 input tokens. The 256K requests have 261,132 input tokens, which with the 256 output tokens comes close to the 262,144-token maximum context of both models. This workload corresponds to handing a very long document, a large part of a codebase, or a very long conversation to the model in one go. For the test environment setup, test method, and other details, see [Method](#method).

## Summary

<!-- benchmark:scaling ironmlx-long-context -->

- **Both lengths run to completion on both models.** All 48 scored requests at 128K and 256K completed with the full input and 256 output tokens. Peak memory while serving stayed at or below 35.3 GiB on a 128 GB Mac.
- **Time to first token grows faster than the input.** At 128K the dense model takes 272.15 s to the first token and the MoE model 68.93 s; at 256K it is 802.34 s (about 13.4 minutes) and 243.94 s. Doubling the input from 128K to 256K makes the wait 2.95 times as long on the dense model and 3.54 times as long on the MoE model.
- **Decode slows as the context grows.** The dense model decodes at 42.0 tok/s at 128K and 28.9 tok/s at 256K, against 66.5 tok/s at 30K; the MoE model at 143.2 and 100.4 tok/s, against 189.3 tok/s at 30K. Speed varies a lot by task: on the dense model the coding tasks stay at 35–65 tok/s, while the knowledge tasks run at 15–30 tok/s.
- **Memory grows moderately.** From 30K to 256K, peak memory rises from 20.6 to 35.3 GiB on the dense model and from 22.0 to 26.5 GiB on the MoE model.
- **Stable and reproducible.** Each task's output was byte-identical in both rounds, and the two round medians differ by at most 1.8% in any metric.

### Results by metric

The table below lists each model's results at each input length. Values are medians of the rounds, with the minimum and maximum in parentheses; each round's value is the median of that round's six tasks. The 30K rows are IronMLX's results from the run behind the 30K comparison pages (same build, four rounds, 32,768-token context) and were not measured in this test. Prefill speed is the prompt processing speed that the server reports for each request in its log.

| Model | Input | Time to first token (s) | Prefill speed (tok/s) | Decode speed (tok/s) | Completion time (s) | Peak memory while serving (GiB) |
|---|---|---|---|---|---|---|
| Qwen3.8 27B | 30K | 35.84 (35.82–⁠36.39) | 862 (850–⁠862) | 66.5 (66.1–⁠66.7) | 41.05 (41.00–⁠41.38) | 20.6 (20.5–⁠21.0) |
| Qwen3.8 27B | 128K | 272.15 (271.25–⁠273.05) | 482 (480–⁠483) | 42.0 (41.6–⁠42.3) | 279.85 (279.13–⁠280.58) | 26.8 (26.8–⁠26.8) |
| Qwen3.8 27B | 256K | 802.34 (801.24–⁠803.45) | 326 (325–⁠326) | 28.9 (28.9–⁠28.9) | 814.30 (810.88–⁠817.72) | 35.3 (35.2–⁠35.3) |
| Qwen3.6 35B A3B | 30K | 6.47 (6.46–⁠6.53) | 4,776 (4,730–⁠4,784) | 189.3 (188.9–⁠189.5) | 7.89 (7.86–⁠7.96) | 22.0 (21.9–⁠22.0) |
| Qwen3.6 35B A3B | 128K | 68.93 (68.85–⁠69.01) | 1,904 (1,902–⁠1,907) | 143.2 (142.8–⁠143.6) | 70.82 (70.75–⁠70.89) | 23.9 (23.8–⁠23.9) |
| Qwen3.6 35B A3B | 256K | 243.94 (243.77–⁠244.11) | 1,071 (1,070–⁠1,072) | 100.4 (100.1–⁠100.7) | 247.16 (247.08–⁠247.23) | 26.5 (26.4–⁠26.6) |

With inputs this long, almost all of the completion time is the time to first token, which is mostly the prefill that reads the input; generating the 256 output tokens takes 1.3–16.4 s depending on the task.

### How time to first token grows

The table below compares each length with 30K.

| Model | Input | Input vs. 30K | Time to first token vs. 30K |
|---|---|---|---|
| Qwen3.8 27B | 30K | 1× | 1× |
| Qwen3.8 27B | 128K | 4.3× | 7.6× |
| Qwen3.8 27B | 256K | 8.5× | 22.4× |
| Qwen3.6 35B A3B | 30K | 1× | 1× |
| Qwen3.6 35B A3B | 128K | 4.3× | 10.7× |
| Qwen3.6 35B A3B | 256K | 8.5× | 37.7× |

The prefill gets slower per token as the input grows, as the prefill speed in [Results by metric](#results-by-metric) shows. Both models mix linear-attention layers with full-attention layers (16 of 64 layers on the dense model, 10 of 40 on the MoE model), and in a full-attention layer each new token attends to every token before it, so the work grows faster than the input; the longer the input, the larger this share. The MoE model's other layers are light (about 3B active parameters per token), which is likely why this share dominates sooner there and its time to first token grows faster relative to 30K than the dense model's. Two conditions also differ from the 30K runs: thermal pressure was high during almost every request on this page (see [Hardware and software](#hardware-and-software)), and on the dense model the ANE takes part in the prefill only for the first 65,536 positions of the input (see [Applications and configuration](#applications-and-configuration)), while at 30K it covers the whole input.

### Results by task

The six tasks at each length have the same input length and the same output length (256 tokens), but draft acceptance varies with the content, so the tasks decode at different speeds. The tables below show the median of two rounds for each task.

Time to first token (s):

| Task | Qwen3.8 27B 128K | Qwen3.8 27B 256K | Qwen3.6 35B A3B 128K | Qwen3.6 35B A3B 256K |
|---|---|---|---|---|
| Coding 1 | 239.38 | 794.26 | 56.67 | 212.67 |
| Coding 2 | 267.33 | 810.49 | 60.87 | 242.44 |
| Coding 3 | 273.64 | 802.54 | 68.02 | 246.46 |
| Knowledge 1 | 272.81 | 802.52 | 69.84 | 245.46 |
| Knowledge 2 | 272.84 | 801.71 | 70.07 | 244.21 |
| Knowledge 3 | 274.91 | 801.54 | 70.48 | 243.42 |

Completion time (s):

| Task | Qwen3.8 27B 128K | Qwen3.8 27B 256K | Qwen3.6 35B A3B 128K | Qwen3.6 35B A3B 256K |
|---|---|---|---|---|
| Coding 1 | 243.30 | 800.45 | 57.92 | 214.62 |
| Coding 2 | 271.28 | 815.43 | 62.37 | 244.72 |
| Coding 3 | 278.37 | 809.78 | 69.59 | 248.68 |
| Knowledge 1 | 283.79 | 818.95 | 72.06 | 248.32 |
| Knowledge 2 | 281.34 | 813.02 | 72.12 | 247.11 |
| Knowledge 3 | 286.06 | 815.12 | 73.34 | 247.28 |

Decode speed (tok/s, higher is better), with the average number of output tokens per verification window in parentheses:

| Task | Qwen3.8 27B 128K | Qwen3.8 27B 256K | Qwen3.6 35B A3B 128K | Qwen3.6 35B A3B 256K |
|---|---|---|---|---|
| Coding 1 | 65.1 (6.6) | 41.2 (6.6) | 205.2 (3.7) | 131.0 (3.6) |
| Coding 2 | 64.5 (6.7) | 51.6 (8.0) | 169.8 (3.1) | 111.5 (3.1) |
| Coding 3 | 53.9 (5.7) | 35.2 (5.4) | 162.5 (3.0) | 115.2 (3.2) |
| Knowledge 1 | 23.2 (2.5) | 15.5 (2.4) | 115.2 (2.1) | 89.2 (2.3) |
| Knowledge 2 | 30.0 (3.3) | 22.6 (3.5) | 123.9 (2.3) | 88.2 (2.3) |
| Knowledge 3 | 22.9 (2.5) | 18.8 (2.9) | 89.1 (1.6) | 66.0 (1.6) |

- **Decode speed follows draft acceptance.** Each verification window runs the target model once over the whole context, and its cost grows with the context. Tasks whose drafts are accepted well produce more tokens per window and decode faster. On the dense model the coding tasks produce 5.4–8.0 tokens per window, and the knowledge tasks only 2.4–3.5, so they decode at less than half the speed.
- **Coding 1 has a shorter time to first token.** All six prompts at a length have the same number of tokens. Even so, Coding 1, the first scored request of every session, reaches its first token 12–20% sooner than Coding 3 through Knowledge 3 on the dense model at 128K and on the MoE model at both lengths. On the MoE model at 128K, Coding 1 is also the only request whose thermal pressure stayed nominal; on the dense model at 256K the pattern does not appear (Coding 1 took 818.7 s in round 1 and 769.8 s in round 2). One likely reason is that the first request runs before thermal throttling sets in fully, but the task order is fixed, so this cannot be fully separated from the content of the tasks.
- **Speculative decoding.** Every scored request had drafts accepted. On the dense model every verification window is a tree window with a draft budget of 7, as on the 30K page; on the MoE model every window uses linear draft verification, with the number of drafts per window set automatically by the default window-cost policy.

### Memory

Peak memory rises with the context mainly because of the KV cache of the full-attention layers: about 64 KiB per token of context on the dense model (16 layers, 4 KV heads of 256 dimensions, BF16) and about 20 KiB on the MoE model (10 layers, 2 KV heads). For the extra input tokens from 30K to 128K and from 128K to 256K, that comes to about 6.1 GiB and then 7.9 GiB on the dense model, and about 1.9 GiB and then 2.5 GiB on the MoE model. The measured increases, 6.2 and 8.5 GiB on the dense model and 1.9 and 2.6 GiB on the MoE model, are of the same size.

## Method

### Hardware and software

| Item | Configuration |
|---|---|
| Computer | MacBook Pro (Mac17,6), Apple M5 Max with a 40-core GPU, 128 GB unified memory |
| Operating system | macOS 26.4 (25E246) |
| Test date | November 1, 2026 (15:41–20:39 UTC) |
| Isolation | One inference server at a time; no other inference server or GPU compute job ran during the test. The benchmark client ran on the same Mac and connected over `127.0.0.1` |
| Thermal control | Before every session the Mac stayed idle for at least two minutes after the previous server stopped, and macOS thermal pressure had read nominal for the last 60 seconds in a row; the first session of each of the four runs was the exception, waiting for the thermal condition only (95–105 s). Thermal pressure was sampled every second during each session. The long prefills heat the Mac: thermal pressure rose to a high level during every scored request except Coding 1 on the MoE model at 128K, which stayed nominal in both rounds |

### Applications and configuration

| Application | Source | Configuration |
|---|---|---|
| IronMLX 0.2.0 | Release build (`cargo build --locked --release`) of tag `v0.2.0-rc.4` (commit `a0afebe55`); server and MLX Metal runtime frozen together. The 0.2.0 release ships this code unchanged | Started as the app starts it; `--dflash2-block-size 8 --dflash2-draft-bits 4 --max-sequences 4 --max-cache-cap 139264` at 128K or `--max-cache-cap 262144` at 256K, and `--kv-quant none`, the same as on the 30K pages except the context; default M5 profile, no tuning switches. On the dense model the ANE is on by default (`auto`, active) and takes part of the MLP computation in prefill chunks that start before position 65,536, as the default profile in the startup log states; the ANE does not apply to the MoE model |

The context was 139,264 tokens for the 128K requests and 262,144 tokens for the 256K requests, the `max_position_embeddings` of both models. The KV cache is BF16, and reusable cross-request prefix caching is off.

### Models

| Role | Model | Source | Revision | Format |
|---|---|---|---|---|
| Dense target model | `mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff) | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` (2026-08-14) | MLX safetensors, 4-bit affine quantization, group size 64; Apache-2.0 |
| Dense draft model | `z-lab/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (2026-08-19) | BF16 safetensors, 5 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |
| MoE target model | `mlx-community/Qwen3.6-35B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (2026-04-16) | MLX safetensors, mixture of experts, 4-bit affine quantization, group size 64 (8-bit expert router gates); Apache-2.0 |
| MoE draft model | `incoai/Qwen3.6-35B-A3B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/tree/51ef7b6923ad6c14cb1bb41c37a9041446496ab6) | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` (2026-09-25) | BF16 safetensors, 6 layers, block size 8; loaded with 4-bit draft quantization; Apache-2.0 |

> [!NOTE]
> These are the same model files as on the 30K pages; every weight, configuration, and tokenizer file matches the Hugging Face revision in the table above by SHA-256.

### Workload and procedure

- **Requests:** streaming `/v1/chat/completions`, one user message, temperature 0, thinking off, `max_tokens` 256, no `ignore_eos`. A scored request may end by length (exactly 256 tokens counted by the server) or naturally (200–256 tokens counted by the server); in this run every scored request ended by length with exactly 256 tokens counted by the server.
- **Prompts:** six prompts per length (three coding, three knowledge), built the same way as the 30K prompts: synthetic project material related to the question, followed by the question. The longer prompts extend the material with more records and keep the same question. The user text is exactly 131,072 tokens at 128K and 261,120 tokens at 256K with the target tokenizer, and 131,084 and 261,132 tokens with the chat template. The six prompts at a length all differ, and any two of them share only their first few tokens (template and heading).
- **Full input:** every request's server-side prompt token count was 131,084 or 261,132, matching the client's count with the target tokenizer; no input was truncated, compressed, or dropped.
- **Sessions:** each session starts a fresh server, sends two short warm-up requests, then the six scored requests one at a time (each sent 1 second after the previous one finished), then stops the server. Each model and length ran two rounds, one session each, in the order MoE 128K, MoE 256K, dense 128K, dense 256K. A rule fixed before the test would have added two more rounds if the two round medians of time to first token, decode speed, or completion time differed by more than 5%; the largest difference was 1.8% (decode speed on the dense model at 128K), so no rounds were added: 8 sessions in all.
- **Caching:** the prefix cache was off for every request (zero cached tokens reported by the server); speculative decoding was verified active from the server's logs.
- **Metrics:** time to first token runs from the send to the first non-empty content and includes transfer, queueing, and reading the full long input, so it is not the same as the GPU prefill time alone; decode speed is (output tokens − tokens in the first content) ÷ (time between the first and last content), counted on the client with the target tokenizer; completion runs from the send to `[DONE]`. Each round takes the median of the six tasks, and the table takes the median of the rounds. Memory is the peak process-tree physical footprint, sampled every 50 ms while the six scored requests ran (including the gaps between them). Prefill speed (the server's prompt tokens per second) and tokens per verification window come from the server's log, and prefill speed takes the same medians as the other metrics.
- **Validity:** 8 of 8 sessions and 48 of 48 scored requests passed every check; no session was retried or dropped.
- **Output counts:** the server's count is authoritative; the client recounts the full output with the target tokenizer as a cross-check and allows a difference of up to 3 tokens. In this run 46 of the 48 recounts were exactly 256; the other two (Knowledge 2 on the MoE model at 256K, both rounds) were 257, within the allowed difference.
- **Output reproducibility:** each task's output was byte-identical in both rounds at every length.

## Scope and limitations

The conclusions on this page apply only under the conditions described in [Method](#method). The following are not covered by this test or need to be kept in mind:

- **Not a comparison.** This page measures IronMLX only and says nothing about how other applications perform at these lengths.
- **The 30K rows come from other runs.** They are IronMLX's results on the 30K comparison pages, measured with the same build but in separate sessions, with four rounds, a 32,768-token context, less thermal pressure, and the ANE covering the whole dense-model prefill. The ratios to 30K describe the overall change and do not isolate the effect of input length alone.
- **A hot Mac.** Every session started from nominal thermal pressure, but almost every scored request ran under high thermal pressure, and the results include that slowdown. A Mac with different cooling, or a machine that has been under load for longer, may perform differently.
- Other Macs and memory sizes. Peak memory reached 35.3 GiB on the dense model at 256K; this page does not show whether Macs with less memory can serve these lengths.
- Other input lengths, other models, concurrent requests, or real multi-user traffic. This page covers single requests at about 128K and 256K tokens only.
- Tail latency or statistical significance: two rounds on one machine are not enough for either.
- Answer quality. This test measures speed and memory only and does not evaluate answer quality at these lengths.
- Reusing long context across turns. Every scored request is a fresh long input with no prefix cache; when a conversation reuses context it already has, the time to first token can be very different.
- Run-to-run variation. Across the two rounds of the same task, time to first token varied by up to 48.9 s on the dense model at 256K (Coding 1: 769.8 to 818.7 s), 7.9 s on the dense model at 128K, 2.9 s on the MoE model at 256K, and 0.3 s on the MoE model at 128K; decode speed varied by up to 7.0% on the dense model at 128K (Knowledge 3) and by 2.4% or less elsewhere.
