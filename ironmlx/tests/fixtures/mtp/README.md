# P3b4 MTP fixture

Tiny synthetic fixture for verifying numerical correctness of `nn::Mtp::forward`
against an independent Python reference built only from `mlx.core` primitives
(no `mlx-lm` patch dependency, no quantization, no MoE).

## Config

- B=1, S=4, hidden_size=32
- Hq=4, Hkv=2, head_dim=8, intermediate_size=64
- num_mtp_layers=1, rms_norm_eps=1e-6
- partial_rotary_factor=1.0 (rot_dim=8, half=4)
- mrope sections=[2, 1, 1]
- attention_bias=false

## Files

| File | Shape | dtype | Notes |
|---|---|---|---|
| `input_hidden.npy` | `[1, 4, 32]` | bf16 | post-norm hidden state from main model (synthetic) |
| `input_next_embeds.npy` | `[1, 4, 32]` | bf16 | embedding of (synthetic) next-token ids |
| `input_position_ids.npy` | `[3, 1, 4]` | i32 | mrope 3-stream position ids |
| `input_inv_freq.npy` | `[4]` | fp32 | precomputed by Mrope::new |
| `pre_fc_norm_hidden_weight.npy` | `[32]` | fp32 | RmsNorm weight |
| `pre_fc_norm_embedding_weight.npy` | `[32]` | fp32 | RmsNorm weight |
| `fc_weight.npy` | `[32, 64]` | bf16 | Linear 2H -> H, no bias |
| `layer0_input_layernorm_weight.npy` | `[32]` | fp32 | DecoderLayer.input_layernorm |
| `layer0_q_proj_weight.npy` | `[64, 32]` | bf16 | Hq*D*2 (queries + gate) |
| `layer0_k_proj_weight.npy` | `[16, 32]` | bf16 | Hkv*D |
| `layer0_v_proj_weight.npy` | `[16, 32]` | bf16 | Hkv*D |
| `layer0_o_proj_weight.npy` | `[32, 32]` | bf16 | hidden_size <- Hq*D |
| `layer0_q_norm_weight.npy` | `[8]` | fp32 | per-head dim |
| `layer0_k_norm_weight.npy` | `[8]` | fp32 | per-head dim |
| `layer0_post_attention_layernorm_weight.npy` | `[32]` | fp32 | DecoderLayer.post_attention_layernorm |
| `layer0_mlp_gate_proj_weight.npy` | `[64, 32]` | bf16 | SwiGLU gate |
| `layer0_mlp_up_proj_weight.npy` | `[64, 32]` | bf16 | SwiGLU up |
| `layer0_mlp_down_proj_weight.npy` | `[32, 64]` | bf16 | SwiGLU down |
| `norm_weight.npy` | `[32]` | fp32 | mtp.norm |
| `expected_cos.npy` | `[1, 4, 8]` | fp32 | Mrope::cos_sin output |
| `expected_sin.npy` | `[1, 4, 8]` | fp32 | Mrope::cos_sin output |
| `expected_mtp_out.npy` | `[1, 4, 32]` | fp32 | post-`mtp.norm` hidden state |

`expected_mtp_out` ends up at fp32 because all RmsNorm weights in this fixture are
fp32. Mixed-precision matmul with bf16 attn / mlp weights upgrades intermediates
to fp32 by the time they hit a fp32 RmsNorm — the final `mtp.norm` outputs fp32.

## Regenerate

```text
cd ironmlx/tests/fixtures/mtp && python gen_fixture.py
```

Pinned MLX version: `0.31.1` (script will refuse to run on a different version).
