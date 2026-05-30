# GLM-4.7-Flash (`glm4_moe_lite`) 接入设计

> 状态：设计稿，待 Boss 复审。
> 范围：在 ironmlx（Rust/MLX 推理引擎）中接入 `model_type=glm4_moe_lite`、架构 `Glm4MoeLiteForCausalLM` 的 GLM-4.7-Flash（首个验证目标 = `mlx-community/GLM-4.7-Flash-4bit`，4-bit affine group_size=64）。
> Worktree：`ironmlx-glm47-flash`（基于 `codex/gemma-4-vision` @ `2a40194`）。

## 来源与方法学说明（binding）

本设计的规格（spec）一律来自一手来源：**DeepSeek-V2 论文**（arXiv:2405.04434 §2.1）、**DeepSeek-V3 论文**（arXiv:2412.19437 §2.1.2）、**GLM-4.7-Flash 的 config.json**、**实测 checkpoint 张量形状**、以及 **MLX C++/Metal API 契约**（读自 `/Users/xin/workspace/iron-rivals/mlx`）。竞品实现（HF `modeling_glm4_moe_lite.py`、mlx-lm `glm4_moe_lite.py`、MLX-VLM）仅作 **observation / triangulation**，文中以 **[OBS]** 标注；竞品不是 specification。

本设计的数值规格经过一个多 agent workflow 的"推导 → 对抗验证 → 综合"流程产出，3 个对抗验证（MLA / 路由 / RoPE）全部应用。

---

## 0. 结论摘要

- **架构定性：MoE**。DeepSeek 风格细粒度 MoE：absorbed-MLA 注意力 + noaux_tc（sigmoid + bias）路由 + ungated 共享专家 + 首层 Dense + MTP（本 checkpoint 已剔除）。47 层中第 0 层 Dense，第 1–46 层 MoE。
- **目标：一步到位** = 正确性对齐 mlx-vlm + 性能**硬验收线：追平/超过 omlx**。
- **方案：Approach 1** = Rust MLX 算子组合管线；自研 MLA Metal kernel 仅作"达不到才上"的有条件项，须先过 feasibility gate（见 §9）。
- **Boss 已决策**：(a) 新建独立 `Glm4MoeBlock`（不参数化 Qwen 的 `SparseMoeBlock`）；(b) RoPE 用 glm-本地封装（不提升共享 FastRope）；(c) KV cache 用新建独立 `MlaLatentCache` 类型。

---

## 1. 模型事实（authoritative）

### 1.1 config.json 关键字段

| 字段 | 值 | | 字段 | 值 |
|---|---|---|---|---|
| hidden_size | 2048 | | n_routed_experts | 64 |
| num_hidden_layers | 47 | | num_experts_per_tok | 4 |
| first_k_dense_replace | 1 | | n_shared_experts | 1 |
| num_attention_heads | 20 | | moe_intermediate_size | 1536 |
| num_key_value_heads | 20 | | intermediate_size（dense层0） | 10240 |
| q_lora_rank | 768 | | norm_topk_prob | true |
| kv_lora_rank | 512 | | routed_scaling_factor | 1.8 |
| qk_nope_head_dim | 192 | | topk_method | noaux_tc |
| qk_rope_head_dim | 64 | | n_group / topk_group | 1 / 1 |
| v_head_dim | 256 | | rope_theta | 1000000 |
| vocab_size | 154880 | | partial_rotary_factor | 1.0 |
| rms_norm_eps | 1e-5 | | max_position_embeddings | 202752 |
| tie_word_embeddings | false | | hidden_act | silu |
| eos_token_id | [154820, 154827, 154829] | | num_nextn_predict_layers | 1 |
| quantization | 4-bit affine, group_size=64 | | rope_scaling | null |

### 1.2 checkpoint 张量形状（4-bit 打包：`.weight` 末维 = in_features/8；逐头堆叠张量带 leading head 轴）

```
self_attn.q_a_proj.weight           [768, 256]       hidden2048 -> q_lora768
self_attn.q_a_layernorm.weight      [768]
self_attn.q_b_proj.weight           [5120, 96]       768 -> 5120 = 20*256 (256=192 nope + 64 rope)
self_attn.kv_a_proj_with_mqa.weight [576, 256]       hidden2048 -> 576 = 512 c_kv + 64 k_pe
self_attn.kv_a_layernorm.weight     [512]
self_attn.embed_q.weight            [20, 512, 24]    per head 192->512  = 吸收的 W^UK^T
self_attn.unembed_out.weight        [20, 256, 64]    per head 512->256  = 吸收的 W^UV
self_attn.o_proj.weight             [2048, 640]      5120 -> hidden2048
layer0.mlp(dense): gate_proj[10240,256] up_proj[10240,256] down_proj[2048,1280]
layer≥1.mlp(moe):  gate.weight[64,2048] (PLAIN 非量化)  gate.e_score_correction_bias[64]
                   switch_mlp.gate_proj[64,1536,256] up_proj[64,1536,256] down_proj[64,2048,192]
                   shared_experts.{gate,up,down}_proj  (无 shared_expert_gate 张量 -> ungated)
model.embed_tokens.weight [154880,256]   lm_head.weight [154880,256]   model.norm.weight [2048]
checkpoint 中无 mtp/nextn 张量（转换时已剔除）
```

`self_attn` **无 `kv_b_proj`**，却有 `embed_q` + `unembed_out` —— 证明 checkpoint 以 **absorbed（矩阵吸收）形态**存 MLA（DeepSeek-V2 §2.1.3）。

吸收恒等式（精确、非近似）：
- `embed_q[h]`（192→512）= W^UK^T：`q_nope_latent = q_nope @ embed_q[h]`，使 `q_nope·(W^UK c_kv) = (W^UK^T q_nope)·c_kv = q_nope_latent·c_kv`。
- `unembed_out[h]`（512→256）= W^UV：`(softmax)@(W^UV c_kv) = ((softmax)@c_kv)@W^UV`。

---

## 2. 模块结构

沿用 per-model 模块惯例（`qwen3_5` / `qwen3_5_moe` / `qwen3_6_moe` / `gemma4`），新增独立模块，**不写兼容代码**：

```
ironmlx/src/models/glm4_moe_lite/
  mod.rs            导出
  config.rs         Glm4MoeLiteConfig（顶层扁平 config，无 text_config 嵌套）
  model.rs          Glm4MoeLiteModel（impl trait Model + DenseVlMethods 文本桩）
  decoder_layer.rs  Glm4DecoderLayer（layer0=Dense；layer1..46=MoE；MLA 注意力）
  mla_attention.rs  MlaAttention（两形态解耦前向）
  mla_cache.rs      MlaLatentCache（latent MQA cache）
  moe.rs            Glm4MoeBlock（sigmoid+noaux_tc 路由 + ungated 共享 + 复用 RoutedExperts）
  rope.rs           Glm4Rope（mlx::fast::rope_on interleaved 薄封装）
```

复用现有件：`nn::RmsNorm`、`nn::Mlp`（layer0 dense）、`qwen3_5_moe::sparse_moe::RoutedExperts`（专家 gather_qmm）、`mlx::fast::rope_on`、`core::cache` 框架、`core::loader`、`EosTokenId::Multi`。

---

## 3. `MlaLatentCache`（新类型）

- **内容**：每 token 存 **归一化后的 `c_kv[512]` + rope 后的 `k_pe[64]`**，单 kv 头 → 576 floats/token（vs MHA 的 20×256×2）。
- **公共契约**（镜像 `KVCache`，见 `core/cache/kv_cache.rs`）：`new / with_step / update_and_fetch_on / offsets / cap / reset / grow_cap / adopt_row_from`。
- **接入**：`Model::make_cache`（`core/model.rs:13`）返回 `Vec<LayerCache>`，故**必须**给共享枚举 `LayerCache`（`nn/decoder_layer.rs:65`）加 `Mla(MlaLatentCache)` 变体。`AttnPath` 是 Qwen 逐层 Full/Linear 混合派发枚举，**GLM 无需扩**——GLM 47 层注意力同质，`Glm4DecoderLayer` 直接持有 `MlaAttention`，其 `forward_on` 对 `&mut LayerCache` 做 `match LayerCache::Mla(c) => …`。GLM 用独立 decoder layer，不混进 Qwen 的 `DecoderLayerMoe`。
- **caveats**：
  - 缓存的是 **归一化后** c_kv（`kv_a_layernorm` 增益已折进），读时**不可**再归一。
  - 缓存的是 **rope 后** k_pe（相对偏移一次性烘焙）。
  - KV-floor 记忆是关于 cache **容量/步长**（`cap`/`step` < 256 触发 cache-update Metal 慢路径），**不是** head-dim 宽度。`MlaLatentCache` 须沿用 `KVCache` 的 `step=256` 默认（`with_step(cap)` 一次性预分配）来规避该 cliff；c_kv 宽 512 / k_pe 宽 64 与该 cliff 无关。（订正：原"buffer width 576 ≥ 256"论据有误——SDPA fused-kernel 选择看 query 长度+head_dim∈支持集，与 buffer 宽度无关；absorbed-MLA 两路本就走 fallback，见 §4.5。）

---

## 4. `MlaAttention`（两形态解耦前向）

> **核心**：真实前向**解耦**——rope 分数 `pe_scores` 作为**加性 mask** 喂进只算 latent/nope 点积的 SDPA；且 decode 与 prefill 用**不同的 SDPA 形状**。这是对抗验证 verdict 1 的主修正，不是"单个 576 宽 SDPA"。

### 4.1 共享前缀（prefill / decode 相同）

| 步 | 操作 | 输出 | 备注 |
|---|---|---|---|
| 1 | `q_a_proj`: x[B,S,2048] @ Wᵀ (4bit→768) | c_q[B,S,768] | 量化 matmul g64 |
| 2 | `q_a_layernorm`(c_q, eps=1e-5) | c_q_n[B,S,768] | **在 q_b_proj 之前** |
| 3 | `q_b_proj`: →5120，reshape[B,S,20,256] | q[B,S,20,256] | |
| 4 | split `q_nope=q[...,:192]`, `q_pe=q[...,192:]` | [.,20,192],[.,20,64] | **NoPE-first/RoPE-second** |
| 5 | `kv_a_proj_with_mqa`: →576，split `c_kv[512]`,`k_pe[64]` | | MQA 单 k_pe |
| 6 | `kv_a_layernorm`(c_kv, eps=1e-5) | c_kv_n[B,S,512] | **归一后入缓存** |
| 7 | `q_pe = rope(q_pe, dims=64, traditional=true, base=1e6, offset)` | [B,S,20,64] | 见 §5 |
| 8 | `k_pe = rope(k_pe, …同上…)`，reshape[B,1,S,64] | [B,1,S,64] | 与 q_pe 同 op |
| 9 | **入缓存**：`c_kv ← [past;c_kv_n]`, `k_pe ← [past;k_pe]` | [B,1,L,512],[B,1,L,64] | L=past+S |

公共常量：`scale = 1/√(192+64) = 1/√256 = 1/16 = 0.0625`（DeepSeek-V2 Eq 18；`rope_scaling=null` ⇒ 无 YaRN/mscale 额外因子）。
公共 rope 分数：`pe_scores = (q_pe * 1/16) @ k_peᵀ`（64 维点积，broadcast 到单 kv 头）。

### 4.2 DECODE 形态（S==1）

```
q_nope[B,20,1,192] --embed_q(transpose=T)--> q_nope_latent[B,20,1,512]   # query 侧吸收 W^UK
k = v = cache_c_kv[B,1,L,512]
mask = pe_scores[B,20,1,L]
O = SDPA(q_nope_latent, k, v, scale=1/16, mask=pe_scores)                  # 头维 512/512/512
out[B,20,1,256] = O --unembed_out(transpose=T)--> 256                      # 输出侧吸收 W^UV
```

### 4.3 PREFILL 形态（S>1）

```
q_nope[B,20,S,192]                                                          # 不吸收
k = embed_q(cache_c_kv, transpose=False)[B,20,L,192]                        # 反吸收 latent->每头 k_nope
v = unembed_out(cache_c_kv)[B,20,L,256]                                     # latent->每头 v
mask = pe_scores + causal(-inf)   [B,20,S,L]                                # 单一加性 bias
O = SDPA(q_nope, k, v, scale=1/16, mask=mask)                              # 头维 192/192/256
out[B,20,S,256] = O                                                         # 已是 256，无后置 unembed
```

> 吸收在 decode 施于 query、在 prefill 反施（latent 展回每头 k/v）；两路**不共享张量形状**，仅共享 scale 与 pe-mask 技巧。

### 4.4 SDPA 收尾（两形态相同）

```
reshape heads: out[B,S,20,256] -> [B,S,5120]   (5120 = 20 * v_head_dim 256)
o_proj: [B,S,5120] @ Wᵀ (4bit 5120->2048)      -> u[B,S,2048]
```

### 4.5 SDPA fallback 事实（MLX 契约，本会话核实）

- 校验（`mlx/fast.cpp:677-702`）：仅强制 `q.shape(-1)==k.shape(-1)`、`k.shape(-3)==v.shape(-3)`（头数）、`n_q%n_kv==0`；**V 末维不与 Q/K 比较** → prefill 192/192/256 合法。
- fused kernel 选择（`scaled_dot_product_attention.cpp:621-626`）要求 `query_head_dim==value_head_dim` **且** head_dim∈{64,96,128,256}(vector)/{64,80,128}(full)。
  - decode 512/512/512：相等但 512 不在支持集 → fallback。
  - prefill 192/192/256：192≠256 → fallback。
- **结论：两路都走 unfused fallback**（`fast.cpp:724,733`：`multiply(scale,Q)→matmul(Q,Kᵀ)→+mask→softmax→matmul(scores,V)`，MQA 经 `expand_dims` broadcast）。与 omlx/mlx-lm 同款 → "追平 omlx"不依赖自研 kernel。
- mask 用**显式 array-mask 路径**（pe_scores 必须折入；prefill 再叠加 causal -inf），不走纯 `do_causal`。

### 4.6 regime 派发 + mixed-batch 风险

- 按本次 `forward_on` 的 query 长度 **S 派发**：`S==1` → decode；`S>1` → prefill。
- **待 plan 核实的风险**：ironmlx 批量调度是否会在单次 forward 混合 prefill 行与 decode 行。若是 chunked-prefill→decode 分步（S 在一次调用内同质），S-派发安全；否则需 per-row 拆分或统一走 prefill 路。→ 列入 §10 开放问题。

---

## 5. RoPE（glm-本地封装）

- 布局 = **INTERLEAVED**（`traditional=true` / rotate-every-two）。证据：MLX `rope.metal` traditional 分支取连续对 `(2i, 2i+1)`；[OBS] mlx-lm `glm4_moe_lite.py` `initialize_rope(traditional=True)`；[OBS] HF `rope_interleave=True`(默认)。**非** half-split/neox。
- 参数：`dims=64`（`partial_rotary_factor=1.0` ⇒ int(64*1.0)=64，32 个频率对），`base=1e6`，`scale=1.0`，`offset=cache_len`。512 个 latent/nope 通道是**结构性 decoupled-rope 拆分**（不旋转），不是 partial-rotary 尾巴。
- q_pe（每头[B,20,S,64]）与 k_pe（单头[B,1,S,64]）施同一 rope op。
- 入口：`glm4_moe_lite/rope.rs` 薄封装 `mlx::fast::rope_on`（`mlx/src/fast/mod.rs:94`）。**不用** `nn::mrope`（其 Metal shader 硬编码 split-half，`interleaved` 字段为死元数据）。
- offset 用法（scalar `rope_on` vs array `rope_with_array_offset_on`）按 decode 是否非均匀 batch 决定 → 列入 §10。

---

## 6. `Glm4MoeBlock`（路由 + 专家 + 首层 Dense）

layer 1..46 用本 block；layer 0 是 Dense FFN（`first_k_dense_replace=1`，复用 `nn::Mlp`，2048→10240→2048，不路由）。block 返回 `routed_out + shared_out`，外层 decoder 残差另加。

### 6.1 路由前向（per token x[.,2048]，noaux_tc）

| 步 | 操作 | 形状 | 来源 |
|---|---|---|---|
| 1 | `logits = x @ gate.weightᵀ`（gate **plain float**，非量化） | [.,64] | checkpoint |
| 2 | `s = sigmoid(logits)`（**非 softmax**；逐专家独立 (0,1)，不归一） | [.,64] | DeepSeek-V3 Eq 15 |
| 3 | `s_for_choice = s + e_score_correction_bias`（bias[64]，**仅选择用**） | [.,64] | Eq 16 |
| 4 | group mask：n_group=1/topk_group=1 → **no-op**（全 64 为一组） | [.,64] | DeepSeek-V3 节点限制路由 |
| 5 | `topk_idx = argtopk(s_for_choice, 4)` | [.,4] | Eq 16 选择支 |
| 6 | `topk_w = gather(s, topk_idx)` —— 取**原始 sigmoid s（不含 bias）** | [.,4] | Eq 16 值支（"bias 仅用于路由"） |
| 7 | if norm_topk_prob: `topk_w /= (sum(topk_w,-1) [+1e-20])` | [.,4] | Eq 13；`+1e-20` 为 [OBS] HF-only，可选 |
| 8 | `topk_w *= routed_scaling_factor`（=1.8） | [.,4] | [OBS] 归一后乘（HF+MLX-VLM 一致，无论文背书） |
| 9 | `routed_out = Σ_k w_k * expert_k(x)`；`expert=down(silu(gate(x))*up(x))` | [.,2048] | SwiGLU，gather_qmm over switch_mlp |
| 10 | `shared_out = shared_down(silu(shared_gate(x))*shared_up(x))` —— **ungated** | [.,2048] | n_shared=1，无门控（DeepSeek-V3 Eq 12） |
| 11 | `final = routed_out + shared_out` | [.,2048] | |

专家权重：`switch_mlp.{gate,up}_proj[64,1536,256]`、`down_proj[64,2048,192]`（4-bit g64），`moe_intermediate_size=1536`。**复用 `RoutedExperts` 原样**（prefix 参数化，64 专家，lazy gate/up fusion）。

### 6.2 静默 bug 警示（对抗验证确认）

1. sigmoid **不是** softmax。
2. 组合权重取**原始 sigmoid s**，不是 bias-corrected `s_for_choice`。
3. 共享专家 **ungated**（权重 1.0 直加；Qwen 的 sigmoid 门控在此是错的）。
4. router gate.weight 是 **plain float**，不走量化路径。
5. 归一化只在 **4 个选中**权重上，不是全 64。

---

## 7. 接线

| 关注点 | 位置 | 做法 |
|---|---|---|
| Registry | `cli/serve.rs:154-205` + `bin/ironmlx-core-bench.rs:139-162` | 两处 match 各加 `"glm4_moe_lite"` arm → `Glm4MoeLiteModel::from_loader` → `serve_with_model`/`run_for_model`。impl `Model` + `DenseVlMethods`(文本桩，vision_input=None)。 |
| Config | 新 `glm4_moe_lite/config.rs`（仿 `qwen3_5_moe/config.rs`） | 顶层扁平反序列化；`eos_token_id`→`EosTokenId::Multi`。 |
| Loader | `core/loader.rs:217-320` 通用 sanitize | 预期通用 sanitize 可用：`gate.weight` 走 `loader.tensor()`（plain）；量化权重走 `quant_meta_for`。layer0 dense `Mlp::from_loader("…layers.0.mlp")`；layer≥1 `RoutedExperts::from_loader("…mlp.switch_mlp")`。**须核实**：sanitize 中 Qwen/Gemma 系的 RMSNorm 约定（plain weight vs (1+w)）与 MTP 触发的 norm-offset（+1.0 shift）**不会误用于 GLM**（GLM 用标准 RMSNorm，`num_nextn_predict_layers=1` 但无 mtp 张量）→ §10。 |
| 多 EOS | `core/tokenizer.rs:177-192`、`core/loader.rs:32-40` | `[154820,154827,154829]`→`EosTokenId::Multi`，端到端已支持。 |
| lm_head | `tie_word_embeddings=false` | 独立 `lm_head.weight[154880,256]`。 |
| MTP | checkpoint 无 mtp 张量 | 确认 loader `mtp.*` strip 在缺失时不报错；model 代码不依赖（§10）。 |
| P5h 插桩 | `core/p5h.rs`、`sparse_moe.rs:562-583`(8 substep 范式) | `MlaAttention::forward_on`、`MlaLatentCache::update_and_fetch_on`、`Glm4MoeBlock` 包 `try_with_p5h_span_from_current_trace`，`cfg(feature="p5h-profile")` 门控，关闭时 no-op。decode/prefill substep 名分离。 |
| Model 结构 | 新 `glm4_moe_lite/model.rs` | embed_tokens、47 层（0 dense，1..46 MoE）、norm、lm_head。impl `Model`：`make_cache`→`Vec<LayerCache>`（每层 `LayerCache::Mla(MlaLatentCache)`）、`forward_on`、`num_hidden_layers=47`。position_ids 需求待核实：GLM 用简单 rope+offset，可能不需要 Qwen mrope 式 position_ids（rope offset 取每行 cache 长度）→ §10。 |

---

## 8. 正确性验收

- 基线：**mlx-vlm**（correctness baseline 约定）。
- 指标：固定 prompt 的 **logits 数值对齐**（top-k 一致 + 关键位 logit 差在容差内）+ 短文本 **PPL/greedy 续写一致**。
- `+1e-20` 归一 epsilon：按所对照的参照实现取值（HF 含、MLX-VLM 省）。是否要求 bit-exact 见 §10。
- 单元/集成：MLA 两形态各自的小张量数值测试（与手算/参照对拍）；router 各步（sigmoid/bias-select/原始权重/norm/scale/ungated-shared）的定向测试；dense-layer0 vs MoE-layer 装配测试。

---

## 9. 性能验收 + feasibility-gated kernel 候选

- 工具：**iron-bench**（串行跑，避免 GPU/swap 互污）。
- 硬验收线：**ironmlx tok/s（prefill + decode）≥ omlx**（同 4-bit 模型）。
  - **前置**：先确认 omlx（mlx-lm）能跑 `glm4_moe_lite` 并建基线；若不支持，则"追平 omlx"无从定义，回退为约定 tok/s 目标（届时与 Boss 确认）。
- 预期：op-composition 大概率已追平（omlx 同走 fallback + Python 调度开销；ironmlx Rust 管线 + 已有 fused projection / 优化 gather_qmm / KV cache 复用 有结构性优势）。
- **唯一明确的 beat-omlx 性能杠杆**（feasibility-gated kernel 候选）：把 decode 路 **上投影到 256 维 MHA**（`head_dim=256 ∈ {64,96,128,256}` 且 256==256 → 命中 fused decode kernel）。代价：放弃 576/token 的小 latent cache，改存每头 k_nope+v 全量。
  - **政策约束**：按 feasibility-gate（Amdahl 上限 + MLX-kernel 饱和度 pre-screen + 历史证据）须先过 **PP=128/512 双点 e2e ≥5%** 门，方可进入 kernel 实现。**不凭直觉做**（Stage β 反例）。

---

## 10. 开放问题 / 计划阶段须解决的风险

1. **[设计] decode/prefill regime 派发 + mixed-batch**（§4.6）：核实 ironmlx 调度是否单次 forward 混合 prefill+decode 行；决定 S-派发 vs per-row 拆分 vs 统一 prefill 路。
2. **[性能-feasibility gate] 上投影 256-MHA**（§9）：进 kernel 前须过 PP=128/512 双点 e2e ≥5%。
3. **[正确性] `+1e-20` 归一 epsilon**：是否要求与参照 bit-exact。
4. **[config 复核] `routed_scaling_factor=1.8`**（与 DeepSeek-V3 的 2.5 不同，无论文推导）+ 归一后乘的 **ordering**（[OBS]-only）→ 以 runtime config.json 为准复核。
5. **[仅自研 kernel 时] RoPE 通道序**：HF de-interleave 与 MLX traditional 旋转值同、通道序不同。GLM 仅经 `q_pe·k_peᵀ` 共享置换不变（数值 diff=0 已验），故走 `rope_on` 无碍；若改自研 cos/sin/融合 kernel 须保持一致布局。
6. **[RoPE offset] scalar vs array**：decode 非均匀 batch 决定 `rope_on` vs `rope_with_array_offset_on`。
7. **[路由通用性] top-2-per-group 评分**：GLM n_group=1 是 no-op；实现按 n_group=1 faithful（不做投机性通用分组）。未来 n_group>1 变体须对其参照重核分组规则（[OBS] top-2-per-group 非论文文字）。
8. **[长上下文] YaRN/mscale**：`max_position_embeddings=202752` 但 config 无 `rope_scaling` 块；以 runtime config.json 复核是否真无（若有会改 1/16 scale）。
9. **[MTP] `num_nextn_predict_layers=1`** 但 checkpoint 无 mtp/nextn 张量：确认 loader strip 不报错、model 不依赖。
10. **[正确性-loader] RMSNorm 约定 + MTP norm-offset 误用风险**（§7 loader 行）：核实 `core/loader.rs` sanitize 不会把 Qwen/Gemma 系的 RMSNorm `(1+w)` 约定或 MTP 触发的 `+1.0` norm-offset 误施于 GLM（GLM 用标准 RMSNorm；误施会污染全部 norm 权重）。确认 `nn::RmsNorm` 对 GLM 用 plain-weight 约定。
11. **[设计] position_ids / rope offset 来源**（§7 model 行）：核实引擎 `forward_on` 是否要求 model 声明 `requires_position_ids`，以及 GLM 简单 rope 的 offset 应取自每行 cache 长度（`MlaLatentCache::offsets`）而非 mrope 式 position_ids；非均匀 batch 下与 §10.6 的 scalar/array offset 选择联动。

---

## 11. 参考文件（绝对路径）

- MLX：`/Users/xin/workspace/iron-rivals/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`（use_fallback 591-640）、`mlx/fast.cpp`（SDPA 校验 677-702、fallback 724-733）、`mlx/src/fast/mod.rs`（rope_on:94、rope_with_array_offset_on:144）、`mlx/ops.cpp`（quantized_matmul 批量广播 40-43）、`mlx/backend/metal/quantized.cpp`（批量 `_batch_1` 142-214）。
- ironmlx：`nn/mrope.rs`（split-half shader 408-504、guard 287-291）、`models/gemma4/rope.rs`、`models/qwen3_5_moe/sparse_moe.rs`（RoutedExperts 233-332、SparseMoeBlock 434-508、p5h 562-583）、`nn/decoder_layer.rs`（AttnPath/LayerCache 53-91）、`core/cache/kv_cache.rs`、`cli/serve.rs`(154-205)、`bin/ironmlx-core-bench.rs`(139-162)、`core/loader.rs`（sanitize 217-320、EosTokenId 32-40）。
- 论文：DeepSeek-V2 arXiv:2405.04434 §2.1（MLA + 矩阵吸收）、DeepSeek-V3 arXiv:2412.19437 §2.1.2（noaux_tc，Eq 12-16）。
