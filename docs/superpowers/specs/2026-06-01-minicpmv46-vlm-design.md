# MiniCPM-V-4.6 (`minicpmv4_6`) 完整 VLM 视觉栈接入设计

状态：设计待 Boss 复审。worktree `ironmlx-backend-minicpmv46`，branch `minicpmv46-text-support`。
前置：text-only 文本骨干已落地并验证（commit `236db39`）。本设计在其上接入视觉栈。

## 来源与方法学说明（binding）

- spec 的 authoritative 来源 = 真实 `config.json` + safetensors header（张量形状/dtype）+ MLX API 契约。
- mlx-vlm `models/minicpmv4_6/{vision,minicpmv4_6,processing,config}.py` 仅作 **observation / triangulation**，用于理解前向语义与确认默认值；**不作 specification**，ironmlx 独立做最优实现（见 [[feedback_no_spec_from_competitors]] [[feedback_design_philosophy]]）。
- 正确性判据 = 同输入下与 mlx-vlm 的 **logits / 中间张量 数值对齐**（argmax + top-k + max_abs/cosine），非 greedy 文本（BOS 协议差异，见 [[project_minicpm5_llama_integration]]）。

## 0. 结论摘要

MiniCPM-V-4.6 是 VLM：SigLIP 视觉编码器 + 中插式 VitMerger 重采样 + Merger 投影 + Qwen3.5-text 语言骨干（已支持）。本设计新增视觉栈，新建 `MiniCpmV46Model`（持有复用的 `Qwen35TextModel` + `Option<MiniCpmV46Vision>`，实现 `Model + DenseVlMethods`），**吸并现有 text-only facade**（vision=None 即纯文本，数值与今日一致）。范围：**仅图像（单图+多图），暂缓视频**。分三阶段实现，每阶段独立数值验收门。

关键简化（相对 Qwen3.5-VL）：
1. **LM 用纯顺序 1D 位置**（图像 token 不用 2D MRoPE）→ 复用 `build_position_ids`，无需新 get_rope_index。
2. **整个视觉栈是 bf16（未量化）** → 视觉路径无 dequant。
3. **跨模态 scatter 复用** `cross_modal::replace_image_tokens`。

## 1. 模型事实（authoritative，自 config.json + safetensors header）

### 1.1 顶层 config
- `model_type = "minicpmv4_6"`，`architectures = ["MiniCPMV4_6ForConditionalGeneration"]`
- `image_token_id = 248056`；`video_token_id = 248057`（out of scope）
- `insert_layer_id = 6`（VitMerger 插入位置，**编码器中部**）
- `image_size = 1120`（顶层；processor 用）；`tie_word_embeddings = true`
- `quantization`：4-bit affine gs=64（**仅作用于 LM**；视觉栈张量为 bf16）

### 1.2 vision_config（`minicpmv4_6_vision`，SigLIP）
- `hidden_size = 1152`，`intermediate_size = 4304`，`num_hidden_layers = 27`
- `num_attention_heads = 16`（head_dim = 72），`patch_size = 14`，`image_size = 980`
- `hidden_act = "gelu_pytorch_tanh"`，`layer_norm_eps = 1e-6`，`num_channels = 3`
- 派生默认（mlx-vlm ModelConfig 观测；ironmlx 以 config + 张量形状为准）：`query_num=64`、`downsample_mode="16x"`、`merge_kernel_size=(2,2)`、`merger_times=1`、`window_kernel_size=(2,2)`、`slice_mode=true`

### 1.3 视觉张量形状（bf16，未量化）
| 张量 | 形状 | 含义 |
|---|---|---|
| `vision_tower.embeddings.patch_embedding.weight` | `[1152,14,14,3]` | Conv2d (out, kh, kw, in)；可等价为 patch-flatten Linear `[1152, 14*14*3=588]` |
| `vision_tower.embeddings.patch_embedding.bias` | `[1152]` | |
| `vision_tower.embeddings.position_embedding.weight` | `[4900,1152]` | 学习位置表 70×70（=image_size/patch=980/14） |
| `vision_tower.encoder.layers.{0..26}.{layer_norm1,layer_norm2}.{weight,bias}` | `[1152]` | 标准 LayerNorm（含 bias） |
| `…self_attn.{q,k,v,out}_proj.{weight,bias}` | w `[1152,1152]`, b `[1152]` | 标准 MHA（含 bias，无 RoPE，无 QK-norm） |
| `…mlp.fc1.{weight,bias}` / `fc2` | `[4304,1152]` / `[1152,4304]` | GELU-tanh |
| `vision_tower.post_layernorm.{weight,bias}` | `[1152]` | |
| `vit_merger.layer_norm1.{weight,bias}` | `[1152]` | 窗口注意力前 norm |
| `vit_merger.self_attn.{q,k,v,out}_proj.{weight,bias}` | `[1152,1152]` | CrossAttention(1152, 16 heads) |
| `vit_merger.pre_norm.{weight,bias}` | `[4608]` | group_hidden=1152*4 |
| `vit_merger.linear_1.{weight,bias}` | `[17216,4608]` | window_intermediate=4304*4 |
| `vit_merger.linear_2.{weight,bias}` | `[1152,17216]` | |
| `merger.mlp.0.pre_norm.{weight,bias}` | `[4608]` | |
| `merger.mlp.0.linear_1.{weight,bias}` | `[4608,4608]` | |
| `merger.mlp.0.linear_2.{weight,bias}` | `[1024,4608]` | 输出 1024 = LM hidden ✓ |

> 注意：sanitize 现丢弃 `vision_tower.*`（`Loader::open`）。VL 路径须 `Loader::open_multimodal`，并须保留 `vit_merger.*` / `merger.*`（当前 sanitize 仅按 `vision_tower.` 前缀保留——**须扩展保留 `vit_merger.`/`merger.` 前缀**，见 §8.4）。

### 1.4 关键结构语义（observation 自 mlx-vlm，待实现期 fixture 验证）
- **VitMerger 中插**：encoder L0–6 → VitMerger（grid÷2×2）→ encoder L7–26 → post_layernorm → Merger（grid÷2×2）。总 16× 下采样。
- **VitMerger 前向**：把 `[grid_h, grid_w, 1152]` 按 2×2 窗口重排成 `[merged_h*merged_w, 4, 1152]`；`layer_norm1`→窗口内 self-attn→残差；`residual = mean(窗口, axis=1)`；窗口 flatten 成 `[*, 4608]`→`pre_norm`→`linear_1`→GELU→`linear_2`→ `+ residual`。
- **Merger 前向**：`[grid_h,grid_w,1152]` 按 2×2 重排成 `[*, 4608]` → `pre_norm`→`linear_1`→GELU→`linear_2` → `[*, 1024]`。
- **SigLIP embeddings**：patch（Conv 或 packed-matmul）+ NaViT 式分桶位置插值（按 tgt 网格把每 patch 映射到 70×70 学习位置表的 bucket id）。
- **LM 位置**：`_set_position_state` = `arange(S)` 广播到 `[3,B,S]`（图像也用顺序 1D）。

## 2. 模块结构（新增 `ironmlx/src/models/minicpmv4_6/`）

```
minicpmv4_6/
  mod.rs            ← 导出 MiniCpmV46Model + config；model_from_loader 改为构造 MiniCpmV46Model
  config.rs         ← 已有(扩展)：text_config→Qwen35Config + 解析 vision_config(MiniCpmV46VisionConfig) + 顶层 insert_layer_id/image_token_id/merge 参数
  model.rs          ← 新 MiniCpmV46Model：Qwen35TextModel(复用) + Option<MiniCpmV46Vision>；impl Model + DenseVlMethods
  vision/
    mod.rs          ← MiniCpmV46Vision::{from_loader, compute_vision_embeds}；编排中插前向
    embeddings.rs   ← SiglipEmbeddings：patch embed(Conv/packed) + 分桶位置插值
    encoder.rs      ← SiglipEncoderLayer × 27（LN→MHA(bias)→res→LN→MLP(gelu_tanh)→res）
    merger.rs       ← VitMerger（窗口 cross-attn）+ Merger（2×2 投影到 LM hidden）
  image_processor.rs ← LLaVA-UHD 预处理（P2 单图无切片 → P3 自适应多切片）
```

复用：`nn::{Linear, RmsNorm, Embedding}`、`mlx::fast::scaled_dot_product_attention`、`cross_modal::replace_image_tokens`、`core::generate::build_position_ids`（顺序 1D，VL 与纯文本同）、KVCache/LayerCache、`Qwen35TextModel`。
> 注：SigLIP 用 **标准 LayerNorm（含 bias）**，需确认 `nn` 有 LayerNorm；若无则新增最小 `nn::LayerNorm`（weight+bias+eps，按最后一维归一）。这是唯一可能缺失的基元。

## 3. SiglipEmbeddings（§vision/embeddings.rs）
- 输入：预处理后的 pixel_values + tgt_size `(grid_h, grid_w)`。
- patch embed 两路径（与权重布局对齐，实现期按 fixture 选定其一）：
  - packed：pixel_values `[B, 14, n*14, 3]` → reshape `[B, n, 588]` → `@ W^T(588×1152)` + bias。
  - Conv：`Conv2d(3→1152, k=14, s=14)`。
- 位置插值：对 `(grid_h, grid_w)`，用分桶 `frac = arange(n)/n`，`bucket = sum(frac ≥ boundaries)`，`pos_id = bucket_h*70 + bucket_w`，取 `position_embedding[pos_id]` 加到 patch embed。
- 输出 `[B, grid_h*grid_w, 1152]`。

## 4. SiglipEncoder（§vision/encoder.rs）
- 27 × 层：`h += SDPA(LN1(h)的 q/k/v, 16 heads, scale=72^-0.5)` 经 out_proj；`h += MLP(LN2(h))`，MLP = `fc2(gelu_tanh(fc1(·)))`。
- 全 bf16；无 RoPE、无 QK-norm、无 attention mask（单图全可见）。

## 5. VitMerger + Merger（§vision/merger.rs）
- 见 §1.4 前向语义。grid 不整除 2×2 → 明确 `Err`（§11）。
- VitMerger 在 encoder 中部由编排层（§6）调用一次（`insert_layer_id=6` 之后）。
- Merger 在 post_layernorm 之后调用，输出 `[N_tokens, 1024]`。

## 6. 视觉前向编排（§vision/mod.rs `compute_vision_embeds`）
```
embed = SiglipEmbeddings(pixel, tgt)          # [1, G, 1152], G=grid_h*grid_w
h = embed
for i in 0..27:
    h = encoder.layers[i](h)
    if i == insert_layer_id(=6):
        h, grid_h, grid_w = vit_merger(h[0], grid_h, grid_w)   # grid ÷2×2
        h = h[None]
h = post_layernorm(h)[0]
merged, _, _ = merger(h, grid_h, grid_w)        # grid ÷2×2 → [N, 1024]
return merged
```
多图：逐图调用后沿 axis 0 拼接（复用 Qwen35Model 多图 concat 模式）。

## 7. 图像预处理（§image_processor.rs，分阶段）
- **P2（单图，max_slice_nums=1，无切片）**：resize（保持 patch 整除）→ 归一化（mean/std）→ patch-pack 成 SiglipEmbeddings 期望布局 → 产出 `pixel_values + (grid_h, grid_w)`。验证 pixel_values + grid 逐位对齐 mlx-vlm processor（slice_mode 关）。
- **P3（自适应多切片 + 多图）**：`_find_best_resize` / `_get_refine_size` 网格搜索（max_slice_nums=9）→ 切分 slices + global thumbnail → 各自 pack + grid → prompt 插入 `<slice></slice>` 标记与对应 image_token 数。**最大正确性风险**，逐函数对齐 mlx-vlm/HF processor。
- 归一化常量、resize 插值方式（bicubic?）、pack 布局以 mlx-vlm processor + HF preprocessor_config.json 为 observation，fixture 逐位校验。

## 8. MiniCpmV46Model + 接线（§model.rs + dispatch）

### 8.1 结构（吸并 facade）
```rust
pub struct MiniCpmV46Model {
    text: Qwen35TextModel,            // 复用
    lm_head: Option<Linear>,          // tied → None
    vision: Option<MiniCpmV46Vision>, // 仅 open_multimodal + 权重存在时 Some
    image_token_id: i32,              // 248056
}
```
- `from_loader`：config 适配器产出 `Qwen35Config` + `MiniCpmV46VisionConfig`；vision 仅当 `loader.contains("vision_tower.embeddings.patch_embedding.weight")` 时加载（哨兵，仿 Qwen35Model）。
- impl `Model`（text 路径：forward_on/batched_prefill/forward_text_hidden/make_cache/model_meta）+ `DenseVlMethods`（compute_vision_embeds/forward_vl_chunk/forward_vl_hidden/batched_prefill_vl）。

### 8.2 跨模态 + 位置
- scatter：`cross_modal::replace_image_tokens(hidden, input_ids, vision_embeds, image_token_id=248056)`（复用，含 token 数=vision 行数校验）。
- 位置：复用 `build_position_ids`（顺序 1D），VL 与纯文本同。

### 8.3 dispatch（generate/serve/bench 三处）
- `MiniCpmV46 =>` 构造 `MiniCpmV46Model`，替换现有 `minicpmv4_6::model_from_loader -> Qwen35Model`。
- generate：有 `--image` → `Loader::open_multimodal` + 走 VL 路径；无 → `Loader::open` + vision=None 纯文本（与今日一致）。
- serve：`open_multimodal`；新增 `VisionInputConfig::MiniCpmV46`（仿 Gemma4 分支）驱动 server 图像预处理。
- 现有 `minicpmv4_6_text_logits_match` 回归测试：改为驱动 `MiniCpmV46Model` 的文本路径（或保留 helper），保持 LM 回归有效。

### 8.4 Loader sanitize 扩展
- `open_multimodal(keep_vision_tower=true)` 当前仅保留 `vision_tower.` 前缀；**须同时保留 `vit_merger.` 与 `merger.` 前缀**（否则 resampler 权重被丢弃）。`Loader::open`（text-only）仍丢弃三者。

## 9. 分阶段实现 + 验收门

| 阶段 | 范围 | 验收门（对齐 mlx-vlm） |
|---|---|---|
| **P1 视觉栈** | embeddings + encoder(27) + VitMerger + Merger + 编排 | 喂 mlx-vlm 抓取的单图 pixel_values fixture → `compute_vision_embeds` 输出 vs mlx-vlm `get_vision_embedding`：max_abs < 阈(bf16 噪声，首测定标) + cosine > 0.999 |
| **P2 单图 e2e** | MiniCpmV46Model + 跨模态 + 单图无切片预处理 + dispatch/CLI/serve + sanitize 扩展 | (a) preprocess pixel_values+grid 逐位对齐；(b) 端到端单图 VL logits：argmax + top-5 集一致 + max_abs 结构阈；(c) `ironmlx generate --image` 连贯正确 |
| **P3 多切片+多图** | LLaVA-UHD 自适应切片(max_slice_nums=9) + 多图 + slice 标记 | 多切片/多图 preprocess 逐位对齐 + e2e VL logits 对齐 |

每阶段：fmt + 规范 clippy gate + build --release + lib 测试保持绿；新增测试 `#[ignore]` env-gated（真实 checkpoint）。

## 10. 正确性验收（fixtures，仿 p6）
- gen 脚本 `tests/fixtures/minicpmv46_vl/gen_*.py`（mlx-vlm 驱动）产出：preprocessed pixel_values、vision-embeds、final VL logits 参照；`expected_*.npy` 按 p4/p6 惯例 **gitignore**，gen 脚本入库。
- 判据分层：中间张量（vision-embeds）用 max_abs + cosine；最终 logits 用 argmax + top-5 集相等 + max_abs 结构阈（阈值按本模型噪声第一性原理定，见 [[feedback_first_principles_feasibility_gate]]，不事后放松）。
- 测试图像 fixture：复用 `tests/fixtures/p6_qwen35_vl/*.jpg` 或新增小图。

## 11. 错误处理 / 开放问题 / 风险

**错误处理**：grid 不整除 `merge_group_size(2,2)` → `Err`（VitMerger/Merger 各自校验）；image_token 数 ≠ vision token 数 → 复用 scatter 既有校验；预处理 slice 数越界（>max_slice_nums）→ 显式 bound。

**开放问题（计划阶段须解决）**：
1. patch embed 走 Conv2d 还是 packed-matmul？取决于预处理产出布局；P1 fixture 定夺（两者数值须等价）。
2. resize 插值算法（bicubic/bilinear）+ 归一化常量：以 HF `preprocessor_config.json` + mlx-vlm processor 为 observation，P2/P3 fixture 逐位校验。
3. `nn::LayerNorm`（含 bias）是否已存在；若无则新增最小实现。
4. server 端图像预处理接线（`VisionInputConfig::MiniCpmV46` + GenerationStream VL 路径）细节，P2 落实。

**风险**：P3 LLaVA-UHD 切片预处理是最大正确性风险（网格搜索/packing/pos 插值须逐位对齐，p6 同族曾在 preprocess diff 上大量调试）。P1/P2 的分阶段 fixture 验收用于隔离编码器/重采样正确性与切片正确性。

**性能**（[[feedback_performance_stability_priority]]）：视觉栈 bf16，单图前向一次性；本阶段先保证正确性，性能调优（如 vision SDPA tile）留后续，不在本 spec 范围。

## 12. 参考文件（绝对路径）
- 模型：`~/.ironmlx/models/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19/`
- mlx-vlm observation：`/Users/xin/workspace/iron-rivals/mlx-vlm/mlx_vlm/models/minicpmv4_6/{vision,minicpmv4_6,processing_minicpmv4_6,config}.py`
- ironmlx 复用：`ironmlx/src/models/qwen3_5/{model,text_model,cross_modal}.rs`、`ironmlx/src/models/vision/`、`ironmlx/src/models/gemma4/vision.rs`（SigLIP 家族参考）、`ironmlx/src/core/loader.rs`(sanitize)、`ironmlx/src/core/generate.rs`(build_position_ids/VL)
- text-only 基线：commit `236db39` + `ironmlx/tests/minicpmv46_text_logits_match.rs`
