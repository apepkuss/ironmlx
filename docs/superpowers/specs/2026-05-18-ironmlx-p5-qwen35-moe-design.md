# ironmlx P5 — Qwen3.5 MoE Foundation 设计

| 字段 | 值 |
|---|---|
| 日期 | 2026-05-18 |
| 状态 | Brainstorming approved，准备 writing-plans |
| 范围 | Qwen3.5 MoE 模型 LM text-only 推理链路（验证模型 `mlx-community/Qwen3.5-35B-A3B-4bit`） |
| 工作分支 | `ironmlx-p5-moe` |
| 上游分支 | `ironmlx` |
| 验证基线 | omlx CLI 数值对齐为主 + mlx-lm 源码算法 reference 为辅（**不对齐实现**） |
| 显式 out-of-scope | VL 路径、MoE-aware MTP、Expert offloading、Fused Metal expert kernel |

## § 1 调研发现与决策摘要

### 1.1 目标模型关键事实（mlx-community/Qwen3.5-35B-A3B-4bit）

| 字段 | 值 | 含义 |
|---|---|---|
| `model_type` | `qwen3_5_moe` | 顶层 dispatch 入口区分 dense / moe |
| `text_config.model_type` | `qwen3_5_moe_text` | 与 dense `qwen3_5_text` 平行 |
| `num_hidden_layers` | 40 | 比 dense 4B (28~32) 多 |
| `num_attention_heads / num_kv_heads / head_dim` | 16 / 2 / 256 | GQA 8:1，attn dim = 4096 |
| `hidden_size` | 2048 | 比 dense 4B 略小（4B 是 2560） |
| `full_attention_interval` | 4 | 与 dense 4B 完全一致的 hybrid linear/full 模式 |
| `linear_num_value_heads / num_key_heads / value_head_dim / key_head_dim / conv_kernel_dim` | 32 / 16 / 128 / 128 / 4 | 与 dense 4B 不同 |
| **`num_experts`** | **256** | 高 routing 空间 |
| **`num_experts_per_tok`** | **8** | top-8 routing |
| **`moe_intermediate_size`** | **512** | 单 expert FFN intermediate 维度 |
| **`shared_expert_intermediate_size`** | **512** | 常驻 SwiGLU（与 routed 并行求和） |
| `mlp_only_layers` | `[]` | 所有 layer 都 MoE，无 dense MLP fallback layer |
| `tie_word_embeddings` | **false** | 独立 lm_head（dense 4B 是 `true`） |
| `vocab_size` | 248320 | 与 dense 4B-VL 共享 tokenizer |
| `vision_config` 存在 + `mtp_num_hidden_layers: 1` | 是 | 本身是 VL + MTP 模型 → **P5 显式跳过这两条路径** |
| `quantization` | `{group_size:64, bits:4, mode:"affine"}` | 4-bit affine 量化，与现有 nn::Linear quant 路径兼容 |
| 模型权重估算 | ~17.5 GB | 32GB+ Apple Silicon 可全 resident |

### 1.2 决策摘要

| 决策项 | 选定方案 | 理由 |
|---|---|---|
| **D1 共存** | `trait Model` 抽象 | 与 `scheduler-actor design` 已标 deferred 的 trait debt 对齐；API surface 小（5 method）；后续模型零成本接入 |
| **D2 范围** | 仅 LM text-only | 风险隔离最大化；VL/MTP 留独立 phase |
| **D3 kernel 策略** | P5b plan 阶段先调研 `mlx::gather_qmm` | 复用 mlx 已有 op 优先；fallback per-expert scatter+qmm |
| **Expert 策略** | 全 resident，不 offload | Apple Silicon UMA 下 offload 无 "快 + 省内存" 双赢路径；17.5GB 在 32GB+ Mac 完全够用；dmlx 实测数据印证（见 [project_dmlx_moe_offload_observation](../../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/project_dmlx_moe_offload_observation.md)） |
| **Q1 trait 形状** | `Generic <M: Model>` 编译期 monomorphize | 零运行时开销；与 ironmlx hot-path 哲学一致；与现有 `impl Into<StreamOrDevice>` generic pattern 一致 |
| **Q2 expert 权重布局** | Hybrid：`shared_expert` per-Linear + `routed_experts` fused stacked | 与 mlx-lm SwitchGLU + dmlx 事实标准一致；适配未来 mlx::gather_qmm；shared 与 routed 数据流本就不同自然分开 |
| **Q3 stage 拆分** | 4 sub-phase（P5a/P5b/P5c/P5d） | 风险逐 phase 隔离；与 b1-p2-3 历史拆分哲学一致；遵守 [feedback_task_breakdown_bounded](../../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_task_breakdown_bounded.md) "拆到 sub-phase 一层为止" |
| **Q4 验证基线** | omlx CLI 数值主对齐 + mlx-lm 算法辅参考 | 实现独立（不对齐代码），但输出 logits / token 在同 prompt 下与 omlx greedy 一致是终极正确性标准 |

## § 2 算法 / 数据流

### 2.1 整体 HTTP request → MoE token stream

```
[HTTP request]
   ↓
SchedulerActor<Qwen35MoeModel>     ← AppState<M: Model> generic
   ↓
Scheduler<Qwen35MoeModel>::step()
   ↓
Qwen35MoeModel::forward_on(input_ids[B,S], position_ids[3,B,S], cache)
   ↓
Qwen35MoeTextModel::forward_on
   ↓
embed_tokens → [B, S, 2048]
   ↓
for layer_idx in 0..40:
   DecoderLayerMoe::forward_on
      input_layernorm
      → Full(layer_idx in {3,7,...,39}) | Linear(others)  attention
      → +residual
      → post_attention_layernorm
      → SparseMoeBlock          ← MoE 替换 dense Mlp
      → +residual
   ↓
norm
   ↓
slice last position
   ↓
lm_head (Linear, untied)         → [B, 1, 248320]
```

### 2.2 SparseMoeBlock forward 数据流

```
x: [B, S, hidden=2048]
flat_x = reshape(x, [B*S, 2048])

(1) gate logits
    logits = router_linear(flat_x)            # [B*S, num_experts=256]

(2) softmax + topk + (optional) renorm
    probs = softmax(logits, axis=-1)          # [B*S, 256]
    (topk_probs, topk_idx) = topk(probs, k=8) # [B*S, 8] × 2
    if norm_topk_prob:
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=true)

(3) routed experts via gather_qmm
    routed = SwiGLU_gather(
        flat_x,                                # [B*S, 2048]
        topk_idx,                              # [B*S, 8] expert ids
        topk_probs,                            # [B*S, 8] weights
        routed_experts.gate_weight,            # [256, 512, 2048_packed]
        routed_experts.up_weight,
        routed_experts.down_weight,
        ...scales/biases
    )                                          # [B*S, 2048]

(4) shared expert (parallel path)
    shared = shared_expert.forward_on(flat_x)  # [B*S, 2048]

(5) sum + reshape
    out = routed + shared
    return reshape(out, [B, S, 2048])
```

具体 `SwiGLU_gather` 实现取决于 `mlx::gather_qmm` 是否暴露（D3 plan 阶段调研），有两条备选路径：
- **路径 G1（首选）**：`mlx::quantization::gather_qmm` 单 fused op + SwiGLU 应用
- **路径 G2（fallback）**：per-token-expert scatter → 8 次 per-expert quantized_matmul → 加权 sum

### 2.3 DecoderLayer dispatch（per layer）

```
DecoderLayerMoe (qwen3_5_moe::decoder_layer::DecoderLayerMoe)
├─ input_layernorm: RmsNorm
├─ attn: AttnPath
│   ├─ Full(GatedAttention)    # layer_idx in {3,7,11,15,19,23,27,31,35,39}
│   └─ Linear(GatedDeltaNet)   # 其它 30 层
├─ post_attention_layernorm: RmsNorm
└─ ffn: SparseMoeBlock         # ← 区别 dense（dense 是 nn::Mlp）
```

DecoderLayerMoe 内的 attention 部分（input_layernorm / attn / post_attention_layernorm）**继续使用 nn:: 共享 primitives**（`RmsNorm` / `GatedAttention` / `GatedDeltaNet`），它们已是跨架构 shared 组件，不算 cross-model imports。仅 `ffn` 字段从 `Mlp` 替换为 `SparseMoeBlock`，且 layer 容器自身在 `qwen3_5_moe/` 目录独立 copy（避免 dense DecoderLayer.ffn 字段污染）。

## § 3 详细设计

### 3.1 `core::model::Model` trait

```rust
// ironmlx/src/core/model.rs
use mlx::{Array, Dtype, StreamOrDevice};
use crate::core::memory_budget::ModelMeta;
use crate::nn::LayerCache;
use crate::Result;

pub trait Model {
    fn make_cache(&self, batch: i32, cap: i32, dtype: Dtype) -> Result<Vec<LayerCache>>;

    #[allow(clippy::too_many_arguments)]
    fn forward_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        per_row_lens: Option<&[i32]>,
        decode_mask: Option<&Array>,
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn batched_prefill(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        linear_attention_mask: &Array,
        per_row_lens: &[i32],
        cache: Option<&mut [LayerCache]>,
        target: StreamOrDevice,
    ) -> Result<Array>;

    fn model_meta(&self) -> ModelMeta;

    fn num_hidden_layers(&self) -> usize;
}
```

**关键设计决定**：
- 不使用 associated type `Cache`；两个 model 共享 `Vec<LayerCache>` 形态（Linear/Full 已通过 enum 区分），保持 trait object-friendly + 简化 scheduler 通用代码
- 不在 trait 内放 VL 方法（`forward_vl_chunk` / `batched_prefill_vl` / `compute_vision_embeds`），由 dense `Qwen35Model` 作为额外 inherent method（不影响 generic scheduler），P6.x 引入 VL trait 扩展（如 `trait MultimodalModel: Model`）

### 3.2 `Qwen35MoeConfig`

```rust
// ironmlx/src/models/qwen3_5_moe/config.rs
use serde::Deserialize;
use crate::core::Loader;
use crate::nn::AttnKind;
use crate::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub mrope_section: Vec<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MoeConfig {
    // ─ Dense shared 字段 ─
    pub hidden_size: i32,
    pub intermediate_size: i32,                // 注意：这是 shared_expert 的 intermediate；与 dense 字段同名
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub full_attention_interval: i32,
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,
    #[serde(default)]
    pub rope_parameters: RopeParams,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,

    // ─ MoE 专属字段 ─
    pub num_experts: i32,                      // 256
    pub num_experts_per_tok: i32,              // 8 (topk)
    pub moe_intermediate_size: i32,            // 512 (per-routed-expert FFN)
    pub shared_expert_intermediate_size: i32,  // 512 (shared expert FFN)
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,             // [] for A3B
    #[serde(default)]
    pub norm_topk_prob: bool,                  // 默认 false; plan 阶段读 mlx-lm 确认 Qwen3.5-MoE 默认
    #[serde(default)]
    pub router_aux_loss_coef: f32,             // 推理时忽略
}
```

`Qwen35MoeConfig::from_loader` 从 `loader.config_raw_value()["text_config"]` 解析；`layer_kind(layer_idx)` 同 dense 实现 `(idx + 1) % full_attention_interval == 0 ? Full : Linear`（不读 `layer_types` 数组，因 `full_attention_interval` 已完整描述 hybrid 模式）。

### 3.3 `Qwen35MoeTextModel`

```rust
// ironmlx/src/models/qwen3_5_moe/text_model.rs
pub struct Qwen35MoeTextModel {
    embed_tokens: Embedding,                 // 来自 nn::Embedding
    layers: Vec<DecoderLayerMoe>,            // 来自 qwen3_5_moe::decoder_layer
    norm: RmsNorm,                           // 来自 nn::RmsNorm
    mrope: Mrope,                            // 来自 nn::Mrope
    cfg: Qwen35MoeConfig,
}
```

API surface 与 `Qwen35TextModel` 对等：`from_loader / from_components(test) / config / num_layers / embed_on / forward_post_embedding_on / forward_on / as_output_on`。

### 3.4 `Qwen35MoeModel` + `impl Model`

```rust
// ironmlx/src/models/qwen3_5_moe/model.rs
pub struct Qwen35MoeModel {
    text: Qwen35MoeTextModel,
    lm_head: Linear,                         // 必有（tie_word_embeddings = false）
}

impl Qwen35MoeModel {
    pub fn from_loader(loader: &Loader) -> Result<Self> { ... }
    pub fn config(&self) -> &Qwen35MoeConfig { self.text.config() }
    pub fn approx_weight_bytes(&self) -> usize {
        let cfg = self.config();
        let h = cfg.hidden_size as usize;
        let l = cfg.num_hidden_layers as usize;
        let e = cfg.num_experts as usize;
        let me = cfg.moe_intermediate_size as usize;
        let se = cfg.shared_expert_intermediate_size as usize;
        // attention: 4*h*h*l / 2 (4-bit)
        // routed: 3 * e * h * me * l / 2
        // shared: 3 * h * se * l / 2
        // embed + lm_head: 2 * vocab * h / 2
        let attn = 4 * h * h * l / 2;
        let routed = 3 * e * h * me * l / 2;
        let shared = 3 * h * se * l / 2;
        let embed_head = 2 * (cfg.vocab_size as usize) * h / 2;
        attn + routed + shared + embed_head
    }
}

impl Model for Qwen35MoeModel {
    fn make_cache(...) -> Result<Vec<LayerCache>> { ... }     // 与 dense 实现一致，只 cfg 来源不同
    fn forward_on(...) -> Result<Array> { ... }                // text.forward_on → slice last → lm_head
    fn batched_prefill(...) -> Result<Array> { ... }
    fn model_meta(&self) -> ModelMeta { ... }
    fn num_hidden_layers(&self) -> usize { self.config().num_hidden_layers as usize }
}
```

同样为 `Qwen35Model` 提供 `impl Model`（P5a sub-phase 完成），dense 既有 inherent VL 方法保留为 inherent（不进 trait）。

### 3.5 `qwen3_5_moe::DecoderLayerMoe`

```rust
// ironmlx/src/models/qwen3_5_moe/decoder_layer.rs
pub struct DecoderLayerMoe {
    input_layernorm: RmsNorm,
    attn: AttnPath,                          // 复用 nn::AttnPath enum (Full | Linear)
    post_attention_layernorm: RmsNorm,
    ffn: SparseMoeBlock,                     // ← 核心区别
    cfg: DecoderLayerMoeConfig,
}
```

`forward_on` 数据流与 dense `nn::DecoderLayer::forward_on` 完全相同，除了 `ffn` 用 `SparseMoeBlock::forward_on` 代替 `nn::Mlp::forward_on`。`from_loader` 处理 `{prefix}.mlp.experts.*` + `{prefix}.mlp.shared_expert.*` + `{prefix}.mlp.gate.*` 三套 key 命名。

### 3.6 `SparseMoeBlock` + `Router`

```rust
// ironmlx/src/models/qwen3_5_moe/sparse_moe.rs
pub struct SparseMoeBlock {
    router: Router,
    routed: RoutedExperts,
    shared: Mlp,                             // 复用 nn::Mlp (SwiGLU)
    num_experts_per_tok: i32,
    norm_topk_prob: bool,
}

pub struct Router {
    gate: Linear,                            // [num_experts, hidden]
}

pub struct RoutedExperts {
    gate_weight:  Array,                     // [E, moe_inter, hidden_packed]
    gate_scales:  Array,                     // [E, moe_inter, groups]
    gate_biases:  Option<Array>,
    up_weight:    Array,
    up_scales:    Array,
    up_biases:    Option<Array>,
    down_weight:  Array,                     // [E, hidden, moe_inter_packed]
    down_scales:  Array,
    down_biases:  Option<Array>,
    group_size:   i32,
    bits:         i32,
    num_experts:  i32,
}

impl SparseMoeBlock {
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: &SparseMoeBlockConfig) -> Result<Self> {
        // gate:   {prefix}.gate.weight  (router Linear)
        // routed: {prefix}.experts.gate_proj.weight + .scales + .biases
        //         {prefix}.experts.up_proj.weight   + .scales + .biases
        //         {prefix}.experts.down_proj.weight + .scales + .biases
        // shared: {prefix}.shared_expert.gate_proj + .up_proj + .down_proj
        ...
    }

    pub fn forward_on(&self, x: &Array, target: StreamOrDevice) -> Result<Array> {
        // (1) flatten [B, S, H] → [B*S, H]
        // (2) gate logits → softmax → topk(num_experts_per_tok) [+renorm]
        // (3) gather_qmm-based SwiGLU per (token, top-k expert) → weighted sum
        // (4) shared.forward_on(x) → element-wise add
        // (5) reshape [B*S, H] → [B, S, H]
        ...
    }
}
```

`forward_on` 内 gather_qmm 路径由 P5b 子任务调研后定。算法 reference 在 mlx-lm `qwen3_moe.py::Qwen3MoeSparseMoeBlock`，但**仅读不抄**。

### 3.7 `core::loader::Loader::sanitize` MoE 扩展

在 P5b plan 阶段拉取 `mlx-community/Qwen3.5-35B-A3B-4bit` snapshot 到 `~/.ironmlx/models/` 后，列出实际 expert tensor key 命名。预期模式：
- `model.layers.{i}.mlp.experts.gate_proj.weight`  shape `[256, 512, hidden_packed]`
- `model.layers.{i}.mlp.experts.gate_proj.scales`  shape `[256, 512, groups]`
- `model.layers.{i}.mlp.experts.gate_proj.biases`
- 同上 `up_proj` / `down_proj`
- `model.layers.{i}.mlp.shared_expert.{gate|up|down}_proj.weight/.scales/.biases`
- `model.layers.{i}.mlp.gate.weight` （router gate Linear，**非** experts.gate_proj）

sanitize 增量：
- **不做** stacked → per-expert 转置（fused 是性能路径）
- mtp.* / vision_tower.* / language_model.* prefix 处理沿用现有 sanitize 规则
- **`mtp.*` 触发器 + norm +1.0 shift 在 35B-A3B 上仍会触发**（因 `mtp_num_hidden_layers: 1`），P5b T0 任务必须验证此路径与 omlx 输出一致；若 MoE checkpoint 的 norm shift 与 dense 路径语义不同（如 MoE 不需要 +1.0），sanitize 增量需引入 model_type 条件分支
- `conv1d.weight` transpose_axes 仅对 hybrid linear-attn 层适用，dense MoE checkpoint 无此 conv1d 时不触发，应天然兼容

### 3.8 `core::memory_budget` MoE-aware

`ModelMeta` 字段不变（`num_hidden_layers / num_attention_heads / num_key_value_heads / hidden_size / head_dim / weight_bytes`），但 `Qwen35MoeModel::model_meta()` 调用 `approx_weight_bytes` MoE 公式（见 §3.4）。`kv_bytes_per_token` 公式天然 GQA-aware，**MoE 不影响 KV 部分**。

### 3.9 `GenerationStream` / `Scheduler` / `SchedulerActor` generic 化

```rust
// 改造前 / 改造后
pub struct GenerationStream<'m> {                pub struct GenerationStream<'m, M: Model> {
    model: &'m Qwen35Model,                          model: &'m M,
    ...                                              ...
}                                                }

pub struct Scheduler {                            pub struct Scheduler<M: Model> {
    ...                                              _marker: PhantomData<M>,
}                                                    ...
                                                 }

impl Scheduler {                                  impl<M: Model> Scheduler<M> {
    pub fn step(                                      pub fn step(
        &mut self,                                        &mut self,
        model: &Qwen35Model,                              model: &M,
    ) -> Result<...> { ... }                          ) -> Result<...> { ... }
                                                  }

pub struct SchedulerActor {                       pub struct SchedulerActor<M: Model> {
    model: Arc<Mutex<Qwen35Model>>,                   model: Arc<Mutex<M>>,
    ...                                              ...
}                                                 }
```

**VL 处理边界**（关键约束）：
- VL 方法（`forward_vl_chunk` / `batched_prefill_vl` / `compute_vision_embeds`）仅保留在 dense `Qwen35Model` 的 inherent impl 上，**不进 `trait Model`**
- P5a trait 化期间，scheduler/server 内**走 VL code path 的代码段保持现状**（继续硬绑 `&Qwen35Model`），不改其类型签名；仅 LM text-only path 的方法（forward_on / batched_prefill / make_cache / model_meta / step / prefill_admitted）改 generic over `M: Model`
- 调用 VL code path 的入口（如 multipart HTTP handler）需要在 P5a 后被 Qwen35Model 专属代码段守护，例如通过两个并存的 SchedulerActor instantiation 或者在 model_type 分发处把 VL endpoint 完全屏蔽给 MoE 走（**P5c plan 阶段确认具体实现**，spec 不固化）
- 后续 P6.x 引入 VL phase 时再统一抽象（如 `trait MultimodalModel: Model`），P5 不预先抽象

### 3.10 CLI dispatch by `model_type`

```rust
// ironmlx/src/cli/generate.rs (改造后)
fn run_with_loader(loader: Loader, args: GenerateArgs, tokenizer: Tokenizer) -> Result<()> {
    let model_type = loader.config_raw_value()
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("config.json missing model_type"))?;

    match model_type {
        "qwen3_5" => {
            let model = Qwen35Model::from_loader(&loader)?;
            run_generation_with_model(&model, &tokenizer, args)
        }
        "qwen3_5_moe" => {
            let model = Qwen35MoeModel::from_loader(&loader)?;
            run_generation_with_model(&model, &tokenizer, args)
        }
        other => Err(anyhow!("unsupported model_type: {other}")),
    }
}

fn run_generation_with_model<M: Model>(
    model: &M, tokenizer: &Tokenizer, args: GenerateArgs,
) -> Result<()> { ... }
```

`cli/serve.rs` 同模式；`SchedulerActor` 的 spawn 路径也按 `model_type` 分发到对应 generic instantiation。

### 3.11 文件结构总览

```
ironmlx/src/
├─ core/
│  ├─ model.rs                   # 新增 trait Model
│  ├─ loader.rs                  # sanitize 增加 MoE expert key 兼容
│  ├─ memory_budget.rs           # 通用，不动；MoE model 自己实现 approx_weight_bytes
│  ├─ generate.rs                # GenerationStream<'m, M: Model>
│  ├─ scheduler.rs               # Scheduler<M: Model>
│  └─ server/
│     ├─ mod.rs                  # AppState<M: Model>
│     └─ scheduler_actor.rs      # SchedulerActor<M: Model>
├─ models/
│  ├─ mod.rs                     # 解开 pub mod qwen3_5_moe
│  ├─ qwen3_5/                   # 不动
│  └─ qwen3_5_moe/               # 新增
│     ├─ mod.rs
│     ├─ config.rs               # Qwen35MoeConfig
│     ├─ model.rs                # Qwen35MoeModel + impl Model
│     ├─ text_model.rs           # Qwen35MoeTextModel
│     ├─ decoder_layer.rs        # DecoderLayerMoe（独立 copy，ffn 字段 = SparseMoeBlock）
│     └─ sparse_moe.rs           # SparseMoeBlock + Router + RoutedExperts
└─ cli/
   ├─ generate.rs                # model_type dispatch
   └─ serve.rs                   # model_type dispatch
```

## § 4 测试策略

### 4.1 单元测试（CI 跑，无外部 checkpoint）

| 测试模块 | 覆盖 |
|---|---|
| `core::model::tests` | Trait 自身可见性 + Send + Sync 边界（如适用） |
| `models::qwen3_5_moe::config::tests` | Qwen35MoeConfig 解析真实 35B-A3B config.json subset；num_experts/num_experts_per_tok/moe_intermediate_size 字段读取 |
| `models::qwen3_5_moe::sparse_moe::tests` | Router shape; topk + renorm 数值；shared+routed 求和（用 stub 权重） |
| `models::qwen3_5_moe::model::tests` | `from_cfg_for_test` stub + `make_cache` layer kind 分布（同 dense 既有测试模式） |
| `models::qwen3_5_moe::sparse_moe::tests::sparse_moe_zero_routed_yields_shared` | `gate=0` weights 退化为纯 shared 路径 |

### 4.2 集成测试（`#[ignore]`，本地真实 checkpoint）

| 测试 | 验证目标 |
|---|---|
| `tests/p5_qwen35_moe_logits_match.rs` | 与 omlx CLI 同 prompt + 同 sampler 配置下 greedy argmax 100% 一致；top-K logits max_abs_diff < 1e-3 |
| `tests/p5_qwen35_moe_http_smoke.rs` | HTTP serve 完整 prompt → token stream，无 OOM/panic |
| `tests/p5_qwen35_moe_batched_prefill.rs` | B=2/B=4 batched prefill 与 B=1 single-stream 等价（沿用现有 dense 等价测试模式） |
| `tests/p5_trait_dense_regression.rs` | P5a 完成后跑全套 dense 既有集成测试不退化 |

### 4.3 性能 gate（P5d）

| 指标 | 基线 | 验收 |
|---|---|---|
| iron-bench `qwen3.5-moe` profile prefill PP=2048 | 待 P5b 实测落定 | 与 omlx CLI 同硬件下相对差 < 30% 即过 |
| iron-bench decode steady ITL | 待落定 | 与 omlx 同上 |
| HTTP smoke p99 latency | 待落定 | 与 dense 4B 同协议 |

性能 gate 阈值在 P5b 拿到第一个可跑数后定，不在 spec 提前固化（避免根据现实调整 spec 的反馈环路）。

## § 5 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `mlx::quantization::gather_qmm` 不暴露 | SparseMoeBlock forward 必须走 per-expert scatter + qmm，性能严重退化 | P5b T0 任务读 `/Users/xin/workspace/iron-rivals/mlx` headers 决定；fallback 路径设计可接受性能损失但功能完整 |
| HF expert key 命名实际 ≠ `experts.{gate,up,down}_proj` | sanitize 必须调整；可能涉及 per-expert ↔ stacked 转换 | P5b T0 同时拉取 snapshot 列 keys；如需大量转换则提早 surface 给 Boss |
| 35B-A3B-4bit 在 M1 Pro 32GB 实际 OOM | 长 context decode 失败 | memory_budget MoE 公式准确预估；admit gate 在启动时拒绝超额 b_max × cap_max；若硬件不够留 close-out 报告说明 |
| Trait 化 4500 行 scheduler 引入回归 | dense 路径性能 / 行为退化 | P5a 唯一验收点：dense 既有所有 test 通过 + perf gate 不退化；任何回归 P5a 不闭环 |
| omlx CLI 未支持 35B-A3B-4bit | 失去主验证基线 | P5b T0 验证 omlx 可加载 35B-A3B；若不支持，退路用 mlx-lm 作为唯一基线（仍只读不抄） |
| `norm_topk_prob` 默认值 / softmax 顺序与 mlx-lm 实现不一致 | logits 数值与 omlx 偏差超阈值 | P5b T1 读 mlx-lm + omlx 同时确认；spec 不预先固化公式 |

## § 6 实施任务划分（给 writing-plans）

按 4 sub-phase 拆分，每个 sub-phase 独立 plan，每个 plan 5-7 个 task（遵守 [feedback_task_breakdown_bounded](../../../../.claude/projects/-Users-xin-workspace-ironmlx-backend/memory/feedback_task_breakdown_bounded.md) 约束，不再细拆 sub-task）。

| Sub-phase | Plan 文件 | 核心目标 | 关键 task 数 |
|---|---|---|---|
| **P5a** Trait Refactor | `2026-05-18-ironmlx-p5a-trait-refactor.md` | 定义 trait Model；Qwen35Model 实现；Scheduler/GenerationStream/SchedulerActor/AppState 改 generic；CLI 不变（仍硬接 dense） | 5-7 |
| **P5b** MoE Forward | `2026-05-18-ironmlx-p5b-moe-forward.md` | 拉 snapshot；读 mlx headers 决定 gather_qmm 路径；Qwen35MoeConfig + Qwen35MoeTextModel + Qwen35MoeModel + SparseMoeBlock + impl Model；loader sanitize MoE 兼容；首个单元测试与 omlx 单 prompt argmax 对齐 | 5-7 |
| **P5c** Scheduler/Server 集成 | `2026-05-18-ironmlx-p5c-scheduler-integration.md` | CLI dispatch by model_type；SchedulerActor MoE spawn；memory_budget MoE 公式；HTTP smoke；batched_prefill MoE 路径 | 5-7 |
| **P5d** Perf Gate + Validation | `2026-05-18-ironmlx-p5d-perf-validation.md` | iron-bench MoE profile；omlx 跨 prompt 对齐；mlx-lm 抽样对齐；perf 基线落定；close-out 报告 | 5-7 |

任意 sub-phase 不闭环则后续不启动。

## § 7 验收标准

**P5 整体验收**（4 sub-phase 全部 closeout 后）：
- `mlx-community/Qwen3.5-35B-A3B-4bit` 通过 `ironmlx generate` 与 `ironmlx serve` CLI 跑通 text-only generation
- 与 omlx CLI 同 prompt greedy 输出 token 序列 100% 一致（至少 50 prompt 抽样 + 至少 200 token decode 长度）
- top-K logits max_abs_diff < 1e-3（与 omlx）
- dense Qwen3.5-4B 路径所有现有 unit + integration test 通过，perf 不退化
- iron-bench 35B-A3B-4bit profile 录入历史基线
- close-out 报告写明：实际 prefill PP / decode ITL / 内存峰值 / 与 omlx 相对性能 / 已知问题

**P5 整体 out-of-scope**（出现于实施过程中需求要立即拒绝并 Boss 决策是否新开 phase）：
- VL inference（pixel_values / vision_tower / cross_modal）
- MoE-aware MTP
- Expert offloading / paging
- 自写 fused Metal expert kernel
- 跨芯片 tile tuning
- norm_topk_prob 决策被推翻为运行时可配
