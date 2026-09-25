use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ironmlx_core::nn::LayerNorm;
use ironmlx_core::weights::{WeightMap, WeightSource};
use mlx::compile::{compile, CompiledFn, ShapeMode};
use mlx::{fast, ops, Array, Dtype};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct EncoderConfig {
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    norm_eps: f32,
    local_attention: i32,
    layer_types: Vec<String>,
    rope_parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    head_layers: usize,
    max_len: usize,
    head_max_len: usize,
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: std::collections::HashMap<String, f32>,
}

pub(super) struct NativeModel {
    gelu: CompiledFn,
    embedding: Array,
    embed_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    head_layers: Vec<HeadLayer>,
    type_emb: Array,
    scorer_norm: LayerNorm,
    scorer_in: Linear,
    scorer_out: Linear,
    head_dim: i32,
    num_heads: i32,
    local_attention: i32,
    pub(super) max_len: usize,
    pub(super) head_max_len: usize,
    pub(super) temperature: Vec<f32>,
    pub(super) temperature_by_options: std::collections::HashMap<String, f32>,
}

enum Forward {
    Eager(Box<NativeModel>),
    Compiled(CompiledFn),
}

pub(super) struct NativeInference {
    forward: Forward,
    local_attention: i32,
    pub(super) max_len: usize,
    pub(super) head_max_len: usize,
    pub(super) temperature: Vec<f32>,
    pub(super) temperature_by_options: std::collections::HashMap<String, f32>,
}

impl NativeInference {
    pub(super) fn load(dir: &Path, dtype: Dtype, compiled: bool) -> Result<Self> {
        let model = NativeModel::load(dir, dtype)?;
        let metadata = (
            model.local_attention,
            model.max_len,
            model.head_max_len,
            model.temperature.clone(),
            model.temperature_by_options.clone(),
        );
        let forward = if compiled {
            Forward::Compiled(compile(
                move |inputs| {
                    model
                        .forward(inputs)
                        .map(|array| vec![array])
                        .map_err(|e| mlx::Error::Mlx(e.to_string()))
                },
                ShapeMode::Fixed,
            )?)
        } else {
            Forward::Eager(Box::new(model))
        };
        Ok(Self {
            forward,
            local_attention: metadata.0,
            max_len: metadata.1,
            head_max_len: metadata.2,
            temperature: metadata.3,
            temperature_by_options: metadata.4,
        })
    }
}

// Match mlx.nn.Linear's fused bias addition, including CPU FP16 rounding.
struct Linear {
    weight: Array,
    bias: Option<Array>,
}
impl Linear {
    fn from_loader(weights: &WeightMap, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(&format!("{prefix}.weight"))?.clone(),
            bias: weights.tensor_opt(&format!("{prefix}.bias")).cloned(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let transposed = self.weight.transpose()?;
        Ok(match &self.bias {
            Some(bias) => ops::matmul::addmm(bias, x, &transposed, 1.0, 1.0)?,
            None => x.matmul(&transposed)?,
        })
    }
}

struct EncoderLayer {
    attn_norm: Option<LayerNorm>,
    qkv: Linear,
    out: Linear,
    mlp_norm: LayerNorm,
    wi: Linear,
    wo: Linear,
    attention_type: String,
    rope_theta: f32,
}

struct HeadLayer {
    norm1: LayerNorm,
    qkv: Linear,
    out: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

fn norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        weights.tensor(&format!("{prefix}.weight"))?.clone(),
        weights.tensor_opt(&format!("{prefix}.bias")).cloned(),
        eps,
    ))
}

impl NativeModel {
    pub(super) fn load(dir: &Path, dtype: Dtype) -> Result<Self> {
        let encoder: EncoderConfig =
            serde_json::from_reader(std::fs::File::open(dir.join("encoder/config.json"))?)?;
        let agent: AgentConfig =
            serde_json::from_reader(std::fs::File::open(dir.join("rl_agent_config.json"))?)?;
        if encoder.model_type != "modernbert"
            || encoder.hidden_size <= 0
            || encoder.intermediate_size <= 0
            || encoder.num_attention_heads <= 0
            || encoder.hidden_size % encoder.num_attention_heads != 0
            || encoder.layer_types.len() != encoder.num_hidden_layers
            || agent.head_layers != 2
            || agent.temperature.len() != 3
            || !(4 < agent.head_max_len
                && agent.head_max_len < agent.max_len
                && agent.max_len <= 8192)
        {
            bail!("unsupported Laya encoder or decision-head configuration");
        }
        let head_dim = encoder.hidden_size / encoder.num_attention_heads;
        if head_dim % 2 != 0 {
            bail!("Laya attention head dimension must be even");
        }
        let (tensors, _) = mlx::io::load_safetensors(
            dir.join("model.safetensors")
                .to_str()
                .ok_or_else(|| anyhow!("non-UTF8 model path"))?,
        )
        .context("loading Laya safetensors")?;
        if tensors.len() != 170 {
            bail!("expected 170 Laya tensors, found {}", tensors.len());
        }
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.astype(dtype)?)))
            .collect::<Result<_>>()?;
        let weights = WeightMap::new(tensors, None, Default::default());
        let embedding = weights
            .tensor("encoder.embeddings.tok_embeddings.weight")?
            .clone();
        if embedding.shape().as_slice() != [encoder.vocab_size, encoder.hidden_size] {
            bail!("Laya embedding shape disagrees with encoder config");
        }
        let mut layers = Vec::with_capacity(encoder.num_hidden_layers);
        for (index, kind) in encoder.layer_types.iter().enumerate() {
            if kind != "full_attention" && kind != "sliding_attention" {
                bail!("unsupported Laya attention type {kind}");
            }
            let theta = encoder
                .rope_parameters
                .get(kind)
                .and_then(|v| v.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| anyhow!("missing RoPE theta for {kind}"))?
                as f32;
            let prefix = format!("encoder.layers.{index}");
            let wi = Linear::from_loader(&weights, &format!("{prefix}.mlp.Wi"))?;
            let wo = Linear::from_loader(&weights, &format!("{prefix}.mlp.Wo"))?;
            if weights
                .tensor(&format!("{prefix}.mlp.Wi.weight"))?
                .shape()
                .as_slice()
                != [2 * encoder.intermediate_size, encoder.hidden_size]
                || weights
                    .tensor(&format!("{prefix}.mlp.Wo.weight"))?
                    .shape()
                    .as_slice()
                    != [encoder.hidden_size, encoder.intermediate_size]
            {
                bail!("Laya encoder MLP shape disagrees with config at layer {index}");
            }
            layers.push(EncoderLayer {
                attn_norm: (index != 0)
                    .then(|| norm(&weights, &format!("{prefix}.attn_norm"), encoder.norm_eps))
                    .transpose()?,
                qkv: Linear::from_loader(&weights, &format!("{prefix}.attn.Wqkv"))?,
                out: Linear::from_loader(&weights, &format!("{prefix}.attn.Wo"))?,
                mlp_norm: norm(&weights, &format!("{prefix}.mlp_norm"), encoder.norm_eps)?,
                wi,
                wo,
                attention_type: kind.clone(),
                rope_theta: theta,
            });
        }
        let mut head_layers = Vec::with_capacity(agent.head_layers);
        for index in 0..agent.head_layers {
            let prefix = format!("head.layers.{index}");
            head_layers.push(HeadLayer {
                norm1: norm(&weights, &format!("{prefix}.norm1"), 1e-5)?,
                qkv: Linear::from_loader(&weights, &format!("{prefix}.self_attn.in_proj"))?,
                out: Linear::from_loader(&weights, &format!("{prefix}.self_attn.out_proj"))?,
                norm2: norm(&weights, &format!("{prefix}.norm2"), 1e-5)?,
                linear1: Linear::from_loader(&weights, &format!("{prefix}.linear1"))?,
                linear2: Linear::from_loader(&weights, &format!("{prefix}.linear2"))?,
            });
        }
        Ok(Self {
            gelu: build_gelu(dtype)?,
            embedding,
            embed_norm: norm(&weights, "encoder.embeddings.norm", encoder.norm_eps)?,
            layers,
            final_norm: norm(&weights, "encoder.final_norm", encoder.norm_eps)?,
            head_layers,
            type_emb: weights.tensor("type_emb.weight")?.clone(),
            scorer_norm: norm(&weights, "scorer.layers.0", 1e-5)?,
            scorer_in: Linear::from_loader(&weights, "scorer.layers.1")?,
            scorer_out: Linear::from_loader(&weights, "scorer.layers.3")?,
            head_dim,
            num_heads: encoder.num_attention_heads,
            local_attention: encoder.local_attention,
            max_len: agent.max_len,
            head_max_len: agent.head_max_len,
            temperature: agent.temperature,
            temperature_by_options: agent.temperature_by_options,
        })
    }

    fn gelu(&self, x: &Array) -> Result<Array> {
        Ok(self.gelu.invoke(&[x])?.remove(0))
    }

    fn forward(&self, inputs: &[&Array]) -> Result<Array> {
        let [token_ids, full_mask, local_mask, qtype_array, index] = inputs else {
            bail!("Laya forward expects five input tensors");
        };
        let batch = token_ids.shape().as_slice()[0];
        let length = token_ids.shape().as_slice()[1];
        let mut hidden = self
            .embed_norm
            .forward(&self.embedding.take(token_ids, 0)?)?;
        for layer in &self.layers {
            let normalized = match &layer.attn_norm {
                Some(norm) => norm.forward(&hidden)?,
                None => hidden.clone(),
            };
            let mask = if layer.attention_type == "sliding_attention" {
                Some(*local_mask)
            } else {
                Some(*full_mask)
            };
            let attended = attention(
                &normalized,
                &layer.qkv,
                &layer.out,
                self.num_heads,
                self.head_dim,
                Some(layer.rope_theta),
                mask,
            )?;
            hidden = &hidden + &attended;
            let mlp = layer.mlp_norm.forward(&hidden)?;
            let chunks = ops::shape::split_n(&layer.wi.forward(&mlp)?, 2, -1)?;
            let gated = &self.gelu(&chunks[0])? * &chunks[1];
            hidden = &hidden + &layer.wo.forward(&gated)?;
        }
        hidden = self.final_norm.forward(&hidden)?;
        let type_embedding = self.type_emb.take(qtype_array, 0)?.reshape((
            batch,
            1,
            self.num_heads * self.head_dim,
        ))?;
        hidden = &hidden + &type_embedding;
        for layer in &self.head_layers {
            let normalized = layer.norm1.forward(&hidden)?;
            hidden = &hidden
                + &attention(
                    &normalized,
                    &layer.qkv,
                    &layer.out,
                    self.num_heads,
                    self.head_dim,
                    None,
                    Some(*full_mask),
                )?;
            let normalized = layer.norm2.forward(&hidden)?;
            let activated = relu(&layer.linear1.forward(&normalized)?)?;
            hidden = &hidden + &layer.linear2.forward(&activated)?;
        }
        let selected = hidden
            .reshape((batch * length, self.num_heads * self.head_dim))?
            .take(index, 0)?
            .reshape((batch, -1, self.num_heads * self.head_dim))?;
        let normalized = self.scorer_norm.forward(&selected)?;
        let scored = self
            .scorer_out
            .forward(&self.gelu(&self.scorer_in.forward(&normalized)?)?)?;
        Ok(scored)
    }
}

impl NativeInference {
    pub(super) fn logits_batch(
        &self,
        rows: &[super::prompt::PreparedQuestion],
        qtypes: &[u32],
        pad: u32,
        pad_to_multiple: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = rows.len() as i32;
        let longest = rows.iter().map(|row| row.ids.len()).max().unwrap_or(0);
        let length = pad_to_multiple
            .map_or(longest, |n| longest.div_ceil(n) * n)
            .min(self.max_len) as i32;
        let mut ids = vec![pad; batch as usize * length as usize];
        let mut full_mask = Vec::new();
        let mut local_mask_values = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            ids[index * length as usize..index * length as usize + row.ids.len()]
                .copy_from_slice(&row.ids);
            for q in 0..length {
                for k in 0..length {
                    let valid = (k as usize) < row.ids.len();
                    if q == 0 {
                        full_mask.push(valid);
                    }
                    local_mask_values.push(
                        (valid && (q - k).abs() <= self.local_attention / 2)
                            || ((q as usize) >= row.ids.len() && k == q),
                    );
                }
            }
        }
        let token_ids: Array = (&ids[..], (batch, length)).try_into()?;
        let full_mask: Array = (&full_mask[..], (batch, 1, 1, length)).try_into()?;
        let local_mask: Array = (&local_mask_values[..], (batch, 1, length, length)).try_into()?;
        let marker_count = rows
            .iter()
            .map(|row| row.markers.len())
            .max()
            .unwrap_or(2)
            .max(2);
        let positions: Vec<u32> = rows
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                (0..marker_count).map(move |m| {
                    (i * length as usize + row.markers.get(m).copied().unwrap_or(0)) as u32
                })
            })
            .collect();
        let index: Array = (&positions[..], (positions.len() as i32,)).try_into()?;
        let qtype_array: Array = (qtypes, (batch,)).try_into()?;
        let inputs = [&token_ids, &full_mask, &local_mask, &qtype_array, &index];
        let scored = match &self.forward {
            Forward::Eager(model) => model.forward(&inputs)?,
            Forward::Compiled(function) => function.invoke(&inputs)?.remove(0),
        };
        let values = scored
            .astype(Dtype::Float32)?
            .reshape((-1,))?
            .to_vec::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            bail!("Laya returned non-finite decision logits");
        }
        let mut offset = 0;
        Ok(rows
            .iter()
            .map(|row| {
                let result = values[offset..offset + row.markers.len()].to_vec();
                offset += marker_count;
                result
            })
            .collect())
    }
}

fn attention(
    x: &Array,
    qkv: &Linear,
    out: &Linear,
    heads: i32,
    head_dim: i32,
    theta: Option<f32>,
    mask: Option<&Array>,
) -> Result<Array> {
    let batch = x.shape().as_slice()[0];
    let length = x.shape().as_slice()[1];
    let projected = qkv
        .forward(x)?
        .reshape(&[batch, length, 3, heads, head_dim][..])?;
    let pieces = ops::shape::split_n(&projected, 3, 2)?;
    let mut q = pieces[0].squeeze(2)?.transpose_axes((0, 2, 1, 3))?;
    let mut k = pieces[1].squeeze(2)?.transpose_axes((0, 2, 1, 3))?;
    let v = pieces[2].squeeze(2)?.transpose_axes((0, 2, 1, 3))?;
    if let Some(theta) = theta {
        q = fast::rope(&q, head_dim, false, Some(theta), 1.0, 0, None)?;
        k = fast::rope(&k, head_dim, false, Some(theta), 1.0, 0, None)?;
    }
    let attended = fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        (head_dim as f32).powf(-0.5),
        "",
        mask,
        None,
    )?;
    out.forward(&attended.transpose_axes((0, 2, 1, 3))?.reshape((
        batch,
        length,
        heads * head_dim,
    ))?)
}

fn dtype_scalar(x: &Array, value: f32) -> Result<Array> {
    let scalar: Array = (&[value][..], ()).try_into()?;
    Ok(scalar.astype(x.dtype())?)
}

fn build_gelu(dtype: Dtype) -> Result<CompiledFn> {
    // Python weak scalars are materialized in the input dtype before tracing.
    // Materialize them here too: a lazy FP32 -> FP16 cast inside a compiled
    // graph can otherwise fuse with the division and change FP16 rounding.
    let scalar = |value: f32| -> Result<Array> {
        let value: Array = (&[value][..], ()).try_into()?;
        let value = value.astype(dtype)?;
        value.eval()?;
        Ok(value)
    };
    let two = scalar(2.0)?;
    let one = scalar(1.0)?;
    let sqrt_two = scalar(std::f32::consts::SQRT_2)?;
    Ok(compile(
        move |inputs| {
            let x = inputs[0];
            Ok(vec![x * (&one + &(x / &sqrt_two).erf()?) / &two])
        },
        ShapeMode::Shapeless,
    )?)
}

fn relu(x: &Array) -> Result<Array> {
    Ok(x.maximum(&dtype_scalar(x, 0.0)?)?)
}
