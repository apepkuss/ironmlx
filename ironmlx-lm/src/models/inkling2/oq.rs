//! Adapt the packed oQ checkpoint layout to the native Inkling model.
use super::config::Inkling2Config;
use crate::core::weights::QuantMeta;
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::collections::HashMap;

type Tensors = HashMap<String, Array>;

fn take(tensors: &mut Tensors, key: &str) -> Result<Array> {
    tensors
        .remove(key)
        .with_context(|| format!("missing oQ tensor {key}"))
}

fn insert(tensors: &mut Tensors, key: String, value: Array) -> Result<()> {
    ensure!(
        !tensors.contains_key(&key),
        "duplicate Inkling tensor {key}"
    );
    tensors.insert(key, value);
    Ok(())
}

fn rename(tensors: &mut Tensors, old: &str, new: String) -> Result<()> {
    let value = take(tensors, old)?;
    insert(tensors, new, value)
}

fn rename_projection(
    tensors: &mut Tensors,
    overrides: &mut HashMap<String, QuantMeta>,
    old: &str,
    new: String,
    rows: i32,
) -> Result<()> {
    let quant = overrides
        .remove(old)
        .with_context(|| format!("missing oQ projection quantization {old}"))?;
    for suffix in ["weight", "scales", "biases"] {
        let array = take(tensors, &format!("{old}.{suffix}"))?;
        ensure!(
            array.ndim() == 2 && array.shape_at(0) == rows,
            "{old}.{suffix}: expected {rows} output rows"
        );
        insert(tensors, format!("{new}.{suffix}"), array)?;
    }
    overrides.insert(new, quant);
    Ok(())
}

pub(crate) fn normalize(
    tensors: &mut Tensors,
    overrides: &mut HashMap<String, QuantMeta>,
    config: &Inkling2Config,
    keep_vision: bool,
) -> Result<()> {
    if !tensors.contains_key("model.embed.weight") {
        return Ok(());
    }
    let t = StreamOrDevice::default();
    let c = &config.text_config;
    let prefix = "model.language_model";
    rename_projection(
        tensors,
        overrides,
        "model.embed",
        format!("{prefix}.embed_tokens"),
        c.vocab_size,
    )?;
    for key in ["embed_norm.weight", "norm.weight", "final_input.query"] {
        rename(tensors, &format!("model.{key}"), format!("{prefix}.{key}"))?;
    }
    let mut message_queries = Vec::new();
    for i in 0..c.num_hidden_layers {
        let p = format!("{prefix}.layers.{i}");
        let src = format!("model.layers.{i}");
        let (heads, kv, dim) = c.attention_dims(i);
        let local = c.is_local(i);
        let dq = heads * dim;
        let dkv = kv * dim;
        let attn = |name: &str| format!("{p}.self_attn.{name}");
        if local {
            rename_projection(
                tensors,
                overrides,
                &format!("{src}.attn.in_proj"),
                attn("packed_input"),
                2 * dq + 2 * dkv + heads * (c.d_rel + 1),
            )?;
            rename(
                tensors,
                &format!("{src}.attn_norm.weight"),
                format!("{p}.input_layernorm.weight"),
            )?;
        } else {
            rename_projection(
                tensors,
                overrides,
                &format!("{src}.attn.q_proj"),
                attn("packed_query"),
                2 * dq + heads * (c.d_rel + 1),
            )?;
            for name in ["k", "v"] {
                rename_projection(
                    tensors,
                    overrides,
                    &format!("{src}.attn.{name}_proj"),
                    attn(&format!("packed_{name}")),
                    dkv + kv * c.sconv_kernel_size,
                )?;
            }
            for name in ["q", "k", "v"] {
                rename(
                    tensors,
                    &format!("{src}.{name}_input_norm.weight"),
                    format!("{p}.{name}_input_layernorm.weight"),
                )?;
            }
        }
        rename_projection(
            tensors,
            overrides,
            &format!("{src}.attn.o_proj"),
            attn("o_proj"),
            c.hidden_size,
        )?;
        for name in ["k", "v"] {
            let suffix = if local { "conv" } else { "static" };
            let conv = take(tensors, &format!("{src}.attn.{name}_{suffix}"))?;
            let (key, value) = if local {
                ("conv1d.weight", conv.expand_dims_on(-1, t)?)
            } else {
                ("static_weight", conv)
            };
            insert(tensors, format!("{p}.self_attn.{name}_sconv.{key}"), value)?;
        }
        for (old, new) in [
            ("attn.q_scale", "self_attn.q_l2_clip.scale"),
            ("attn.r_scale", "self_attn.rel_l2_clip.scale"),
            ("attn.rel_proj", "self_attn.rel_logits_proj.proj"),
            ("mlp_norm.weight", "post_attention_layernorm.weight"),
            ("mlp.router_weight", "mlp.gate.weight"),
            ("mlp.router_bias", "mlp.gate.e_score_correction_bias"),
            ("mlp_input.query", "mlp_input.query"),
        ] {
            rename(tensors, &format!("{src}.{old}"), format!("{p}.{new}"))?;
        }
        if i != 0 {
            rename(
                tensors,
                &format!("{src}.attn_input.query"),
                format!("{p}.attention_input.query"),
            )?;
        }
        for (old, new) in [
            ("attn_conv", "post_attention_convolution"),
            ("mlp_conv", "post_mlp_convolution"),
        ] {
            let conv = take(tensors, &format!("{src}.{old}"))?;
            insert(
                tensors,
                format!("{p}.{new}.conv1d.weight"),
                conv.expand_dims_on(-1, t)?,
            )?;
        }
        for (old, new) in [("w13", "gate_up_proj"), ("w2", "down_proj")] {
            let old = format!("{src}.mlp.experts.{old}");
            let new = format!("{p}.mlp.experts.{new}");
            for suffix in ["weight", "scales", "biases", "global_scale"] {
                if let Some(value) = tensors.remove(&format!("{old}.{suffix}")) {
                    insert(tensors, format!("{new}.{suffix}"), value)?;
                }
            }
            if let Some(quant) = overrides.remove(&old) {
                overrides.insert(new, quant);
            }
        }
        if let Some(loop_config) = c
            .loop_config
            .as_ref()
            .filter(|l| (l.core_start..l.core_end).contains(&i))
        {
            message_queries.push(take(tensors, &format!("{src}.loop_message.query"))?);
            let down = take(tensors, &format!("{src}.loop_down"))?;
            let gain_forget = take(tensors, &format!("{src}.loop_gain_forget"))?;
            ensure!(
                down.shape().as_slice() == [4, 2, c.hidden_size, loop_config.gate_rank]
                    && gain_forget.shape().as_slice()
                        == [4, loop_config.gate_rank, 2 * c.hidden_size],
                "{src}: invalid loop controller layout"
            );
            insert(
                tensors,
                format!("{p}.packed_loop_down"),
                down.reshape_on((4, 2 * c.hidden_size, loop_config.gate_rank), t)?
                    .astype_on(Dtype::Bfloat16, t)?,
            )?;
            insert(tensors, format!("{p}.packed_loop_out"), gain_forget)?;
        }
    }
    if !message_queries.is_empty() {
        let refs: Vec<_> = message_queries.iter().collect();
        insert(
            tensors,
            format!("{prefix}.message_input.query"),
            mlx::ops::shape::stack_on(&refs, 0, t)?,
        )?;
    }

    let vision: Vec<_> = tensors
        .keys()
        .filter(|key| key.starts_with("vision_tower."))
        .cloned()
        .collect();
    for key in vision {
        let value = take(tensors, &key)?;
        if !keep_vision {
            continue;
        }
        let relative = key.trim_start_matches("vision_tower.");
        let relative = match relative {
            "backbone.embeddings.position_embedding" => {
                "embeddings.position_embedding.weight".to_owned()
            }
            _ if relative.starts_with("proj_in.") => {
                relative.replace("proj_in.", "merger.linear_1.")
            }
            _ if relative.starts_with("proj_out.") => {
                relative.replace("proj_out.", "merger.linear_2.")
            }
            _ => relative
                .trim_start_matches("backbone.")
                .replace("ln_pre.", "pre_layernorm.")
                .replace(".ln_1.", ".layernorm_before.")
                .replace(".ln_2.", ".layernorm_after.")
                .replace(".ls_1.gamma", ".lambda_1")
                .replace(".ls_2.gamma", ".lambda_2")
                .replace(".mlp.c_fc.", ".mlp.fc1.")
                .replace(".mlp.c_proj.", ".mlp.fc2."),
        };
        insert(tensors, format!("model.vision_tower.{relative}"), value)?;
    }
    Ok(())
}
