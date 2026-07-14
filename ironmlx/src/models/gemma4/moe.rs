use anyhow::{anyhow, Context};
use mlx::ops::indexing::{slice_on, take_along_axis_on, take_on};
use mlx::ops::sort::argpartition_on;
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::models::qwen3_5_moe::RoutedExperts;
use crate::nn::{Linear, RmsNorm};
use crate::Result;

use super::config::{Gemma4LayerKind, Gemma4TextConfig};
use super::mlp::Gemma4GeGluMlp;
use super::profile;

pub(crate) enum Gemma4FeedForward {
    Dense(Gemma4GeGluMlp),
    Moe(Box<Gemma4MoeBlock>),
}

pub(crate) struct Gemma4MoeBlock {
    dense_mlp: Gemma4GeGluMlp,
    post_dense_norm: RmsNorm,
    router: Gemma4Router,
    pre_expert_norm: RmsNorm,
    experts: RoutedExperts,
    post_expert_norm: RmsNorm,
    layer_idx: usize,
    layer_kind: Gemma4LayerKind,
}

struct Gemma4Router {
    proj: Linear,
    scale: Array,
    per_expert_scale: Array,
    hidden_root: f32,
    eps: f32,
    top_k: i32,
    num_experts: i32,
}

impl Gemma4FeedForward {
    pub(crate) fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        layer_kind: Gemma4LayerKind,
        dense_intermediate_size: i32,
    ) -> Result<Self> {
        if cfg.enable_moe_block {
            Ok(Self::Moe(Box::new(Gemma4MoeBlock::from_loader(
                loader,
                prefix,
                cfg,
                layer_idx,
                layer_kind,
                dense_intermediate_size,
            )?)))
        } else {
            Ok(Self::Dense(Gemma4GeGluMlp::from_loader(
                loader,
                &format!("{prefix}.mlp"),
                dense_intermediate_size,
                layer_idx,
                layer_kind,
            )?))
        }
    }

    pub(crate) fn forward_on(
        &self,
        hidden: &Array,
        pre_feedforward_layernorm: &RmsNorm,
        target: StreamOrDevice,
    ) -> Result<Array> {
        match self {
            Self::Dense(mlp) => {
                let ffn = pre_feedforward_layernorm.forward_on(hidden, target)?;
                mlp.forward_on(&ffn, target)
            }
            Self::Moe(moe) => moe.forward_on(hidden, pre_feedforward_layernorm, target),
        }
    }
}

impl Gemma4MoeBlock {
    fn from_loader(
        loader: &Loader,
        prefix: &str,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        layer_kind: Gemma4LayerKind,
        dense_intermediate_size: i32,
    ) -> Result<Self> {
        let dense_mlp = Gemma4GeGluMlp::from_loader(
            loader,
            &format!("{prefix}.mlp"),
            dense_intermediate_size,
            layer_idx,
            layer_kind,
        )
        .with_context(|| format!("loading Gemma4 MoE dense branch at `{prefix}`"))?;
        let post_dense_norm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_feedforward_layernorm_1"),
            cfg.rms_norm_eps,
        )
        .with_context(|| format!("loading Gemma4 MoE dense post norm at `{prefix}`"))?;
        let router = Gemma4Router::from_loader(loader, &format!("{prefix}.router"), cfg)
            .with_context(|| format!("loading Gemma4 MoE router at `{prefix}`"))?;
        let pre_expert_norm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.pre_feedforward_layernorm_2"),
            cfg.rms_norm_eps,
        )
        .with_context(|| format!("loading Gemma4 MoE expert pre norm at `{prefix}`"))?;
        let experts =
            RoutedExperts::from_loader(loader, &format!("{prefix}.experts.switch_glu"))
                .with_context(|| format!("loading Gemma4 MoE routed experts at `{prefix}`"))?;
        let expected_intermediate = cfg
            .moe_intermediate_size
            .expect("Gemma4TextConfig validation requires moe_intermediate_size");
        if experts.moe_intermediate != expected_intermediate {
            return Err(anyhow!(
                "Gemma4MoeBlock `{prefix}`: expert intermediate {} != config moe_intermediate_size {}",
                experts.moe_intermediate,
                expected_intermediate
            ));
        }
        if experts.num_experts != cfg.num_experts_value() {
            return Err(anyhow!(
                "Gemma4MoeBlock `{prefix}`: expert count {} != config num_experts {}",
                experts.num_experts,
                cfg.num_experts_value()
            ));
        }
        let post_expert_norm = RmsNorm::from_loader(
            loader,
            &format!("{prefix}.post_feedforward_layernorm_2"),
            cfg.rms_norm_eps,
        )
        .with_context(|| format!("loading Gemma4 MoE expert post norm at `{prefix}`"))?;

        Ok(Self {
            dense_mlp,
            post_dense_norm,
            router,
            pre_expert_norm,
            experts,
            post_expert_norm,
            layer_idx,
            layer_kind,
        })
    }

    fn forward_on(
        &self,
        hidden: &Array,
        pre_feedforward_layernorm: &RmsNorm,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let profile = profile::vl_layer_enabled();

        let t0 = std::time::Instant::now();
        let dense = pre_feedforward_layernorm.forward_on(hidden, target)?;
        let dense = self.dense_mlp.forward_on(&dense, target)?;
        let dense = self.post_dense_norm.forward_on(&dense, target)?;
        profile::eval_layer(
            "gemma4_text_moe_dense_branch",
            self.layer_idx,
            self.layer_kind,
            &[&dense],
            t0,
            profile,
        )?;

        let dims = hidden.shape();
        let shape = dims.as_slice();
        if shape.len() != 3 {
            return Err(anyhow!(
                "Gemma4MoeBlock::forward_on: hidden must be [B,S,H], got {shape:?}"
            ));
        }
        let (batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);
        let bs = batch * seq;

        let t0 = std::time::Instant::now();
        let (expert_indices, expert_weights) = self.router.route_on(hidden, target)?;
        profile::eval_layer(
            "gemma4_text_moe_router",
            self.layer_idx,
            self.layer_kind,
            &[&expert_indices, &expert_weights],
            t0,
            profile,
        )?;

        let t0 = std::time::Instant::now();
        let expert_input = self.pre_expert_norm.forward_on(hidden, target)?;
        let expert_input = expert_input
            .reshape_on((bs, hidden_size), target)
            .context("Gemma4MoeBlock: reshape expert input to [BS,H]")?;
        let expert = self.experts.apply_experts_geglu(
            &expert_input,
            &expert_indices,
            &expert_weights,
            target,
            self.layer_idx as i32,
        )?;
        let expert = expert
            .reshape_on((batch, seq, hidden_size), target)
            .context("Gemma4MoeBlock: reshape expert output to [B,S,H]")?;
        let expert = self.post_expert_norm.forward_on(&expert, target)?;
        profile::eval_layer(
            "gemma4_text_moe_expert_branch",
            self.layer_idx,
            self.layer_kind,
            &[&expert],
            t0,
            profile,
        )?;

        Ok(&dense + &expert)
    }
}

impl Gemma4Router {
    fn from_loader(loader: &Loader, prefix: &str, cfg: &Gemma4TextConfig) -> Result<Self> {
        let proj = Linear::from_loader(loader, &format!("{prefix}.proj"))
            .context("Gemma4Router: loading router projection")?;
        let num_experts = cfg.num_experts_value();
        if proj.out_features() != num_experts as usize {
            return Err(anyhow!(
                "Gemma4Router `{prefix}`: proj output features {} != num_experts {}",
                proj.out_features(),
                num_experts
            ));
        }
        Ok(Self {
            proj,
            scale: loader
                .tensor(&format!("{prefix}.scale"))
                .context("Gemma4Router: loading scale")?
                .clone(),
            per_expert_scale: loader
                .tensor(&format!("{prefix}.per_expert_scale"))
                .context("Gemma4Router: loading per_expert_scale")?
                .clone(),
            hidden_root: (cfg.hidden_size as f32).powf(-0.5),
            eps: cfg.rms_norm_eps,
            top_k: cfg.top_k_experts_value(),
            num_experts,
        })
    }

    fn route_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<(Array, Array)> {
        let dims = hidden.shape();
        let shape = dims.as_slice();
        if shape.len() != 3 {
            return Err(anyhow!(
                "Gemma4Router::route_on: hidden must be [B,S,H], got {shape:?}"
            ));
        }
        let (batch, seq, hidden_size) = (shape[0], shape[1], shape[2]);
        let bs = batch * seq;
        let flat = hidden
            .reshape_on((bs, hidden_size), target)
            .context("Gemma4Router: reshape hidden to [BS,H]")?;
        let router_scale = &self.scale * self.hidden_root;
        let flat = mlx::fast::rms_norm_on(&flat, Some(&router_scale), self.eps, target)
            .context("Gemma4Router: rms_norm")?;
        let logits = self.proj.forward_on(&flat, target)?;
        let logits_shape = logits.shape();
        let logits_shape = logits_shape.as_slice();
        if logits_shape != [bs, self.num_experts] {
            return Err(anyhow!(
                "Gemma4Router: logits shape {logits_shape:?} != [{bs}, {}]",
                self.num_experts
            ));
        }

        let part = argpartition_on(&logits, -self.top_k, -1, target)
            .context("Gemma4Router: argpartition top-k")?;
        let indices = slice_on(
            &part,
            [0_i32, self.num_experts - self.top_k],
            [bs, self.num_experts],
            target,
        )
        .context("Gemma4Router: slice top-k indices")?;
        let weights_raw = take_along_axis_on(&logits, &indices, -1, target)
            .context("Gemma4Router: take top-k logits")?;
        let weights = mlx::ops::softmax_on(&weights_raw, -1_i32, true, target)
            .context("Gemma4Router: softmax top-k logits")?;
        let expert_scale = take_on(&self.per_expert_scale, &indices, 0, target)
            .context("Gemma4Router: take per-expert scale")?;
        let weights = &weights * &expert_scale;
        let indices_u32 = mlx::ops::cast::astype_on(&indices, mlx::Dtype::Uint32, target)
            .context("Gemma4Router: cast top-k indices")?;
        Ok((indices_u32, weights))
    }
}
