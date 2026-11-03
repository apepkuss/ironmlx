//! Flash-Next text graph. Metadata-only memory admission lives in load_plan;
//! runtime request/cache adaptation is kept separate from the native graph.
use super::{
    attention::{QsaAttention, QsaState},
    config::FlashTextConfig,
    hyper::HyperConnection,
    math::capture_hc_mean_on,
    paged_moe::PagedMoe,
    ple::{PleLayer, PleState},
    ple_rows::PleRows,
};
use crate::{
    core::{
        cache::{gated_delta::GatedDeltaCacheSnapshot, GatedDeltaCache},
        paged_experts::{ExpertCacheStats, ExpertTableSpec, PagedExperts},
        tensor_file::ModelTensorFiles,
        Loader,
    },
    models::qwen3_5_moe::SparseMoeBlock,
    nn::{Embedding, GatedDeltaNet, Linear},
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::sync::{Arc, Mutex};

enum Attention {
    Linear(Box<GatedDeltaNet>),
    Sparse(Box<QsaAttention>),
}
enum Moe {
    Resident(Box<SparseMoeBlock>),
    Paged(Box<PagedMoe>),
}
#[derive(Debug, Clone, Copy)]
pub struct ExpertPagingOptions {
    pub cache_bytes: usize,
    pub staging_bytes: usize,
    pub pack_bytes: usize,
}
struct Layer {
    attention: Attention,
    ple: Option<PleLayer>,
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    mlp: Moe,
}
pub(crate) enum LayerState {
    Linear {
        gdn: Box<GatedDeltaCache>,
        ple: Option<PleState>,
    },
    Sparse(QsaState),
}
#[derive(Clone)]
pub(crate) enum LayerSnapshot {
    Linear(GatedDeltaCacheSnapshot, Option<PleState>),
    Sparse(QsaState),
}

pub struct FlashCache {
    pub(crate) layers: Vec<LayerState>,
    pub position: i32,
    pub(crate) cap: i32,
    pub(crate) instance: Arc<()>,
}
#[derive(Clone)]
pub struct FlashCacheSnapshot {
    layers: Vec<LayerSnapshot>,
    position: i32,
    instance: Arc<()>,
}
pub struct FlashOutput {
    pub logits: Array,
    pub captures: Vec<Array>,
}
pub struct FlashHiddenOutput {
    pub hidden: Array,
    pub captures: Vec<Array>,
}

impl FlashCache {
    pub fn snapshot(&self) -> FlashCacheSnapshot {
        FlashCacheSnapshot {
            position: self.position,
            instance: Arc::clone(&self.instance),
            layers: self
                .layers
                .iter()
                .map(|state| match state {
                    LayerState::Linear { gdn, ple } => {
                        LayerSnapshot::Linear(gdn.snapshot(), ple.clone())
                    }
                    LayerState::Sparse(state) => LayerSnapshot::Sparse(state.clone()),
                })
                .collect(),
        }
    }
    pub fn restore(&mut self, snapshot: &FlashCacheSnapshot) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.instance, &snapshot.instance)
                && self.layers.len() == snapshot.layers.len(),
            "Flash cache snapshot shape mismatch"
        );
        for (state, snapshot) in self.layers.iter_mut().zip(&snapshot.layers) {
            match (state, snapshot) {
                (LayerState::Linear { gdn, ple }, LayerSnapshot::Linear(base, base_ple)) => {
                    gdn.restore(base)?;
                    *ple = base_ple.clone();
                }
                (LayerState::Sparse(state), LayerSnapshot::Sparse(base)) => *state = base.clone(),
                _ => anyhow::bail!("Flash cache snapshot architecture mismatch"),
            }
        }
        self.position = snapshot.position;
        Ok(())
    }
}

pub struct FlashTextModel {
    ane_shared: std::sync::OnceLock<Option<crate::nn::flash_ane::Context>>,
    embedding: Embedding,
    layers: Vec<Layer>,
    mixer: HyperConnection,
    head: Linear,
    pub config: FlashTextConfig,
    pub(crate) instance: Arc<()>,
    pub(crate) dflash_instance: crate::models::dflash2::DFlash2Instance,
    pub(crate) resident_bytes: usize,
    pub(crate) quant: Option<crate::core::weights::QuantMeta>,
    experts: Option<Arc<Mutex<PagedExperts>>>,
    pub(crate) layer_major_verify: bool,
    pub(crate) vision: crate::models::vision::VisionSlot,
    pub(crate) vision_config: Option<crate::models::qwen3_5::VisionConfig>,
}

impl FlashTextModel {
    pub fn from_loader(
        loader: &Loader,
        files: Arc<ModelTensorFiles>,
        ple_cache_bytes: usize,
        ple_staging_bytes: usize,
        qsa_scratch_bytes: usize,
    ) -> Result<Self> {
        Self::from_loader_with_paging(
            loader,
            files,
            ple_cache_bytes,
            ple_staging_bytes,
            qsa_scratch_bytes,
            None,
        )
    }

    pub fn from_loader_with_paging(
        loader: &Loader,
        files: Arc<ModelTensorFiles>,
        ple_cache_bytes: usize,
        ple_staging_bytes: usize,
        qsa_scratch_bytes: usize,
        paging: Option<ExpertPagingOptions>,
    ) -> Result<Self> {
        let cfg = FlashTextConfig::from_root(loader.config_raw_value())?;
        let layer_major_verify = match std::env::var("IRONMLX_FLASH_VERIFY_ORDER")
            .unwrap_or_else(|_| "layer".to_owned())
            .as_str()
        {
            "layer" => true,
            "token" => false,
            _ => anyhow::bail!("IRONMLX_FLASH_VERIFY_ORDER must be layer or token"),
        };
        let experts = paging
            .map(|options| -> Result<_> {
                let mut tables = Vec::new();
                for layer in 0..cfg.num_hidden_layers {
                    let prefix = format!("model.layers.{layer}.mlp.switch_mlp");
                    ensure!(!loader.contains(&format!("{prefix}.gate_proj.weight")),
                    "paged graph requires a loader that excluded routed experts before evaluation");
                    let quant = loader
                        .quant_meta_for(&format!("{prefix}.gate_proj"))
                        .context("missing expert quantization")?;
                    ensure!(
                        ["up_proj", "down_proj"]
                            .iter()
                            .all(|name| loader.quant_meta_for(&format!("{prefix}.{name}"))
                                == Some(quant)),
                        "paged Flash experts require matching gate/up/down quantization"
                    );
                    tables.push(ExpertTableSpec {
                        prefix: format!("language_model.{prefix}"),
                        experts: cfg.num_experts as u32,
                        hidden: cfg.hidden_size,
                        intermediate: cfg.moe_intermediate_size,
                        quant,
                    });
                }
                Ok(Arc::new(Mutex::new(PagedExperts::new(
                    Arc::clone(&files),
                    tables,
                    options.cache_bytes,
                    options.staging_bytes,
                    options.pack_bytes,
                )?)))
            })
            .transpose()?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (index, kind) in cfg.layer_types.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            let attention = if kind == "linear_attention" {
                Attention::Linear(Box::new(GatedDeltaNet::from_flash_weights(
                    loader,
                    &format!("{prefix}.linear_attn"),
                    &cfg,
                )?))
            } else {
                Attention::Sparse(Box::new(QsaAttention::from_weights(
                    loader,
                    &format!("{prefix}.self_attn"),
                    &cfg,
                    qsa_scratch_bytes,
                )?))
            };
            let ple = cfg
                .ple_layer_ids
                .iter()
                .position(|&id| id == index + 1)
                .map(|ple_index| {
                    let address = cfg.ple_index(ple_index)?;
                    let width = cfg.ple_embed_dim as usize / address.heads();
                    let rows = PleRows::new(
                        Arc::clone(&files),
                        &format!("language_model.{prefix}.ple.ple_embedding.ngram_embedding"),
                        address,
                        width,
                        ple_cache_bytes,
                        ple_staging_bytes,
                    )?;
                    PleLayer::from_weights(loader, &format!("{prefix}.ple"), &cfg, rows)
                })
                .transpose()?;
            layers.push(Layer {
                attention,
                ple,
                attn_hc: HyperConnection::from_weights(
                    loader,
                    &format!("{prefix}.attn_hyper_connection"),
                    &cfg,
                    true,
                )?,
                mlp_hc: HyperConnection::from_weights(
                    loader,
                    &format!("{prefix}.mlp_hyper_connection"),
                    &cfg,
                    true,
                )?,
                mlp: if let Some(store) = &experts {
                    Moe::Paged(Box::new(PagedMoe::from_weights(
                        loader,
                        &format!("{prefix}.mlp"),
                        &cfg,
                        Arc::clone(store),
                        index,
                    )?))
                } else {
                    Moe::Resident(Box::new(SparseMoeBlock::from_loader_flash(
                        loader,
                        &format!("{prefix}.mlp"),
                        cfg.num_experts_per_tok,
                        cfg.norm_topk_prob,
                    )?))
                },
            });
        }
        let vision_config: Option<crate::models::qwen3_5::VisionConfig> = loader
            .config_raw_value()
            .get("vision_config")
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?;
        let vision =
            crate::models::vision::VisionSlot::from_loader(loader, vision_config.as_ref())?;
        Ok(Self {
            ane_shared: std::sync::OnceLock::new(),
            vision,
            vision_config,
            embedding: Embedding::from_loader(loader, "model.embed_tokens")?,
            layers,
            mixer: HyperConnection::from_weights(
                loader,
                "model.hyper_connection_mixer",
                &cfg,
                false,
            )?,
            head: Linear::from_loader(loader, "lm_head")?,
            config: cfg.clone(),
            instance: Arc::new(()),
            dflash_instance: crate::models::dflash2::DFlash2Instance::new(),
            quant: loader.quant_meta(),
            resident_bytes: loader
                .loaded_tensor_bytes()
                .checked_add(
                    ple_cache_bytes
                        .checked_mul(cfg.ple_layer_ids.len())
                        .context("PLE cache reservation overflow")?,
                )
                .and_then(|n| n.checked_add(paging.map_or(0, |options| options.cache_bytes)))
                .context("Flash model residency overflow")?,
            experts,
            layer_major_verify,
        })
    }

    pub(crate) fn ane_scope(
        &self,
        batch: usize,
        seq: i32,
        prefill: bool,
    ) -> Option<crate::nn::flash_ane::Scope> {
        if !prefill || batch != 1 || seq != 2048 || !crate::nn::flash_ane::requested() {
            return None;
        }
        self.ane_shared
            .get_or_init(|| match crate::nn::flash_ane::SharedExpert::new(2048) {
                Ok(context) => {
                    tracing::info!(
                        surface_bytes = context.surface_bytes(),
                        "Flash shared-expert ANE context created"
                    );
                    Some(Arc::new(Mutex::new(context)))
                }
                Err(error) => {
                    tracing::warn!(%error,"Flash ANE unavailable; shared experts stay on GPU");
                    None
                }
            })
            .as_ref()
            .map(|context| crate::nn::flash_ane::scope(Arc::clone(context)))
    }
    pub fn expert_cache_stats(&self) -> Result<Option<ExpertCacheStats>> {
        self.experts
            .as_ref()
            .map(|store| {
                Ok(store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert cache lock poisoned"))?
                    .stats())
            })
            .transpose()
    }
    pub(crate) fn expert_paging_enabled(&self) -> bool {
        self.experts.is_some()
    }

    pub fn new_cache(&self, cap: i32) -> Result<FlashCache> {
        let cfg = &self.config;
        ensure!(
            cap > 0 && cap as u32 <= cfg.max_position_embeddings,
            "invalid Flash context capacity"
        );
        let mut layers = Vec::new();
        for layer in &self.layers {
            layers.push(match &layer.attention {
                Attention::Linear(gdn) => {
                    let c = gdn.config();
                    LayerState::Linear {
                        gdn: Box::new(GatedDeltaCache::new_with_cap(
                            1,
                            c.conv_kernel_size,
                            c.conv_dim(),
                            c.num_v_heads,
                            c.head_v_dim,
                            c.head_k_dim,
                            Dtype::Bfloat16,
                            cap,
                        )?),
                        ple: layer.ple.as_ref().map(PleLayer::new_state).transpose()?,
                    }
                }
                Attention::Sparse(_) => LayerState::Sparse(QsaState::new(cap)?),
            });
        }
        Ok(FlashCache {
            layers,
            position: 0,
            cap,
            instance: Arc::clone(&self.instance),
        })
    }

    pub fn forward_tokens_on(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        target: StreamOrDevice,
    ) -> Result<FlashOutput> {
        let base = cache.snapshot();
        let result = self
            .forward_hidden_tokens_on(tokens, cache, capture_layers, target)
            .and_then(|output| {
                let logits = self.head.forward_on(&output.hidden, target)?;
                logits.eval()?;
                Ok(FlashOutput {
                    logits,
                    captures: output.captures,
                })
            });
        if result.is_err() {
            cache.restore(&base)?;
        }
        result
    }

    pub fn forward_hidden_tokens_on(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        target: StreamOrDevice,
    ) -> Result<FlashHiddenOutput> {
        self.forward_inputs_on(tokens, cache, capture_layers, None, None, target)
    }

    pub(crate) fn forward_inputs_on(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        positions: Option<&Array>,
        embeddings: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<FlashHiddenOutput> {
        self.validate_request(tokens, cache, capture_layers)?;
        if let Some(p) = positions {
            ensure!(
                p.shape().as_slice() == [3, 1, tokens.len() as i32],
                "Flash positions must be [3,1,S]"
            );
        }
        if let Some(e) = embeddings {
            ensure!(
                e.shape().as_slice() == [1, tokens.len() as i32, self.config.hidden_size],
                "Flash embeddings must be [1,S,H]"
            );
        }
        let base = cache.snapshot();
        let result =
            self.forward_inner(tokens, cache, capture_layers, positions, embeddings, target);
        if result.is_err() {
            cache
                .restore(&base)
                .context("restoring failed Flash forward")?;
        }
        result
    }

    fn validate_request(
        &self,
        tokens: &[u32],
        cache: &FlashCache,
        capture_layers: &[usize],
    ) -> Result<()> {
        let seq = i32::try_from(tokens.len())?;
        ensure!(
            Arc::ptr_eq(&self.instance, &cache.instance),
            "Flash cache belongs to another loaded model"
        );
        ensure!(
            seq > 0 && tokens.iter().all(|&id| id < self.config.vocab_size),
            "invalid Flash token IDs"
        );
        ensure!(
            cache
                .position
                .checked_add(seq)
                .is_some_and(|end| end <= cache.cap)
                && cache.layers.len() == self.layers.len(),
            "Flash cache capacity/architecture mismatch"
        );
        ensure!(
            capture_layers.windows(2).all(|v| v[0] < v[1])
                && capture_layers.iter().all(|&id| id < self.layers.len()),
            "invalid Flash capture layers"
        );
        Ok(())
    }

    /// Exact Q1 arithmetic with layer-major scheduling. Within a layer the
    /// request advances token by token; every projection/attention still has
    /// its Q1 shape. The whole window shares bounded graph submissions and
    /// the currently hot expert table. Prefix snapshots include PLE history.
    pub(crate) fn forward_verify_hidden_tokens_on(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        positions: Option<&Array>,
        embeddings: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<(FlashHiddenOutput, Vec<Vec<LayerSnapshot>>)> {
        self.validate_request(tokens, cache, capture_layers)?;
        ensure!(
            tokens.len() <= 8,
            "Flash verifier exceeds qualified Q8 width"
        );
        let base = cache.snapshot();
        let result = (|| {
            let seq = tokens.len() as i32;
            let ids = Array::try_from((tokens, [1, seq]))?;
            let embedding = match embeddings {
                Some(e) => e.clone(),
                None => self.embedding.forward_on(&ids, target)?,
            };
            let streams = vec![&embedding; self.config.hc_count as usize];
            let mut hyper = mlx::ops::shape::concatenate_on(&streams, -1, target)?;
            let mut captures = Vec::new();
            let mut prefixes: Vec<Vec<LayerSnapshot>> =
                (0..self.layers.len()).map(|_| Vec::new()).collect();
            for (index, state) in cache.layers.iter_mut().enumerate() {
                let mut outputs = Vec::new();
                for token in 0..seq {
                    let input = hyper.slice(
                        &[0, token, 0][..],
                        &[1, token + 1, self.config.hc_count * self.config.hidden_size][..],
                    )?;
                    outputs.push(
                        self.forward_layer_on(
                            index,
                            &tokens[token as usize..token as usize + 1],
                            state,
                            base.position + token,
                            &input,
                            positions
                                .map(|p| p.slice(&[0, 0, token][..], &[3, 1, token + 1][..]))
                                .transpose()?
                                .as_ref(),
                            target,
                        )?,
                    );
                    if matches!(state, LayerState::Linear { .. }) {
                        prefixes[index].push(state.snapshot());
                    }
                }
                let refs: Vec<&Array> = outputs.iter().collect();
                hyper = mlx::ops::shape::concatenate_on(&refs, 1, target)?;
                if capture_layers.binary_search(&index).is_ok() {
                    captures.push(capture_hc_mean_on(
                        &hyper,
                        self.config.hc_count,
                        self.config.hidden_size,
                        target,
                    )?);
                }
                if index % 4 == 3 {
                    hyper.eval()?;
                }
            }
            let mut mixed = Vec::new();
            for token in 0..seq {
                let input = hyper.slice(
                    &[0, token, 0][..],
                    &[1, token + 1, self.config.hc_count * self.config.hidden_size][..],
                )?;
                mixed.push(self.mixer.final_mix_on(&input, target)?);
            }
            let refs: Vec<&Array> = mixed.iter().collect();
            let hidden = mlx::ops::shape::concatenate_on(&refs, 1, target)?;
            hidden.eval()?;
            for capture in &captures {
                capture.eval()?;
            }
            cache.position += seq;
            Ok((FlashHiddenOutput { hidden, captures }, prefixes))
        })();
        if result.is_err() {
            cache.restore(&base)?;
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_on(
        &self,
        index: usize,
        tokens: &[u32],
        state: &mut LayerState,
        position: i32,
        input: &Array,
        positions: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let layer = &self.layers[index];
        let mut hyper = input.clone();
        if let (
            Some(ple),
            LayerState::Linear {
                ple: Some(state), ..
            },
        ) = (&layer.ple, &mut *state)
        {
            let delta = ple.forward_on(&hyper, tokens, state, target)?;
            hyper = mlx::ops::binary::add_on(&hyper, &delta, target)?;
        }
        let branch = layer.attn_hc.branch_on(&hyper, target)?;
        let output = match (&layer.attention, &mut *state) {
            (Attention::Linear(gdn), LayerState::Linear { gdn: state, .. }) => {
                ensure!(state.offsets() == [position], "Flash GDN position mismatch");
                gdn.forward_on(&branch.input, None, None, Some(state), target, index as i32)?
            }
            (Attention::Sparse(attention), LayerState::Sparse(state)) => {
                ensure!(state.offset == position, "Flash QSA position mismatch");
                attention.forward_positions_on(&branch.input, state, positions, target)?
            }
            _ => anyhow::bail!("Flash layer state mismatch"),
        };
        hyper = branch.combine_on(&output, target)?;
        let branch = layer.mlp_hc.branch_on(&hyper, target)?;
        let output = match &layer.mlp {
            Moe::Resident(moe) => moe.forward_on(&branch.input, target, index as i32)?,
            Moe::Paged(moe) => moe.forward_on(&branch.input, target, index as i32)?,
        };
        hyper = branch.combine_on(&output, target)?;
        Ok(hyper)
    }

    fn forward_inner(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        positions: Option<&Array>,
        embeddings: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<FlashHiddenOutput> {
        let seq = tokens.len() as i32;
        let ids = Array::try_from((tokens, [1, seq]))?;
        let embedding = match embeddings {
            Some(e) => e.clone(),
            None => self.embedding.forward_on(&ids, target)?,
        };
        let streams = vec![&embedding; self.config.hc_count as usize];
        let mut hyper = mlx::ops::shape::concatenate_on(&streams, -1, target)?;
        let mut captures = Vec::new();
        for (index, state) in cache.layers.iter_mut().enumerate() {
            hyper = self.forward_layer_on(
                index,
                tokens,
                state,
                cache.position,
                &hyper,
                positions,
                target,
            )?;
            if capture_layers.binary_search(&index).is_ok() {
                captures.push(capture_hc_mean_on(
                    &hyper,
                    self.config.hc_count,
                    self.config.hidden_size,
                    target,
                )?);
            }
            // Submit bounded subgraphs; do not retain a 48-layer lazy graph.
            if index % 4 == 3 {
                hyper.eval()?;
            }
        }
        let hidden = self.mixer.final_mix_on(&hyper, target)?;
        hidden.eval()?;
        for capture in &captures {
            capture.eval()?;
        }
        cache.position += seq;
        Ok(FlashHiddenOutput { hidden, captures })
    }

    pub(crate) fn embed_on(&self, ids: &Array, target: StreamOrDevice) -> Result<Array> {
        self.embedding.forward_on(ids, target)
    }
    pub(crate) fn project_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        self.head.forward_on(hidden, target)
    }
}
