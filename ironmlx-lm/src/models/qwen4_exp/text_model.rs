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
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ExpertPagingOptions {
    pub cache_bytes: usize,
    pub staging_bytes: usize,
    pub pack_bytes: usize,
}
/// Order in which a speculative window is verified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlashVerifyOrder {
    #[default]
    Layer,
    Token,
}
/// Whether batches above one row run paged experts as one native group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlashPagedBatching {
    /// Native on a qualified GPU, per-row elsewhere.
    #[default]
    Auto,
    On,
    Off,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlashExecutionOptions {
    pub verify_order: FlashVerifyOrder,
    pub paged_batching: FlashPagedBatching,
}
impl FlashExecutionOptions {
    /// Optional explicit process configuration. Invalid settings are errors.
    pub fn from_environment() -> Result<Self> {
        let verify_order = match std::env::var("IRONMLX_FLASH_VERIFY_ORDER")
            .unwrap_or_else(|_| "layer".to_owned())
            .as_str()
        {
            "layer" => FlashVerifyOrder::Layer,
            "token" => FlashVerifyOrder::Token,
            _ => anyhow::bail!("IRONMLX_FLASH_VERIFY_ORDER must be layer or token"),
        };
        let paged_batching = match std::env::var("IRONMLX_FLASH_PAGED_BATCHING")
            .unwrap_or_else(|_| "auto".to_owned())
            .as_str()
        {
            "auto" => FlashPagedBatching::Auto,
            "on" => FlashPagedBatching::On,
            "off" => FlashPagedBatching::Off,
            _ => anyhow::bail!("IRONMLX_FLASH_PAGED_BATCHING must be auto, on or off"),
        };
        Ok(Self {
            verify_order,
            paged_batching,
        })
    }
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

pub(crate) type GroupPrefixes = Vec<Vec<Vec<LayerSnapshot>>>;
type GroupOutput = (Vec<FlashHiddenOutput>, GroupPrefixes);
pub struct FlashTextModel {
    embedding: Embedding,
    layers: Vec<Layer>,
    mixer: HyperConnection,
    head: Linear,
    pub config: FlashTextConfig,
    pub(crate) active_pages: Option<Arc<crate::core::active_pages::ActivePagesPool>>,
    pub(crate) paging_telemetry: Arc<crate::core::paging_stats::PagingTelemetry>,
    pub(crate) instance: Arc<()>,
    pub(crate) artifact_identity: String,
    pub(crate) dflash_instance: crate::models::dflash2::DFlash2Instance,
    pub(crate) resident_bytes: usize,
    pub(crate) quant: Option<crate::core::weights::QuantMeta>,
    experts: Option<Arc<Mutex<PagedExperts>>>,
    pub(crate) layer_major_verify: bool,
    pub(crate) native_paged_batching: bool,
    pub(crate) native_batch_unqualified: bool,
    pub(crate) vision: crate::models::vision::VisionSlot,
    pub(crate) vision_config: Option<crate::models::qwen3_5::VisionConfig>,
    // Kept until model teardown; does not own KV or paged expert buffers.
    _weight_residency: Option<mlx::memory::WeightResidency>,
    residency_binding_status: std::sync::atomic::AtomicU8,
}

impl FlashTextModel {
    pub(crate) fn bind_weight_residency(&self, target: StreamOrDevice) {
        if let Some(lease) = &self._weight_residency {
            let result = lease.bind_on(target);
            let status = match &result {
                Ok(true) => 1,
                Ok(false) => 2,
                Err(_) => 3,
            };
            let previous = self
                .residency_binding_status
                .swap(status, std::sync::atomic::Ordering::Relaxed);
            if previous != status {
                match result {
                    Err(error) => {
                        tracing::warn!(%error, "Flash weight residency binding failed; using automatic Metal residency")
                    }
                    Ok(false) => tracing::debug!(
                        "Flash weight residency paused or queue capacity unavailable"
                    ),
                    Ok(true) => tracing::debug!("Flash weight residency bound to execution queue"),
                }
            }
        }
    }
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
            FlashExecutionOptions::default(),
        )
    }

    pub fn from_loader_with_paging(
        loader: &Loader,
        files: Arc<ModelTensorFiles>,
        ple_cache_bytes: usize,
        ple_staging_bytes: usize,
        qsa_scratch_bytes: usize,
        paging: Option<ExpertPagingOptions>,
        execution: FlashExecutionOptions,
    ) -> Result<Self> {
        let cfg = FlashTextConfig::from_root(loader.config_raw_value())?;
        let layer_major_verify = execution.verify_order == FlashVerifyOrder::Layer;
        let qualified =
            native_batch_qualified(ironmlx_core::m5_profile::hardware_architecture().as_deref());
        let native_paged_batching = match execution.paged_batching {
            FlashPagedBatching::Auto => qualified,
            FlashPagedBatching::On => true,
            FlashPagedBatching::Off => false,
        };
        let native_batch_unqualified =
            execution.paged_batching == FlashPagedBatching::Auto && !qualified;
        let telemetry = Arc::new(crate::core::paging_stats::PagingTelemetry::new(
            paging.is_some(),
        ));
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
                Ok(Arc::new(Mutex::new(
                    PagedExperts::new(
                        Arc::clone(&files),
                        tables,
                        options.cache_bytes,
                        options.staging_bytes,
                        options.pack_bytes,
                    )?
                    .with_telemetry(Arc::clone(&telemetry)),
                )))
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
                    )?
                    .with_telemetry(Arc::clone(&telemetry));
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
            artifact_identity: crate::core::artifact_identity::artifact_identity(
                loader.model_dir(),
            )?,
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
            active_pages: None,
            paging_telemetry: telemetry,
            layer_major_verify,
            native_paged_batching,
            native_batch_unqualified,
            _weight_residency: super::weight_residency::prepare(loader, paging.is_none())?,
            residency_binding_status: std::sync::atomic::AtomicU8::new(0),
        })
    }

    pub(crate) fn active_cache_cost(&self) -> (usize, usize) {
        let (token, fixed) = super::load_plan::flash_cache_cost(self.active_pages.is_some());
        (token as usize, fixed as usize)
    }
    /// Configure disposable sparse KV storage before creating request caches.
    /// Rebinding invalidates older request/draft identities; they cannot be
    /// mixed with the arithmetic/storage contract of the returned model.
    pub fn with_active_state(
        mut self,
        config: crate::core::active_pages::ActivePagesConfig,
    ) -> Result<Self> {
        ensure!(
            self.active_pages.is_none(),
            "Flash active state already configured"
        );
        super::load_plan::validate_active_config(&config, &self.config)?;
        let pool = crate::core::active_pages::ActivePagesPool::new(config)?;
        pool.stats.set_hot_window_tokens(
            (pool.config.cache_bytes as u64
                / (self.config.num_key_value_heads as u64 * self.config.head_dim as u64 * 4))
                .max(4096),
        );
        tracing::info!(
            read_cache_bytes = pool.config.cache_bytes,
            gather_payload_bytes = pool.config.pack_bytes,
            disk_quota_bytes = pool.config.disk_bytes,
            hot_window_tokens = pool.stats.snapshot().hot_window_tokens,
            "Flash active sparse KV paging enabled; indexer and recurrent state remain in RAM"
        );
        self.paging_telemetry
            .attach_active_state(Arc::clone(&pool.stats));
        self.active_pages = Some(pool);
        self.instance = Arc::new(());
        self.dflash_instance = crate::models::dflash2::DFlash2Instance::new();
        Ok(self)
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
    /// Whether batches above one row run paged experts as one native group.
    /// False with fully resident experts, with batching off, and with `auto`
    /// on a GPU that does not qualify.
    pub fn native_paged_batching_enabled(&self) -> bool {
        self.expert_paging_enabled() && self.native_paged_batching
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
                Attention::Sparse(_) => {
                    let state = QsaState::new(cap)?;
                    LayerState::Sparse(if let Some(pool) = &self.active_pages {
                        state.with_active_pages(Arc::clone(pool))
                    } else {
                        state
                    })
                }
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
        let _phase = crate::core::paging_stats::PhaseGuard::enter(if tokens.len() == 1 {
            crate::core::paging_stats::PagingPhase::Decode
        } else {
            crate::core::paging_stats::PagingPhase::Prefill
        });
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
        self.bind_weight_residency(target);
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
        let _phase = crate::core::paging_stats::PhaseGuard::enter(
            crate::core::paging_stats::PagingPhase::Verify,
        );
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
        let branch = self.forward_attention_branch_on(
            index, tokens, state, position, input, positions, target,
        )?;
        let output = match &layer.mlp {
            Moe::Resident(moe) => moe.forward_on(&branch.input, target, index as i32)?,
            Moe::Paged(moe) => moe.forward_on(&branch.input, target, index as i32)?,
        };
        branch.combine_on(&output, target)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_branch_on(
        &self,
        index: usize,
        tokens: &[u32],
        state: &mut LayerState,
        position: i32,
        input: &Array,
        positions: Option<&Array>,
        target: StreamOrDevice,
    ) -> Result<super::hyper::HyperBranch> {
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
        layer.mlp_hc.branch_on(&hyper, target)
    }

    /// Layer-major requests with isolated attention/recurrent state and one
    /// shared routed expert execution per layer. Verification keeps Q1
    /// attention/HC arithmetic and records each token's rollback snapshot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_group_on(
        &self,
        tokens: &[Vec<u32>],
        rows: &mut [FlashCache],
        captures: &[usize],
        positions: &[Option<Array>],
        embeddings: &[Option<Array>],
        verify: bool,
        target: StreamOrDevice,
    ) -> Result<GroupOutput> {
        let mut hyper = Vec::new();
        let mut captured: Vec<Vec<Array>> = (0..rows.len()).map(|_| Vec::new()).collect();
        let mut prefixes: Vec<Vec<Vec<LayerSnapshot>>> = (0..self.layers.len())
            .map(|_| (0..rows.len()).map(|_| Vec::new()).collect())
            .collect();
        for (r, t) in tokens.iter().enumerate() {
            if t.is_empty() {
                hyper.push(None);
                continue;
            }
            self.validate_request(t, &rows[r], captures)?;
            let ids = Array::try_from((t.as_slice(), [1, t.len() as i32]))?;
            let embedding = match &embeddings[r] {
                Some(e) => e.clone(),
                None => self.embedding.forward_on(&ids, target)?,
            };
            hyper.push(Some(mlx::ops::shape::concatenate_on(
                &vec![&embedding; self.config.hc_count as usize],
                -1,
                target,
            )?));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let mut branches = Vec::new();
            let mut owners = Vec::new();
            for (r, t) in tokens.iter().enumerate() {
                let Some(hyper) = &hyper[r] else {
                    continue;
                };
                let chunks = if verify { t.len() } else { 1 };
                for token in 0..chunks {
                    let (start, end) = if verify {
                        (token as i32, token as i32 + 1)
                    } else {
                        (0, t.len() as i32)
                    };
                    let input = hyper.slice(
                        &[0, start, 0][..],
                        &[1, end, self.config.hc_count * self.config.hidden_size][..],
                    )?;
                    let p = positions[r]
                        .as_ref()
                        .map(|p| p.slice(&[0, 0, start][..], &[3, 1, end][..]))
                        .transpose()?;
                    let position = rows[r].position + start;
                    let branch = self.forward_attention_branch_on(
                        index,
                        &t[start as usize..end as usize],
                        &mut rows[r].layers[index],
                        position,
                        &input,
                        p.as_ref(),
                        target,
                    )?;
                    if verify && matches!(rows[r].layers[index], LayerState::Linear { .. }) {
                        prefixes[index][r].push(rows[r].layers[index].snapshot());
                    }
                    branches.push(branch);
                    owners.push(r);
                }
            }
            if branches.is_empty() {
                continue;
            }
            let inputs: Vec<_> = branches.iter().map(|b| b.input.clone()).collect();
            let output = match &layer.mlp {
                Moe::Paged(moe) => moe.forward_rows_on(&inputs, &owners, target, index as i32)?,
                Moe::Resident(moe) => inputs
                    .iter()
                    .map(|x| moe.forward_on(x, target, index as i32))
                    .collect::<Result<Vec<_>>>()?,
            };
            let mut per_row: Vec<Vec<Array>> = (0..rows.len()).map(|_| Vec::new()).collect();
            for ((branch, output), r) in branches.into_iter().zip(output).zip(owners) {
                per_row[r].push(branch.combine_on(&output, target)?);
            }
            for (r, parts) in per_row.into_iter().enumerate() {
                if parts.is_empty() {
                    continue;
                }
                let refs: Vec<_> = parts.iter().collect();
                let joined = mlx::ops::shape::concatenate_on(&refs, 1, target)?;
                if captures.binary_search(&index).is_ok() {
                    captured[r].push(capture_hc_mean_on(
                        &joined,
                        self.config.hc_count,
                        self.config.hidden_size,
                        target,
                    )?);
                }
                if index % 4 == 3 {
                    joined.eval()?;
                }
                hyper[r] = Some(joined);
            }
        }
        let mut outputs = Vec::new();
        for (r, h) in hyper.into_iter().enumerate() {
            let hidden = if let Some(h) = h {
                let hidden = if verify {
                    let parts = (0..tokens[r].len() as i32)
                        .map(|t| {
                            self.mixer.final_mix_on(
                                &h.slice(
                                    &[0, t, 0][..],
                                    &[1, t + 1, self.config.hc_count * self.config.hidden_size][..],
                                )?,
                                target,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    mlx::ops::shape::concatenate_on(&parts.iter().collect::<Vec<_>>(), 1, target)?
                } else {
                    self.mixer.final_mix_on(&h, target)?
                };
                hidden.eval()?;
                hidden
            } else {
                Array::zeros([1, 0, self.config.hidden_size], Dtype::Bfloat16)?
            };
            for c in &captured[r] {
                c.eval()?;
            }
            rows[r].position += tokens[r].len() as i32;
            outputs.push(FlashHiddenOutput {
                hidden,
                captures: std::mem::take(&mut captured[r]),
            });
        }
        Ok((outputs, prefixes))
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
        // Resident single-token decode reads nothing back between groups, so
        // its groups are submitted without waiting. Paged paths keep the wait.
        let nonblocking = seq == 1 && self.experts.is_none() && self.active_pages.is_none();
        let mut submitted = SubmittedGroup(None);
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
                if nonblocking {
                    mlx::transforms::async_eval(&[&hyper])?;
                    submitted.0 = Some(hyper.clone());
                } else {
                    hyper.eval()?;
                }
            }
        }
        let hidden = self.mixer.final_mix_on(&hyper, target)?;
        hidden.eval()?;
        for capture in &captures {
            capture.eval()?;
        }
        // `hidden` depends on every submitted group, so nothing is in flight.
        submitted.0 = None;
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

/// The last group submitted without waiting. If the forward returns early,
/// dropping this waits for that group so no submitted work outlives the call.
struct SubmittedGroup(Option<Array>);

impl Drop for SubmittedGroup {
    fn drop(&mut self) {
        if let Some(group) = self.0.take() {
            let _ = group.eval();
        }
    }
}

/// `auto` runs native paged batches on Apple GPU generation 17 (M5 family) or
/// newer, by Metal architecture name such as `applegpu_g17s`. Measured on M5
/// Max only.
fn native_batch_qualified(architecture: Option<&str>) -> bool {
    architecture
        .and_then(ironmlx_core::m5_profile::gpu_generation)
        .is_some_and(|generation| generation >= ironmlx_core::m5_profile::MIN_GPU_GENERATION)
}

#[cfg(test)]
mod native_batch_tests {
    use super::*;
    #[test]
    fn auto_requires_gpu_generation_17_or_newer() {
        for architecture in [
            "applegpu_g17",
            "applegpu_g17p",
            "applegpu_g17s",
            "applegpu_g18s",
        ] {
            assert!(native_batch_qualified(Some(architecture)), "{architecture}");
        }
        for architecture in [
            None,
            Some("applegpu_g13s"),
            Some("applegpu_g16s"),
            Some("applegpu_gx"),
            Some("Apple M5 Max"),
        ] {
            assert!(!native_batch_qualified(architecture), "{architecture:?}");
        }
    }
}
