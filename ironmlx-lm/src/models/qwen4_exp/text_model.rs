//! Unregistered text bring-up graph. Runtime admission is deliberately kept
//! separate until complete-model, batching and speculative gates pass.
use super::{
    attention::{QsaAttention, QsaState},
    config::FlashTextConfig,
    hyper::HyperConnection,
    math::capture_hc_mean_on,
    ple::{PleLayer, PleState},
    ple_rows::PleRows,
};
use crate::{
    core::{
        cache::{gated_delta::GatedDeltaCacheSnapshot, GatedDeltaCache},
        tensor_file::ModelTensorFiles,
        Loader,
    },
    models::qwen3_5_moe::SparseMoeBlock,
    nn::{Embedding, GatedDeltaNet, Linear},
};
use anyhow::{ensure, Context, Result};
use mlx::{Array, Dtype, StreamOrDevice};
use std::sync::Arc;

enum Attention {
    Linear(Box<GatedDeltaNet>),
    Sparse(Box<QsaAttention>),
}
struct Layer {
    attention: Attention,
    ple: Option<PleLayer>,
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    mlp: SparseMoeBlock,
}
enum LayerState {
    Linear {
        gdn: GatedDeltaCache,
        ple: Option<PleState>,
    },
    Sparse(QsaState),
}
enum LayerSnapshot {
    Linear(GatedDeltaCacheSnapshot, Option<PleState>),
    Sparse(QsaState),
}

pub struct FlashCache {
    layers: Vec<LayerState>,
    pub position: i32,
    cap: i32,
    instance: Arc<()>,
}
pub struct FlashCacheSnapshot {
    layers: Vec<LayerSnapshot>,
    position: i32,
    instance: Arc<()>,
}
pub struct FlashOutput {
    pub logits: Array,
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
    embedding: Embedding,
    layers: Vec<Layer>,
    mixer: HyperConnection,
    head: Linear,
    pub config: FlashTextConfig,
    instance: Arc<()>,
}

impl FlashTextModel {
    pub fn from_loader(
        loader: &Loader,
        files: Arc<ModelTensorFiles>,
        ple_cache_bytes: usize,
        ple_staging_bytes: usize,
        qsa_scratch_bytes: usize,
    ) -> Result<Self> {
        let cfg = FlashTextConfig::from_root(loader.config_raw_value())?;
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
                mlp: SparseMoeBlock::from_loader(
                    loader,
                    &format!("{prefix}.mlp"),
                    cfg.num_experts_per_tok,
                    cfg.norm_topk_prob,
                )?,
            });
        }
        Ok(Self {
            embedding: Embedding::from_loader(loader, "model.embed_tokens")?,
            layers,
            mixer: HyperConnection::from_weights(
                loader,
                "model.hyper_connection_mixer",
                &cfg,
                false,
            )?,
            head: Linear::from_loader(loader, "lm_head")?,
            config: cfg,
            instance: Arc::new(()),
        })
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
                        gdn: GatedDeltaCache::new_with_cap(
                            1,
                            c.conv_kernel_size,
                            c.conv_dim(),
                            c.num_v_heads,
                            c.head_v_dim,
                            c.head_k_dim,
                            Dtype::Bfloat16,
                            cap,
                        )?,
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
        let base = cache.snapshot();
        let result = self.forward_inner(tokens, cache, capture_layers, target);
        if result.is_err() {
            cache
                .restore(&base)
                .context("restoring failed Flash forward")?;
        }
        result
    }

    fn forward_inner(
        &self,
        tokens: &[u32],
        cache: &mut FlashCache,
        capture_layers: &[usize],
        target: StreamOrDevice,
    ) -> Result<FlashOutput> {
        let seq = tokens.len() as i32;
        let ids = Array::try_from((tokens, [1, seq]))?;
        let embedding = self.embedding.forward_on(&ids, target)?;
        let streams = vec![&embedding; self.config.hc_count as usize];
        let mut hyper = mlx::ops::shape::concatenate_on(&streams, -1, target)?;
        let mut captures = Vec::new();
        for (index, (layer, state)) in self.layers.iter().zip(&mut cache.layers).enumerate() {
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
                    ensure!(
                        state.offsets() == [cache.position],
                        "Flash GDN position mismatch"
                    );
                    gdn.forward_on(&branch.input, None, None, Some(state), target, index as i32)?
                }
                (Attention::Sparse(attention), LayerState::Sparse(state)) => {
                    ensure!(
                        state.offset == cache.position,
                        "Flash QSA position mismatch"
                    );
                    attention.forward_on(&branch.input, state, target)?
                }
                _ => anyhow::bail!("Flash layer state mismatch"),
            };
            hyper = branch.combine_on(&output, target)?;
            let branch = layer.mlp_hc.branch_on(&hyper, target)?;
            let output = layer.mlp.forward_on(&branch.input, target, index as i32)?;
            hyper = branch.combine_on(&output, target)?;
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
        let logits = self.head.forward_on(&hidden, target)?;
        logits.eval()?;
        for capture in &captures {
            capture.eval()?;
        }
        cache.position += seq;
        Ok(FlashOutput { logits, captures })
    }
}
