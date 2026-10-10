//! DFlash2 target capability for the Qwen3.6-35B-A3B MoE checkpoints.
//!
//! The draft loop is shared with the dense Qwen3.5 family; this module owns
//! only the MoE target contract: which checkpoints are qualified, which verify
//! shapes they certify, and the execution route that keeps wide verification
//! row-exact with ordinary Q=1 decoding.

use anyhow::anyhow;
use mlx::{Array, StreamOrDevice};

use crate::core::cache::layer::{LayerCache, LayerCacheSnapshot};
use crate::models::dflash2::{
    DFlash2DraftTree, DFlash2HybridCacheGeometry, DFlash2Instance, DFlash2Target,
    DFlash2TargetCacheCost, DFlash2TargetForwardMode, DFlash2TargetOutput, DFlash2TargetSpec,
    DFlash2VerifyCapabilities, DFlash2VerifyShape,
};
use crate::Result;

use super::model::Qwen35MoeModel;

/// Largest B1 verify width (current token plus drafts) certified per target
/// affine bit width. Each entry is backed by the real-checkpoint
/// qualification `qwen36_moe_dflash2_verify_matches_ordinary_decode`, which
/// compares every tap, the final hidden state and the logits of each width
/// against ordinary Q=1 decoding of the same checkpoint, including prefix
/// restore after partial acceptance. A width absent here is not executable.
pub(crate) fn qwen36_moe_dflash2_max_verify_width(bits: i32) -> usize {
    match bits {
        4 | 5 | 6 | 8 => 8,
        _ => 0,
    }
}

/// Largest flat draft tree (nodes excluding the root, so one B1 tree verify
/// of `nodes + 1` rows) certified per target bit width by
/// `qwen36_moe_dflash2_tree_matches_ancestor_decode`, which compares every
/// node against ordinary decoding of its ancestor path and the committed
/// accepted-path state against sequential decoding.
pub(crate) fn qwen36_moe_dflash2_flat_tree_max_nodes(bits: i32) -> usize {
    match bits {
        4 | 5 | 6 | 8 => DFlash2DraftTree::MAX_NODES,
        _ => 0,
    }
}

/// Whether the explicit experimental setting arms the expert-grouped gather
/// qmv for multi-token verification (not part of the M5 MoE profile).
fn m5_grouped_experts() -> bool {
    ironmlx_core::m5_profile::flag(ironmlx_core::m5_profile::settings::M5_MOE_GROUPED_QMV)
}

/// Narrowest verify (rows) for which the M5 expert-grouped route runs, per
/// target bit width. Measured on M5 Max: it is faster than MLX's sorted
/// gather only for affine6/affine8 from eight rows up, and slower for
/// affine4/affine5 at every width, so those keep MLX's route.
pub(crate) fn qwen36_moe_grouped_experts_min_rows(bits: i32) -> Option<usize> {
    match bits {
        6 | 8 => Some(8),
        _ => None,
    }
}

impl Qwen35MoeModel {
    /// Default affine bit width of a DFlash2-qualified Qwen3.6 MoE target,
    /// `None` for every other MoE checkpoint.
    pub fn dflash2_target_bits(&self) -> Option<i32> {
        self.dflash2_target_bits
    }

    /// Shared B1 DFlash2 target forward for linear windows and flat trees
    /// (the tree plan, when present, is carried by the thread-local scope).
    fn forward_dflash2_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target_layer_ids: &[usize],
        mode: DFlash2TargetForwardMode,
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let input_shape = input_ids.shape();
        let input_dims = input_shape.as_slice();
        if input_dims.len() != 2 || input_dims[0] != 1 || input_dims[1] <= 0 {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 is qualified only for B1 inputs [1,S], got {input_dims:?}"
            ));
        }
        let verify_width = input_dims[1] as usize;
        // Verification keeps every quantized projection on the
        // product-stable route (one dispatch, ordinary Q=1 accumulation
        // tree), full attention on the position-stable bulk route, and
        // GatedDelta replay capture for exact prefix restore. Routed experts
        // run one unisolated gather per verify block; B1 routing keeps the
        // per-token expert products of ordinary decoding.
        let _position_stable_linear = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_linear::scope);
        let _position_stable_qmm = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_qmm::scope);
        let _dflash2_bulk_attention = mode
            .requires_position_stability()
            .then(crate::nn::position_stable_qmm::dflash2_bulk_attention_scope);
        let _product_stable_qmm =
            (mode.is_verify() && verify_width > 1).then(crate::nn::product_stable_qmm::scope);
        // M5 MoE affine4 projections (experimental): every target forward of
        // the stream, including ordinary Q=1 decoding, uses the same
        // tensor-unit arithmetic so verify and serial decoding agree.
        let _m5_affine4 = self
            .m5_affine4_projections()
            .then(crate::nn::m5_affine4::scope);
        // M5 MoE route: rows of every expert share one weight stream while
        // each row keeps the MLX gather_qmv_fast arithmetic.
        let _grouped_experts = (mode.is_verify() && self.m5_grouped_route(verify_width))
            .then(crate::nn::moe_grouped_qmv::scope);
        let (hidden, context_hidden) = self.text().forward_with_dflash2_taps_on(
            input_ids,
            position_ids,
            cache,
            target_layer_ids,
            target,
        )?;
        Ok(DFlash2TargetOutput {
            hidden,
            context_hidden,
        })
    }

    /// Whether every DFlash2 target forward and projection of this target
    /// runs the experimental M5 affine4 projections. They are exact against
    /// the same path decoding token by token, not against ordinary MLX
    /// decoding, so the execution fingerprint names them.
    /// The setting cannot bypass the hardware: the kernels need the
    /// tensor units of a real Apple GPU of generation 17 or newer; elsewhere
    /// the product-stable projections run and the fingerprint says so.
    fn m5_affine4_projections(&self) -> bool {
        self.dflash2_target_bits == Some(4)
            && ironmlx_core::m5_profile::tensor_unit_flag(
                ironmlx_core::m5_profile::settings::M5_MOE_AFFINE4_PROJECTIONS,
            )
    }

    /// Whether a verify of `rows` target rows runs the M5 expert-grouped
    /// route on this target.
    fn m5_grouped_route(&self, rows: usize) -> bool {
        m5_grouped_experts()
            && self
                .dflash2_target_bits
                .and_then(qwen36_moe_grouped_experts_min_rows)
                .is_some_and(|min_rows| rows >= min_rows)
    }

    fn dflash2_max_verify_width(&self) -> usize {
        self.dflash2_target_bits
            .map(qwen36_moe_dflash2_max_verify_width)
            .unwrap_or(0)
    }
}

impl DFlash2Target for Qwen35MoeModel {
    fn dflash2_target_spec(&self) -> DFlash2TargetSpec {
        self.config().into()
    }

    fn dflash2_prewarm(&self) -> Result<()> {
        if self.dflash2_target_bits.is_none() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let target = StreamOrDevice::default();
        let mut cache = self.make_cache(1, 64, crate::core::Model::cache_dtype(self))?;
        let ids = |start: u32, len: usize| -> Result<Array> {
            let tokens: Vec<u32> = (0..len as u32).map(|i| start + i).collect();
            Ok((&tokens[..], &[1_i32, len as i32][..]).try_into()?)
        };
        let taps = [0_usize];
        let prefill = self.dflash2_forward_target_on(
            &ids(1000, 8)?,
            &crate::core::model_input::build_position_ids(0, 8)?,
            Some(&mut cache),
            &taps,
            DFlash2TargetForwardMode::Prefill,
            target,
        )?;
        mlx::transforms::eval(&[&prefill.hidden, &prefill.context_hidden])?;
        let base: Vec<_> = cache.iter().map(LayerCache::snapshot).collect();
        let widths = 1..=self.dflash2_max_verify_width().max(1);
        for width in widths {
            for layer in cache.iter_mut() {
                layer.begin_speculative_prefix_capture()?;
            }
            let mode = if width == 1 {
                DFlash2TargetForwardMode::OrdinaryDecode
            } else {
                DFlash2TargetForwardMode::GreedyVerify
            };
            let output = self.dflash2_forward_target_on(
                &ids(2000, width)?,
                &crate::core::model_input::build_position_ids(8, width as i32)?,
                Some(&mut cache),
                &taps,
                mode,
                target,
            )?;
            let logits = self.dflash2_project_hidden_on(&output.hidden, target)?;
            mlx::transforms::eval(&[&logits, &output.context_hidden])?;
            for (layer, snapshot) in cache.iter_mut().zip(&base) {
                layer.discard_speculative_prefix_capture();
                layer.restore(snapshot)?;
            }
        }
        drop(cache);
        mlx::clear_cache();
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Qwen3.6 MoE DFlash2 kernels prewarmed"
        );
        Ok(())
    }

    fn dflash2_draft_fast_paths(&self) -> bool {
        self.text().affine4_fast_paths()
    }

    /// Qualified Qwen3.6 MoE recipes run the B1 linear window-cost policy.
    fn dflash2_window_cost_budget_policy(&self) -> bool {
        self.dflash2_target_bits.is_some()
    }

    fn dflash2_instance(&self) -> Option<&DFlash2Instance> {
        Some(&self.instance)
    }

    fn dflash2_draft_vocab_enabled(&self) -> Result<bool> {
        Ok(self.draft_vocab()?.is_some())
    }

    fn dflash2_target_cache_cost(&self) -> DFlash2TargetCacheCost {
        let cfg = self.config();
        let positive = |value: i32, field: &str| {
            usize::try_from(value)
                .unwrap_or_else(|_| panic!("validated Qwen3.5 MoE {field} must be non-negative"))
        };
        DFlash2HybridCacheGeometry {
            layers: positive(cfg.num_hidden_layers, "layer count"),
            full_attention_interval: positive(
                cfg.full_attention_interval,
                "full-attention interval",
            ),
            kv_heads: positive(cfg.num_key_value_heads, "KV head count"),
            head_dim: positive(cfg.effective_head_dim(), "head dimension"),
            linear_value_heads: positive(cfg.linear_num_value_heads, "linear value head count"),
            linear_key_heads: positive(cfg.linear_num_key_heads, "linear key head count"),
            linear_key_dim: positive(cfg.linear_key_head_dim, "linear key dimension"),
            linear_value_dim: positive(cfg.linear_value_head_dim, "linear value dimension"),
            linear_conv_kernel: positive(cfg.linear_conv_kernel_dim, "linear convolution width"),
        }
        .cache_cost()
    }

    fn dflash2_verify_capabilities(&self) -> DFlash2VerifyCapabilities {
        // B1 only: batched MoE verification interleaves expert routes across
        // requests and has not been qualified for this target.
        let supported_shapes = (2..=self.dflash2_max_verify_width())
            .map(|verify_width| DFlash2VerifyShape {
                batch_width: 1,
                verify_width,
            })
            .collect::<Vec<_>>();
        let certified = !supported_shapes.is_empty();
        DFlash2VerifyCapabilities {
            profile: match self.dflash2_target_bits {
                Some(bits)
                    if m5_grouped_experts()
                        && qwen36_moe_grouped_experts_min_rows(bits).is_some() =>
                {
                    format!(
                        "qwen36-moe-affine{bits}-b1-v1+m5-grouped-experts-v2-q{}",
                        qwen36_moe_grouped_experts_min_rows(bits).unwrap_or(0)
                    )
                }
                Some(bits) => format!("qwen36-moe-affine{bits}-b1-v1"),
                None => "qwen35-moe-unqualified".to_owned(),
            },
            row_bit_exact_qmm: certified,
            row_bit_exact_attention: certified,
            transactional_state_restore: certified,
            supported_shapes,
            lane_kernel_pack: None,
            flat_tree_max_nodes: self
                .dflash2_target_bits
                .map(qwen36_moe_dflash2_flat_tree_max_nodes)
                .unwrap_or(0),
        }
    }

    fn dflash2_execution_fingerprint(&self) -> String {
        let experts = match self
            .dflash2_target_bits
            .and_then(qwen36_moe_grouped_experts_min_rows)
        {
            Some(min_rows) if m5_grouped_experts() => {
                format!("expert-grouped-gather-qmv-v2-from-q{min_rows}")
            }
            _ => "gather-qmm-unisolated".to_owned(),
        };
        let projections = if self.m5_affine4_projections() {
            "m5-affine4-v1"
        } else {
            "product-stable-v2"
        };
        format!(
            "qwen36-moe-dflash2-v1;experts={experts};projections={projections};attention=bulk-position-stable-v1;recurrent=logical-prefix-v2;{}",
            self.dflash2_verify_capabilities().stable_fingerprint()
        )
    }

    fn dflash2_embed_on(&self, input_ids: &Array, target: StreamOrDevice) -> Result<Array> {
        self.text().embed_on(input_ids, target)
    }

    fn dflash2_forward_tree_on(
        &self,
        input_ids: &Array,
        parents: &[i32],
        start: i32,
        cache: &mut [LayerCache],
        target_layer_ids: &[usize],
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        let nodes = parents.len().saturating_sub(1);
        let max_nodes = self.dflash2_verify_capabilities().flat_tree_max_nodes;
        anyhow::ensure!(
            nodes >= 1 && nodes <= max_nodes,
            "Qwen3.6 MoE DFlash2 flat tree with {nodes} nodes is outside the certified 1..={max_nodes}"
        );
        anyhow::ensure!(
            input_ids.shape().as_slice() == [1, parents.len() as i32],
            "Qwen3.6 MoE DFlash2 flat tree input must be [1,{}], got {:?}",
            parents.len(),
            input_ids.shape().as_slice()
        );
        // Each node attends to the prompt and its own ancestors, advances
        // the GatedDelta state from its parent's state, and takes the RoPE
        // position of its depth.
        let plan = crate::nn::dflash_tree::Plan::new(parents)?;
        let positions = plan.positions(start)?;
        let _tree = crate::nn::dflash_tree::enter(plan)?;
        self.forward_dflash2_on(
            input_ids,
            &positions,
            Some(cache),
            target_layer_ids,
            DFlash2TargetForwardMode::GreedyVerify,
            target,
        )
    }

    fn dflash2_commit_tree_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[LayerCacheSnapshot],
        rows: &[i32],
        target: StreamOrDevice,
    ) -> Result<()> {
        anyhow::ensure!(
            cache.len() == snapshots.len() && !rows.is_empty() && rows[0] == 0,
            "Qwen3.6 MoE DFlash2 tree commit requires one snapshot per layer and a root-first path"
        );
        // Keep only the accepted root-to-node path: attention K/V rows are
        // compacted to contiguous positions and GatedDelta replays the path
        // from the pre-verify state.
        for (live, saved) in cache.iter_mut().zip(snapshots) {
            match (live, saved) {
                (LayerCache::Full(kv), LayerCacheSnapshot::Full(saved)) => {
                    kv.commit_tree_rows(saved.offsets()[0], rows, target)?
                }
                (LayerCache::Linear(gdn), LayerCacheSnapshot::Linear(_)) => {
                    gdn.select_tree_replay_rows(rows, target)?
                }
                _ => anyhow::bail!("Qwen3.6 MoE DFlash2 tree commit received incompatible cache"),
            }
        }
        self.text()
            .restore_dflash2_speculative_prefix_on(cache, snapshots, rows.len(), target)
    }

    fn dflash2_forward_target_on(
        &self,
        input_ids: &Array,
        position_ids: &Array,
        cache: Option<&mut [LayerCache]>,
        target_layer_ids: &[usize],
        mode: DFlash2TargetForwardMode,
        target: StreamOrDevice,
    ) -> Result<DFlash2TargetOutput> {
        anyhow::ensure!(
            crate::nn::dflash_tree::current().is_none(),
            "Qwen3.6 MoE linear DFlash2 forward called inside a flat tree scope"
        );
        let input_shape = input_ids.shape();
        let input_dims = input_shape.as_slice();
        if input_dims.len() == 2 && input_dims[0] == 1 && input_dims[1] > 0 {
            let verify_width = input_dims[1] as usize;
            if mode.is_verify()
                && verify_width > 1
                && !self.dflash2_verify_capabilities().supports(1, verify_width)
            {
                return Err(anyhow!(
                    "Qwen3.6 MoE DFlash2 verify is not qualified for B1/Q{verify_width} on this target"
                ));
            }
        }
        self.forward_dflash2_on(
            input_ids,
            position_ids,
            cache,
            target_layer_ids,
            mode,
            target,
        )
    }

    fn dflash2_restore_target_prefix_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[LayerCacheSnapshot],
        accepted_len: usize,
        target: StreamOrDevice,
    ) -> Result<()> {
        self.text()
            .restore_dflash2_speculative_prefix_on(cache, snapshots, accepted_len, target)
    }

    fn dflash2_restore_target_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: StreamOrDevice,
    ) -> Result<()> {
        self.text().restore_dflash2_speculative_prefix_rows_on(
            cache,
            snapshots,
            accepted_lens,
            target,
        )
    }

    fn dflash2_project_hidden_on(&self, hidden: &Array, target: StreamOrDevice) -> Result<Array> {
        let shape = hidden.shape();
        let dims = shape.as_slice();
        if dims.len() != 3 || dims[0] <= 0 {
            return Err(anyhow!(
                "Qwen3.6 MoE DFlash2 projection requires [B,S,H] with B>0, got {dims:?}"
            ));
        }
        let _m5_affine4 = self
            .m5_affine4_projections()
            .then(crate::nn::m5_affine4::scope);
        // Multi-position projection uses the product-stable affine kernel so
        // each row keeps the ordinary Q=1 accumulation tree.
        let _product_stable = (dims[1] > 1).then(crate::nn::product_stable_qmm::scope);
        let _fast_paths = self
            .text()
            .affine4_fast_paths()
            .then(crate::nn::moe_fast_path::scope);
        self.lm_head().forward_on(hidden, target)
    }

    fn dflash2_draft_vocab_project_on(
        &self,
        hidden: &Array,
        extra: &[u32],
        target: StreamOrDevice,
    ) -> Result<Option<(Array, Array)>> {
        let Some(vocab) = self.draft_vocab()? else {
            return Ok(None);
        };
        let _fast_paths = crate::nn::moe_fast_path::scope();
        vocab
            .project_on(self.lm_head(), hidden, extra, target)
            .map(Some)
    }

    fn dflash2_draft_vocab_lacks(&self, token: u32) -> Result<bool> {
        Ok(self
            .draft_vocab()?
            .is_some_and(|vocab| !vocab.contains(token)))
    }
}

impl Qwen35MoeModel {
    /// The DFlash2 draft vocabulary of an affine4 target, built on first use.
    fn draft_vocab(&self) -> Result<Option<&super::draft_vocab::DraftVocab>> {
        if !self.text().affine4_fast_paths() {
            return Ok(None);
        }
        let vocab = match self.draft_vocab.get() {
            Some(vocab) => vocab,
            None => {
                let built = super::draft_vocab::DraftVocab::from_lm_head(self.lm_head())?;
                self.draft_vocab.get_or_init(|| built)
            }
        };
        Ok(vocab.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::layer::LayerCache;
    use crate::core::model_input::build_position_ids;
    use crate::core::Model;
    use serial_test::serial;

    const TAPS: [usize; 8] = [1, 6, 11, 16, 22, 27, 32, 37];
    const PREFILL_CHUNK: usize = 2048;
    const TEXT: &str =
        "DFlash2 drafts a block of tokens from tapped target hidden states, and the \
        target verifies the block in one forward. Accepted tokens are committed, the first \
        mismatch is replaced by the target's own token, and every cache is restored to the \
        accepted prefix before generation continues. ";

    fn checkpoint() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("QWEN36_MOE_DFLASH2_TARGET").ok()?);
        path.is_dir().then_some(path)
    }

    fn contexts() -> Vec<usize> {
        std::env::var("QWEN36_MOE_DFLASH2_CONTEXTS")
            .unwrap_or_else(|_| "8,64,1024,8192".to_owned())
            .split(',')
            .map(|value| value.trim().parse().expect("context length"))
            .collect()
    }

    fn ids(tokens: &[u32]) -> Array {
        (tokens, &[1_i32, tokens.len() as i32][..])
            .try_into()
            .expect("input ids")
    }

    fn row(array: &Array, index: usize) -> Array {
        let dims = array.shape();
        let dims = dims.as_slice();
        mlx::ops::indexing::slice_strided(
            array,
            &[0_i32, index as i32, 0][..],
            &[1_i32, index as i32 + 1, dims[2]][..],
            &[1_i32, 1, 1][..],
        )
        .expect("slice row")
    }

    fn assert_exact(label: &str, expected: &Array, actual: &Array) {
        assert_eq!(expected.shape(), actual.shape(), "{label}: shape");
        let to_f32 = |array: &Array| {
            mlx::ops::cast::astype(array, mlx::Dtype::Float32)
                .expect("cast")
                .to_vec::<f32>()
                .expect("read")
        };
        let (expected, actual) = (to_f32(expected), to_f32(actual));
        let mismatches = expected
            .iter()
            .zip(&actual)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        assert_eq!(mismatches, 0, "{label}: {mismatches} elements differ");
    }

    fn argmax(logits: &Array) -> u32 {
        mlx::ops::reduction::argmax(logits, -1, false)
            .expect("argmax")
            .reshape((-1,))
            .expect("flatten")
            .to_vec::<u32>()
            .expect("read argmax")
            .last()
            .copied()
            .expect("argmax value")
    }

    fn assert_cache_exact(label: &str, expected: &[LayerCache], actual: &[LayerCache]) {
        for (layer, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            match (expected, actual) {
                (LayerCache::Full(expected), LayerCache::Full(actual)) => {
                    assert_eq!(
                        expected.offsets(),
                        actual.offsets(),
                        "{label}: layer {layer} KV offsets"
                    );
                    let offset = expected.offsets()[0];
                    for (index, (left, right)) in expected
                        .diagnostic_buffers()
                        .into_iter()
                        .zip(actual.diagnostic_buffers())
                        .enumerate()
                    {
                        let prefix = |array: &Array| {
                            let dims = array.shape();
                            let dims = dims.as_slice();
                            mlx::ops::indexing::slice_strided(
                                array,
                                &[0_i32, 0, 0, 0][..],
                                &[dims[0], dims[1], offset, dims[3]][..],
                                &[1_i32, 1, 1, 1][..],
                            )
                            .expect("KV prefix")
                        };
                        assert_exact(
                            &format!("{label}: layer {layer} KV buffer {index}"),
                            &prefix(left),
                            &prefix(right),
                        );
                    }
                }
                (LayerCache::Linear(expected), LayerCache::Linear(actual)) => {
                    assert_eq!(
                        expected.offsets(),
                        actual.offsets(),
                        "{label}: layer {layer} GDN offsets"
                    );
                    assert_exact(
                        &format!("{label}: layer {layer} GDN conv state"),
                        expected.conv_state(),
                        actual.conv_state(),
                    );
                    assert_exact(
                        &format!("{label}: layer {layer} GDN recurrent state"),
                        expected.recurrent_state(),
                        actual.recurrent_state(),
                    );
                }
                _ => panic!("{label}: layer {layer} cache kind mismatch"),
            }
        }
    }

    /// Ordinary decoding reference: the scheduler/GenerationStream graph.
    /// `QWEN36_MOE_DFLASH2_REFERENCE=same-path` makes every reference step a
    /// DFlash2 target forward (prefill / ordinary decode) of the same
    /// execution profile, for routes that change arithmetic versus MLX (B-type
    /// qualification). Otherwise the reference is ordinary MLX decoding.
    fn same_path_reference() -> bool {
        std::env::var("QWEN36_MOE_DFLASH2_REFERENCE").as_deref() == Ok("same-path")
    }

    fn reference_project(model: &Qwen35MoeModel, hidden: &Array) -> Array {
        if same_path_reference() {
            model
                .dflash2_project_hidden_on(hidden, StreamOrDevice::default())
                .expect("same-path logits")
        } else {
            Model::project_hidden_on(model, hidden, StreamOrDevice::default()).expect("logits")
        }
    }

    fn ordinary_step(
        model: &Qwen35MoeModel,
        tokens: &[u32],
        start: usize,
        cache: &mut [LayerCache],
    ) -> (Array, Array) {
        if same_path_reference() {
            let mode = if tokens.len() > 1 {
                DFlash2TargetForwardMode::Prefill
            } else {
                DFlash2TargetForwardMode::OrdinaryDecode
            };
            let (output, logits) = dflash_step(model, tokens, start, cache, mode);
            return (output.hidden, logits);
        }
        let positions = build_position_ids(start as i32, tokens.len() as i32).expect("positions");
        let hidden = Model::forward_text_hidden(
            model,
            &ids(tokens),
            &positions,
            None,
            None,
            Some(cache),
            StreamOrDevice::default(),
        )
        .expect("ordinary forward");
        let logits =
            Model::project_hidden_on(model, &hidden, StreamOrDevice::default()).expect("logits");
        mlx::transforms::eval(&[&hidden, &logits]).expect("eval ordinary");
        (hidden, logits)
    }

    fn dflash_step(
        model: &Qwen35MoeModel,
        tokens: &[u32],
        start: usize,
        cache: &mut [LayerCache],
        mode: DFlash2TargetForwardMode,
    ) -> (DFlash2TargetOutput, Array) {
        let positions = build_position_ids(start as i32, tokens.len() as i32).expect("positions");
        let output = model
            .dflash2_forward_target_on(
                &ids(tokens),
                &positions,
                Some(cache),
                &TAPS,
                mode,
                StreamOrDevice::default(),
            )
            .expect("DFlash2 target forward");
        let logits = model
            .dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())
            .expect("DFlash2 logits");
        mlx::transforms::eval(&[&output.hidden, &output.context_hidden, &logits])
            .expect("eval DFlash2");
        (output, logits)
    }

    fn begin_transaction(cache: &mut [LayerCache]) -> Vec<LayerCacheSnapshot> {
        let snapshots = cache
            .iter()
            .map(|layer| layer.dflash2_transaction_snapshot().expect("snapshot"))
            .collect();
        for layer in cache.iter_mut() {
            layer.begin_speculative_prefix_capture().expect("capture");
        }
        snapshots
    }

    fn restore(cache: &mut [LayerCache], snapshots: &[LayerCacheSnapshot]) {
        for (layer, snapshot) in cache.iter_mut().zip(snapshots) {
            layer.discard_speculative_prefix_capture();
            layer.restore(snapshot).expect("restore snapshot");
        }
    }

    /// Real-checkpoint DFlash2 qualification of one Qwen3.6 MoE target bit
    /// width (`QWEN36_MOE_DFLASH2_TARGET=<snapshot>`). For every context
    /// length and every certified verify width it checks, against ordinary
    /// Q=1 decoding of the same checkpoint: the prefill state, all eight
    /// taps, the final hidden state and the logits of each verify row in
    /// greedy and sampled verify modes, then full, zero and partial
    /// acceptance with prefix restore, the bonus/correction continuation and
    /// the resulting GDN and attention cache state.
    #[test]
    #[ignore = "loads a full local Qwen3.6 MoE checkpoint (QWEN36_MOE_DFLASH2_TARGET)"]
    #[serial(mlx_metal)]
    fn qwen36_moe_dflash2_verify_matches_ordinary_decode() {
        let Some(dir) = checkpoint() else {
            eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET to a Qwen3.6 MoE snapshot");
            return;
        };
        let loader = crate::core::Loader::open(&dir).expect("open target");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen35MoeModel::from_loader(&loader).expect("load target");
        let bits = model
            .dflash2_target_bits()
            .expect("checkpoint must match a qualified Qwen3.6 MoE recipe");
        let max_q = qwen36_moe_dflash2_max_verify_width(bits);
        assert!(max_q >= 2, "affine{bits} certifies no verify width");
        let capabilities = model.dflash2_verify_capabilities();
        assert!(!capabilities.certifies_batched_execution());
        assert!(capabilities.supports(1, max_q) && !capabilities.supports(1, max_q + 1));
        assert!(!capabilities.supports(2, 2));
        let mut text_tokens = tokenizer.encode(TEXT, false).expect("encode");
        let max_context = contexts().into_iter().max().unwrap_or(8);
        while text_tokens.len() < max_context + max_q + 8 {
            let copy = text_tokens.clone();
            text_tokens.extend(copy);
        }

        for context in contexts() {
            let prompt = &text_tokens[..context];
            let verify = &text_tokens[context..context + max_q];
            let cap = (context + max_q + 16) as i32;
            let mut ordinary = model
                .make_cache(1, cap, model.cache_dtype())
                .expect("cache");
            let mut dflash = model
                .make_cache(1, cap, model.cache_dtype())
                .expect("cache");

            // Prefill both caches with the same chunking. Generation
            // projects only the last prompt position (a single row), so the
            // first-token logits are compared on that row.
            let mut last = None;
            for start in (0..context).step_by(PREFILL_CHUNK) {
                let end = (start + PREFILL_CHUNK).min(context);
                let (ordinary_hidden, _) =
                    ordinary_step(&model, &prompt[start..end], start, &mut ordinary);
                let (dflash_output, _) = dflash_step(
                    &model,
                    &prompt[start..end],
                    start,
                    &mut dflash,
                    DFlash2TargetForwardMode::Prefill,
                );
                last = Some((ordinary_hidden, dflash_output.hidden, end - start));
            }
            let (ordinary_hidden, dflash_hidden, chunk) = last.expect("prefill chunk");
            let ordinary_last = row(&ordinary_hidden, chunk - 1);
            let dflash_last = row(&dflash_hidden, chunk - 1);
            assert_exact(
                &format!("C{context} prefill last hidden"),
                &ordinary_last,
                &dflash_last,
            );
            assert_exact(
                &format!("C{context} prefill first-token logits"),
                &reference_project(&model, &ordinary_last),
                &model
                    .dflash2_project_hidden_on(&dflash_last, StreamOrDevice::default())
                    .expect("DFlash2 first logits"),
            );
            assert_cache_exact(&format!("C{context} prefill cache"), &ordinary, &dflash);
            let ordinary_base: Vec<_> = ordinary.iter().map(LayerCache::snapshot).collect();
            let dflash_base: Vec<_> = dflash.iter().map(LayerCache::snapshot).collect();

            // Per-position references: ordinary Q=1 logits/hidden and the
            // DFlash2 ordinary-decode taps, which must agree with each other.
            let mut reference_logits = Vec::new();
            let mut reference_hidden = Vec::new();
            let mut reference_taps = Vec::new();
            for (index, &token) in verify.iter().enumerate() {
                let (hidden, logits) =
                    ordinary_step(&model, &[token], context + index, &mut ordinary);
                let (output, dflash_logits) = dflash_step(
                    &model,
                    &[token],
                    context + index,
                    &mut dflash,
                    DFlash2TargetForwardMode::OrdinaryDecode,
                );
                let label = format!("C{context} ordinary-decode position {index}");
                assert_exact(&format!("{label} hidden"), &hidden, &output.hidden);
                assert_exact(&format!("{label} logits"), &logits, &dflash_logits);
                assert_eq!(
                    output.context_hidden.shape().as_slice(),
                    &[1, 1, 8 * 2048],
                    "{label} tap width"
                );
                reference_logits.push(logits);
                reference_hidden.push(hidden);
                reference_taps.push(output.context_hidden);
            }

            for width in 1..=max_q {
                // Sampled verify must produce the same rows as greedy verify.
                restore(&mut dflash, &dflash_base);
                let snapshots = begin_transaction(&mut dflash);
                let (sampled, sampled_logits) = dflash_step(
                    &model,
                    &verify[..width],
                    context,
                    &mut dflash,
                    DFlash2TargetForwardMode::SampledVerify,
                );
                drop(snapshots);
                for index in 0..width {
                    let label = format!("C{context} Q{width} sampled row {index}");
                    assert_exact(
                        &format!("{label} logits"),
                        &reference_logits[index],
                        &row(&sampled_logits, index),
                    );
                    assert_exact(
                        &format!("{label} taps"),
                        &reference_taps[index],
                        &row(&sampled.context_hidden, index),
                    );
                }

                let mut accepted_lengths = vec![1, width.div_ceil(2), width];
                accepted_lengths.dedup();
                for accepted in accepted_lengths {
                    let label = format!("C{context} Q{width} accepted={accepted}");
                    restore(&mut dflash, &dflash_base);
                    let snapshots = begin_transaction(&mut dflash);
                    let (output, logits) = dflash_step(
                        &model,
                        &verify[..width],
                        context,
                        &mut dflash,
                        DFlash2TargetForwardMode::GreedyVerify,
                    );
                    for index in 0..width {
                        assert_exact(
                            &format!("{label} row {index} logits"),
                            &reference_logits[index],
                            &row(&logits, index),
                        );
                        assert_exact(
                            &format!("{label} row {index} hidden"),
                            &reference_hidden[index],
                            &row(&output.hidden, index),
                        );
                        assert_exact(
                            &format!("{label} row {index} taps"),
                            &reference_taps[index],
                            &row(&output.context_hidden, index),
                        );
                    }
                    // Full acceptance keeps the verify state and continues
                    // with the bonus token; otherwise restore the accepted
                    // prefix and continue with the target correction token.
                    if accepted == width {
                        for layer in dflash.iter_mut() {
                            layer.discard_speculative_prefix_capture();
                        }
                    } else {
                        model
                            .dflash2_restore_target_prefix_on(
                                &mut dflash,
                                &snapshots,
                                accepted,
                                StreamOrDevice::default(),
                            )
                            .expect("restore accepted prefix");
                    }
                    let next = argmax(&row(&logits, accepted - 1));

                    restore(&mut ordinary, &ordinary_base);
                    for (index, &token) in verify[..accepted].iter().enumerate() {
                        ordinary_step(&model, &[token], context + index, &mut ordinary);
                    }
                    assert_cache_exact(&format!("{label} committed cache"), &ordinary, &dflash);
                    let mut continuation = vec![next];
                    let mut continuation_logits = Vec::new();
                    for step in 0..3 {
                        let (_, logits) = ordinary_step(
                            &model,
                            &[continuation[step]],
                            context + accepted + step,
                            &mut ordinary,
                        );
                        if step < 2 {
                            continuation.push(argmax(&logits));
                        }
                        continuation_logits.push(logits);
                    }
                    let snapshots = begin_transaction(&mut dflash);
                    let (_, logits) = dflash_step(
                        &model,
                        &continuation,
                        context + accepted,
                        &mut dflash,
                        DFlash2TargetForwardMode::GreedyVerify,
                    );
                    drop(snapshots);
                    for layer in dflash.iter_mut() {
                        layer.discard_speculative_prefix_capture();
                    }
                    for (index, expected) in continuation_logits.iter().enumerate() {
                        assert_exact(
                            &format!("{label} continuation row {index}"),
                            expected,
                            &row(&logits, index),
                        );
                    }
                    assert_cache_exact(&format!("{label} cache"), &ordinary, &dflash);
                    eprintln!("[qwen36-moe-dflash2 affine{bits}] {label}: exact");
                }
            }
        }
        if model.m5_grouped_route(crate::nn::moe_grouped_qmv::MAX_RUN as usize) {
            let dispatches = crate::nn::moe_grouped_qmv::dispatch_count();
            assert!(
                dispatches > 0,
                "grouped expert route was armed but never dispatched"
            );
            eprintln!("[qwen36-moe-dflash2 affine{bits}] grouped expert dispatches={dispatches}");
        }
    }

    /// Flat-tree shapes (parents of nodes 1..; node 0 is the root). Chain,
    /// star, binary best-first and an unbalanced deep tree, all within the
    /// certified node budget.
    fn tree_shapes(max_nodes: usize) -> Vec<(&'static str, Vec<i32>)> {
        let mut shapes = Vec::new();
        shapes.push(("chain7", (0..7).collect()));
        shapes.push(("star8", vec![0; 8]));
        let mut binary = Vec::new();
        for node in 1..=max_nodes {
            binary.push(if node <= 2 {
                0
            } else {
                ((node - 1) / 2) as i32
            });
        }
        shapes.push(("binary", binary));
        shapes.push((
            "unbalanced",
            vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 8, 10, 3, 4, 13][..max_nodes.min(15)].to_vec(),
        ));
        shapes
            .into_iter()
            .map(|(name, parents)| {
                let mut all = vec![-1];
                all.extend(parents.into_iter().take(max_nodes));
                (name, all)
            })
            .collect()
    }

    fn path_to(parents: &[i32], node: usize) -> Vec<usize> {
        let mut path = vec![node];
        while parents[*path.last().unwrap()] >= 0 {
            path.push(parents[*path.last().unwrap()] as usize);
        }
        path.reverse();
        path
    }

    /// Real-checkpoint flat-tree qualification of one Qwen3.6 MoE target bit
    /// width. Every tree node must match ordinary Q=1 decoding of its own
    /// ancestor path (logits, final hidden, all eight taps), so a node can
    /// see only the prompt and its ancestors. Committing an accepted path
    /// (root only, a mid path and the deepest path) must leave attention
    /// K/V, GDN recurrent and convolution state identical to sequential
    /// decoding of that path, and the bonus continuation must match.
    #[test]
    #[ignore = "loads a full local Qwen3.6 MoE checkpoint (QWEN36_MOE_DFLASH2_TARGET)"]
    #[serial(mlx_metal)]
    fn qwen36_moe_dflash2_tree_matches_ancestor_decode() {
        let Some(dir) = checkpoint() else {
            eprintln!("skip: set QWEN36_MOE_DFLASH2_TARGET to a Qwen3.6 MoE snapshot");
            return;
        };
        let loader = crate::core::Loader::open(&dir).expect("open target");
        let tokenizer = crate::core::Tokenizer::from_loader(&loader).expect("tokenizer");
        let model = Qwen35MoeModel::from_loader(&loader).expect("load target");
        let bits = model.dflash2_target_bits().expect("qualified recipe");
        let max_nodes = model.dflash2_verify_capabilities().flat_tree_max_nodes;
        assert!(max_nodes >= 1, "affine{bits} certifies no flat tree");
        let mut text_tokens = tokenizer.encode(TEXT, false).expect("encode");
        let max_context = contexts().into_iter().max().unwrap_or(8);
        while text_tokens.len() < max_context + max_nodes + 8 {
            let copy = text_tokens.clone();
            text_tokens.extend(copy);
        }

        for context in contexts() {
            let prompt = &text_tokens[..context];
            let cap = (context + max_nodes + 16) as i32;
            let mut ordinary = model
                .make_cache(1, cap, model.cache_dtype())
                .expect("cache");
            let mut dflash = model
                .make_cache(1, cap, model.cache_dtype())
                .expect("cache");
            for start in (0..context).step_by(PREFILL_CHUNK) {
                let end = (start + PREFILL_CHUNK).min(context);
                ordinary_step(&model, &prompt[start..end], start, &mut ordinary);
                dflash_step(
                    &model,
                    &prompt[start..end],
                    start,
                    &mut dflash,
                    DFlash2TargetForwardMode::Prefill,
                );
            }
            let ordinary_base: Vec<_> = ordinary.iter().map(LayerCache::snapshot).collect();
            let dflash_base: Vec<_> = dflash.iter().map(LayerCache::snapshot).collect();

            for (name, parents) in tree_shapes(max_nodes) {
                let width = parents.len();
                // Node tokens: the root is the next prompt token, the rest
                // are distinct text tokens so siblings differ.
                let tokens = (0..width)
                    .map(|node| text_tokens[context + node])
                    .collect::<Vec<_>>();

                // References: decode each node after its parent's state.
                let mut ordinary_states: Vec<Option<Vec<LayerCacheSnapshot>>> =
                    (0..width).map(|_| None).collect();
                let mut reference = Vec::with_capacity(width);
                for node in 0..width {
                    let parent = parents[node];
                    if parent < 0 {
                        restore(&mut ordinary, &ordinary_base);
                    } else {
                        restore(
                            &mut ordinary,
                            ordinary_states[parent as usize]
                                .as_ref()
                                .expect("parent state"),
                        );
                    }
                    let depth = path_to(&parents, node).len() - 1;
                    let (hidden, logits) =
                        ordinary_step(&model, &[tokens[node]], context + depth, &mut ordinary);
                    ordinary_states[node] = Some(
                        ordinary
                            .iter()
                            .map(LayerCache::snapshot)
                            .collect::<Vec<_>>(),
                    );
                    reference.push((hidden, logits));
                }
                let mut tap_reference = Vec::with_capacity(width);
                let mut dflash_states: Vec<Option<Vec<LayerCacheSnapshot>>> =
                    (0..width).map(|_| None).collect();
                for node in 0..width {
                    let parent = parents[node];
                    if parent < 0 {
                        restore(&mut dflash, &dflash_base);
                    } else {
                        restore(
                            &mut dflash,
                            dflash_states[parent as usize]
                                .as_ref()
                                .expect("parent state"),
                        );
                    }
                    let depth = path_to(&parents, node).len() - 1;
                    let (output, _) = dflash_step(
                        &model,
                        &[tokens[node]],
                        context + depth,
                        &mut dflash,
                        DFlash2TargetForwardMode::OrdinaryDecode,
                    );
                    assert_exact(
                        &format!("C{context} {name} node {node} ordinary-decode hidden"),
                        &reference[node].0,
                        &output.hidden,
                    );
                    dflash_states[node] =
                        Some(dflash.iter().map(LayerCache::snapshot).collect::<Vec<_>>());
                    tap_reference.push(output.context_hidden);
                }

                let deepest = (0..width)
                    .max_by_key(|&node| (path_to(&parents, node).len(), std::cmp::Reverse(node)))
                    .expect("deepest node");
                let deepest_path = path_to(&parents, deepest);
                let mid = deepest_path[deepest_path.len() / 2];
                let mut commit_nodes = vec![0, mid, deepest];
                commit_nodes.dedup();
                for commit in commit_nodes {
                    let label = format!("C{context} {name}(W{width}) commit node {commit}");
                    restore(&mut dflash, &dflash_base);
                    let snapshots = begin_transaction(&mut dflash);
                    let output = model
                        .dflash2_forward_tree_on(
                            &ids(&tokens),
                            &parents,
                            context as i32,
                            &mut dflash,
                            &TAPS,
                            StreamOrDevice::default(),
                        )
                        .expect("tree verify");
                    let logits = model
                        .dflash2_project_hidden_on(&output.hidden, StreamOrDevice::default())
                        .expect("tree logits");
                    mlx::transforms::eval(&[&output.hidden, &output.context_hidden, &logits])
                        .expect("eval tree");
                    for node in 0..width {
                        assert_exact(
                            &format!("{label} node {node} logits"),
                            &reference[node].1,
                            &row(&logits, node),
                        );
                        assert_exact(
                            &format!("{label} node {node} hidden"),
                            &reference[node].0,
                            &row(&output.hidden, node),
                        );
                        assert_exact(
                            &format!("{label} node {node} taps"),
                            &tap_reference[node],
                            &row(&output.context_hidden, node),
                        );
                    }
                    let path = path_to(&parents, commit);
                    let rows = path.iter().map(|&node| node as i32).collect::<Vec<_>>();
                    model
                        .dflash2_commit_tree_on(
                            &mut dflash,
                            &snapshots,
                            &rows,
                            StreamOrDevice::default(),
                        )
                        .expect("commit accepted path");
                    restore(
                        &mut ordinary,
                        ordinary_states[commit].as_ref().expect("committed state"),
                    );
                    assert_cache_exact(&format!("{label} committed cache"), &ordinary, &dflash);

                    // Bonus continuation after the committed path.
                    let next = argmax(&row(&logits, commit));
                    let mut continuation = vec![next];
                    let mut continuation_logits = Vec::new();
                    for step in 0..3 {
                        let (_, logits) = ordinary_step(
                            &model,
                            &[continuation[step]],
                            context + path.len() + step,
                            &mut ordinary,
                        );
                        if step < 2 {
                            continuation.push(argmax(&logits));
                        }
                        continuation_logits.push(logits);
                    }
                    let snapshots = begin_transaction(&mut dflash);
                    let (_, logits) = dflash_step(
                        &model,
                        &continuation,
                        context + path.len(),
                        &mut dflash,
                        DFlash2TargetForwardMode::GreedyVerify,
                    );
                    drop(snapshots);
                    for layer in dflash.iter_mut() {
                        layer.discard_speculative_prefix_capture();
                    }
                    for (index, expected) in continuation_logits.iter().enumerate() {
                        assert_exact(
                            &format!("{label} continuation row {index}"),
                            expected,
                            &row(&logits, index),
                        );
                    }
                    assert_cache_exact(&format!("{label} continued cache"), &ordinary, &dflash);
                    eprintln!("[qwen36-moe-dflash2-tree affine{bits}] {label}: exact");
                }
            }
        }
        if model.m5_grouped_route(crate::nn::moe_grouped_qmv::MAX_RUN as usize) {
            let dispatches = crate::nn::moe_grouped_qmv::dispatch_count();
            assert!(
                dispatches > 0,
                "grouped expert route was armed but never dispatched"
            );
            eprintln!(
                "[qwen36-moe-dflash2-tree affine{bits}] grouped expert dispatches={dispatches}"
            );
        }
    }
}
