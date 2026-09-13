//! Model-side MTP capability and model implementations. No scheduling policy or generation loops.

use super::cache::MtpCache;
use super::{Loader, Model};
use crate::core::cache::layer::{LayerCache, LayerCacheSnapshot};
use crate::models::{Qwen35Model, Qwen35MoeModel, Qwen35MoeMtp, Qwen36MoeModel};
use crate::nn::{Mtp, MtpStepOutput};
use crate::Result;
use anyhow::anyhow;
use mlx::{Array, Dtype, StreamOrDevice};

/// Narrow model capability required by single-request MTP speculative decoding.
pub trait MtpSpeculativeModel: Model {
    type MtpHead;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead>;

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache>;

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32;

    fn mtp_hidden_dtype(&self, mtp: &Self::MtpHead) -> Dtype;

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Model::project_hidden_on(self, hidden, target.into())
    }

    fn supports_mtp_accepted_prefix_restore(&self) -> bool {
        false
    }

    fn supports_affine8_b4_mtp_exact_hot_path(
        &self,
        _batch_width: usize,
        _verify_width: usize,
    ) -> bool {
        false
    }

    fn begin_mtp_accepted_prefix_capture(&self, _cache: &mut [LayerCache]) -> Result<()> {
        Err(anyhow!(
            "{} does not support MTP accepted-prefix capture",
            std::any::type_name::<Self>()
        ))
    }

    fn restore_mtp_accepted_prefix_rows_on(
        &self,
        _cache: &mut [LayerCache],
        _snapshots: &[LayerCacheSnapshot],
        _accepted_lens: &[usize],
        _target: StreamOrDevice,
    ) -> Result<()> {
        Err(anyhow!(
            "{} does not support MTP accepted-prefix restore",
            std::any::type_name::<Self>()
        ))
    }

    fn discard_mtp_accepted_prefix_capture(&self, cache: &mut [LayerCache]) {
        for layer in cache {
            layer.discard_speculative_prefix_capture();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput>;
}

impl MtpSpeculativeModel for Qwen35Model {
    type MtpHead = Mtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35Model::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn supports_mtp_accepted_prefix_restore(&self) -> bool {
        true
    }

    fn supports_affine8_b4_mtp_exact_hot_path(
        &self,
        batch_width: usize,
        verify_width: usize,
    ) -> bool {
        Qwen35Model::supports_affine8_b4_mtp_exact_hot_path(self, batch_width, verify_width)
    }

    fn begin_mtp_accepted_prefix_capture(&self, cache: &mut [LayerCache]) -> Result<()> {
        for layer in cache {
            layer.begin_speculative_prefix_capture()?;
        }
        Ok(())
    }

    fn restore_mtp_accepted_prefix_rows_on(
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

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35Model::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen35MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen36MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen36MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen36MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}
