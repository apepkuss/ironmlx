use anyhow::{anyhow, Context};
use mlx::ops::indexing::{slice_on, take_along_axis_on, take_on};
use mlx::ops::sort::{argpartition_on, argsort_on};
use mlx::{Array, StreamOrDevice};

use crate::core::Loader;
use crate::nn::Linear;
use crate::Result;

const SORTED_ROUTING_MIN_BS_K: i32 = 64;
const MAX_EXACT_U32_IN_F32: i32 = 1 << 24;

pub struct DiffusionGemmaRouter {
    proj: Linear,
    scale: Array,
    per_expert_scale: Array,
    hidden_root: f32,
    top_k: i32,
    num_experts: i32,
    eps: f32,
}

pub struct DiffusionGemmaExperts {
    gate_up_weight: Array,
    gate_up_scales: Array,
    gate_up_biases: Option<Array>,
    down_weight: Array,
    down_scales: Array,
    down_biases: Option<Array>,
    group_size: i32,
    bits: i32,
    moe_intermediate: i32,
}

impl DiffusionGemmaRouter {
    pub fn from_loader(
        loader: &Loader,
        prefix: &str,
        hidden_size: i32,
        num_experts: i32,
        top_k: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_loader(loader, &format!("{prefix}.proj"))
                .with_context(|| format!("loading DiffusionGemma router `{prefix}`"))?,
            scale: loader
                .tensor(&format!("{prefix}.scale"))
                .with_context(|| format!("loading DiffusionGemma router scale `{prefix}`"))?
                .clone(),
            per_expert_scale: loader
                .tensor(&format!("{prefix}.per_expert_scale"))
                .with_context(|| {
                    format!("loading DiffusionGemma router per_expert_scale `{prefix}`")
                })?
                .clone(),
            hidden_root: (hidden_size as f32).powf(-0.5),
            top_k,
            num_experts,
            eps,
        })
    }

    pub fn route_on(&self, x: &Array, target: StreamOrDevice) -> Result<(Array, Array)> {
        let x = mlx::fast::rms_norm_on(x, None, self.eps, target)?;
        let scale: Array = (&[self.hidden_root][..], ()).try_into()?;
        let x = &(&x * &self.scale) * &scale;
        let logits = self.proj.forward_on(&x, target)?;
        let part = argpartition_on(&logits, -self.top_k, -1, target)
            .context("DiffusionGemmaRouter: argpartition top-k")?;
        let shape = logits.shape();
        let dims = shape.as_slice();
        let bs = dims[0];
        let inds = slice_on(
            &part,
            [0_i32, self.num_experts - self.top_k],
            [bs, self.num_experts],
            target,
        )
        .context("DiffusionGemmaRouter: slice top-k indices")?;
        let weights_raw = take_along_axis_on(&logits, &inds, -1, target)
            .context("DiffusionGemmaRouter: take top-k logits")?;
        let weights = mlx::ops::softmax_on(&weights_raw, -1_i32, true, target)
            .context("DiffusionGemmaRouter: softmax top-k logits")?;
        let expert_scale = take_on(&self.per_expert_scale, &inds, 0, target)
            .context("DiffusionGemmaRouter: take per-expert scale")?;
        let weights = &weights * &expert_scale;
        let inds_u32 = mlx::ops::cast::astype_on(&inds, mlx::Dtype::Uint32, target)
            .context("DiffusionGemmaRouter: cast top-k indices")?;
        Ok((weights, inds_u32))
    }
}

impl DiffusionGemmaExperts {
    pub fn from_loader(loader: &Loader, prefix: &str) -> Result<Self> {
        let gate_up_prefix = format!("{prefix}.gate_up_proj");
        let down_prefix = format!("{prefix}.down_proj");
        let qmeta = loader
            .quant_meta_for(&gate_up_prefix)
            .ok_or_else(|| anyhow!("DiffusionGemmaExperts requires quantization metadata"))?;
        let gate_up_weight = loader.tensor(&format!("{gate_up_prefix}.weight"))?.clone();
        let gate_up_scales = loader.tensor(&format!("{gate_up_prefix}.scales"))?.clone();
        let gate_up_biases = loader
            .tensor_opt(&format!("{gate_up_prefix}.biases"))
            .cloned();
        let down_weight = loader.tensor(&format!("{down_prefix}.weight"))?.clone();
        let down_scales = loader.tensor(&format!("{down_prefix}.scales"))?.clone();
        let down_biases = loader.tensor_opt(&format!("{down_prefix}.biases")).cloned();
        let dims = gate_up_weight.shape();
        let shape = dims.as_slice();
        if shape.len() != 3 || shape[1] % 2 != 0 {
            return Err(anyhow!(
                "DiffusionGemmaExperts: gate_up weight must be [E,2I,K], got {:?}",
                shape
            ));
        }
        Ok(Self {
            gate_up_weight,
            gate_up_scales,
            gate_up_biases,
            down_weight,
            down_scales,
            down_biases,
            group_size: qmeta.group_size,
            bits: qmeta.bits,
            moe_intermediate: shape[1] / 2,
        })
    }

    pub fn forward_on(
        &self,
        x: &Array,
        inds: &Array,
        weights: &Array,
        target: StreamOrDevice,
    ) -> Result<Array> {
        let xdims = x.shape();
        let xshape = xdims.as_slice();
        if xshape.len() != 2 {
            return Err(anyhow!(
                "DiffusionGemmaExperts: x must be [BS,H], got {:?}",
                xshape
            ));
        }
        let (bs, h) = (xshape[0], xshape[1]);
        let idims = inds.shape();
        let ishape = idims.as_slice();
        if ishape.len() != 2 || ishape[0] != bs {
            return Err(anyhow!(
                "DiffusionGemmaExperts: inds must be [BS,k], got {:?}",
                ishape
            ));
        }
        let k = ishape[1];
        let bs_k = bs * k;
        let use_sorted = bs_k >= SORTED_ROUTING_MIN_BS_K;

        let (gate, up, rhs_idx, sorted, sort_perm) = if use_sorted {
            let flat_topk = mlx::ops::shape::reshape(inds, [bs_k])
                .context("DiffusionGemmaExperts: reshape inds")?;
            let sort_perm =
                argsort_on(&flat_topk, -1_i32, target).context("DiffusionGemmaExperts: argsort")?;
            let sorted_topk = take_along_axis_on(&flat_topk, &sort_perm, -1_i32, target)
                .context("DiffusionGemmaExperts: sort top-k")?;
            let sorted_token_idx =
                sorted_token_indices_from_sort_perm(&sort_perm, k, bs_k, target)?;
            let sorted_x = take_on(x, &sorted_token_idx, 0_i32, target)
                .context("DiffusionGemmaExperts: gather sorted x")?;
            let sorted_x = mlx::ops::shape::expand_dims_on(&sorted_x, -2_i32, target)?;
            let gate_up = mlx::quantization::gather_quantized_matmul_on(
                &sorted_x,
                &self.gate_up_weight,
                &self.gate_up_scales,
                self.gate_up_biases.as_ref(),
                None,
                Some(&sorted_topk),
                true,
                Some(self.group_size),
                Some(self.bits),
                "affine",
                true,
                target,
            )
            .context("DiffusionGemmaExperts: sorted gate_up gather_qmm")?;
            let i = self.moe_intermediate;
            let gate = slice_on(&gate_up, [0_i32, 0, 0], [bs_k, 1, i], target)?;
            let up = slice_on(&gate_up, [0_i32, 0, i], [bs_k, 1, 2 * i], target)?;
            (gate, up, sorted_topk, true, Some(sort_perm))
        } else {
            let x_in = mlx::ops::shape::expand_dims_on(x, &[-2_i32, -3_i32][..], target)?;
            let gate_up = mlx::quantization::gather_quantized_matmul_on(
                &x_in,
                &self.gate_up_weight,
                &self.gate_up_scales,
                self.gate_up_biases.as_ref(),
                None,
                Some(inds),
                true,
                Some(self.group_size),
                Some(self.bits),
                "affine",
                false,
                target,
            )
            .context("DiffusionGemmaExperts: gate_up gather_qmm")?;
            let i = self.moe_intermediate;
            let gate = slice_on(&gate_up, [0_i32, 0, 0, 0], [bs, k, 1, i], target)?;
            let up = slice_on(&gate_up, [0_i32, 0, 0, i], [bs, k, 1, 2 * i], target)?;
            (gate, up, inds.clone(), false, None)
        };

        let act = geglu_on(&gate, &up, target)?;
        let down_raw = mlx::quantization::gather_quantized_matmul_on(
            &act,
            &self.down_weight,
            &self.down_scales,
            self.down_biases.as_ref(),
            None,
            Some(&rhs_idx),
            true,
            Some(self.group_size),
            Some(self.bits),
            "affine",
            sorted,
            target,
        )
        .context("DiffusionGemmaExperts: down gather_qmm")?;

        let down = if let Some(sort_perm) = sort_perm {
            let inv = argsort_on(&sort_perm, -1_i32, target)?;
            let down_2d = mlx::ops::shape::reshape(&down_raw, [bs_k, h])?;
            let unsorted = take_on(&down_2d, &inv, 0_i32, target)?;
            mlx::ops::shape::reshape(&unsorted, [bs, k, h])?
        } else {
            mlx::ops::shape::squeeze_on(&down_raw, &[-2_i32][..], target)?
        };
        let weights = mlx::ops::shape::expand_dims_on(weights, -1_i32, target)?;
        let weighted = &down * &weights;
        Ok(mlx::ops::sum_on(&weighted, -2_i32, false, target)?)
    }
}

fn geglu_on(gate: &Array, up: &Array, target: StreamOrDevice) -> Result<Array> {
    let gate = crate::nn::activations::gelu_tanh(gate, target)?;
    Ok(&gate * up)
}

fn sorted_token_indices_from_sort_perm(
    sort_perm: &Array,
    k: i32,
    bs_k: i32,
    target: StreamOrDevice,
) -> Result<Array> {
    if k <= 0 {
        return Err(anyhow!("DiffusionGemmaExperts: top-k must be positive"));
    }
    if bs_k <= MAX_EXACT_U32_IN_F32 {
        let sort_perm_f32 = mlx::ops::cast::astype_on(sort_perm, mlx::Dtype::Float32, target)?;
        let k_scalar: Array = (&[k as f32][..], ()).try_into()?;
        let div = sort_perm_f32.try_div_on(&k_scalar, target)?;
        let idx = div.floor_on(target)?;
        return mlx::ops::cast::astype_on(&idx, mlx::Dtype::Uint32, target)
            .context("DiffusionGemmaExperts: cast sorted token idx");
    }

    let bs_k_usize = usize::try_from(bs_k)?;
    let k_usize = usize::try_from(k)?;
    let token_idx_vec: Vec<u32> = (0..bs_k_usize).map(|i| (i / k_usize) as u32).collect();
    let token_idx: Array = (token_idx_vec.as_slice(), [bs_k]).try_into()?;
    take_along_axis_on(&token_idx, sort_perm, -1_i32, target)
        .context("DiffusionGemmaExperts: take sorted token idx")
}
