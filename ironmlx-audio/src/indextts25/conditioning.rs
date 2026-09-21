// Adapted from the fixed mlx-indextts2 Conformer/Perceiver and length regulator.
use super::layers::{finite, relu, scalar, scale, swish, Weights};
use crate::{error::invalid, Result, SessionControl};
use mlx::{ops, Array};

pub(super) struct Conditioning {
    gpt: Weights,
    s2mel: Weights,
}

impl Conditioning {
    pub fn new(
        gpt: ironmlx_core::weights::WeightMap,
        s2mel: ironmlx_core::weights::WeightMap,
    ) -> Self {
        Self {
            gpt: Weights(gpt),
            s2mel: Weights(s2mel),
        }
    }
    // Published GPT/S2Mel convolution kernels already use MLX layout.
    fn conv(w: &Weights, x: &Array, name: &str, padding: i32, groups: i32) -> Result<Array> {
        let y = ops::conv1d(x, w.get(&format!("{name}.weight"))?, 1, padding, 1, groups)?;
        Ok(ops::add(&y, w.get(&format!("{name}.bias"))?)?)
    }
    fn check_semantic(x: &Array) -> Result<()> {
        let s = x.shape();
        if s.len() != 3 || s[0] != 1 || s[2] != 1024 || !(49..=750).contains(&s[1]) {
            return Err(invalid(
                "conditioning",
                "expected bounded [1,T,1024] semantic features",
            ));
        }
        finite(x, "semantic embedding")
    }
    pub fn emotion(&self, semantic: &Array, control: &dyn SessionControl) -> Result<Array> {
        control.check()?;
        Self::check_semantic(semantic)?;
        let w = &self.gpt;
        let root = "emo_conditioning_encoder";
        let input = ops::expand_dims(semantic, -1)?;
        let embed = ops::conv2d(
            &input,
            w.get(&format!("{root}.embed.conv.weight"))?,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )?;
        let embed = relu(&ops::add(
            &embed,
            w.get(&format!("{root}.embed.conv.bias"))?,
        )?)?;
        let t = embed.shape()[1];
        let embed = embed
            .transpose_axes([0, 1, 3, 2])?
            .reshape([1, t, 512 * 511])?;
        let mut x = scale(
            &w.linear(&embed, &format!("{root}.embed.out"))?,
            (512f32).sqrt(),
        )?;
        let mut positions = Vec::with_capacity(t as usize * 512);
        for position in 0..t {
            for channel in 0..256 {
                // NumPy computes div_term in float64, then stores the PE in float32.
                let angle =
                    f64::from(position) * (f64::from(channel * 2) * -(10000_f64.ln() / 512.)).exp();
                positions.extend([angle.sin() as f32, angle.cos() as f32]);
            }
        }
        let pos = Array::try_from((positions.as_slice(), [1, t, 512]))?;
        for i in 0..4 {
            control.check()?;
            let base = format!("{root}.encoders.{i}");
            let norm = w.norm(&x, &format!("{base}.norm_mha"), 1e-5)?;
            let attn = format!("{base}.self_attn");
            let q = w
                .linear(&norm, &format!("{attn}.linear_q"))?
                .reshape([1, t, 4, 128])?;
            let k = w
                .linear(&norm, &format!("{attn}.linear_k"))?
                .reshape([1, t, 4, 128])?
                .transpose_axes([0, 2, 3, 1])?;
            let v = w
                .linear(&norm, &format!("{attn}.linear_v"))?
                .reshape([1, t, 4, 128])?
                .transpose_axes([0, 2, 1, 3])?;
            let p = w
                .linear(&pos, &format!("{attn}.linear_pos"))?
                .reshape([1, t, 4, 128])?
                .transpose_axes([0, 2, 3, 1])?;
            let qu = ops::add(&q, w.get(&format!("{attn}.pos_bias_u"))?)?
                .transpose_axes([0, 2, 1, 3])?;
            let qv = ops::add(&q, w.get(&format!("{attn}.pos_bias_v"))?)?
                .transpose_axes([0, 2, 1, 3])?;
            let scores = scale(
                &ops::add(&ops::matmul(&qu, &k)?, &ops::matmul(&qv, &p)?)?,
                1. / 128f32.sqrt(),
            )?;
            let attended = ops::matmul(&ops::softmax(&scores, -1, false)?, &v)?
                .transpose_axes([0, 2, 1, 3])?
                .reshape([1, t, 512])?;
            x = ops::add(&x, &w.linear(&attended, &format!("{attn}.linear_out"))?)?;
            let conv = format!("{base}.conv_module");
            let norm = w.norm(&x, &format!("{base}.norm_conv"), 1e-5)?;
            let y = Self::conv(w, &norm, &format!("{conv}.pointwise_conv1"), 0, 1)?;
            let parts = ops::split_n(&y, 2, -1)?;
            let y = ops::multiply(&parts[0], &ops::sigmoid(&parts[1])?)?;
            let y = Self::conv(w, &y, &format!("{conv}.depthwise_conv"), 7, 512)?;
            let y = swish(&w.norm(&y, &format!("{conv}.norm"), 1e-5)?)?;
            x = ops::add(
                &x,
                &Self::conv(w, &y, &format!("{conv}.pointwise_conv2"), 0, 1)?,
            )?;
            let norm = w.norm(&x, &format!("{base}.norm_ff"), 1e-5)?;
            let y = swish(&w.linear(&norm, &format!("{base}.feed_forward.w_1"))?)?;
            x = ops::add(&x, &w.linear(&y, &format!("{base}.feed_forward.w_2"))?)?;
            x = w.norm(&x, &format!("{base}.norm_final"), 1e-5)?;
            x.eval()?;
        }
        let context = w.norm(&x, &format!("{root}.after_norm"), 1e-5)?;
        let root = "emo_perceiver_encoder";
        let context = w.linear(&context, &format!("{root}.proj_context"))?;
        let mut latent = w.get(&format!("{root}.latents"))?.reshape([1, 1, 1024])?;
        for i in 0..2 {
            control.check()?;
            let base = format!("{root}.layers.{i}");
            let both = ops::concatenate(&[&latent, &context], 1)?;
            let q = w
                .linear(&latent, &format!("{base}.0.linear_q"))?
                .reshape([1, 1, 4, 64])?
                .transpose_axes([0, 2, 1, 3])?;
            let k = w
                .linear(&both, &format!("{base}.0.linear_k"))?
                .reshape([1, t + 1, 4, 64])?
                .transpose_axes([0, 2, 3, 1])?;
            let v = w
                .linear(&both, &format!("{base}.0.linear_v"))?
                .reshape([1, t + 1, 4, 64])?
                .transpose_axes([0, 2, 1, 3])?;
            let scores = scale(&ops::matmul(&q, &k)?, 0.125)?;
            let attended = ops::matmul(&ops::softmax(&scores, -1, false)?, &v)?
                .transpose_axes([0, 2, 1, 3])?
                .reshape([1, 1, 256])?;
            latent = ops::add(
                &latent,
                &w.linear(&attended, &format!("{base}.0.linear_out"))?,
            )?;
            let y = w.linear(&latent, &format!("{base}.1.w_1"))?;
            let parts = ops::split_n(&y, 2, -1)?;
            let gelu = scale(
                &ops::multiply(
                    &parts[1],
                    &ops::add(
                        &scalar(1.)?,
                        &ops::erf(&scale(&parts[1], std::f32::consts::FRAC_1_SQRT_2)?)?,
                    )?,
                )?,
                0.5,
            )?;
            let y = ops::multiply(&parts[0], &gelu)?;
            latent = ops::add(&latent, &w.linear(&y, &format!("{base}.1.w_2"))?)?;
        }
        let rms = ops::sqrt(&ops::add(
            &ops::mean(&ops::multiply(&latent, &latent)?, -1, true)?,
            &scalar(1e-8)?,
        )?)?;
        latent = ops::multiply(
            &ops::divide(&latent, &rms)?,
            w.get(&format!("{root}.norm.weight"))?,
        )?
        .reshape([1, 1024])?;
        let emotion = w.linear(&w.linear(&latent, "emovec_layer")?, "emo_layer")?;
        emotion.eval()?;
        control.check()?;
        finite(&emotion, "emotion")?;
        Ok(emotion)
    }
    pub fn gpt_conditioning(&self, style: &Array, emotion: &Array) -> Result<Array> {
        if style.shape().as_slice() != [1, 192] || emotion.shape().as_slice() != [1, 1280] {
            return Err(invalid(
                "conditioning",
                "expected one speaker and emotion vector",
            ));
        }
        let first =
            ops::add(&self.gpt.linear(style, "spk_emb_proj")?, emotion)?.reshape([1, 1, 1280])?;
        let zero = Array::zeros([1, 2, 1280], first.dtype())?;
        let result = ops::concatenate(&[&first, &zero], 1)?;
        finite(&result, "GPT conditioning")?;
        Ok(result)
    }
    pub fn prompt(
        &self,
        semantic: &Array,
        frames: usize,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        control.check()?;
        Self::check_semantic(semantic)?;
        if !(86..=1292).contains(&frames) {
            return Err(invalid("mel frames", "outside reference duration bounds"));
        }
        length_regulate(&self.s2mel, semantic, frames, control)
    }
}

/// Shared length regulator; prompt and generated semantics use the same weights.
pub(super) fn length_regulate(
    w: &Weights,
    semantic: &Array,
    frames: usize,
    control: &dyn SessionControl,
) -> Result<Array> {
    control.check()?;
    let shape = semantic.shape();
    if shape.len() != 3
        || shape[0] != 1
        || shape[2] != 1024
        || !(1..=3000).contains(&shape[1])
        || !(1..=5160).contains(&frames)
    {
        return Err(invalid(
            "length regulator",
            "unbounded semantic or mel length",
        ));
    }
    finite(semantic, "length regulator input")?;
    let root = "length_regulator";
    let mut x = w.linear_fused(semantic, &format!("{root}.content_in_proj"))?;
    let length = x.shape()[1];
    let indices: Vec<i32> = (0..frames)
        .map(|i| ((i as f32) * (length as f32 / frames as f32)) as i32)
        .collect();
    x = ops::take(
        &x,
        &Array::try_from((indices.as_slice(), [frames as i32]))?,
        1,
    )?;
    for i in 0..4 {
        control.check()?;
        x = Conditioning::conv(w, &x, &format!("{root}.model.{}", i * 3), 1, 1)?;
        let norm = format!("{root}.model.{}", i * 3 + 1);
        // GroupNorm(groups=1) reduces time and channels for each batch row.
        x = x
            .transpose_axes([0, 2, 1])?
            .reshape([1, 1, 512, frames as i32])?;
        let mean = ops::mean(&x, [2, 3], true)?;
        let centered = ops::subtract(&x, &mean)?;
        let var = ops::mean(&ops::multiply(&centered, &centered)?, [2, 3], true)?;
        x = ops::divide(
            &centered,
            &ops::sqrt(&ops::add(&var, &scalar(1e-5)?.astype(x.dtype())?)?)?,
        )?;
        x = x
            .reshape([1, 512, frames as i32])?
            .transpose_axes([0, 2, 1])?;
        x = ops::add(
            &ops::multiply(&x, w.get(&format!("{norm}.weight"))?)?,
            w.get(&format!("{norm}.bias"))?,
        )?;
        // Preserve the input dtype in the reference Mish expression.
        x = ops::multiply(
            &x,
            &ops::tanh(&ops::log(&ops::add(
                &scalar(1.)?.astype(x.dtype())?,
                &ops::exp(&x)?,
            )?)?)?,
        )?;
        x.eval()?;
    }
    x = Conditioning::conv(w, &x, &format!("{root}.model.12"), 0, 1)?;
    // A single unpadded reference has an all-ones length mask.
    x.eval()?;
    control.check()?;
    finite(&x, "prompt conditioning")?;
    Ok(x.astype(mlx::Dtype::Float32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{load_component, IndexTts25Component};
    #[test]
    #[ignore = "requires pinned snapshot and IRONMLX_REFERENCE_ACOUSTIC; MLX_ENABLE_TF32=0"]
    fn real_generated_length_parity() {
        struct Active;
        impl SessionControl for Active {
            fn check(&self) -> Result<()> {
                Ok(())
            }
        }
        let snapshot =
            std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let w = Weights(
            load_component(
                &snapshot.join("s2mel.safetensors"),
                &IndexTts25Component::S2Mel.spec().unwrap(),
            )
            .unwrap(),
        );
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ACOUSTIC").unwrap())
                .unwrap();
        let actual = length_regulate(
            &w,
            &reference["semantic"],
            reference["condition"].shape()[1] as usize,
            &Active,
        )
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
        let expected = reference["condition"].to_vec::<f32>().unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("length regulator max={max} mean={mean}");
        assert!(max < 2e-3 && mean < 2e-4);
    }
    #[test]
    #[ignore = "requires pinned snapshot and IRONMLX_REFERENCE_ENCODERS; launch with MLX_ENABLE_TF32=0"]
    fn real_conditioning_parity() {
        struct Active;
        impl SessionControl for Active {
            fn check(&self) -> Result<()> {
                Ok(())
            }
        }
        let snapshot =
            std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let load = |component: IndexTts25Component| {
            load_component(
                &snapshot.join(component.file_name()),
                &component.spec().unwrap(),
            )
            .unwrap()
        };
        let model = Conditioning::new(
            load(IndexTts25Component::Gpt),
            load(IndexTts25Component::S2Mel),
        );
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ENCODERS").unwrap())
                .unwrap();
        let emotion = model.emotion(&reference["semantic"], &Active).unwrap();
        let gpt = model
            .gpt_conditioning(&reference["style"], &emotion)
            .unwrap();
        let prompt = model
            .prompt(
                &reference["semantic"],
                reference["mel"].shape()[1] as usize,
                &Active,
            )
            .unwrap();
        for (key, actual) in [
            ("emotion", emotion),
            ("conditioning", gpt),
            ("prompt_condition", prompt),
        ] {
            assert_eq!(actual.shape(), reference[key].shape());
            let actual = actual.to_vec::<f32>().unwrap();
            let expected = reference[key].to_vec::<f32>().unwrap();
            let errors: Vec<_> = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .collect();
            let maximum = errors.iter().copied().fold(0., f32::max);
            let mean = errors.iter().sum::<f32>() / errors.len() as f32;
            eprintln!("{key}: max={maximum}, mean={mean}");
            assert!(actual.iter().all(|x| x.is_finite()));
            assert!(maximum < 2e-3 && mean < 2e-4, "{key}");
        }
    }
}
