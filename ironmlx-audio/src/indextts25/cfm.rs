//! Fixed non-causal S2Mel DiT/WaveNet estimator and one-step Euler progression.
use super::layers::{finite, scalar, scale, swish, Weights};
use crate::{error::invalid, AudioError, Result, SessionControl};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array, Dtype};

pub(super) struct Cfm {
    weights: Weights,
}
pub(super) struct CfmState {
    pub x: Array,
    prompt: Array,
    condition: Array,
    style: Array,
    prompt_frames: i32,
    step: usize,
    times: Vec<f32>,
    time: f32,
}
impl CfmState {
    pub fn finished(&self) -> bool {
        self.step == 25
    }
}
impl Cfm {
    pub fn new(weights: WeightMap) -> Self {
        Self {
            weights: Weights(weights),
        }
    }
    pub fn start(
        condition: &Array,
        prompt: &Array,
        style: &Array,
        noise: Array,
    ) -> Result<CfmState> {
        let cs = condition.shape();
        let ps = prompt.shape();
        if cs.len() != 3
            || cs[0] != 1
            || cs[2] != 512
            || !(87..=6452).contains(&cs[1])
            || ps.len() != 3
            || ps[0] != 1
            || ps[1] != 80
            || !(86..=1292).contains(&ps[2])
            || ps[2] >= cs[1]
            || style.shape().as_slice() != [1, 192]
            || noise.shape().as_slice() != [1, 80, cs[1]]
        {
            return Err(invalid(
                "CFM input",
                "invalid or unbounded condition, prompt, style or noise",
            ));
        }
        for (name, value) in [
            ("CFM condition", condition),
            ("CFM prompt", prompt),
            ("CFM style", style),
            ("CFM noise", &noise),
        ] {
            finite(value, name)?;
        }
        let total = cs[1];
        let prompt_frames = ps[2];
        let x = ops::concatenate(
            &[
                &Array::zeros([1, 80, prompt_frames], Dtype::Float32)?,
                &ops::slice(&noise, [0, 0, prompt_frames], [1, 80, total])?,
            ],
            2,
        )?;
        let prompt = ops::concatenate(
            &[
                prompt,
                &Array::zeros([1, 80, total - prompt_frames], Dtype::Float32)?,
            ],
            2,
        )?;
        let prompt = ops::concatenate(&[&prompt, &ops::zeros_like(&prompt)?], 0)?;
        let condition = ops::concatenate(&[condition, &ops::zeros_like(condition)?], 0)?;
        let style = ops::concatenate(&[style, &ops::zeros_like(style)?], 0)?;
        let times = ops::linspace(0., 1., 26, Dtype::Float32)?.to_vec::<f32>()?;
        Ok(CfmState {
            x,
            prompt,
            condition,
            style,
            prompt_frames,
            step: 0,
            times,
            time: 0.,
        })
    }
    fn time_embedding(&self, t: &Array, base: &str) -> Result<Array> {
        let frequencies = self.weights.get(&format!("{base}.freqs"))?;
        let phase = ops::multiply(
            &scale(&t.reshape([2, 1])?, 1000.)?,
            &frequencies.reshape([1, 128])?,
        )?;
        let embedded = ops::concatenate(&[&ops::cos(&phase)?, &ops::sin(&phase)?], -1)?;
        let w = &self.weights;
        w.linear_fused(
            &swish(&w.linear_fused(&embedded, &format!("{base}.linear1"))?)?,
            &format!("{base}.linear2"),
        )
    }
    fn norm(&self, x: &Array, condition: &Array, base: &str) -> Result<Array> {
        let w = &self.weights;
        let rms = ops::sqrt(&ops::add(
            &ops::mean(&ops::multiply(x, x)?, -1, true)?,
            &scalar(1e-5)?,
        )?)?;
        let norm = ops::multiply(
            &ops::divide(x, &rms)?,
            w.get(&format!("{base}.norm.weight"))?,
        )?;
        let projected = w.linear_fused(
            &condition.reshape([2, 1, 512])?,
            &format!("{base}.project_layer"),
        )?;
        let weight = ops::slice(&projected, [0, 0, 0], [2, 1, 512])?;
        let bias = ops::slice(&projected, [0, 0, 512], [2, 1, 1024])?;
        Ok(ops::add(&ops::multiply(&weight, &norm)?, &bias)?)
    }
    fn rope(x: &Array, cosine: &Array, sine: &Array) -> Result<Array> {
        let shape = x.shape();
        let t = shape[1];
        let paired = x.reshape([2, t, 8, 32, 2])?;
        let re = ops::slice(&paired, [0, 0, 0, 0, 0], [2, t, 8, 32, 1])?.reshape([2, t, 8, 32])?;
        let im = ops::slice(&paired, [0, 0, 0, 0, 1], [2, t, 8, 32, 2])?.reshape([2, t, 8, 32])?;
        let real = ops::subtract(&ops::multiply(&re, cosine)?, &ops::multiply(&im, sine)?)?;
        let imag = ops::add(&ops::multiply(&im, cosine)?, &ops::multiply(&re, sine)?)?;
        Ok(ops::stack(&[&real, &imag], -1)?.reshape([2, t, 8, 64])?)
    }
    fn conv(&self, x: &Array, base: &str, pad: i32) -> Result<Array> {
        let w = &self.weights;
        Ok(ops::add(
            &ops::conv1d(x, w.get(&format!("{base}.weight"))?, 1, pad, 1, 1)?,
            w.get(&format!("{base}.bias"))?,
        )?)
    }
    fn estimator(
        &self,
        x: &Array,
        prompt: &Array,
        condition: &Array,
        style: &Array,
        t: f32,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let w = &self.weights;
        let root = "cfm.estimator";
        let length = x.shape()[2];
        let time = Array::try_from((&[t, t][..], [2]))?;
        let t1 = self.time_embedding(&time, &format!("{root}.t_embedder"))?;
        let content = w.linear_fused(condition, &format!("{root}.cond_projection"))?;
        let original = x.transpose_axes([0, 2, 1])?;
        let style = ops::broadcast_to(&style.reshape([2, 1, 192])?, [2, length, 192])?;
        let merged = ops::concatenate(
            &[
                &original,
                &prompt.transpose_axes([0, 2, 1])?,
                &content,
                &style,
            ],
            -1,
        )?;
        let mut x = w.linear_fused(&merged, &format!("{root}.cond_x_merge_linear"))?;
        let frequencies: Vec<_> = (0..32).map(|i| (2 * i) as f32 / 64.).collect();
        let frequencies = ops::divide(
            &scalar(1.)?,
            &ops::power(
                &scalar(10000.)?,
                &Array::try_from((frequencies.as_slice(), [32]))?,
            )?,
        )?;
        let positions: Vec<_> = (0..length).map(|i| i as f32).collect();
        let phases = ops::multiply(
            &Array::try_from((positions.as_slice(), [length, 1]))?,
            &frequencies.reshape([1, 32])?,
        )?;
        let cosine = ops::cos(&phases)?.reshape([1, length, 1, 32])?;
        let sine = ops::sin(&phases)?.reshape([1, length, 1, 32])?;
        let mut skips = Vec::with_capacity(6);
        for layer in 0..13 {
            control.check()?;
            let base = format!("{root}.transformer.layers.{layer}");
            if layer > 6 {
                x = w.linear_fused(
                    &ops::concatenate(
                        &[
                            &x,
                            &skips
                                .pop()
                                .ok_or_else(|| invalid("CFM", "missing U-ViT skip"))?,
                        ],
                        -1,
                    )?,
                    &format!("{base}.skip_in_linear"),
                )?;
            }
            let norm = self.norm(&x, &t1, &format!("{base}.attention_norm"))?;
            let qkv = w.linear_fused(&norm, &format!("{base}.attention.wqkv"))?;
            let q = Self::rope(
                &ops::slice(&qkv, [0, 0, 0], [2, length, 512])?.reshape([2, length, 8, 64])?,
                &cosine,
                &sine,
            )?
            .transpose_axes([0, 2, 1, 3])?;
            let k = Self::rope(
                &ops::slice(&qkv, [0, 0, 512], [2, length, 1024])?.reshape([2, length, 8, 64])?,
                &cosine,
                &sine,
            )?
            .transpose_axes([0, 2, 3, 1])?;
            let v = ops::slice(&qkv, [0, 0, 1024], [2, length, 1536])?
                .reshape([2, length, 8, 64])?
                .transpose_axes([0, 2, 1, 3])?;
            // Both CFG rows have the full, identical valid length: the mask is all zero.
            let scores = scale(&ops::matmul(&q, &k)?, 0.125)?;
            let attended = ops::matmul(&ops::softmax(&scores, -1, false)?, &v)?
                .transpose_axes([0, 2, 1, 3])?
                .reshape([2, length, 512])?;
            x = ops::add(
                &x,
                &w.linear_fused(&attended, &format!("{base}.attention.wo"))?,
            )?;
            let norm = self.norm(&x, &t1, &format!("{base}.ffn_norm"))?;
            let activated = swish(&w.linear_fused(&norm, &format!("{base}.feed_forward.w1"))?)?;
            let gated = ops::multiply(
                &activated,
                &w.linear_fused(&norm, &format!("{base}.feed_forward.w3"))?,
            )?;
            x = ops::add(
                &x,
                &w.linear_fused(&gated, &format!("{base}.feed_forward.w2"))?,
            )?;
            x.eval()?;
            if layer < 6 {
                skips.push(x.clone());
            }
        }
        x = self.norm(&x, &t1, &format!("{root}.transformer.norm"))?;
        let residual = w.linear_fused(
            &ops::concatenate(&[&x, &original], -1)?,
            &format!("{root}.skip_linear"),
        )?;
        x = w.linear_fused(&residual, &format!("{root}.conv1"))?;
        let t2 = self.time_embedding(&time, &format!("{root}.t_embedder2"))?;
        let wn = format!("{root}.wavenet");
        let g = self.conv(
            &t2.reshape([2, 1, 512])?,
            &format!("{wn}.cond_layer.conv"),
            0,
        )?;
        let mut output = ops::zeros_like(&x)?;
        let indices: Vec<i32> = (0..length + 4)
            .map(|i| {
                let j = i - 2;
                if j < 0 {
                    -j
                } else if j >= length {
                    2 * length - j - 2
                } else {
                    j
                }
            })
            .collect();
        let indices = Array::try_from((indices.as_slice(), [length + 4]))?;
        for layer in 0..8 {
            control.check()?;
            let input = self.conv(
                &ops::take(&x, &indices, 1)?,
                &format!("{wn}.in_layers.{layer}.conv"),
                0,
            )?;
            let conditioned = ops::add(
                &input,
                &ops::slice(&g, [0, 0, layer * 1024], [2, 1, (layer + 1) * 1024])?,
            )?;
            let gate = ops::multiply(
                &ops::tanh(&ops::slice(&conditioned, [0, 0, 0], [2, length, 512])?)?,
                &ops::sigmoid(&ops::slice(&conditioned, [0, 0, 512], [2, length, 1024])?)?,
            )?;
            let res_skip = self.conv(&gate, &format!("{wn}.res_skip_layers.{layer}.conv"), 0)?;
            if layer < 7 {
                x = ops::add(&x, &ops::slice(&res_skip, [0, 0, 0], [2, length, 512])?)?;
                output = ops::add(
                    &output,
                    &ops::slice(&res_skip, [0, 0, 512], [2, length, 1024])?,
                )?;
            } else {
                output = ops::add(&output, &res_skip)?;
            }
        }
        x = ops::add(
            &output,
            &w.linear_fused(&residual, &format!("{root}.res_projection"))?,
        )?;
        let modulation = w.linear_fused(
            &swish(&t1)?,
            &format!("{root}.final_layer.adaLN_modulation.layers.1"),
        )?;
        let shift = ops::slice(&modulation, [0, 0], [2, 512])?.reshape([2, 1, 512])?;
        let factor = ops::slice(&modulation, [0, 512], [2, 1024])?.reshape([2, 1, 512])?;
        let mean = ops::mean(&x, -1, true)?;
        let centered = ops::subtract(&x, &mean)?;
        let variance = ops::mean(&ops::multiply(&centered, &centered)?, -1, true)?;
        x = ops::divide(
            &ops::subtract(&x, &mean)?,
            &ops::sqrt(&ops::add(&variance, &scalar(1e-6)?)?)?,
        )?;
        x = ops::add(
            &ops::multiply(&x, &ops::add(&factor, &scalar(1.)?)?)?,
            &shift,
        )?;
        x = w.linear_fused(&x, &format!("{root}.final_layer.linear"))?;
        Ok(self
            .conv(&x, &format!("{root}.conv2"), 0)?
            .transpose_axes([0, 2, 1])?)
    }
    pub fn advance(&self, state: &mut CfmState, control: &dyn SessionControl) -> Result<()> {
        control.check()?;
        if state.finished() {
            return Err(AudioError::InvalidSessionState);
        }
        let total = state.x.shape()[2];
        let stacked = ops::concatenate(&[&state.x, &state.x], 0)?;
        let prediction = self.estimator(
            &stacked,
            &state.prompt,
            &state.condition,
            &state.style,
            state.time,
            control,
        )?;
        let conditional = ops::slice(&prediction, [0, 0, 0], [1, 80, total])?;
        let null = ops::slice(&prediction, [1, 0, 0], [2, 80, total])?;
        let derivative = ops::subtract(&scale(&conditional, 1.7)?, &scale(&null, 0.7)?)?;
        let dt = state.times[state.step + 1] - state.times[state.step];
        let updated = ops::add(&state.x, &scale(&derivative, dt)?)?;
        state.x = ops::concatenate(
            &[
                &Array::zeros([1, 80, state.prompt_frames], Dtype::Float32)?,
                &ops::slice(&updated, [0, 0, state.prompt_frames], [1, 80, total])?,
            ],
            2,
        )?;
        finite(&state.x, "CFM Euler state")?;
        state.step += 1;
        state.time += dt;
        control.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{load_component, IndexTts25Component};
    struct Active;
    impl SessionControl for Active {
        fn check(&self) -> Result<()> {
            Ok(())
        }
    }
    #[test]
    #[ignore = "requires pinned snapshot and IRONMLX_REFERENCE_ACOUSTIC; MLX_ENABLE_TF32=0"]
    fn real_cfm_euler_parity() {
        let root = std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ACOUSTIC").unwrap())
                .unwrap();
        let model = Cfm::new(
            load_component(
                &root.join("s2mel.safetensors"),
                &IndexTts25Component::S2Mel.spec().unwrap(),
            )
            .unwrap(),
        );
        let mut state = Cfm::start(
            &reference["combined"],
            &reference["ref_mel"],
            &reference["style"],
            reference["noise"].clone(),
        )
        .unwrap();
        for step in 1..=25 {
            model.advance(&mut state, &Active).unwrap();
            assert_eq!(state.finished(), step == 25);
        }
        assert!(matches!(
            model.advance(&mut state, &Active),
            Err(AudioError::InvalidSessionState)
        ));
        assert_eq!(state.x.shape(), reference["mel"].shape());
        let actual = state.x.to_vec::<f32>().unwrap();
        let expected = reference["mel"].to_vec::<f32>().unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("CFM mel max={max} mean={mean}");
        assert!(max < 2e-3 && mean < 2e-4);
    }
}
