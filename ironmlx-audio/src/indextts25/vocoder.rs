//! Fixed BigVGAN V2 (80 mel bins, 256x) with anti-aliased SnakeBeta activations.
use super::layers::{finite, scalar, scale, Weights};
use crate::{error::invalid, Result, SessionControl};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array};

pub(super) struct Vocoder {
    weights: Weights,
    filter: Array,
}
impl Vocoder {
    pub fn new(weights: WeightMap) -> Result<Self> {
        // Fixed reference activations.kaiser_sinc_filter1d(0.25, 0.3, 12),
        // normalized in NumPy float64 then stored as float32. No model data.
        let coefficients = [
            0x3b04f869, 0x3c19d644, 0xbcd14087, 0xbd6c2a24, 0x3e03a88a, 0x3ee2ec65, 0x3ee2ec65,
            0x3e03a88a, 0xbd6c2a24, 0xbcd14087, 0x3c19d644, 0x3b04f869,
        ]
        .map(f32::from_bits);
        Ok(Self {
            weights: Weights(weights),
            filter: Array::try_from((&coefficients[..], [1, 12, 1]))?,
        })
    }
    fn conv(&self, x: &Array, name: &str, padding: i32, dilation: i32) -> Result<Array> {
        let mut y = ops::conv1d(
            x,
            self.weights.get(&format!("{name}.weight"))?,
            1,
            padding,
            dilation,
            1,
        )?;
        if let Some(bias) = self.weights.0.tensors().get(&format!("{name}.bias")) {
            y = ops::add(&y, bias)?;
        }
        Ok(y)
    }
    fn edge_pad(x: &Array, left: i32, right: i32) -> Result<Array> {
        let length = x.shape()[1];
        let indices: Vec<_> = (-left..length + right)
            .map(|i| i.clamp(0, length - 1))
            .collect();
        Ok(ops::take(
            x,
            &Array::try_from((indices.as_slice(), [indices.len() as i32]))?,
            1,
        )?)
    }
    fn activation(&self, x: &Array, name: &str) -> Result<Array> {
        let length = x.shape()[1];
        let channels = x.shape()[2];
        let filter = ops::broadcast_to(&self.filter, [channels, 12, 1])?;
        let x = scale(
            &ops::conv_transpose1d(&Self::edge_pad(x, 5, 5)?, &filter, 2, 0, 1, 0, channels)?,
            2.,
        )?;
        let x = ops::slice(&x, [0, 15, 0], [1, 15 + 2 * length, channels])?;
        let alpha = ops::exp(self.weights.get(&format!("{name}.act.alpha"))?)?;
        let beta = ops::exp(self.weights.get(&format!("{name}.act.beta"))?)?;
        // Reference scalar operations on beta retain its parameter dtype.
        let epsilon = scalar(1e-9)?.astype(beta.dtype())?;
        let one = scalar(1.)?.astype(beta.dtype())?;
        let reciprocal = ops::divide(&one, &ops::add(&beta, &epsilon)?)?;
        let periodic = ops::power(&ops::sin(&ops::multiply(&alpha, &x)?)?, &scalar(2.)?)?;
        let x = ops::add(&x, &ops::multiply(&reciprocal, &periodic)?)?;
        Ok(ops::conv1d(
            &Self::edge_pad(&x, 5, 6)?,
            &filter,
            2,
            0,
            1,
            channels,
        )?)
    }
    pub fn synthesize(&self, mel: &Array, control: &dyn SessionControl) -> Result<Array> {
        control.check()?;
        let shape = mel.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 80 || !(1..=5160).contains(&shape[2]) {
            return Err(invalid("vocoder mel", "expected bounded [1,80,T] input"));
        }
        finite(mel, "vocoder input")?;
        let mut x = self.conv(&mel.transpose_axes([0, 2, 1])?, "conv_pre", 3, 1)?;
        for (stage, rate) in [4, 4, 2, 2, 2, 2].into_iter().enumerate() {
            control.check()?;
            let base = format!("ups.{stage}");
            x = ops::add(
                &ops::conv_transpose1d(
                    &x,
                    self.weights.get(&format!("{base}.weight"))?,
                    rate,
                    rate / 2,
                    1,
                    0,
                    1,
                )?,
                self.weights.get(&format!("{base}.bias"))?,
            )?;
            let mut summed = None;
            for (branch, kernel) in [3, 7, 11].into_iter().enumerate() {
                let root = format!("resblocks.{}", stage * 3 + branch);
                let mut residual = x.clone();
                for (layer, dilation) in [1, 3, 5].into_iter().enumerate() {
                    control.check()?;
                    let y =
                        self.activation(&residual, &format!("{root}.activations.{}", layer * 2))?;
                    let y = self.conv(
                        &y,
                        &format!("{root}.convs1.{layer}"),
                        (kernel - 1) * dilation / 2,
                        dilation,
                    )?;
                    let y =
                        self.activation(&y, &format!("{root}.activations.{}", layer * 2 + 1))?;
                    let y =
                        self.conv(&y, &format!("{root}.convs2.{layer}"), (kernel - 1) / 2, 1)?;
                    residual = ops::add(&y, &residual)?;
                    residual.eval()?;
                }
                summed = Some(match summed {
                    None => residual,
                    Some(previous) => ops::add(&previous, &residual)?,
                });
            }
            x = ops::divide(
                &summed.ok_or_else(|| invalid("vocoder", "no residual branches"))?,
                &scalar(3.)?,
            )?;
        }
        let x = self.conv(&self.activation(&x, "activation_post")?, "conv_post", 3, 1)?;
        let result = ops::minimum(&ops::maximum(&x, &scalar(-1.)?)?, &scalar(1.)?)?
            .reshape([shape[2] * 256])?;
        finite(&result, "vocoder waveform")?;
        control.check()?;
        Ok(result)
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
    fn real_vocoder_parity() {
        let root = std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ACOUSTIC").unwrap())
                .unwrap();
        let model = Vocoder::new(
            load_component(
                &root.join("bigvgan.safetensors"),
                &IndexTts25Component::BigVgan.spec().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let actual = model
            .synthesize(&reference["generated_mel"], &Active)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = reference["audio"].to_vec::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        let errors: Vec<_> = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("vocoder max={max} mean={mean}");
        assert!(max < 2e-3 && mean < 2e-4);
    }
}
