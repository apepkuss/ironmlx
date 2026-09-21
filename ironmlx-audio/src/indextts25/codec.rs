//! EnhancedCodecV25 semantic decoder. Published kernels already use MLX layout.
use super::layers::{finite, Weights};
use crate::{error::invalid, Result, SessionControl};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array};

pub(super) struct Codec {
    weights: Weights,
    gelu: mlx::compile::CompiledFn,
}
impl Codec {
    pub fn new(weights: WeightMap) -> Result<Self> {
        let gelu = mlx::compile::compile(
            |inputs| {
                let x = inputs[0];
                let scalar = |v| -> mlx::Result<Array> {
                    Array::try_from((&[v][..], []))?.astype(x.dtype())
                };
                let y = ops::divide(
                    &ops::multiply(
                        x,
                        &ops::add(
                            &scalar(1f32)?,
                            &ops::erf(&ops::divide(x, &scalar(2f32.sqrt())?)?)?,
                        )?,
                    )?,
                    &scalar(2f32)?,
                )?;
                Ok(vec![y])
            },
            mlx::compile::ShapeMode::Shapeless,
        )?;
        Ok(Self {
            weights: Weights(weights),
            gelu,
        })
    }
    fn conv(&self, x: &Array, name: &str, padding: i32, groups: i32) -> Result<Array> {
        let w = &self.weights;
        Ok(ops::add(
            &ops::conv1d(x, w.get(&format!("{name}.weight"))?, 1, padding, 1, groups)?,
            w.get(&format!("{name}.bias"))?,
        )?)
    }
    pub fn decode(&self, codes: &[u32], control: &dyn SessionControl) -> Result<Array> {
        self.decode_traced(codes, control, |_, _| Ok(()))
    }
    fn decode_traced(
        &self,
        codes: &[u32],
        control: &dyn SessionControl,
        mut trace: impl FnMut(&str, &Array) -> Result<()>,
    ) -> Result<Array> {
        control.check()?;
        if codes.is_empty() || codes.len() > 1500 || codes.iter().any(|&x| x >= 8192) {
            return Err(invalid(
                "semantic codes",
                "expected 1–1500 code IDs in 0..8192",
            ));
        }
        let w = &self.weights;
        let root = "quantizer.quantizers.0";
        let ids = Array::try_from((codes, [codes.len() as i32]))?;
        let x = ops::take(w.get(&format!("{root}.codebook.weight"))?, &ids, 0)?.reshape([
            1,
            codes.len() as i32,
            8,
        ])?;
        let mut x = self.conv(&x, &format!("{root}.out_project"), 0, 1)?;
        trace("quantized", &x)?;
        x = w.norm(
            &self.conv(&x, "decoder.0.embed", 3, 1)?,
            "decoder.0.norm",
            1e-6,
        )?;
        trace("embed", &x)?;
        for layer in 0..12 {
            control.check()?;
            let base = format!("decoder.0.convnext.{layer}");
            let y = self.conv(&x, &format!("{base}.dwconv"), 3, 384)?;
            trace(&format!("{layer}.dwconv"), &y)?;
            let y = w.norm(&y, &format!("{base}.norm"), 1e-6)?;
            trace(&format!("{layer}.norm"), &y)?;
            let y = w.linear_fused(&y, &format!("{base}.pwconv1"))?;
            trace(&format!("{layer}.gelu_input"), &y)?;
            let y = self.gelu.invoke(&[&y])?.remove(0);
            trace(&format!("{layer}.gelu"), &y)?;
            let y = w.linear_fused(&y, &format!("{base}.pwconv2"))?;
            x = ops::add(&x, &ops::multiply(&y, w.get(&format!("{base}.gamma"))?)?)?;
            x.eval()?;
            trace(&format!("block{layer}"), &x)?;
        }
        let x = w.linear_fused(
            &w.norm(&x, "decoder.0.final_layer_norm", 1e-6)?,
            "decoder.1",
        )?;
        let result = self.conv(&ops::repeat(&x, 2, 1)?, "up", 1, 1)?;
        finite(&result, "decoded semantic features")?;
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
    fn real_codec_decode_parity() {
        let root = std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ACOUSTIC").unwrap())
                .unwrap();
        let model = Codec::new(
            load_component(
                &root.join("codec.safetensors"),
                &IndexTts25Component::Codec.spec().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let ids = reference["codes"].to_vec::<u32>().unwrap();
        let actual = model
            .decode_traced(&ids, &Active, |name, value| {
                let expected = &reference[&format!("codec.{name}")];
                let a = value.astype(mlx::Dtype::Float32)?.to_vec::<f32>()?;
                let b = expected.astype(mlx::Dtype::Float32)?.to_vec::<f32>()?;
                let max = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                eprintln!("codec.{name}: dtype={:?} max={max}", value.dtype());
                Ok(())
            })
            .unwrap();
        assert_eq!(actual.shape(), reference["semantic"].shape());
        let actual = actual
            .astype(mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = reference["semantic"]
            .astype(mlx::Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("codec semantic max={max} mean={mean}");
        assert!(max < 2e-3 && mean < 2e-4);
    }
}
