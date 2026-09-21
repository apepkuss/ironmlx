// CAMPPlus architecture adapted from Alibaba 3D-Speaker (Apache-2.0).
use super::layers::{finite, relu, scale, Weights};
use crate::{error::invalid, Result, SessionControl};
use mlx::{ops, Array};

pub(super) struct CampPlus {
    weights: Weights,
}
impl CampPlus {
    pub fn new(weights: ironmlx_core::weights::WeightMap) -> Self {
        Self {
            weights: Weights(weights),
        }
    }
    fn nonlinear(&self, x: &Array, base: &str) -> Result<Array> {
        relu(&self.weights.batch_norm(x, &format!("{base}.batchnorm"))?)
    }
    fn res_block(&self, x: &Array, base: &str, stride: i32) -> Result<Array> {
        let w = &self.weights;
        let y = relu(&w.batch_norm(
            &w.conv2(x, &format!("{base}.conv1"), (stride, 1), (1, 1))?,
            &format!("{base}.bn1"),
        )?)?;
        let y = w.batch_norm(
            &w.conv2(&y, &format!("{base}.conv2"), (1, 1), (1, 1))?,
            &format!("{base}.bn2"),
        )?;
        let shortcut = if stride == 1 {
            x.clone()
        } else {
            w.batch_norm(
                &w.conv2(x, &format!("{base}.shortcut.0"), (stride, 1), (0, 0))?,
                &format!("{base}.shortcut.1"),
            )?
        };
        relu(&ops::add(&y, &shortcut)?)
    }
    fn seg_pool(x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (time, channels) = (shape[1], shape[2]);
        let mut segments = Vec::new();
        for start in (0..time).step_by(100) {
            let end = (start + 100).min(time);
            let slice = ops::slice(x, [0, start, 0], [1, end, channels])?;
            segments.push(ops::broadcast_to(
                &ops::mean(&slice, 1, true)?,
                [1, end - start, channels],
            )?);
        }
        Ok(ops::concatenate(&segments.iter().collect::<Vec<_>>(), 1)?)
    }
    pub fn encode(&self, fbank: &Array, control: &dyn SessionControl) -> Result<Array> {
        control.check()?;
        let shape = fbank.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 80 || !(98..=1498).contains(&shape[1]) {
            return Err(invalid("campplus", "expected [1,T,80] reference fbank"));
        }
        finite(fbank, "speaker fbank")?;
        let w = &self.weights;
        let mut x = fbank
            .reshape([1, shape[1], 80, 1])?
            .transpose_axes([0, 2, 1, 3])?;
        x = relu(&w.batch_norm(&w.conv2(&x, "head.conv1", (1, 1), (1, 1))?, "head.bn1")?)?;
        for stage in 1..=2 {
            for block in 0..2 {
                control.check()?;
                x = self.res_block(
                    &x,
                    &format!("head.layer{stage}.{block}"),
                    if block == 0 { 2 } else { 1 },
                )?;
            }
        }
        x = relu(&w.batch_norm(&w.conv2(&x, "head.conv2", (2, 1), (1, 1))?, "head.bn2")?)?;
        // Preserve PyTorch's [channel,frequency] flatten order, not NHWC's reverse.
        x = x
            .transpose_axes([0, 2, 3, 1])?
            .reshape([1, shape[1], 320])?;
        x = self.nonlinear(
            &w.conv1(&x, "xvector.tdnn.linear", 2, 2, 1)?,
            "xvector.tdnn.nonlinear",
        )?;
        for (block, layers, dilation) in [(1, 12, 1), (2, 24, 2), (3, 16, 2)] {
            for layer in 1..=layers {
                control.check()?;
                let base = format!("xvector.block{block}.tdnnd{layer}");
                let y = w.conv1(
                    &self.nonlinear(&x, &format!("{base}.nonlinear1"))?,
                    &format!("{base}.linear1"),
                    1,
                    0,
                    1,
                )?;
                let y = self.nonlinear(&y, &format!("{base}.nonlinear2"))?;
                let local = w.conv1_dilated(
                    &y,
                    &format!("{base}.cam_layer.linear_local"),
                    1,
                    dilation,
                    dilation,
                    1,
                )?;
                let context = ops::add(&ops::mean(&y, 1, true)?, &Self::seg_pool(&y)?)?;
                let context =
                    relu(&w.conv1(&context, &format!("{base}.cam_layer.linear1"), 1, 0, 1)?)?;
                let mask = ops::sigmoid(&w.conv1(
                    &context,
                    &format!("{base}.cam_layer.linear2"),
                    1,
                    0,
                    1,
                )?)?;
                x = ops::concatenate(&[&x, &ops::multiply(&local, &mask)?], 2)?;
                // Bound graph retention across the dense block and expose a cancellation boundary.
                x.eval()?;
            }
            let base = format!("xvector.transit{block}");
            x = w.conv1(
                &self.nonlinear(&x, &format!("{base}.nonlinear"))?,
                &format!("{base}.linear"),
                1,
                0,
                1,
            )?;
        }
        x = self.nonlinear(&x, "xvector.out_nonlinear")?;
        let time = x.shape()[1];
        let mean = ops::mean(&x, 1, true)?;
        let centered = ops::subtract(&x, &mean)?;
        let var = scale(
            &ops::sum(&ops::multiply(&centered, &centered)?, 1, true)?,
            1. / (time - 1) as f32,
        )?;
        let pooled = ops::concatenate(&[&mean, &ops::sqrt(&var)?], 2)?;
        let result = w
            .batch_norm(
                &w.conv1(&pooled, "xvector.dense.linear", 1, 0, 1)?,
                "xvector.dense.nonlinear.batchnorm",
            )?
            .reshape([1, 192])?;
        result.eval()?;
        control.check()?;
        finite(&result, "CAMPPlus style")?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::derived::{load_auxiliary, AuxiliaryComponent};
    #[test]
    #[ignore = "requires derived resources and IRONMLX_REFERENCE_ENCODERS baseline"]
    fn real_campplus_style_parity() {
        struct Active;
        impl SessionControl for Active {
            fn check(&self) -> Result<()> {
                Ok(())
            }
        }
        let root = std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_DERIVED").unwrap());
        let model = CampPlus::new(load_auxiliary(&root, AuxiliaryComponent::CampPlus).unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ENCODERS").unwrap())
                .unwrap();
        let actual = model
            .encode(&reference["fbank"], &Active)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = reference["style"].to_vec::<f32>().unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let maximum = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("campplus style: max={maximum}, mean={mean}");
        assert!(maximum < 2e-3 && mean < 2e-4);
    }
}
