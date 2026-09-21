use super::layers::{finite, scale, swish, Weights};
use crate::{error::invalid, Result, SessionControl};
use mlx::{ops, Array, Dtype};

pub(super) struct W2vBert {
    weights: Weights,
}
impl W2vBert {
    pub fn new(weights: ironmlx_core::weights::WeightMap) -> Self {
        Self {
            weights: Weights(weights),
        }
    }
    fn ffn(&self, x: &Array, base: &str) -> Result<Array> {
        self.weights.linear(
            &swish(
                &self
                    .weights
                    .linear(x, &format!("{base}.intermediate_dense"))?,
            )?,
            &format!("{base}.output_dense"),
        )
    }
    /// Exactly the state after 17 encoder layers (hidden_states[17]); no final adapter.
    pub fn hidden17(
        &self,
        features: &Array,
        mask: &[i32],
        control: &dyn SessionControl,
    ) -> Result<Array> {
        control.check()?;
        if features.shape().as_slice() != [1, mask.len() as i32, 160]
            || mask.is_empty()
            || mask.len() > 750
            || mask.iter().any(|x| !matches!(x, 0 | 1))
            || mask.iter().all(|&x| x == 0)
        {
            return Err(invalid(
                "semantic_features",
                "expected bounded [1,T,160] features and a binary attention mask",
            ));
        }
        finite(features, "semantic features")?;
        let t = mask.len() as i32;
        let mask_f32: Vec<f32> = mask.iter().map(|&x| x as f32).collect();
        let valid = Array::try_from((mask_f32.as_slice(), [1, t, 1]))?;
        let bias: Vec<f32> = mask
            .iter()
            .map(|&x| if x == 0 { f32::MIN } else { 0. })
            .collect();
        let bias = Array::try_from((bias.as_slice(), [1, 1, 1, t]))?;
        let distances: Vec<i32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| (j - i).clamp(-64, 8) + 64))
            .collect();
        let distances = Array::try_from((distances.as_slice(), [1, 1, t, t]))?;
        let w = &self.weights;
        let mut x = w.linear(
            &w.norm(features, "feature_projection.layer_norm", 1e-5)?,
            "feature_projection.projection",
        )?;
        x = ops::multiply(&x, &valid)?;
        for layer in 0..17 {
            control.check()?;
            let base = format!("encoder.layers.{layer}");
            let ffn = self.ffn(
                &w.norm(&x, &format!("{base}.ffn1_layer_norm"), 1e-5)?,
                &format!("{base}.ffn1"),
            )?;
            x = ops::add(&x, &scale(&ffn, 0.5)?)?;
            let normalized = w.norm(&x, &format!("{base}.self_attn_layer_norm"), 1e-5)?;
            let attention = format!("{base}.self_attn");
            let project = |suffix| -> Result<Array> {
                Ok(
                    w.linear(&normalized, &format!("{attention}.linear_{suffix}"))?
                        .reshape([1, t, 16, 64])?
                        .transpose_axes([0, 2, 1, 3])?,
                )
            };
            let (q, k, v) = (project("q")?, project("k")?, project("v")?);
            let dot = ops::matmul(&q, &k.transpose_axes([0, 1, 3, 2])?)?;
            // Project onto all 73 relative keys, then gather per query/key distance.
            // This avoids materializing [heads,T,T,64] pairwise products.
            let relative = ops::matmul(
                &q,
                &w.get(&format!("{attention}.distance_embedding.weight"))?
                    .transpose()?,
            )?;
            let relative = ops::take_along_axis(&relative, &distances, -1)?;
            let scores = ops::add(&scale(&ops::add(&dot, &relative)?, 0.125)?, &bias)?;
            let attended = ops::matmul(&ops::softmax(&scores, -1, true)?, &v)?
                .transpose_axes([0, 2, 1, 3])?
                .reshape([1, t, 1024])?;
            x = ops::add(
                &x,
                &w.linear(&attended, &format!("{attention}.linear_out"))?,
            )?;
            let conv = format!("{base}.conv_module");
            let normalized =
                ops::multiply(&w.norm(&x, &format!("{conv}.layer_norm"), 1e-5)?, &valid)?;
            let projected = w.conv1(&normalized, &format!("{conv}.pointwise_conv1"), 1, 0, 1)?;
            let pieces = ops::split_n(&projected, 2, -1)?;
            let glu = ops::multiply(&pieces[0], &ops::sigmoid(&pieces[1])?)?;
            let padding = Array::zeros([1, 30, 1024], Dtype::Float32)?;
            let padded = ops::concatenate(&[&padding, &glu], 1)?;
            let depthwise = w.conv1(&padded, &format!("{conv}.depthwise_conv"), 1, 0, 1024)?;
            let normalized =
                swish(&w.norm(&depthwise, &format!("{conv}.depthwise_layer_norm"), 1e-5)?)?;
            x = ops::add(
                &x,
                &w.conv1(&normalized, &format!("{conv}.pointwise_conv2"), 1, 0, 1)?,
            )?;
            let ffn = self.ffn(
                &w.norm(&x, &format!("{base}.ffn2_layer_norm"), 1e-5)?,
                &format!("{base}.ffn2"),
            )?;
            x = w.norm(
                &ops::add(&x, &scale(&ffn, 0.5)?)?,
                &format!("{base}.final_layer_norm"),
                1e-5,
            )?;
            x.eval()?;
        }
        control.check()?;
        finite(&x, "w2v hidden_states[17]")?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{load_component, IndexTts25Component};
    #[test]
    #[ignore = "requires pinned snapshot and IRONMLX_REFERENCE_ENCODERS baseline"]
    fn real_w2v_hidden17_parity() {
        struct Active;
        impl SessionControl for Active {
            fn check(&self) -> Result<()> {
                Ok(())
            }
        }
        let snapshot =
            std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let weights = load_component(
            &snapshot.join("model.safetensors"),
            &IndexTts25Component::W2vBert.spec().unwrap(),
        )
        .unwrap();
        let model = W2vBert::new(weights);
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_ENCODERS").unwrap())
                .unwrap();
        let mask = reference["mask"].to_vec::<i32>().unwrap();
        let actual = model
            .hidden17(&reference["features"], &mask, &Active)
            .unwrap();
        assert_eq!(actual.shape(), reference["hidden17"].shape());
        let actual = actual.to_vec::<f32>().unwrap();
        let expected = reference["hidden17"].to_vec::<f32>().unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let maximum = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("w2v hidden17: max={maximum}, mean={mean}");
        assert!(maximum < 2e-3 && mean < 2e-4);
        struct CancelAfterLayer(std::cell::Cell<usize>);
        impl SessionControl for CancelAfterLayer {
            fn check(&self) -> Result<()> {
                self.0.set(self.0.get() + 1);
                if self.0.get() >= 3 {
                    Err(crate::AudioError::Cancelled)
                } else {
                    Ok(())
                }
            }
        }
        // One layer has evaluated before cancellation at the next layer boundary.
        assert!(matches!(
            model.hidden17(
                &reference["features"],
                &mask,
                &CancelAfterLayer(std::cell::Cell::new(0))
            ),
            Err(crate::AudioError::Cancelled)
        ));
    }
}
