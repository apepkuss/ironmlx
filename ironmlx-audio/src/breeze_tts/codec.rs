//! Native Qwen3-TTS 12.5 Hz codec. NLC convolution layout throughout.
//! Architecture adapted from mlx-audio's Qwen3-TTS and Kyutai Mimi implementations.
use crate::{PcmBuffer, SessionControl};
use anyhow::{anyhow, ensure, Context, Result};
use mlx::{fast, ops, Array, Dtype};
use std::{collections::HashMap, path::Path};

#[derive(Default)]
struct State {
    conv: HashMap<String, Array>,
    overflow: HashMap<String, Array>,
    kv: HashMap<String, (Array, Array)>,
    offset: i32,
}
pub(super) struct Codec {
    w: HashMap<String, Array>,
    books: HashMap<String, Array>,
    state: State,
}
fn scalar(x: f32) -> Result<Array> {
    Ok(Array::try_from((&[x][..], []))?)
}
fn time(x: &Array, a: i32, b: i32) -> Result<Array> {
    let s = x.shape();
    Ok(ops::slice(x, &[0, a, 0][..], &[s[0], b, s[2]][..])?)
}
fn pad(x: &Array, left: i32, right: i32, edge: bool) -> Result<Array> {
    let s = x.shape();
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(if edge {
            ops::broadcast_to(&time(x, 0, 1)?, &[s[0], left, s[2]][..])?
        } else {
            Array::zeros(&[s[0], left, s[2]][..], x.dtype())?
        });
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(if edge {
            ops::broadcast_to(&time(x, s[1] - 1, s[1])?, &[s[0], right, s[2]][..])?
        } else {
            Array::zeros(&[s[0], right, s[2]][..], x.dtype())?
        });
    }
    Ok(ops::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?)
}
fn gelu(x: &Array, approx: bool) -> Result<Array> {
    let cdf = if approx {
        ops::tanh(&ops::multiply(
            &scalar(0.7978846)?,
            &ops::add(
                x,
                &ops::multiply(&scalar(0.044715)?, &ops::multiply(&ops::square(x)?, x)?)?,
            )?,
        )?)?
    } else {
        ops::erf(&ops::divide(x, &scalar(std::f32::consts::SQRT_2)?)?)?
    };
    Ok(ops::multiply(
        &ops::multiply(x, &scalar(0.5)?)?,
        &ops::add(&scalar(1.)?, &cdf)?,
    )?)
}
fn elu(x: &Array) -> Result<Array> {
    Ok(ops::where_(
        &ops::greater(x, &scalar(0.)?)?,
        x,
        &ops::subtract(&ops::exp(x)?, &scalar(1.)?)?,
    )?)
}
impl Codec {
    pub(super) fn load(snapshot: &Path) -> Result<Self> {
        let root = if snapshot.join("audio_tokenizer").is_dir() {
            snapshot.join("audio_tokenizer")
        } else {
            snapshot.to_path_buf()
        };
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        ensure!(
            config["model_type"] == "qwen3_tts_tokenizer_12hz"
                && config["encoder_valid_num_quantizers"] == 16
                && config["output_sample_rate"] == 24000
                && config["decode_upsample_rate"] == 1920,
            "unsupported Breeze audio tokenizer config"
        );
        let (raw, _) = mlx::io::load_safetensors(
            root.join("model.safetensors")
                .to_str()
                .context("codec path is not UTF-8")?,
        )?;
        ensure!(
            raw.len() == 496,
            "expected 496 codec tensors, got {}",
            raw.len()
        );
        let mut w = HashMap::new();
        for (k, v) in raw {
            ensure!(v.dtype() == Dtype::Float32, "codec tensor {k} must be F32");
            let v = if v.ndim() == 3 && k.ends_with("weight") {
                if (k.starts_with("decoder.upsample.") && k.contains(".0.conv.weight"))
                    || (k.starts_with("decoder.decoder.") && k.ends_with(".block.1.conv.weight"))
                {
                    v.transpose_axes(&[1, 2, 0][..])?
                } else {
                    v.transpose_axes(&[0, 2, 1][..])?
                }
            } else {
                v
            };
            w.insert(k, v);
        }
        let mut codec = Self {
            w,
            books: HashMap::new(),
            state: State::default(),
        };
        for (prefix, count, suffix) in [
            (
                "decoder.quantizer.rvq_first.vq",
                1,
                "_codebook.embedding_sum",
            ),
            (
                "decoder.quantizer.rvq_rest.vq",
                15,
                "_codebook.embedding_sum",
            ),
            (
                "encoder.quantizer.semantic_residual_vector_quantizer",
                1,
                "codebook.embed_sum",
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                31,
                "codebook.embed_sum",
            ),
        ] {
            for i in 0..count {
                let p = format!("{prefix}.layers.{i}");
                let book = ops::divide(
                    codec.t(&format!("{p}.{suffix}"))?,
                    &ops::maximum(
                        codec.t(&format!(
                            "{p}.{}cluster_usage",
                            if suffix.starts_with('_') {
                                "_codebook."
                            } else {
                                "codebook."
                            }
                        ))?,
                        &scalar(1e-5)?,
                    )?
                    .reshape(&[-1, 1][..])?,
                )?;
                book.eval()?;
                codec.books.insert(p, book);
            }
        }
        Ok(codec)
    }
    fn t(&self, key: &str) -> Result<&Array> {
        self.w
            .get(key)
            .ok_or_else(|| anyhow!("missing codec tensor {key}"))
    }
    fn linear(&self, x: &Array, p: &str) -> Result<Array> {
        let y = ops::matmul(x, &self.t(&format!("{p}.weight"))?.transpose()?)?;
        Ok(if let Some(b) = self.w.get(&format!("{p}.bias")) {
            ops::add(&y, b)?
        } else {
            y
        })
    }
    fn norm(&self, x: &Array, p: &str, rms: bool) -> Result<Array> {
        Ok(if rms {
            fast::rms_norm(x, Some(self.t(&format!("{p}.weight"))?), 1e-5)?
        } else {
            fast::layer_norm(
                x,
                Some(self.t(&format!("{p}.weight"))?),
                self.w.get(&format!("{p}.bias")),
                if p.ends_with(".norm") { 1e-6 } else { 1e-5 },
            )?
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        x: &Array,
        p: &str,
        stride: i32,
        dilation: i32,
        groups: i32,
        edge: bool,
        state: Option<&mut State>,
    ) -> Result<Array> {
        let w = self.t(&format!("{p}.weight"))?;
        let padding = (w.shape()[1] - 1) * dilation + 1 - stride;
        let padded = if let Some(s) = state {
            let v = if let Some(prev) = s.conv.remove(p) {
                ops::concatenate(&[&prev, x], 1)?
            } else {
                pad(x, padding, 0, edge)?
            };
            if padding > 0 {
                s.conv
                    .insert(p.into(), time(&v, v.shape()[1] - padding, v.shape()[1])?);
            }
            v
        } else {
            pad(x, padding, (stride - x.shape()[1] % stride) % stride, edge)?
        };
        let y = ops::conv1d(&padded, w, stride, 0, dilation, groups)?;
        Ok(if let Some(b) = self.w.get(&format!("{p}.bias")) {
            ops::add(&y, b)?
        } else {
            y
        })
    }
    fn upconv(&self, x: &Array, p: &str, stride: i32, state: Option<&mut State>) -> Result<Array> {
        let w = self.t(&format!("{p}.weight"))?;
        let trim = w.shape()[1] - stride;
        // Keep overlap free of bias: the bias is applied exactly once per output sample.
        let mut y = ops::conv_transpose1d(x, w, stride, 0, 1, 0, 1)?;
        if let Some(s) = state {
            if let Some(prev) = s.overflow.remove(p) {
                let n = prev.shape()[1];
                y = ops::concatenate(
                    &[
                        &ops::add(&time(&y, 0, n)?, &prev)?,
                        &time(&y, n, y.shape()[1])?,
                    ],
                    1,
                )?;
            }
            if trim > 0 {
                s.overflow
                    .insert(p.into(), time(&y, y.shape()[1] - trim, y.shape()[1])?);
            }
        }
        if trim > 0 {
            y = time(&y, 0, y.shape()[1] - trim)?;
        }
        Ok(if let Some(b) = self.w.get(&format!("{p}.bias")) {
            ops::add(&y, b)?
        } else {
            y
        })
    }
    fn snake(&self, x: &Array, p: &str) -> Result<Array> {
        let a = ops::exp(self.t(&format!("{p}.alpha"))?)?;
        let b = ops::add(&ops::exp(self.t(&format!("{p}.beta"))?)?, &scalar(1e-9)?)?;
        Ok(ops::add(
            x,
            &ops::divide(&ops::square(&ops::sin(&ops::multiply(x, &a)?)?)?, &b)?,
        )?)
    }
    fn transformer(
        &self,
        x: &Array,
        decoder: bool,
        mut state: Option<&mut State>,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let p = if decoder {
            "decoder.pre_transformer"
        } else {
            "encoder.encoder_transformer"
        };
        let mut x = if decoder {
            self.linear(x, &format!("{p}.input_proj"))?
        } else {
            x.clone()
        };
        let n = x.shape()[1];
        let offset = state.as_ref().map_or(0, |s| s.offset);
        let heads = if decoder { 16 } else { 8 };
        let cached = state
            .as_ref()
            .and_then(|s| s.kv.values().next())
            .map_or(0, |(k, _)| k.shape()[2]);
        let mut mask = Vec::with_capacity((n * (cached + n)) as usize);
        for q in 0..n {
            for k in 0..cached + n {
                let delta = q + cached - k;
                mask.push(if delta >= 0 && (!decoder || delta < 72) {
                    0.
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mask = Array::try_from((mask.as_slice(), &[n, cached + n][..]))?;
        for i in 0..8 {
            control.check()?;
            let l = format!("{p}.layers.{i}");
            let h = self.norm(&x, &format!("{l}.input_layernorm"), decoder)?;
            let proj = |name: &str| -> Result<Array> {
                Ok(self
                    .linear(&h, &format!("{l}.self_attn.{name}_proj"))?
                    .reshape(&[1, n, heads, 64][..])?
                    .transpose_axes(&[0, 2, 1, 3][..])?)
            };
            let q = fast::rope(&proj("q")?, 64, false, Some(10000.), 1., offset, None)?;
            let mut k = fast::rope(&proj("k")?, 64, false, Some(10000.), 1., offset, None)?;
            let mut v = proj("v")?;
            if let Some(s) = state.as_deref_mut() {
                if let Some((pk, pv)) = s.kv.remove(&l) {
                    k = ops::concatenate(&[&pk, &k], 2)?;
                    v = ops::concatenate(&[&pv, &v], 2)?;
                }
                let retained = k.shape()[2].min(72);
                let keep = |a: &Array| -> Result<Array> {
                    Ok(ops::slice(
                        a,
                        &[0, 0, a.shape()[2] - retained, 0][..],
                        &[1, heads, a.shape()[2], 64][..],
                    )?)
                };
                s.kv.insert(l.clone(), (keep(&k)?, keep(&v)?));
            }
            let a = fast::scaled_dot_product_attention(&q, &k, &v, 0.125, "", Some(&mask), None)?
                .transpose_axes(&[0, 2, 1, 3][..])?
                .reshape(&[1, n, heads * 64][..])?;
            x = ops::add(
                &x,
                &ops::multiply(
                    &self.linear(&a, &format!("{l}.self_attn.o_proj"))?,
                    self.t(&format!("{l}.self_attn_layer_scale.scale"))?,
                )?,
            )?;
            let h = self.norm(&x, &format!("{l}.post_attention_layernorm"), decoder)?;
            let ff = if decoder {
                let gate = self.linear(&h, &format!("{l}.mlp.gate_proj"))?;
                let silu = ops::multiply(&gate, &ops::sigmoid(&gate)?)?;
                self.linear(
                    &ops::multiply(&silu, &self.linear(&h, &format!("{l}.mlp.up_proj"))?)?,
                    &format!("{l}.mlp.down_proj"),
                )?
            } else {
                self.linear(
                    &gelu(&self.linear(&h, &format!("{l}.mlp.fc1"))?, true)?,
                    &format!("{l}.mlp.fc2"),
                )?
            };
            x = ops::add(
                &x,
                &ops::multiply(&ff, self.t(&format!("{l}.mlp_layer_scale.scale"))?)?,
            )?;
            x.eval()?;
        }
        if let Some(s) = state {
            s.offset += n;
        }
        if decoder {
            x = self.linear(
                &self.norm(&x, &format!("{p}.norm"), true)?,
                &format!("{p}.output_proj"),
            )?;
        }
        Ok(x)
    }
    pub(super) fn encode(&self, pcm: &PcmBuffer, control: &dyn SessionControl) -> Result<Array> {
        control.check()?;
        let frames = pcm.frames()?;
        ensure!(
            frames > 0 && frames <= pcm.format.sample_rate as usize * 60,
            "reference must contain at most 60 seconds"
        );
        let mono = pcm
            .samples
            .chunks_exact(pcm.format.channels as usize)
            .map(|c| c.iter().map(|v| v / c.len() as f32).sum())
            .collect::<Vec<f32>>();
        let samples = crate::signal::resample_mono(&mono, pcm.format.sample_rate, 24000, control)?;
        let mut x = Array::try_from((samples.as_slice(), &[1, samples.len() as i32, 1][..]))?;
        x = self.conv(&x, "encoder.encoder.layers.0.conv", 1, 1, 1, false, None)?;
        for (i, stride) in [4, 5, 6, 8].iter().enumerate() {
            control.check()?;
            let p = format!("encoder.encoder.layers.{}", i * 3 + 1);
            let h = self.conv(
                &elu(&x)?,
                &format!("{p}.block.1.conv"),
                1,
                1,
                1,
                false,
                None,
            )?;
            let h = self.conv(
                &elu(&h)?,
                &format!("{p}.block.3.conv"),
                1,
                1,
                1,
                false,
                None,
            )?;
            x = ops::add(&x, &h)?;
            x = self.conv(
                &elu(&x)?,
                &format!("encoder.encoder.layers.{}.conv", i * 3 + 3),
                *stride,
                1,
                1,
                false,
                None,
            )?;
            x.eval()?;
        }
        x = self.conv(
            &elu(&x)?,
            "encoder.encoder.layers.14.conv",
            1,
            1,
            1,
            false,
            None,
        )?;
        x = self.transformer(&x, false, None, control)?;
        x = self.conv(&x, "encoder.downsample.conv", 2, 1, 1, true, None)?;
        let mut codes = Vec::new();
        for (p, count) in [
            ("encoder.quantizer.semantic_residual_vector_quantizer", 1),
            ("encoder.quantizer.acoustic_residual_vector_quantizer", 15),
        ] {
            let mut residual = self.conv(&x, &format!("{p}.input_proj"), 1, 1, 1, false, None)?;
            for i in 0..count {
                control.check()?;
                let book = &self.books[&format!("{p}.layers.{i}")];
                let c2 = ops::multiply(&ops::sum(&ops::square(book)?, -1, false)?, &scalar(0.5)?)?;
                let dist = ops::subtract(&c2, &ops::matmul(&residual, &book.transpose()?)?)?;
                let ids = ops::argmin(&dist, -1, false)?.astype(Dtype::Int32)?;
                residual = ops::subtract(&residual, &ops::take(book, &ids, 0)?)?;
                ids.eval()?;
                residual.eval()?;
                codes.push(ids);
            }
        }
        Ok(ops::stack(&codes.iter().collect::<Vec<_>>(), -1)?)
    }
    fn decode_inner(
        &self,
        codes: &Array,
        mut state: Option<&mut State>,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let shape = codes.shape();
        ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[1] > 0 && shape[2] == 16,
            "codec codes must be [1,time,16]"
        );
        let mut decoded = Vec::new();
        for (p, start, count) in [
            ("decoder.quantizer.rvq_first.vq", 0, 1),
            ("decoder.quantizer.rvq_rest.vq", 1, 15),
        ] {
            let mut x = Array::zeros(&[1, shape[1], 256][..], Dtype::Float32)?;
            for i in 0..count {
                let ids = ops::slice(
                    codes,
                    &[0, 0, start + i][..],
                    &[1, shape[1], start + i + 1][..],
                )?
                .reshape(&[1, shape[1]][..])?;
                x = ops::add(
                    &x,
                    &ops::take(&self.books[&format!("{p}.layers.{i}")], &ids, 0)?,
                )?;
            }
            decoded.push(self.conv(
                &x,
                &format!("{}.output_proj", p.trim_end_matches(".vq")),
                1,
                1,
                1,
                false,
                None,
            )?);
        }
        let mut x = ops::add(&decoded[0], &decoded[1])?;
        x = self.conv(
            &x,
            "decoder.pre_conv.conv",
            1,
            1,
            1,
            false,
            state.as_deref_mut(),
        )?;
        x = self.transformer(&x, true, state.as_deref_mut(), control)?;
        for i in 0..2 {
            control.check()?;
            let p = format!("decoder.upsample.{i}");
            x = self.upconv(&x, &format!("{p}.0.conv"), 2, None)?;
            let h = self.conv(
                &x,
                &format!("{p}.1.dwconv.conv"),
                1,
                1,
                1024,
                false,
                state.as_deref_mut(),
            )?;
            let h = self.norm(&h, &format!("{p}.1.norm"), false)?;
            let h = self.linear(
                &gelu(&self.linear(&h, &format!("{p}.1.pwconv1"))?, false)?,
                &format!("{p}.1.pwconv2"),
            )?;
            x = ops::add(&x, &ops::multiply(&h, self.t(&format!("{p}.1.gamma"))?)?)?;
            x.eval()?;
        }
        x = self.conv(
            &x,
            "decoder.decoder.0.conv",
            1,
            1,
            1,
            false,
            state.as_deref_mut(),
        )?;
        for (i, stride) in [8, 5, 4, 3].iter().enumerate() {
            control.check()?;
            let p = format!("decoder.decoder.{}.block", i + 1);
            x = self.snake(&x, &format!("{p}.0"))?;
            x = self.upconv(&x, &format!("{p}.1.conv"), *stride, state.as_deref_mut())?;
            for (j, d) in [1, 3, 9].iter().enumerate() {
                let r = format!("{p}.{}", j + 2);
                let h = self.conv(
                    &self.snake(&x, &format!("{r}.act1"))?,
                    &format!("{r}.conv1.conv"),
                    1,
                    *d,
                    1,
                    false,
                    state.as_deref_mut(),
                )?;
                let h = self.conv(
                    &self.snake(&h, &format!("{r}.act2"))?,
                    &format!("{r}.conv2.conv"),
                    1,
                    1,
                    1,
                    false,
                    state.as_deref_mut(),
                )?;
                x = ops::add(&x, &h)?;
            }
            x.eval()?;
        }
        x = self.conv(
            &self.snake(&x, "decoder.decoder.5")?,
            "decoder.decoder.6.conv",
            1,
            1,
            1,
            false,
            state,
        )?;
        Ok(ops::clip(&x, Some(&scalar(-1.)?), Some(&scalar(1.)?))?.reshape(&[-1][..])?)
    }
    #[cfg(test)]
    pub(super) fn decode(&self, codes: &Array, control: &dyn SessionControl) -> Result<Array> {
        self.decode_inner(codes, None, control)
    }
    pub(super) fn reset_stream(&mut self) {
        self.state = State::default();
    }
    pub(super) fn decode_step(
        &mut self,
        codes: &Array,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let mut s = std::mem::take(&mut self.state);
        let r = self.decode_inner(codes, Some(&mut s), control);
        self.state = s;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Control;
    impl SessionControl for Control {
        fn check(&self) -> crate::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn transposed_convolution_stream_preserves_bias_and_partition() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let w = HashMap::from([
            (
                "up.weight".into(),
                Array::try_from((&[1.0f32, 2., 3., 4.][..], &[1, 4, 1][..]))?,
            ),
            (
                "up.bias".into(),
                Array::try_from((&[0.75f32][..], &[1][..]))?,
            ),
        ]);
        let codec = Codec {
            w,
            books: HashMap::new(),
            state: State::default(),
        };
        let x = Array::try_from((&[1.0f32, 2., 3.][..], &[1, 3, 1][..]))?;
        let full = codec.upconv(&x, "up", 2, None)?;
        let mut state = State::default();
        let a = codec.upconv(&time(&x, 0, 1)?, "up", 2, Some(&mut state))?;
        let b = codec.upconv(&time(&x, 1, 3)?, "up", 2, Some(&mut state))?;
        let stream = ops::concatenate(&[&a, &b], 1)?;
        assert_eq!(
            full.to_vec::<f32>()?,
            vec![1.75, 2.75, 5.75, 8.75, 9.75, 14.75]
        );
        assert_eq!(stream.to_vec::<f32>()?, full.to_vec::<f32>()?);
        Ok(())
    }
    #[test]
    #[ignore = "requires IRONMLX_BREEZE_SNAPSHOT and IRONMLX_BREEZE_CODEC_REFERENCE"]
    fn real_codec_numerical_parity() -> Result<()> {
        let root = std::env::var("MLX_DIR")?;
        mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        let snapshot = std::env::var("IRONMLX_BREEZE_SNAPSHOT")?;
        let fixture = std::env::var("IRONMLX_BREEZE_CODEC_REFERENCE")?;
        let (r, _) = mlx::io::load_safetensors(&fixture)?;
        let mut codec = Codec::load(Path::new(&snapshot))?;
        let pcm = PcmBuffer {
            format: crate::PcmFormat {
                sample_rate: 24000,
                channels: 1,
            },
            samples: r["wave"].to_vec::<f32>()?,
        };
        let encoded = codec.encode(&pcm, &Control)?;
        assert_eq!(encoded.shape(), r["encoded"].shape());
        let a = encoded.to_vec::<i32>()?;
        let b = r["encoded"].astype(Dtype::Int32)?.to_vec::<i32>()?;
        assert_eq!(a, b, "reference encoder codebook indices");
        let decoded = codec.decode(&r["codes"], &Control)?;
        let compare = |name: &str, x: &Array, y: &Array| -> Result<()> {
            assert_eq!(x.shape(), y.shape());
            let a = x.to_vec::<f32>()?;
            let b = y.to_vec::<f32>()?;
            let max = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let rmse = (a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f32>()
                / a.len() as f32)
                .sqrt();
            eprintln!("{name}: max={max} rmse={rmse}");
            ensure!(
                max < 0.00003 && rmse < 0.000005,
                "{name} waveform numerical mismatch"
            );
            Ok(())
        };
        compare("decode", &decoded, &r["decoded"])?;
        codec.reset_stream();
        let n = r["codes"].shape()[1];
        let mut chunks = Vec::new();
        for start in (0..n).step_by(4) {
            let audio =
                codec.decode_step(&time(&r["codes"], start, (start + 4).min(n))?, &Control)?;
            audio.eval()?;
            chunks.push(audio);
            assert!(codec.state.kv.values().all(|(k, _)| k.shape()[2] <= 72));
        }
        let stream = ops::concatenate(&chunks.iter().collect::<Vec<_>>(), 0)?;
        compare("partition", &stream, &decoded)?;
        compare("stream", &stream, &r["stream"])?;
        Ok(())
    }
}
