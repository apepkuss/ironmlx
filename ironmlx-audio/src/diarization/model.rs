//! Adapted from mlx-audio 4ef6c3a577ade12d946e4b1d1870e4ba67fb5b9a (MIT).
//! Rust adds strict checkpoint contracts, cancellation and bounded session state.
use super::{cache, Config, Diarization, Options, SAMPLE_RATE};
use crate::{PcmBuffer, SessionControl};
use anyhow::{ensure, Context, Result};
use mlx::{ops, Array, Dtype};
use std::{collections::HashMap, path::Path};

pub struct NemotronDiarization {
    weights: HashMap<String, Array>,
    weight_bytes: u64,
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
    #[ignore = "requires the local BF16 Nemotron checkpoint and a reference recording"]
    fn nemotron_real_checkpoint_streaming_and_probability_parity() -> Result<()> {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib"))?;
        }
        let path = std::env::var("IRONMLX_DIARIZATION_DIR")?;
        let audio = std::env::var("IRONMLX_DIARIZATION_AUDIO")?;
        let bytes = std::fs::read(audio)?;
        let pcm = crate::AudioIo::decode(
            &crate::io::NativeAudioIo,
            &bytes,
            &crate::DecodeLimits {
                max_duration_seconds: 7200,
                max_encoded_bytes: 128 * 1024 * 1024,
                max_pcm_bytes: 512 * 1024 * 1024,
                ..crate::DecodeLimits::default()
            },
        )?;
        let model = NemotronDiarization::load(Path::new(&path))?;
        let options = Options {
            latency: serde_json::from_value(
                std::env::var("IRONMLX_DIARIZATION_LATENCY")
                    .unwrap_or_else(|_| "offline".into())
                    .into(),
            )?,
            ..Options::default()
        };
        let mut probabilities = Vec::new();
        let output = model.diarize(&pcm, options, &Control, &mut |_, p| {
            probabilities.extend_from_slice(p);
            Ok(())
        })?;
        ensure!(
            probabilities
                .iter()
                .flatten()
                .all(|p| p.is_finite() && (0. ..=1.).contains(p)),
            "invalid activity probabilities"
        );
        ensure!(
            (output.duration - probabilities.len() as f64 * 0.01).abs() < 0.010000001,
            "missing diarization frames"
        );
        if let Ok(prefix) = std::env::var("IRONMLX_DIARIZATION_OUTPUT") {
            std::fs::write(
                format!("{prefix}.json"),
                serde_json::to_vec_pretty(&output)?,
            )?;
            std::fs::write(
                format!("{prefix}.f32"),
                probabilities
                    .iter()
                    .flatten()
                    .flat_map(|p| p.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        // Incremental PCM partitioning cannot reset context or speaker IDs.
        if pcm.format.sample_rate == 16000 && pcm.format.channels == 1 && pcm.samples.len() < 500000
        {
            let mut state = model.init_stream(options)?;
            let mut partitioned = Vec::new();
            for chunk in pcm.samples.chunks(1237) {
                partitioned.extend(model.feed(&mut state, chunk, false, &Control)?);
            }
            partitioned.extend(model.feed(&mut state, &[], true, &Control)?);
            ensure!(
                partitioned == probabilities,
                "PCM partitioning changes speaker probabilities"
            );
            ensure!(
                model.feed(&mut state, &[], true, &Control).is_err(),
                "finished stream accepted more audio"
            );
        }
        Ok(())
    }
}
pub struct StreamingState {
    cache: Array,
    cache_predictions: Array,
    fifo: Array,
    fifo_predictions: Array,
    compressed: bool,
    pub frames_processed: usize,
    received: usize,
    offset: usize,
    audio: Vec<f32>,
    finished: bool,
    options: Options,
}
fn slice(x: &Array, start: usize, end: usize) -> Result<Array> {
    let shape = x.shape();
    let dims = shape.as_slice();
    Ok(x.slice([0, start as i32, 0], [1, end as i32, dims[2]])?)
}
impl NemotronDiarization {
    pub fn load(directory: &Path) -> Result<Self> {
        Config::load(directory)?;
        let path = directory.join("model.safetensors");
        let weight_bytes = path.metadata()?.len();
        let (weights, _) =
            mlx::io::load_safetensors(path.to_str().context("invalid checkpoint path")?)?;
        let model = Self {
            weights,
            weight_bytes,
        };
        model.validate_weights()?;
        Ok(model)
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    fn weight(&self, name: &str) -> Result<&Array> {
        self.weights
            .get(name)
            .with_context(|| format!("missing Nemotron tensor {name}"))
    }
    fn linear(&self, x: &Array, p: &str) -> Result<Array> {
        let w = self.weight(&format!("{p}.weight"))?.transpose()?;
        Ok(match self.weights.get(&format!("{p}.bias")) {
            Some(b) => ops::addmm(b, x, &w, 1., 1.)?,
            None => ops::matmul(x, &w)?,
        })
    }
    fn norm(&self, x: &Array, p: &str) -> Result<Array> {
        Ok(mlx::fast::layer_norm(
            x,
            Some(self.weight(&format!("{p}.weight"))?),
            Some(self.weight(&format!("{p}.bias"))?),
            1e-5,
        )?)
    }
    fn validate_weights(&self) -> Result<()> {
        let check = |p: &str, s: &[i32], dtype: Dtype| -> Result<()> {
            let w = self.weight(p)?;
            ensure!(
                w.shape().as_slice() == s && w.dtype() == dtype,
                "invalid Nemotron tensor {p}"
            );
            Ok(())
        };
        let bf = Dtype::Bfloat16;
        check("encoder.pre_encode.proj.weight", &[512, 1024], bf)?;
        for p in ["encoder.embed_norm", "encoder.final_norm"] {
            for suffix in ["weight", "bias"] {
                check(&format!("{p}.{suffix}"), &[512], bf)?;
            }
        }
        for i in 0..31 {
            let p = format!("encoder.layers.{i}");
            for n in ["norm1", "norm2"] {
                for s in ["weight", "bias"] {
                    check(&format!("{p}.{n}.{s}"), &[512], bf)?;
                }
            }
            for (n, o, input, bias) in [
                ("attn.w_qkv", 1536, 512, false),
                ("attn.out_proj", 512, 512, true),
                ("ffn.linear1", 2048, 512, true),
                ("ffn.linear2", 512, 2048, true),
            ] {
                check(&format!("{p}.{n}.weight"), &[o, input], bf)?;
                if bias {
                    check(&format!("{p}.{n}.bias"), &[o], bf)?;
                }
            }
        }
        for (name, o, input) in [
            ("encoder_proj", 192, 512),
            ("first_hidden_to_hidden", 192, 192),
            ("single_hidden_to_spks", 8, 192),
        ] {
            check(
                &format!("sortformer_modules.{name}.weight"),
                &[o, input],
                bf,
            )?;
            check(&format!("sortformer_modules.{name}.bias"), &[o], bf)?;
        }
        check(
            "sortformer_modules.subpixel_upsample.weight",
            &[1536, 3, 192],
            bf,
        )?;
        check("sortformer_modules.subpixel_upsample.bias", &[1536], bf)?;
        check("sortformer_modules.learnable_sil_emb", &[512], bf)?;
        check("preprocessor.fb", &[1, 128, 257], Dtype::Float32)?;
        check("preprocessor.window", &[400], Dtype::Float32)?;
        for (name, w) in &self.weights {
            ensure!(
                ops::all(&ops::isfinite(w)?, ops::All, false)?.item::<bool>()?,
                "non-finite Nemotron tensor {name}"
            );
        }
        Ok(())
    }
    fn features(
        &self,
        state: &StreamingState,
        count: usize,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let window = self.weight("preprocessor.window")?.to_vec::<f32>()?;
        let mut frames = vec![0f32; count * 512];
        let get = |i: isize| -> f32 {
            if i < 0 || i >= state.received as isize {
                0.
            } else {
                state.audio[(i - state.offset as isize) as usize]
            }
        };
        for f in 0..count {
            if f % 256 == 0 {
                control.check()?;
            }
            for i in 56..456 {
                let source = ((state.frames_processed + f) * 160 + i) as isize - 256;
                if source >= 0 && source < state.received as isize {
                    frames[f * 512 + i] = (get(source) - 0.97 * get(source - 1)) * window[i - 56];
                }
            }
        }
        let x = Array::try_from((frames.as_slice(), [count as i32, 512]))?;
        let spectrum = ops::fft::rfft(&x, 512, -1, ops::fft::FftNorm::Backward)?
            .abs()?
            .square()?;
        let filters = self
            .weight("preprocessor.fb")?
            .reshape([128, 257])?
            .transpose()?;
        let mel = (ops::matmul(&spectrum, &filters)? + 2f32.powi(-24)).log()?;
        let valid = state.received / 160 - state.frames_processed;
        let mask = (0..count)
            .map(|i| if i < valid { 1f32 } else { 0. })
            .collect::<Vec<_>>();
        let mask = Array::try_from((mask.as_slice(), [count as i32, 1]))?;
        let mel = ops::multiply(&mel, &mask)?;
        let padded = count.div_ceil(8) * 8;
        let mel = if padded > count {
            ops::concatenate(
                &[
                    &mel,
                    &Array::zeros([(padded - count) as i32, 128], Dtype::Float32)?,
                ],
                0,
            )?
        } else {
            mel
        };
        self.linear(
            &mel.astype(Dtype::Bfloat16)?
                .reshape([1, (padded / 8) as i32, 1024])?,
            "encoder.pre_encode.proj",
        )
    }
    fn forward(
        &self,
        features: &Array,
        valid: usize,
        control: &dyn SessionControl,
    ) -> Result<Array> {
        let t = features.shape().as_slice()[1];
        let mut x = self.norm(features, "encoder.embed_norm")?;
        let mask = (0..t).map(|i| i < (valid as i32)).collect::<Vec<_>>();
        let mask = Array::try_from((mask.as_slice(), [1, 1, 1, t]))?;
        for layer in 0..31 {
            control.check()?;
            let p = format!("encoder.layers.{layer}");
            let normalized = self.norm(&x, &format!("{p}.norm1"))?;
            let qkv = self
                .linear(&normalized, &format!("{p}.attn.w_qkv"))?
                .reshape([1, t, 3, 8, 64])?;
            let mut arrays = Vec::with_capacity(3);
            for i in 0..3 {
                arrays.push(
                    qkv.slice([0, 0, i, 0, 0], [1, t, i + 1, 8, 64])?
                        .reshape([1, t, 8, 64])?
                        .transpose_axes([0, 2, 1, 3])?,
                );
            }
            let q = mlx::fast::rope(&arrays[0], 64, false, Some(10000.), 1., 0, None)?;
            let k = mlx::fast::rope(&arrays[1], 64, false, Some(10000.), 1., 0, None)?;
            let attention = mlx::fast::scaled_dot_product_attention(
                &q,
                &k,
                &arrays[2],
                0.125,
                "",
                Some(&mask),
                None,
            )?
            .transpose_axes([0, 2, 1, 3])?
            .reshape([1, t, 512])?;
            x = ops::add(&x, &self.linear(&attention, &format!("{p}.attn.out_proj"))?)?;
            let hidden = self.linear(
                &self.norm(&x, &format!("{p}.norm2"))?,
                &format!("{p}.ffn.linear1"),
            )?;
            let scalar = |v: f32| -> Result<Array> {
                Ok(Array::try_from((&[v][..], []))?.astype(hidden.dtype())?)
            };
            let cdf =
                ops::erf(&ops::divide(&hidden, &scalar(std::f32::consts::SQRT_2)?)?)? + scalar(1.)?;
            let gelu = ops::multiply(&hidden, &cdf)? / scalar(2.)?;
            x = ops::add(&x, &self.linear(&gelu, &format!("{p}.ffn.linear2"))?)?;
            // Bound the lazy graph and release intermediates between layers.
            mlx::eval(&[&x])?;
        }
        let x = self.linear(
            &self.norm(&x, "encoder.final_norm")?,
            "sortformer_modules.encoder_proj",
        )?;
        let x = ops::conv1d(
            &x,
            self.weight("sortformer_modules.subpixel_upsample.weight")?,
            1,
            1,
            1,
            1,
        )?;
        let x = (x + self.weight("sortformer_modules.subpixel_upsample.bias")?).reshape([
            1,
            t * 8,
            192,
        ])?;
        let zero = Array::try_from((&[0f32][..], []))?.astype(x.dtype())?;
        let x = self.linear(
            &x.maximum(&zero)?,
            "sortformer_modules.first_hidden_to_hidden",
        )?;
        let x = self.linear(
            &x.maximum(&zero)?,
            "sortformer_modules.single_hidden_to_spks",
        )?;
        Ok(ops::sigmoid(&x)?)
    }
    pub fn init_stream(&self, options: Options) -> Result<StreamingState> {
        options.validate()?;
        Ok(StreamingState {
            cache: Array::zeros([1, 0, 512], Dtype::Bfloat16)?,
            cache_predictions: Array::zeros([1, 0, 8], Dtype::Float32)?,
            fifo: Array::zeros([1, 0, 512], Dtype::Bfloat16)?,
            fifo_predictions: Array::zeros([1, 0, 8], Dtype::Float32)?,
            compressed: false,
            frames_processed: 0,
            received: 0,
            offset: 0,
            audio: Vec::new(),
            finished: false,
            options,
        })
    }
    fn step(
        &self,
        state: &mut StreamingState,
        count: usize,
        central: usize,
        control: &dyn SessionControl,
    ) -> Result<Vec<[f32; 8]>> {
        let chunk = self.features(state, count, control)?;
        let c = state.cache.shape().as_slice()[1] as usize;
        let f = state.fifo.shape().as_slice()[1] as usize;
        let combined = ops::concatenate(&[&state.cache, &state.fifo, &chunk], 1)?;
        let valid = (state.received / 160 - state.frames_processed)
            .min(count)
            .div_ceil(8)
            + c
            + f;
        let high = self
            .forward(&combined, valid, control)?
            .astype(Dtype::Float32)?;
        let len = high.shape().as_slice()[1] as usize;
        let masks = (0..len)
            .map(|i| if i < valid * 8 { 1f32 } else { 0. })
            .collect::<Vec<_>>();
        let high = ops::multiply(
            &high,
            &Array::try_from((masks.as_slice(), [1, len as i32, 1]))?,
        )?;
        let low = ops::mean(&high.reshape([1, (len / 8) as i32, 8, 8])?, 2, false)?;
        let output = slice(&high, (c + f) * 8, (c + f) * 8 + central)?
            .to_vec::<f32>()?
            .chunks_exact(8)
            .map(|v| v.try_into().unwrap())
            .collect();
        let n = central.div_ceil(8);
        state.fifo = ops::concatenate(&[&state.fifo, &slice(&chunk, 0, n)?], 1)?;
        state.fifo_predictions = slice(&low, c, c + f + n)?;
        let (_, _, fifo, period) = state.options.latency.windows();
        let len = state.fifo.shape().as_slice()[1] as usize;
        if len > fifo {
            let pop = period.max(len - fifo).min(len);
            state.cache = ops::concatenate(&[&state.cache, &slice(&state.fifo, 0, pop)?], 1)?;
            let previous = if state.compressed {
                state.cache_predictions.clone()
            } else {
                slice(&low, 0, c)?
            };
            state.cache_predictions =
                ops::concatenate(&[&previous, &slice(&state.fifo_predictions, 0, pop)?], 1)?;
            state.fifo = slice(&state.fifo, pop, len)?;
            state.fifo_predictions = slice(&state.fifo_predictions, pop, len)?;
            if state.cache.shape().as_slice()[1] > 264 {
                (state.cache, state.cache_predictions) = cache::compress(
                    &state.cache,
                    &state.cache_predictions,
                    self.weight("sortformer_modules.learnable_sil_emb")?,
                )?;
                state.compressed = true;
            }
        }
        state.frames_processed += central;
        mlx::eval(&[
            &state.cache,
            &state.cache_predictions,
            &state.fifo,
            &state.fifo_predictions,
        ])?;
        Ok(output)
    }
    /// Mono 16 kHz PCM. The final call flushes right context exactly once.
    pub fn feed(
        &self,
        state: &mut StreamingState,
        samples: &[f32],
        final_chunk: bool,
        control: &dyn SessionControl,
    ) -> Result<Vec<[f32; 8]>> {
        ensure!(!state.finished, "diarization stream is already finished");
        ensure!(
            samples.iter().all(|v| v.is_finite()),
            "non-finite diarization PCM"
        );
        ensure!(
            state.received + samples.len()
                <= super::MAX_DURATION_SECONDS as usize * SAMPLE_RATE as usize,
            "diarization recording exceeds two hours"
        );
        state.audio.extend_from_slice(samples);
        state.received += samples.len();
        let (central, right, _, _) = state.options.latency.windows();
        let central = central * 8;
        let right = right * 8;
        let mut result = Vec::new();
        loop {
            control.check()?;
            let available = state.received / 160 - state.frames_processed;
            let needed = (state.frames_processed + central + right - 1) * 160 + 256;
            if available == 0 || (!final_chunk && state.received < needed) {
                break;
            }
            let n = central.min(available);
            let total = (state.received / 160 + 1).div_ceil(16) * 16;
            let count = if final_chunk {
                (central + right).min(total - state.frames_processed)
            } else {
                central + right
            };
            result.extend(self.step(state, count, n, control)?);
            let keep = (state.frames_processed * 160)
                .saturating_sub(257)
                .min(state.received);
            state.audio.drain(..keep - state.offset);
            state.offset = keep;
        }
        state.finished = final_chunk;
        if final_chunk {
            state.audio.clear();
        }
        Ok(result)
    }
    pub fn diarize(
        &self,
        pcm: &PcmBuffer,
        options: Options,
        control: &dyn SessionControl,
        progress: &mut super::ProgressCallback<'_>,
    ) -> Result<Diarization> {
        options.validate()?;
        pcm.validate()?;
        ensure!(
            pcm.frames()? as u64 <= u64::from(pcm.format.sample_rate) * 7200
                && pcm.format.channels <= 2
                && (8000..=96000).contains(&pcm.format.sample_rate),
            "unsupported diarization audio duration or format"
        );
        control.check()?;
        let mono = pcm
            .samples
            .chunks_exact(pcm.format.channels as usize)
            .map(|v| v.iter().map(|x| x / v.len() as f32).sum())
            .collect::<Vec<_>>();
        let audio = crate::signal::resample_mono_with_limit(
            &mono,
            pcm.format.sample_rate,
            SAMPLE_RATE,
            7200,
            control,
        )?;
        let mut state = self.init_stream(options)?;
        let mut probabilities = Vec::new();
        for chunk in audio.chunks(16000) {
            let output = self.feed(&mut state, chunk, false, control)?;
            if !output.is_empty() {
                progress(probabilities.len(), &output)?;
                probabilities.extend(output);
            }
        }
        let output = self.feed(&mut state, &[], true, control)?;
        if !output.is_empty() {
            progress(probabilities.len(), &output)?;
            probabilities.extend(output);
        }
        let segments = super::segments(&probabilities, 0, options);
        let num_speakers = segments
            .iter()
            .map(|s| &s.speaker)
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok(Diarization {
            duration: audio.len() as f64 / 16000.,
            num_speakers,
            segments,
        })
    }
}
