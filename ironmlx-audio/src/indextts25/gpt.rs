//! Fixed 2.5 GPT generation backbone; all state belongs to one synthesis session.
use super::layers::{finite, scalar, scale, Weights};
use crate::{error::invalid, AudioError, Result, SessionControl};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array, Dtype};

pub(super) struct Gpt {
    weights: Weights,
}
pub(super) struct GptState {
    input: Array,
    padding: Vec<bool>,
    cache: Vec<(Array, Array)>,
    pub generated: Vec<u32>,
    pub finished: bool,
}
impl Gpt {
    pub fn new(weights: WeightMap) -> Self {
        Self {
            weights: Weights(weights),
        }
    }
    fn embedding(&self, name: &str, ids: &[u32]) -> Result<Array> {
        Ok(ops::take(
            self.weights.get(&format!("{name}.weight"))?,
            &Array::try_from((ids, [ids.len() as i32]))?,
            0,
        )?)
    }
    fn mel_embedding(&self, token: u32, position: usize) -> Result<Array> {
        if token >= 8194 || position >= 1818 {
            return Err(invalid("mel_position", "invalid token or position"));
        }
        Ok(ops::add(
            &self.embedding("mel_embedding", &[token])?,
            &self.embedding("mel_pos_embedding.emb", &[position as u32])?,
        )?
        .reshape([1, 1, 1280])?)
    }
    pub fn start(&self, conditioning: &Array, text: &[u32], language: u32) -> Result<GptState> {
        if conditioning.shape().as_slice() != [1, 3, 1280]
            || text.is_empty()
            || text.len() > 602
            || text.iter().any(|&x| x >= 60509)
            || ![0, 1, 3, 7, 13].contains(&language)
        {
            return Err(invalid(
                "gpt_input",
                "invalid conditioning, tokens or language",
            ));
        }
        finite(conditioning, "GPT conditioning")?;
        let canonical: Vec<_> = std::iter::once(0)
            .chain(text.iter().copied().filter(|x| *x != 0 && *x != 1))
            .chain(std::iter::once(1))
            .collect();
        if canonical.len() > 602 {
            return Err(AudioError::CapacityExceeded {
                resource: "GPT text positions",
            });
        }
        let positions: Vec<_> = (0..canonical.len() as u32).collect();
        let embeddings = ops::add(
            &ops::add(
                &self.embedding("text_embedding", &canonical)?,
                &self.embedding("text_pos_embedding.emb", &positions)?,
            )?,
            &self.embedding("lang_embedding", &[language])?,
        )?
        .reshape([1, canonical.len() as i32, 1280])?;
        let pad = text.len() + 2 - canonical.len();
        let zeros = Array::zeros([1, pad as i32, 1280], embeddings.dtype())?;
        let input = ops::concatenate(
            &[
                &zeros,
                conditioning,
                &embeddings,
                &self.mel_embedding(8192, 0)?,
            ],
            1,
        )?;
        let mut padding = vec![false; pad];
        padding.resize(input.shape()[1] as usize, true);
        Ok(GptState {
            input,
            padding,
            cache: Vec::new(),
            generated: Vec::new(),
            finished: false,
        })
    }
    pub fn logits(&self, state: &mut GptState, control: &dyn SessionControl) -> Result<Array> {
        control.check()?;
        if state.finished {
            return Err(AudioError::InvalidSessionState);
        }
        if state.generated.len() >= 1500 {
            return Err(AudioError::GenerationLimitExceeded);
        }
        let query = state.input.shape()[1];
        let keys = state.padding.len() as i32;
        if keys > 2420 {
            return Err(AudioError::GenerationLimitExceeded);
        }
        let mask: Vec<f32> = (0..query)
            .flat_map(|q| {
                let padding = &state.padding;
                (0..keys).map(move |k| {
                    (if k > keys - query + q { -1e9 } else { 0. })
                        + (if padding[k as usize] { 0. } else { -1e9 })
                })
            })
            .collect();
        let mask = Array::try_from((mask.as_slice(), [1, 1, query, keys]))?;
        let w = &self.weights;
        let mut x = state.input.clone();
        let mut cache = Vec::with_capacity(24);
        for layer in 0..24 {
            control.check()?;
            let base = format!("gpt.h.{layer}");
            let norm = w.norm(&x, &format!("{base}.ln_1"), 1e-5)?;
            let qkv = w.linear_fused(&norm, &format!("{base}.attn.c_attn"))?;
            let q = ops::slice(&qkv, [0, 0, 0], [1, query, 1280])?;
            let k = ops::slice(&qkv, [0, 0, 1280], [1, query, 2560])?;
            let v = ops::slice(&qkv, [0, 0, 2560], [1, query, 3840])?;
            let (k, v) = if state.cache.is_empty() {
                (k, v)
            } else {
                (
                    ops::concatenate(&[&state.cache[layer].0, &k], 1)?,
                    ops::concatenate(&[&state.cache[layer].1, &v], 1)?,
                )
            };
            let q = q
                .reshape([1, query, 20, 64])?
                .transpose_axes([0, 2, 1, 3])?;
            let kh = k.reshape([1, keys, 20, 64])?.transpose_axes([0, 2, 3, 1])?;
            let vh = v.reshape([1, keys, 20, 64])?.transpose_axes([0, 2, 1, 3])?;
            let scores = ops::add(&scale(&ops::matmul(&q, &kh)?, 0.125)?, &mask)?;
            let attended = ops::matmul(&ops::softmax(&scores, -1, false)?, &vh)?
                .transpose_axes([0, 2, 1, 3])?
                .reshape([1, query, 1280])?;
            x = ops::add(
                &x,
                &w.linear_fused(&attended, &format!("{base}.attn.c_proj"))?,
            )?;
            let norm = w.norm(&x, &format!("{base}.ln_2"), 1e-5)?;
            let hidden = w.linear_fused(&norm, &format!("{base}.mlp.c_fc"))?;
            let cubic = ops::power(&hidden, &scalar(3.)?)?;
            let inner = scale(
                &ops::add(&hidden, &scale(&cubic, 0.044715)?)?,
                (2. / std::f32::consts::PI).sqrt(),
            )?;
            let gelu = ops::multiply(
                &scale(&hidden, 0.5)?,
                &ops::add(&scalar(1.)?, &ops::tanh(&inner)?)?,
            )?;
            x = ops::add(&x, &w.linear_fused(&gelu, &format!("{base}.mlp.c_proj"))?)?;
            mlx::transforms::eval(&[&x, &k, &v])?;
            cache.push((k, v));
        }
        let hidden = w.norm(&x, "gpt.ln_f", 1e-5)?;
        let last = ops::slice(&hidden, [0, query - 1, 0], [1, query, 1280])?;
        let logits = w
            .linear_fused(&w.norm(&last, "final_norm", 1e-5)?, "mel_head")?
            .reshape([1, 8194])?;
        finite(&logits, "GPT logits")?;
        control.check()?;
        state.cache = cache;
        Ok(logits)
    }
    pub fn accept(&self, state: &mut GptState, token: u32) -> Result<()> {
        if state.finished {
            return Err(AudioError::InvalidSessionState);
        }
        if token == 8193 {
            state.finished = true;
            return Ok(());
        }
        if token >= 8192 {
            return Err(AudioError::InferenceFailed {
                reason: "GPT emitted a non-code token".into(),
            });
        }
        if state.generated.len() >= 1500 {
            return Err(AudioError::GenerationLimitExceeded);
        }
        state.generated.push(token);
        state.input = self.mel_embedding(token, state.generated.len())?;
        state.padding.push(true);
        Ok(())
    }
}

/// Reference silence cleanup runs only after normal EOS, never on the sampling history.
pub(super) fn compress_silence(codes: &mut Vec<u32>) {
    if codes.iter().filter(|&&code| code == 52).count() <= 30 {
        return;
    }
    let mut consecutive = 0;
    codes.retain(|&code| {
        if code != 52 {
            consecutive = 0;
            true
        } else if consecutive < 10 {
            consecutive += 1;
            true
        } else {
            false
        }
    });
}

/// Fixed reference sampling, including its probability floor; never uses global PRNG state.
fn sampling_logits(logits: &Array, generated: &[u32]) -> Result<Array> {
    finite(logits, "sampling logits")?;
    let mut logits = logits.clone();
    let mut ids: Vec<_> = generated.iter().copied().filter(|&x| x < 8194).collect();
    ids.sort_unstable();
    ids.dedup();
    if !ids.is_empty() {
        let ids = Array::try_from((ids.as_slice(), [1, ids.len() as i32]))?;
        let selected = ops::take_along_axis(&logits, &ids, -1)?;
        let penalty = ops::where_(
            &ops::greater(&selected, &scalar(0.)?)?,
            &scale(&selected, 0.1)?,
            &scale(&selected, 10.)?,
        )?;
        logits = ops::put_along_axis(&logits, &ids, &penalty, -1)?;
    }
    logits = ops::divide(&logits, &scalar(0.8)?)?;
    let threshold = ops::slice(&ops::topk(&logits, 30, -1)?, [0, 0], [1, 1])?;
    logits = ops::where_(
        &ops::less(&logits, &threshold)?,
        &scalar(f32::NEG_INFINITY)?,
        &logits,
    )?;
    let ascending = ops::argsort(&logits, -1)?;
    let reverse: Vec<_> = (0..8194u32).rev().collect();
    let indices = ops::take(
        &ascending,
        &Array::try_from((reverse.as_slice(), [8194]))?,
        -1,
    )?;
    let sorted = ops::take_along_axis(&logits, &indices, -1)?;
    let cumulative = ops::cumsum(&ops::softmax(&sorted, -1, false)?, -1, false, true)?;
    let removed = ops::greater(&cumulative, &scalar(0.8)?)?;
    let removed = ops::concatenate(
        &[
            &Array::zeros([1, 1], Dtype::Bool)?,
            &ops::slice(&removed, [0, 0], [1, 8193])?,
        ],
        -1,
    )?;
    let mask = ops::put_along_axis(
        &Array::zeros([1, 8194], Dtype::Bool)?,
        &indices,
        &removed,
        -1,
    )?;
    logits = ops::where_(&mask, &scalar(f32::NEG_INFINITY)?, &logits)?;
    let probabilities = ops::softmax(&logits, -1, false)?;
    Ok(ops::log(&ops::add(&probabilities, &scalar(1e-10)?)?)?)
}
pub(super) fn sample(logits: &Array, generated: &[u32], key: &Array) -> Result<u32> {
    // The pinned MLX 0.31 reference uses Gumbel-max. MLX 0.32's categorical
    // inverse-CDF optimization changes same-key results for a single row.
    let noise = mlx::random::gumbel().shape([1, 8194]).key(key).sample()?;
    let sampled = ops::argmax(
        &ops::add(&sampling_logits(logits, generated)?, &noise)?,
        -1,
        false,
    )?;
    Ok(sampled.astype(Dtype::Uint32)?.item::<u32>()?)
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
    fn silence_cleanup_uses_total_threshold_and_limits_each_run() {
        let mut short = vec![52; 30];
        compress_silence(&mut short);
        assert_eq!(short.len(), 30);
        let mut long: Vec<_> = std::iter::repeat_n(52, 16)
            .chain([7])
            .chain(std::iter::repeat_n(52, 15))
            .collect();
        compress_silence(&mut long);
        assert_eq!(
            long,
            std::iter::repeat_n(52, 10)
                .chain([7])
                .chain(std::iter::repeat_n(52, 10))
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn generation_cap_is_not_eos_and_terminal_cannot_resume() {
        let model = Gpt::new(WeightMap::new(
            std::collections::HashMap::new(),
            None,
            std::collections::HashMap::new(),
        ));
        let mut state = GptState {
            input: Array::zeros([1, 1, 1280], Dtype::Float32).unwrap(),
            padding: vec![true],
            cache: vec![],
            generated: vec![7; 1500],
            finished: false,
        };
        assert!(matches!(
            model.logits(&mut state, &Active),
            Err(AudioError::GenerationLimitExceeded)
        ));
        assert!(!state.finished);
        state.generated.pop(); // EOS is valid within the 1500-step budget.
        model.accept(&mut state, 8193).unwrap();
        assert!(state.finished);
        assert!(matches!(
            model.logits(&mut state, &Active),
            Err(AudioError::InvalidSessionState)
        ));
        assert!(matches!(
            model.accept(&mut state, 42),
            Err(AudioError::InvalidSessionState)
        ));
    }
    #[test]
    fn sampling_rejects_nonfinite_and_does_not_depend_on_global_rng() {
        let bad = Array::try_from((vec![f32::NAN; 8194].as_slice(), [1, 8194])).unwrap();
        let key = mlx::random::key(15).unwrap();
        assert!(sample(&bad, &[], &key).is_err());
        let logits = Array::zeros([1, 8194], Dtype::Float32).unwrap();
        let a = sample(&logits, &[1, 1, 2], &key).unwrap();
        let _unrelated = mlx::random::normal()
            .shape([20])
            .sample()
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(a, sample(&logits, &[1, 1, 2], &key).unwrap());
    }
    fn close(name: &str, actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape(), "{name}");
        let actual = actual
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = expected
            .astype(Dtype::Float32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let errors: Vec<_> = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max = errors.iter().copied().fold(0., f32::max);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        eprintln!("{name}: max={max}, mean={mean}");
        assert!(max < 2e-3 && mean < 2e-4, "{name}");
    }
    #[test]
    #[ignore = "requires IRONMLX_REFERENCE_GPT"]
    fn reference_sampler_parity() {
        let (r, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_GPT").unwrap()).unwrap();
        for name in ["zh", "en"] {
            let mut generated = Vec::new();
            for step in 0..4 {
                let prefix = format!("{name}.{step}");
                let expected = r[&format!("{prefix}.sample")]
                    .astype(Dtype::Uint32)
                    .unwrap()
                    .item::<u32>()
                    .unwrap();
                let key = mlx::random::key(2025 + step).unwrap();
                let actual = ops::argmax(
                    &ops::add(
                        &r[&format!("{prefix}.sampling_logits")],
                        &mlx::random::gumbel()
                            .shape([1, 8194])
                            .key(&key)
                            .sample()
                            .unwrap(),
                    )
                    .unwrap(),
                    -1,
                    false,
                )
                .unwrap()
                .astype(Dtype::Uint32)
                .unwrap()
                .item::<u32>()
                .unwrap();
                eprintln!("{prefix} direct RNG: actual={actual} expected={expected}");
                close(
                    &format!("{prefix}.sampling_logits"),
                    &sampling_logits(&r[&format!("{prefix}.logits")], &generated).unwrap(),
                    &r[&format!("{prefix}.sampling_logits")],
                );
                assert_eq!(actual, expected);
                generated.push(expected);
            }
        }
    }
    #[test]
    #[ignore = "requires pinned snapshot and IRONMLX_REFERENCE_SYNTHESIS; MLX_ENABLE_TF32=0"]
    fn real_complete_generation_parity() {
        use crate::indextts25::{
            cfm::Cfm, codec::Codec, conditioning::length_regulate, vocoder::Vocoder,
        };
        let root = std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_SYNTHESIS").unwrap())
                .unwrap();
        let load = |component: IndexTts25Component| {
            load_component(
                &root.join(component.file_name()),
                &component.spec().unwrap(),
            )
            .unwrap()
        };
        let gpt = Gpt::new(load(IndexTts25Component::Gpt));
        let codec = Codec::new(load(IndexTts25Component::Codec)).unwrap();
        let s2mel = load(IndexTts25Component::S2Mel);
        let length = Weights(WeightMap::new(
            s2mel
                .tensors()
                .iter()
                .filter(|(k, _)| k.starts_with("length_regulator."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            None,
            std::collections::HashMap::new(),
        ));
        let cfm = Cfm::new(s2mel);
        let vocoder = Vocoder::new(load(IndexTts25Component::BigVgan)).unwrap();
        for name in ["zh", "en"] {
            let text = reference[&format!("{name}.text")]
                .astype(Dtype::Uint32)
                .unwrap()
                .to_vec::<u32>()
                .unwrap();
            let language = reference[&format!("{name}.language")]
                .item::<i32>()
                .unwrap() as u32;
            let mut state = gpt
                .start(&reference["conditioning"], &text, language)
                .unwrap();
            let mut key = mlx::random::key(2025).unwrap();
            while !state.finished {
                let logits = gpt.logits(&mut state, &Active).unwrap();
                let (next, draw) = mlx::random::split(&key).unwrap();
                key = next;
                let token = sample(&logits, &state.generated, &draw).unwrap();
                gpt.accept(&mut state, token).unwrap();
            }
            assert_eq!(
                state.generated,
                reference[&format!("{name}.raw_codes")]
                    .to_vec::<u32>()
                    .unwrap(),
                "{name} complete sampled sequence"
            );
            compress_silence(&mut state.generated);
            assert_eq!(
                state.generated,
                reference[&format!("{name}.codes")].to_vec::<u32>().unwrap()
            );
            let semantic = codec.decode(&state.generated, &Active).unwrap();
            close(
                &format!("{name}.semantic"),
                &semantic,
                &reference[&format!("{name}.semantic")],
            );
            let frames = (f64::from(semantic.shape()[1]) * 1.72) as usize;
            let condition = length_regulate(&length, &semantic, frames, &Active).unwrap();
            close(
                &format!("{name}.condition"),
                &condition,
                &reference[&format!("{name}.condition")],
            );
            let combined =
                ops::concatenate(&[&reference["prompt_condition"], &condition], 1).unwrap();
            let (_, draw) = mlx::random::split(&key).unwrap();
            let noise = mlx::random::normal()
                .shape([1, 80, combined.shape()[1]])
                .key(&draw)
                .sample()
                .unwrap();
            close(
                &format!("{name}.noise"),
                &noise,
                &reference[&format!("{name}.noise")],
            );
            let mut flow =
                Cfm::start(&combined, &reference["ref_mel"], &reference["style"], noise).unwrap();
            for _ in 0..25 {
                cfm.advance(&mut flow, &Active).unwrap();
            }
            let mel = ops::slice(
                &flow.x,
                [0, 0, reference["ref_mel"].shape()[2]],
                [1, 80, combined.shape()[1]],
            )
            .unwrap();
            close(
                &format!("{name}.mel"),
                &mel,
                &reference[&format!("{name}.mel")],
            );
            let audio = vocoder.synthesize(&mel, &Active).unwrap();
            close(
                &format!("{name}.audio"),
                &audio,
                &reference[&format!("{name}.audio")].reshape([-1]).unwrap(),
            );
        }
    }
    #[test]
    #[ignore = "requires pinned snapshot, IRONMLX_REFERENCE_GPT, and MLX_ENABLE_TF32=0"]
    fn real_gpt_prefill_decode_parity() {
        let snapshot =
            std::path::PathBuf::from(std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap());
        let (reference, _) =
            mlx::io::load_safetensors(&std::env::var("IRONMLX_REFERENCE_GPT").unwrap()).unwrap();
        let model = Gpt::new(
            load_component(
                &snapshot.join("gpt.safetensors"),
                &IndexTts25Component::Gpt.spec().unwrap(),
            )
            .unwrap(),
        );
        for name in ["zh", "en"] {
            let tokens = reference[&format!("{name}.text")]
                .astype(Dtype::Uint32)
                .unwrap()
                .to_vec::<u32>()
                .unwrap();
            let language = reference[&format!("{name}.language")]
                .item::<i32>()
                .unwrap() as u32;
            let mut state = model
                .start(&reference["conditioning"], &tokens, language)
                .unwrap();
            for step in 0..4 {
                let prefix = format!("{name}.{step}");
                close(
                    &format!("{prefix}.input"),
                    &state.input,
                    &reference[&format!("{prefix}.input")],
                );
                assert_eq!(
                    state.padding,
                    reference[&format!("{prefix}.mask")]
                        .to_vec::<i32>()
                        .unwrap()
                        .iter()
                        .map(|&x| x == 1)
                        .collect::<Vec<_>>()
                );
                let logits = model.logits(&mut state, &Active).unwrap();
                close(
                    &format!("{prefix}.logits"),
                    &logits,
                    &reference[&format!("{prefix}.logits")],
                );
                close(
                    &format!("{prefix}.k0"),
                    &state.cache[0].0,
                    &reference[&format!("{prefix}.k0")],
                );
                close(
                    &format!("{prefix}.v23"),
                    &state.cache[23].1,
                    &reference[&format!("{prefix}.v23")],
                );
                let key = mlx::random::key(2025 + step).unwrap();
                let expected = reference[&format!("{prefix}.sample")]
                    .astype(Dtype::Uint32)
                    .unwrap()
                    .item::<u32>()
                    .unwrap();
                assert_eq!(
                    sample(
                        &reference[&format!("{prefix}.logits")],
                        &state.generated,
                        &key
                    )
                    .unwrap(),
                    expected,
                    "sampling {prefix}"
                );
                assert_eq!(
                    sample(&logits, &state.generated, &key).unwrap(),
                    expected,
                    "native sampling {prefix}"
                );
                model.accept(&mut state, expected).unwrap();
            }
            model.accept(&mut state, 8193).unwrap();
            assert!(matches!(
                model.logits(&mut state, &Active),
                Err(AudioError::InvalidSessionState)
            ));
        }
    }
}
