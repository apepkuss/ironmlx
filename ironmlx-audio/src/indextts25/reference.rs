use super::{campplus::CampPlus, conditioning::Conditioning, layers::finite, w2v::W2vBert};
use crate::{
    error::invalid,
    features,
    resources::{
        self,
        derived::{self, AuxiliaryComponent},
        IndexTts25Component,
    },
    signal, AudioError, PcmBuffer, ResourceIssue, ResourceReport, Result, SessionControl,
};
use ironmlx_core::weights::WeightMap;
use mlx::{ops, Array};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

/// Reference conditioning stays on the execution worker for reuse during synthesis.
/// This value is not a synthesis result or transport payload.
pub struct ReferenceConditioning {
    pub(crate) semantic: Array,
    pub(crate) mel: Array,
    pub(crate) style: Array,
    pub(crate) emotion: Array,
    pub(crate) prompt: Array,
    pub(crate) gpt: Array,
    source_frames: usize,
    used_frames: usize,
    source_sample_rate: u32,
}

impl ReferenceConditioning {
    pub fn source_frames(&self) -> usize {
        self.source_frames
    }
    pub fn used_frames(&self) -> usize {
        self.used_frames
    }
    pub fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }
    pub fn semantic_frames(&self) -> usize {
        self.semantic.shape()[1] as usize
    }
    pub fn mel_frames(&self) -> usize {
        self.mel.shape()[2] as usize
    }
    /// Total evaluated tensor payload, for runtime accounting (excluding model weights).
    pub fn tensor_bytes(&self) -> usize {
        [
            &self.semantic,
            &self.mel,
            &self.style,
            &self.emotion,
            &self.prompt,
            &self.gpt,
        ]
        .iter()
        .map(|x| x.size() * x.dtype().byte_size())
        .sum()
    }
}

/// Fixed native reference encoder; exclusive worker-local use and no internal downloads.
/// Set MLX_ENABLE_TF32=0 before any MLX initialization. This library never changes
/// that process-wide setting, which is cached by MLX on its first use.
pub struct IndexTts25ReferenceEncoder {
    w2v: W2vBert,
    campplus: CampPlus,
    conditioning: Conditioning,
    auxiliary: WeightMap,
}
impl IndexTts25ReferenceEncoder {
    /// Inspect resources used by reference encoding, not full synthesis readiness.
    pub fn inspect(snapshot: &Path, derived_root: &Path) -> ResourceReport {
        let mut report = derived::inspect_derived(derived_root);
        for component in [
            IndexTts25Component::W2vBert,
            IndexTts25Component::Gpt,
            IndexTts25Component::S2Mel,
        ] {
            match component.spec().and_then(|s| {
                resources::inspect_component(&snapshot.join(component.file_name()), &s)
            }) {
                Ok(bytes) => report.static_tensor_bytes += bytes,
                Err(error) => {
                    report.complete = false;
                    report.issues.push(ResourceIssue {
                        component: component.file_name().into(),
                        reason: error.to_string(),
                    });
                }
            }
        }
        if let Err(error) = Self::check_config(snapshot) {
            report.complete = false;
            report.issues.push(ResourceIssue {
                component: "config.json".into(),
                reason: error.to_string(),
            });
        }
        report
    }
    pub(super) fn check_config(snapshot: &Path) -> Result<()> {
        let path = snapshot.join("config.json");
        let profile: serde_json::Value =
            serde_json::from_str(include_str!("../../resources/indextts25/sources.json"))
                .map_err(|e| invalid("profile", e.to_string()))?;
        let file = profile["source"]["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["path"] == "config.json")
            .unwrap();
        if fs::metadata(&path)?.len() != file["bytes"].as_u64().unwrap() {
            return Err(invalid("config", "configuration size mismatch"));
        }
        if Some(format!("{:x}", Sha256::digest(fs::read(path)?)).as_str())
            != file["sha256"].as_str()
        {
            return Err(invalid("config", "configuration hash mismatch"));
        }
        Ok(())
    }
    pub fn load(snapshot: &Path, derived_root: &Path) -> Result<Self> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
            return Err(invalid(
                "MLX_ENABLE_TF32",
                "set to 0 before process MLX initialization",
            ));
        }
        Self::check_config(snapshot)?;
        let get = |component: IndexTts25Component| {
            resources::load_component(&snapshot.join(component.file_name()), &component.spec()?)
        };
        Self::from_weights(
            snapshot,
            derived_root,
            get(IndexTts25Component::Gpt)?,
            get(IndexTts25Component::S2Mel)?,
        )
    }
    pub(super) fn from_weights(
        snapshot: &Path,
        derived_root: &Path,
        mut gpt: WeightMap,
        mut s2mel: WeightMap,
    ) -> Result<Self> {
        if std::env::var("MLX_ENABLE_TF32").as_deref() != Ok("0") {
            return Err(invalid(
                "MLX_ENABLE_TF32",
                "reference encoding requires MLX_ENABLE_TF32=0 before process MLX initialization",
            ));
        }
        Self::check_config(snapshot)?;
        let auxiliary = derived::load_auxiliary(derived_root, AuxiliaryComponent::Matrices)?;
        if !ops::all(
            &ops::greater(
                &auxiliary.tensors()["w2v_var"],
                &Array::try_from((&[0f32][..], []))?,
            )?,
            ops::All,
            false,
        )?
        .item::<bool>()?
        {
            return Err(AudioError::ResourceMismatch {
                component: "w2v_var".into(),
                reason: "variance must be positive".into(),
            });
        }
        let campplus = CampPlus::new(derived::load_auxiliary(
            derived_root,
            AuxiliaryComponent::CampPlus,
        )?);
        let get = |component: IndexTts25Component| {
            resources::load_component(&snapshot.join(component.file_name()), &component.spec()?)
        };
        let w2v = W2vBert::new(get(IndexTts25Component::W2vBert)?);
        // Retain only the reference modules; shared arrays do not duplicate storage.
        gpt.retain(|key, _| {
            key.starts_with("emo_")
                || key.starts_with("emovec_layer.")
                || key.starts_with("spk_emb_proj.")
        });
        s2mel.retain(|key, _| key.starts_with("length_regulator."));
        Ok(Self {
            w2v,
            campplus,
            conditioning: Conditioning::new(gpt, s2mel),
            auxiliary,
        })
    }
    pub fn encode(
        &mut self,
        pcm: &PcmBuffer,
        control: &dyn SessionControl,
    ) -> Result<ReferenceConditioning> {
        control.check()?;
        let wave = signal::prepare_reference(pcm, control)?;
        let mel = features::reference_mel(&wave.mel_22050, control)?;
        let mel_frames = mel.frames();
        let mel = Array::try_from((mel.values(), [1, mel_frames as i32, 80]))?
            .transpose_axes([0, 2, 1])?;
        let fbank = features::speaker_fbank(&wave.semantic_16k, control)?;
        let fbank = Array::try_from((fbank.values(), [1, fbank.frames() as i32, 80]))?;
        let features = features::semantic_features(&wave.semantic_16k, control)?;
        let input = Array::try_from((
            features.features.values(),
            [1, features.features.frames() as i32, 160],
        ))?;
        let hidden = self
            .w2v
            .hidden17(&input, &features.attention_mask, control)?;
        let tensors = self.auxiliary.tensors();
        let mean = &tensors["w2v_mean"];
        let var = &tensors["w2v_var"];
        let semantic = ops::divide(&ops::subtract(&hidden, mean)?, &ops::sqrt(var)?)?;
        finite(&semantic, "normalized semantic embedding")?;
        let style = self.campplus.encode(&fbank, control)?;
        let prompt = self.conditioning.prompt(&semantic, mel_frames, control)?;
        let emotion = self.conditioning.emotion(&semantic, control)?;
        let gpt = self.conditioning.gpt_conditioning(&style, &emotion)?;
        mlx::transforms::eval(&[&semantic, &mel, &style, &prompt, &emotion, &gpt])?;
        control.check()?;
        Ok(ReferenceConditioning {
            semantic,
            mel,
            style,
            prompt,
            emotion,
            gpt,
            source_frames: wave.source_frames,
            used_frames: wave.used_frames,
            source_sample_rate: wave.source_sample_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcmFormat;

    struct Active;
    impl SessionControl for Active {
        fn check(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    #[ignore = "requires pinned snapshot, derived resources, features and reference encoder fixtures"]
    fn real_reference_pipeline_parity() {
        let snapshot = std::env::var("IRONMLX_INDEXTTS25_SNAPSHOT").unwrap();
        let derived = std::env::var("IRONMLX_INDEXTTS25_DERIVED").unwrap();
        let features_path = std::env::var("IRONMLX_REFERENCE_FEATURES").unwrap();
        let reference_directory = std::env::var("IRONMLX_REFERENCE_DIRECTORY").unwrap();
        let (waves, _) = mlx::io::load_safetensors(&features_path).unwrap();
        let mut encoder =
            IndexTts25ReferenceEncoder::load(Path::new(&snapshot), Path::new(&derived)).unwrap();
        for (case, file) in [
            ("speech_1s", "encoders-fp32-1s.safetensors"),
            ("speech_odd", "encoders-fp32-odd.safetensors"),
            ("speech_5s", "encoders-fp32-5s.safetensors"),
        ] {
            let (expected, _) = mlx::io::load_safetensors(
                Path::new(&reference_directory).join(file).to_str().unwrap(),
            )
            .unwrap();
            let pcm = PcmBuffer {
                format: PcmFormat {
                    sample_rate: 16000,
                    channels: 1,
                },
                samples: waves[&format!("{case}.wave16")].to_vec().unwrap(),
            };
            let result = encoder.encode(&pcm, &Active).unwrap();
            assert_eq!(result.source_frames(), pcm.samples.len());
            assert_eq!(result.used_frames(), pcm.samples.len());
            assert_eq!(result.source_sample_rate(), 16000);
            assert!(result.tensor_bytes() > 0);
            let mel = result.mel.transpose_axes([0, 2, 1]).unwrap();
            for (name, actual, max_limit, mean_limit) in [
                ("semantic", &result.semantic, 2e-3, 2e-4),
                ("mel", &mel, 5e-4, 2e-5),
                ("style", &result.style, 2e-3, 2e-4),
                ("emotion", &result.emotion, 2e-3, 2e-4),
                ("conditioning", &result.gpt, 2e-3, 2e-4),
                ("prompt_condition", &result.prompt, 2e-3, 2e-4),
            ] {
                assert_eq!(actual.shape(), expected[name].shape(), "{case}.{name}");
                let actual = actual.to_vec::<f32>().unwrap();
                let expected = expected[name].to_vec::<f32>().unwrap();
                assert!(actual.iter().all(|x| x.is_finite()));
                let errors: Vec<_> = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .collect();
                let maximum = errors.iter().copied().fold(0., f32::max);
                let mean = errors.iter().sum::<f32>() / errors.len() as f32;
                eprintln!("{case}.{name}: max={maximum}, mean={mean}");
                assert!(maximum <= max_limit && mean <= mean_limit, "{case}.{name}");
            }
        }
        struct Cancelled;
        impl SessionControl for Cancelled {
            fn check(&self) -> Result<()> {
                Err(AudioError::Cancelled)
            }
        }
        let pcm = PcmBuffer {
            format: PcmFormat {
                sample_rate: 16000,
                channels: 1,
            },
            samples: vec![0.; 16000],
        };
        assert!(matches!(
            encoder.encode(&pcm, &Cancelled),
            Err(AudioError::Cancelled)
        ));
    }
}
