use ironmlx_audio::{
    indextts25::IndexTts25Loader, text::TextLimits, AudioError, Language, OutputPolicy, PcmBuffer,
    PcmFormat, ResolvedModelResources, Result, SessionControl, TtsModel, TtsRequest, TtsStep,
};
use std::{
    cell::Cell,
    path::PathBuf,
    time::{Duration, Instant},
};

struct Control {
    cancelled: Cell<bool>,
    deadline: Instant,
}
impl SessionControl for Control {
    fn check(&self) -> Result<()> {
        if self.cancelled.get() {
            Err(AudioError::Cancelled)
        } else if Instant::now() >= self.deadline {
            Err(AudioError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}
fn path(name: &str) -> PathBuf {
    std::env::var(name).expect(name).into()
}
#[test]
#[ignore = "requires pinned snapshot, derived, text resources and IRONMLX_REFERENCE_FEATURES; MLX_ENABLE_TF32=0"]
fn real_synthesis_session_acceptance() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let resources = ResolvedModelResources {
        source_snapshot: path("IRONMLX_INDEXTTS25_SNAPSHOT"),
        derived_resources: path("IRONMLX_INDEXTTS25_DERIVED"),
        resource_lock: root.join("resources/indextts25/sources.json"),
    };
    let loader = IndexTts25Loader::new(path("IRONMLX_WETEXT_FSTS"), path("IRONMLX_UNIDIC_DIR"))
        .with_limits(
            TextLimits {
                segment_tokens: 12,
                ..TextLimits::default()
            },
            600 * 22050,
        )
        .unwrap();
    let started = Instant::now();
    let mut model = loader.load_model(&resources).unwrap();
    eprintln!(
        "model load: {:?}; memory={:?}",
        started.elapsed(),
        mlx::memory::snapshot()
    );
    let (waves, _) =
        mlx::io::load_safetensors(path("IRONMLX_REFERENCE_FEATURES").to_str().unwrap()).unwrap();
    let pcm = PcmBuffer {
        format: PcmFormat {
            sample_rate: 16000,
            channels: 1,
        },
        samples: waves["speech_1s.wave16"].to_vec().unwrap(),
    };
    let control = Control {
        cancelled: Cell::new(false),
        deadline: Instant::now() + Duration::from_secs(900),
    };
    let request = |text: &str, output_policy| TtsRequest {
        text: text.into(),
        reference: pcm.clone(),
        language: Language::Auto,
        seed: Some(2025),
        output_policy,
    };
    for (name, text, language) in [
        ("zh", "你好，这是语音测试。今天的天气很好。", Language::Zh),
        (
            "en",
            "Hello world. This is a speech test. Have a good day.",
            Language::En,
        ),
    ] {
        let mut runs = Vec::new();
        for chunks in [true, false] {
            let started = Instant::now();
            let mut session = model
                .start(
                    request(
                        text,
                        if chunks {
                            OutputPolicy::Chunks
                        } else {
                            OutputPolicy::Collect
                        },
                    ),
                    &control,
                )
                .unwrap();
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "start must not synthesize"
            );
            let mut output = Vec::new();
            let mut resumed_after_audio = false;
            let mut first = None;
            let summary = loop {
                match session.advance().unwrap() {
                    TtsStep::Progress => {
                        if !output.is_empty() {
                            resumed_after_audio = true;
                        }
                    }
                    TtsStep::Audio(chunk) => {
                        first.get_or_insert_with(|| started.elapsed());
                        assert_eq!(chunk.start_frame, output.len() as u64);
                        assert_eq!(chunk.pcm.format, model_format());
                        assert!(!chunk.pcm.samples.is_empty() && chunk.pcm.samples.len() <= 8192);
                        assert!(chunk
                            .pcm
                            .samples
                            .iter()
                            .all(|x| x.is_finite() && x.abs() <= 0.99));
                        output.extend(chunk.pcm.samples);
                    }
                    TtsStep::Finished(summary) => break summary,
                }
            };
            assert!(matches!(
                session.advance(),
                Err(AudioError::InvalidSessionState)
            ));
            assert_eq!(summary.total_frames, output.len() as u64);
            assert!(summary.segments >= 2);
            assert_eq!(summary.resolved_language, language);
            assert_eq!(
                resumed_after_audio, chunks,
                "chunks must emit before subsequent segments are generated"
            );
            assert!(
                output.iter().any(|x| x.abs() > 1e-4),
                "non-silent synthesis"
            );
            eprintln!(
                "{name} chunks={chunks}: segments={} frames={} first={:?} total={:?}",
                summary.segments,
                summary.total_frames,
                first.unwrap(),
                started.elapsed()
            );
            if let Ok(directory) = std::env::var("IRONMLX_SYNTHESIS_OUTPUT") {
                std::fs::create_dir_all(&directory).unwrap();
                use ironmlx_audio::AudioIo;
                let bytes = ironmlx_audio::io::NativeAudioIo
                    .encode_wav(&PcmBuffer {
                        format: model_format(),
                        samples: output.clone(),
                    })
                    .unwrap();
                std::fs::write(
                    PathBuf::from(directory).join(format!(
                        "{name}-{}.wav",
                        if chunks { "chunks" } else { "collect" }
                    )),
                    bytes,
                )
                .unwrap();
            }
            runs.push(output);
        }
        eprintln!(
            "{name} memory after sessions: {:?}",
            mlx::memory::snapshot()
        );
        assert_eq!(
            runs[0], runs[1],
            "same-seed collect and chunks must be bit-identical"
        );
    }
    // Cancellation after start is terminal, and dropping it releases the exclusive lease.
    let mut session = model
        .start(request("你好。", OutputPolicy::Chunks), &control)
        .unwrap();
    control.cancelled.set(true);
    let started = Instant::now();
    assert!(matches!(session.advance(), Err(AudioError::Cancelled)));
    assert!(started.elapsed() < Duration::from_millis(100));
    assert!(matches!(
        session.advance(),
        Err(AudioError::InvalidSessionState)
    ));
    drop(session);
    control.cancelled.set(false);
    let mut session = model
        .start(request("", OutputPolicy::Chunks), &control)
        .unwrap();
    assert!(matches!(
        session.advance(),
        Err(AudioError::InvalidInput { .. })
    ));
    assert!(matches!(
        session.advance(),
        Err(AudioError::InvalidSessionState)
    ));
}
fn model_format() -> PcmFormat {
    PcmFormat {
        sample_rate: 22050,
        channels: 1,
    }
}
