use super::*;
use std::{cell::Cell, path::PathBuf, time::Instant};
fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if let Ok(root) = std::env::var("MLX_DIR") {
            mlx::metal::set_metallib_path(&format!("{root}/lib/mlx.metallib")).unwrap();
        }
    });
}
struct Control(Cell<bool>);
impl SessionControl for Control {
    fn check(&self) -> crate::Result<()> {
        if self.0.get() {
            Err(AudioError::Cancelled)
        } else {
            Ok(())
        }
    }
}
fn compare(label: &str, actual: &Array, expected: &Array) {
    assert_eq!(actual.shape(), expected.shape(), "{label}");
    let a = actual
        .astype(Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let b = expected
        .astype(Dtype::Float32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mse = a
        .iter()
        .zip(&b)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    let rms = (b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
    eprintln!("{label}: max={max}, rel_rms={}", mse.sqrt() / rms.max(1e-8));
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(
        mse.sqrt() / rms.max(1e-8) < 0.03,
        "{label}: excessive numerical drift"
    );
}
#[test]
fn options_and_reserved_sampling() {
    init();
    let mut o = BreezeOptions::default();
    o.temperature = 0.;
    let mut logits = vec![0f32; 2052];
    logits[2049] = 100.;
    logits[7] = 10.;
    let mut key = mlx::random::key(1).unwrap();
    let a = Array::try_from((logits.as_slice(), [2052])).unwrap();
    assert_eq!(sample(&a, true, &o, &mut key).unwrap(), 7);
    o.cfg_scale = f32::NAN;
    assert!(o.validate().is_err());
    o = BreezeOptions::default();
    o.max_tokens = 0;
    assert!(o.validate().is_err());
    o = BreezeOptions::default();
    o.instructions = Some(" ".into());
    assert!(o.validate().is_err());
}
#[test]
fn local_mask_is_bidirectional_and_bounded() {
    init();
    let m = transformer::local_mask(7, 4)
        .unwrap()
        .to_vec::<bool>()
        .unwrap();
    assert!(m[3 * 7 + 2]);
    assert!(!m[3 * 7 + 1]);
    assert!(m[3 * 7 + 5]);
    assert!(!m[3 * 7]);
    assert!(!m[3 * 7 + 6]);
    assert!(m[1]); // future token is visible, unlike causal attention
    assert_eq!(
        split_text(&"好".repeat(1201))
            .iter()
            .map(|s| s.chars().count())
            .collect::<Vec<_>>(),
        [600, 600, 1]
    );
}
#[test]
#[ignore = "requires IRONMLX_BREEZE_SNAPSHOT and IRONMLX_BREEZE_REFERENCE, MLX_ENABLE_TF32=0"]
fn real_network_parity() {
    init();
    let path = PathBuf::from(std::env::var("IRONMLX_BREEZE_SNAPSHOT").unwrap());
    let reference = std::env::var("IRONMLX_BREEZE_REFERENCE").unwrap();
    let (r, _) = mlx::io::load_safetensors(&reference).unwrap();
    let m = BreezeTts::load(&path, 600 * 24000).unwrap();
    let c = Control(Cell::new(false));
    for (label, text, instruction) in [
        ("en", "Hello.", "A clear and calm English narrator."),
        ("zh", "你好。", "一位温柔清晰的中文女声。"),
    ] {
        let ids = m
            .tokenizer
            .encode(format!("[S0]<ins_bos>{instruction}<ins_eos>{text}"), true)
            .unwrap();
        assert_eq!(
            ids.get_ids(),
            r[&format!("{label}.ids")]
                .astype(Dtype::Uint32)
                .unwrap()
                .to_vec::<u32>()
                .unwrap()
        );
        let prompt = m
            .encode_text(&format!("[S0]<ins_bos>{instruction}<ins_eos>{text}"), &c)
            .unwrap();
        compare(
            &format!("{label}.prompt"),
            &prompt,
            &r[&format!("{label}.prompt")],
        );
        let mut cache = m.backbone.cache();
        let h = last(m.backbone.forward(prompt, &mut cache, &c).unwrap()).unwrap();
        compare(
            &format!("{label}.hidden"),
            &h,
            &r[&format!("{label}.hidden")],
        );
        compare(
            &format!("{label}.logits"),
            &m.head.forward(&h).unwrap(),
            &r[&format!("{label}.logits")],
        );
        let frame = r[&format!("{label}.frame")].to_vec::<i32>().unwrap();
        let mut dc = m.depth.cache();
        m.depth
            .forward(m.depth_proj.forward(&h).unwrap(), &mut dc, &c)
            .unwrap()
            .eval()
            .unwrap();
        for i in 0..15 {
            let ids = Array::try_from((&[frame[i] + i as i32 * 2051][..], [1, 1])).unwrap();
            let x = m
                .depth_proj
                .forward(&m.depth_embed.forward(&ids).unwrap())
                .unwrap();
            let d = m.depth.forward(x, &mut dc, &c).unwrap();
            let head = m
                .depth_head
                .slice([i as i32, 0, 0], [i as i32 + 1, 1024, 2051])
                .unwrap()
                .reshape([1024, 2051])
                .unwrap();
            let logits = ops::matmul(&d, &head).unwrap().reshape([1, 2051]).unwrap();
            compare(
                &format!("{label}.depth.{i}"),
                &logits,
                &r[&format!("{label}.depth.{i}")],
            );
        }
        let codes = Array::try_from((frame.as_slice(), [1, 1, 16])).unwrap();
        let next = last(
            m.backbone
                .forward(m.audio_embedding(&codes).unwrap(), &mut cache, &c)
                .unwrap(),
        )
        .unwrap();
        compare(
            &format!("{label}.next"),
            &next,
            &r[&format!("{label}.next")],
        );
    }
}
#[test]
#[ignore = "requires IRONMLX_BREEZE_SNAPSHOT; optionally IRONMLX_BREEZE_OUTPUT"]
fn real_synthesis_acceptance() {
    init();
    let path = PathBuf::from(std::env::var("IRONMLX_BREEZE_SNAPSHOT").unwrap());
    let c = Control(Cell::new(false));
    let start = Instant::now();
    let mut model = BreezeTts::load(&path, 600 * 24000).unwrap();
    eprintln!(
        "load {:?} memory {:?}",
        start.elapsed(),
        mlx::memory::snapshot()
    );
    let mut reference = None;
    for (name, text, instructions, cfg) in [
        (
            "design-en",
            "Hello.",
            Some("A clear and calm English narrator."),
            1.,
        ),
        ("design-zh", "你好。", Some("一位温柔清晰的中文女声。"), 4.),
        ("clone", "Welcome.", None, 1.),
        (
            "direction",
            "Good morning.",
            Some("Speak softly and warmly."),
            4.,
        ),
    ] {
        let use_reference = matches!(name, "clone" | "direction");
        let options = BreezeOptions {
            instructions: instructions.map(String::from),
            cfg_scale: cfg,
            ref_text: if use_reference {
                Some("Hello.".into())
            } else {
                None
            },
            ..Default::default()
        };
        let request = TtsRequest {
            text: text.into(),
            reference: if use_reference {
                reference.clone().unwrap()
            } else {
                PcmBuffer {
                    format: PcmFormat {
                        sample_rate: 24000,
                        channels: 1,
                    },
                    samples: vec![],
                }
            },
            language: Language::Auto,
            seed: Some(42),
            output_policy: OutputPolicy::Chunks,
            breeze: Some(options),
        };
        let start = Instant::now();
        let mut session = model.start(request, &c).unwrap();
        let mut samples = Vec::new();
        let mut first = true;
        loop {
            match session.advance().unwrap() {
                TtsStep::Progress => {}
                TtsStep::Audio(chunk) => {
                    if first {
                        eprintln!("{name} first audio {:?}", start.elapsed());
                        first = false;
                    }
                    assert_eq!(chunk.start_frame, samples.len() as u64);
                    samples.extend(chunk.pcm.samples);
                }
                TtsStep::Finished(summary) => {
                    assert_eq!(summary.total_frames, samples.len() as u64);
                    break;
                }
            }
        }
        drop(session);
        assert!(samples.len() > 2400, "{name}: empty/too short");
        assert!(samples.iter().all(|v| v.is_finite()));
        let seconds = samples.len() as f64 / 24000.;
        eprintln!(
            "{name} seconds={seconds:.3} elapsed={:?} RTF={:.3} memory={:?}",
            start.elapsed(),
            start.elapsed().as_secs_f64() / seconds,
            mlx::memory::snapshot()
        );
        let pcm = PcmBuffer {
            format: PcmFormat {
                sample_rate: 24000,
                channels: 1,
            },
            samples,
        };
        if name == "design-en" {
            reference = Some(pcm.clone());
        }
        if let Ok(out) = std::env::var("IRONMLX_BREEZE_OUTPUT") {
            std::fs::create_dir_all(&out).unwrap();
            let mut bytes = Vec::new();
            crate::io::write_wav(&pcm, &mut bytes).unwrap();
            std::fs::write(Path::new(&out).join(format!("{name}.wav")), bytes).unwrap();
        }
    }
    let design_request = |output_policy| TtsRequest {
        text: "Hello.".into(),
        reference: PcmBuffer {
            format: PcmFormat {
                sample_rate: 24000,
                channels: 1,
            },
            samples: Vec::new(),
        },
        language: Language::Auto,
        seed: Some(42),
        output_policy,
        breeze: Some(BreezeOptions {
            instructions: Some("A clear and calm English narrator.".into()),
            ..Default::default()
        }),
    };
    let mut session = model
        .start(design_request(OutputPolicy::Collect), &c)
        .unwrap();
    let mut collected = Vec::new();
    loop {
        match session.advance().unwrap() {
            TtsStep::Audio(chunk) => {
                assert_eq!(chunk.start_frame, collected.len() as u64);
                collected.extend(chunk.pcm.samples);
            }
            TtsStep::Finished(_) => break,
            TtsStep::Progress => {}
        }
    }
    assert!(matches!(
        session.advance(),
        Err(AudioError::InvalidSessionState)
    ));
    drop(session);
    assert_eq!(
        collected,
        reference.unwrap().samples,
        "same seed and collect/chunks must produce identical samples across requests"
    );
    let mut session = model
        .start(design_request(OutputPolicy::Chunks), &c)
        .unwrap();
    assert!(matches!(session.advance().unwrap(), TtsStep::Progress));
    c.0.set(true);
    assert!(matches!(session.advance(), Err(AudioError::Cancelled)));
    assert!(matches!(
        session.advance(),
        Err(AudioError::InvalidSessionState)
    ));
    drop(session);
    let request = TtsRequest {
        text: "Hello".into(),
        reference: PcmBuffer {
            format: PcmFormat {
                sample_rate: 24000,
                channels: 1,
            },
            samples: vec![],
        },
        language: Language::Auto,
        seed: Some(42),
        output_policy: OutputPolicy::Chunks,
        breeze: Some(BreezeOptions {
            instructions: Some("A calm narrator".into()),
            ..Default::default()
        }),
    };
    assert!(matches!(
        model.start(request, &c),
        Err(AudioError::Cancelled)
    ));
}
