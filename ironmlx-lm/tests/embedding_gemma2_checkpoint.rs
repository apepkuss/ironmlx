//! Real checkpoint qualification. See docs/text-embeddings.md for environment variables.
use ironmlx_lm::models::embedding_gemma2::{EmbeddingGemma2Model, MAX_INPUT_TOKENS};
use serde_json::Value;

#[test]
#[ignore = "requires local BF16 and affine4 EmbeddingGemma 2 checkpoints"]
fn embedding_gemma2_matches_pinned_mlx_reference() -> anyhow::Result<()> {
    if let Ok(dir) = std::env::var("MLX_DIR") {
        mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib"))?;
    }
    let device = mlx::Device::gpu(0);
    mlx::set_default_device(device);
    mlx::set_default_stream(mlx::new_stream(device)?);
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/embedding_gemma2/reference.json"))?;
    let texts: Vec<String> = serde_json::from_value(reference["inputs"].clone())?;
    for (precision, variable) in [
        ("bf16", "IRONMLX_EMBEDDING_BF16_DIR"),
        ("4bit", "IRONMLX_EMBEDDING_AFFINE4_DIR"),
    ] {
        let path = std::env::var(variable)?;
        let model = EmbeddingGemma2Model::load(std::path::Path::new(&path))?;
        let output = model.encode(&texts, 768)?;
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(reference[precision]["embeddings"].clone())?;
        let ids: Vec<Vec<u32>> = serde_json::from_value(reference[precision]["tokens"].clone())?;
        assert_eq!(output.input_tokens, ids.iter().map(Vec::len).sum::<usize>());
        for (actual, expected) in output.embeddings.iter().zip(&expected) {
            assert_eq!(actual.len(), 768);
            let maximum = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                maximum < 0.0001,
                "{precision}: maximum vector error {maximum}"
            );
        }
        for dimensions in [128, 256, 512] {
            let reduced = model.encode(&texts[..1], dimensions)?;
            let vector = &reduced.embeddings[0];
            let norm = output.embeddings[0][..dimensions]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            for (a, b) in vector.iter().zip(&output.embeddings[0]) {
                assert!((a - b / norm).abs() < 0.000001);
            }
            assert!((vector.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.00001);
        }
        assert!(model
            .encode_tokens(&vec![2; MAX_INPUT_TOKENS + 1], 768)
            .is_err());
        assert!(model.encode_tokens(&[u32::MAX], 768).is_err());
        assert!(model.encode_tokens(&[0], 768).is_err());
        assert!(model.encode(&[" ".into()], 768).is_err());
        assert!(model.encode(&["a ".repeat(8192)], 768).is_err());
        assert!(model.encode(&texts, 129).is_err());
    }
    Ok(())
}

#[test]
fn preflight_accepts_only_supported_embedding_formats() -> anyhow::Result<()> {
    let root = std::env::temp_dir().join(format!(
        "embedding-gemma2-preflight-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&root)?;
    let mut config: Value =
        serde_json::from_str(include_str!("fixtures/embedding_gemma2/config.json"))?;
    let check = |config: &Value| -> anyhow::Result<_> {
        std::fs::write(root.join("config.json"), serde_json::to_vec(config)?)?;
        ironmlx_lm::core::preflight_model_metadata(&root)
    };
    assert_eq!(check(&config)?.artifact_role, "embedding");
    config["quantization"] = serde_json::json!({"mode":"affine","bits":4,"group_size":64});
    assert_eq!(check(&config)?.quantization.unwrap().bits, 4);
    config["quantization"]["bits"] = 8.into();
    assert!(check(&config).is_err());
    config["quantization"] = serde_json::json!({"mode":"mxfp4","bits":4,"group_size":32});
    assert!(check(&config).is_err());
    config.as_object_mut().unwrap().remove("quantization");
    let vision: Value =
        serde_json::from_str(include_str!("fixtures/embedding_gemma2/vision-config.json"))?;
    for (key, value) in vision.as_object().unwrap() {
        config[key] = value.clone();
    }
    assert_eq!(check(&config)?.artifact_role, "embedding");
    assert!(ironmlx_lm::models::embedding_gemma2::checkpoint_supports_images(&root)?);
    config["vision_config"]["head_dim"] = 65.into();
    assert!(check(&config).is_err());
    config["vision_config"]["head_dim"] = 64.into();
    config["text_config"]["dtype"] = "float16".into();
    assert!(check(&config).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires local BF16 and affine4 EmbeddingGemma 2 checkpoints"]
fn embedding_gemma2_images_match_reference_and_retrieve() -> anyhow::Result<()> {
    use ironmlx_lm::models::embedding_gemma2::{EmbeddingContent, EmbeddingSample};
    let dir = std::env::var("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib"))?;
    let device = mlx::Device::gpu(0);
    mlx::set_default_device(device);
    mlx::set_default_stream(mlx::new_stream(device)?);
    let reference: Value = serde_json::from_str(include_str!(
        "fixtures/embedding_gemma2/image-reference.json"
    ))?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embedding_gemma2/images");
    let samples: Vec<EmbeddingSample> = reference["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sample| -> anyhow::Result<_> {
            let content = if let Some(text) = sample.as_str() {
                vec![EmbeddingContent::Text(text.to_owned())]
            } else {
                sample["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|part| -> anyhow::Result<_> {
                        Ok(if let Some(text) = part["text"].as_str() {
                            EmbeddingContent::Text(text.to_owned())
                        } else {
                            EmbeddingContent::Image(std::fs::read(
                                root.join(part["image"].as_str().unwrap()),
                            )?)
                        })
                    })
                    .collect::<anyhow::Result<_>>()?
            };
            Ok(EmbeddingSample { content })
        })
        .collect::<anyhow::Result<_>>()?;
    for (precision, variable) in [
        ("bf16", "IRONMLX_EMBEDDING_BF16_DIR"),
        ("4bit", "IRONMLX_EMBEDDING_AFFINE4_DIR"),
    ] {
        let path = std::env::var(variable)?;
        let model = EmbeddingGemma2Model::load(std::path::Path::new(&path))?;
        assert!(model.supports_images());
        let output = model.encode_inputs(&samples, 768)?;
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(reference[precision]["embeddings"].clone())?;
        let ids: Vec<Vec<u32>> = serde_json::from_value(reference[precision]["tokens"].clone())?;
        assert_eq!(output.input_tokens, ids.iter().map(Vec::len).sum::<usize>());
        for (i, (actual, expected)) in output.embeddings.iter().zip(&expected).enumerate() {
            let maximum = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                maximum < 0.0001,
                "{precision} image sample {i}: error {maximum}"
            );
            assert!((actual.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 0.00001);
            let single = model.encode_inputs(&samples[i..=i], 768)?;
            assert_eq!(single.embeddings[0], *actual);
        }
        for dimensions in [128, 256, 512] {
            let reduced = model.encode_inputs(&samples[..1], dimensions)?;
            let norm = output.embeddings[0][..dimensions]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(reduced.embeddings[0]
                .iter()
                .zip(&output.embeddings[0])
                .all(|(a, b)| (a - b / norm).abs() < 0.000001));
        }
        for (i, query) in output.embeddings[5..].iter().enumerate() {
            let scores: Vec<f32> = output.embeddings[..3]
                .iter()
                .map(|image| image.iter().zip(query).map(|(a, b)| a * b).sum())
                .collect();
            let top = (0..3)
                .max_by(|a, b| scores[*a].total_cmp(&scores[*b]))
                .unwrap();
            assert_eq!(top, i, "{precision} cross-modal retrieval: {scores:?}");
        }
        let image = samples[0].content[0].clone();
        let maximum_images = EmbeddingSample {
            content: vec![image.clone(); 8],
        };
        let maximum_output = model.encode_inputs(&[maximum_images], 768)?;
        assert_eq!(maximum_output.input_tokens, 2178);
        assert!(maximum_output.embeddings[0].iter().all(|x| x.is_finite()));
        let boundary = EmbeddingSample {
            content: vec![EmbeddingContent::Text("a ".repeat(7917)), image.clone()],
        };
        let boundary_output = model.encode_inputs(&[boundary], 768)?;
        assert_eq!(boundary_output.input_tokens, 8192);
        assert!(boundary_output.embeddings[0].iter().all(|x| x.is_finite()));
        let over_boundary = EmbeddingSample {
            content: vec![EmbeddingContent::Text("a ".repeat(7918)), image.clone()],
        };
        assert!(model.encode_inputs(&[over_boundary], 768).is_err());
        for bad in [
            EmbeddingSample {
                content: vec![EmbeddingContent::Image(vec![0; 16])],
            },
            EmbeddingSample {
                content: vec![image.clone(); 9],
            },
            EmbeddingSample {
                content: vec![EmbeddingContent::Text("a ".repeat(8192)), image],
            },
            EmbeddingSample {
                content: vec![EmbeddingContent::Text("<|image|>".into())],
            },
            EmbeddingSample {
                content: vec![EmbeddingContent::Text("<|audio|>".into())],
            },
            EmbeddingSample {
                content: vec![EmbeddingContent::Text("<|video|>".into())],
            },
        ] {
            assert!(model.encode_inputs(&[bad], 768).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires local BF16 and affine4 EmbeddingGemma 2 checkpoints"]
fn embedding_gemma2_audio_matches_reference_and_retrieves() -> anyhow::Result<()> {
    use ironmlx_lm::{
        core::audio_input::EmbeddingAudio,
        models::embedding_gemma2::{EmbeddingContent, EmbeddingSample},
    };
    let dir = std::env::var("MLX_DIR")?;
    mlx::metal::set_metallib_path(&format!("{dir}/lib/mlx.metallib"))?;
    let device = mlx::Device::gpu(0);
    mlx::set_default_device(device);
    mlx::set_default_stream(mlx::new_stream(device)?);
    let reference: Value = serde_json::from_str(include_str!(
        "fixtures/embedding_gemma2/audio-reference.json"
    ))?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embedding_gemma2/audio");
    let samples: Vec<EmbeddingSample> = reference["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sample| -> anyhow::Result<_> {
            let content = sample["content"]
                .as_array()
                .unwrap()
                .iter()
                .map(|part| -> anyhow::Result<_> {
                    Ok(if let Some(text) = part["text"].as_str() {
                        EmbeddingContent::Text(text.into())
                    } else {
                        EmbeddingContent::Audio(EmbeddingAudio::decode(&std::fs::read(
                            root.join(part["audio"].as_str().unwrap()),
                        )?)?)
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(EmbeddingSample { content })
        })
        .collect::<anyhow::Result<_>>()?;
    for (precision, variable) in [
        ("bf16", "IRONMLX_EMBEDDING_BF16_DIR"),
        ("4bit", "IRONMLX_EMBEDDING_AFFINE4_DIR"),
    ] {
        let model = EmbeddingGemma2Model::load(std::path::Path::new(&std::env::var(variable)?))?;
        assert!(model.supports_audio());
        assert!(model.supports_images());
        let mut vectors = Vec::new();
        for (group_index, group) in samples.chunks(6).enumerate() {
            let output = model.encode_inputs(group, 768)?;
            let expected_tokens: usize = reference[precision].as_array().unwrap()
                [group_index * 6..group_index * 6 + group.len()]
                .iter()
                .map(|row| row["tokens"].as_u64().unwrap() as usize)
                .sum();
            assert_eq!(output.input_tokens, expected_tokens);
            vectors.extend(output.embeddings);
        }
        for (i, actual) in vectors.iter().enumerate() {
            let expected: Vec<f32> =
                serde_json::from_value(reference[precision][i]["embedding"].clone())?;
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let cosine = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
            // Audio's FP32 Conformer feeds BF16 text activations. Floating-point
            // kernel rounding can cross BF16 boundaries; validate both metrics.
            assert!(
                max < 0.005 && cosine > 0.9995,
                "{precision} sample {i}: max={max}, cosine={cosine}"
            );
            assert!(actual.iter().all(|x| x.is_finite()));
            assert!((actual.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        }
        for query in 0..3 {
            let best = (0..3)
                .max_by(|a, b| {
                    let score = |i: usize| {
                        vectors[query + 3]
                            .iter()
                            .zip(&vectors[i])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                    };
                    score(*a).total_cmp(&score(*b))
                })
                .unwrap();
            assert_eq!(best, query, "{precision} speech retrieval");
        }
        assert_eq!(
            model.encode_inputs(&samples[6..8], 768)?.embeddings,
            vectors[6..8]
        );
        for dimensions in [128, 256, 512] {
            let output = model.encode_inputs(&samples[..1], dimensions)?;
            let norm = vectors[0][..dimensions]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            for (actual, full) in output.embeddings[0].iter().zip(&vectors[0]) {
                assert!((actual - full / norm).abs() < 1e-6);
            }
        }
        let long = EmbeddingSample {
            content: vec![EmbeddingContent::Audio(EmbeddingAudio::from_mono_16k(
                vec![0.0; 480000],
            )?)],
        };
        let output = model.encode_inputs(&[long], 768)?;
        assert_eq!(output.input_tokens, 754);
        assert!(output.embeddings[0].iter().all(|x| x.is_finite()));
        let nine = EmbeddingSample {
            content: vec![samples[0].content[0].clone(); 9],
        };
        assert!(model.encode_inputs(&[nine], 768).is_err());
        let aggregate = EmbeddingSample {
            content: vec![
                EmbeddingContent::Audio(EmbeddingAudio::from_mono_16k(vec![0.0; 480000])?);
                3
            ],
        };
        assert!(model.encode_inputs(&[aggregate], 768).is_err());
        let image = EmbeddingContent::Image(std::fs::read(
            root.parent().unwrap().join("images/red-circle.png"),
        )?);
        let mixed = EmbeddingSample {
            content: vec![
                samples[0].content[0].clone(),
                image,
                EmbeddingContent::Text("A recording and an image.".into()),
            ],
        };
        let output = model.encode_inputs(&[mixed], 256)?;
        assert!(output.embeddings[0].iter().all(|x| x.is_finite()));
    }
    Ok(())
}
