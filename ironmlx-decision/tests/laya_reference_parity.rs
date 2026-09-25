//! Real-checkpoint comparison with laya-mlx 0a859518 at the pinned HF revision.
//! Run with LAYA_MODEL_DIR and LAYA_METALLIB set to the checkpoint and metallib paths.

use std::path::Path;

use ironmlx_decision::{ComputeDtype, DecisionRequest, DecisionSettings, Laya};
use serde_json::Value;

#[test]
fn pinned_checkpoint_matches_reference_decisions() {
    let Some(model_dir) = std::env::var_os("LAYA_MODEL_DIR") else {
        return;
    };
    if let Some(metallib) = std::env::var_os("LAYA_METALLIB") {
        mlx::metal::set_metallib_path(metallib.to_str().expect("UTF-8 metallib path"))
            .expect("load metallib");
    }
    for dtype in [ComputeDtype::Float16, ComputeDtype::Float32] {
        for batch_size in [1, 2, 16] {
            for cache_prompts in [false, true] {
                let settings = DecisionSettings {
                    dtype,
                    batch_size,
                    cache_prompts,
                    ..Default::default()
                };
                let model = Laya::load_with_settings(Path::new(&model_dir), settings)
                    .expect("load settings");
                verify_reference(&model, dtype, None);
                // Repeated prompts with varied states must still produce fresh decisions.
                if cache_prompts {
                    verify_reference(&model, dtype, None);
                }
            }
        }
    }
}

fn verify_reference(model: &Laya, dtype: ComputeDtype, advanced: Option<DecisionSettings>) {
    for index in 0..4 {
        let request: DecisionRequest = serde_json::from_slice(
            &std::fs::read(format!(
                "tests/fixtures/laya-reference/{index}.request.json"
            ))
            .expect("read request fixture"),
        )
        .expect("parse request fixture");
        let reference: Value = serde_json::from_slice(
            &std::fs::read(format!(
                "tests/fixtures/laya-reference/{index}{}.reference.json",
                if let Some(settings) = advanced {
                    format!(
                        ".{}.{}.{}",
                        if settings.device == ironmlx_decision::ComputeDevice::Cpu {
                            "cpu"
                        } else {
                            "gpu"
                        },
                        if settings.compile {
                            "compiled"
                        } else {
                            "eager"
                        },
                        if dtype == ComputeDtype::Float32 {
                            "float32"
                        } else {
                            "float16"
                        }
                    )
                } else if dtype == ComputeDtype::Float32 {
                    ".float32".to_string()
                } else {
                    String::new()
                }
            ))
            .expect("read reference fixture"),
        )
        .expect("parse reference fixture");
        let actual = serde_json::to_value(model.predict(&request).expect("native prediction"))
            .expect("serialize native prediction");
        assert_eq!(actual["model"], "aac6fef/laya-multilingual-mlx");
        assert_eq!(
            actual["usage"], reference["usage"],
            "fixture {index} token usage"
        );
        let answers = actual["answers"].as_object().expect("answers map");
        assert_eq!(
            answers.len(),
            reference["answers"].as_object().unwrap().len()
        );
        for (name, answer) in answers {
            let expected = &reference["answers"][name];
            assert_eq!(answer["type"], expected["type"], "fixture {index}, {name}");
            assert!(answer.get("action").is_none(), "action head is not public");
            match answer["type"].as_str().unwrap() {
                "choice" => assert_eq!(answer["choice"], expected["choice"]),
                "score" => {
                    for (key, value) in answer["legend"].as_object().unwrap() {
                        let original = &expected["legend"][key];
                        let text = value.as_str().expect("string legend value");
                        if let Some(original_text) = original.as_str() {
                            assert_eq!(text, original_text);
                        } else {
                            assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *original);
                        }
                    }
                }
                "noul" => assert!(answer.get("confidence").is_none()),
                kind => panic!("unexpected answer type {kind}"),
            }
            for field in ["confidence", "score", "noul"] {
                if let Some(value) = answer.get(field) {
                    let error = (value.as_f64().unwrap() - expected[field].as_f64().unwrap()).abs();
                    assert!(
                        error <= 0.001 + 1e-12,
                        "fixture {index}, {name}.{field}: {error}"
                    );
                }
            }
            if let Some(probabilities) = answer.get("probabilities") {
                for (option, value) in probabilities.as_object().unwrap() {
                    let error = (value.as_f64().unwrap()
                        - expected["probabilities"][option].as_f64().unwrap())
                    .abs();
                    assert!(
                        error <= 0.001 + 1e-12,
                        "fixture {index}, {name}.{option}: {error}"
                    );
                }
            }
        }
    }

    let criteria: serde_json::Map<String, Value> = (0..255)
        .map(|index| {
            (
                format!("option-{index}"),
                Value::String("description".into()),
            )
        })
        .collect();
    let oversized: DecisionRequest = serde_json::from_value(serde_json::json!({
        "model": "aac6fef/laya-multilingual-mlx",
        "state": "hello",
        "questions": {"too_many_tokens": {
            "type": "choice", "instructions": "Choose one", "criteria": criteria
        }}
    }))
    .unwrap();
    assert!(
        model.predict(&oversized).is_err(),
        "oversized options must be rejected"
    );
}

#[test]
fn advanced_settings_preserve_decisions_across_devices_and_shapes() {
    let Some(model_dir) = std::env::var_os("LAYA_MODEL_DIR") else {
        return;
    };
    if let Some(metallib) = std::env::var_os("LAYA_METALLIB") {
        mlx::metal::set_metallib_path(metallib.to_str().unwrap()).unwrap();
    }
    for device in [
        ironmlx_decision::ComputeDevice::Gpu,
        ironmlx_decision::ComputeDevice::Cpu,
    ] {
        for dtype in [ComputeDtype::Float16, ComputeDtype::Float32] {
            for compile in [false, true] {
                let settings = DecisionSettings {
                    device,
                    dtype,
                    compile,
                    pad_to_multiple: Some(16),
                    batch_size: 2,
                    ..Default::default()
                };
                eprintln!("Checking {settings:?}");
                let model = Laya::load_with_settings(Path::new(&model_dir), settings).unwrap();
                verify_reference(&model, dtype, Some(settings));
                if compile {
                    verify_reference(&model, dtype, Some(settings));
                }
            }
        }
    }
}
