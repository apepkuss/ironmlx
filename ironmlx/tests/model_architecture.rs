use ironmlx::models::ModelArchitecture;
use serde_json::json;

#[test]
fn qwen_declared_model_types_map_to_execution_architectures() {
    let qwen35_dense = json!({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"]
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&qwen35_dense).unwrap(),
        ModelArchitecture::Qwen35Dense
    );

    let qwen36_27b_dense = json!({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120
        },
        "vision_config": {
            "model_type": "qwen3_5"
        },
        "image_token_id": 248056
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&qwen36_27b_dense).unwrap(),
        ModelArchitecture::Qwen35Dense
    );

    let qwen36_moe = json!({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8
        },
        "vision_config": {
            "model_type": "qwen3_5_moe"
        },
        "image_token_id": 248056,
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "language_model.model.layers.0.mlp.gate": {
                "bits": 8,
                "group_size": 64
            }
        }
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&qwen36_moe).unwrap(),
        ModelArchitecture::Qwen35Moe
    );

    let glm47_flash = json!({
        "model_type": "glm4_moe_lite",
        "architectures": ["Glm4MoeLiteForCausalLM"]
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&glm47_flash).unwrap(),
        ModelArchitecture::Glm4MoeLite
    );
}

#[test]
fn llama_declared_model_type_maps_to_execution_architecture() {
    // MiniCPM5-1B ships `model_type = "llama"` (architectures =
    // ["LlamaForCausalLM"]); it is a standard GQA dense Llama checkpoint.
    let minicpm5 = json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&minicpm5).unwrap(),
        ModelArchitecture::Llama
    );
}

#[test]
fn llama_round_trips_model_type_string() {
    assert_eq!(ModelArchitecture::Llama.model_type(), "llama");
}

#[test]
fn unsupported_model_type_reports_supported_architectures() {
    let config = json!({ "model_type": "qwen2_vl" });

    let err = ModelArchitecture::from_config_value(&config).unwrap_err();

    assert_eq!(
        err.to_string(),
        "unsupported model_type: qwen2_vl (expected 'qwen3_5', 'qwen3_5_moe', 'gemma4', 'glm4_moe_lite', or 'llama')"
    );
}
