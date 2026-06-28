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
fn gemma4_unified_declared_model_type_maps_to_gemma4_execution_architecture() {
    let gemma4_unified = json!({
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_unified_text",
            "enable_moe_block": false,
            "num_hidden_layers": 48
        },
        "vision_config": {
            "model_type": "gemma4_unified_vision",
            "num_soft_tokens": 280
        },
        "image_token_id": 258880
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&gemma4_unified).unwrap(),
        ModelArchitecture::Gemma4
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
fn minicpmv46_declared_model_type_maps_to_execution_architecture() {
    // MiniCPM-V-4.6 ships `model_type = "minicpmv4_6"` with a nested
    // `text_config.model_type = "qwen3_5_text"` Qwen3.5-text backbone. It maps
    // to a dedicated execution architecture that runs the text-only Qwen3.5
    // dense graph.
    let minicpmv46 = json!({
        "model_type": "minicpmv4_6",
        "architectures": ["MiniCPMV4_6ForConditionalGeneration"],
        "text_config": { "model_type": "qwen3_5_text" },
        "vision_config": { "model_type": "minicpmv4_6_vision" },
        "image_token_id": 248056
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&minicpmv46).unwrap(),
        ModelArchitecture::MiniCpmV46
    );
}

#[test]
fn minicpmv46_round_trips_model_type_string() {
    assert_eq!(ModelArchitecture::MiniCpmV46.model_type(), "minicpmv4_6");
}

#[test]
fn diffusion_gemma_declared_model_type_maps_to_execution_architecture() {
    let diffusion_gemma = json!({
        "model_type": "diffusion_gemma",
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "canvas_length": 256,
        "text_config": { "model_type": "diffusion_gemma_text" }
    });
    assert_eq!(
        ModelArchitecture::from_config_value(&diffusion_gemma).unwrap(),
        ModelArchitecture::DiffusionGemma
    );
}

#[test]
fn diffusion_gemma_round_trips_model_type_string() {
    assert_eq!(
        ModelArchitecture::DiffusionGemma.model_type(),
        "diffusion_gemma"
    );
}

#[test]
fn unsupported_model_type_reports_supported_architectures() {
    let config = json!({ "model_type": "qwen2_vl" });

    let err = ModelArchitecture::from_config_value(&config).unwrap_err();

    assert_eq!(
        err.to_string(),
        "unsupported model_type: qwen2_vl (expected 'qwen3_5', 'qwen3_5_moe', 'gemma4', 'gemma4_unified', 'glm4_moe_lite', 'llama', 'minicpmv4_6', or 'diffusion_gemma')"
    );
}
