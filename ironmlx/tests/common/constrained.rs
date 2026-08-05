use ironmlx::core::constrained::{ConstraintPlan, ConstraintTokenizer, ToolConstraintOptions};
use ironmlx::core::tool_calling::ToolDefinition;
use serde_json::{json, Map, Value};

const BYTE_VOCAB_SIZE: usize = 257;
const BYTE_EOS_TOKEN: u32 = 256;

pub fn byte_vocab_size() -> usize {
    BYTE_VOCAB_SIZE
}

pub fn weather_constraint_plan() -> ConstraintPlan {
    weather_constraint_plan_with_options(&ToolConstraintOptions::default())
}

pub fn weather_constraint_plan_with_options(options: &ToolConstraintOptions) -> ConstraintPlan {
    let mut vocab = Map::new();
    for byte in 0_u16..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), Value::from(byte));
    }
    let tokenizer_json = json!({
        "added_tokens": [{
            "id": BYTE_EOS_TOKEN,
            "content": "<eos>",
            "special": true
        }],
        "decoder": {
            "type": "Sequence",
            "decoders": [{"type": "ByteFallback"}]
        },
        "model": {"vocab": vocab}
    });
    let tokenizer =
        ConstraintTokenizer::from_tokenizer_json(&tokenizer_json, &[BYTE_EOS_TOKEN], &[])
            .expect("build byte-level constraint tokenizer");
    let tool: ToolDefinition = serde_json::from_value(json!({
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
            "additionalProperties": false
        }
    }))
    .expect("weather tool definition");
    tokenizer
        .compile_qwen_tools(&[tool], options)
        .expect("compile byte-level weather constraint")
}
