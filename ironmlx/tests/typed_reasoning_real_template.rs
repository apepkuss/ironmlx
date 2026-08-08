use std::path::Path;

use ironmlx::core::generated_output::{GeneratedOutputDecoder, GeneratedOutputEvent};
use ironmlx::core::native_output::NativeOutputDialect;
use ironmlx::core::tool_calling::{AgentMessage, TemplateFunctionCall, TemplateToolCall};
use ironmlx::core::Tokenizer;

fn message(role: &str, content: &str, reasoning_content: Option<&str>) -> AgentMessage {
    AgentMessage {
        role: role.to_owned(),
        content: Some(content.to_owned()),
        reasoning_content: reasoning_content.map(str::to_owned),
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn assistant_tool_call(reasoning_content: &str) -> AgentMessage {
    AgentMessage {
        role: "assistant".to_owned(),
        content: None,
        reasoning_content: Some(reasoning_content.to_owned()),
        tool_calls: vec![TemplateToolCall {
            id: "call_weather".to_owned(),
            kind: "function",
            function: TemplateFunctionCall {
                name: "get_weather".to_owned(),
                arguments: serde_json::json!({"city":"Tokyo"}),
            },
        }],
        tool_call_id: None,
    }
}

fn tool_result() -> AgentMessage {
    AgentMessage {
        role: "tool".to_owned(),
        content: Some(r#"{"temperature_c":22}"#.to_owned()),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: Some("call_weather".to_owned()),
    }
}

#[test]
#[ignore = "requires IRONMLX_REASONING_MODEL pointing to a supported real checkpoint"]
fn real_template_and_tokenizer_round_trip_native_reasoning() {
    let model = std::env::var("IRONMLX_REASONING_MODEL")
        .expect("IRONMLX_REASONING_MODEL must point to a real checkpoint");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model)).expect("load tokenizer");
    let dialect = tokenizer
        .native_output_dialect()
        .expect("checkpoint must expose an exact native reasoning contract");
    assert!(tokenizer
        .capability_profile(false)
        .output
        .reasoning
        .is_supported());
    if matches!(dialect, NativeOutputDialect::Glm) {
        let user_role = tokenizer
            .encode("<|user|>", false)
            .expect("encode GLM user role token");
        assert_eq!(user_role.len(), 1);
        assert!(tokenizer.eos_token_ids().contains(&user_role[0]));
    }

    let kwargs = serde_json::json!({"enable_thinking":true});
    let prompt = tokenizer
        .apply_chat_template(
            &[
                message("user", "first question", None),
                assistant_tool_call("previous plan"),
                tool_result(),
            ],
            true,
            Some(&kwargs),
        )
        .expect("render native reasoning history");
    assert!(prompt.contains("previous plan"));

    let native = match dialect {
        NativeOutputDialect::Gemma => "<|channel>thought\nnative plan<channel|>final answer",
        _ => "native plan</think>\n\nfinal answer",
    };
    let token_ids = tokenizer
        .encode(native, false)
        .expect("encode native output");
    let config = tokenizer
        .native_output_decoder_config(Some(&kwargs))
        .expect("resolve decoder config");

    let structured_native = match dialect {
        NativeOutputDialect::Gemma => {
            "<|channel>thought\nnative plan<channel|>{\"answer\":\"sunny\"}"
        }
        _ => "native plan</think>\n\n{\"answer\":\"sunny\"}",
    };
    let structured_tokens = tokenizer
        .encode(structured_native, false)
        .expect("encode reasoning plus structured output");
    let plan = tokenizer
        .compile_json_output_constraint_with_reasoning(
            &serde_json::json!({
                "type": "object",
                "properties": {"answer": {"const": "sunny"}},
                "required": ["answer"],
                "additionalProperties": false
            }),
            config.expect("reasoning-aware constraint config"),
        )
        .expect("compile reasoning-aware structured output constraint");
    let mut constraint = plan.start_session().expect("start structured matcher");
    constraint
        .commit_tokens(&structured_tokens)
        .expect("consume reasoning plus structured output");
    assert!(constraint
        .is_accepting()
        .expect("structured accepting state"));

    let mut decoder = GeneratedOutputDecoder::new_with_native(&tokenizer, None, config)
        .expect("construct decoder");
    let mut reasoning = String::new();
    let mut text = String::new();
    for token in token_ids {
        for event in decoder.push_token(token).expect("decode token") {
            match event {
                GeneratedOutputEvent::ReasoningDelta(delta) => reasoning.push_str(&delta),
                GeneratedOutputEvent::TextDelta(delta) => text.push_str(&delta),
                other => panic!("unexpected nonterminal event: {other:?}"),
            }
        }
    }
    for event in decoder.finish("stop").expect("finish decoder") {
        match event {
            GeneratedOutputEvent::ReasoningDelta(delta) => reasoning.push_str(&delta),
            GeneratedOutputEvent::TextDelta(delta) => text.push_str(&delta),
            GeneratedOutputEvent::Finished(_) => {}
            other => panic!("unexpected terminal event: {other:?}"),
        }
    }
    assert_eq!(reasoning, "native plan");
    assert_eq!(text, "final answer");
}
