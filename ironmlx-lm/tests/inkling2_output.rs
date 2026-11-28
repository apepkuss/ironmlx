use anyhow::Result;
use ironmlx_lm::core::{
    generated_output::GeneratedOutputEvent as Event,
    native_output::{NativeOutputDecoderConfig, NativeOutputDialect, NativeOutputParser},
    tool_calling::{ToolCallParser, ToolDefinition, ToolDialect},
};
use serde_json::json;

fn config() -> NativeOutputDecoderConfig {
    NativeOutputDecoderConfig {
        dialect: NativeOutputDialect::Inkling2,
        reasoning_enabled: true,
    }
}

#[test]
fn inkling_channels_and_tools_survive_every_utf8_split() -> Result<()> {
    let wire = concat!("<|message_model|><|content_thinking|>查天气。<|end_message|>",
        "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"东京\"}}<|end_message|>",
        "<|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|>");
    let tools = vec![ToolDefinition {
        name: "weather".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}),
        strict: Some(true),
    }];
    for split in (0..=wire.len()).filter(|n| wire.is_char_boundary(*n)) {
        let mut native = NativeOutputParser::new(config());
        let mut tool = ToolCallParser::new(ToolDialect::Inkling2, "test", &tools)?;
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut calls = Vec::new();
        let mut events = native.push(&wire[..split])?;
        events.extend(native.push(&wire[split..])?);
        events.extend(native.finish("stop")?);
        let mut decoded = Vec::new();
        for event in events {
            match event {
                Event::TextDelta(delta) => decoded.extend(tool.push(&delta)?),
                Event::ReasoningDelta(delta) => reasoning.push_str(&delta),
                _ => panic!("unexpected event"),
            }
        }
        let (tail, saw) = tool.finish()?;
        assert!(saw);
        decoded.extend(tail);
        for event in decoded {
            match event {
                Event::TextDelta(delta) => text.push_str(&delta),
                Event::ToolCall(call) => calls.push(call),
                _ => panic!("unexpected tool event"),
            }
        }
        assert_eq!(reasoning, "查天气。");
        assert_eq!(text, "Done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(calls[0].arguments, json!({"city":"东京"}));
    }
    Ok(())
}

#[test]
fn inkling_length_stop_does_not_expose_partial_headers() -> Result<()> {
    for text in [
        "<|message_model|>",
        "<|message_model|><|content_th",
        "<|message_model|><|content_thinking|>",
    ] {
        let mut parser = NativeOutputParser::new(config());
        assert!(parser.push(text)?.is_empty());
        assert!(parser.finish("length")?.is_empty());
    }
    Ok(())
}

#[test]
fn inkling_output_cut_inside_a_call_keeps_completed_calls() -> Result<()> {
    let tools = vec![ToolDefinition {
        name: "weather".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}),
        strict: Some(true),
    }];
    let call = |city: &str| {
        format!("<|message_model|>weather<|content_invoke_tool_json|>{{\"name\":\"weather\",\"args\":{{\"city\":\"{city}\"}}}}<|end_message|>")
    };
    let complete = call("Oslo");
    let second = call("Lima");
    let cut = &second[..second.find("Lima").unwrap()];
    let wire = format!("{complete}{cut}");
    let mut native = NativeOutputParser::new(config());
    let mut tool = ToolCallParser::new(ToolDialect::Inkling2, "cut", &tools)?;
    let mut events = native.push(&wire)?;
    events.extend(native.finish("length")?);
    let mut decoded = Vec::new();
    for event in events {
        if let Event::TextDelta(delta) = event {
            decoded.extend(tool.push(&delta)?);
        }
    }
    let (tail, saw, dropped) = tool.finish_truncated()?;
    decoded.extend(tail);
    assert!(saw && dropped);
    let calls: Vec<_> = decoded
        .into_iter()
        .filter_map(|event| match event {
            Event::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, json!({"city":"Oslo"}));
    Ok(())
}
