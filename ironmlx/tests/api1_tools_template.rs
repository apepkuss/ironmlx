use std::path::Path;

use ironmlx::core::tool_calling::{AgentMessage, TemplateToolCall, ToolCall, ToolDialect};
use ironmlx::core::Tokenizer;

#[test]
#[ignore = "requires QWEN35_MODEL pointing to a local Qwen3.5/3.6 checkpoint"]
fn real_qwen_template_renders_tools_calls_and_results() {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Qwen35));

    let call = ToolCall {
        id: "call_history_0".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "东京"}),
    };
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("东京天气如何？".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: Some("需要查询天气。\n</think>\n\n".into()),
            reasoning_content: None,
            tool_calls: vec![TemplateToolCall::from(call)],
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("晴，25°C".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_history_0".into()),
        },
    ];
    let kwargs = serde_json::json!({
        "enable_thinking": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render tool prompt");
    assert!(prompt.contains("<function=get_weather>"));
    assert!(prompt.contains("<parameter=city>\n东京\n</parameter>"));
    assert!(prompt.contains("<think>\n需要查询天气。\n</think>"));
    assert!(prompt.contains("<tool_response>\n晴，25°C\n</tool_response>"));
}

#[test]
#[ignore = "requires GEMMA4_MODEL pointing to a local Gemma 4 checkpoint"]
fn real_gemma_template_renders_tools_calls_and_results() {
    let model_dir = std::env::var("GEMMA4_MODEL").expect("GEMMA4_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Gemma));

    let call = ToolCall {
        id: "call_history_0".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "东京", "days": 2}),
    };
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("东京天气如何？".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: vec![TemplateToolCall::from(call)],
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("晴，25°C".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_history_0".into()),
        },
        AgentMessage {
            role: "user".into(),
            content: Some("谢谢".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
    ];
    let kwargs = serde_json::json!({
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render Gemma tool prompt");
    assert!(prompt.contains("<|tool>declaration:get_weather"));
    assert!(prompt.contains(concat!(
        "<|tool_call>call:get_weather{city:<|\"|>东京<|\"|>,days:2}",
        "<tool_call|>"
    )));
    assert!(prompt.contains(concat!(
        "<|tool_response>response:get_weather{value:<|\"|>晴，25°C<|\"|>}",
        "<tool_response|>"
    )));
    assert_eq!(prompt.matches("<|turn>model\n").count(), 2);
}

#[test]
#[ignore = "requires GLM47_MODEL pointing to a local GLM-4.7-Flash checkpoint"]
fn real_glm_template_renders_tools_calls_and_results() {
    let model_dir = std::env::var("GLM47_MODEL").expect("GLM47_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Glm));

    let call = ToolCall {
        id: "call_history_0".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "东京", "days": 2}),
    };
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("东京天气如何？".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: vec![TemplateToolCall::from(call)],
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("晴，25°C".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_history_0".into()),
        },
        AgentMessage {
            role: "user".into(),
            content: Some("谢谢".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
    ];
    let kwargs = serde_json::json!({
        "enable_thinking": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render GLM tool prompt");
    assert!(prompt.starts_with("[gMASK]<sop>"));
    assert!(prompt.contains("# Tools"));
    assert!(prompt.contains("\"name\":\"get_weather\""));
    assert!(prompt.contains(concat!(
        "<tool_call>get_weather",
        "<arg_key>city</arg_key><arg_value>东京</arg_value>",
        "<arg_key>days</arg_key><arg_value>2</arg_value>",
        "</tool_call>"
    )));
    assert!(prompt.contains("<|observation|><tool_response>晴，25°C</tool_response>"));
    assert!(prompt.ends_with("<|assistant|></think>"));
}

#[test]
#[ignore = "requires LLAMA32_MODEL pointing to a local Llama 3.1/3.2 Instruct checkpoint"]
fn real_llama_template_renders_tools_single_call_history_and_ipython_result() {
    let model_dir = std::env::var("LLAMA32_MODEL").expect("LLAMA32_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Llama));

    let call = ToolCall {
        id: "call_history_0".into(),
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "东京", "days": 2}),
    };
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("东京天气如何？".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: vec![TemplateToolCall::from(call.clone())],
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("晴，25°C".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_history_0".into()),
        },
        AgentMessage {
            role: "user".into(),
            content: Some("谢谢".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
    ];
    let kwargs = serde_json::json!({
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render Llama tool prompt");
    assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>system"));
    assert!(prompt.contains("Environment: ipython"));
    assert!(prompt.contains(concat!(
        "Respond in the format {\"name\": function name, ",
        "\"parameters\": dictionary of argument name and its value}."
    )));
    assert!(prompt.contains("    \"type\": \"function\""));
    assert!(prompt.contains(concat!(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "{\"name\": \"get_weather\", \"parameters\": ",
        "{\"city\":\"东京\",\"days\":2}}<|eot_id|>"
    )));
    assert!(prompt.contains(concat!(
        "<|start_header_id|>ipython<|end_header_id|>\n\n",
        "\"晴，25°C\"<|eot_id|>"
    )));
    assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

    let parallel_history = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("比较天气".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: vec![
                TemplateToolCall::from(call.clone()),
                TemplateToolCall::from(call),
            ],
            tool_call_id: None,
        },
    ];
    let error = tokenizer
        .apply_chat_template(&parallel_history, true, Some(&kwargs))
        .expect_err("Llama native template must reject parallel call history");
    assert!(error.to_string().contains("single tool-calls"));
}

#[test]
#[ignore = "requires MINICPMV46_MODEL pointing to a local MiniCPM-V 4.6 checkpoint"]
fn real_minicpmv46_template_mixes_image_placeholders_parallel_calls_and_results() {
    let model_dir = std::env::var("MINICPMV46_MODEL").expect("MINICPMV46_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::MiniCpmV46));

    let image = ironmlx::models::minicpmv4_6::image_placeholder_string(3);
    let calls = [
        ToolCall {
            id: "call_tokyo".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "东京"}),
        },
        ToolCall {
            id: "call_osaka".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "大阪"}),
        },
    ];
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some(format!("{image}\n比较图片地点的天气")),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: Some("需要查询两个城市。".into()),
            reasoning_content: None,
            tool_calls: calls.iter().cloned().map(TemplateToolCall::from).collect(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("东京：晴".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_tokyo".into()),
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("大阪：雨".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_osaka".into()),
        },
        AgentMessage {
            role: "user".into(),
            content: Some("给出结论".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
    ];
    let kwargs = serde_json::json!({
        "enable_thinking": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render MiniCPM-V tool prompt");
    assert!(prompt.contains(&image));
    assert_eq!(
        prompt
            .matches("<tool_call>\n<function=get_weather>")
            .count(),
        2
    );
    assert!(prompt.contains("<parameter=city>\n东京\n</parameter>"));
    assert!(prompt.contains("<parameter=city>\n大阪\n</parameter>"));
    assert!(prompt.contains(concat!(
        "<|im_start|>user\n<tool_response>\n东京：晴\n</tool_response>",
        "\n<tool_response>\n大阪：雨\n</tool_response><|im_end|>"
    )));
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
#[ignore = "requires MINICPM5_MODEL pointing to a local MiniCPM5 checkpoint"]
fn real_minicpm5_template_renders_cdata_parallel_history_and_results() {
    let model_dir = std::env::var("MINICPM5_MODEL").expect("MINICPM5_MODEL");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&model_dir)).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::MiniCpm5));

    let calls = [
        ToolCall {
            id: "call_special".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "东京\n<&"}),
        },
        ToolCall {
            id: "call_plain".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "大阪"}),
        },
    ];
    let messages = vec![
        AgentMessage {
            role: "user".into(),
            content: Some("比较天气".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: calls.iter().cloned().map(TemplateToolCall::from).collect(),
            tool_call_id: None,
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("东京：晴".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_special".into()),
        },
        AgentMessage {
            role: "tool".into(),
            content: Some("大阪：雨".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_plain".into()),
        },
    ];
    let kwargs = serde_json::json!({
        "enable_thinking": false,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    });

    let prompt = tokenizer
        .apply_chat_template(&messages, true, Some(&kwargs))
        .expect("render MiniCPM5 tool prompt");
    assert!(prompt.contains(concat!(
        "<function name=\"get_weather\"><param name=\"city\">",
        "<![CDATA[东京\n<&]]></param></function>"
    )));
    assert!(prompt.contains(concat!(
        "<function name=\"get_weather\"><param name=\"city\">",
        "大阪</param></function>"
    )));
    assert!(prompt.contains(concat!(
        "<|im_start|>user\n<tool_response>\n东京：晴\n</tool_response>",
        "\n<tool_response>\n大阪：雨\n</tool_response><|im_end|>"
    )));
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}
