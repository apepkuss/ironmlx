//! Real-tokenizer acceptance for native constrained tool decoding.

use std::path::PathBuf;

use ironmlx::core::constrained::{ToolChoiceConstraint, ToolConstraintOptions};
use ironmlx::core::tool_calling::{
    AssistantOutputEvent, ToolCallParser, ToolDefinition, ToolDialect,
};
use ironmlx::core::Tokenizer;

fn weather_tool() -> ToolDefinition {
    serde_json::from_value(serde_json::json!({
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "enum": [1, 2, 3]},
                "units": {"anyOf": [
                    {"type": "string", "const": "celsius"},
                    {"type": "null"}
                ]}
            },
            "required": ["city", "days"],
            "additionalProperties": false
        }
    }))
    .expect("tool definition")
}

#[test]
#[ignore = "requires IRONMLX_CONSTRAINT_MODEL_DIR pointing to a real Qwen3.5/3.6 model"]
fn real_qwen_tokenizer_compiles_and_enforces_tool_grammar() {
    let model_dir = PathBuf::from(
        std::env::var("IRONMLX_CONSTRAINT_MODEL_DIR")
            .expect("IRONMLX_CONSTRAINT_MODEL_DIR must be set"),
    );
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Qwen35));
    let plan = tokenizer
        .compile_tool_constraint(&[weather_tool()], &ToolConstraintOptions::default())
        .expect("compile constraint");

    let valid = "<tool_call>\n<function=get_weather>\n<parameter=days>2</parameter>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>";
    let valid_tokens = tokenizer.encode(valid, false).expect("encode valid call");
    let mut session = plan.start_session().expect("start matcher");
    session
        .commit_tokens(&valid_tokens)
        .expect("consume valid call");
    assert!(session.is_accepting().expect("accepting state"));

    let mut detok = tokenizer.decode_stream(false);
    let mut parser = ToolCallParser::new(ToolDialect::Qwen35, "real", &[weather_tool()])
        .expect("create Qwen parser");
    let mut events = Vec::new();
    for token in &valid_tokens {
        if let Some(text) = detok.step(*token).expect("decode native token") {
            events.extend(parser.push(&text).expect("parse native token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish Qwen parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert!(matches!(
        events.as_slice(),
        [AssistantOutputEvent::ToolCall(_)]
    ));

    let invalid = "<tool_call><function=unknown>";
    let invalid_tokens = tokenizer
        .encode(invalid, false)
        .expect("encode invalid call");
    let session = plan.start_session().expect("start matcher");
    assert!(
        session
            .validate_tokens(&invalid_tokens)
            .expect("validate invalid prefix")
            < invalid_tokens.len()
    );

    let required = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: false,
            },
        )
        .expect("compile required serial constraint");
    let plain = tokenizer
        .encode("ordinary answer", false)
        .expect("encode plain text");
    let mut required_session = required.start_session().expect("start required matcher");
    required_session
        .commit_tokens(&plain)
        .expect("plain thinking prefix remains legal");
    assert!(!required_session.is_accepting().expect("required state"));

    let forced = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Function("get_weather".into()),
                allow_parallel_calls: true,
            },
        )
        .expect("compile forced function constraint");
    let mut forced_session = forced.start_session().expect("start forced matcher");
    forced_session
        .commit_tokens(&valid_tokens)
        .expect("consume forced call");
    assert!(forced_session.is_accepting().expect("forced state"));
    assert!(
        forced_session
            .validate_tokens(&valid_tokens)
            .expect("reject second forced call")
            < valid_tokens.len()
    );
}

#[test]
#[ignore = "requires GEMMA4_MODEL pointing to a real Gemma 4 model"]
fn real_gemma_tokenizer_compiles_and_enforces_tool_grammar() {
    let model_dir = PathBuf::from(std::env::var("GEMMA4_MODEL").expect("GEMMA4_MODEL must be set"));
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Gemma));
    let plan = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: false,
            },
        )
        .expect("compile Gemma constraint");

    let valid = concat!(
        "<|tool_call>call:get_weather{days:2,city:<|\"|>Tokyo<|\"|>,",
        "units:<|\"|>celsius<|\"|>}<tool_call|>"
    );
    let valid_tokens = tokenizer.encode(valid, false).expect("encode valid call");
    let mut session = plan.start_session().expect("start matcher");
    session
        .commit_tokens(&valid_tokens)
        .expect("consume valid call");
    assert!(session.is_accepting().expect("accepting state"));

    let parallel_forced_plan = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Function("get_weather".into()),
                allow_parallel_calls: true,
            },
        )
        .expect("compile parallel forced Gemma constraint");
    let empty_string_prefix = tokenizer
        .encode("<|tool_call>call:get_weather{city:", false)
        .expect("encode empty-string prefix");
    let string_delimiter = tokenizer
        .encode("<|\"|>", false)
        .expect("encode Gemma string delimiter");
    assert_eq!(string_delimiter.len(), 1, "delimiter must be one token");
    let mut empty_string_session = parallel_forced_plan.start_session().expect("start matcher");
    for token in empty_string_prefix {
        assert!(
            empty_string_session
                .compute_mask()
                .expect("compute prefix mask")
                .is_allowed(token),
            "prefix token {token} must be allowed"
        );
        empty_string_session
            .commit_token(token)
            .expect("commit mask-allowed prefix token");
    }
    let mut empty_string_session = empty_string_session.fork();
    let closing_delimiter = string_delimiter[0];
    assert!(
        empty_string_session
            .compute_mask()
            .expect("compute opening-string mask")
            .is_allowed(closing_delimiter),
        "mask must permit the opening string delimiter"
    );
    empty_string_session
        .commit_token(closing_delimiter)
        .expect("commit mask-allowed opening string delimiter");
    assert!(
        empty_string_session
            .compute_mask()
            .expect("compute empty-string mask")
            .is_allowed(closing_delimiter),
        "mask must permit the closing string delimiter"
    );
    empty_string_session
        .commit_token(closing_delimiter)
        .expect("commit mask-allowed closing string delimiter");
    empty_string_session
        .commit_tokens(
            &tokenizer
                .encode(",days:2}", false)
                .expect("encode object suffix"),
        )
        .expect("consume empty-string suffix");
    let tool_call_close = tokenizer
        .encode("<tool_call|>", false)
        .expect("encode Gemma tool-call close");
    assert_eq!(
        tool_call_close.len(),
        1,
        "tool-call close must be one token"
    );
    let mut empty_string_session = empty_string_session.fork();
    assert!(
        empty_string_session
            .compute_mask()
            .expect("compute tool-call close mask")
            .is_allowed(tool_call_close[0]),
        "mask must permit the tool-call close token"
    );
    let tool_response = tokenizer
        .encode("<|tool_response>", false)
        .expect("encode Gemma tool-response open");
    assert_eq!(
        tool_response.len(),
        1,
        "tool-response open must be one token"
    );
    let mut speculative_resolution = vec![tool_call_close[0], tool_response[0]];
    empty_string_session
        .truncate_invalid_speculative_bonus(&mut speculative_resolution)
        .expect("truncate invalid target bonus after a complete tool call");
    assert_eq!(speculative_resolution, tool_call_close);
    empty_string_session
        .commit_token(tool_call_close[0])
        .expect("commit mask-allowed tool-call close token");
    assert!(empty_string_session
        .is_accepting()
        .expect("empty string accepting state"));

    let invalid = "<|tool_call>call:unknown{";
    let invalid_tokens = tokenizer
        .encode(invalid, false)
        .expect("encode invalid call");
    let session = plan.start_session().expect("start matcher");
    assert!(
        session
            .validate_tokens(&invalid_tokens)
            .expect("validate invalid prefix")
            < invalid_tokens.len()
    );
}

#[test]
#[ignore = "requires GLM47_MODEL pointing to a real GLM-4.7-Flash model"]
fn real_glm_tokenizer_compiles_and_enforces_tool_grammar() {
    let model_dir = PathBuf::from(std::env::var("GLM47_MODEL").expect("GLM47_MODEL must be set"));
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Glm));
    let plan = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: false,
            },
        )
        .expect("compile GLM constraint");

    let valid = concat!(
        "<tool_call>get_weather",
        "<arg_key>days</arg_key><arg_value>2</arg_value>",
        "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
        "<arg_key>units</arg_key><arg_value>celsius</arg_value>",
        "</tool_call>"
    );
    let valid_tokens = tokenizer.encode(valid, false).expect("encode valid call");
    let mut session = plan.start_session().expect("start matcher");
    for (index, token) in valid_tokens.iter().enumerate() {
        assert!(
            session
                .compute_mask()
                .expect("compute GLM mask")
                .is_allowed(*token),
            "GLM native token {token} at index {index} ({:?}) after prefix {:?} must be allowed",
            tokenizer.decode(&[*token], false).expect("decode token"),
            tokenizer
                .decode(&valid_tokens[..index], false)
                .expect("decode accepted prefix")
        );
        session.commit_token(*token).expect("commit GLM token");
    }
    assert!(session.is_accepting().expect("accepting state"));

    let mut detok = tokenizer.decode_stream(false);
    let mut parser = ToolCallParser::new(ToolDialect::Glm, "real", &[weather_tool()])
        .expect("create GLM parser");
    let mut events = Vec::new();
    for token in &valid_tokens {
        if let Some(text) = detok.step(*token).expect("decode GLM native token") {
            events.extend(parser.push(&text).expect("parse GLM native token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish GLM parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert!(matches!(
        events.as_slice(),
        [AssistantOutputEvent::ToolCall(_)]
    ));

    let parallel_plan = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: true,
            },
        )
        .expect("compile parallel GLM constraint");
    let adjacent_calls = concat!(
        "<tool_call>get_weather",
        "<arg_key>days</arg_key><arg_value>1</arg_value>",
        "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>",
        "</tool_call>",
        "<tool_call>get_weather",
        "<arg_key>days</arg_key><arg_value>2</arg_value>",
        "<arg_key>city</arg_key><arg_value>Osaka</arg_value>",
        "</tool_call>"
    );
    let adjacent_tokens = tokenizer
        .encode(adjacent_calls, false)
        .expect("encode adjacent GLM calls");
    let mut parallel_session = parallel_plan
        .start_session()
        .expect("start parallel matcher");
    parallel_session
        .commit_tokens(&adjacent_tokens)
        .expect("consume adjacent GLM calls");
    assert!(parallel_session
        .is_accepting()
        .expect("adjacent calls accepting state"));

    let mut detok = tokenizer.decode_stream(false);
    let mut parser = ToolCallParser::new(ToolDialect::Glm, "parallel-real", &[weather_tool()])
        .expect("create parallel GLM parser");
    let mut events = Vec::new();
    for token in &adjacent_tokens {
        if let Some(text) = detok.step(*token).expect("decode adjacent GLM token") {
            events.extend(parser.push(&text).expect("parse adjacent GLM token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish parallel GLM parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert_eq!(events.len(), 2);
    assert!(events
        .iter()
        .all(|event| matches!(event, AssistantOutputEvent::ToolCall(_))));

    let empty_string = concat!(
        "<tool_call>get_weather",
        "<arg_key>days</arg_key><arg_value>2</arg_value>",
        "<arg_key>city</arg_key><arg_value></arg_value>",
        "</tool_call>"
    );
    let empty_tokens = tokenizer
        .encode(empty_string, false)
        .expect("encode empty-string call");
    let mut empty_session = plan.start_session().expect("start empty matcher");
    empty_session
        .commit_tokens(&empty_tokens)
        .expect("consume empty-string call");
    assert!(empty_session.is_accepting().expect("empty accepting state"));

    let prefix = empty_string
        .strip_suffix("</tool_call>")
        .expect("known GLM close suffix");
    let prefix_tokens = tokenizer.encode(prefix, false).expect("encode GLM prefix");
    let close_tokens = tokenizer
        .encode("</tool_call>", false)
        .expect("encode GLM close");
    let observation_tokens = tokenizer
        .encode("<|observation|>", false)
        .expect("encode GLM observation token");
    let mut speculative = plan.start_session().expect("start speculative matcher");
    speculative
        .commit_tokens(&prefix_tokens)
        .expect("consume GLM prefix");
    let mut resolution = close_tokens.clone();
    resolution.extend(observation_tokens);
    speculative
        .truncate_invalid_speculative_bonus(&mut resolution)
        .expect("truncate GLM bonus after complete call");
    assert_eq!(resolution, close_tokens);

    let invalid = "<tool_call>unknown";
    let invalid_tokens = tokenizer
        .encode(invalid, false)
        .expect("encode invalid call");
    let session = plan.start_session().expect("start matcher");
    assert!(
        session
            .validate_tokens(&invalid_tokens)
            .expect("validate invalid prefix")
            < invalid_tokens.len()
    );
}

#[test]
#[ignore = "requires LLAMA32_MODEL pointing to a real Llama 3.1/3.2 Instruct model"]
fn real_llama_tokenizer_compiles_parses_and_enforces_single_json_call() {
    let model_dir =
        PathBuf::from(std::env::var("LLAMA32_MODEL").expect("LLAMA32_MODEL must be set"));
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::Llama));

    let required = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: true,
            },
        )
        .expect("compile Llama constraint");
    let valid =
        r#"{"name":"get_weather","parameters":{"city":"Tokyo","days":2,"units":"celsius"}}"#;
    let valid_tokens = tokenizer.encode(valid, false).expect("encode valid call");
    let mut session = required.start_session().expect("start matcher");
    session
        .commit_tokens(&valid_tokens)
        .expect("consume valid Llama call");
    assert!(session.is_accepting().expect("accepting state"));
    assert!(
        session
            .validate_tokens(&valid_tokens)
            .expect("reject second native call")
            < valid_tokens.len()
    );

    let mut detok = tokenizer.decode_stream(ToolDialect::MiniCpmV46.skip_special_tokens());
    let mut parser = ToolCallParser::new(ToolDialect::Llama, "real", &[weather_tool()])
        .expect("create Llama parser");
    let mut events = Vec::new();
    for token in &valid_tokens {
        if let Some(text) = detok.step(*token).expect("decode native token") {
            events.extend(parser.push(&text).expect("parse native token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish Llama parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert!(matches!(
        events.as_slice(),
        [AssistantOutputEvent::ToolCall(_)]
    ));

    let plain = tokenizer
        .encode("ordinary answer", false)
        .expect("encode plain text");
    let session = required.start_session().expect("start required matcher");
    assert_eq!(
        session
            .validate_tokens(&plain)
            .expect("reject required plain text"),
        0
    );

    let auto = tokenizer
        .compile_tool_constraint(&[weather_tool()], &ToolConstraintOptions::default())
        .expect("compile auto Llama constraint");
    let mut text = auto.start_session().expect("start auto matcher");
    text.commit_tokens(&plain).expect("consume ordinary text");
    assert!(text.is_accepting().expect("text accepting state"));

    let invalid = tokenizer
        .encode(
            r#"{"name":"get_weather","parameters":{"city":"Tokyo","days":4}}"#,
            false,
        )
        .expect("encode invalid enum call");
    let session = required.start_session().expect("start invalid matcher");
    assert!(
        session
            .validate_tokens(&invalid)
            .expect("validate invalid enum")
            < invalid.len()
    );
}

#[test]
#[ignore = "requires MINICPMV46_MODEL pointing to a real MiniCPM-V 4.6 model"]
fn real_minicpmv46_tokenizer_compiles_parses_and_enforces_native_xml() {
    let model_dir =
        PathBuf::from(std::env::var("MINICPMV46_MODEL").expect("MINICPMV46_MODEL must be set"));
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::MiniCpmV46));

    let required = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: true,
            },
        )
        .expect("compile MiniCPM-V constraint");
    let call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=days>\n2\n</parameter>\n",
        "<parameter=city>\nTokyo\n</parameter>\n",
        "</function>\n</tool_call>"
    );
    let two_calls = format!("{call}\n{call}");
    let tokens = tokenizer
        .encode(&two_calls, false)
        .expect("encode MiniCPM-V calls");
    let mut session = required.start_session().expect("start matcher");
    session
        .commit_tokens(&tokens)
        .expect("consume MiniCPM-V calls");
    assert!(session.is_accepting().expect("accepting state"));

    let mut detok = tokenizer.decode_stream(ToolDialect::MiniCpmV46.skip_special_tokens());
    let mut parser = ToolCallParser::new(ToolDialect::MiniCpmV46, "real-v46", &[weather_tool()])
        .expect("create MiniCPM-V parser");
    let mut events = Vec::new();
    for token in &tokens {
        if let Some(text) = detok.step(*token).expect("decode native token") {
            events.extend(parser.push(&text).expect("parse native token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish MiniCPM-V parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert_eq!(events.len(), 2);
    assert!(events
        .iter()
        .all(|event| matches!(event, AssistantOutputEvent::ToolCall(_))));
}

#[test]
#[ignore = "requires MINICPM5_MODEL pointing to a real MiniCPM5 model"]
fn real_minicpm5_tokenizer_compiles_parses_and_enforces_xml_cdata() {
    let model_dir =
        PathBuf::from(std::env::var("MINICPM5_MODEL").expect("MINICPM5_MODEL must be set"));
    let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("load tokenizer");
    assert_eq!(tokenizer.tool_dialect(), Some(ToolDialect::MiniCpm5));

    let required = tokenizer
        .compile_tool_constraint(
            &[weather_tool()],
            &ToolConstraintOptions {
                choice: ToolChoiceConstraint::Required,
                allow_parallel_calls: false,
            },
        )
        .expect("compile MiniCPM5 constraint");
    let valid = concat!(
        "<function name=\"get_weather\">",
        "<param name=\"city\"><![CDATA[Tokyo\n<&]]></param>",
        "<param name=\"days\">2</param>",
        "</function>"
    );
    let tokens = tokenizer
        .encode(valid, false)
        .expect("encode MiniCPM5 call");
    let mut session = required.start_session().expect("start matcher");
    session
        .commit_tokens(&tokens)
        .expect("consume MiniCPM5 call");
    assert!(session.is_accepting().expect("accepting state"));
    assert!(
        session
            .validate_tokens(&tokens)
            .expect("reject second serial call")
            < tokens.len()
    );

    let mut detok = tokenizer.decode_stream(ToolDialect::MiniCpm5.skip_special_tokens());
    let mut parser = ToolCallParser::new(ToolDialect::MiniCpm5, "real-mini5", &[weather_tool()])
        .expect("create MiniCPM5 parser");
    let mut events = Vec::new();
    for token in &tokens {
        if let Some(text) = detok.step(*token).expect("decode native token") {
            events.extend(parser.push(&text).expect("parse native token"));
        }
    }
    let (tail, saw_tool_call) = parser.finish().expect("finish MiniCPM5 parser");
    events.extend(tail);
    assert!(saw_tool_call);
    assert!(matches!(
        events.as_slice(),
        [AssistantOutputEvent::ToolCall(_)]
    ));

    let auto = tokenizer
        .compile_tool_constraint(&[weather_tool()], &ToolConstraintOptions::default())
        .expect("compile MiniCPM5 auto constraint");
    let plain = tokenizer
        .encode("ordinary answer", false)
        .expect("encode plain text");
    let mut text = auto.start_session().expect("start text matcher");
    text.commit_tokens(&plain).expect("consume plain text");
    assert!(text.is_accepting().expect("text accepting state"));
}
