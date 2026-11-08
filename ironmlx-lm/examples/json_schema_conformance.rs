//! Assess the bounded public schema subset against a pinned official test tree.
//! Rejected schemas are reported as unsupported, never counted as passing tests.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ironmlx_lm::core::constrained::{
    normalize_json_schema, schema_uses_raw_string, validate_schema_value, validate_tool_schemas,
    ConstraintPlan, ConstraintTokenizer, ToolChoiceConstraint, ToolConstraintOptions,
};
use ironmlx_lm::core::tool_calling::{ToolCallParser, ToolDefinition, ToolDialect};
use ironmlx_lm::core::GeneratedOutputEvent;
use serde_json::{json, Map, Value};

fn inputs(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            inputs(&path, files)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            files.push(path);
        }
    }
    Ok(())
}

// Construct the same complete byte vocabulary as the internal grammar tests,
// through the production tokenizer entry point, without loading any MLX model.
fn tokenizer() -> Result<ConstraintTokenizer> {
    let vocab: Map<String, Value> = (0..256_u32)
        .map(|byte| (format!("<0x{byte:02X}>"), json!(byte)))
        .collect();
    ConstraintTokenizer::from_tokenizer_json(
        &json!({"model":{"vocab":vocab},
            "decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"}]},
            "added_tokens":[{"id":256,"content":"<eos>","special":true}]}),
        &[256],
        &[],
    )
}

fn grammar_accepts(plan: &ConstraintPlan, text: &str) -> Result<bool> {
    let mut session = plan.start_session()?;
    for byte in text.bytes() {
        let token = u32::from(byte);
        if !session.compute_mask()?.is_allowed(token) {
            return Ok(false);
        }
        session.commit_token(token)?;
    }
    session.is_accepting()
}

fn caught<T>(action: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .unwrap_or_else(|_| anyhow::bail!("implementation panicked"))
}

fn outcome(result: Result<bool>) -> Value {
    match result {
        Ok(accepted) => json!({"accepted":accepted}),
        Err(error) => json!({"error":format!("{error:#}")}),
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: json_schema_conformance SUITE_DRAFT_DIRECTORY REPORT_JSON [REQUEST_OR_SCHEMA_JSON ...]"
    );
    std::panic::set_hook(Box::new(|_| {}));
    let root = Path::new(&args[1]);
    let mut files = Vec::new();
    inputs(root, &mut files)?;
    files.sort();
    let tokenizer = tokenizer()?;
    let options = ToolConstraintOptions {
        choice: ToolChoiceConstraint::Required,
        allow_parallel_calls: false,
    };
    let mut records = Vec::new();
    let mut load_errors = Vec::new();
    for file in files {
        let relative = file.strip_prefix(root)?.to_string_lossy();
        let groups: Vec<Value> = match serde_json::from_slice(&std::fs::read(&file)?) {
            Ok(groups) => groups,
            Err(error) => {
                load_errors.push(json!({"file":relative,"error":error.to_string()}));
                continue;
            }
        };
        for (index, group) in groups.iter().enumerate() {
            let tests = group["tests"].as_array().context("group.tests")?;
            let mut record = json!({"file":relative,"group_index":index,
                "description":group["description"],"instances":tests.len()});
            let normalized = match caught(|| normalize_json_schema(&group["schema"])) {
                Ok(schema) => schema,
                Err(error) => {
                    record["status"] = json!("unsupported");
                    record["error"] = json!(format!("{error:#}"));
                    records.push(record);
                    continue;
                }
            };
            record["status"] = json!("accepted");
            let mut nested = normalized.clone();
            if let Some(object) = nested.as_object_mut() {
                object
                    .entry("$id")
                    .or_insert_with(|| json!("https://ironmlx.invalid/assessed-schema"));
            }
            let wrapped = json!({"type":"object","properties":{"value":nested},
                "required":["value"],"additionalProperties":false});
            let tool = ToolDefinition {
                name: "probe".into(),
                description: None,
                parameters: wrapped.clone(),
                strict: None,
            };
            let tools = std::slice::from_ref(&tool);
            let json_plan = caught(|| tokenizer.compile_json_output(&wrapped));
            let qwen_plan = caught(|| {
                validate_tool_schemas(tools)?;
                tokenizer.compile_qwen_tools(tools, &options)
            });
            if let Err(error) = &json_plan {
                record["json_compile_error"] = json!(format!("{error:#}"));
            }
            if let Err(error) = &qwen_plan {
                record["qwen_compile_error"] = json!(format!("{error:#}"));
            }
            let mut results = Vec::new();
            for (test_index, test) in tests.iter().enumerate() {
                let data = &test["data"];
                let expected = test["valid"].as_bool().context("test.valid")?;
                let mut result = json!({"test_index":test_index,"description":test["description"],"expected":expected,
                    "validator":outcome(caught(|| Ok(validate_schema_value(&group["schema"],data).is_ok())))});
                let wrapped_text = serde_json::to_string(&json!({"value":data}))?;
                if wrapped_text.len() <= 32_768 {
                    if let Ok(plan) = &json_plan {
                        result["json_grammar"] =
                            outcome(caught(|| grammar_accepts(plan, &wrapped_text)));
                    }
                    if let Ok(plan) = &qwen_plan {
                        let string_only = schema_uses_raw_string(&normalized);
                        if string_only && !data.is_string() {
                            result["qwen_skip"] = json!("A native raw-string parameter cannot represent this non-string source instance; serializing it as text would test a different value.");
                        } else {
                            let argument = if string_only {
                                data.as_str().expect("string instance").to_owned()
                            } else {
                                serde_json::to_string(data)?
                            };
                            let text = format!("<tool_call><function=probe><parameter=value>{argument}</parameter></function></tool_call>");
                            result["qwen_grammar"] =
                                outcome(caught(|| grammar_accepts(plan, &text)));
                            let parsed = caught(|| {
                                let mut parser = ToolCallParser::new(
                                    ToolDialect::Qwen35,
                                    "official-suite",
                                    tools,
                                )?;
                                let mut events = parser.push(&text)?;
                                events.extend(parser.finish()?.0);
                                events
                                    .into_iter()
                                    .find_map(|event| match event {
                                        GeneratedOutputEvent::ToolCall(call) => {
                                            Some(call.arguments["value"].clone())
                                        }
                                        _ => None,
                                    })
                                    .context("parser emitted no tool call")
                            });
                            result["qwen_parser"] = json!({"accepted":parsed.is_ok()});
                            if let Ok(actual) = parsed {
                                result["qwen_roundtrip"] =
                                    json!({"matches_instance":actual==*data,"actual":actual});
                            }
                            result["qwen_comparison_eligible"] = json!(result
                                .get("qwen_roundtrip")
                                .is_none_or(|roundtrip| roundtrip["matches_instance"] == true));
                        }
                    }
                } else {
                    result["grammar_skip"] =
                        json!("serialized instance exceeds 32768-byte probe limit");
                }
                let mismatch = ["validator", "json_grammar", "qwen_grammar", "qwen_parser"]
                    .iter()
                    .any(|key| {
                        result.get(*key).is_some_and(|value| {
                            value.get("error").is_some()
                                || value["accepted"].as_bool() != Some(expected)
                        })
                    });
                if mismatch {
                    result["schema"] = group["schema"].clone();
                    result["data"] = data.clone();
                }
                results.push(result);
            }
            record["tests"] = json!(results);
            records.push(record);
        }
    }
    let mut inventories = Vec::new();
    for source in &args[3..] {
        let value: Value = serde_json::from_slice(&std::fs::read(source)?)?;
        let definitions: Vec<ToolDefinition> =
            if let Some(tools) = value.get("tools").and_then(Value::as_array) {
                tools
                    .iter()
                    .map(|tool| serde_json::from_value(tool["function"].clone()))
                    .collect::<std::result::Result<_, _>>()?
            } else {
                vec![ToolDefinition {
                    name: "aft_callgraph".into(),
                    description: None,
                    parameters: value,
                    strict: None,
                }]
            };
        for tool in definitions {
            let tools = std::slice::from_ref(&tool);
            let validation = caught(|| validate_tool_schemas(tools));
            let json_compile = caught(|| tokenizer.compile_json_output(&tool.parameters));
            let qwen_compile = caught(|| tokenizer.compile_qwen_tools(tools, &options));
            inventories.push(json!({"source":source,"name":tool.name,
                "validation":validation.as_ref().map(|_|"accepted").map_err(|error|format!("{error:#}")),
                "json_compile":json_compile.as_ref().map(|_|"compiled").map_err(|error|format!("{error:#}")),
                "qwen_compile":qwen_compile.as_ref().map(|_|"compiled").map_err(|error|format!("{error:#}")),
                "boundary":"Schema admission and grammar compilation only; no model/plugin tool execution."}));
        }
    }
    let report = json!({"suite_directory":root,"load_errors":load_errors,"groups":records,"inventory":inventories,
        "method":"Unmodified official schemas/instances. Unsupported schemas are distinct from passing cases. Accepted schemas are wrapped as one required tool argument for JSON/Qwen token-mask probes; root dialect declarations are removed only from that nested wrapper. Byte vocabulary; no model execution. Qwen string-only schemas use native raw strings; non-string instances under those schemas are not representable and are explicitly excluded from native probes. Other values use JSON serialization. Native parser round-trip values are recorded. Grammar results test the representative serialization, not every semantically equivalent JSON spelling.",
        "limits":"Core and optional files reported separately by path; no remote schemas are fetched; output probes capped at 32768 bytes per serialized instance. JSON/Qwen compile failures and runtime errors are recorded separately. This is an assessment, not a success gate."});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    println!("Assessment written to {}", args[2]);
    Ok(())
}
