//! Deterministic replay regressions using the real native decoder and wire adapters.
use super::*;
use ironmlx_lm::core::native_output::NativeOutputDialect;
use serde_json::{json, Value};

fn normalize(input: Value) -> anyhow::Result<NormalizedRequest> {
    serde_json::from_value::<ResponsesRequest>(json!({
        "model":"local", "input":input, "store":false,
    }))?
    .normalize()
}

fn reasoning(text: &str, status: Option<&str>) -> Value {
    let mut item = json!({"type":"reasoning", "id":"rs_test", "summary":[],
        "content":[{"type":"reasoning_text", "text":text}]});
    if let Some(status) = status {
        item["status"] = json!(status);
    }
    item
}

fn user(text: &str) -> Value {
    json!({"role":"user", "content":text})
}

fn meta() -> ResponseMeta {
    ResponseMeta::from_normalized(
        &normalize(json!([user("question")])).unwrap(),
        "local".into(),
    )
}

fn events(frames: Vec<Bytes>) -> Vec<Value> {
    frames
        .into_iter()
        .map(|frame| {
            let wire = String::from_utf8(frame.to_vec()).unwrap();
            serde_json::from_str(
                wire.lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn orphan_plaintext_is_dropped_without_losing_other_history() {
    for status in [
        None,
        Some("incomplete"),
        Some("completed"),
        Some("in_progress"),
    ] {
        for boundary in [
            user("continue"),
            json!({"role":"developer", "content":"instruction"}),
            json!({"type":"function_call_output", "call_id":"call_1", "output":"result"}),
        ] {
            let with_orphan = normalize(json!([
                user("question"),
                reasoning("unfinished", status),
                boundary.clone()
            ]))
            .unwrap();
            let without = normalize(json!([user("question"), boundary])).unwrap();
            assert_eq!(
                serde_json::to_value(with_orphan.chat.messages).unwrap(),
                serde_json::to_value(without.chat.messages).unwrap()
            );
        }
        let trailing =
            normalize(json!([user("question"), reasoning("unfinished", status)])).unwrap();
        assert_eq!(trailing.chat.messages.len(), 1);
        assert!(trailing.chat.messages[0].reasoning_content.is_none());
    }
    // Discarding an orphan must not make an otherwise empty request valid.
    assert!(normalize(json!([reasoning("unfinished", None)])).is_err());
}

#[test]
fn paired_reasoning_is_preserved_but_not_attached_across_turns() {
    let normalized = normalize(json!([
        user("first"), reasoning("orphan", None), user("second"),
        reasoning("also orphan", Some("incomplete")), reasoning("paired", None),
        {"role":"assistant", "content":"answer"}, user("third"),
        reasoning("tool plan", Some("completed")),
        {"type":"function_call", "id":"fc_1", "call_id":"call_1", "name":"weather", "arguments":"{}"},
        {"type":"function_call_output", "call_id":"call_1", "output":"sunny"}
    ])).unwrap();
    let messages = normalized.chat.messages;
    assert_eq!(messages.len(), 6);
    assert!(messages[1].reasoning_content.is_none());
    assert_eq!(messages[2].reasoning_content.as_deref(), Some("paired"));
    assert_eq!(messages[4].reasoning_content.as_deref(), Some("tool plan"));
    assert_eq!(messages[4].tool_calls[0].id, "call_1");
    assert_eq!(messages[5].tool_call_id.as_deref(), Some("call_1"));
    for content in [
        json!(""),
        json!([]),
        json!([{"type":"output_text", "text":""}]),
    ] {
        let empty = normalize(json!([user("question"), reasoning("orphan", None),
            {"role":"assistant", "content":content}, user("continue")]))
        .unwrap();
        assert!(empty
            .chat
            .messages
            .iter()
            .all(|message| message.reasoning_content.is_none()));
    }
}

#[test]
fn orphan_handling_does_not_bypass_validation_or_decrypt_opaque_history() {
    for content in [
        json!([]),
        json!([{"type":"reasoning_text", "text":""}]),
        json!([{"type":"reasoning_text", "text":"  "}]),
    ] {
        let opaque = json!({"type":"reasoning", "status":"incomplete", "encrypted_content":"opaque", "content":content});
        for tail in [
            vec![],
            vec![user("continue")],
            vec![
                reasoning("next", None),
                json!({"role":"assistant", "content":"answer"}),
            ],
        ] {
            let mut input = vec![user("question"), opaque.clone()];
            input.extend(tail);
            assert!(normalize(json!(input))
                .unwrap_err()
                .to_string()
                .contains("encrypted reasoning cannot be replayed"));
        }
    }
    for (field, value) in [("id", json!("")), ("status", json!("invalid"))] {
        let mut item = reasoning("orphan", None);
        item[field] = value;
        assert!(normalize(json!([user("question"), item, user("continue")])).is_err());
    }
}

#[tokio::test]
async fn reasoning_status_matches_native_channel_in_stream_and_unary_replay() {
    for (dialect, open, close) in [
        (NativeOutputDialect::Qwen35, "", "</think>"),
        (
            NativeOutputDialect::Gemma,
            "<|channel>thought\n",
            "<channel|>",
        ),
    ] {
        for (suffix, finish, expected) in [
            ("".to_string(), "length", "incomplete"),
            (close[..close.len() - 1].to_string(), "length", "incomplete"),
            (close.to_string(), "length", "completed"),
            (format!("{close}partial answer"), "length", "completed"),
            (format!("{close}answer"), "stop", "completed"),
        ] {
            let mut decoder = GeneratedOutputDecoder::from_decoded_with_native(
                None,
                Some(NativeOutputDecoderConfig {
                    dialect,
                    reasoning_enabled: true,
                }),
            )
            .unwrap();
            let mut output = CollectedOutput::new();
            let mut stream = ResponsesStream::new(meta());
            let mut frames = vec![stream.created()];
            let mut calls = Vec::new();
            let mut typed_finish = None;
            // Character-sized chunks exercise marker buffering and flush paths.
            for ch in format!("{open}inspect{suffix}").chars() {
                let decoded = decoder.push_text_delta(&ch.to_string()).unwrap();
                output.collect(decoded.clone()).unwrap();
                frames.extend(
                    stream_generated_events(&mut stream, decoded, &mut calls, &mut typed_finish)
                        .unwrap(),
                );
            }
            let tail = decoder.finish(finish).unwrap();
            output.collect(tail.clone()).unwrap();
            output.reasoning_incomplete = decoder.reasoning_incomplete();
            frames.extend(
                stream_generated_events(&mut stream, tail, &mut calls, &mut typed_finish).unwrap(),
            );
            frames.extend(stream.completed(
                typed_finish.unwrap(),
                Usage::new(3, 8),
                decoder.reasoning_incomplete(),
            ));
            let wire = events(frames);
            let added = wire
                .iter()
                .find(|event| event["type"] == "response.output_item.added")
                .unwrap();
            assert_eq!(added["item"]["status"], "in_progress");
            let done = wire
                .iter()
                .find(|event| {
                    event["type"] == "response.output_item.done"
                        && event["item"]["type"] == "reasoning"
                })
                .unwrap();
            assert_eq!(done["item"]["status"], expected, "{dialect:?}: {suffix}");
            let final_response = &wire.last().unwrap()["response"];
            assert_eq!(final_response["output"][0], done["item"]);
            assert_eq!(
                final_response["status"],
                if finish == "length" {
                    "incomplete"
                } else {
                    "completed"
                }
            );
            let response = unary_response(meta(), 3, output);
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let unary: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(unary["output"][0]["status"], expected);
            // Replay exactly the item pi-ai saves from output_item.done, then
            // also replay the complete final output (both response modes).
            for history in [
                json!([done["item"].clone()]),
                final_response["output"].clone(),
                unary["output"].clone(),
            ] {
                let mut input = vec![user("question")];
                input.extend(history.as_array().unwrap().iter().cloned());
                input.push(user("continue"));
                let replay = normalize(json!(input)).unwrap();
                assert!(replay
                    .chat
                    .messages
                    .last()
                    .unwrap()
                    .reasoning_content
                    .is_none());
                if suffix.is_empty() {
                    assert_eq!(replay.chat.messages.len(), 2);
                }
            }
        }
    }
}

#[test]
fn later_tool_call_does_not_retroactively_mark_reasoning_incomplete() {
    let mut stream = ResponsesStream::new(meta());
    let mut frames = stream.reasoning_delta("plan".into());
    frames.extend(
        stream
            .tool_call(ToolCall {
                id: "call_1".into(),
                name: "weather".into(),
                arguments: json!({}),
            })
            .unwrap(),
    );
    frames.extend(stream.completed("length", Usage::new(3, 8), false));
    let wire = events(frames);
    let done = wire
        .iter()
        .find(|event| {
            event["type"] == "response.output_item.done" && event["item"]["type"] == "reasoning"
        })
        .unwrap();
    assert_eq!(done["item"]["status"], "completed");
    assert_eq!(wire.last().unwrap()["response"]["output"][0], done["item"]);
}
