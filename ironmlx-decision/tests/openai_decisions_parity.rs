//! Real-checkpoint comparison of `/v1/decisions` and System One on equivalent input.
//! Run with LAYA_MODEL_DIR and LAYA_METALLIB set to the checkpoint and metallib paths.

use std::path::Path;

use ironmlx_decision::{openai::parse_request, DecisionRequest, Laya};
use serde_json::{json, Value};

const MODEL: &str = ironmlx_decision::contract::MULTILINGUAL_MODEL_ID;

fn load() -> Option<Laya> {
    let model_dir = std::env::var_os("LAYA_MODEL_DIR")?;
    if let Some(metallib) = std::env::var_os("LAYA_METALLIB") {
        mlx::metal::set_metallib_path(metallib.to_str().expect("UTF-8 metallib path"))
            .expect("load metallib");
    }
    Some(Laya::load(Path::new(&model_dir)).expect("load Laya"))
}

/// Both APIs must produce identical numbers; `names` maps each Decisions
/// answer, in order, to its System One key.
fn assert_equivalent(model: &Laya, system_one: Value, decisions: Value, names: &[&str]) -> Value {
    let request: DecisionRequest = serde_json::from_value(system_one).unwrap();
    let expected = serde_json::to_value(model.predict(&request).unwrap()).unwrap();
    let prepared = parse_request(&decisions).unwrap();
    let actual = serde_json::to_value(
        prepared
            .response(&model.score(&prepared.scoring).unwrap())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual["model"], MODEL);
    assert_eq!(
        actual["usage"]["input_tokens"],
        expected["usage"]["input_tokens"]
    );
    assert_eq!(
        actual["usage"]["total_tokens"],
        expected["usage"]["input_tokens"]
    );
    let answers = actual["answers"].as_array().unwrap();
    assert_eq!(answers.len(), names.len());
    for (answer, name) in answers.iter().zip(names) {
        let reference = &expected["answers"][*name];
        match answer["type"].as_str().unwrap() {
            "predicate" => {
                assert_eq!(reference["type"], "noul");
                assert_eq!(answer["probability"], reference["noul"], "{name}");
            }
            "choice" => {
                assert_eq!(answer["confidence"], reference["confidence"], "{name}");
                let probabilities = reference["probabilities"].as_object().unwrap();
                let listed = answer["probabilities"].as_array().unwrap();
                assert_eq!(listed.len(), probabilities.len());
                for (entry, (_, probability)) in listed.iter().zip(probabilities) {
                    assert_eq!(&entry["probability"], probability, "{name}");
                }
                let best = listed
                    .iter()
                    .position(|entry| entry["value"] == answer["choice"])
                    .unwrap();
                assert_eq!(
                    probabilities.keys().nth(best).unwrap(),
                    reference["choice"].as_str().unwrap()
                );
            }
            "score" => {
                assert_eq!(answer["score"], reference["score"], "{name}");
                assert_eq!(answer["confidence"], reference["confidence"], "{name}");
                for (index, entry) in answer["probabilities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    assert_eq!(entry["value"], index);
                    assert_eq!(
                        entry["probability"],
                        reference["probabilities"][index.to_string()],
                        "{name}"
                    );
                }
            }
            other => panic!("unexpected answer type {other}"),
        }
    }
    actual
}

#[test]
fn decisions_match_system_one_on_equivalent_inputs() {
    let Some(model) = load() else {
        eprintln!("LAYA_MODEL_DIR not set; skipping real-checkpoint parity");
        return;
    };

    // Fixture 0: System One renders the object state as JSON text.
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/laya-reference/0.request.json")).unwrap();
    let actual = assert_equivalent(
        &model,
        fixture,
        json!({"model": MODEL, "input": "{\"message\": \"发票被重复扣款，请退款。\"}", "questions": [
            {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
             "choices": [{"value": "billing", "description": "refunds"},
                         {"value": "technical", "description": "bugs"}]},
            {"type": "predicate", "name": "refund", "instructions": "Is a refund requested?"},
            {"type": "score", "name": "urgency", "instructions": "How urgent is this?",
             "levels": [{"label": "can wait"}, {"label": "soon"}, {"label": "today"}]}
        ]}),
        &["department", "refund", "urgency"],
    );
    assert_eq!(actual["usage"]["input_tokens"], 128);
    assert_eq!(actual["answers"][0]["choice"], "billing");
    assert_eq!(actual["answers"][2]["probabilities"][2]["label"], "today");

    // Empty input and right-truncated long context keep Laya's budget rules.
    assert_equivalent(
        &model,
        json!({"model": MODEL, "state": "", "questions": {
            "empty": {"type": "noul", "instructions": "Is there any content?"}}}),
        json!({"model": MODEL, "input": "", "questions": [
            {"type": "predicate", "name": "empty", "instructions": "Is there any content?"}]}),
        &["empty"],
    );
    let long = "Long context. ".repeat(600);
    let actual = assert_equivalent(
        &model,
        json!({"model": MODEL, "state": long, "questions": {"long": {"type": "choice",
            "instructions": "Which class?",
            "criteria": {"first": "Class one", "second": "Class two", "third": "Class three"}}}}),
        json!({"model": MODEL, "input": long, "questions": [{"type": "choice", "name": "long",
            "instructions": "Which class?", "choices": [
                {"value": "first", "description": "Class one"},
                {"value": "second", "description": "Class two"},
                {"value": "third", "description": "Class three"}]}]}),
        &["long"],
    );
    assert_eq!(actual["usage"]["input_tokens"], 1024);

    // Message parts join with newlines; mixed order, unnamed and repeated names.
    assert_equivalent(
        &model,
        json!({"model": MODEL, "state": "El pago falló dos veces.\nNecesito ayuda hoy.", "questions": {
            "q0": {"type": "score", "instructions": "How urgent is this?",
                   "criteria": ["later: not today", "soon", "today: within hours"]},
            "q1": {"type": "noul", "instructions": "Is the customer complaining?"},
            "q2": {"type": "choice", "instructions": "Choose route",
                   "criteria": {"support": "technical issue", "billing": "payment issue", "sales": null}},
            "q3": {"type": "noul", "instructions": "Is a payment involved?"}}}),
        json!({"model": MODEL, "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "El pago falló dos veces."},
                {"type": "input_text", "text": "Necesito ayuda hoy."}]}],
            "questions": [
                {"type": "score", "name": "same", "instructions": "How urgent is this?",
                 "levels": [{"label": "later", "description": "not today"}, {"label": "soon"},
                            {"label": "today", "description": "within hours"}]},
                {"type": "predicate", "instructions": "Is the customer complaining?"},
                {"type": "choice", "name": "same", "instructions": "Choose route", "choices": [
                    {"value": "support", "description": "technical issue"},
                    {"value": "billing", "description": "payment issue"},
                    {"value": "sales"}]},
                {"type": "predicate", "instructions": "Is a payment involved?"}]}),
        &["q0", "q1", "q2", "q3"],
    );

    // Boolean and string values with distinct descriptions are separate options.
    let actual = assert_equivalent(
        &model,
        json!({"model": MODEL, "state": "The customer confirmed the order.", "questions": {
            "q": {"type": "choice", "instructions": "Did the customer confirm?",
                  "criteria": {"true: confirmed": null, "true: the literal word true": null,
                               "false: not confirmed": null}}}}),
        json!({"model": MODEL, "input": "The customer confirmed the order.", "questions": [
            {"type": "choice", "instructions": "Did the customer confirm?", "choices": [
                {"value": true, "description": "confirmed"},
                {"value": "true", "description": "the literal word true"},
                {"value": false, "description": "not confirmed"}]}]}),
        &["q"],
    );
    let values = actual["answers"][0]["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["value"].clone())
        .collect::<Vec<_>>();
    assert_eq!(values, [json!(true), json!("true"), json!(false)]);

    // Options that cannot fit are rejected with a typed budget error, never dropped.
    let choices = (0..255)
        .map(|index| json!({"value": format!("option-{index}"), "description": "description"}))
        .collect::<Vec<_>>();
    let prepared = parse_request(&json!({"model": MODEL, "input": "hello", "questions": [
        {"type": "choice", "instructions": "Choose one", "choices": choices}]}))
    .unwrap();
    let error = model.score(&prepared.scoring).unwrap_err();
    assert!(
        error.is::<ironmlx_decision::TokenBudgetExceeded>(),
        "{error:#}"
    );
}
