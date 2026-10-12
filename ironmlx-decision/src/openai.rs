//! OpenAI Decisions API (`POST /v1/decisions`) mapped onto Laya scoring.
//!
//! Requests are parsed from raw JSON so every rejection carries the offending
//! parameter path. Typed choice values and question names stay in this layer;
//! the model only ever sees the rendered option text.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::contract::{Question, Scores, ScoringRequest};
use crate::laya::{confidence, round4};

const MIN_CHOICES: usize = 2;
const MAX_CHOICES: usize = 255;
const MIN_LEVELS: usize = 2;
const MAX_LEVELS: usize = 10;
const MAX_SAFETY_IDENTIFIER_CHARS: usize = 128;
/// Separator between text parts when `input` is an array of messages.
pub const TEXT_PART_SEPARATOR: &str = "\n";

/// A client error with an OpenAI-style `code` and `param`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionInputError {
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
}

impl DecisionInputError {
    fn new(code: &'static str, param: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            param: Some(param.into()),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DecisionInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecisionInputError {}

type Parsed<T> = Result<T, DecisionInputError>;

/// Choice values are typed: `true` and `"true"` are distinct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    Bool(bool),
    String(String),
}

impl ChoiceValue {
    fn prompt_text(&self) -> &str {
        match self {
            Self::Bool(true) => "true",
            Self::Bool(false) => "false",
            Self::String(value) => value,
        }
    }
}

#[derive(Debug, Clone)]
enum AnswerKind {
    Predicate,
    Choice(Vec<ChoiceValue>),
    Score(Vec<String>),
}

#[derive(Debug, Clone)]
struct AnswerSpec {
    name: Option<String>,
    kind: AnswerKind,
}

/// A validated request: transport-neutral scoring input plus what is needed
/// to format answers in question order.
#[derive(Debug, Clone)]
pub struct PreparedDecision {
    pub model: String,
    pub scoring: ScoringRequest,
    answers: Vec<AnswerSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    pub model: String,
    pub answers: Vec<Answer>,
    pub usage: DecisionUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Predicate {
        name: Option<String>,
        probability: f64,
    },
    Choice {
        name: Option<String>,
        choice: ChoiceValue,
        probabilities: Vec<ChoiceProbability>,
        confidence: f64,
    },
    Score {
        name: Option<String>,
        score: f64,
        probabilities: Vec<ScoreProbability>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreProbability {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: usize,
}

/// Laya's CPU prompt-prefix cache skips tokenization only; every token is
/// still evaluated by the encoder, so no model tokens are reported as cached.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: usize,
}

impl DecisionUsage {
    fn from_input_tokens(input_tokens: usize) -> Self {
        Self {
            input_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: 0,
                cache_write_tokens: 0,
            },
            output_tokens: 0,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: input_tokens,
        }
    }
}

impl PreparedDecision {
    pub fn question_count(&self) -> usize {
        self.answers.len()
    }

    /// Format scores in question order. Confidence is Laya's normalized
    /// entropy; probabilities and scores are rounded to four decimals.
    pub fn response(&self, scores: &Scores) -> anyhow::Result<Decision> {
        anyhow::ensure!(
            scores.probabilities.len() == self.answers.len(),
            "decision answer count mismatch"
        );
        let answers = self
            .answers
            .iter()
            .zip(&scores.probabilities)
            .map(|(spec, probs)| spec.answer(probs))
            .collect::<anyhow::Result<_>>()?;
        Ok(Decision {
            model: self.model.clone(),
            answers,
            usage: DecisionUsage::from_input_tokens(scores.input_tokens),
        })
    }
}

impl AnswerSpec {
    fn answer(&self, probs: &[f32]) -> anyhow::Result<Answer> {
        let name = self.name.clone();
        Ok(match &self.kind {
            AnswerKind::Predicate => {
                anyhow::ensure!(probs.len() == 2, "predicate marker count mismatch");
                Answer::Predicate {
                    name,
                    probability: round4(probs[1]),
                }
            }
            AnswerKind::Choice(values) => {
                anyhow::ensure!(probs.len() == values.len(), "choice marker count mismatch");
                // The first option wins ties, matching System One.
                let mut best = 0;
                for (index, &prob) in probs.iter().enumerate() {
                    if prob > probs[best] {
                        best = index;
                    }
                }
                Answer::Choice {
                    name,
                    choice: values[best].clone(),
                    probabilities: values
                        .iter()
                        .zip(probs)
                        .map(|(value, &prob)| ChoiceProbability {
                            value: value.clone(),
                            probability: round4(prob),
                        })
                        .collect(),
                    confidence: round4(confidence(probs)),
                }
            }
            AnswerKind::Score(labels) => {
                anyhow::ensure!(probs.len() == labels.len(), "score marker count mismatch");
                let score = probs
                    .iter()
                    .enumerate()
                    .map(|(index, &prob)| index as f32 * prob)
                    .sum::<f32>();
                Answer::Score {
                    name,
                    score: round4(score),
                    probabilities: labels
                        .iter()
                        .zip(probs)
                        .enumerate()
                        .map(|(value, (label, &prob))| ScoreProbability {
                            value,
                            label: label.clone(),
                            probability: round4(prob),
                        })
                        .collect(),
                    confidence: round4(confidence(probs)),
                }
            }
        })
    }
}

/// Parse and validate a Decisions request body.
pub fn parse_request(body: &Value) -> Parsed<PreparedDecision> {
    let object = body.as_object().ok_or_else(|| DecisionInputError {
        code: "invalid_type",
        param: None,
        message: "The request body must be a JSON object.".into(),
    })?;
    reject_unknown(
        object,
        &["model", "input", "questions", "safety_identifier"],
        "",
    )?;
    let model = required_string(object, "model", "model")?;
    if model.is_empty() {
        return Err(DecisionInputError::new(
            "invalid_value",
            "model",
            "`model` must not be empty.",
        ));
    }
    match object.get("safety_identifier") {
        None | Some(Value::Null) => {}
        Some(Value::String(value)) if value.chars().count() <= MAX_SAFETY_IDENTIFIER_CHARS => {}
        Some(Value::String(_)) => {
            return Err(DecisionInputError::new(
                "invalid_value",
                "safety_identifier",
                "`safety_identifier` must be at most 128 characters.",
            ))
        }
        Some(_) => return Err(invalid_type("safety_identifier", "a string or null")),
    }
    let state = parse_input(
        object
            .get("input")
            .ok_or_else(|| missing("input", "input"))?,
    )?;
    let questions = match object.get("questions") {
        None => return Err(missing("questions", "questions")),
        Some(Value::Array(questions)) => questions,
        Some(_) => return Err(invalid_type("questions", "an array")),
    };
    if questions.is_empty() {
        return Err(DecisionInputError::new(
            "invalid_value",
            "questions",
            "`questions` must contain at least one question.",
        ));
    }
    let (questions, answers) = questions
        .iter()
        .enumerate()
        .map(|(index, question)| parse_question(question, &format!("questions[{index}]")))
        .collect::<Parsed<(Vec<_>, Vec<_>)>>()?;
    Ok(PreparedDecision {
        model: model.to_owned(),
        scoring: ScoringRequest {
            state: Value::String(state),
            questions,
        },
        answers,
    })
}

/// Normalize text evidence in order. Laya is text-only; every non-text
/// item is rejected rather than dropped.
fn parse_input(input: &Value) -> Parsed<String> {
    let messages = match input {
        Value::String(text) => return Ok(text.clone()),
        Value::Array(messages) => messages,
        _ => {
            return Err(invalid_type(
                "input",
                "a string or an array of user messages",
            ))
        }
    };
    if messages.is_empty() {
        return Err(DecisionInputError::new(
            "invalid_value",
            "input",
            "`input` must contain at least one message.",
        ));
    }
    let mut texts = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let path = format!("input[{index}]");
        let message = message
            .as_object()
            .ok_or_else(|| invalid_type(&path, "a message object"))?;
        match message.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "message" => {}
            Some(Value::String(kind)) => {
                return Err(unsupported(
                    format!("{path}.type"),
                    format!("Input items of type `{kind}` are not supported; only user messages are accepted."),
                ))
            }
            Some(_) => return Err(invalid_type(format!("{path}.type"), "a string")),
        }
        let role = required_string(message, "role", &format!("{path}.role"))?;
        if role != "user" {
            return Err(unsupported(
                format!("{path}.role"),
                format!("Messages with role `{role}` are not supported; only `user` is accepted."),
            ));
        }
        reject_unknown(message, &["type", "role", "content"], &path)?;
        match message
            .get("content")
            .filter(|content| !content.is_null())
            .ok_or_else(|| missing("content", format!("{path}.content")))?
        {
            Value::String(text) => texts.push(text.as_str()),
            Value::Array(parts) => {
                if parts.is_empty() {
                    return Err(DecisionInputError::new(
                        "invalid_value",
                        format!("{path}.content"),
                        "Message content must contain at least one part.",
                    ));
                }
                for (part_index, part) in parts.iter().enumerate() {
                    texts.push(parse_part(part, &format!("{path}.content[{part_index}]"))?);
                }
            }
            _ => {
                return Err(invalid_type(
                    format!("{path}.content"),
                    "a string or an array of content parts",
                ))
            }
        }
    }
    Ok(texts.join(TEXT_PART_SEPARATOR))
}

fn parse_part<'a>(part: &'a Value, path: &str) -> Parsed<&'a str> {
    let part = part
        .as_object()
        .ok_or_else(|| invalid_type(path, "a content part object"))?;
    let kind = required_string(part, "type", &format!("{path}.type"))?;
    match kind {
        "input_text" => {
            reject_unknown(part, &["type", "text"], path)?;
            required_string(part, "text", &format!("{path}.text"))
        }
        "input_image" => Err(unsupported(
            format!("{path}.type"),
            "`input_image` is not supported: aac6fef/laya-multilingual-mlx accepts text only.",
        )),
        other => Err(unsupported(
            format!("{path}.type"),
            format!(
                "Content parts of type `{other}` are not supported; only `input_text` is accepted."
            ),
        )),
    }
}

fn parse_question(question: &Value, path: &str) -> Parsed<(Question, AnswerSpec)> {
    let object = question
        .as_object()
        .ok_or_else(|| invalid_type(path, "a question object"))?;
    let kind = required_string(object, "type", &format!("{path}.type"))?;
    let allowed: &[&str] = match kind {
        "predicate" => &["type", "name", "instructions"],
        "choice" => &["type", "name", "instructions", "choices"],
        "score" => &["type", "name", "instructions", "levels"],
        other => {
            return Err(DecisionInputError::new(
                "invalid_value",
                format!("{path}.type"),
                format!(
                "Unsupported question type `{other}`; expected `predicate`, `choice`, or `score`."
            ),
            ))
        }
    };
    reject_unknown(object, allowed, path)?;
    let name = optional_string(object, "name", &format!("{path}.name"))?.map(str::to_owned);
    let instructions = Value::String(
        required_string(object, "instructions", &format!("{path}.instructions"))?.to_owned(),
    );
    Ok(match kind {
        "predicate" => (
            Question::Noul {
                instructions,
                criteria: None,
            },
            AnswerSpec {
                name,
                kind: AnswerKind::Predicate,
            },
        ),
        "choice" => {
            let (criteria, values) = parse_choices(object, path)?;
            (
                Question::Choice {
                    instructions,
                    criteria,
                },
                AnswerSpec {
                    name,
                    kind: AnswerKind::Choice(values),
                },
            )
        }
        _ => {
            let (criteria, labels) = parse_levels(object, path)?;
            (
                Question::Score {
                    instructions,
                    criteria,
                },
                AnswerSpec {
                    name,
                    kind: AnswerKind::Score(labels),
                },
            )
        }
    })
}

type ChoiceCriteria = (Vec<(String, Value)>, Vec<ChoiceValue>);

fn parse_choices(question: &Map<String, Value>, path: &str) -> Parsed<ChoiceCriteria> {
    let param = format!("{path}.choices");
    let choices = match question.get("choices") {
        None => return Err(missing("choices", param)),
        Some(Value::Array(choices)) => choices,
        Some(_) => return Err(invalid_type(param, "an array")),
    };
    if !(MIN_CHOICES..=MAX_CHOICES).contains(&choices.len()) {
        return Err(DecisionInputError::new(
            "invalid_value",
            param,
            "`choices` must contain 2 to 255 options.",
        ));
    }
    let mut criteria = Vec::with_capacity(choices.len());
    let mut values: Vec<ChoiceValue> = Vec::with_capacity(choices.len());
    for (index, choice) in choices.iter().enumerate() {
        let path = format!("{param}[{index}]");
        let choice = choice
            .as_object()
            .ok_or_else(|| invalid_type(&path, "a choice object"))?;
        reject_unknown(choice, &["value", "description"], &path)?;
        let value = match choice.get("value") {
            None => return Err(missing("value", format!("{path}.value"))),
            Some(Value::String(value)) => ChoiceValue::String(value.clone()),
            Some(Value::Bool(value)) => ChoiceValue::Bool(*value),
            Some(_) => return Err(invalid_type(format!("{path}.value"), "a string or boolean")),
        };
        if values.contains(&value) {
            return Err(DecisionInputError::new(
                "invalid_value",
                format!("{path}.value"),
                "Choice values must be unique within a question.",
            ));
        }
        let description = optional_string(choice, "description", &format!("{path}.description"))?;
        let text = option_text(value.prompt_text(), description);
        if text.is_empty() {
            return Err(DecisionInputError::new(
                "invalid_value",
                format!("{path}.description"),
                "An empty-string choice value needs a nonempty description.",
            ));
        }
        // The rendered text is the prompt label; the typed value never reaches the model.
        criteria.push((text, Value::Null));
        values.push(value);
    }
    Ok((criteria, values))
}

fn parse_levels(question: &Map<String, Value>, path: &str) -> Parsed<(Vec<Value>, Vec<String>)> {
    let param = format!("{path}.levels");
    let levels = match question.get("levels") {
        None => return Err(missing("levels", param)),
        Some(Value::Array(levels)) => levels,
        Some(_) => return Err(invalid_type(param, "an array")),
    };
    if !(MIN_LEVELS..=MAX_LEVELS).contains(&levels.len()) {
        return Err(DecisionInputError::new(
            "invalid_value",
            param,
            "`levels` must contain 2 to 10 levels.",
        ));
    }
    let mut criteria = Vec::with_capacity(levels.len());
    let mut labels = Vec::with_capacity(levels.len());
    for (index, level) in levels.iter().enumerate() {
        let path = format!("{param}[{index}]");
        let level = level
            .as_object()
            .ok_or_else(|| invalid_type(&path, "a level object"))?;
        reject_unknown(level, &["label", "description"], &path)?;
        let label = required_string(level, "label", &format!("{path}.label"))?;
        let description = optional_string(level, "description", &format!("{path}.description"))?;
        criteria.push(Value::String(option_text(label, description)));
        labels.push(label.to_owned());
    }
    Ok((criteria, labels))
}

/// Render `label: description` the way System One renders its criteria.
fn option_text(label: &str, description: Option<&str>) -> String {
    match description.filter(|description| !description.is_empty()) {
        None => label.to_owned(),
        Some(description) if label.is_empty() => description.to_owned(),
        Some(description) => format!("{label}: {description}"),
    }
}

fn reject_unknown(object: &Map<String, Value>, allowed: &[&str], path: &str) -> Parsed<()> {
    match object.keys().find(|key| !allowed.contains(&key.as_str())) {
        None => Ok(()),
        Some(key) => {
            let param = if path.is_empty() {
                key.clone()
            } else {
                format!("{path}.{key}")
            };
            Err(DecisionInputError::new(
                "unknown_parameter",
                param.clone(),
                format!("Unrecognized request argument supplied: `{param}`."),
            ))
        }
    }
}

fn required_string<'a>(object: &'a Map<String, Value>, key: &str, param: &str) -> Parsed<&'a str> {
    match object.get(key) {
        None | Some(Value::Null) => Err(missing(key, param)),
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(invalid_type(param, "a string")),
    }
}

/// Optional strings treat an explicit `null` as omitted.
fn optional_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    param: &str,
) -> Parsed<Option<&'a str>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(invalid_type(param, "a string")),
    }
}

fn missing(key: &str, param: impl Into<String>) -> DecisionInputError {
    DecisionInputError::new(
        "missing_required_parameter",
        param,
        format!("Missing required parameter: `{key}`."),
    )
}

fn invalid_type(param: impl Into<String>, expected: &str) -> DecisionInputError {
    let param = param.into();
    DecisionInputError::new(
        "invalid_type",
        param.clone(),
        format!("Invalid type for `{param}`: expected {expected}."),
    )
}

fn unsupported(param: impl Into<String>, message: impl Into<String>) -> DecisionInputError {
    DecisionInputError::new("unsupported_value", param, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laya::system_one_response;
    use serde_json::json;

    const MODEL: &str = crate::contract::MULTILINGUAL_MODEL_ID;

    fn parse(body: Value) -> Parsed<PreparedDecision> {
        parse_request(&body)
    }

    fn error(body: Value) -> DecisionInputError {
        parse(body).expect_err("request must be rejected")
    }

    fn base(questions: Value) -> Value {
        json!({"model": MODEL, "input": "charged twice", "questions": questions})
    }

    #[test]
    fn mixed_questions_keep_order_names_and_rendered_prompts() {
        let prepared = parse(base(json!([
            {"type": "score", "name": "urgency", "instructions": "How urgent?",
             "levels": [{"label": "low", "description": "can wait"}, {"label": "high"}]},
            {"type": "predicate", "instructions": "Refund?"},
            {"type": "choice", "name": "urgency", "instructions": "Team?",
             "choices": [{"value": "billing", "description": "refunds"}, {"value": true},
                         {"value": "true"}, {"value": "", "description": "none of these"}]}
        ])))
        .unwrap();
        assert_eq!(prepared.scoring.state, json!("charged twice"));
        assert_eq!(prepared.question_count(), 3);
        match &prepared.scoring.questions[0] {
            Question::Score { criteria, .. } => {
                assert_eq!(criteria, &vec![json!("low: can wait"), json!("high")])
            }
            other => panic!("expected score, got {other:?}"),
        }
        assert!(matches!(
            prepared.scoring.questions[1],
            Question::Noul { criteria: None, .. }
        ));
        match &prepared.scoring.questions[2] {
            Question::Choice { criteria, .. } => assert_eq!(
                criteria.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
                ["billing: refunds", "true", "true", "none of these"]
            ),
            other => panic!("expected choice, got {other:?}"),
        }

        let scores = Scores {
            probabilities: vec![vec![0.25, 0.75], vec![0.2, 0.8], vec![0.1, 0.2, 0.6, 0.1]],
            input_tokens: 77,
        };
        let response = serde_json::to_value(prepared.response(&scores).unwrap()).unwrap();
        assert_eq!(
            response,
            json!({
                "model": MODEL,
                "answers": [
                    {"type": "score", "name": "urgency", "score": 0.75, "confidence": 0.1887,
                     "probabilities": [
                        {"value": 0, "label": "low", "probability": 0.25},
                        {"value": 1, "label": "high", "probability": 0.75}]},
                    {"type": "predicate", "name": null, "probability": 0.8},
                    {"type": "choice", "name": "urgency", "choice": "true", "confidence": 0.2145,
                     "probabilities": [
                        {"value": "billing", "probability": 0.1},
                        {"value": true, "probability": 0.2},
                        {"value": "true", "probability": 0.6},
                        {"value": "", "probability": 0.1}]}
                ],
                "usage": {
                    "input_tokens": 77,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 77
                }
            })
        );
    }

    #[test]
    fn boolean_choice_is_returned_as_boolean_and_ties_pick_first() {
        let prepared = parse(base(json!([{"type": "choice", "instructions": "?",
            "choices": [{"value": false}, {"value": "false"}]}])))
        .unwrap();
        let response = prepared
            .response(&Scores {
                probabilities: vec![vec![0.5, 0.5]],
                input_tokens: 1,
            })
            .unwrap();
        match &response.answers[0] {
            Answer::Choice { choice, .. } => assert_eq!(choice, &ChoiceValue::Bool(false)),
            other => panic!("expected choice, got {other:?}"),
        }
    }

    #[test]
    fn equivalent_system_one_request_renders_identical_questions() {
        let decisions = parse(json!({"model": MODEL, "input": "发票被重复扣款，请退款。", "questions": [
            {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
             "choices": [{"value": "billing", "description": "refunds"},
                         {"value": "technical", "description": "bugs"}]},
            {"type": "predicate", "name": "refund", "instructions": "Is a refund requested?"},
            {"type": "score", "name": "urgency", "instructions": "How urgent is this?",
             "levels": [{"label": "can wait"}, {"label": "soon"}, {"label": "today"}]}
        ]}))
        .unwrap();
        let system_one: crate::DecisionRequest = serde_json::from_value(json!({
            "model": MODEL, "state": "发票被重复扣款，请退款。", "questions": {
                "department": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "refunds", "technical": "bugs"}},
                "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                "urgency": {"type": "score", "instructions": "How urgent is this?",
                            "criteria": ["can wait", "soon", "today"]}}
        }))
        .unwrap();
        let named = system_one.validate().unwrap();
        assert_eq!(decisions.scoring.state, system_one.state);
        for ((_, expected), actual) in named.iter().zip(&decisions.scoring.questions) {
            assert_eq!(
                crate::laya::render_for_test(expected),
                crate::laya::render_for_test(actual)
            );
        }
        let scores = Scores {
            probabilities: vec![vec![0.9, 0.1], vec![0.3, 0.7], vec![0.2, 0.3, 0.5]],
            input_tokens: 128,
        };
        let s1 = serde_json::to_value(system_one_response(&named, &scores).unwrap()).unwrap();
        let oa = serde_json::to_value(decisions.response(&scores).unwrap()).unwrap();
        assert_eq!(
            oa["answers"][0]["choice"],
            s1["answers"]["department"]["choice"]
        );
        assert_eq!(
            oa["answers"][0]["confidence"],
            s1["answers"]["department"]["confidence"]
        );
        assert_eq!(
            oa["answers"][1]["probability"],
            s1["answers"]["refund"]["noul"]
        );
        assert_eq!(oa["answers"][2]["score"], s1["answers"]["urgency"]["score"]);
        assert_eq!(oa["usage"]["input_tokens"], s1["usage"]["input_tokens"]);
    }

    #[test]
    fn choice_count_is_bounded_to_two_through_255() {
        let question = |count: usize| {
            let choices = (0..count)
                .map(|i| json!({"value": format!("option-{i}")}))
                .collect::<Vec<_>>();
            base(json!([{"type": "predicate", "instructions": "?"},
                {"type": "choice", "instructions": "?", "choices": choices}]))
        };
        for count in [0, 1, 256] {
            let failure = error(question(count));
            assert_eq!(failure.code, "invalid_value", "{count}");
            assert_eq!(failure.param.as_deref(), Some("questions[1].choices"));
            assert_eq!(failure.message, "`choices` must contain 2 to 255 options.");
        }
        for count in [2, 255] {
            match &parse(question(count)).unwrap().scoring.questions[1] {
                Question::Choice { criteria, .. } => assert_eq!(criteria.len(), count),
                other => panic!("expected choice, got {other:?}"),
            }
        }
        // System One keeps its own 1 to 255 range.
        let single: crate::DecisionRequest = serde_json::from_value(json!({
            "model": MODEL, "state": "x", "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": {"only": null}}}
        }))
        .unwrap();
        assert!(single.validate().is_ok());
    }

    #[test]
    fn text_messages_are_joined_in_order() {
        let prepared = parse(json!({"model": MODEL, "questions": [
        {"type": "predicate", "instructions": "?"}],
        "input": [
            {"role": "user", "content": "first"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_text", "text": "third"}]}
        ]}))
        .unwrap();
        assert_eq!(prepared.scoring.state, json!("first\nsecond\nthird"));
    }

    #[test]
    fn optional_fields_accept_null_and_safety_identifier_is_bounded() {
        let mut body = base(json!([{"type": "choice", "name": null, "instructions": "?",
            "choices": [{"value": "a", "description": null}, {"value": "b"}]}]));
        body["safety_identifier"] = json!("user-123");
        assert!(parse(body.clone()).is_ok());
        body["safety_identifier"] = Value::Null;
        assert!(parse(body.clone()).is_ok());
        body["safety_identifier"] = json!("x".repeat(129));
        assert_eq!(error(body).param.as_deref(), Some("safety_identifier"));
    }

    #[test]
    fn unsupported_inputs_are_rejected_not_dropped() {
        let q = json!([{"type": "predicate", "instructions": "?"}]);
        for (input, param) in [
            (
                json!([{"role": "user", "content": [{"type": "input_text", "text": "a"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]),
                "input[0].content[1].type",
            ),
            (
                json!([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]),
                "input[0].content[0].type",
            ),
            (
                json!([{"role": "user", "content": [{"type": "input_audio"}]}]),
                "input[0].content[0].type",
            ),
            (
                json!([{"type": "function_call", "name": "f", "arguments": "{}"}]),
                "input[0].type",
            ),
            (
                json!([{"type": "item_reference", "id": "x"}]),
                "input[0].type",
            ),
            (
                json!([{"role": "assistant", "content": "hi"}]),
                "input[0].role",
            ),
            (
                json!([{"role": "system", "content": "hi", "id": "msg_1"}]),
                "input[0].role",
            ),
        ] {
            let failure = error(json!({"model": MODEL, "input": input, "questions": q}));
            assert_eq!(failure.code, "unsupported_value", "{failure:?}");
            assert_eq!(failure.param.as_deref(), Some(param));
        }
    }

    #[test]
    fn malformed_requests_name_the_parameter() {
        let cases = [
            (
                json!({"input": "x", "questions": []}),
                "missing_required_parameter",
                "model",
            ),
            (
                json!({"model": MODEL, "questions": []}),
                "missing_required_parameter",
                "input",
            ),
            (
                json!({"model": MODEL, "input": "x"}),
                "missing_required_parameter",
                "questions",
            ),
            (base(json!([])), "invalid_value", "questions"),
            (
                json!({"model": MODEL, "input": [], "questions": [{"type":"predicate","instructions":"?"}]}),
                "invalid_value",
                "input",
            ),
            (
                json!({"model": MODEL, "input": 3, "questions": []}),
                "invalid_type",
                "input",
            ),
            (
                json!({"model": MODEL, "input": [{"role": "user", "content": null}], "questions": []}),
                "missing_required_parameter",
                "input[0].content",
            ),
            (
                json!({"model": MODEL, "input": "x", "questions": [], "store": true}),
                "unknown_parameter",
                "store",
            ),
            (
                base(json!([{"type": "noul", "instructions": "?"}])),
                "invalid_value",
                "questions[0].type",
            ),
            (
                base(json!([{"type": "predicate"}])),
                "missing_required_parameter",
                "questions[0].instructions",
            ),
            (
                base(json!([{"type": "predicate", "instructions": {"a": 1}}])),
                "invalid_type",
                "questions[0].instructions",
            ),
            (
                base(json!([{"type": "predicate", "instructions": "?", "choices": []}])),
                "unknown_parameter",
                "questions[0].choices",
            ),
            (
                base(json!([{"type": "predicate", "instructions": "?", "name": 1}])),
                "invalid_type",
                "questions[0].name",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?"}])),
                "missing_required_parameter",
                "questions[0].choices",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?", "choices": []}])),
                "invalid_value",
                "questions[0].choices",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?",
                "choices": [{"value": 1}, {"value": "b"}]}])),
                "invalid_type",
                "questions[0].choices[0].value",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?",
                "choices": [{"value": "a"}, {"value": "a"}]}])),
                "invalid_value",
                "questions[0].choices[1].value",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?",
                "choices": [{"value": true}, {"value": true}]}])),
                "invalid_value",
                "questions[0].choices[1].value",
            ),
            (
                base(json!([{"type": "choice", "instructions": "?",
                "choices": [{"value": ""}, {"value": "b"}]}])),
                "invalid_value",
                "questions[0].choices[0].description",
            ),
            (
                base(json!([{"type": "score", "instructions": "?", "levels": [{"label": "a"}]}])),
                "invalid_value",
                "questions[0].levels",
            ),
            (
                base(json!([{"type": "score", "instructions": "?",
                "levels": [{"label": "a"}, {"description": "b"}]}])),
                "missing_required_parameter",
                "questions[0].levels[1].label",
            ),
        ];
        for (body, code, param) in cases {
            let failure = error(body.clone());
            assert_eq!(
                (failure.code, failure.param.as_deref()),
                (code, Some(param)),
                "{body}"
            );
        }
        let many = (0..256)
            .map(|i| json!({"value": i.to_string()}))
            .collect::<Vec<_>>();
        let failure = error(base(
            json!([{"type": "choice", "instructions": "?", "choices": many}]),
        ));
        assert_eq!(failure.param.as_deref(), Some("questions[0].choices"));
    }
}
