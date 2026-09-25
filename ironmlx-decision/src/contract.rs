//! In-process contract shared by native inference and the later System One transport.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const MULTILINGUAL_MODEL_ID: &str = "aac6fef/laya-multilingual-mlx";

#[derive(Debug, Clone, Deserialize)]
pub struct DecisionRequest {
    pub model: String,
    pub state: Value,
    pub questions: Map<String, Value>,
}

impl DecisionRequest {
    pub fn validate(&self) -> Result<Vec<(String, Question)>> {
        if self.model != MULTILINGUAL_MODEL_ID {
            bail!("unsupported decision model {:?}", self.model);
        }
        if !matches!(
            self.state,
            Value::String(_) | Value::Object(_) | Value::Array(_)
        ) {
            bail!("state must be a string, object, or array");
        }
        if self.questions.is_empty() {
            bail!("questions must not be empty");
        }
        self.questions
            .iter()
            .map(|(name, value)| {
                if name.is_empty() {
                    bail!("question names must not be empty");
                }
                let question: Question = serde_json::from_value(value.clone())?;
                question.validate()?;
                Ok((name.clone(), question))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice {
        instructions: Value,
        criteria: Map<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: Option<Map<String, Value>>,
    },
}

impl Question {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
            Self::Noul { .. } => "noul",
        }
    }

    pub fn kind_index(&self) -> u32 {
        match self {
            Self::Choice { .. } => 0,
            Self::Score { .. } => 1,
            Self::Noul { .. } => 2,
        }
    }

    pub fn instructions(&self) -> &Value {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_entry(self.instructions(), "instructions", false)?;
        match self {
            Self::Choice { criteria, .. } => {
                if criteria.is_empty() || criteria.len() > 255 {
                    bail!("choice criteria must have 1 to 255 options");
                }
                for (label, value) in criteria {
                    if label.is_empty() {
                        bail!("choice labels must not be empty");
                    }
                    validate_entry(value, "choice criterion", true)?;
                }
            }
            Self::Score { criteria, .. } => {
                if !(2..=10).contains(&criteria.len()) {
                    bail!("score criteria must have 2 to 10 levels");
                }
                for value in criteria {
                    validate_entry(value, "score criterion", false)?;
                }
            }
            Self::Noul { criteria, .. } => {
                if let Some(criteria) = criteria {
                    for (key, value) in criteria {
                        if key != "true" && key != "false" {
                            bail!("noul criteria accepts only true and false");
                        }
                        validate_entry(value, "noul criterion", true)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_entry(value: &Value, field: &str, allow_null: bool) -> Result<()> {
    if matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
        || (allow_null && value.is_null())
    {
        Ok(())
    } else {
        bail!(
            "{field} must be a string, object, array{}",
            if allow_null { ", or null" } else { "" }
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_questions_preserve_names_and_choice_order() {
        let request: DecisionRequest = serde_json::from_str(
            r#"{"model":"aac6fef/laya-multilingual-mlx","state":"hello","questions":{"z":{"type":"choice","instructions":"choose","criteria":{"billing":null,"sales":null}},"a":{"type":"noul","instructions":"yes?"}}}"#,
        )
        .unwrap();
        let parsed = request.validate().unwrap();
        assert_eq!(
            parsed.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["z", "a"]
        );
        match &parsed[0].1 {
            Question::Choice { criteria, .. } => {
                assert_eq!(
                    criteria.keys().map(String::as_str).collect::<Vec<_>>(),
                    ["billing", "sales"]
                );
            }
            _ => panic!("expected choice"),
        }
    }

    #[test]
    fn instructions_are_required_by_the_system_one_contract() {
        let request: DecisionRequest = serde_json::from_str(
            r#"{"model":"aac6fef/laya-multilingual-mlx","state":"hello","questions":{"q":{"type":"noul"}}}"#,
        ).unwrap();
        assert!(request.validate().is_err());
    }
}
