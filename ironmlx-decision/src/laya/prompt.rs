use std::{cell::RefCell, collections::VecDeque};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::contract::Question;

pub(super) struct PromptTokenizer {
    backend: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    mask_text: String,
    pub pad: u32,
    cache: RefCell<VecDeque<(String, PreparedQuestion)>>,
}

#[derive(Clone)]
pub(super) struct PreparedQuestion {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

impl PromptTokenizer {
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let tokenizer_dir = dir.join("tokenizer");
        let backend = Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading Laya tokenizer: {e}"))?;
        let config: Value = serde_json::from_reader(std::fs::File::open(
            tokenizer_dir.join("tokenizer_config.json"),
        )?)?;
        let token = |name: &str| -> Result<(String, u32)> {
            let value = config.get(name).ok_or_else(|| anyhow!("missing {name}"))?;
            let text = value
                .as_str()
                .or_else(|| value.get("content").and_then(Value::as_str))
                .ok_or_else(|| anyhow!("invalid {name}"))?;
            let id = backend
                .token_to_id(text)
                .ok_or_else(|| anyhow!("tokenizer lacks {name}"))?;
            Ok((text.to_string(), id))
        };
        let (_, cls) = token("cls_token")?;
        let (_, sep) = token("sep_token")?;
        let (mask_text, mask) = token("mask_token")?;
        let (_, pad) = token("pad_token")?;
        Ok(Self {
            backend,
            cls,
            sep,
            mask,
            mask_text,
            pad,
            cache: RefCell::new(VecDeque::new()),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .backend
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing Laya input: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn prepare_prefix(
        &self,
        question: &Question,
        max_len: usize,
        head_max_len: usize,
        cache_prompts: bool,
    ) -> Result<PreparedQuestion> {
        // Rendered text preserves criterion order and contains no request state.
        let key = format!(
            "{max_len}:{head_max_len}:{}:{}:{}",
            question.kind(),
            python_json(question.instructions(), true),
            serde_json::to_string(&render_options(question))?
        );
        if cache_prompts {
            let mut cache = self.cache.borrow_mut();
            if let Some(index) = cache.iter().position(|(candidate, _)| candidate == &key) {
                let entry = cache.remove(index).expect("known cache index");
                let prepared = entry.1.clone();
                cache.push_back(entry);
                return Ok(prepared);
            }
        }
        let instructions = match question.instructions() {
            Value::String(value) => value.clone(),
            value => python_json(value, true),
        };
        let instructions = instructions.replace(&self.mask_text, " ");
        let mut head = self.encode(&format!("{} question: {}", question.kind(), instructions))?;
        let options = render_options(question);
        let mut option_ids = Vec::with_capacity(options.len());
        for option in &options {
            let mut encoded = vec![self.mask];
            encoded.extend(
                self.encode(&format!(" {}", option.replace(&self.mask_text, " ")))?
                    .into_iter()
                    .take(48),
            );
            option_ids.push(encoded);
        }
        let mut option_budget =
            head_max_len.saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        if option_budget < 16 {
            let per = ((head_max_len.saturating_sub(16)) / option_ids.len().max(1)).max(4);
            for ids in &mut option_ids {
                ids.truncate(per);
            }
            option_budget =
                head_max_len.saturating_sub(option_ids.iter().map(Vec::len).sum::<usize>());
        }
        head.truncate(option_budget.max(8));
        let mut ids = vec![self.cls];
        ids.extend(head);
        ids.push(self.sep);
        let mut markers = Vec::with_capacity(option_ids.len());
        for option in option_ids {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(self.sep);
        if ids.len() + 1 > max_len || markers.iter().any(|&m| m >= max_len) {
            bail!("question options exceed the Laya token budget");
        }
        let prepared = PreparedQuestion { ids, markers };
        if cache_prompts {
            let mut cache = self.cache.borrow_mut();
            if cache.len() == 128 {
                cache.pop_front();
            }
            cache.push_back((key, prepared.clone()));
        }
        Ok(prepared)
    }

    pub fn encode_state(&self, state: &Value) -> Result<Vec<u32>> {
        let text = match state {
            Value::String(s) => s.clone(),
            value => python_json(value, false),
        }
        .replace(&self.mask_text, " ");
        self.encode(&text)
    }

    pub fn append_state(
        &self,
        mut prefix: PreparedQuestion,
        state: &[u32],
        max_len: usize,
    ) -> PreparedQuestion {
        let room = max_len - prefix.ids.len() - 1;
        prefix.ids.extend(state.iter().take(room));
        prefix.ids.push(self.sep);
        prefix
    }
}

fn render_options(question: &Question) -> Vec<String> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(label, description)| {
                if description.is_null() || description.as_str() == Some("") {
                    label.clone()
                } else {
                    format!("{label}: {}", render_criterion(description))
                }
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, value)| format!("level {i}: {}", render_criterion(value)))
            .collect(),
        Question::Noul { criteria, .. } => {
            let get = |name: &str, fallback: &str| {
                criteria
                    .as_ref()
                    .and_then(|c| c.get(name))
                    .filter(|v| !v.is_null() && v.as_str() != Some(""))
                    .map(render_criterion)
                    .unwrap_or_else(|| fallback.to_string())
            };
            vec![
                format!("false: {}", get("false", "no, the statement does not hold")),
                format!("true: {}", get("true", "yes, the statement holds")),
            ]
        }
    }
}

pub(super) fn render_criterion(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_json(value, false))
}

/// Python `json.dumps` formatting used by the upstream prompt path.
fn python_json(value: &Value, ascii: bool) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            let mut encoded = serde_json::to_string(s).expect("JSON strings serialize");
            if ascii && !s.is_ascii() {
                encoded.clear();
                encoded.push('"');
                for ch in s.chars() {
                    if ch.is_ascii() {
                        match ch {
                            '"' => encoded.push_str("\\\""),
                            '\\' => encoded.push_str("\\\\"),
                            '\n' => encoded.push_str("\\n"),
                            '\r' => encoded.push_str("\\r"),
                            '\t' => encoded.push_str("\\t"),
                            c if c.is_control() => {
                                encoded.push_str(&format!("\\u{:04x}", c as u32))
                            }
                            c => encoded.push(c),
                        }
                    } else {
                        let mut utf16 = [0u16; 2];
                        for code in ch.encode_utf16(&mut utf16) {
                            encoded.push_str(&format!("\\u{code:04x}"));
                        }
                    }
                }
                encoded.push('"');
            }
            encoded
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|v| python_json(v, ascii))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        python_json(&Value::String(k.clone()), ascii),
                        python_json(v, ascii)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_cache_is_opt_in_bounded_and_excludes_state() {
        let Some(path) = std::env::var_os("LAYA_MODEL_DIR") else {
            return;
        };
        let tokenizer = PromptTokenizer::load(std::path::Path::new(&path)).unwrap();
        let question = |i| {
            serde_json::from_value::<Question>(serde_json::json!({
                "type":"noul", "instructions":format!("Question {i}")
            }))
            .unwrap()
        };
        tokenizer
            .prepare_prefix(&question(0), 1024, 256, false)
            .unwrap();
        assert!(tokenizer.cache.borrow().is_empty());
        let prefix = tokenizer
            .prepare_prefix(&question(0), 1024, 256, true)
            .unwrap();
        let a = tokenizer.append_state(prefix.clone(), &[10, 11], 1024);
        let b = tokenizer.append_state(
            tokenizer
                .prepare_prefix(&question(0), 1024, 256, true)
                .unwrap(),
            &[12],
            1024,
        );
        assert_ne!(a.ids, b.ids);
        assert_eq!(tokenizer.cache.borrow().len(), 1);
        assert_eq!(tokenizer.cache.borrow()[0].1.ids, prefix.ids);
        for i in 1..130 {
            tokenizer
                .prepare_prefix(&question(i), 1024, 256, true)
                .unwrap();
        }
        assert_eq!(tokenizer.cache.borrow().len(), 128);
        assert!(!tokenizer
            .cache
            .borrow()
            .iter()
            .any(|(_, value)| value.ids == prefix.ids));
    }

    #[test]
    fn python_json_keeps_order_and_spacing() {
        let value: Value = serde_json::from_str(r#"{"message":"退款","urgent":true}"#).unwrap();
        assert_eq!(
            python_json(&value, false),
            "{\"message\": \"退款\", \"urgent\": true}"
        );
        assert_eq!(
            python_json(&value, true),
            "{\"message\": \"\\u9000\\u6b3e\", \"urgent\": true}"
        );
    }
}
