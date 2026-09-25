mod model;
mod prompt;

use std::path::Path;

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

use crate::contract::{DecisionRequest, DecisionResponse, Question, Usage, MULTILINGUAL_MODEL_ID};
use crate::DecisionSettings;
use model::NativeInference;
use prompt::{render_criterion, PromptTokenizer};

pub struct Laya {
    model: NativeInference,
    tokenizer: PromptTokenizer,
    settings: DecisionSettings,
}

impl Laya {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_settings(dir, DecisionSettings::default())
    }

    pub fn load_with_settings(dir: &Path, settings: DecisionSettings) -> Result<Self> {
        settings.validate()?;
        let mlx_config: Value =
            serde_json::from_reader(std::fs::File::open(dir.join("mlx_config.json"))?)?;
        if mlx_config.get("format").and_then(Value::as_str) != Some("laya-mlx")
            || mlx_config.get("format_version").and_then(Value::as_u64) != Some(1)
            || mlx_config.get("repository").and_then(Value::as_str) != Some(MULTILINGUAL_MODEL_ID)
        {
            bail!("not the supported Laya multilingual MLX checkpoint");
        }
        mlx::set_default_device(settings.device.resolve()?);
        Ok(Self {
            model: NativeInference::load(dir, settings.dtype.mlx(), settings.compile)?,
            settings,
            tokenizer: PromptTokenizer::load(dir)?,
        })
    }

    pub fn predict(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        mlx::set_default_device(self.settings.device.resolve()?);
        let questions = request.validate()?;
        let mut answers = Map::new();
        let mut input_tokens = 0;
        let state = self.tokenizer.encode_state(&request.state)?;
        for chunk in questions.chunks(self.settings.batch_size) {
            let prepared = chunk
                .iter()
                .map(|(_, question)| {
                    let prefix = self.tokenizer.prepare_prefix(
                        question,
                        self.model.max_len,
                        self.model.head_max_len,
                        self.settings.cache_prompts,
                    )?;
                    Ok(self
                        .tokenizer
                        .append_state(prefix, &state, self.model.max_len))
                })
                .collect::<Result<Vec<_>>>()?;
            input_tokens += prepared.iter().map(|p| p.ids.len()).sum::<usize>();
            let kinds = chunk
                .iter()
                .map(|(_, q)| q.kind_index())
                .collect::<Vec<_>>();
            let logits = self.model.logits_batch(
                &prepared,
                &kinds,
                self.tokenizer.pad,
                self.settings.pad_to_multiple,
            )?;
            for ((name, question), logits) in chunk.iter().zip(logits) {
                let temperature = self.temperature(question, logits.len());
                let probs = softmax(&logits.iter().map(|&v| v / temperature).collect::<Vec<_>>());
                answers.insert(name.clone(), answer(question, &probs)?);
            }
        }
        Ok(DecisionResponse {
            model: MULTILINGUAL_MODEL_ID.into(),
            answers,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }

    fn temperature(&self, question: &Question, count: usize) -> f32 {
        let bucket = if count <= 2 {
            "2"
        } else if count <= 5 {
            "3-5"
        } else if count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let key = format!("{}:{bucket}", question.kind());
        self.model
            .temperature_by_options
            .get(&key)
            .copied()
            .unwrap_or(self.model.temperature[question.kind_index() as usize])
            .clamp(0.5, 5.0)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<_> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn round4(x: f32) -> f64 {
    ((x as f64) * 10_000.0).round() / 10_000.0
}

fn confidence(probs: &[f32]) -> f32 {
    if probs.len() < 2 {
        return 1.0;
    }
    let entropy: f32 = probs.iter().map(|&p| -p * p.max(1e-12).ln()).sum();
    (1.0 - entropy / (probs.len() as f32).ln()).clamp(0.0, 1.0)
}

fn answer(question: &Question, probs: &[f32]) -> Result<Value> {
    match question {
        Question::Choice { criteria, .. } => {
            if criteria.len() != probs.len() {
                bail!("choice marker count mismatch");
            }
            let mut probabilities = Map::new();
            let (mut best, mut best_p) = ("", f32::NEG_INFINITY);
            for ((name, _), &prob) in criteria.iter().zip(probs) {
                probabilities.insert(name.clone(), json!(round4(prob)));
                if prob > best_p {
                    best = name;
                    best_p = prob;
                }
            }
            Ok(
                json!({"type":"choice", "choice":best, "confidence":round4(confidence(probs)), "probabilities":probabilities}),
            )
        }
        Question::Score { criteria, .. } => {
            if criteria.len() != probs.len() {
                bail!("score marker count mismatch");
            }
            let mut legend = Map::new();
            let mut probabilities = Map::new();
            let mut score = 0.0;
            for (i, (criterion, &prob)) in criteria.iter().zip(probs).enumerate() {
                legend.insert(i.to_string(), Value::String(render_criterion(criterion)));
                probabilities.insert(i.to_string(), json!(round4(prob)));
                score += i as f32 * prob;
            }
            Ok(
                json!({"type":"score", "score":round4(score), "confidence":round4(confidence(probs)), "legend":legend, "probabilities":probabilities}),
            )
        }
        Question::Noul { .. } => {
            if probs.len() != 2 {
                bail!("noul marker count mismatch");
            }
            Ok(json!({"type":"noul", "noul":round4(probs[1])}))
        }
    }
}
