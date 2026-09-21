use super::{japanese::Japanese, normalize::Normalizer, IndexTts25Tokenizer};
use crate::{error::invalid, AudioError, Language, Result, SessionControl};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use std::{path::Path, sync::LazyLock};
static ANNOTATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<([^|>\n]+)\|([^>\n]+)>").expect("annotation regex"));
static PROTECTED: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(r"<\|SPECIAL_TOKEN_(\d+)\|>.*?<\|SPECIAL_TOKEN_\1\|>").expect("protected regex")
});
static SPECIAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|([^|]+)\|>").expect("special regex"));
fn han(c: char) -> bool {
    matches!(c as u32,0x3400..=0x4dbf|0x4e00..=0x9fff)
}
fn kana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30ff)
}
impl Language {
    pub fn code(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Zh => "zh",
            Self::En => "en",
            Self::Ja => "ja",
            Self::Es => "es",
            Self::Ar => "ar",
        }
    }
    pub fn model_id(self) -> Result<u32> {
        match self {
            Self::En => Ok(0),
            Self::Zh => Ok(1),
            Self::Es => Ok(3),
            Self::Ja => Ok(7),
            Self::Ar => Ok(13),
            Self::Auto => Err(invalid("language", "auto must be resolved")),
        }
    }
}
/// Select the pinned script heuristic; plain Latin text reports ambiguity.
pub fn resolve_language(text: &str, requested: Language) -> Result<(Language, bool)> {
    if text.trim().is_empty() {
        return Err(invalid("text", "empty input"));
    }
    if requested != Language::Auto {
        return Ok((requested, false));
    }
    let arabic = text
        .chars()
        .any(|c| matches!(c as u32,0x600..=0x6ff|0x750..=0x77f|0x8a0..=0x8ff));
    let japanese = text.chars().any(kana);
    let chinese = text.chars().any(han);
    if arabic && (japanese || chinese) {
        return Err(invalid(
            "language",
            "mixed Arabic/CJK needs an explicit language",
        ));
    }
    if arabic {
        Ok((Language::Ar, false))
    } else if japanese {
        Ok((Language::Ja, false))
    } else if chinese {
        Ok((Language::Zh, false))
    } else if text.contains([
        'á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡', 'Á', 'É', 'Í', 'Ó', 'Ú', 'Ü', 'Ñ',
    ]) {
        Ok((Language::Es, false))
    } else if text.chars().any(|c| c.is_ascii_alphabetic()) {
        Ok((Language::En, true))
    } else {
        Err(invalid(
            "language",
            "cannot identify script; choose an explicit language",
        ))
    }
}
#[derive(Debug)]
pub struct PreparedText {
    pub language: Language,
    pub language_id: u32,
    pub language_ambiguous: bool,
    pub normalized_text: String,
    pub segments: Vec<String>,
    /// Prefix plus segment plus the frontend EOS, before GPT canonicalization.
    pub token_ids: Vec<Vec<u32>>,
    /// GPT removes existing start/stop IDs, then adds exactly one BOS/EOS pair.
    pub canonical_token_ids: Vec<Vec<u32>>,
}
#[derive(Clone, Debug)]
pub struct TextLimits {
    pub max_input_bytes: usize,
    pub max_normalized_bytes: usize,
    pub max_total_tokens: usize,
    pub max_segments: usize,
    pub segment_tokens: usize,
    pub position_capacity: usize,
}
impl Default for TextLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 65536,
            max_normalized_bytes: 262144,
            max_total_tokens: 16384,
            max_segments: 256,
            segment_tokens: 120,
            position_capacity: 602,
        }
    }
}
pub struct IndexTts25TextFrontend {
    tokenizer: IndexTts25Tokenizer,
    normalizer: Normalizer,
    japanese: Japanese,
}
impl IndexTts25TextFrontend {
    /// Read verified local resources; no optional host packages or silent fallback.
    pub fn load(vocab: &Path, fst_root: &Path, unidic_root: &Path) -> Result<Self> {
        Ok(Self {
            tokenizer: IndexTts25Tokenizer::from_file(vocab)?,
            normalizer: Normalizer::load(fst_root)?,
            japanese: Japanese::load(unidic_root)?,
        })
    }
    pub fn prepare(
        &mut self,
        text: &str,
        language: Language,
        limits: &TextLimits,
        control: &dyn SessionControl,
    ) -> Result<PreparedText> {
        control.check()?;
        if text.len() > limits.max_input_bytes {
            return Err(AudioError::CapacityExceeded {
                resource: "text input bytes",
            });
        }
        if text.contains('\0') {
            return Err(invalid("text", "NUL is not supported"));
        }
        if limits.segment_tokens == 0 || !(4..=602).contains(&limits.position_capacity) {
            return Err(invalid("text_limits", "invalid segment/position capacity"));
        }
        let (language, ambiguous) = resolve_language(text, language)?;
        let prepared = self.normalizer.normalize(text, language, control)?;
        let prepared = match language {
            Language::Zh | Language::En | Language::Ja => prepared.to_lowercase(),
            Language::Es => prepared.to_uppercase(),
            _ => prepared,
        };
        let prepared = ANNOTATION
            .replace_all(&prepared, |cap: &regex::Captures<'_>| {
                let pronunciation = cap[2].to_uppercase();
                if !pronunciation.is_empty() && pronunciation.chars().all(kana) {
                    format!(" {pronunciation} ")
                } else {
                    let marker = if cap[1].chars().any(han) { 2 } else { 1 };
                    format!("<|SPECIAL_TOKEN_{marker}|>{pronunciation}<|SPECIAL_TOKEN_{marker}|>")
                }
            })
            .into_owned();
        let prepared = if language == Language::Ja {
            self.japanese.process(&prepared, control)?
        } else {
            prepared
        };
        let prepared = SPECIAL
            .replace_all(&prepared, |cap: &regex::Captures<'_>| {
                format!("<|{}|>", cap[1].to_uppercase())
            })
            .into_owned();
        if prepared.trim().is_empty() {
            return Err(invalid("text", "normalization produced empty text"));
        }
        if prepared.len() > limits.max_normalized_bytes {
            return Err(AudioError::CapacityExceeded {
                resource: "normalized text bytes",
            });
        }
        let prefix = format!("<|{}|> ", language.code());
        let budget = limits
            .segment_tokens
            .min(limits.position_capacity - 2)
            .saturating_sub(self.tokenizer.token_count(&prefix))
            .max(1);
        let segments = split_text(&prepared, &self.tokenizer, budget, control)?;
        if segments.len() > limits.max_segments {
            return Err(AudioError::CapacityExceeded {
                resource: "text segments",
            });
        }
        let mut token_ids = Vec::new();
        let mut canonical_token_ids = Vec::new();
        let mut total = 0;
        for segment in &segments {
            control.check()?;
            let mut raw = self.tokenizer.encode(&(prefix.clone() + segment));
            raw.push(1);
            let canonical: Vec<u32> = std::iter::once(0)
                .chain(raw.iter().copied().filter(|t| *t != 0 && *t != 1))
                .chain(std::iter::once(1))
                .collect();
            if canonical.len() > limits.position_capacity {
                return Err(AudioError::CapacityExceeded {
                    resource: "text positions",
                });
            }
            total += canonical.len();
            if total > limits.max_total_tokens {
                return Err(AudioError::CapacityExceeded {
                    resource: "text tokens",
                });
            }
            token_ids.push(raw);
            canonical_token_ids.push(canonical);
        }
        Ok(PreparedText {
            language,
            language_id: language.model_id()?,
            language_ambiguous: ambiguous,
            normalized_text: prepared,
            segments,
            token_ids,
            canonical_token_ids,
        })
    }
}
fn split_text(
    text: &str,
    tokenizer: &IndexTts25Tokenizer,
    budget: usize,
    control: &dyn SessionControl,
) -> Result<Vec<String>> {
    if tokenizer.token_count(text) <= budget {
        return Ok(vec![text.into()]);
    }
    let mut pieces = Vec::new();
    let mut start = 0;
    for capture in PROTECTED.find_iter(text) {
        let m = capture.map_err(|e| invalid("text", e.to_string()))?;
        if m.start() > start {
            pieces.push((&text[start..m.start()], false));
        }
        pieces.push((m.as_str(), true));
        start = m.end();
    }
    if start < text.len() {
        pieces.push((&text[start..], false));
    }
    let mut chunks = Vec::new();
    for (piece, atomic) in pieces {
        control.check()?;
        if atomic {
            chunks.push(piece.to_owned());
            continue;
        }
        for part in piece.split_inclusive([
            '，', '。', '！', '？', '、', '；', '：', ',', '.', '!', '?', ';', ':', '\n',
        ]) {
            if tokenizer.token_count(part) <= budget {
                chunks.push(part.to_owned());
                continue;
            }
            let mut current = String::new();
            for character in part.chars() {
                control.check()?;
                let candidate = format!("{current}{character}");
                if !current.is_empty() && tokenizer.token_count(&candidate) > budget {
                    chunks.push(current);
                    current = character.to_string();
                } else {
                    current = candidate;
                }
            }
            if !current.is_empty() {
                chunks.push(current);
            }
        }
    }
    let mut segments = Vec::new();
    let mut current = String::new();
    for chunk in chunks {
        control.check()?;
        if !current.is_empty() && tokenizer.token_count(&(current.clone() + &chunk)) > budget {
            segments.push(current);
            current = chunk;
        } else {
            current += &chunk;
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    Ok(segments)
}
