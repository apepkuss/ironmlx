//! IndexTTS wrapper ordering around WeText; external glossaries are disabled.
use super::fst::WeText;
use crate::{error::invalid, Language, Result, SessionControl};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use std::{path::Path, sync::LazyLock};
static PINYIN: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(r"(?i)(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])").expect("pinyin regex")
});
static TECH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+").expect("tech regex"));
static NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}").expect("name regex")
});
static CONTRACTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(what|where|who|which|how|t?here|it|s?he|that|this)'s")
        .expect("contraction regex")
});
static JQX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)([jqx])[uü](n|e|an)*(\d)").expect("jqx regex"));
static RESTORE_TECH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*<H>\s*").expect("tech restoration"));
static MAP: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(
        "../../resources/indextts25/character-map.json"
    ))
    .expect("fixed character map")
});
fn replace_characters(text: &str, sorted: bool, chinese: bool) -> String {
    let mut replacements = MAP.clone();
    if chinese {
        replacements.insert(0, ("$".into(), ".".into()));
    }
    if sorted {
        replacements.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
    }
    let mut result = String::new();
    let mut index = 0;
    while index < text.len() {
        if let Some((key, value)) = replacements
            .iter()
            .find(|(k, _)| text[index..].starts_with(k))
        {
            result.push_str(value);
            index += key.len();
        } else {
            let ch = text[index..]
                .chars()
                .next()
                .expect("valid character boundary");
            result.push(ch);
            index += ch.len_utf8();
        }
    }
    result
}
fn emoji(ch: char) -> bool {
    matches!(ch as u32,0x1f600..=0x1f64f|0x1f300..=0x1f5ff|0x1f680..=0x1f6ff|0x1f1e0..=0x1f1ff|0x2600..=0x26ff|0x2700..=0x27bf|0x1f900..=0x1f9ff|0x1fa00..=0x1faff|0xfe00..=0xfe0f|0x200d)
}
fn placeholder(prefix: &str, index: usize) -> String {
    format!(
        "<{prefix}_{}>",
        char::from_u32('a' as u32 + index as u32).expect("bounded placeholders")
    )
}
pub(super) struct Normalizer {
    wetext: WeText,
}
impl Normalizer {
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            wetext: WeText::load(root)?,
        })
    }
    pub fn normalize(
        &self,
        text: &str,
        language: Language,
        control: &dyn SessionControl,
    ) -> Result<String> {
        let mut text = replace_characters(text, true, false);
        if !matches!(language, Language::Zh | Language::En) {
            return Ok(text);
        }
        text = text
            .chars()
            .map(|ch| if emoji(ch) { ' ' } else { ch })
            .collect();
        // The upstream IndexTTS wrapper expands this limited pattern even though
        // WeText's optional contractions package expansion is disabled.
        text = CONTRACTION.replace_all(&text, "${1} is").into_owned();
        let chinese = text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            || !text.chars().any(|c| c.is_ascii_alphabetic())
            || PINYIN
                .is_match(&text)
                .map_err(|e| invalid("text", e.to_string()))?;
        if chinese {
            text = text.trim_end().to_owned();
        }
        let tech = TECH.is_match(&text);
        text = TECH
            .replace_all(&text, |cap: &regex::Captures<'_>| {
                cap[0].replace('-', "<H>")
            })
            .into_owned();
        let mut pinyin = Vec::new();
        let mut names = Vec::new();
        if chinese {
            for capture in PINYIN.captures_iter(&text) {
                let capture = capture.map_err(|e| invalid("text", e.to_string()))?;
                let word = capture[0].to_owned();
                if !pinyin.contains(&word) {
                    pinyin.push(word);
                }
            }
            for (i, word) in pinyin.iter().enumerate() {
                text = text.replace(word, &placeholder("pinyin", i));
            }
            for capture in NAME.find_iter(&text) {
                let word = capture.as_str().to_owned();
                if !names.contains(&word) {
                    names.push(word);
                }
            }
            for (i, word) in names.iter().enumerate() {
                text = text.replace(word, &placeholder("n", i));
            }
        }
        text = self.wetext.normalize(&text, chinese, control)?;
        for (i, word) in names.iter().enumerate() {
            text = text.replace(&placeholder("n", i), word);
        }
        for (i, word) in pinyin.iter().enumerate() {
            let corrected = if word.starts_with(['j', 'q', 'x', 'J', 'Q', 'X']) {
                JQX.replace_all(word, "${1}v${2}${3}").into_owned()
            } else {
                word.clone()
            };
            text = text.replace(&placeholder("pinyin", i), &corrected.to_uppercase());
        }
        if tech {
            text = RESTORE_TECH.replace_all(&text, "-").into_owned();
        }
        Ok(replace_characters(&text, false, chinese))
    }
}
