//! IndexTTS 2.5 BPE vocabulary and special-token ordering.
//! Token order follows vanch007/mlx-indextts2 a7666367 (tokenizer_v25.py).
use crate::{error::invalid, AudioError, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, path::Path};
const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
];
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
pub struct IndexTts25Tokenizer {
    encoding: tiktoken_rs::CoreBPE,
}
impl IndexTts25Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() != 907395 {
            return Err(invalid("tokenizer", "unexpected vocabulary size"));
        }
        let bytes = std::fs::read(path)?;
        if format!("{:x}", Sha256::digest(&bytes))
            != "747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718"
        {
            return Err(AudioError::ResourceMismatch {
                component: "tokenizer".into(),
                reason: "unexpected vocabulary SHA-256".into(),
            });
        }
        let text = std::str::from_utf8(&bytes).map_err(|e| invalid("tokenizer", e.to_string()))?;
        let mut entries = Vec::new();
        let mut ranks = HashSet::new();
        let mut tokens = HashSet::new();
        for line in text.lines().filter(|v| !v.is_empty()) {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(invalid("tokenizer", "expected token and rank"));
            }
            // The SHA-pinned vocabulary contains `= 48474`: Python's base64
            // reader maps that entry to empty bytes. Preserve its rank rather
            // than rejecting, deleting, or shifting the published vocabulary.
            let token = if parts[0] == "=" && parts[1] == "48474" {
                Vec::new()
            } else {
                STANDARD
                    .decode(parts[0])
                    .map_err(|e| invalid("tokenizer", e.to_string()))?
            };
            let rank: u32 = parts[1]
                .parse()
                .map_err(|e: std::num::ParseIntError| invalid("tokenizer", e.to_string()))?;
            if !ranks.insert(rank) || !tokens.insert(token.clone()) {
                return Err(invalid("tokenizer", "duplicate token or rank"));
            }
            entries.push((token, rank));
        }
        if !(0..entries.len() as u32).all(|r| ranks.contains(&r))
            || !(0..=255u8).all(|b| tokens.contains(&vec![b]))
        {
            return Err(invalid("tokenizer", "incomplete vocabulary"));
        }
        let mut specials = vec![
            "<|endoftext|>".to_owned(),
            "<|startoftranscript|>".to_owned(),
        ];
        specials.extend(LANGUAGES[..99].iter().map(|l| format!("<|{l}|>")));
        specials.extend(
            [
                "ASR",
                "AED",
                "SER",
                "Speech",
                "/Speech",
                "BGM",
                "/BGM",
                "Laughter",
                "/Laughter",
                "Applause",
                "/Applause",
                "HAPPY",
                "SAD",
                "ANGRY",
                "NEUTRAL",
                "translate",
                "transcribe",
                "startoflm",
                "startofprev",
                "nospeech",
                "notimestamps",
            ]
            .iter()
            .map(|s| format!("<|{s}|>")),
        );
        specials.extend((1..=30).map(|i| format!("<|SPECIAL_TOKEN_{i}|>")));
        specials.extend(
            [
                "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
            ]
            .iter()
            .map(|s| format!("<|{s}|>")),
        );
        specials.extend((1..=13).map(|i| format!("<|TTS/SP{i:02}|>")));
        specials.extend((0..=1500).map(|i| format!("<|{}.{:02}|>", i * 2 / 100, i * 2 % 100)));
        if entries.len() + specials.len() != 60509 {
            return Err(invalid("tokenizer", "unexpected vocabulary size"));
        }
        let special_base = entries.len() as u32;
        let encoding = tiktoken_rs::CoreBPE::new(
            entries.into_iter().collect(),
            specials
                .into_iter()
                .enumerate()
                .map(|(i, s)| (s, special_base + i as u32))
                .collect(),
            PATTERN,
        )
        .map_err(|e| invalid("tokenizer", e.to_string()))?;
        Ok(Self { encoding })
    }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoding.encode_with_special_tokens(text)
    }
    pub fn token_count(&self, text: &str) -> usize {
        self.encode(text).len()
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.encoding
            .decode(tokens)
            .map_err(|e| invalid("tokenizer", e.to_string()))
    }
}
