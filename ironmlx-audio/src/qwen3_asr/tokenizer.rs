use anyhow::{anyhow, ensure, Context, Result};
use std::path::Path;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteDecoder,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
    AddedToken, SplitDelimiterBehavior, Tokenizer,
};

pub(super) fn load(directory: &Path) -> Result<Tokenizer> {
    let consolidated = directory.join("tokenizer.json");
    let mut tokenizer = if consolidated.is_file() {
        Tokenizer::from_file(consolidated).map_err(|e| anyhow!("ASR tokenizer: {e}"))?
    } else {
        let vocab = directory.join("vocab.json");
        let merges = directory.join("merges.txt");
        let model = BPE::from_file(
            vocab.to_str().context("invalid tokenizer path")?,
            merges.to_str().context("invalid merges path")?,
        )
        .build()
        .map_err(|e| anyhow!("ASR BPE: {e}"))?;
        let mut tokenizer = Tokenizer::new(model);
        let split = Split::new(SplitPattern::Regex(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".into()
        ), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("ASR pretokenizer: {e}"))?;
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::default()
                .add_prefix_space(false)
                .use_regex(false)
                .into(),
        ])));
        tokenizer.with_decoder(Some(ByteDecoder::default()));
        tokenizer
    };
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("tokenizer_config.json"))?)?;
    let added = config["added_tokens_decoder"]
        .as_object()
        .context("missing ASR special tokens")?;
    let mut tokens = added
        .iter()
        .map(|(id, value)| Ok((id.parse::<u32>()?, value)))
        .collect::<Result<Vec<_>>>()?;
    tokens.sort_by_key(|(id, _)| *id);
    for (id, value) in tokens {
        let content = value["content"]
            .as_str()
            .context("invalid ASR special token")?;
        let token = AddedToken::from(content, value["special"].as_bool().unwrap_or(false))
            .single_word(value["single_word"].as_bool().unwrap_or(false))
            .lstrip(value["lstrip"].as_bool().unwrap_or(false))
            .rstrip(value["rstrip"].as_bool().unwrap_or(false))
            .normalized(value["normalized"].as_bool().unwrap_or(false));
        tokenizer.add_tokens(&[token]);
        ensure!(
            tokenizer.token_to_id(content) == Some(id),
            "ASR token ID mismatch for {content}"
        );
    }
    for (token, id) in [
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<|audio_start|>", 151669),
        ("<|audio_end|>", 151670),
        ("<|audio_pad|>", 151676),
        ("<asr_text>", 151704),
    ] {
        ensure!(
            tokenizer.token_to_id(token) == Some(id),
            "missing ASR token {token}"
        );
    }
    Ok(tokenizer)
}
