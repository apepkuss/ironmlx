use ironmlx_audio::{
    text::{resolve_language, IndexTts25TextFrontend, TextLimits},
    AudioError, Language, Result, SessionControl,
};
use std::path::PathBuf;
struct Active;
impl SessionControl for Active {
    fn check(&self) -> Result<()> {
        Ok(())
    }
}
fn language(code: &str) -> Language {
    match code {
        "zh" => Language::Zh,
        "en" => Language::En,
        "ja" => Language::Ja,
        "es" => Language::Es,
        "ar" => Language::Ar,
        _ => Language::Auto,
    }
}
#[test]
fn language_heuristics_and_ambiguity() {
    assert_eq!(
        resolve_language("hello", Language::Auto).unwrap(),
        (Language::En, true)
    );
    assert_eq!(
        resolve_language("漢字", Language::Auto).unwrap(),
        (Language::Zh, false)
    );
    assert_eq!(
        resolve_language("漢字かな", Language::Auto).unwrap(),
        (Language::Ja, false)
    );
    assert!(resolve_language("مرحبا中文", Language::Auto).is_err());
    assert!(resolve_language("123", Language::Auto).is_err());
    assert_eq!(
        resolve_language("123", Language::En).unwrap(),
        (Language::En, false)
    );
}
/// Requires fixed model vocabulary, WeText FSTs and UniDic-lite; no Python at execution.
#[test]
#[ignore = "requires IRONMLX_INDEXTTS25_SNAPSHOT, IRONMLX_WETEXT_FSTS and IRONMLX_UNIDIC_DIR"]
fn real_text_frontend_matches_fixed_python_reference() {
    let path = |key| PathBuf::from(std::env::var(key).expect(key));
    let mut frontend = IndexTts25TextFrontend::load(
        &path("IRONMLX_INDEXTTS25_SNAPSHOT").join("multilingual_zh_ja_yue_char_del.tiktoken"),
        &path("IRONMLX_WETEXT_FSTS"),
        &path("IRONMLX_UNIDIC_DIR"),
    )
    .unwrap();
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/text.json")).unwrap();
    for (index, case) in cases["cases"].as_array().unwrap().iter().enumerate() {
        let text = case["input"].as_str().unwrap();
        let actual = frontend
            .prepare(
                text,
                language(case["requested_language"].as_str().unwrap()),
                &TextLimits::default(),
                &Active,
            )
            .unwrap_or_else(|e| panic!("case {index}: {text}: {e}"));
        assert_eq!(
            actual.normalized_text,
            case["normalized_text"].as_str().unwrap(),
            "case {index}: {text}"
        );
        assert_eq!(actual.language.code(), case["language"].as_str().unwrap());
        assert_eq!(
            actual.language_id,
            case["language_id"].as_u64().unwrap() as u32
        );
        assert_eq!(
            actual.language_ambiguous,
            case["language_ambiguous"].as_bool().unwrap()
        );
        assert_eq!(
            serde_json::to_value(&actual.segments).unwrap(),
            case["segments"],
            "segments {index}"
        );
        assert_eq!(
            serde_json::to_value(&actual.token_ids).unwrap(),
            case["token_ids"],
            "tokens {index}"
        );
        assert_eq!(
            serde_json::to_value(&actual.canonical_token_ids).unwrap(),
            case["canonical_token_ids"],
            "canonical {index}"
        );
    }
    let overlong = format!("<word|{}>", "AH ".repeat(1000));
    assert!(matches!(
        frontend.prepare(&overlong, Language::En, &TextLimits::default(), &Active),
        Err(AudioError::CapacityExceeded { .. })
    ));
    assert!(frontend
        .prepare(
            "hello",
            Language::En,
            &TextLimits {
                max_total_tokens: 1,
                ..Default::default()
            },
            &Active
        )
        .is_err());
    assert!(frontend
        .prepare("\0test", Language::En, &TextLimits::default(), &Active)
        .is_err());
}
