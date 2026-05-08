//! Dump ironmlx's rendered chat template for a fixed input — to compare
//! byte-for-byte against HF transformers' output for the same input. Used
//! only as a debug harness for the early-EOS bug investigation.

use ironmlx::core::{Loader, Message, Tokenizer};

#[test]
#[ignore = "requires QWEN35_MODEL env var"]
fn dump_ironmlx_chat_template_render() {
    let snap =
        std::env::var("QWEN35_MODEL").expect("set QWEN35_MODEL to the Qwen3.5-4B-MLX-4bit dir");
    let loader = Loader::open(std::path::Path::new(&snap)).expect("open loader");
    let tok = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");

    let msgs = vec![Message {
        role: "user".into(),
        content: "Say hi in 5 words.".into(),
    }];

    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tok
        .apply_chat_template(&msgs, /*add_generation_prompt=*/ true, Some(&kw))
        .expect("apply_chat_template");

    println!("=== ironmlx rendered (enable_thinking=false) ===");
    println!("{rendered:?}");
    println!();
    println!("len: {} chars", rendered.len());

    let ids = tok
        .encode(&rendered, /*add_special_tokens=*/ false)
        .expect("encode");
    println!();
    println!("tokens ({}): {:?}", ids.len(), &ids[..ids.len().min(30)]);
}
