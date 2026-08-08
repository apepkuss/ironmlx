use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ironmlx::core::tool_calling::{AgentMessage, ToolDialect};
use ironmlx::core::Tokenizer;

fn user(content: &str) -> AgentMessage {
    AgentMessage {
        role: "user".to_owned(),
        content: Some(content.to_owned()),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn assistant(content: &str) -> AgentMessage {
    AgentMessage {
        role: "assistant".to_owned(),
        content: Some(content.to_owned()),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn tool_kwargs(dialect: ToolDialect, description: &str, tool_count: usize) -> serde_json::Value {
    let tools = (0..tool_count)
        .map(|index| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": format!("lookup_weather_{index}"),
                    "description": format!("{description} {index}"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"}
                        },
                        "required": ["city"]
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let mut kwargs = serde_json::json!({"tools": tools});
    if matches!(dialect, ToolDialect::Qwen35 | ToolDialect::Glm) {
        kwargs["enable_thinking"] = serde_json::Value::Bool(false);
    }
    kwargs
}

fn uncached(
    tokenizer: &Tokenizer,
    messages: &[AgentMessage],
    kwargs: &serde_json::Value,
) -> Vec<u32> {
    let prompt = tokenizer
        .apply_chat_template(messages, true, Some(kwargs))
        .expect("render prompt");
    tokenizer.encode(&prompt, false).expect("encode prompt")
}

fn model_from_env() -> Tokenizer {
    let model = std::env::var("IRONMLX_TOOL_CACHE_MODEL")
        .expect("IRONMLX_TOOL_CACHE_MODEL must point to a real supported model snapshot");
    Tokenizer::from_model_dir(Path::new(&model)).expect("load tokenizer")
}

#[test]
#[ignore = "requires IRONMLX_TOOL_CACHE_MODEL pointing to a real supported model snapshot"]
fn real_tool_cache_is_exact_for_cold_hot_history_and_schema_change() {
    let tokenizer = Arc::new(model_from_env());
    let dialect = tokenizer.tool_dialect().expect("supported tool dialect");
    let kwargs = tool_kwargs(dialect, "查询指定城市未来天气", 4);
    let first = vec![user("东京明天会下雨吗？")];

    let expected_first = uncached(&tokenizer, &first, &kwargs);
    let cold = tokenizer
        .render_and_encode_tool_prompt(&first, &kwargs)
        .expect("cold cache request");
    assert_eq!(cold, expected_first, "cold token IDs diverged");

    let hot = tokenizer
        .render_and_encode_tool_prompt(&first, &kwargs)
        .expect("exact cache request");
    assert_eq!(hot, expected_first, "exact-hit token IDs diverged");

    let grown = vec![
        user("东京明天会下雨吗？"),
        assistant("东京明天降雨概率较低。"),
        user("那后天呢？"),
    ];
    let expected_grown = uncached(&tokenizer, &grown, &kwargs);
    let cached_grown = tokenizer
        .render_and_encode_tool_prompt(&grown, &kwargs)
        .expect("history-prefix cache request");
    assert_eq!(
        cached_grown, expected_grown,
        "prefix-hit token IDs diverged"
    );

    let changed_kwargs = tool_kwargs(dialect, "查询城市天气与空气质量", 4);
    let expected_changed = uncached(&tokenizer, &first, &changed_kwargs);
    let changed = tokenizer
        .render_and_encode_tool_prompt(&first, &changed_kwargs)
        .expect("changed-schema cache request");
    assert_eq!(
        changed, expected_changed,
        "schema-change token IDs diverged"
    );

    let before_threads = tokenizer.tool_prompt_cache_stats();
    let handles = (0..8)
        .map(|_| {
            let tokenizer = Arc::clone(&tokenizer);
            let messages = first.clone();
            let kwargs = kwargs.clone();
            std::thread::spawn(move || {
                tokenizer
                    .render_and_encode_tool_prompt(&messages, &kwargs)
                    .expect("concurrent cache request")
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        assert_eq!(
            handle.join().expect("cache thread panicked"),
            expected_first
        );
    }

    let stats = tokenizer.tool_prompt_cache_stats();
    assert_eq!(stats.misses, 2, "cold and changed schema must miss");
    assert_eq!(stats.prefix_hits, 1, "grown history must reuse a prefix");
    assert_eq!(stats.exact_hits - before_threads.exact_hits, 8);
    assert!(stats.reused_tokens > 0);
    assert_eq!(stats.entries, 3);
    assert!(stats.bytes > 0 && stats.bytes <= 16 * 1024 * 1024);
    eprintln!("dialect={dialect:?} cache_stats={stats:?}");
}

fn measure<F>(iterations: usize, mut operation: F) -> Duration
where
    F: FnMut() -> Vec<u32>,
{
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    start.elapsed()
}

#[test]
#[ignore = "manual fixed-environment microbenchmark; requires IRONMLX_TOOL_CACHE_MODEL"]
fn tool_cache_balanced_abba_microbenchmark() {
    const ITERATIONS: usize = 500;
    let tokenizer = model_from_env();
    let dialect = tokenizer.tool_dialect().expect("supported tool dialect");
    let kwargs = tool_kwargs(dialect, "查询城市天气、湿度、风速和空气质量", 24);
    let messages = vec![user("请比较东京、大阪和札幌未来七天的天气。")];
    let expected = uncached(&tokenizer, &messages, &kwargs);
    assert_eq!(
        tokenizer
            .render_and_encode_tool_prompt(&messages, &kwargs)
            .expect("warm cache"),
        expected
    );

    let a1 = measure(ITERATIONS, || uncached(&tokenizer, &messages, &kwargs));
    let b1 = measure(ITERATIONS, || {
        tokenizer
            .render_and_encode_tool_prompt(&messages, &kwargs)
            .expect("cached prompt")
    });
    let b2 = measure(ITERATIONS, || {
        tokenizer
            .render_and_encode_tool_prompt(&messages, &kwargs)
            .expect("cached prompt")
    });
    let a2 = measure(ITERATIONS, || uncached(&tokenizer, &messages, &kwargs));

    eprintln!(
        "dialect={dialect:?} iterations={ITERATIONS} A1_uncached_us={} B1_cached_us={} B2_cached_us={} A2_uncached_us={} exact_hits={}",
        a1.as_micros(),
        b1.as_micros(),
        b2.as_micros(),
        a2.as_micros(),
        tokenizer.tool_prompt_cache_stats().exact_hits,
    );
}

fn growing_histories(turns: usize) -> Vec<Vec<AgentMessage>> {
    let mut messages = vec![user("东京明天会下雨吗？")];
    let mut histories = vec![messages.clone()];
    for index in 1..turns {
        messages.push(assistant(&format!(
            "第 {index} 次查询结果：天气稳定，适合出行。"
        )));
        messages.push(user(&format!("请继续查询第 {index} 个日期。")));
        histories.push(messages.clone());
    }
    histories
}

fn measure_histories<F>(histories: &[Vec<AgentMessage>], mut operation: F) -> Duration
where
    F: FnMut(&[AgentMessage]) -> Vec<u32>,
{
    let start = Instant::now();
    for messages in histories {
        black_box(operation(messages));
    }
    start.elapsed()
}

#[test]
#[ignore = "manual fixed-environment prefix microbenchmark; requires IRONMLX_TOOL_CACHE_MODEL"]
fn tool_prefix_cache_balanced_abba_microbenchmark() {
    const TURNS: usize = 48;
    let model = std::env::var("IRONMLX_TOOL_CACHE_MODEL")
        .expect("IRONMLX_TOOL_CACHE_MODEL must point to a model snapshot");
    let tokenizer_a = Tokenizer::from_model_dir(Path::new(&model)).expect("load tokenizer A");
    let dialect = tokenizer_a.tool_dialect().expect("supported tool dialect");
    let kwargs = tool_kwargs(dialect, "查询城市天气、湿度、风速和空气质量", 24);
    let histories = growing_histories(TURNS);

    let a1 = measure_histories(&histories, |messages| {
        uncached(&tokenizer_a, messages, &kwargs)
    });
    let tokenizer_b1 = Tokenizer::from_model_dir(Path::new(&model)).expect("load tokenizer B1");
    let b1 = measure_histories(&histories, |messages| {
        tokenizer_b1
            .render_and_encode_tool_prompt(messages, &kwargs)
            .expect("prefix-cached prompt B1")
    });
    let tokenizer_b2 = Tokenizer::from_model_dir(Path::new(&model)).expect("load tokenizer B2");
    let b2 = measure_histories(&histories, |messages| {
        tokenizer_b2
            .render_and_encode_tool_prompt(messages, &kwargs)
            .expect("prefix-cached prompt B2")
    });
    let a2 = measure_histories(&histories, |messages| {
        uncached(&tokenizer_a, messages, &kwargs)
    });

    let stats_b1 = tokenizer_b1.tool_prompt_cache_stats();
    let stats_b2 = tokenizer_b2.tool_prompt_cache_stats();
    assert_eq!(stats_b1.prefix_hits, TURNS as u64 - 1);
    assert_eq!(stats_b2.prefix_hits, TURNS as u64 - 1);
    assert_eq!(stats_b1.misses, 1);
    assert_eq!(stats_b2.misses, 1);
    eprintln!(
        "dialect={dialect:?} turns={TURNS} A1_uncached_us={} B1_prefix_us={} B2_prefix_us={} A2_uncached_us={} B1_reused_tokens={} B2_reused_tokens={}",
        a1.as_micros(),
        b1.as_micros(),
        b2.as_micros(),
        a2.as_micros(),
        stats_b1.reused_tokens,
        stats_b2.reused_tokens,
    );
}
