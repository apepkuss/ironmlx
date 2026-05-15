//! B1-p2.3c-3 — Continuous batching (mid-batch admit/evict) integration tests.
//!
//! Three scenarios:
//!   1. continuous_batching_mid_decode_admit — central correctness gate.
//!      B=2 with A (max_new=3) + B (max_new=8). After A finishes, admit
//!      C (max_new=5) mid-decode. Verify all three rows produce correct
//!      tokens (bit-id >= 0.95 vs B=1 baselines).
//!   2. continuous_batching_full_reject — b_max=2 saturated by A+B both
//!      with max_new=20; admit C while decoding; verify C reply is Err
//!      "scheduler full".
//!   3. continuous_batching_drains_to_empty — admit A, drain, admit B
//!      100ms later; verify B prefills + completes through the actor's
//!      second outer batch iteration (batch_count == 2).
//!
//! All gated `#[ignore]`; drive via SchedulerActor cmd_tx (3c-3's
//! value lives in driver_loop's rolling decode loop).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use ironmlx::core::generate::{GenerateRequest, GenerationStream};
use ironmlx::core::sampler::Sampler;
use ironmlx::core::scheduler::StepEvent;
use ironmlx::core::server::scheduler_actor::{spawn_scheduler_actor, AdmitReply, SchedulerCommand};
use ironmlx::core::{Loader, Message, Tokenizer};
use ironmlx::models::qwen3_5::Qwen35Model;

const ARGMAX_BITID_GATE: f64 = 0.95;

fn load_fixture() -> (Arc<Mutex<Qwen35Model>>, Arc<Tokenizer>) {
    let model_dir = std::env::var("QWEN35_MODEL").expect("QWEN35_MODEL env var required");
    let model_path = Path::new(&model_dir);
    let loader = Loader::open(model_path).expect("Loader::open");
    let model = Qwen35Model::from_loader(&loader).expect("Qwen35Model::from_loader");
    let tokenizer = Tokenizer::from_loader(&loader).expect("Tokenizer::from_loader");
    (Arc::new(Mutex::new(model)), Arc::new(tokenizer))
}

fn tokenize_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let msgs = vec![Message {
        role: "user".into(),
        content: text.into(),
    }];
    let kw = serde_json::json!({"enable_thinking": false});
    let rendered = tokenizer
        .apply_chat_template(&msgs, true, Some(&kw))
        .expect("apply_chat_template");
    tokenizer.encode(&rendered, false).expect("encode")
}

fn make_request(prompt_ids: Vec<u32>, max_new_tokens: usize, stop: Vec<u32>) -> GenerateRequest {
    GenerateRequest {
        prompt_ids,
        max_new_tokens,
        sampler: Sampler::greedy(),
        stop_token_ids: stop,
        prefill_chunk_size: 256,
        pixel_values: None,
        image_grid_thw: None,
        image_spatial_merge_size: 2,
        image_token_id: 248056,
    }
}

fn run_b1_baseline(
    model: &Mutex<Qwen35Model>,
    tokenizer: &Tokenizer,
    request: GenerateRequest,
) -> Vec<u32> {
    let model_guard = model.blocking_lock();
    let mut stream = GenerationStream::new(&model_guard, tokenizer, request).expect("new stream");
    let mut tokens = Vec::new();
    loop {
        match stream.next_token().expect("next_token") {
            Some(ev) => {
                tokens.push(ev.token);
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            None => break,
        }
    }
    tokens
}

fn argmax_bit_id_ratio(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / n as f64
}

async fn submit_admit(
    cmd_tx: &tokio::sync::mpsc::Sender<SchedulerCommand>,
    req: GenerateRequest,
) -> ironmlx::Result<AdmitReply> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    cmd_tx
        .send(SchedulerCommand::Admit {
            request: req,
            reply_tx,
        })
        .await
        .map_err(|e| anyhow::anyhow!("cmd_tx.send: {e:?}"))?;
    reply_rx
        .await
        .map_err(|e| anyhow::anyhow!("reply_rx.await: {e:?}"))?
}

async fn drain_until_finished(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<StepEvent>,
) -> Vec<StepEvent> {
    let mut events = Vec::new();
    loop {
        match rx.recv().await {
            Some(ev) => {
                let done = ev.finish_reason.is_some();
                events.push(ev);
                if done {
                    break;
                }
            }
            None => break, // channel closed (EOF)
        }
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_mid_decode_admit() {
    let (model, tokenizer) = load_fixture();

    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let prompt_c = tokenize_prompt(&tokenizer, "Goodbye");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let max_new_a: usize = 3;
    let max_new_b: usize = 8;
    let max_new_c: usize = 5;

    // B=1 baselines.
    let baseline_a = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_a.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_a, stop))
        })
        .await
        .expect("baseline A")
    };
    let baseline_b = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_b.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_b, stop))
        })
        .await
        .expect("baseline B")
    };
    let baseline_c = {
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let prompt = prompt_c.clone();
        let stop = stop.clone();
        tokio::task::spawn_blocking(move || {
            run_b1_baseline(&model, &tokenizer, make_request(prompt, max_new_c, stop))
        })
        .await
        .expect("baseline C")
    };

    // Drive actor.
    let handle = spawn_scheduler_actor(model.clone(), 2);

    let reply_a = submit_admit(
        &handle.cmd_tx,
        make_request(prompt_a.clone(), max_new_a, stop.clone()),
    )
    .await
    .expect("admit A");
    let reply_b = submit_admit(
        &handle.cmd_tx,
        make_request(prompt_b.clone(), max_new_b, stop.clone()),
    )
    .await
    .expect("admit B");

    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;

    // Drain A.
    let events_a = drain_until_finished(&mut rx_a).await;
    assert_eq!(
        events_a.len(),
        max_new_a,
        "A should produce {} events; got {}",
        max_new_a,
        events_a.len()
    );
    assert_eq!(events_a.last().unwrap().finish_reason, Some("length"));

    // After A finishes (and gc clears slot 0), submit C — should admit_mid.
    let reply_c = submit_admit(
        &handle.cmd_tx,
        make_request(prompt_c.clone(), max_new_c, stop.clone()),
    )
    .await
    .expect("admit C mid-decode");
    let mut rx_c = reply_c.event_rx;

    // Drain B + C concurrently.
    let (events_b, events_c) = tokio::join!(
        drain_until_finished(&mut rx_b),
        drain_until_finished(&mut rx_c),
    );

    assert_eq!(events_b.len(), max_new_b);
    assert_eq!(events_b.last().unwrap().finish_reason, Some("length"));
    assert_eq!(events_c.len(), max_new_c);
    assert_eq!(events_c.last().unwrap().finish_reason, Some("length"));

    let tokens_a: Vec<u32> = events_a.iter().map(|e| e.token).collect();
    let tokens_b: Vec<u32> = events_b.iter().map(|e| e.token).collect();
    let tokens_c: Vec<u32> = events_c.iter().map(|e| e.token).collect();

    let ratio_a = argmax_bit_id_ratio(&tokens_a, &baseline_a);
    let ratio_b = argmax_bit_id_ratio(&tokens_b, &baseline_b);
    let ratio_c = argmax_bit_id_ratio(&tokens_c, &baseline_c);

    println!("[continuous_batching] tokens_a={tokens_a:?} bit-id={ratio_a:.4}");
    println!("[continuous_batching] tokens_b={tokens_b:?} bit-id={ratio_b:.4}");
    println!("[continuous_batching] tokens_c={tokens_c:?} bit-id={ratio_c:.4}");

    assert!(
        ratio_a >= ARGMAX_BITID_GATE,
        "A bit-id {} < {}",
        ratio_a,
        ARGMAX_BITID_GATE
    );
    assert!(
        ratio_b >= ARGMAX_BITID_GATE,
        "B bit-id {} < {}",
        ratio_b,
        ARGMAX_BITID_GATE
    );
    assert!(
        ratio_c >= ARGMAX_BITID_GATE,
        "C bit-id {} < {}",
        ratio_c,
        ARGMAX_BITID_GATE
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_full_reject() {
    let (model, tokenizer) = load_fixture();
    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let prompt_c = tokenize_prompt(&tokenizer, "Goodbye");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let handle = spawn_scheduler_actor(model.clone(), 2);

    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a, 20, stop.clone()))
        .await
        .expect("admit A");
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b, 20, stop.clone()))
        .await
        .expect("admit B");

    // Wait briefly so A + B reach Decoding phase.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Now submit C — both slots full + Decoding -> admit_mid Err.
    let admit_c_result =
        submit_admit(&handle.cmd_tx, make_request(prompt_c, 5, stop.clone())).await;
    match admit_c_result {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("scheduler full") || msg.contains("no row available"),
                "expected 'scheduler full' Err; got: {msg}"
            );
        }
        Ok(_) => panic!("C admit should have failed but succeeded"),
    }

    // Drain A + B normally.
    let mut rx_a = reply_a.event_rx;
    let mut rx_b = reply_b.event_rx;
    let (_, _) = tokio::join!(
        drain_until_finished(&mut rx_a),
        drain_until_finished(&mut rx_b),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn continuous_batching_drains_to_empty() {
    let (model, tokenizer) = load_fixture();
    let prompt_a = tokenize_prompt(&tokenizer, "Hello");
    let prompt_b = tokenize_prompt(&tokenizer, "World");
    let stop: Vec<u32> = tokenizer.eos_token_ids().to_vec();

    let handle = spawn_scheduler_actor(model.clone(), 2);

    // First admit + drain.
    let reply_a = submit_admit(&handle.cmd_tx, make_request(prompt_a, 4, stop.clone()))
        .await
        .expect("admit A");
    let mut rx_a = reply_a.event_rx;
    let events_a = drain_until_finished(&mut rx_a).await;
    assert_eq!(events_a.len(), 4);

    tokio::time::sleep(Duration::from_millis(150)).await;

    let bc_after_a = handle
        .batch_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        bc_after_a, 1,
        "expected 1 batch after A; got {}",
        bc_after_a
    );

    // Second admit triggers new outer batch.
    let reply_b = submit_admit(&handle.cmd_tx, make_request(prompt_b, 5, stop.clone()))
        .await
        .expect("admit B");
    let mut rx_b = reply_b.event_rx;
    let events_b = drain_until_finished(&mut rx_b).await;
    assert_eq!(events_b.len(), 5);

    tokio::time::sleep(Duration::from_millis(150)).await;
    let bc_after_b = handle
        .batch_count
        .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        bc_after_b, 2,
        "expected 2 batches after B; got {}",
        bc_after_b
    );
}
