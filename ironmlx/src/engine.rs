use std::collections::{HashMap, HashSet, VecDeque};

use tokio::sync::mpsc;

use ironmlx_core::cache::CacheManager;
use ironmlx_core::generate::{BatchGenerator, FinishReason, SamplerConfig, SeqUid, Tokenizer};
use ironmlx_core::media::ProcessedMedia;
use ironmlx_core::model::Model;

/// Chip-specific Metal command buffer defaults, read from MLX at startup.
#[derive(Clone, Copy)]
struct MetalBufferDefaults {
    ops: i32,
    mb: i32,
}

impl MetalBufferDefaults {
    fn from_device() -> Self {
        let ops = ironmlx_core::metal::get_max_ops_per_buffer().unwrap_or(40);
        let mb = ironmlx_core::metal::get_max_mb_per_buffer().unwrap_or(40);
        eprintln!(
            "[engine] Metal buffer defaults: max_ops={}, max_mb={}",
            ops, mb
        );
        Self { ops, mb }
    }
}

/// Output for each generation step
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RequestOutput {
    pub request_id: String,
    pub token_id: i32,
    pub token_text: String,
    /// None while generating, Some("stop"|"length") when finished
    pub finish_reason: Option<String>,
}

/// Commands sent to the engine
#[allow(dead_code)]
pub enum EngineCommand {
    AddRequest {
        request_id: String,
        prompt_token_ids: Vec<i32>,
        sampling_params: SamplerConfig,
        max_tokens: usize,
        eos_token_id: i32,
        token_tx: mpsc::Sender<RequestOutput>,
        media: Option<Vec<ProcessedMedia>>,
    },
    AbortRequest {
        request_id: String,
    },
    Shutdown,
}

/// A pending request in the waiting queue
struct PendingRequest {
    request_id: String,
    prompt_token_ids: Vec<i32>,
    sampling_params: SamplerConfig,
    max_tokens: usize,
    eos_token_id: i32,
    token_tx: mpsc::Sender<RequestOutput>,
    media: Option<Vec<ProcessedMedia>>,
}

/// Maps SeqUid to running request metadata
struct RunningRequest {
    request_id: String,
    token_tx: mpsc::Sender<RequestOutput>,
}

// SAFETY: MLX C handles are reference-counted and internally synchronized.
// EngineCore runs on a single blocking thread; all model access is sequential.
unsafe impl Send for EngineCore {}

/// The engine core that owns the model and processes requests using BatchGenerator.
pub struct EngineCore {
    cmd_rx: mpsc::Receiver<EngineCommand>,
    model: Model,
    tokenizer: Tokenizer,
    cache_manager: Option<CacheManager>,
    max_num_seqs: usize,
}

impl EngineCore {
    #[allow(dead_code)]
    pub fn new(cmd_rx: mpsc::Receiver<EngineCommand>, model: Model, tokenizer: Tokenizer) -> Self {
        Self {
            cmd_rx,
            model,
            tokenizer,
            cache_manager: None,
            max_num_seqs: 256,
        }
    }

    pub fn with_cache_manager(
        cmd_rx: mpsc::Receiver<EngineCommand>,
        model: Model,
        tokenizer: Tokenizer,
        cache_manager: CacheManager,
        max_num_seqs: usize,
    ) -> Self {
        Self {
            cmd_rx,
            model,
            tokenizer,
            cache_manager: Some(cache_manager),
            max_num_seqs,
        }
    }

    /// Blocking run loop — call from a dedicated thread.
    pub fn run(&mut self) {
        let metal_defaults = MetalBufferDefaults::from_device();
        let mut batch = if let Some(cm) = self.cache_manager.take() {
            BatchGenerator::with_cache_manager(&self.model, cm)
        } else {
            BatchGenerator::new(&self.model)
        };
        let mut waiting: VecDeque<PendingRequest> = VecDeque::new();
        let mut running: HashMap<SeqUid, RunningRequest> = HashMap::new();
        let mut prefill_pending: PrefillRunning = HashMap::new();
        let mut pending_aborts: HashSet<String> = HashSet::new();
        let max_num_seqs = self.max_num_seqs;
        let mut step_counter: u64 = 0;
        loop {
            // Periodic GPU memory cleanup (every 32 steps, following omlx pattern)
            step_counter += 1;
            if step_counter % 32 == 0 {
                ironmlx_core::memory::clear_cache().ok();
            }
            // 1. Drain all pending commands
            match drain_commands(&mut self.cmd_rx, &mut waiting, &mut pending_aborts) {
                DrainResult::Continue => {}
                DrainResult::Shutdown => return,
            }

            // 2. Process pending aborts
            process_aborts(&mut pending_aborts, &mut running, &mut batch);

            // 3. Enqueue waiting requests into chunked prefill
            enqueue_to_prefill(
                &mut waiting,
                &mut batch,
                &mut prefill_pending,
                max_num_seqs,
                metal_defaults,
            );

            // 4. If nothing is active and nothing is prefilling, block-wait
            if batch.active_count() == 0 && batch.prefilling_count() == 0 {
                if waiting.is_empty() {
                    match self.cmd_rx.blocking_recv() {
                        Some(EngineCommand::Shutdown) | None => return,
                        Some(cmd) => {
                            handle_command(cmd, &mut waiting, &mut pending_aborts);
                        }
                    }
                    process_aborts(&mut pending_aborts, &mut running, &mut batch);
                    enqueue_to_prefill(
                        &mut waiting,
                        &mut batch,
                        &mut prefill_pending,
                        max_num_seqs,
                        metal_defaults,
                    );
                }

                if batch.active_count() == 0 && batch.prefilling_count() == 0 {
                    continue;
                }
            }

            // 5. Process one chunk of prefill for all prefilling sequences
            step_prefill(
                &mut batch,
                &mut prefill_pending,
                &mut running,
                &self.tokenizer,
                metal_defaults,
            );

            // 6. Execute one decode step for all active sequences
            let active = batch.active_count();
            if active == 0 {
                continue;
            }
            let use_batched = self.model.supports_batched_forward() && active > 1;
            let step_t0 = std::time::Instant::now();
            let step_result = if use_batched {
                batch.step_true_batched()
            } else {
                batch.step_batched()
            };
            let step_ms = step_t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[engine] decode step: active={}, path={}, took={:.1}ms",
                active,
                if use_batched { "batched" } else { "sequential" },
                step_ms
            );
            match step_result {
                Ok(responses) => {
                    process_batch_responses(responses, &mut running, &mut batch, &self.tokenizer);
                }
                Err(e) => {
                    eprintln!("Engine step error: {}", e);
                    let uids: Vec<SeqUid> = running.keys().copied().collect();
                    for uid in uids {
                        batch.remove(uid);
                        if let Some(req) = running.remove(&uid) {
                            let _ = req.token_tx.blocking_send(RequestOutput {
                                request_id: req.request_id,
                                token_id: 0,
                                token_text: String::new(),
                                finish_reason: Some("stop".to_string()),
                            });
                        }
                    }
                }
            }
        }
    }
}

enum DrainResult {
    Continue,
    Shutdown,
}

fn drain_commands(
    cmd_rx: &mut mpsc::Receiver<EngineCommand>,
    waiting: &mut VecDeque<PendingRequest>,
    pending_aborts: &mut HashSet<String>,
) -> DrainResult {
    loop {
        match cmd_rx.try_recv() {
            Ok(EngineCommand::Shutdown) => return DrainResult::Shutdown,
            Ok(cmd) => {
                handle_command(cmd, waiting, pending_aborts);
            }
            Err(mpsc::error::TryRecvError::Empty) => return DrainResult::Continue,
            Err(mpsc::error::TryRecvError::Disconnected) => return DrainResult::Shutdown,
        }
    }
}

fn handle_command(
    cmd: EngineCommand,
    waiting: &mut VecDeque<PendingRequest>,
    pending_aborts: &mut HashSet<String>,
) {
    match cmd {
        EngineCommand::Shutdown => {} // handled by caller
        EngineCommand::AddRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            max_tokens,
            eos_token_id,
            token_tx,
            media,
        } => {
            waiting.push_back(PendingRequest {
                request_id,
                prompt_token_ids,
                sampling_params,
                max_tokens,
                eos_token_id,
                token_tx,
                media,
            });
        }
        EngineCommand::AbortRequest { request_id } => {
            waiting.retain(|r| r.request_id != request_id);
            pending_aborts.insert(request_id);
        }
    }
}

fn process_aborts(
    pending_aborts: &mut HashSet<String>,
    running: &mut HashMap<SeqUid, RunningRequest>,
    batch: &mut BatchGenerator<'_>,
) {
    if pending_aborts.is_empty() {
        return;
    }

    let abort_ids: Vec<String> = pending_aborts.drain().collect();
    let mut uids_to_remove = Vec::new();
    for (&uid, req) in running.iter() {
        if abort_ids.contains(&req.request_id) {
            uids_to_remove.push(uid);
        }
    }
    for uid in uids_to_remove {
        batch.remove(uid);
        if let Some(req) = running.remove(&uid) {
            let _ = req.token_tx.blocking_send(RequestOutput {
                request_id: req.request_id,
                token_id: 0,
                token_text: String::new(),
                finish_reason: Some("stop".to_string()),
            });
        }
    }
}

/// Map from SeqUid to PendingRequest for sequences in chunked prefill.
type PrefillRunning = HashMap<SeqUid, PendingRequest>;

/// Enqueue waiting requests into chunked prefill (not full prefill).
/// VLM requests fall back to full prefill since they need vision encoding.
fn enqueue_to_prefill(
    waiting: &mut VecDeque<PendingRequest>,
    batch: &mut BatchGenerator<'_>,
    prefill_pending: &mut PrefillRunning,
    max_num_seqs: usize,
    metal_defaults: MetalBufferDefaults,
) {
    while (batch.active_count() + batch.prefilling_count()) < max_num_seqs {
        // Memory pressure check
        if let Ok(active) = ironmlx_core::memory::get_active_memory()
            && let Ok(limit) = ironmlx_core::memory::get_memory_limit()
            && limit > 0
            && active > limit * 9 / 10
        {
            let _ = ironmlx_core::memory::clear_cache();
            if let Ok(after) = ironmlx_core::memory::get_active_memory()
                && after > limit * 9 / 10
            {
                break;
            }
        }

        let Some(req) = waiting.pop_front() else {
            break;
        };

        if req.token_tx.is_closed() {
            continue;
        }

        // VLM requests use full prefill (need vision encoding)
        if req.media.is_some() {
            let uid = batch.begin_prefill(
                &req.prompt_token_ids,
                req.sampling_params.clone(),
                req.eos_token_id,
                req.max_tokens,
            );
            eprintln!(
                "[engine] enqueue VLM prefill: req={}, prompt_tokens={}",
                req.request_id,
                req.prompt_token_ids.len()
            );
            prefill_pending.insert(uid, req);
            continue;
        }

        let prompt_len = req.prompt_token_ids.len();

        // Dynamically adjust Metal command buffer limits based on prompt size.
        // Scale down from chip-specific defaults to prevent Metal command buffer
        // timeout on large prompts.
        if prompt_len > 8192 {
            let divisor = if prompt_len > 16384 { 8 } else { 4 };
            let ops = (metal_defaults.ops / divisor).max(2);
            let mb = (metal_defaults.mb / divisor).max(5);
            ironmlx_core::metal::set_max_ops_per_buffer(ops).ok();
            ironmlx_core::metal::set_max_mb_per_buffer(mb).ok();
        }

        let uid = batch.begin_prefill(
            &req.prompt_token_ids,
            req.sampling_params.clone(),
            req.eos_token_id,
            req.max_tokens,
        );
        eprintln!(
            "[engine] enqueue chunked prefill: req={}, prompt_tokens={}, uid={}",
            req.request_id, prompt_len, uid
        );
        prefill_pending.insert(uid, req);
    }
}

/// Process one round of chunked prefill + handle completed sequences.
fn step_prefill(
    batch: &mut BatchGenerator<'_>,
    prefill_pending: &mut PrefillRunning,
    running: &mut HashMap<SeqUid, RunningRequest>,
    tokenizer: &Tokenizer,
    metal_defaults: MetalBufferDefaults,
) {
    if batch.prefilling_count() == 0 {
        return;
    }

    let prefill_t0 = std::time::Instant::now();
    let count = batch.prefilling_count();
    let results = match batch.step_prefill_chunk() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[engine] prefill chunk error: {}", e);
            return;
        }
    };
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[engine] prefill chunk: prefilling={}, completed={}, took={:.1}ms",
        count,
        results.len(),
        prefill_ms
    );

    // Restore chip-specific Metal buffer defaults after prefill completes
    if !results.is_empty() && batch.prefilling_count() == 0 {
        ironmlx_core::metal::set_max_ops_per_buffer(metal_defaults.ops).ok();
        ironmlx_core::metal::set_max_mb_per_buffer(metal_defaults.mb).ok();
    }

    for (uid, response) in results {
        let Some(req) = prefill_pending.remove(&uid) else {
            continue;
        };

        let finish_reason = response.finish_reason.map(|r| match r {
            FinishReason::Eos => "stop".to_string(),
            FinishReason::MaxTokens => "length".to_string(),
        });
        let text = tokenizer.decode(&[response.token_id]).unwrap_or_default();

        let _ = req.token_tx.blocking_send(RequestOutput {
            request_id: req.request_id.clone(),
            token_id: response.token_id,
            token_text: text,
            finish_reason: finish_reason.clone(),
        });

        if finish_reason.is_none() {
            running.insert(
                uid,
                RunningRequest {
                    request_id: req.request_id,
                    token_tx: req.token_tx,
                },
            );
        }
    }
}

fn process_batch_responses(
    responses: Vec<ironmlx_core::generate::BatchResponse>,
    running: &mut HashMap<SeqUid, RunningRequest>,
    batch: &mut BatchGenerator<'_>,
    tokenizer: &Tokenizer,
) {
    for resp in responses {
        let Some(req) = running.get(&resp.uid) else {
            continue;
        };

        let finish_reason = resp.finish_reason.map(|r| match r {
            FinishReason::Eos => "stop".to_string(),
            FinishReason::MaxTokens => "length".to_string(),
        });

        let text = tokenizer.decode(&[resp.token_id]).unwrap_or_default();

        let send_result = req.token_tx.blocking_send(RequestOutput {
            request_id: req.request_id.clone(),
            token_id: resp.token_id,
            token_text: text,
            finish_reason: finish_reason.clone(),
        });

        if send_result.is_err() {
            batch.remove(resp.uid);
            running.remove(&resp.uid);
            continue;
        }

        if finish_reason.is_some() {
            running.remove(&resp.uid);
        }
    }
}
