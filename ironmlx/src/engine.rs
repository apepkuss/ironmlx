use std::collections::{HashMap, HashSet, VecDeque};

use tokio::sync::mpsc;

use ironmlx_core::cache::CacheManager;
use ironmlx_core::generate::{BatchGenerator, FinishReason, SamplerConfig, SeqUid, Tokenizer};
use ironmlx_core::media::ProcessedMedia;
use ironmlx_core::model::Model;

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
}

impl EngineCore {
    #[allow(dead_code)]
    pub fn new(cmd_rx: mpsc::Receiver<EngineCommand>, model: Model, tokenizer: Tokenizer) -> Self {
        Self {
            cmd_rx,
            model,
            tokenizer,
            cache_manager: None,
        }
    }

    pub fn with_cache_manager(
        cmd_rx: mpsc::Receiver<EngineCommand>,
        model: Model,
        tokenizer: Tokenizer,
        cache_manager: CacheManager,
    ) -> Self {
        Self {
            cmd_rx,
            model,
            tokenizer,
            cache_manager: Some(cache_manager),
        }
    }

    /// Blocking run loop — call from a dedicated thread.
    pub fn run(&mut self) {
        let mut batch = if let Some(cm) = self.cache_manager.take() {
            BatchGenerator::with_cache_manager(&self.model, cm)
        } else {
            BatchGenerator::new(&self.model)
        };
        let mut waiting: VecDeque<PendingRequest> = VecDeque::new();
        let mut running: HashMap<SeqUid, RunningRequest> = HashMap::new();
        let mut pending_aborts: HashSet<String> = HashSet::new();
        let max_num_seqs: usize = 256;

        loop {
            // 1. Drain all pending commands
            match drain_commands(&mut self.cmd_rx, &mut waiting, &mut pending_aborts) {
                DrainResult::Continue => {}
                DrainResult::Shutdown => return,
            }

            // 2. Process pending aborts
            process_aborts(&mut pending_aborts, &mut running, &mut batch);

            // 3. Schedule waiting requests into the batch
            schedule_waiting(
                &mut waiting,
                &mut running,
                &mut batch,
                &self.tokenizer,
                max_num_seqs,
            );

            // 4. If nothing is running, block-wait for next command
            if batch.active_count() == 0 {
                if waiting.is_empty() {
                    match self.cmd_rx.blocking_recv() {
                        Some(EngineCommand::Shutdown) | None => return,
                        Some(cmd) => {
                            handle_command(cmd, &mut waiting, &mut pending_aborts);
                        }
                    }
                    process_aborts(&mut pending_aborts, &mut running, &mut batch);
                    schedule_waiting(
                        &mut waiting,
                        &mut running,
                        &mut batch,
                        &self.tokenizer,
                        max_num_seqs,
                    );
                }

                if batch.active_count() == 0 {
                    continue;
                }
            }

            // 5. Execute one decode step for all active sequences
            match batch.step() {
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

fn schedule_waiting(
    waiting: &mut VecDeque<PendingRequest>,
    running: &mut HashMap<SeqUid, RunningRequest>,
    batch: &mut BatchGenerator<'_>,
    tokenizer: &Tokenizer,
    max_num_seqs: usize,
) {
    while batch.active_count() < max_num_seqs {
        let Some(req) = waiting.pop_front() else {
            break;
        };

        if req.token_tx.is_closed() {
            continue;
        }

        let insert_result = if let Some(ref media) = req.media {
            batch.insert_vlm(
                &req.prompt_token_ids,
                media,
                req.sampling_params,
                req.eos_token_id,
                req.max_tokens,
            )
        } else {
            batch.insert(
                &req.prompt_token_ids,
                req.sampling_params,
                req.eos_token_id,
                req.max_tokens,
            )
        };

        match insert_result {
            Ok((uid, first_response)) => {
                let finish_reason = first_response.finish_reason.map(|r| match r {
                    FinishReason::Eos => "stop".to_string(),
                    FinishReason::MaxTokens => "length".to_string(),
                });
                let text = tokenizer
                    .decode(&[first_response.token_id])
                    .unwrap_or_default();

                let _ = req.token_tx.blocking_send(RequestOutput {
                    request_id: req.request_id.clone(),
                    token_id: first_response.token_id,
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
            Err(e) => {
                eprintln!("Failed to insert request {}: {}", req.request_id, e);
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
