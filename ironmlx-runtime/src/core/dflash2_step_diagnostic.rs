//! Default-off DFlash2 actor step diagnostic.
//!
//! With `IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG=<path>` the DFlash2 actor appends
//! one JSON line per batch step: the step width, the wall time of each step
//! phase, the ragged linear window it ran (rows, cache build, window time and,
//! with `IRONMLX_DIAGNOSTIC_DFLASH2_WINDOW_STAGES=1`, its synchronized stages)
//! and the drained tree-window stage records. Without the stage switch the
//! phase times show where the host waits; GPU work issued asynchronously is
//! attributed to the phase that first waits for it. Unset, nothing is
//! recorded and no clock is read.

use std::io::Write;
use std::time::Instant;

const STEP_LOG_ENV: &str = "IRONMLX_DIAGNOSTIC_DFLASH2_STEP_LOG";

pub(crate) struct DFlash2StepDiagnostic {
    file: std::fs::File,
    step: u64,
    current: Option<StepRecord>,
}

struct StepRecord {
    width: usize,
    started: Instant,
    last: Instant,
    phases: Vec<(&'static str, u64)>,
    ragged: Option<serde_json::Value>,
    notes: Vec<(&'static str, u64)>,
}

impl DFlash2StepDiagnostic {
    pub(crate) fn from_env() -> Option<Self> {
        let path = std::env::var_os(STEP_LOG_ENV)?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => {
                tracing::warn!(
                    target: "ironmlx::dflash2",
                    path = %std::path::Path::new(&path).display(),
                    "DFlash2 step diagnostic enabled"
                );
                Some(Self {
                    file,
                    step: 0,
                    current: None,
                })
            }
            Err(error) => {
                tracing::error!(%error, "DFlash2 step diagnostic file cannot be opened");
                None
            }
        }
    }

    pub(crate) fn begin(this: &mut Option<Self>, width: usize) {
        if let Some(this) = this {
            let now = Instant::now();
            this.current = Some(StepRecord {
                width,
                started: now,
                last: now,
                phases: Vec::new(),
                ragged: None,
                notes: Vec::new(),
            });
        }
    }

    pub(crate) fn mark(this: &mut Option<Self>, phase: &'static str) {
        if let Some(record) = this.as_mut().and_then(|this| this.current.as_mut()) {
            let now = Instant::now();
            record.phases.push((phase, micros(now - record.last)));
            record.last = now;
        }
    }

    pub(crate) fn note(this: &mut Option<Self>, name: &'static str, value: u64) {
        if let Some(record) = this.as_mut().and_then(|this| this.current.as_mut()) {
            record.notes.push((name, value));
        }
    }

    pub(crate) fn ragged(
        this: &mut Option<Self>,
        built: bool,
        timing: &super::dflash2::DFlash2RaggedWindowTiming,
    ) {
        if let Some(record) = this.as_mut().and_then(|this| this.current.as_mut()) {
            record.ragged = Some(serde_json::json!({
                "built": built,
                "timing": timing,
            }));
        }
    }

    pub(crate) fn finish(this: &mut Option<Self>) {
        let Some(this) = this.as_mut() else {
            return;
        };
        let Some(record) = this.current.take() else {
            return;
        };
        this.step += 1;
        let windows = super::dflash2::take_window_stage_records();
        let line = serde_json::json!({
            "step": this.step,
            "width": record.width,
            "total_us": micros(record.started.elapsed()),
            "phases": record.phases,
            "notes": record.notes,
            "ragged": record.ragged,
            "tree_windows": windows,
        });
        if let Err(error) = writeln!(this.file, "{line}") {
            tracing::error!(%error, "DFlash2 step diagnostic write failed");
        }
    }
}

fn micros(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
