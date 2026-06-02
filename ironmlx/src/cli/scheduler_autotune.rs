//! `ironmlx scheduler-autotune` — post-process offline calibration results.

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Args, Subcommand, ValueEnum};

use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, merge_scheduler_autotune_calibrations,
    select_scheduler_autotune_profile, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneMergeOptions,
};
use crate::Result;

#[derive(Args, Debug)]
pub struct SchedulerAutotuneArgs {
    #[command(subcommand)]
    pub action: Option<SchedulerAutotuneAction>,

    /// JSON file containing offline scheduler calibration measurements.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneOutputFormat::Text)]
    pub format: SchedulerAutotuneOutputFormat,

    /// Write the selected runtime scheduler profile to this JSON path.
    #[arg(long)]
    pub write_profile: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum SchedulerAutotuneAction {
    /// Select a scheduler/autotune profile from one calibration JSON.
    Select(SchedulerAutotuneSelectArgs),
    /// Merge multiple candidate calibration JSON files into one calibration.
    Merge(SchedulerAutotuneMergeArgs),
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneSelectArgs {
    /// JSON file containing offline scheduler calibration measurements.
    #[arg(long)]
    pub input: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneOutputFormat::Text)]
    pub format: SchedulerAutotuneOutputFormat,

    /// Write the selected runtime scheduler profile to this JSON path.
    #[arg(long)]
    pub write_profile: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneMergeArgs {
    /// Candidate calibration JSON files to merge.
    #[arg(long, required = true, num_args = 1..)]
    pub input: Vec<PathBuf>,

    /// Output path for merged calibration JSON. Prints to stdout when omitted.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Allow candidate configs that do not cover the same scenario set.
    #[arg(long)]
    pub allow_incomplete_coverage: bool,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAutotuneOutputFormat {
    Text,
    Json,
}

pub fn run(args: SchedulerAutotuneArgs) -> Result<()> {
    match args.action {
        Some(SchedulerAutotuneAction::Select(select)) => run_select(select),
        Some(SchedulerAutotuneAction::Merge(merge)) => run_merge(merge),
        None => {
            let input = args
                .input
                .context("--input is required unless a scheduler-autotune subcommand is used")?;
            run_select(SchedulerAutotuneSelectArgs {
                input,
                format: args.format,
                write_profile: args.write_profile,
            })
        }
    }
}

fn run_select(args: SchedulerAutotuneSelectArgs) -> Result<()> {
    let input = read_calibration(&args.input)?;
    let selection = select_scheduler_autotune_profile(input);

    if let Some(path) = &args.write_profile {
        let profile = build_scheduler_autotune_runtime_profile(&selection)?;
        let output = serde_json::to_string_pretty(&profile)?;
        std::fs::write(path, format!("{output}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }

    match args.format {
        SchedulerAutotuneOutputFormat::Text => {
            print!("{}", selection.render_text());
        }
        SchedulerAutotuneOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&selection)?);
        }
    }
    Ok(())
}

fn run_merge(args: SchedulerAutotuneMergeArgs) -> Result<()> {
    let mut inputs = Vec::with_capacity(args.input.len());
    for path in &args.input {
        inputs.push(read_calibration(path)?);
    }
    let merged = merge_scheduler_autotune_calibrations(
        inputs,
        SchedulerAutotuneMergeOptions {
            require_complete_coverage: !args.allow_incomplete_coverage,
        },
    )?;
    let output = serde_json::to_string_pretty(&merged)?;

    match args.output {
        Some(path) => {
            std::fs::write(&path, format!("{output}\n"))
                .with_context(|| format!("writing {}", path.display()))?;
        }
        None => {
            println!("{output}");
        }
    }
    Ok(())
}

fn read_calibration(path: &Path) -> Result<SchedulerAutotuneCalibrationInput> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}
