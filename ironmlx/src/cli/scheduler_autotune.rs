//! `ironmlx scheduler-autotune` — post-process offline calibration results.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Args, ValueEnum};

use crate::core::scheduler_autotune::{
    select_scheduler_autotune_profile, SchedulerAutotuneCalibrationInput,
};
use crate::Result;

#[derive(Args, Debug)]
pub struct SchedulerAutotuneArgs {
    /// JSON file containing offline scheduler calibration measurements.
    #[arg(long)]
    pub input: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneOutputFormat::Text)]
    pub format: SchedulerAutotuneOutputFormat,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAutotuneOutputFormat {
    Text,
    Json,
}

pub fn run(args: SchedulerAutotuneArgs) -> Result<()> {
    let raw = std::fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;
    let input: SchedulerAutotuneCalibrationInput =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", args.input.display()))?;
    let selection = select_scheduler_autotune_profile(input);

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
