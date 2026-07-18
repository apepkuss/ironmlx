use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use crate::core::mtp_draft_cap_calibration::{
    calibrate_mtp_draft_cap, MtpDraftCapBenchInput, MtpDraftCapCalibrationConfig,
};

#[derive(Args, Debug)]
pub struct MtpDraftCapArgs {
    /// Benchmark JSON files from configured-cap Gemma4 drafter policy runs.
    #[arg(long, required = true)]
    pub input: Vec<PathBuf>,

    /// Minimum homogeneous windows required for each candidate cap.
    #[arg(long, default_value_t = 32)]
    pub min_windows: usize,

    /// Minimum valid benchmark records required for each candidate cap.
    #[arg(long, default_value_t = 3)]
    pub min_records: usize,

    /// Gain required before selecting a higher cap over a lower cap.
    #[arg(long, default_value_t = 3.0)]
    pub min_improvement_percent: f64,

    /// Write the calibration report to this JSON path instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: MtpDraftCapArgs) -> Result<()> {
    let mut inputs = Vec::with_capacity(args.input.len());
    for path in &args.input {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let input = serde_json::from_slice::<MtpDraftCapBenchInput>(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        inputs.push(input);
    }
    let report = calibrate_mtp_draft_cap(
        inputs,
        MtpDraftCapCalibrationConfig {
            min_windows: args.min_windows,
            min_records: args.min_records,
            min_improvement_percent: args.min_improvement_percent,
        },
    )?;
    let json = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        println!("calibration: {}", path.display());
    } else {
        print!("{json}");
    }
    Ok(())
}
