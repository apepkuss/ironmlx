//! `ironmlx scheduler-autotune` — post-process offline calibration results.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use clap::{Args, Subcommand, ValueEnum};
use serde::Serialize;

use super::scheduler_profile_context::SchedulerProfileRuntimeArgs;
use super::scheduler_profile_store::{
    detect_scheduler_profile_hardware_label, SchedulerProfileStore,
};
use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, evaluate_scheduler_autotune_profile_health,
    merge_scheduler_autotune_calibrations, select_scheduler_autotune_profile_with_options,
    SchedulerAutotuneCalibrationInput, SchedulerAutotuneMergeOptions,
    SchedulerAutotuneProfileHealthInput, SchedulerAutotuneProfileHealthReport,
    SchedulerAutotuneProfileHealthStatus, SchedulerAutotuneRuntimeProfile,
    SchedulerAutotuneSelectionOptions, SchedulerAutotuneSelectionProfile,
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

    /// Profile used to weight calibration scenarios during selection.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneSelectionProfileArg::AgentLongPrompt)]
    pub selection_profile: SchedulerAutotuneSelectionProfileArg,

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
    /// Run local scheduler/autotune calibration candidates and write a profile.
    Calibrate(Box<super::scheduler_autotune_calibrate::SchedulerAutotuneCalibrateArgs>),
    /// Inspect or remove persisted local scheduler profiles.
    Profile(SchedulerAutotuneProfileArgs),
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneSelectArgs {
    /// JSON file containing offline scheduler calibration measurements.
    #[arg(long)]
    pub input: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneOutputFormat::Text)]
    pub format: SchedulerAutotuneOutputFormat,

    /// Profile used to weight calibration scenarios during selection.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneSelectionProfileArg::AgentLongPrompt)]
    pub selection_profile: SchedulerAutotuneSelectionProfileArg,

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

#[derive(Args, Debug)]
pub struct SchedulerAutotuneProfileArgs {
    #[command(subcommand)]
    pub action: SchedulerAutotuneProfileAction,
}

#[derive(Subcommand, Debug)]
pub enum SchedulerAutotuneProfileAction {
    /// List persisted scheduler profiles in ~/.ironmlx.
    List,
    /// Print one persisted scheduler profile as JSON.
    Show(SchedulerAutotuneProfileShowArgs),
    /// Diagnose the matching local scheduler profile for one model.
    Doctor(SchedulerAutotuneProfileDoctorArgs),
    /// Remove one persisted scheduler profile and its JSON file.
    Remove(SchedulerAutotuneProfileRemoveArgs),
    /// Import one runtime scheduler profile into ~/.ironmlx for one model path.
    Import(SchedulerAutotuneProfileImportArgs),
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneProfileShowArgs {
    /// Profile id from `scheduler-autotune profile list`.
    pub id: String,
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneProfileDoctorArgs {
    /// Local model directory to match against ~/.ironmlx scheduler profiles.
    #[arg(long)]
    pub model: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = SchedulerAutotuneOutputFormat::Text)]
    pub format: SchedulerAutotuneOutputFormat,

    /// Maximum accepted profile age before warning.
    #[arg(long, default_value_t = 30)]
    pub max_age_days: u64,

    #[command(flatten)]
    pub runtime: SchedulerProfileRuntimeArgs,
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneProfileRemoveArgs {
    /// Profile id from `scheduler-autotune profile list`.
    pub id: String,
}

#[derive(Args, Debug)]
pub struct SchedulerAutotuneProfileImportArgs {
    /// Local model directory this runtime profile applies to.
    #[arg(long)]
    pub model: PathBuf,

    /// Runtime scheduler profile JSON to import.
    #[arg(long)]
    pub profile: PathBuf,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAutotuneOutputFormat {
    Text,
    Json,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAutotuneSelectionProfileArg {
    Balanced,
    AgentLongPrompt,
}

impl From<SchedulerAutotuneSelectionProfileArg> for SchedulerAutotuneSelectionProfile {
    fn from(value: SchedulerAutotuneSelectionProfileArg) -> Self {
        match value {
            SchedulerAutotuneSelectionProfileArg::Balanced => Self::Balanced,
            SchedulerAutotuneSelectionProfileArg::AgentLongPrompt => Self::AgentLongPrompt,
        }
    }
}

pub fn run(args: SchedulerAutotuneArgs) -> Result<()> {
    match args.action {
        Some(SchedulerAutotuneAction::Select(select)) => run_select(select),
        Some(SchedulerAutotuneAction::Merge(merge)) => run_merge(merge),
        Some(SchedulerAutotuneAction::Calibrate(calibrate)) => {
            super::scheduler_autotune_calibrate::run(*calibrate)
        }
        Some(SchedulerAutotuneAction::Profile(profile)) => run_profile(profile),
        None => {
            let input = args
                .input
                .context("--input is required unless a scheduler-autotune subcommand is used")?;
            run_select(SchedulerAutotuneSelectArgs {
                input,
                format: args.format,
                selection_profile: args.selection_profile,
                write_profile: args.write_profile,
            })
        }
    }
}

fn run_profile(args: SchedulerAutotuneProfileArgs) -> Result<()> {
    let store = SchedulerProfileStore::default()?;
    match args.action {
        SchedulerAutotuneProfileAction::List => run_profile_list(&store),
        SchedulerAutotuneProfileAction::Show(show) => run_profile_show(&store, show),
        SchedulerAutotuneProfileAction::Doctor(doctor) => run_profile_doctor(&store, doctor),
        SchedulerAutotuneProfileAction::Remove(remove) => run_profile_remove(&store, remove),
        SchedulerAutotuneProfileAction::Import(import) => run_profile_import(&store, import),
    }
}

fn run_profile_list(store: &SchedulerProfileStore) -> Result<()> {
    let records = store.list_profiles()?;
    println!("store: {}", store.root().display());
    println!("profiles: {}", records.len());
    if records.is_empty() {
        return Ok(());
    }
    println!(
        "id\tmodel_name\thardware_label\truntime_context\tstatus\truntime_schema\tironmlx_version\tupdated_at_unix_ms\tmodel_path\tprofile_path"
    );
    for record in records {
        let status = if record.profile_exists {
            "ok"
        } else {
            "missing"
        };
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            record.id,
            record.model_name,
            record.hardware_label,
            record.runtime_context_fingerprint,
            status,
            record.runtime_schema_version,
            record.ironmlx_version,
            record.updated_at_unix_ms,
            record.model_path,
            record.profile_path.display()
        );
    }
    Ok(())
}

fn run_profile_show(
    store: &SchedulerProfileStore,
    args: SchedulerAutotuneProfileShowArgs,
) -> Result<()> {
    let Some(profile) = store.read_profile(&args.id)? else {
        bail!(
            "scheduler profile not found or profile file missing: {}",
            args.id
        );
    };
    println!("{}", serde_json::to_string_pretty(&profile)?);
    Ok(())
}

fn run_profile_doctor(
    store: &SchedulerProfileStore,
    args: SchedulerAutotuneProfileDoctorArgs,
) -> Result<()> {
    let model_name = profile_model_name(&args.model)?;
    let hardware_label = detect_scheduler_profile_hardware_label();
    let runtime_context = args.runtime.context_for_model(&args.model)?;
    let runtime_context_fingerprint = runtime_context.fingerprint();
    let Some(profile_path) =
        store.find_profile(&args.model, &hardware_label, &runtime_context_fingerprint)?
    else {
        bail!(
            "no matching scheduler profile found for model={} model_name={} hardware_label={} store={}",
            args.model.display(),
            model_name,
            hardware_label,
            store.root().display()
        );
    };
    let profile = read_runtime_profile(&profile_path)?;
    let report = evaluate_scheduler_autotune_profile_health(SchedulerAutotuneProfileHealthInput {
        profile: &profile,
        expected_model_name: &model_name,
        expected_hardware_label: &hardware_label,
        expected_runtime_context: &runtime_context,
        current_ironmlx_version: env!("CARGO_PKG_VERSION"),
        now_unix_ms: unix_time_ms(),
        max_age_days: args.max_age_days,
    });
    let recalibrate_command = recalibrate_command(&args.model, report.status);

    match args.format {
        SchedulerAutotuneOutputFormat::Text => {
            println!("store: {}", store.root().display());
            println!("profile_path: {}", profile_path.display());
            print!("{}", report.render_text());
            if let Some(command) = recalibrate_command {
                println!("recalibrate: {command}");
            }
        }
        SchedulerAutotuneOutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&SchedulerAutotuneProfileDoctorOutput {
                    store: store.root().display().to_string(),
                    profile_path: profile_path.display().to_string(),
                    report,
                    recalibrate_command,
                })?
            );
        }
    }

    Ok(())
}

fn run_profile_remove(
    store: &SchedulerProfileStore,
    args: SchedulerAutotuneProfileRemoveArgs,
) -> Result<()> {
    let Some(record) = store.remove_profile(&args.id)? else {
        bail!("scheduler profile not found: {}", args.id);
    };
    println!(
        "removed: {} model_name={} hardware_label={} profile_path={}",
        record.id,
        record.model_name,
        record.hardware_label,
        record.profile_path.display()
    );
    Ok(())
}

fn run_profile_import(
    store: &SchedulerProfileStore,
    args: SchedulerAutotuneProfileImportArgs,
) -> Result<()> {
    let path = import_profile(store, args)?;
    println!("imported: {}", path.display());
    Ok(())
}

fn import_profile(
    store: &SchedulerProfileStore,
    args: SchedulerAutotuneProfileImportArgs,
) -> Result<PathBuf> {
    let raw = std::fs::read_to_string(&args.profile)
        .with_context(|| format!("reading {}", args.profile.display()))?;
    let profile: SchedulerAutotuneRuntimeProfile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing {}", args.profile.display()))?;
    store.persist_profile(&args.model, &profile)
}

#[derive(Debug, Serialize)]
struct SchedulerAutotuneProfileDoctorOutput {
    store: String,
    profile_path: String,
    report: SchedulerAutotuneProfileHealthReport,
    recalibrate_command: Option<String>,
}

fn run_select(args: SchedulerAutotuneSelectArgs) -> Result<()> {
    let input = read_calibration(&args.input)?;
    let selection = select_scheduler_autotune_profile_with_options(
        input,
        SchedulerAutotuneSelectionOptions {
            profile: args.selection_profile.into(),
        },
    );

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

fn read_runtime_profile(path: &Path) -> Result<SchedulerAutotuneRuntimeProfile> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn profile_model_name(model: &Path) -> Result<String> {
    model
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            anyhow::anyhow!("--model has no directory name for scheduler profile lookup")
        })
}

fn recalibrate_command(
    model: &Path,
    status: SchedulerAutotuneProfileHealthStatus,
) -> Option<String> {
    if status == SchedulerAutotuneProfileHealthStatus::Healthy {
        return None;
    }
    Some(format!(
        "ironmlx scheduler-autotune calibrate --model {}",
        model.display()
    ))
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{import_profile, SchedulerAutotuneProfileImportArgs};
    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeContext,
        SchedulerAutotuneRuntimeProfile, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
    };

    #[test]
    fn import_profile_persists_runtime_profile_for_model() {
        let temp_dir = unique_temp_dir("scheduler-profile-import");
        let model_dir = temp_dir.join("model");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let profile_path = temp_dir.join("scheduler-profile.json");
        let profile = runtime_profile();
        let output = serde_json::to_string_pretty(&profile).expect("serialize profile");
        std::fs::write(&profile_path, format!("{output}\n")).expect("write profile");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));

        let stored = import_profile(
            &store,
            SchedulerAutotuneProfileImportArgs {
                model: model_dir.clone(),
                profile: profile_path,
            },
        )
        .expect("import profile");

        assert_eq!(
            stored,
            store.profile_path(
                "GLM-4.7-Flash-4bit",
                "test-host",
                profile.metadata.selection_profile,
                &model_dir,
                &profile.runtime_context.fingerprint(),
            )
        );
        let loaded_path = store
            .find_profile(
                &model_dir,
                "test-host",
                &profile.runtime_context.fingerprint(),
            )
            .expect("find profile")
            .expect("profile should exist");
        assert_eq!(loaded_path, stored);

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "GLM-4.7-Flash-4bit".to_string(),
            hardware_label: "test-host".to_string(),
            runtime_context: SchedulerAutotuneRuntimeContext::local_default(32768),
            config: SchedulerAutotuneProfileConfig {
                b_max: 1,
                prefill_chunk_size: 2048,
                admission_deadline_ms: 5,
                admission_queue_max: 32,
                max_cache_cap: 32768,
                decode_cadence_mid_chunk_cap: 128,
            },
            rules: Vec::new(),
            metadata:
                crate::core::scheduler_autotune::SchedulerAutotuneRuntimeProfileMetadata::synthetic(
                    1811606400000,
                ),
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }
}
