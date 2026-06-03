//! `ironmlx scheduler-autotune` — post-process offline calibration results.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use clap::{Args, Subcommand, ValueEnum};

use super::scheduler_profile_store::SchedulerProfileStore;
use crate::core::scheduler_autotune::{
    build_scheduler_autotune_runtime_profile, merge_scheduler_autotune_calibrations,
    select_scheduler_autotune_profile_with_options, SchedulerAutotuneCalibrationInput,
    SchedulerAutotuneMergeOptions, SchedulerAutotuneRuntimeProfile,
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
        "id\tmodel_name\thardware_label\tstatus\truntime_schema\tironmlx_version\tupdated_at_unix_ms\tmodel_path\tprofile_path"
    );
    for record in records {
        let status = if record.profile_exists {
            "ok"
        } else {
            "missing"
        };
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            record.id,
            record.model_name,
            record.hardware_label,
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{import_profile, SchedulerAutotuneProfileImportArgs};
    use crate::cli::scheduler_profile_store::SchedulerProfileStore;
    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
        SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
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
            store.profile_path("GLM-4.7-Flash-4bit", "test-host", &model_dir)
        );
        let loaded_path = store
            .find_profile(&model_dir, "GLM-4.7-Flash-4bit", "test-host")
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
            config: SchedulerAutotuneProfileConfig {
                b_max: 1,
                prefill_chunk_size: 2048,
                admission_deadline_ms: 5,
                admission_queue_max: 32,
                max_cache_cap: 32768,
                decode_cadence_mid_chunk_cap: 128,
            },
            rules: Vec::new(),
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
