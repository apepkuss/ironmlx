use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};

use crate::core::scheduler_autotune::{
    SchedulerAutotuneRuntimeProfile, SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
};
use crate::Result;

const STORE_SCHEMA_VERSION: u32 = 1;
const INDEX_FILE: &str = "index.json";
const PROFILES_DIR: &str = "profiles";

#[derive(Debug, Clone)]
pub(crate) struct SchedulerProfileStore {
    root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SchedulerProfileStoreIndex {
    schema_version: u32,
    profiles: Vec<SchedulerProfileStoreEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SchedulerProfileStoreEntry {
    id: String,
    model_name: String,
    model_path: String,
    hardware_label: String,
    ironmlx_version: String,
    runtime_schema_version: u32,
    profile_path: PathBuf,
    updated_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerProfileStoreRecord {
    pub(crate) id: String,
    pub(crate) model_name: String,
    pub(crate) model_path: String,
    pub(crate) hardware_label: String,
    pub(crate) ironmlx_version: String,
    pub(crate) runtime_schema_version: u32,
    pub(crate) profile_path: PathBuf,
    pub(crate) profile_exists: bool,
    pub(crate) updated_at_unix_ms: u64,
}

impl SchedulerProfileStore {
    pub(crate) fn default() -> Result<Self> {
        let home = dirs::home_dir().context("locating home directory for ~/.ironmlx")?;
        Ok(Self::from_root(
            home.join(".ironmlx").join("scheduler-profiles"),
        ))
    }

    pub(crate) fn from_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn profile_path(
        &self,
        model_name: &str,
        hardware_label: &str,
        model_path: &Path,
    ) -> PathBuf {
        self.root.join(PROFILES_DIR).join(format!(
            "{}.json",
            profile_id(model_name, hardware_label, model_path)
        ))
    }

    pub(crate) fn persist_profile(
        &self,
        model_path: &Path,
        profile: &SchedulerAutotuneRuntimeProfile,
    ) -> Result<PathBuf> {
        if profile.schema_version != SCHEDULER_AUTOTUNE_SCHEMA_VERSION {
            bail!(
                "scheduler profile schema_version mismatch: expected {}, got {}",
                SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
                profile.schema_version
            );
        }

        let profile_path =
            self.profile_path(&profile.model_name, &profile.hardware_label, model_path);
        let relative_profile_path = profile_path
            .strip_prefix(&self.root)
            .unwrap_or(&profile_path)
            .to_path_buf();
        std::fs::create_dir_all(self.root.join(PROFILES_DIR))
            .with_context(|| format!("creating {}", self.root.join(PROFILES_DIR).display()))?;

        let output = serde_json::to_string_pretty(profile)?;
        std::fs::write(&profile_path, format!("{output}\n"))
            .with_context(|| format!("writing {}", profile_path.display()))?;

        let mut index = self.read_index()?;
        let id = profile_id(&profile.model_name, &profile.hardware_label, model_path);
        index.profiles.retain(|entry| entry.id != id);
        index.profiles.push(SchedulerProfileStoreEntry {
            id,
            model_name: profile.model_name.clone(),
            model_path: normalized_model_path(model_path),
            hardware_label: profile.hardware_label.clone(),
            ironmlx_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_schema_version: profile.schema_version,
            profile_path: relative_profile_path,
            updated_at_unix_ms: unix_time_ms(),
        });
        self.write_index(&index)?;

        Ok(profile_path)
    }

    pub(crate) fn find_profile(
        &self,
        model_path: &Path,
        model_name: &str,
        hardware_label: &str,
    ) -> Result<Option<PathBuf>> {
        let index = self.read_index()?;
        let normalized_model_path = normalized_model_path(model_path);
        Ok(index
            .profiles
            .iter()
            .filter(|entry| {
                entry.hardware_label == hardware_label
                    && entry.runtime_schema_version == SCHEDULER_AUTOTUNE_SCHEMA_VERSION
                    && (entry.model_path == normalized_model_path || entry.model_name == model_name)
                    && self.root.join(&entry.profile_path).exists()
            })
            .max_by_key(|entry| entry.updated_at_unix_ms)
            .map(|entry| self.root.join(&entry.profile_path)))
    }

    pub(crate) fn list_profiles(&self) -> Result<Vec<SchedulerProfileStoreRecord>> {
        let index = self.read_index()?;
        let mut records = index
            .profiles
            .iter()
            .map(|entry| self.entry_to_record(entry))
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            right
                .updated_at_unix_ms
                .cmp(&left.updated_at_unix_ms)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(records)
    }

    pub(crate) fn read_profile(&self, id: &str) -> Result<Option<SchedulerAutotuneRuntimeProfile>> {
        let Some(record) = self.record_by_id(id)? else {
            return Ok(None);
        };
        if !record.profile_path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&record.profile_path)
            .with_context(|| format!("reading {}", record.profile_path.display()))?;
        let profile = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", record.profile_path.display()))?;
        Ok(Some(profile))
    }

    pub(crate) fn remove_profile(&self, id: &str) -> Result<Option<SchedulerProfileStoreRecord>> {
        let mut index = self.read_index()?;
        let Some(position) = index.profiles.iter().position(|entry| entry.id == id) else {
            return Ok(None);
        };
        let entry = index.profiles.remove(position);
        let record = self.entry_to_record(&entry);
        match std::fs::remove_file(&record.profile_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("removing {}", record.profile_path.display()));
            }
        }
        self.write_index(&index)?;
        Ok(Some(record))
    }

    fn index_path(&self) -> PathBuf {
        self.root.join(INDEX_FILE)
    }

    fn read_index(&self) -> Result<SchedulerProfileStoreIndex> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(SchedulerProfileStoreIndex {
                schema_version: STORE_SCHEMA_VERSION,
                profiles: Vec::new(),
            });
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let index: SchedulerProfileStoreIndex =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        if index.schema_version != STORE_SCHEMA_VERSION {
            bail!(
                "scheduler profile store schema_version mismatch: expected {}, got {}",
                STORE_SCHEMA_VERSION,
                index.schema_version
            );
        }
        Ok(index)
    }

    fn write_index(&self, index: &SchedulerProfileStoreIndex) -> Result<()> {
        std::fs::create_dir_all(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        let output = serde_json::to_string_pretty(index)?;
        let path = self.index_path();
        std::fs::write(&path, format!("{output}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn record_by_id(&self, id: &str) -> Result<Option<SchedulerProfileStoreRecord>> {
        let index = self.read_index()?;
        Ok(index
            .profiles
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| self.entry_to_record(entry)))
    }

    fn entry_to_record(&self, entry: &SchedulerProfileStoreEntry) -> SchedulerProfileStoreRecord {
        let profile_path = self.root.join(&entry.profile_path);
        SchedulerProfileStoreRecord {
            id: entry.id.clone(),
            model_name: entry.model_name.clone(),
            model_path: entry.model_path.clone(),
            hardware_label: entry.hardware_label.clone(),
            ironmlx_version: entry.ironmlx_version.clone(),
            runtime_schema_version: entry.runtime_schema_version,
            profile_exists: profile_path.exists(),
            profile_path,
            updated_at_unix_ms: entry.updated_at_unix_ms,
        }
    }
}

pub(crate) fn detect_scheduler_profile_hardware_label() -> String {
    hardware_label_from_parts(detect_cpu_label().as_deref(), detect_total_ram_bytes())
}

fn profile_id(model_name: &str, hardware_label: &str, model_path: &Path) -> String {
    format!(
        "{}--{}--{}",
        slugify_component(model_name),
        slugify_component(hardware_label),
        stable_hex_hash(&normalized_model_path(model_path))
    )
}

fn normalized_model_path(model_path: &Path) -> String {
    model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn unix_time_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_millis();
    millis.min(u128::from(u64::MAX)) as u64
}

fn hardware_label_from_parts(cpu_label: Option<&str>, total_ram_bytes: Option<u64>) -> String {
    let cpu = cpu_label
        .map(slugify_component)
        .filter(|label| !label.is_empty())
        .unwrap_or_else(|| slugify_component(std::env::consts::ARCH));

    match total_ram_bytes {
        Some(bytes) => format!("{cpu}-{}gb", rounded_gib(bytes)),
        None => cpu,
    }
}

fn slugify_component(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_separator = false;

    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_separator = false;
        } else if !last_was_separator && !slug.is_empty() {
            slug.push('-');
            last_was_separator = true;
        }
    }

    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        "unknown".to_string()
    } else {
        slug
    }
}

fn rounded_gib(bytes: u64) -> u64 {
    let gib = 1024_u64.pow(3);
    ((bytes + gib / 2) / gib).max(1)
}

fn stable_hex_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(target_os = "macos")]
fn detect_cpu_label() -> Option<String> {
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| command_output("sysctl", &["-n", "hw.model"]))
}

#[cfg(target_os = "linux")]
fn detect_cpu_label() -> Option<String> {
    let raw = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    raw.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name")
            .then(|| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_cpu_label() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn detect_total_ram_bytes() -> Option<u64> {
    command_output("sysctl", &["-n", "hw.memsize"])?
        .parse::<u64>()
        .ok()
}

#[cfg(target_os = "linux")]
fn detect_total_ram_bytes() -> Option<u64> {
    let raw = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return kb.checked_mul(1024);
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_total_ram_bytes() -> Option<u64> {
    None
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::scheduler_autotune::{
        SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    };

    use super::{hardware_label_from_parts, stable_hex_hash, SchedulerProfileStore};

    #[test]
    fn hardware_label_from_parts_slugifies_cpu_and_memory() {
        assert_eq!(
            hardware_label_from_parts(Some("Apple M5 Max"), Some(128 * 1024_u64.pow(3))),
            "apple-m5-max-128gb"
        );
    }

    #[test]
    fn stable_hex_hash_is_deterministic() {
        assert_eq!(stable_hex_hash("/tmp/model"), stable_hex_hash("/tmp/model"));
        assert_ne!(
            stable_hex_hash("/tmp/model-a"),
            stable_hex_hash("/tmp/model-b")
        );
    }

    #[test]
    fn list_profiles_returns_profile_metadata() {
        let temp_dir = unique_temp_dir("scheduler-profile-list");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        let stored_path = store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");

        let records = store.list_profiles().expect("list profiles");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model_name, "test-model");
        assert_eq!(records[0].hardware_label, "test-host");
        assert_eq!(records[0].profile_path, stored_path);
        assert!(records[0].profile_exists);
        assert!(!records[0].id.is_empty());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    #[test]
    fn read_and_remove_profile_by_id() {
        let temp_dir = unique_temp_dir("scheduler-profile-remove");
        let model_dir = temp_dir.join("GLM-4.7-Flash-4bit");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        let store = SchedulerProfileStore::from_root(temp_dir.join("store"));
        let stored_path = store
            .persist_profile(&model_dir, &runtime_profile())
            .expect("persist profile");
        let id = store.list_profiles().expect("list profiles")[0].id.clone();

        let loaded = store
            .read_profile(&id)
            .expect("read profile")
            .expect("stored profile should exist");
        assert_eq!(loaded.model_name, "test-model");

        let removed = store
            .remove_profile(&id)
            .expect("remove profile")
            .expect("profile should be removed");
        assert_eq!(removed.id, id);
        assert!(!stored_path.exists());
        assert!(store
            .read_profile(&removed.id)
            .expect("read removed profile")
            .is_none());

        std::fs::remove_dir_all(temp_dir).expect("cleanup temp dir");
    }

    fn runtime_profile() -> SchedulerAutotuneRuntimeProfile {
        SchedulerAutotuneRuntimeProfile {
            schema_version: crate::core::scheduler_autotune::SCHEDULER_AUTOTUNE_SCHEMA_VERSION,
            model_name: "test-model".to_string(),
            hardware_label: "test-host".to_string(),
            config: SchedulerAutotuneProfileConfig {
                b_max: 2,
                prefill_chunk_size: 1024,
                admission_deadline_ms: 7,
                admission_queue_max: 16,
                max_cache_cap: 8192,
                decode_cadence_mid_chunk_cap: 384,
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
