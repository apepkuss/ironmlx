//! Default DFlash2 serving profile for Apple GPU generation 17+ (M5).
//!
//! On a g17+ GPU, DFlash2 serving enables the qualified candidate settings
//! below (affine4 tensor-unit QMM, lane attention, flat tf-v1 tree, fixed
//! draft budget, ragged linear batching, prefill kernels, ...). Each feature
//! keeps its own quantization, shape, request and memory checks and falls
//! back to the ordinary path when they do not hold. Other GPUs, other serving
//! modes and `--m5-dflash2-profile off` keep the previous defaults.
//!
//! Every setting is still keyed by its former `IRONMLX_EXPERIMENTAL_*`
//! variable: an explicitly set variable wins over the profile (`"1"` on, any
//! other value off), so individual features can be turned off or tested.
//!
//! [`install`] must run once at startup, before model loading; reads before
//! it see an inactive profile.

use std::sync::OnceLock;

/// Former experiment variables, now profile settings.
pub mod settings {
    pub const M5_DFLASH2_QMM: &str = "IRONMLX_EXPERIMENTAL_M5_DFLASH2_QMM";
    pub const M5_LANE_ATTN: &str = "IRONMLX_EXPERIMENTAL_M5_LANE_ATTN";
    pub const M5_ATTN_GROUP_TILES: &str = "IRONMLX_EXPERIMENTAL_M5_ATTN_GROUP_TILES";
    pub const M5_SHARED_WEIGHT_LAYOUT: &str = "IRONMLX_EXPERIMENTAL_M5_SHARED_WEIGHT_LAYOUT";
    pub const M5_PRECOMPILE: &str = "IRONMLX_EXPERIMENTAL_M5_PRECOMPILE";
    pub const DFLASH2_FIXED_BUDGET: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_FIXED_BUDGET";
    pub const DFLASH2_SINGLE_PREFILL: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_SINGLE_PREFILL";
    pub const DFLASH2_RADIX_TOPK: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RADIX_TOPK";
    pub const DFLASH2_FLAT_TREE: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_FLAT_TREE";
    pub const DFLASH2_TREE_PROFILE: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_TREE_PROFILE";
    pub const DFLASH2_RAGGED_LINEAR: &str = "IRONMLX_EXPERIMENTAL_DFLASH2_RAGGED_LINEAR";
    pub const PREFILL_MASKED_SOFTMAX: &str = "IRONMLX_EXPERIMENTAL_PREFILL_MASKED_SOFTMAX";
    pub const PREFILL_CHUNK_CACHE_RESET: &str = "IRONMLX_EXPERIMENTAL_PREFILL_CHUNK_CACHE_RESET";
    pub const PREFILL_QMM_MTILE_METALLIB: &str = "IRONMLX_EXPERIMENTAL_PREFILL_QMM_MTILE_METALLIB";
    pub const PREFILL_D256_NAX_METALLIB: &str = "IRONMLX_EXPERIMENTAL_PREFILL_D256_NAX_METALLIB";
    pub const MLX_CACHE_MAX_MIB: &str = "IRONMLX_EXPERIMENTAL_MLX_CACHE_MAX_MIB";
    pub const QEMBEDDING_RUNTIME_COUNT: &str = "IRONMLX_EXPERIMENTAL_QEMBEDDING_RUNTIME_COUNT";
    /// Wire the loaded DFlash2 weights (MLX residency sets) at serve start.
    pub const WIRED_WEIGHTS: &str = "IRONMLX_EXPERIMENTAL_WIRED_WEIGHTS";
    /// Qwen3.6 MoE: expert-grouped gather qmv for multi-token verification.
    pub const M5_MOE_GROUPED_QMV: &str = "IRONMLX_EXPERIMENTAL_M5_MOE_GROUPED_QMV";
    /// Qwen3.6 MoE affine4: route non-expert projections through the M5
    /// affine4 tensor-unit kernel (changes reduction order versus MLX).
    pub const M5_MOE_AFFINE4_PROJECTIONS: &str = "IRONMLX_EXPERIMENTAL_M5_MOE_AFFINE4_PROJECTIONS";
}

/// Model family a DFlash2 profile is qualified for. Each family carries its
/// own settings table; a family never inherits another family's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileTarget {
    /// Dense Qwen3.5-family targets (the Qwen3.8-27B lane qualification).
    Qwen35Dense,
    /// Qwen3.6-35B-A3B MoE targets.
    Qwen36Moe,
    /// No qualified profile: every setting stays off.
    Unqualified,
}

impl M5ProfileTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35Dense => "qwen35-dense",
            Self::Qwen36Moe => "qwen36-moe",
            Self::Unqualified => "unqualified",
        }
    }

    fn values(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Qwen35Dense => PROFILE_VALUES,
            Self::Qwen36Moe => MOE_PROFILE_VALUES,
            Self::Unqualified => &[],
        }
    }

    /// Experimental settings outside the profile table that still change
    /// this family's execution when set explicitly; reported for diagnosis.
    fn experimental_settings(self) -> &'static [&'static str] {
        match self {
            Self::Qwen36Moe => &[
                settings::M5_MOE_GROUPED_QMV,
                settings::M5_MOE_AFFINE4_PROJECTIONS,
            ],
            Self::Qwen35Dense | Self::Unqualified => &[],
        }
    }
}

/// Qwen3.6 MoE profile values. Only settings whose MoE correctness and
/// end-to-end benefit were measured are listed. The expert-grouped gather
/// qmv is exact but runs only at 8+ verify rows: on M5 Max (same-process A/B,
/// affine6/affine8) adaptive linear windows showed no measurable gain and the
/// wide shapes that use it gained at most ~2.6% while staying slower than
/// adaptive linear. The affine4 projections were slower. Both stay explicit
/// experimental settings. The runtime-count quantized embedding has identical
/// arithmetic and removes a Metal library compile on every new prompt length
/// (affine4 short prompts: six-task outputs unchanged, first-use prefill
/// spikes of 35-116 ms gone). A 512 MiB MLX buffer-cache ceiling (default
/// 2 GiB) lowered the short-prompt serving footprint by 1.0 GiB with unchanged
/// window costs; long-prompt prefill was not measured at this ceiling. Wired
/// weights removed the intermittent 40-80 ms slower prefills after the idle
/// gap between requests (0 of 48 against 6 of 24 in the paired sessions).
const MOE_PROFILE_VALUES: &[(&str, &str)] = &[
    (settings::QEMBEDDING_RUNTIME_COUNT, "1"),
    (settings::MLX_CACHE_MAX_MIB, "512"),
    (settings::WIRED_WEIGHTS, "1"),
];

/// Default flat-tree size of the Qwen3.6 MoE profile when the option is not
/// given explicitly. Linear windows: tree7/tree15 measured slower than linear
/// DFlash2 on every affine width (and below ordinary decoding on 5/6/8-bit).
pub const MOE_DFLASH2_TREE_MAX_NODES: usize = 0;

/// Profile values of the settings (the qualified candidate configuration).
/// The prefill kernel libraries are set to the embedded copies at install.
const PROFILE_VALUES: &[(&str, &str)] = &[
    (settings::M5_DFLASH2_QMM, "1"),
    (settings::M5_LANE_ATTN, "1"),
    (settings::M5_ATTN_GROUP_TILES, "4"),
    (settings::M5_SHARED_WEIGHT_LAYOUT, "1"),
    (settings::M5_PRECOMPILE, "1"),
    (settings::DFLASH2_FIXED_BUDGET, "7"),
    (settings::DFLASH2_SINGLE_PREFILL, "1"),
    (settings::DFLASH2_RADIX_TOPK, "1"),
    (settings::DFLASH2_FLAT_TREE, "1"),
    (settings::DFLASH2_TREE_PROFILE, "tf-v1"),
    (settings::DFLASH2_RAGGED_LINEAR, "1"),
    (settings::PREFILL_MASKED_SOFTMAX, "1"),
    (settings::PREFILL_CHUNK_CACHE_RESET, "1"),
    (settings::MLX_CACHE_MAX_MIB, "4096"),
    (settings::QEMBEDDING_RUNTIME_COUNT, "1"),
];

/// DFlash2 serve defaults of the profile, applied only when the
/// corresponding option is not given explicitly.
pub const DFLASH2_TREE_MAX_NODES: usize = 15;
pub const DFLASH2_ADMISSION_DEADLINE_MS: u64 = 0;

/// Minimum Apple GPU generation of the profile (M5).
pub const MIN_GPU_GENERATION: u32 = 17;

/// Diagnostic only: pretend the GPU has this architecture name when
/// resolving the profile (for fallback tests on M5 hardware).
pub const ARCHITECTURE_OVERRIDE_ENV: &str = "IRONMLX_DIAGNOSTIC_M5_PROFILE_ARCHITECTURE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileMode {
    Auto,
    Off,
}

impl M5ProfileMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for M5ProfileMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            other => Err(format!("expected `auto` or `off`, got `{other}`")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M5ProfileStatus {
    Active,
    /// `--m5-dflash2-profile off`.
    Disabled,
    /// Not DFlash2 serving.
    NotDFlash2,
    /// The GPU is older than generation 17.
    UnsupportedGpu,
    /// The GPU architecture could not be read.
    GpuUnknown,
    /// DFlash2 serving whose target model family was not qualified with the
    /// profile settings (they were tuned for the dense Qwen3.8 lane).
    TargetNotQualified,
}

impl M5ProfileStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Disabled => "disabled",
            Self::NotDFlash2 => "not_dflash2",
            Self::UnsupportedGpu => "unsupported_gpu",
            Self::GpuUnknown => "gpu_unknown",
            Self::TargetNotQualified => "target_not_qualified",
        }
    }
}

/// Apple GPU generation from an architecture name such as `applegpu_g17s`.
pub fn gpu_generation(architecture: &str) -> Option<u32> {
    architecture
        .strip_prefix("applegpu_g")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Pure profile decision.
pub fn resolve(mode: M5ProfileMode, dflash2: bool, architecture: Option<&str>) -> M5ProfileStatus {
    resolve_for_target(mode, dflash2, true, architecture)
}

/// Pure profile decision for a DFlash2 target whose model family may not be
/// covered by the profile qualification.
pub fn resolve_for_target(
    mode: M5ProfileMode,
    dflash2: bool,
    target_qualified: bool,
    architecture: Option<&str>,
) -> M5ProfileStatus {
    if mode == M5ProfileMode::Off {
        return M5ProfileStatus::Disabled;
    }
    if !dflash2 {
        return M5ProfileStatus::NotDFlash2;
    }
    if !target_qualified {
        return M5ProfileStatus::TargetNotQualified;
    }
    match architecture.map(gpu_generation) {
        None | Some(None) => M5ProfileStatus::GpuUnknown,
        Some(Some(generation)) if generation >= MIN_GPU_GENERATION => M5ProfileStatus::Active,
        Some(Some(_)) => M5ProfileStatus::UnsupportedGpu,
    }
}

/// Outcome of configuring one prefill kernel library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillLibrary {
    pub name: &'static str,
    /// `"profile"` (embedded copy), `"environment"` (explicit path) or
    /// `"off"`.
    pub source: &'static str,
    pub path: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct M5Profile {
    pub mode: M5ProfileMode,
    pub architecture: Option<String>,
    pub status: M5ProfileStatus,
    pub target: M5ProfileTarget,
    pub prefill_libraries: Vec<PrefillLibrary>,
}

impl M5Profile {
    pub fn is_active(&self) -> bool {
        self.status == M5ProfileStatus::Active
    }

    /// Effective value of every profile setting, followed by the family's
    /// experimental settings (`None` = off/unset).
    pub fn settings(&self) -> Vec<(&'static str, Option<String>)> {
        self.target
            .values()
            .iter()
            .map(|(name, _)| *name)
            .chain(self.target.experimental_settings().iter().copied())
            .map(|name| (name, effective_setting(name)))
            .collect()
    }

    /// Settings requested with `1` that do not take effect because the real
    /// GPU lacks the tensor units they need.
    pub fn hardware_ignored_settings(&self) -> Vec<&'static str> {
        self.target
            .experimental_settings()
            .iter()
            .copied()
            .filter(|name| {
                TENSOR_UNIT_SETTINGS.contains(name)
                    && setting(name).as_deref() == Some("1")
                    && !tensor_unit_flag(name)
            })
            .collect()
    }

    /// `name=value` of every setting with an effective value, or `none`:
    /// what the profile (or an explicit variable) actually turns on, as
    /// opposed to the profile merely being installed.
    pub fn effective_settings_summary(&self) -> String {
        let effective: Vec<String> = self
            .settings()
            .into_iter()
            .filter_map(|(name, value)| value.map(|value| format!("{name}={value}")))
            .collect();
        if effective.is_empty() {
            "none".to_owned()
        } else {
            effective.join(",")
        }
    }
}

static INSTALLED: OnceLock<M5Profile> = OnceLock::new();

/// The installed profile, if any.
pub fn installed() -> Option<&'static M5Profile> {
    INSTALLED.get()
}

pub fn active() -> bool {
    installed().is_some_and(M5Profile::is_active)
}

fn setting_with(
    explicit: Option<String>,
    active_values: Option<&[(&str, &str)]>,
    name: &str,
) -> Option<String> {
    if explicit.is_some() {
        return explicit;
    }
    active_values
        .and_then(|values| values.iter().find(|(key, _)| *key == name))
        .map(|(_, value)| (*value).to_owned())
}

/// Effective value of a profile setting: an explicitly set variable,
/// otherwise the active profile's value for its target family, otherwise
/// unset.
pub fn setting(name: &str) -> Option<String> {
    let active_values = installed()
        .filter(|profile| profile.is_active())
        .map(|profile| profile.target.values());
    setting_with(std::env::var(name).ok(), active_values, name)
}

/// Boolean profile setting (`"1"` = on).
pub fn flag(name: &str) -> bool {
    setting(name).as_deref() == Some("1")
}

/// Settings that run M5 tensor-unit kernels: besides being set they need a
/// real Apple GPU of generation [`MIN_GPU_GENERATION`] or newer.
const TENSOR_UNIT_SETTINGS: &[&str] = &[settings::M5_MOE_AFFINE4_PROJECTIONS];

/// Metal architecture of the running GPU as MLX reports it. Unlike
/// [`current_architecture`], the diagnostic override is ignored: hardware
/// gates follow the real device.
pub fn hardware_architecture() -> Option<String> {
    static ARCHITECTURE: OnceLock<Option<String>> = OnceLock::new();
    ARCHITECTURE
        .get_or_init(|| mlx::metal::architecture().ok())
        .clone()
}

/// Whether a tensor-unit setting with `value` takes effect on a GPU whose
/// real architecture is `hardware_architecture`: it must be `1` and the
/// architecture a known Apple GPU of generation 17 or newer.
fn tensor_unit_setting_effective(value: Option<&str>, hardware_architecture: Option<&str>) -> bool {
    value == Some("1")
        && hardware_architecture
            .and_then(gpu_generation)
            .is_some_and(|generation| generation >= MIN_GPU_GENERATION)
}

/// [`flag`] for a setting that runs M5 tensor-unit kernels: an explicit
/// variable cannot enable it on a GPU without them.
pub fn tensor_unit_flag(name: &str) -> bool {
    tensor_unit_setting_effective(setting(name).as_deref(), hardware_architecture().as_deref())
}

/// [`setting`] as it takes effect: a tensor-unit setting the GPU cannot run
/// reports as unset.
fn effective_setting(name: &str) -> Option<String> {
    let value = setting(name);
    if TENSOR_UNIT_SETTINGS.contains(&name)
        && value.as_deref() == Some("1")
        && !tensor_unit_flag(name)
    {
        return None;
    }
    value
}

/// GPU architecture used for the profile decision.
pub fn current_architecture() -> Option<String> {
    std::env::var(ARCHITECTURE_OVERRIDE_ENV)
        .ok()
        .or_else(|| mlx::metal::architecture().ok())
}

/// Resolve and install the process profile (first call wins) and configure
/// the prefill kernels accordingly. Call before loading a model.
pub fn install(mode: M5ProfileMode, dflash2: bool) -> &'static M5Profile {
    install_for_target(mode, dflash2, M5ProfileTarget::Qwen35Dense)
}

/// [`install`] for a DFlash2 target family. An unqualified target keeps
/// every profile setting off; each qualified family uses only its own table.
pub fn install_for_target(
    mode: M5ProfileMode,
    dflash2: bool,
    target: M5ProfileTarget,
) -> &'static M5Profile {
    INSTALLED.get_or_init(|| {
        let architecture = current_architecture();
        let status = resolve_for_target(
            mode,
            dflash2,
            target != M5ProfileTarget::Unqualified,
            architecture.as_deref(),
        );
        // The prefill kernel libraries and masked softmax belong to the
        // dense lane qualification.
        let active = status == M5ProfileStatus::Active && target == M5ProfileTarget::Qwen35Dense;
        let prefill_libraries = vec![
            configure_library(
                "prefill_qmm_mtile",
                settings::PREFILL_QMM_MTILE_METALLIB,
                mlx::metal::prefill_shaders::PREFILL_QMM_MTILE_METALLIB,
                active,
                mlx::metal::set_prefill_qmm_mtile_library,
            ),
            configure_library(
                "prefill_d256_nax",
                settings::PREFILL_D256_NAX_METALLIB,
                mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB,
                active,
                mlx::metal::set_prefill_d256_nax_library,
            ),
        ];
        let masked_softmax = setting_with(
            std::env::var(settings::PREFILL_MASKED_SOFTMAX).ok(),
            active.then_some(PROFILE_VALUES),
            settings::PREFILL_MASKED_SOFTMAX,
        )
        .as_deref()
            == Some("1");
        mlx::metal::set_prefill_masked_softmax(masked_softmax);
        M5Profile {
            mode,
            architecture,
            status,
            target,
            prefill_libraries,
        }
    })
}

fn configure_library(
    name: &'static str,
    env: &str,
    embedded: &[u8],
    active: bool,
    set: fn(&str) -> bool,
) -> PrefillLibrary {
    let (source, path, error) = if let Ok(path) = std::env::var(env) {
        ("environment", Some(path), None)
    } else if active {
        match materialize_library(name, embedded) {
            Ok(path) => ("profile", Some(path), None),
            Err(error) => ("off", None, Some(error.to_string())),
        }
    } else {
        ("off", None, None)
    };
    let applied = set(path.as_deref().unwrap_or(""));
    let error = error.or_else(|| (!applied).then(|| "configured after first use".to_owned()));
    PrefillLibrary {
        name,
        source,
        path,
        error,
    }
}

/// Write an embedded library to a content-addressed file (MLX loads
/// libraries by path) and return the path. An existing file is reused only
/// if its bytes are identical.
fn materialize_library(name: &str, bytes: &[u8]) -> std::io::Result<String> {
    let dir = std::env::temp_dir().join("ironmlx-prefill-kernels");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!(
        "{name}-{}-{:016x}.metallib",
        bytes.len(),
        fnv1a64(bytes)
    ));
    if std::fs::read(&path).ok().as_deref() != Some(bytes) {
        let staging = dir.join(format!(
            ".{name}-{}-{}.tmp",
            std::process::id(),
            fnv1a64(bytes)
        ));
        std::fs::write(&staging, bytes)?;
        std::fs::rename(&staging, &path)?;
    }
    path.into_os_string()
        .into_string()
        .map_err(|_| std::io::Error::other("non-UTF-8 kernel cache path"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_unit_settings_require_a_real_g17_gpu() {
        // Supported hardware, explicitly enabled.
        assert!(tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g17s")
        ));
        assert!(tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g18g")
        ));
        // Older GPU, unknown or unreadable architecture: an explicit `1`
        // does not override the hardware requirement.
        assert!(!tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g16s")
        ));
        assert!(!tensor_unit_setting_effective(
            Some("1"),
            Some("applegpu_g13p")
        ));
        assert!(!tensor_unit_setting_effective(Some("1"), Some("unknown")));
        assert!(!tensor_unit_setting_effective(Some("1"), None));
        // Not requested.
        assert!(!tensor_unit_setting_effective(None, Some("applegpu_g17s")));
        assert!(!tensor_unit_setting_effective(
            Some("0"),
            Some("applegpu_g17s")
        ));
        // Only the MoE affine4 projections are gated; the grouped qmv kernel
        // uses ordinary Metal and the dense settings keep their own gates.
        assert_eq!(
            TENSOR_UNIT_SETTINGS,
            &[settings::M5_MOE_AFFINE4_PROJECTIONS]
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn hardware_gate_ignores_the_diagnostic_architecture_override() {
        // The diagnostic override feeds the profile decision only; the
        // hardware gate reads MLX's real architecture.
        let previous = std::env::var(ARCHITECTURE_OVERRIDE_ENV).ok();
        std::env::set_var(ARCHITECTURE_OVERRIDE_ENV, "applegpu_g99s");
        let hardware = hardware_architecture();
        assert_ne!(hardware.as_deref(), Some("applegpu_g99s"));
        assert_eq!(current_architecture().as_deref(), Some("applegpu_g99s"));
        match previous {
            Some(value) => std::env::set_var(ARCHITECTURE_OVERRIDE_ENV, value),
            None => std::env::remove_var(ARCHITECTURE_OVERRIDE_ENV),
        }
    }

    #[test]
    fn generation_parses_apple_gpu_names() {
        assert_eq!(gpu_generation("applegpu_g17s"), Some(17));
        assert_eq!(gpu_generation("applegpu_g16p"), Some(16));
        assert_eq!(gpu_generation("applegpu_g18"), Some(18));
        assert_eq!(gpu_generation("apple_g13s"), None);
        assert_eq!(gpu_generation("applegpu_gx"), None);
    }

    #[test]
    fn profile_is_active_only_for_dflash2_on_g17_and_newer() {
        use M5ProfileMode::{Auto, Off};
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g17s")),
            M5ProfileStatus::Active
        );
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g18p")),
            M5ProfileStatus::Active
        );
        assert_eq!(
            resolve(Auto, true, Some("applegpu_g16s")),
            M5ProfileStatus::UnsupportedGpu
        );
        assert_eq!(
            resolve(Auto, true, Some("apple_g13s")),
            M5ProfileStatus::GpuUnknown
        );
        assert_eq!(resolve(Auto, true, None), M5ProfileStatus::GpuUnknown);
        assert_eq!(
            resolve(Auto, false, Some("applegpu_g17s")),
            M5ProfileStatus::NotDFlash2
        );
        assert_eq!(
            resolve_for_target(Auto, true, false, Some("applegpu_g17s")),
            M5ProfileStatus::TargetNotQualified
        );
        assert_eq!(
            resolve_for_target(Off, true, false, Some("applegpu_g17s")),
            M5ProfileStatus::Disabled
        );
        assert_eq!(
            resolve(Off, true, Some("applegpu_g17s")),
            M5ProfileStatus::Disabled
        );
        assert_eq!("off".parse::<M5ProfileMode>(), Ok(Off));
        assert!("on".parse::<M5ProfileMode>().is_err());
    }

    #[test]
    fn explicit_setting_overrides_profile_value() {
        let dense = Some(M5ProfileTarget::Qwen35Dense.values());
        let name = settings::DFLASH2_FIXED_BUDGET;
        assert_eq!(setting_with(None, dense, name).as_deref(), Some("7"));
        assert_eq!(setting_with(None, None, name), None);
        assert_eq!(
            setting_with(Some("3".into()), dense, name).as_deref(),
            Some("3")
        );
        assert_eq!(
            setting_with(Some("0".into()), dense, settings::M5_LANE_ATTN).as_deref(),
            Some("0")
        );
        assert_eq!(
            setting_with(Some("1".into()), None, settings::M5_LANE_ATTN).as_deref(),
            Some("1")
        );
        // The prefill library paths are not table values; they come from
        // the embedded copies at install.
        assert_eq!(
            setting_with(None, dense, settings::PREFILL_QMM_MTILE_METALLIB),
            None
        );
    }

    #[test]
    fn moe_profile_never_inherits_dense_lane_settings() {
        let moe = Some(M5ProfileTarget::Qwen36Moe.values());
        for (name, _) in PROFILE_VALUES {
            if *name == settings::QEMBEDDING_RUNTIME_COUNT || *name == settings::MLX_CACHE_MAX_MIB {
                continue;
            }
            assert_eq!(setting_with(None, moe, name), None, "{name}");
        }
        // Shared settings with target-specific values.
        assert_eq!(
            setting_with(None, moe, settings::QEMBEDDING_RUNTIME_COUNT).as_deref(),
            Some("1")
        );
        assert_eq!(
            setting_with(None, moe, settings::MLX_CACHE_MAX_MIB).as_deref(),
            Some("512")
        );
        assert_eq!(
            setting_with(None, moe, settings::WIRED_WEIGHTS).as_deref(),
            Some("1")
        );
        assert_eq!(
            setting_with(
                None,
                Some(M5ProfileTarget::Qwen35Dense.values()),
                settings::WIRED_WEIGHTS
            ),
            None
        );
        // The experimental MoE settings are reported, never enabled by the
        // profile; an explicit value still applies.
        for name in [
            settings::M5_MOE_GROUPED_QMV,
            settings::M5_MOE_AFFINE4_PROJECTIONS,
        ] {
            assert_eq!(setting_with(None, moe, name), None, "{name}");
        }
        assert_eq!(
            setting_with(Some("1".into()), moe, settings::M5_MOE_GROUPED_QMV).as_deref(),
            Some("1")
        );
        let dense = Some(M5ProfileTarget::Qwen35Dense.values());
        assert_eq!(
            setting_with(None, dense, settings::M5_MOE_GROUPED_QMV),
            None
        );
        assert!(M5ProfileTarget::Unqualified.values().is_empty());
    }

    #[test]
    fn embedded_libraries_match_the_qualified_kernels() {
        // Same size as the candidate's libraries; the byte identity is
        // checked against the frozen files by the product verification.
        assert_eq!(
            mlx::metal::prefill_shaders::PREFILL_QMM_MTILE_METALLIB.len(),
            70_104
        );
        assert_eq!(
            mlx::metal::prefill_shaders::PREFILL_D256_NAX_METALLIB.len(),
            32_392
        );
    }

    #[test]
    fn materialized_library_is_content_addressed_and_reused() {
        let bytes = b"ironmlx test library";
        let first = materialize_library("unit_test", bytes).expect("written");
        let second = materialize_library("unit_test", bytes).expect("reused");
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&first).expect("readable"), bytes);
        let other = materialize_library("unit_test", b"other bytes").expect("written");
        assert_ne!(first, other);
        let _ = std::fs::remove_file(first);
        let _ = std::fs::remove_file(other);
    }
}
