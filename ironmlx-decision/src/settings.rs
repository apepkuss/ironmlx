//! Model-load settings for native decision inference.
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDtype {
    #[default]
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "float32")]
    Float32,
}

impl ComputeDtype {
    pub fn weight_multiplier(self) -> usize {
        match self {
            Self::Float16 => 1,
            Self::Float32 => 2,
        }
    }
    pub(crate) fn mlx(self) -> mlx::Dtype {
        match self {
            Self::Float16 => mlx::Dtype::Float16,
            Self::Float32 => mlx::Dtype::Float32,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    #[default]
    Auto,
    Gpu,
    Cpu,
}
impl ComputeDevice {
    pub fn resolve(self) -> Result<mlx::Device> {
        let gpu = mlx::Device::gpu(0);
        let device = match self {
            Self::Auto if mlx::is_available(gpu) => gpu,
            Self::Auto | Self::Cpu => mlx::Device::cpu(),
            Self::Gpu => gpu,
        };
        if !mlx::is_available(device) {
            bail!("requested decision device is unavailable");
        }
        Ok(device)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecisionSettings {
    pub dtype: ComputeDtype,
    pub batch_size: usize,
    pub cache_prompts: bool,
    pub device: ComputeDevice,
    pub compile: bool,
    pub pad_to_multiple: Option<usize>,
}
impl Default for DecisionSettings {
    fn default() -> Self {
        Self {
            dtype: ComputeDtype::Float16,
            batch_size: 16,
            cache_prompts: false,
            device: ComputeDevice::Auto,
            compile: false,
            pad_to_multiple: None,
        }
    }
}
impl DecisionSettings {
    pub fn validate(self) -> Result<()> {
        if !(1..=256).contains(&self.batch_size) {
            bail!("decision batch_size must be between 1 and 256");
        }
        if self
            .pad_to_multiple
            .is_some_and(|n| !(1..=1024).contains(&n))
        {
            bail!("decision pad_to_multiple must be between 1 and 1024");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_settings_and_advanced_validation() {
        let legacy: DecisionSettings =
            serde_json::from_str(r#"{"dtype":"float16","batch_size":16,"cache_prompts":false}"#)
                .unwrap();
        assert_eq!(legacy, DecisionSettings::default());
        for value in [0, 1025, usize::MAX] {
            assert!(DecisionSettings {
                pad_to_multiple: Some(value),
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for value in [1, 16, 1024] {
            let settings = DecisionSettings {
                device: ComputeDevice::Cpu,
                compile: true,
                pad_to_multiple: Some(value),
                ..Default::default()
            };
            settings.validate().unwrap();
            assert_eq!(
                settings,
                serde_json::from_value(serde_json::to_value(settings).unwrap()).unwrap()
            );
        }
        assert!(serde_json::from_str::<DecisionSettings>(r#"{"device":"cuda"}"#).is_err());
    }
}
