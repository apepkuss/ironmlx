//! Verified offline auxiliary artifacts for the fixed IndexTTS reference encoder.
use super::{inspect_component, load_component, mismatch, ComponentSpec, ResourceReport};
use crate::{ResourceIssue, Result};
use ironmlx_core::weights::WeightMap;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Clone, Copy)]
pub enum AuxiliaryComponent {
    Matrices,
    CampPlus,
}
impl AuxiliaryComponent {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Matrices => "auxiliary.safetensors",
            Self::CampPlus => "campplus.safetensors",
        }
    }
    pub fn spec(self) -> Result<ComponentSpec> {
        let pinned: Value =
            serde_json::from_str(include_str!("../../resources/indextts25/derived.json"))
                .map_err(|e| mismatch("derived profile", e.to_string()))?;
        let schema = match self {
            Self::Matrices => include_str!("../../resources/indextts25/auxiliary.schema.json"),
            Self::CampPlus => include_str!("../../resources/indextts25/campplus.schema.json"),
        };
        let mut spec = pinned["components"][self.file_name()].clone();
        spec["tensors"] =
            serde_json::from_str(schema).map_err(|e| mismatch("derived schema", e.to_string()))?;
        serde_json::from_value(spec).map_err(|e| mismatch("derived profile", e.to_string()))
    }
}

/// Validate the complete derived artifact, including provenance and copied configuration.
/// Output hashes are pinned to the bit-preserving offline recipe, not trusted from its manifest.
pub fn inspect_derived(directory: &Path) -> ResourceReport {
    let mut report = ResourceReport {
        complete: true,
        issues: Vec::new(),
        static_tensor_bytes: 0,
    };
    let mut add = |component: &str, error: String| {
        report.complete = false;
        report.issues.push(ResourceIssue {
            component: component.into(),
            reason: error,
        });
    };
    let pinned: Value =
        serde_json::from_str(include_str!("../../resources/indextts25/derived.json"))
            .expect("embedded profile");
    let read_manifest = || -> Result<Value> {
        let path = directory.join("manifest.json");
        if fs::metadata(&path)?.len() > 1024 * 1024 {
            return Err(mismatch("manifest", "manifest too large"));
        }
        serde_json::from_slice(&fs::read(path)?).map_err(|e| mismatch("manifest", e.to_string()))
    };
    match read_manifest() {
        Err(error) => add("manifest", error.to_string()),
        Ok(manifest) => {
            for key in [
                "schema_version",
                "family",
                "recipe",
                "source_revision",
                "input_digest",
                "inputs",
                "files",
                "reuse_weight",
                "allowed_unused",
            ] {
                if manifest[key] != pinned[key] {
                    add(
                        "manifest",
                        format!("{key} differs from the fixed conversion profile"),
                    );
                }
            }
            if manifest["components"].as_object().map(|x| x.len()) != Some(2) {
                add("manifest", "expected exactly two components".into());
            }
            for component in [AuxiliaryComponent::Matrices, AuxiliaryComponent::CampPlus] {
                match component.spec() {
                    Ok(spec) => {
                        if serde_json::to_value(&spec).ok().as_ref()
                            != Some(&manifest["components"][component.file_name()])
                        {
                            add(
                                component.file_name(),
                                "manifest component differs from the pinned output".into(),
                            );
                        }
                    }
                    Err(error) => add(component.file_name(), error.to_string()),
                }
            }
        }
    }
    for file in pinned["files"].as_array().expect("embedded file list") {
        let name = file["path"].as_str().expect("embedded file path");
        let verify = || -> Result<()> {
            let path = directory.join(name);
            if fs::metadata(&path)?.len() != file["bytes"].as_u64().unwrap() {
                return Err(mismatch(name, "file size mismatch"));
            }
            let hash = format!("{:x}", Sha256::digest(fs::read(path)?));
            if Some(hash.as_str()) != file["sha256"].as_str() {
                return Err(mismatch(name, "file hash mismatch"));
            }
            Ok(())
        };
        if let Err(error) = verify() {
            add(name, error.to_string());
        }
    }
    // End the issues closure's mutable borrow before accumulating tensor byte counts.
    for component in [AuxiliaryComponent::Matrices, AuxiliaryComponent::CampPlus] {
        let verify = component
            .spec()
            .and_then(|spec| inspect_component(&directory.join(component.file_name()), &spec));
        match verify {
            Ok(bytes) => report.static_tensor_bytes += bytes,
            Err(error) => {
                report.complete = false;
                report.issues.push(ResourceIssue {
                    component: component.file_name().into(),
                    reason: error.to_string(),
                });
            }
        }
    }
    report
}

/// Load only after the whole derived directory passes verification.
pub fn load_auxiliary(directory: &Path, component: AuxiliaryComponent) -> Result<WeightMap> {
    let report = inspect_derived(directory);
    if !report.complete {
        return Err(mismatch(
            "derived resources",
            report
                .issues
                .into_iter()
                .map(|x| x.reason)
                .collect::<Vec<_>>()
                .join("; "),
        ));
    }
    load_component(&directory.join(component.file_name()), &component.spec()?)
}
