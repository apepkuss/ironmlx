//! Assess every optional Draft 2020-12 test, separating default and format assertion profiles.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ironmlx_lm::core::json_schema::SchemaValidator;
use serde_json::{json, Value};

fn files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if recursive && path.is_dir() {
            paths.extend(files(&path, true)?);
        }
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4,
        "usage: json_schema_optional_suite SUITE_ROOT REPORT.json default|formats"
    );
    anyhow::ensure!(
        matches!(args[3].as_str(), "default" | "formats"),
        "profile must be default or formats"
    );
    let root = Path::new(&args[1]);
    let remote_root = root.join("remotes");
    let mut resources = BTreeMap::new();
    for path in files(&remote_root, true)? {
        let uri = format!(
            "http://localhost:1234/{}",
            path.strip_prefix(&remote_root)?.to_string_lossy()
        );
        resources.insert(uri, serde_json::from_slice(&std::fs::read(path)?)?);
    }
    let mut results = Vec::new();
    let mut total = 0;
    let mut passed = 0;
    let mut unsupported = 0;
    let mut failures = Vec::new();
    let mut policy_passed = 0;
    let mut policy_failures = Vec::new();
    let optional = root.join("tests/draft2020-12/optional");
    for path in files(&optional, true)? {
        let groups: Vec<Value> = serde_json::from_slice(&std::fs::read(&path)?)?;
        for (group_index, group) in groups.iter().enumerate() {
            let tests = group["tests"].as_array().context("tests array")?;
            total += tests.len();
            let validator = if args[3] == "formats" {
                SchemaValidator::with_format_assertions(&group["schema"], resources.clone())
            } else {
                SchemaValidator::with_resources(&group["schema"], resources.clone())
            };
            let mut record = json!({"file":path.strip_prefix(&optional)?,"group_index":group_index,"description":group["description"],"instances":tests.len()});
            match validator {
                Ok(validator) => {
                    let mut matches = 0;
                    for (test_index, test) in tests.iter().enumerate() {
                        let actual = validator.is_valid(&test["data"]);
                        let annotation_profile = args[3] == "default"
                            && path.strip_prefix(&optional)?.starts_with("format");
                        let expected = if annotation_profile {
                            true
                        } else {
                            test["valid"].as_bool().context("expected boolean")?
                        };
                        if actual == expected {
                            policy_passed += 1;
                        } else {
                            policy_failures.push(json!({"file":path.strip_prefix(&optional)?,"group_index":group_index,"test_index":test_index,"expected":expected,"actual":actual}));
                        }
                        if actual == test["valid"].as_bool().context("expected boolean")? {
                            matches += 1;
                        } else {
                            failures.push(json!({"file":path.strip_prefix(&optional)?,"group_index":group_index,"test_index":test_index,
                            "description":test["description"],"schema":group["schema"],"instance":test["data"],"expected":test["valid"],"actual":actual}));
                        }
                    }
                    passed += matches;
                    record["passed"] = json!(matches);
                }
                Err(error) => {
                    unsupported += tests.len();
                    record["compile_error"] = json!(format!("{error:#}"));
                }
            }
            results.push(record);
        }
    }
    let report = json!({"total":total,"passed":passed,"unsupported":unsupported,"failures":failures,"groups":results,
        "profile":args[3],"policy_passed":policy_passed,"policy_failures":policy_failures,"resources":resources.len(),"method":"All unmodified optional schemas and instances. The formats profile explicitly enables format assertions through production SchemaValidator; default profile retains dialect defaults. Raw matches compare the official assertion expectations; default-profile policy checks expect all optional/format instances to validate as annotations (these schemas contain only $schema and format). Remote fixtures supplied to production SchemaValidator as an explicit offline resource catalogue. No network or file retrieval by validator. Compile errors count as unsupported and fail the gate."});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "raw assertion matches {passed}/{total}; configured profile checks {policy_passed}/{total}; {unsupported} unsupported; {} profile failures",
        report["policy_failures"].as_array().unwrap().len()
    );

    anyhow::ensure!(
        policy_passed == total && unsupported == 0,
        "optional profile has unsupported schemas or semantic failures"
    );
    Ok(())
}
