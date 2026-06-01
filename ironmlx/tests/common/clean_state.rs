//! Post-sweep / inter-suite GPU + memory hygiene invariant checks.
//! Replaces Boss's manual `pgrep + memory_pressure + alloc-probe`
//! procedure with a single callable; can be invoked from individual
//! integration tests or from `sweep_full.sh` between suites.
//!
//! See spec
//! `docs/superpowers/specs/2026-05-18-b1-p2-5-production-hardening-design.md`
//! §4.3 G4 for rationale.

// Shared test helpers: each integration-test crate uses a subset, so unused-in-this-crate items are expected.
#![allow(dead_code)]

use std::process::Command;
use std::time::Instant;

#[allow(dead_code)]
#[derive(Debug)]
pub struct CleanStateReport {
    pub ironmlx_processes_alive: usize,
    pub zombies: usize,
    pub free_ram_bytes: usize,
    pub small_alloc_probe_us: u128,
}

pub const MIN_FREE_RAM_BYTES: usize = 1 * 1024 * 1024 * 1024; // 1 GiB
pub const MAX_ALLOC_PROBE_US: u128 = 10_000; // 10 ms

/// Run all hygiene checks. Returns `Ok(report)` if all thresholds met,
/// `Err(detailed_message)` if any failed.
pub fn verify_clean_state(label: &str) -> Result<CleanStateReport, String> {
    let ironmlx_processes_alive = count_ironmlx_processes()?;
    let zombies = count_zombies()?;
    let free_ram_bytes = ironmlx::core::server::health::system_free_ram_bytes();
    let small_alloc_probe_us = run_small_alloc_probe()?;

    let mut errs = Vec::new();
    if ironmlx_processes_alive > 0 {
        errs.push(format!(
            "{ironmlx_processes_alive} ironmlx test processes still alive (expected 0)"
        ));
    }
    if zombies > 0 {
        errs.push(format!("{zombies} zombie processes (expected 0)"));
    }
    if free_ram_bytes < MIN_FREE_RAM_BYTES {
        errs.push(format!(
            "free RAM {free_ram_bytes} bytes < {MIN_FREE_RAM_BYTES} threshold"
        ));
    }
    if small_alloc_probe_us > MAX_ALLOC_PROBE_US {
        errs.push(format!(
            "small alloc probe {small_alloc_probe_us}us > {MAX_ALLOC_PROBE_US}us threshold (Metal kernel cache may be degraded)"
        ));
    }

    let report = CleanStateReport {
        ironmlx_processes_alive,
        zombies,
        free_ram_bytes,
        small_alloc_probe_us,
    };

    if errs.is_empty() {
        Ok(report)
    } else {
        Err(format!(
            "[{label}] verify_clean_state failed:\n  - {}\nreport: {report:#?}",
            errs.join("\n  - ")
        ))
    }
}

fn count_ironmlx_processes() -> Result<usize, String> {
    let output = Command::new("pgrep")
        .args([
            "-f",
            "target/release/deps/b1_p2|target/release/deps/p4_|target/release/deps/p6_",
        ])
        .output()
        .map_err(|e| format!("pgrep failed: {e}"))?;
    let stdout = std::str::from_utf8(&output.stdout).map_err(|e| format!("pgrep utf8: {e}"))?;
    Ok(stdout.lines().filter(|l| !l.trim().is_empty()).count())
}

fn count_zombies() -> Result<usize, String> {
    let output = Command::new("ps")
        .args(["-axo", "stat"])
        .output()
        .map_err(|e| format!("ps failed: {e}"))?;
    let stdout = std::str::from_utf8(&output.stdout).map_err(|e| format!("ps utf8: {e}"))?;
    Ok(stdout.lines().filter(|l| l.trim().starts_with('Z')).count())
}

fn run_small_alloc_probe() -> Result<u128, String> {
    let t0 = Instant::now();
    let _arr = mlx::Array::zeros(&[4_i32, 2_i32][..], mlx::Dtype::Uint32)
        .map_err(|e| format!("Array::zeros failed: {e}"))?;
    let elapsed = t0.elapsed();
    Ok(elapsed.as_micros())
}
