/// Apple Silicon chip information.
#[derive(Debug, Clone)]
pub struct ChipInfo {
    pub chip_name: String,
    pub total_memory_gb: f64,
}

impl ChipInfo {
    /// Detect the current Apple Silicon chip.
    pub fn detect() -> Self {
        Self {
            chip_name: detect_chip_name(),
            total_memory_gb: detect_total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
        }
    }

    /// Recommend memory limit (leave ~2GB or 25% for system, whichever is smaller).
    pub fn recommended_memory_limit(&self) -> usize {
        let total_bytes = (self.total_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let reserve = (2 * 1024 * 1024 * 1024usize).min(total_bytes / 4);
        total_bytes - reserve
    }

    pub fn summary(&self) -> String {
        format!("{} ({:.0}GB RAM)", self.chip_name, self.total_memory_gb)
    }
}

fn detect_chip_name() -> String {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "Unknown Apple Silicon".to_string(),
    }
}

fn detect_total_memory() -> usize {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<usize>()
            .unwrap_or(16 * 1024 * 1024 * 1024),
        _ => 16 * 1024 * 1024 * 1024,
    }
}

impl std::fmt::Display for ChipInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}
