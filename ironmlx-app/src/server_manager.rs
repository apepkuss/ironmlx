use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    #[allow(dead_code)]
    Failed,
}

pub struct ServerManager {
    process: Arc<Mutex<Option<Child>>>,
    status: Arc<Mutex<ServerStatus>>,
    host: String,
    port: u16,
    memory_limit_total: f64,
    hot_cache_gb: f64,
    cold_cache_gb: f64,
    max_sequences: usize,
    init_cache_blocks: usize,
    cache_enabled: bool,
    cache_dir: String,
}

impl ServerManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: &str,
        port: u16,
        memory_limit_total: f64,
        hot_cache_gb: f64,
        cold_cache_gb: f64,
        max_sequences: usize,
        init_cache_blocks: usize,
        cache_enabled: bool,
        cache_dir: &str,
    ) -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            host: host.to_string(),
            port,
            memory_limit_total,
            hot_cache_gb,
            cold_cache_gb,
            max_sequences,
            init_cache_blocks,
            cache_enabled,
            cache_dir: cache_dir.to_string(),
        }
    }

    pub fn status(&self) -> ServerStatus {
        *self.status.lock().unwrap()
    }

    pub fn start(&self, model_path: &str) -> Result<(), String> {
        let mut status = self.status.lock().unwrap();
        if *status == ServerStatus::Running {
            return Ok(());
        }
        *status = ServerStatus::Starting;
        drop(status);

        // Find ironmlx binary — same directory as this app, or in PATH
        let binary = Self::find_binary().ok_or("ironmlx binary not found")?;

        let mut cmd = Command::new(&binary);
        cmd.arg("--model")
            .arg(model_path)
            .arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(self.port.to_string());
        if self.memory_limit_total > 0.0 {
            cmd.arg("--memory-limit")
                .arg(self.memory_limit_total.to_string());
        }
        cmd.arg("--max-sequences")
            .arg(self.max_sequences.to_string());
        if self.init_cache_blocks > 0 {
            cmd.arg("--init-cache-blocks")
                .arg(self.init_cache_blocks.to_string());
        }
        if !self.cache_enabled {
            cmd.arg("--no-cache");
        } else {
            if self.hot_cache_gb > 0.0 {
                cmd.arg("--hot-cache-limit")
                    .arg(self.hot_cache_gb.to_string());
            }
            cmd.arg("--cold-cache-limit")
                .arg(self.cold_cache_gb.to_string());
        }
        if !self.cache_dir.is_empty() && self.cache_dir != "~/.ironmlx/cache/kv_cache" {
            cmd.arg("--cache-dir").arg(&self.cache_dir);
        }
        let child = cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start server: {}", e))?;

        *self.process.lock().unwrap() = Some(child);
        *self.status.lock().unwrap() = ServerStatus::Running;

        // Start crash monitor thread — detects backend exit and auto-restarts
        let process_arc = self.process.clone();
        let status_arc = self.status.clone();
        let model = model_path.to_string();
        let binary_clone = binary.clone();
        let host = self.host.clone();
        let port = self.port;
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3));
                let mut guard = process_arc.lock().unwrap();
                if let Some(ref mut child) = *guard {
                    match child.try_wait() {
                        Ok(Some(exit_status)) => {
                            // Process exited — log and attempt restart
                            eprintln!(
                                "[server_manager] backend exited with status: {}. Restarting...",
                                exit_status
                            );
                            *status_arc.lock().unwrap() = ServerStatus::Stopped;
                            drop(guard);

                            // Wait briefly before restart
                            std::thread::sleep(std::time::Duration::from_secs(2));

                            // Restart
                            let restart_result = Command::new(&binary_clone)
                                .arg("--model")
                                .arg(&model)
                                .arg("--host")
                                .arg(&host)
                                .arg("--port")
                                .arg(port.to_string())
                                .stdout(std::process::Stdio::null())
                                .stderr(std::process::Stdio::null())
                                .spawn();

                            match restart_result {
                                Ok(new_child) => {
                                    *process_arc.lock().unwrap() = Some(new_child);
                                    *status_arc.lock().unwrap() = ServerStatus::Running;
                                    eprintln!("[server_manager] backend restarted successfully");
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[server_manager] failed to restart backend: {}",
                                        e
                                    );
                                    return; // Stop monitoring
                                }
                            }
                        }
                        Ok(None) => {} // Still running
                        Err(_) => return, // Can't check — stop monitoring
                    }
                } else {
                    return; // No process — intentionally stopped, exit monitor
                }
            }
        });

        Ok(())
    }

    pub fn stop(&self) {
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *self.status.lock().unwrap() = ServerStatus::Stopped;
    }

    pub fn set_host(&mut self, host: &str) {
        self.host = host.to_string();
    }

    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }

    pub fn set_memory_limit_total(&mut self, limit: f64) {
        self.memory_limit_total = limit;
    }

    pub fn set_hot_cache_gb(&mut self, v: f64) {
        self.hot_cache_gb = v;
    }

    pub fn set_cold_cache_gb(&mut self, v: f64) {
        self.cold_cache_gb = v;
    }

    pub fn set_max_sequences(&mut self, v: usize) {
        self.max_sequences = v;
    }

    pub fn set_init_cache_blocks(&mut self, v: usize) {
        self.init_cache_blocks = v;
    }

    pub fn set_cache_enabled(&mut self, v: bool) {
        self.cache_enabled = v;
    }

    pub fn set_cache_dir(&mut self, v: &str) {
        self.cache_dir = v.to_string();
    }

    pub fn restart(&self, model_path: &str) -> Result<(), String> {
        self.stop();
        self.start(model_path)
    }

    #[allow(dead_code)]
    pub fn is_running(&self) -> bool {
        // Check if process is still alive
        if let Some(ref mut child) = *self.process.lock().unwrap() {
            match child.try_wait() {
                Ok(None) => return true, // Still running
                Ok(Some(_)) => {
                    // Process exited
                    *self.status.lock().unwrap() = ServerStatus::Stopped;
                    return false;
                }
                Err(_) => return false,
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn check_health(&self) -> bool {
        let health_host = if self.host == "0.0.0.0" {
            "127.0.0.1"
        } else {
            &self.host
        };
        let url = format!("http://{}:{}/health", health_host, self.port);
        // Simple blocking check
        std::process::Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
            .unwrap_or(false)
    }

    fn find_binary() -> Option<String> {
        // 1. Same directory as this binary
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            let sibling = dir.join("ironmlx");
            if sibling.exists() {
                return Some(sibling.to_string_lossy().to_string());
            }
        }
        // 2. In PATH
        if std::process::Command::new("which")
            .arg("ironmlx")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some("ironmlx".to_string());
        }
        // 3. Homebrew location
        let brew_path = "/opt/homebrew/bin/ironmlx";
        if std::path::Path::new(brew_path).exists() {
            return Some(brew_path.to_string());
        }
        None
    }
}

impl Drop for ServerManager {
    fn drop(&mut self) {
        self.stop();
    }
}
