use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Failed,
}

pub struct ServerManager {
    process: Arc<Mutex<Option<Child>>>,
    status: Arc<Mutex<ServerStatus>>,
    port: u16,
}

impl ServerManager {
    pub fn new(port: u16) -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            port,
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

        let child = Command::new(&binary)
            .arg("--model")
            .arg(model_path)
            .arg("--port")
            .arg(self.port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start server: {}", e))?;

        *self.process.lock().unwrap() = Some(child);
        *self.status.lock().unwrap() = ServerStatus::Running;
        Ok(())
    }

    pub fn stop(&self) {
        if let Some(mut child) = self.process.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *self.status.lock().unwrap() = ServerStatus::Stopped;
    }

    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }

    pub fn restart(&self, model_path: &str) -> Result<(), String> {
        self.stop();
        self.start(model_path)
    }

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

    pub fn check_health(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        // Simple blocking check
        std::process::Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
            .unwrap_or(false)
    }

    fn find_binary() -> Option<String> {
        // 1. Same directory as this binary
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let sibling = dir.join("ironmlx");
                if sibling.exists() {
                    return Some(sibling.to_string_lossy().to_string());
                }
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
