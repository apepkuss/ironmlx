use std::process::Command;

fn ironmlx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ironmlx"))
}

#[test]
fn default_local_mode_rejects_external_bind_before_model_startup() {
    let output = ironmlx()
        .args(["serve", "--host", "0.0.0.0", "--port", "19070"])
        .output()
        .expect("run ironmlx");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("local_bind_address_required"));
}

#[test]
fn lan_mode_without_security_bootstrap_is_refused_before_model_startup() {
    let output = ironmlx()
        .args([
            "serve",
            "--network-mode",
            "lan",
            "--lan-host",
            "192.168.1.24",
            "--port",
            "19071",
        ])
        .output()
        .expect("run ironmlx");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--security-bootstrap-stdin"));
}
