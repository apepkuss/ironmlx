//! Each metallib condition runs in its own child: the Metal device is a process
//! singleton, and a crash must fail the parent's assertion, not the test binary.

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CASE: &str = "IRONMLX_METALLIB_TEST_CASE";
const PATH: &str = "IRONMLX_METALLIB_TEST_PATH";
const REPORTED: i32 = 42;

// Dispatched by `run_child`; the parent test below owns all assertions.
#[test]
fn child() {
    let Ok(case) = std::env::var(CASE) else {
        return;
    };
    if let Ok(path) = std::env::var(PATH) {
        mlx::metal::set_metallib_path(&path).unwrap();
    }
    match case.as_str() {
        "verify" => match mlx::metal::verify_runtime() {
            Ok(()) => println!("VERIFIED"),
            Err(error) => {
                eprintln!("REPORTED: {error}");
                std::process::exit(REPORTED);
            }
        },
        "counters" => match mlx::memory::try_snapshot() {
            Ok(snapshot) => println!("COUNTERS {}", snapshot.active_bytes),
            Err(error) => {
                eprintln!("REPORTED: {error}");
                std::process::exit(REPORTED);
            }
        },
        "panicking_counters" => {
            let snapshot = mlx::memory::snapshot();
            println!("COUNTERS {}", snapshot.active_bytes);
        }
        "valid_runtime" => valid_runtime(),
        other => panic!("unknown case {other}"),
    }
}

fn valid_runtime() {
    mlx::metal::verify_runtime().unwrap();
    let values = mlx::Array::try_from((&[1_f32, 2., 3., 4.][..], &[2_i32, 2][..])).unwrap();
    let gpu = mlx::ops::matmul_on(&values, &values, mlx::Device::gpu(0)).unwrap();
    let cpu = mlx::ops::matmul_on(&values, &values, mlx::Device::cpu()).unwrap();
    let expected = vec![7_f32, 10., 15., 22.];
    assert_eq!(gpu.to_vec::<f32>().unwrap(), expected);
    assert_eq!(cpu.to_vec::<f32>().unwrap(), expected);
    // A result evaluated on a worker thread stays usable after that thread exits.
    let doubled = std::thread::spawn(move || {
        let doubled = mlx::ops::binary::add(&values, &values).unwrap();
        doubled.eval().unwrap();
        doubled
    })
    .join()
    .unwrap();
    let surviving = mlx::ops::binary::add(&doubled, &doubled).unwrap();
    assert_eq!(
        surviving.to_vec::<f32>().unwrap(),
        vec![4_f32, 8., 12., 16.]
    );
    assert!(mlx::memory::snapshot().active_bytes > 0);
    println!("VALID");
}

fn run_child(case: &str, path: Option<&Path>, dir: &Path) -> Output {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(CASE, case)
        .env_remove(PATH)
        .current_dir(dir);
    if let Some(path) = path {
        command.env(PATH, path);
    }
    command.output().unwrap()
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_reported(case: &str, output: &Output, needle: &str) {
    let log = text(output);
    assert_eq!(
        output.status.signal(),
        None,
        "{case} died from a signal instead of an error:\n{log}"
    );
    assert_eq!(output.status.code(), Some(REPORTED), "{case}:\n{log}");
    assert!(log.contains("REPORTED"), "{case}:\n{log}");
    assert!(log.contains(needle), "{case} missing `{needle}`:\n{log}");
}

fn correct_metallib() -> Option<PathBuf> {
    let root = std::env::var("MLX_DIR").ok()?;
    let path = PathBuf::from(root).join("lib/mlx.metallib");
    path.exists().then_some(path)
}

/// A real Metal library without any MLX kernel, when the toolchain can build one.
fn unrelated_metallib(dir: &Path) -> Option<PathBuf> {
    let source = dir.join("unrelated.metal");
    let library = dir.join("unrelated.metallib");
    std::fs::write(
        &source,
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void unrelated(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 1; }\n",
    )
    .ok()?;
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-o"])
        .arg(&library)
        .arg(&source)
        .status()
        .ok()?;
    status.success().then_some(library)
}

#[test]
fn invalid_resources_report_errors_and_valid_ones_run() {
    let dir = std::env::temp_dir().join(format!("ironmlx-metallib-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let missing = dir.join("missing.metallib");
    let directory = dir.join("directory.metallib");
    std::fs::create_dir_all(&directory).unwrap();
    let corrupt = dir.join("corrupt.metallib");
    std::fs::write(&corrupt, vec![0xA5_u8; 1 << 16]).unwrap();
    for (case, path) in [
        ("missing", &missing),
        ("directory", &directory),
        ("corrupt", &corrupt),
    ] {
        let output = run_child("verify", Some(path), &dir);
        assert_reported(case, &output, "Can not load metallib");
        assert_reported(case, &output, path.to_str().unwrap());
        let output = run_child("counters", Some(path), &dir);
        assert_reported(case, &output, "Can not load metallib");
        // Never zeroed counters: a panic the process can report and exit from.
        let output = run_child("panicking_counters", Some(path), &dir);
        let log = text(&output);
        assert_eq!(output.status.signal(), None, "{case}:\n{log}");
        assert!(!output.status.success(), "{case}:\n{log}");
        assert!(
            log.contains("MLX memory counters unavailable"),
            "{case}:\n{log}"
        );
        assert!(!log.contains("COUNTERS"), "{case}:\n{log}");
    }

    if let Some(correct) = correct_metallib() {
        let bytes = std::fs::read(&correct).unwrap();
        let truncated = dir.join("truncated.metallib");
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        let output = run_child("verify", Some(&truncated), &dir);
        assert_reported("truncated", &output, "Can not load metallib");

        let output = run_child("valid_runtime", Some(&correct), &dir);
        let log = text(&output);
        assert!(output.status.success(), "correct metallib:\n{log}");
        assert!(log.contains("VALID"), "correct metallib:\n{log}");
    } else {
        eprintln!("MLX_DIR metallib unavailable; correct and truncated cases skipped");
    }

    if let Some(unrelated) = unrelated_metallib(&dir) {
        let output = run_child("verify", Some(&unrelated), &dir);
        assert_reported("unrelated", &output, "Unable to load kernel");
    } else {
        eprintln!("Metal compiler unavailable; unrelated-library case skipped");
    }

    // No configured path and nothing beside the binary: MLX's default search
    // either finds its build-tree library or reports every location it tried.
    let output = run_child("verify", None, &dir);
    let log = text(&output);
    assert_eq!(output.status.signal(), None, "default lookup:\n{log}");
    assert!(
        output.status.success() || output.status.code() == Some(REPORTED),
        "default lookup:\n{log}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
