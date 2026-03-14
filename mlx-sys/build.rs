use std::path::PathBuf;

/// Recursively search for a static library under `dir`, returning the
/// directory that contains it (suitable for `cargo:rustc-link-search`).
fn find_lib(dir: &PathBuf, lib_name: &str) -> Option<PathBuf> {
    let target = format!("lib{}.a", lib_name);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_lib(&path, lib_name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == target.as_str()) {
            return Some(path.parent().unwrap().to_path_buf());
        }
    }
    None
}

fn main() {
    // Allow overriding paths via environment variables for portability.
    // MLX_C_DIR: path to mlx-c source (required; falls back to local dev path).
    // MLX_DIR:   path to local mlx source; if absent, cmake FetchContent downloads it.
    let mlx_c_src = PathBuf::from(
        std::env::var("MLX_C_DIR").unwrap_or_else(|_| "/Volumes/CodeHub/mlx-c".to_string()),
    );
    let mlx_src = std::env::var("MLX_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            let default = PathBuf::from("/Volumes/CodeHub/mlx");
            if default.exists() {
                Some(default)
            } else {
                None
            }
        });

    // ── 1. Build mlx-c (mlx is pulled in as a FetchContent dependency) ──────
    let mut cmake_cfg = cmake::Config::new(&mlx_c_src);
    cmake_cfg
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("MLX_C_USE_SYSTEM_MLX", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    // Point FetchContent at a local mlx repo if available — skips network download.
    if let Some(ref src) = mlx_src {
        cmake_cfg.define(
            "FETCHCONTENT_SOURCE_DIR_MLX",
            src.to_str().expect("mlx path is not valid UTF-8"),
        );
    }

    // Allow disabling Metal backend (e.g. when full Xcode is not installed).
    if std::env::var("MLX_NO_METAL").is_ok() {
        cmake_cfg.define("MLX_BUILD_METAL", "OFF");
    }

    let dst = cmake_cfg.build();

    // ── 2. Link mlxc (installed by cmake into dst/lib) ───────────────────────
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlxc");

    // ── 3. Link mlx (built as FetchContent, not installed – must find it) ────
    let build_dir = dst.join("build");
    let mlx_lib_dir =
        find_lib(&build_dir, "mlx").expect("Could not find libmlx.a in cmake build tree");
    println!("cargo:rustc-link-search=native={}", mlx_lib_dir.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // ── 4. macOS system frameworks required by mlx on Apple Silicon ──────────
    for fw in &["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={}", fw);
    }

    // ── 5. C++ standard library ───────────────────────────────────────────────
    println!("cargo:rustc-link-lib=c++");

    // ── 6. Re-run triggers ────────────────────────────────────────────────────
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        mlx_c_src.join("mlx/c/mlx.h").display()
    );
    println!("cargo:rerun-if-env-changed=MLX_C_DIR");
    println!("cargo:rerun-if-env-changed=MLX_DIR");

    // ── 7. Generate Rust FFI bindings via bindgen ────────────────────────────
    let bindings = bindgen::Builder::default()
        .header(
            mlx_c_src
                .join("mlx/c/mlx.h")
                .to_str()
                .expect("header path is not valid UTF-8"),
        )
        // Teach clang where to find the mlx-c headers.
        .clang_arg(format!("-I{}", mlx_c_src.display()))
        // Pass the Rust target triple to clang so architecture-specific
        // types (e.g. __fp16 on ARM) resolve correctly.
        .clang_arg(format!("--target={}", std::env::var("TARGET").unwrap()))
        // Only emit items that originate from mlx-c headers (avoids pulling in
        // system headers into the generated file).
        .allowlist_file(format!("{}.*", mlx_c_src.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings");
}
