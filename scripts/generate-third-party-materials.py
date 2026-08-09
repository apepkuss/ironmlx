#!/usr/bin/env python3
"""Generate normalized third-party inventory, notices, and license texts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def resolve_native_source(source: str, mlx_source: Path, mlx_build: Path) -> Path:
    prefix, separator, relative = source.partition(":")
    if not separator or not relative:
        raise ValueError(f"invalid native license source: {source}")
    roots = {"mlx": mlx_source, "mlx-build": mlx_build}
    if prefix not in roots:
        raise ValueError(f"unsupported native license source prefix: {prefix}")
    resolved = (roots[prefix] / relative).resolve()
    root = roots[prefix].resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"native license source escapes its root: {source}")
    return resolved


def verify_native_source(
    verification: dict[str, Any], mlx_source: Path, mlx_build: Path
) -> dict[str, Any]:
    source_path = resolve_native_source(
        verification["source"], mlx_source, mlx_build
    )
    verification_type = verification["type"]
    if verification_type == "git":
        actual = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        expected = verification["commit"]
        if actual != expected:
            raise ValueError(
                f"native Git source mismatch at {source_path}: "
                f"expected {expected}, found {actual}"
            )
        return {"commit": expected, "type": verification_type}
    if verification_type == "archive":
        if not source_path.is_file():
            raise ValueError(f"native source archive is missing: {source_path}")
        actual = sha256_bytes(source_path.read_bytes())
        expected = verification["sha256"]
        if actual != expected:
            raise ValueError(
                f"native source archive mismatch at {source_path}: "
                f"expected {expected}, found {actual}"
            )
        return {"sha256": expected, "type": verification_type}
    raise ValueError(f"unsupported native source verification: {verification_type}")


def rust_materials(
    cargo_about_documents: list[dict[str, Any]], licenses_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    package_license_files: dict[str, set[str]] = defaultdict(set)
    license_texts: dict[str, dict[str, Any]] = {}

    for cargo_about in cargo_about_documents:
        for license_entry in cargo_about["licenses"]:
            text = license_entry["text"].strip() + "\n"
            digest = sha256_bytes(text.encode())
            filename = f"rust-license-{digest[:16]}.txt"
            license_texts.setdefault(
                digest,
                {
                    "filename": filename,
                    "license_ids": set(),
                    "text": text,
                },
            )["license_ids"].add(license_entry["id"])
            for usage in license_entry["used_by"]:
                package_license_files[usage["crate"]["id"]].add(filename)

    normalized_licenses: list[dict[str, Any]] = []
    for digest, entry in sorted(license_texts.items()):
        write_text(licenses_dir / entry["filename"], entry["text"])
        normalized_licenses.append(
            {
                "file": entry["filename"],
                "license_ids": sorted(entry["license_ids"]),
                "sha256": digest,
            }
        )

    crate_entries: dict[str, dict[str, Any]] = {}
    for cargo_about in cargo_about_documents:
        for crate_entry in cargo_about["crates"]:
            crate_entries[crate_entry["package"]["id"]] = crate_entry

    crates: list[dict[str, Any]] = []
    for package_id, crate_entry in crate_entries.items():
        package = crate_entry["package"]
        files = sorted(package_license_files[package_id])
        if not files:
            raise ValueError(f"cargo-about supplied no license text for {package_id}")
        crates.append(
            {
                "license_expression": crate_entry["license"],
                "license_files": files,
                "name": package["name"],
                "repository": package.get("repository") or package.get("homepage"),
                "source": package.get("source"),
                "version": package["version"],
            }
        )
    crates.sort(key=lambda entry: (entry["name"].casefold(), entry["version"]))
    return crates, normalized_licenses


def native_materials(
    manifest: dict[str, Any], mlx_source: Path, mlx_build: Path, licenses_dir: Path
) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for dependency in manifest["dependencies"]:
        source_path = resolve_native_source(
            dependency["license_source"], mlx_source, mlx_build
        )
        if not source_path.is_file():
            raise ValueError(f"native license source is missing: {source_path}")
        content = source_path.read_bytes()
        actual_hash = sha256_bytes(content)
        if actual_hash != dependency["license_sha256"]:
            raise ValueError(
                f"native license hash mismatch for {dependency['component']}: "
                f"expected {dependency['license_sha256']}, found {actual_hash}"
            )
        shutil.copyfile(source_path, licenses_dir / dependency["license_file"])
        normalized = {
            key: value
            for key, value in dependency.items()
            if key not in {"license_source", "source_verification"}
        }
        normalized["source_integrity"] = verify_native_source(
            dependency["source_verification"], mlx_source, mlx_build
        )
        dependencies.append(normalized)
    dependencies.sort(key=lambda entry: entry["component"].casefold())
    return dependencies


def swift_materials(
    manifest: dict[str, Any],
    package_resolved: dict[str, Any],
    checkout_root: Path,
    licenses_dir: Path,
) -> list[dict[str, Any]]:
    pins = {pin["identity"]: pin for pin in package_resolved.get("pins", [])}
    expected_identities = {entry["identity"] for entry in manifest["dependencies"]}
    if set(pins) != expected_identities:
        raise ValueError(
            "Swift dependency lock differs from reviewed inventory: "
            f"expected {sorted(expected_identities)}, found {sorted(pins)}"
        )

    dependencies: list[dict[str, Any]] = []
    for dependency in manifest["dependencies"]:
        identity = dependency["identity"]
        state = pins[identity]["state"]
        if state.get("version") != dependency["version"]:
            raise ValueError(
                f"Swift package version mismatch for {identity}: "
                f"expected {dependency['version']}, found {state.get('version')}"
            )
        if state.get("revision") != dependency["revision"]:
            raise ValueError(
                f"Swift package revision mismatch for {identity}: "
                f"expected {dependency['revision']}, found {state.get('revision')}"
            )

        checkout = (checkout_root / identity).resolve()
        if not checkout.is_dir():
            raise ValueError(f"Swift package checkout is missing: {checkout}")
        actual_revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_revision != dependency["revision"]:
            raise ValueError(
                f"Swift checkout revision mismatch for {identity}: "
                f"expected {dependency['revision']}, found {actual_revision}"
            )

        license_path = (checkout / dependency["license_source"]).resolve()
        if checkout != license_path and checkout not in license_path.parents:
            raise ValueError(f"Swift license source escapes checkout: {license_path}")
        if not license_path.is_file():
            raise ValueError(f"Swift license source is missing: {license_path}")
        license_content = license_path.read_bytes()
        actual_hash = sha256_bytes(license_content)
        if actual_hash != dependency["license_sha256"]:
            raise ValueError(
                f"Swift license hash mismatch for {identity}: "
                f"expected {dependency['license_sha256']}, found {actual_hash}"
            )
        shutil.copyfile(license_path, licenses_dir / dependency["license_file"])

        dependencies.append(
            {
                "component": dependency["component"],
                "identity": identity,
                "license": dependency["license"],
                "license_file": dependency["license_file"],
                "license_sha256": actual_hash,
                "repository": dependency["repository"],
                "revision": dependency["revision"],
                "version": dependency["version"],
            }
        )
    dependencies.sort(key=lambda entry: entry["component"].casefold())
    return dependencies


def render_notices(inventory: dict[str, Any]) -> str:
    lines = [
        "# IronMLX Third-Party Notices",
        "",
        "This engineering inventory describes third-party software included in the",
        "Apple Silicon macOS Release product. It is generated from the locked Rust",
        "dependency graph and the pinned native MLX build inputs. It is not legal",
        "advice or approval to distribute the product.",
        "",
        "## Native dependencies",
        "",
        "| Component | Revision | License | License text | Source |",
        "|---|---|---|---|---|",
    ]
    for dependency in inventory["native"]["dependencies"]:
        lines.append(
            f"| {dependency['component']} | `{dependency['revision']}` | "
            f"{dependency['license']} | "
            f"`THIRD_PARTY_LICENSES/{dependency['license_file']}` | "
            f"{dependency['repository']} |"
        )

    lines.extend(
        [
            "",
            "The MLX entry identifies the non-official IronMLX fork and its exact",
            "revision. Bundled JACCL sources are part of that checkout and are covered",
            "by the checkout's MLX license file.",
            "",
            "## Rust dependencies",
            "",
            "Scope: `ironmlx` and `iron-bench` Release binaries for",
            "`aarch64-apple-darwin`, default features, locked dependency graphs,",
            "development dependencies excluded, build dependencies retained.",
            "",
            "| Crate | Version | License expression | License text | Source |",
            "|---|---|---|---|---|",
        ]
    )
    for crate in inventory["rust"]["crates"]:
        license_files = "<br>".join(
            f"`THIRD_PARTY_LICENSES/{name}`" for name in crate["license_files"]
        )
        source = crate["repository"] or crate["source"] or "unknown"
        lines.append(
            f"| {crate['name']} | {crate['version']} | "
            f"{crate['license_expression']} | {license_files} | {source} |"
        )

    lines.extend(
        [
            "",
            "## Swift dependencies",
            "",
            "| Package | Version | Revision | License | License text | Source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for dependency in inventory["swift"]["external_packages"]:
        lines.append(
            f"| {dependency['component']} | {dependency['version']} | "
            f"`{dependency['revision']}` | {dependency['license']} | "
            f"`THIRD_PARTY_LICENSES/{dependency['license_file']}` | "
            f"{dependency['repository']} |"
        )

    lines.extend(
        [
            "",
            "## Explicit exclusions",
            "",
            "Apple system frameworks are supplied by macOS and are not copied into the",
            "App Bundle. Model weights are downloaded separately by the user and are",
            "outside this App-binary inventory; each model remains subject to its own",
            "license and usage terms.",
            "",
            "## Review boundary",
            "",
            "The generated materials preserve source license texts and detect dependency",
            "drift. Final license interpretation, attribution review, model-license policy,",
            "CycloneDX SBOM production, and authorization for public distribution remain",
            "P0-8B release gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cargo-about-json", required=True, action="append", type=Path
    )
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--mlx-source", required=True, type=Path)
    parser.add_argument("--mlx-build", required=True, type=Path)
    parser.add_argument("--swift-manifest", required=True, type=Path)
    parser.add_argument("--swift-package-json", required=True, type=Path)
    parser.add_argument("--swift-dependency-manifest", required=True, type=Path)
    parser.add_argument("--swift-package-resolved", required=True, type=Path)
    parser.add_argument("--swift-checkout-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    licenses_dir = output_root / "THIRD_PARTY_LICENSES"
    licenses_dir.mkdir(parents=True, exist_ok=False)

    cargo_about_documents = [read_json(path) for path in args.cargo_about_json]
    native_manifest = read_json(args.native_manifest)
    rust_crates, rust_licenses = rust_materials(cargo_about_documents, licenses_dir)
    native_dependencies = native_materials(
        native_manifest, args.mlx_source, args.mlx_build, licenses_dir
    )
    swift_hash = sha256_bytes(args.swift_manifest.read_bytes())
    swift_package = read_json(args.swift_package_json)
    swift_dependencies = swift_package.get("dependencies", [])
    swift_dependency_manifest = read_json(args.swift_dependency_manifest)
    if len(swift_dependencies) != len(swift_dependency_manifest["dependencies"]):
        raise ValueError(
            "Swift package manifest dependency count differs from reviewed inventory"
        )
    swift_external_packages = swift_materials(
        swift_dependency_manifest,
        read_json(args.swift_package_resolved),
        args.swift_checkout_root,
        licenses_dir,
    )

    inventory = {
        "generation": {
            "cargo_about_config": "about.toml",
            "cargo_about_mode": "offline after locked target fetch",
            "cargo_about_version": "0.9.1",
            "scope": "ironmlx and iron-bench macOS arm64 Release binaries",
        },
        "native": {
            "dependencies": native_dependencies,
            "excluded_system_components": native_manifest[
                "excluded_system_components"
            ],
        },
        "rust": {
            "crates": rust_crates,
            "license_texts": rust_licenses,
            "target": "aarch64-apple-darwin",
        },
        "schema_version": 1,
        "swift": {
            "external_packages": swift_external_packages,
            "manifest": "ironmlx-app/Package.swift",
            "manifest_sha256": swift_hash,
        },
    }
    write_text(
        output_root / "third-party-inventory.json",
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text(output_root / "THIRD_PARTY_NOTICES.md", render_notices(inventory))


if __name__ == "__main__":
    main()
