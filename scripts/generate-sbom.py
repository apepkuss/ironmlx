#!/usr/bin/env python3
"""Generate a deterministic CycloneDX SBOM for the IronMLX release inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from urllib.parse import urlparse


SCHEMA_VERSION = "1"
CYCLONEDX_VERSION = "1.6"
NAMESPACE = uuid.UUID("f6f8e2ad-44c8-4ab2-b0ce-7fd9ec6a9234")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def purl_for_github(repository: str, version: str) -> str | None:
    parsed = urlparse(repository)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        return None
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if path.count("/") != 1:
        return None
    return f"pkg:github/{path}@{version}"


def license_entry(expression: str) -> dict[str, object]:
    # Keep non-SPDX attribution notes as a named license instead of emitting an
    # invalid SPDX expression into the CycloneDX document.
    if "bundled third-party notices" in expression:
        return {"license": {"name": expression}}
    return {"expression": expression}


def property_entry(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def cargo_component(crate: dict) -> dict:
    name = crate["name"]
    version = crate["version"]
    bom_ref = f"cargo:{name}@{version}"
    component = {
        "type": "library",
        "bom-ref": bom_ref,
        "group": "crates.io",
        "name": name,
        "version": version,
        "licenses": [license_entry(crate["license_expression"])],
        "properties": [
            property_entry("ironmlx:source", crate.get("source", "")),
            property_entry(
                "ironmlx:license-files", ",".join(sorted(crate.get("license_files", [])))
            ),
        ],
        "purl": f"pkg:cargo/{name}@{version}",
    }
    repository = crate.get("repository")
    if repository:
        component["externalReferences"] = [
            {"type": "vcs", "url": repository},
        ]
    return component


def native_component(dependency: dict) -> dict:
    name = dependency["component"]
    version = dependency["revision"]
    bom_ref = f"native:{name}@{version}"
    component = {
        "type": "library",
        "bom-ref": bom_ref,
        "name": name,
        "version": version,
        "licenses": [license_entry(dependency["license"])],
        "properties": [
            property_entry("ironmlx:license-file", dependency["license_file"]),
            property_entry("ironmlx:license-sha256", dependency["license_sha256"]),
        ],
    }
    source_verification = dependency.get("source_integrity", {})
    if source_verification.get("commit"):
        component["properties"].append(
            property_entry("ironmlx:source-commit", source_verification["commit"])
        )
    if source_verification.get("sha256"):
        component["properties"].append(
            property_entry("ironmlx:source-sha256", source_verification["sha256"])
        )
    repository = dependency.get("repository")
    if repository:
        component["externalReferences"] = [{"type": "vcs", "url": repository}]
        purl = purl_for_github(repository, version)
        if purl:
            component["purl"] = purl
    return component


def swift_component(dependency: dict) -> dict:
    name = dependency["component"]
    version = dependency["version"]
    revision = dependency["revision"]
    bom_ref = f"swift:{dependency['identity']}@{version}"
    component = {
        "type": "library",
        "bom-ref": bom_ref,
        "name": name,
        "version": version,
        "licenses": [license_entry(dependency["license"])],
        "properties": [
            property_entry("ironmlx:identity", dependency["identity"]),
            property_entry("ironmlx:revision", revision),
            property_entry("ironmlx:license-file", dependency["license_file"]),
        ],
        "externalReferences": [
            {"type": "vcs", "url": dependency["repository"]},
        ],
    }
    purl = purl_for_github(dependency["repository"], revision)
    if purl:
        component["purl"] = purl
    return component


def asset_component(asset: dict) -> dict:
    name = asset["component"]
    revision = asset["revision"]
    bom_ref = f"asset:{name}@{revision}"
    return {
        "type": "file",
        "bom-ref": bom_ref,
        "name": name,
        "version": revision,
        "licenses": [license_entry(asset["license"])],
        "hashes": [{"alg": "SHA-256", "content": asset["bundled_sha256"]}],
        "properties": [
            property_entry("ironmlx:bundled-path", asset["bundled_path"]),
            property_entry("ironmlx:source-path", asset["source_path"]),
            property_entry("ironmlx:license-file", asset["license_file"]),
        ],
        "externalReferences": [
            {"type": "distribution", "url": asset["source_url"]},
            {"type": "vcs", "url": asset["repository"]},
        ],
    }


def build_bom(root: Path) -> dict:
    inventory = json.loads((root / "third-party-inventory.json").read_text())
    product_version = (root / "VERSION").read_text().strip()
    components = []

    for crate in inventory["rust"]["crates"]:
        components.append(cargo_component(crate))
    for dependency in inventory["native"]["dependencies"]:
        components.append(native_component(dependency))
    for dependency in inventory["swift"]["external_packages"]:
        components.append(swift_component(dependency))
    for asset in inventory["bundled_assets"]["assets"]:
        components.append(asset_component(asset))

    components.sort(key=lambda component: component["bom-ref"])
    component_digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    serial = f"urn:uuid:{uuid.uuid5(NAMESPACE, component_digest)}"
    metadata_component = {
        "type": "application",
        "bom-ref": "application:ironmlx",
        "name": "IronMLX",
        "version": product_version,
        "licenses": [license_entry("Apache-2.0")],
        "properties": [
            property_entry("ironmlx:target", inventory["rust"]["target"]),
            property_entry("ironmlx:inventory-schema", str(inventory["schema_version"])),
        ],
        "externalReferences": [
            {"type": "vcs", "url": "https://github.com/apepkuss/ironmlx"},
        ],
    }
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_VERSION,
        "serialNumber": serial,
        "version": int(SCHEMA_VERSION),
        "metadata": {"component": metadata_component},
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=repository_root() / "SBOM.cdx.json")
    args = parser.parse_args()
    root = repository_root()
    bom = build_bom(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bom, indent=2, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
