from __future__ import annotations

import hashlib
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "generate-third-party-materials.py"
SPEC = importlib.util.spec_from_file_location("third_party_materials", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load generator: {SCRIPT_PATH}")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class BundledAssetMaterialsTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, Path]:
        asset_content = b"<svg/>\n"
        license_content = b"MIT fixture\n"
        asset_path = root / "Resources" / "agent.svg"
        license_path = root / "compliance" / "agent-license.txt"
        asset_path.parent.mkdir(parents=True)
        license_path.parent.mkdir(parents=True)
        asset_path.write_bytes(asset_content)
        license_path.write_bytes(license_content)

        manifest = {
            "schema_version": 1,
            "assets": [
                {
                    "bundled_path": "Resources/agent.svg",
                    "bundled_sha256": sha256(asset_content),
                    "component": "Agent logo",
                    "license_file": "asset-agent-mit.txt",
                    "license_sha256": sha256(license_content),
                    "license_source": "compliance/agent-license.txt",
                }
            ],
        }
        licenses_dir = root / "generated-licenses"
        licenses_dir.mkdir()
        return manifest, licenses_dir

    def test_verifies_asset_and_license_hashes_and_copies_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, licenses_dir = self.fixture(root)

            assets = GENERATOR.bundled_asset_materials(
                manifest, root, licenses_dir
            )

            self.assertEqual(assets[0]["component"], "Agent logo")
            self.assertNotIn("license_source", assets[0])
            self.assertEqual(
                (licenses_dir / "asset-agent-mit.txt").read_bytes(),
                b"MIT fixture\n",
            )

    def test_rejects_bundled_asset_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, licenses_dir = self.fixture(root)
            manifest["assets"][0]["bundled_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "bundled asset hash mismatch"):
                GENERATOR.bundled_asset_materials(manifest, root, licenses_dir)

    def test_rejects_repository_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, licenses_dir = self.fixture(root)
            manifest["assets"][0]["bundled_path"] = "../outside.svg"

            with self.assertRaisesRegex(ValueError, "escapes its root"):
                GENERATOR.bundled_asset_materials(manifest, root, licenses_dir)

    def test_rejects_license_output_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, licenses_dir = self.fixture(root)
            manifest["assets"][0]["license_file"] = "../license.txt"

            with self.assertRaisesRegex(ValueError, "invalid generated license"):
                GENERATOR.bundled_asset_materials(manifest, root, licenses_dir)


class SourceAttributionTests(unittest.TestCase):
    def test_shipped_notices_include_mpl_source(self):
        root = SCRIPT_PATH.parents[1]
        inventory = json.loads((root / "third-party-inventory.json").read_text())
        notices = GENERATOR.render_notices(inventory)
        self.assertIn("https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate", notices)
        self.assertIn("source is available under MPL-2.0", notices)

    def test_non_registry_mpl_source_requires_review(self):
        root = SCRIPT_PATH.parents[1]
        inventory = json.loads((root / "third-party-inventory.json").read_text())
        for crate in inventory["rust"]["crates"]:
            if crate["name"] == "option-ext":
                crate["source"] = "git+https://example.com/modified-option-ext"
        with self.assertRaisesRegex(ValueError, "MPL source availability"):
            GENERATOR.render_notices(inventory)


class EmbeddedLogoTests(unittest.TestCase):
    def test_embedded_logo_geometry_and_occurrence_drift(self):
        root = SCRIPT_PATH.parents[1]
        manifest = json.loads((root / "compliance/bundled-assets.json").read_text())
        assets = [a for a in manifest["assets"] if a.get("embedded_svg_marker")]
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for asset in assets:
                for key in ("bundled_path", "license_source"):
                    dest = temp / asset[key]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes((root / asset[key]).read_bytes())
            out = temp / "licenses"
            out.mkdir()
            selected = {"schema_version": 1, "assets": assets}
            self.assertEqual(len(GENERATOR.bundled_asset_materials(selected, temp, out)), 2)
            html = temp / assets[0]["bundled_path"]
            original = html.read_text()
            html.write_text(original.replace("#FF9D0B", "#000000", 1))
            with self.assertRaisesRegex(ValueError, "embedded SVG"):
                GENERATOR.bundled_asset_materials(selected, temp, out)
            html.write_text(original.replace(assets[0]["embedded_svg_marker"], "removed", 1))
            with self.assertRaisesRegex(ValueError, "embedded SVG"):
                GENERATOR.bundled_asset_materials(selected, temp, out)


if __name__ == "__main__":
    unittest.main()
