from __future__ import annotations

import hashlib
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


if __name__ == "__main__":
    unittest.main()
