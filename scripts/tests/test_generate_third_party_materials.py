from __future__ import annotations

import hashlib
import json
import importlib.util
import shutil
import subprocess
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


class RustMaterialsTests(unittest.TestCase):
    def test_normalizes_license_line_endings_before_hashing_and_writing(self) -> None:
        package_id = "mime_guess 2.0.5 (registry+https://github.com/rust-lang/crates.io-index)"
        cargo_about = {
            "licenses": [
                {
                    "id": "MIT",
                    "text": "MIT fixture\r\n\r\nPermission granted.\r\n",
                    "used_by": [{"crate": {"id": package_id}}],
                }
            ],
            "crates": [
                {
                    "package": {
                        "id": package_id,
                        "name": "mime_guess",
                        "version": "2.0.5",
                        "repository": "https://example.com/mime_guess",
                        "source": "registry+https://github.com/rust-lang/crates.io-index",
                    },
                    "license": "MIT",
                }
            ],
        }
        normalized = b"MIT fixture\n\nPermission granted.\n"

        with tempfile.TemporaryDirectory() as directory:
            licenses_dir = Path(directory)
            crates, licenses = GENERATOR.rust_materials([cargo_about], licenses_dir)

            filename = f"rust-license-{sha256(normalized)[:16]}.txt"
            self.assertEqual(crates[0]["license_files"], [filename])
            self.assertEqual(licenses[0]["sha256"], sha256(normalized))
            self.assertEqual((licenses_dir / filename).read_bytes(), normalized)


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


class VendoredNativeSourceTests(unittest.TestCase):
    def git_fixture(self, root: Path) -> tuple[Path, str]:
        import subprocess

        upstream = root / "upstream"
        (upstream / "mlx/kernels").mkdir(parents=True)
        (upstream / "mlx/kernels/a.h").write_text("a\n")
        (upstream / "mlx/kernels/b.h").write_text("b\n")
        for command in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
        ):
            subprocess.run(["git", "-C", str(upstream), *command], check=True)
        commit = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        vendored = root / "repo/vendor/include/mlx/kernels"
        vendored.mkdir(parents=True)
        (vendored / "a.h").write_text("a\n")
        return upstream, commit

    def test_vendored_files_match_the_listed_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, commit = self.git_fixture(root)
            result = GENERATOR.verify_native_source(
                {
                    "commit": commit,
                    "repository": "mlx:.",
                    "source": "repo:vendor/include",
                    "type": "git-files",
                },
                upstream,
                root / "build",
                root / "repo",
            )
            self.assertEqual(result["files"], 1)
            self.assertEqual(result["commit"], commit)
            self.assertEqual(len(result["tree_sha256"]), 64)

    def test_rejects_vendored_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, commit = self.git_fixture(root)
            (root / "repo/vendor/include/mlx/kernels/a.h").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "vendored file differs"):
                GENERATOR.verify_native_source(
                    {
                        "commit": commit,
                        "repository": "mlx:.",
                        "source": "repo:vendor/include",
                        "type": "git-files",
                    },
                    upstream,
                    root / "build",
                    root / "repo",
                )

    def test_repository_sources_require_a_repository_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported native license source prefix"):
            GENERATOR.resolve_native_source("repo:x", Path("/m"), Path("/b"))

    def test_reports_missing_commit_separately_from_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, _ = self.git_fixture(root)
            with self.assertRaisesRegex(ValueError, "verification commit is unavailable"):
                GENERATOR.verify_git_files(
                    root / "repo/vendor/include", upstream, "0" * 40
                )

    def test_reports_missing_upstream_file_separately_from_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, commit = self.git_fixture(root)
            (root / "repo/vendor/include/mlx/kernels/missing.h").write_text("missing\n")
            with self.assertRaisesRegex(ValueError, "cannot read verification file"):
                GENERATOR.verify_git_files(root / "repo/vendor/include", upstream, commit)


class ReleaseMLXCheckoutTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, str, str]:
        fork, build_commit = VendoredNativeSourceTests().git_fixture(root / "fork")
        upstream, upstream_commit = VendoredNativeSourceTests().git_fixture(root / "upstream")
        # Make the verification commit distinct from the build commit, as in CI.
        subprocess.run(
            ["git", "-C", str(upstream), "-c", "user.name=t", "-c",
             "user.email=t@t", "commit", "--allow-empty", "-qm", "verification"],
            check=True,
        )
        upstream_commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        repository = root / "product"
        scripts = repository / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(SCRIPT_PATH.parent / "checkout-release-mlx.sh", scripts / "checkout-release-mlx.sh")
        (scripts / "release-config.sh").write_text(
            f'IRONMLX_MLX_REPOSITORY="{fork}"\nIRONMLX_MLX_COMMIT="{build_commit}"\n'
        )
        manifest = repository / "compliance/native-dependencies.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"dependencies": [
            {"repository": str(source), "source_verification": {
                "type": "git-files", "repository": "mlx:.", "commit": commit,
            }}
            for source, commit in ((fork, build_commit), (upstream, upstream_commit))
        ]}))
        return repository, upstream, build_commit, upstream_commit

    def test_clean_shallow_checkout_includes_verification_commit_without_changing_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, _, build_commit, upstream_commit = self.fixture(root)
            destination = root / "checkout"
            subprocess.run(
                ["bash", str(repository / "scripts/checkout-release-mlx.sh"), str(destination)],
                check=True, capture_output=True, text=True,
            )
            def git(*args: str) -> str:
                return subprocess.check_output(
                    ["git", "-C", str(destination), *args], text=True
                ).strip()
            self.assertEqual(git("rev-parse", "HEAD"), build_commit)
            self.assertEqual(git("rev-parse", "--is-shallow-repository"), "true")
            self.assertEqual(git("status", "--porcelain"), "")
            self.assertEqual(git("cat-file", "-t", upstream_commit), "commit")
            result = GENERATOR.verify_git_files(
                root / "upstream/repo/vendor/include", destination, upstream_commit
            )
            self.assertEqual(result["files"], 1)

    def test_unavailable_verification_commit_fails_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, _, _, _ = self.fixture(root)
            manifest = repository / "compliance/native-dependencies.json"
            data = json.loads(manifest.read_text())
            data["dependencies"][1]["source_verification"]["commit"] = "0" * 40
            manifest.write_text(json.dumps(data))
            result = subprocess.run(
                ["bash", str(repository / "scripts/checkout-release-mlx.sh"), str(root / "checkout")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("MLX checkout ready:", result.stdout)


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
