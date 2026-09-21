"""Offline converter regression tests; no model downloads or snapshot changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
import convert_indextts25 as conversion


class ConversionTests(unittest.TestCase):
    def test_bit_preservation_and_reproducible_bytes(self):
        tensors = {"x": torch.tensor([0., -0., 1.25], dtype=torch.float32),
                   "counter": torch.tensor([2**60 + 1], dtype=torch.int64)}
        expected = conversion.schema(tensors)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = conversion.write_component(root, "one", tensors, expected)
            second = conversion.write_component(root, "two", tensors, expected)
            self.assertEqual(first["sha256"], second["sha256"])
            restored = load_file(str(root / "one"))
            for name in tensors:
                self.assertEqual(tensors[name].numpy().tobytes(), restored[name].numpy().tobytes())

    def test_invalid_values_and_schema_fail_before_write(self):
        for tensors, schema in [
            ({"x": torch.tensor([float("nan")])}, {"x": {"dtype": "F32", "shape": [1]}}),
            ({"w2v_var": torch.tensor([0.])}, {"w2v_var": {"dtype": "F32", "shape": [1]}}),
            ({"x.running_var": torch.tensor([-1.])}, {"x.running_var": {"dtype": "F32", "shape": [1]}}),
            ({"x": torch.tensor([1.])}, {"x": {"dtype": "F32", "shape": [2]}}),
        ]:
            with self.subTest(tensors=tensors), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    conversion.write_component(Path(directory), "output", tensors, schema)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_conversion_removes_stage_and_lock(self):
        # Inject a checkpoint-read failure after source verification and staging.
        # The publication/cleanup path is the real converter path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "derived"
            with patch.object(conversion, "check_file", side_effect=lambda path, expected: expected), \
                 patch.object(torch, "load", side_effect=RuntimeError("injected read failure")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    conversion.convert(root, root, root, output)
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_destination_and_lock_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "derived"
            output.mkdir()
            (output / "sentinel").write_text("retain")
            with patch.object(conversion, "check_file", side_effect=lambda path, expected: expected):
                with self.assertRaises(FileExistsError):
                    conversion.convert(root, root, root, output)
                lock = root / "second.conversion-lock"
                lock.write_text("other owner")
                with self.assertRaises(FileExistsError):
                    conversion.convert(root, root, root, root / "second")
            self.assertEqual((output / "sentinel").read_text(), "retain")
            self.assertEqual(lock.read_text(), "other owner")


if __name__ == "__main__":
    unittest.main()
