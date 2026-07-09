import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import quant_validation_matrix as qvm


class QuantValidationMatrixArgsTests(unittest.TestCase):
    def test_parse_args_accepts_multiple_concurrency_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "quant_validation_matrix.py",
                "--model",
                f"toy={model_dir}",
                "--concurrent",
                "4,8",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = qvm.parse_args()

        self.assertEqual(args.concurrent, [4, 8])

    def test_parse_args_accepts_dedicated_concurrent_prompt_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "quant_validation_matrix.py",
                "--model",
                f"toy={model_dir}",
                "--concurrent-prompt-lens",
                "8192,32768",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = qvm.parse_args()

        self.assertEqual(args.concurrent_prompt_lens, [8192, 32768])


if __name__ == "__main__":
    unittest.main()
