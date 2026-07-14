import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import affine56_prefill as prefill


class Affine56PrefillTests(unittest.TestCase):
    def test_parse_args_uses_clean_prefill_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = ["affine56_prefill.py", "--model", f"toy={model_dir}"]
            with mock.patch.object(sys, "argv", argv):
                args = prefill.parse_args()

        self.assertEqual(args.prompt_lens, [2048, 8192, 32768])
        self.assertEqual(args.runs, 5)
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.inter_run_cooldown_secs, 1)
        self.assertEqual(args.out_root, "reports/affine56-prefill")

    def test_validate_payload_accepts_complete_authoritative_runs(self) -> None:
        payload = {
            "metadata": {"runs_measured": 2, "warmup": 1},
            "stats": [
                {
                    "pp_target": prompt,
                    "tg_target": 1,
                    "n_runs": 2,
                    "ttft_ms_median": float(prompt),
                    "ttft_ms_p95": float(prompt) * 1.1,
                }
                for prompt in (2048, 8192)
            ],
            "raw_runs": [
                {
                    "pp_target": prompt,
                    "tg_target": 1,
                    "finish_reason": "length",
                    "completion_tokens_server": 1,
                    "ttft_ms": float(prompt),
                }
                for prompt in (2048, 8192)
                for _ in range(2)
            ],
        }

        result = prefill.validate_payload(payload, [2048, 8192], runs=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["completed_requests"], 4)
        self.assertEqual(result["cells"][2048]["ttft_ms_median"], 2048.0)

    def test_validate_payload_rejects_missing_prompt_cell(self) -> None:
        payload = {
            "metadata": {"runs_measured": 1, "warmup": 1},
            "stats": [
                {
                    "pp_target": 2048,
                    "tg_target": 1,
                    "n_runs": 1,
                    "ttft_ms_median": 10.0,
                    "ttft_ms_p95": 11.0,
                }
            ],
            "raw_runs": [
                {
                    "pp_target": 2048,
                    "tg_target": 1,
                    "finish_reason": "length",
                    "completion_tokens_server": 1,
                    "ttft_ms": 10.0,
                }
            ],
        }

        result = prefill.validate_payload(payload, [2048, 8192], runs=1)

        self.assertFalse(result["ok"])
        self.assertTrue(any("prompt coverage" in error for error in result["errors"]))

    def test_validate_payload_rejects_missing_server_usage(self) -> None:
        payload = {
            "metadata": {"runs_measured": 1, "warmup": 1},
            "stats": [
                {
                    "pp_target": 2048,
                    "tg_target": 1,
                    "n_runs": 1,
                    "ttft_ms_median": 10.0,
                    "ttft_ms_p95": 11.0,
                }
            ],
            "raw_runs": [
                {
                    "pp_target": 2048,
                    "tg_target": 1,
                    "finish_reason": "length",
                    "completion_tokens_server": None,
                    "ttft_ms": 10.0,
                }
            ],
        }

        result = prefill.validate_payload(payload, [2048], runs=1)

        self.assertFalse(result["ok"])
        self.assertTrue(any("completion_tokens_server" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
