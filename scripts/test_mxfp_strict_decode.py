import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mxfp_strict_decode as strict


class StrictDecodeValidationTests(unittest.TestCase):
    def test_parse_args_defaults_to_ignored_reports_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            argv = [
                "mxfp_strict_decode.py",
                "--model",
                f"toy={model_dir}",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = strict.parse_args()

        self.assertEqual(args.out_root, "reports/mxfp-strict-decode")
        self.assertEqual(args.serve_admission_deadline_ms, 5)
        self.assertEqual(args.serve_admission_queue_max, 32)
        self.assertEqual(args.serve_decode_cadence_mid_chunk_cap, 256)

    def test_accepts_full_length_requests_from_every_worker(self) -> None:
        payload = {
            "cells": [
                {
                    "n_requests": 2,
                    "itl_ms": {"p95": 12.5},
                    "finish_reason_summary": "length=2",
                }
            ],
            "raw_runs": [
                {
                    "worker_id": 0,
                    "finish_reason": "length",
                    "completion_tokens_server": 512,
                },
                {
                    "worker_id": 1,
                    "finish_reason": "length",
                    "completion_tokens_server": 512,
                },
            ],
        }

        result = strict.validate_payload(payload, max_tokens=512, concurrent=2)

        self.assertTrue(result["ok"])
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(result["itl_ms_p95"], 12.5)

    def test_uses_server_token_count_when_detokenized_chunks_are_coalesced(self) -> None:
        payload = {
            "cells": [
                {
                    "n_requests": 1,
                    "itl_ms": {"p95": 12.5},
                    "finish_reason_summary": "length=1",
                }
            ],
            "raw_runs": [
                {
                    "worker_id": 0,
                    "finish_reason": "length",
                    "completion_tokens": 40,
                    "completion_tokens_server": 512,
                }
            ],
        }

        result = strict.validate_payload(payload, max_tokens=512, concurrent=1)

        self.assertTrue(result["ok"])

    def test_rejects_early_stop(self) -> None:
        payload = {
            "cells": [{"n_requests": 1, "itl_ms": {"p95": 10.0}}],
            "raw_runs": [
                {
                    "worker_id": 0,
                    "finish_reason": "stop",
                    "completion_tokens_server": 23,
                }
            ],
        }

        result = strict.validate_payload(payload, max_tokens=512, concurrent=1)

        self.assertFalse(result["ok"])
        self.assertEqual(result["failed_requests"], 1)
        self.assertIn("finish_reason", result["errors"][0])

    def test_rejects_missing_worker_coverage(self) -> None:
        payload = {
            "cells": [{"n_requests": 1, "itl_ms": {"p95": 10.0}}],
            "raw_runs": [
                {
                    "worker_id": 0,
                    "finish_reason": "length",
                    "completion_tokens_server": 512,
                }
            ],
        }

        result = strict.validate_payload(payload, max_tokens=512, concurrent=2)

        self.assertFalse(result["ok"])
        self.assertIn("worker coverage", result["errors"][-1])


if __name__ == "__main__":
    unittest.main()
