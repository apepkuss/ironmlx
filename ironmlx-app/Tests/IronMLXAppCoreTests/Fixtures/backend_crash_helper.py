#!/usr/bin/env python3
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


loaded_models = []


def model_info(body):
    model = body.get("model") or body.get("model_dir") or "test-model"
    is_diffusion = "DiffusionGemma" in model
    return {
        "id": model,
        "model": model,
        "path": body.get("model_dir") or model,
        "architecture": "diffusion_gemma" if is_diffusion else "llm",
        "default": bool(body.get("set_default")),
        "max_position_embeddings": 4096,
        "pinned": bool(body.get("pinned")),
        "mtp_enabled": bool(body.get("mtp_model_dir")),
        "mtp_model_dir": body.get("mtp_model_dir"),
        "mtp_draft_tokens": body.get("mtp_draft_tokens"),
        "prompt_lookup": body.get("prompt_lookup"),
        "runtime_kind": "block_diffusion" if is_diffusion else "causal",
        "supports_streaming": True,
        "supports_vision": is_diffusion,
        "supports_mtp": False,
        "supports_prompt_lookup": not is_diffusion,
        "supports_speculative_decoding": False,
        "supports_kv_cache": not is_diffusion,
        "supported_sampling_parameters": (
            ["max_tokens", "temperature", "seed"]
            if is_diffusion
            else [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
            ]
        ),
        "runtime_state": "loaded",
        "scheduler": "serial_block_diffusion" if is_diffusion else "continuous_batching",
        "active_requests": 0,
        "queued_requests": 0,
        "queue_capacity": 8,
        "usage": {
            "cumulative_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "prefix_cache": None
            if is_diffusion
            else {"hit_tokens": 0, "eligible_tokens": 0},
            "performance": {
                "window_seconds": 60,
                "completed_requests": 0,
                "prefill_tokens_per_second": None,
                "decode_tokens_per_second": None,
                "ttft_ms": None,
            },
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print("helper request " + (format % args), flush=True)

    def json_response(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            self.json_response({
                "status": "ok",
                "mode": "app_daemon",
                "models": loaded_models,
                "uptime_secs": 1,
                "model": {"name": ",".join(m["id"] for m in loaded_models), "max_position_embeddings": 4096},
                "scheduler": {
                    "b_max": 1,
                    "b_active": 1,
                    "b_queued": 0,
                    "queue_max": 8,
                    "admission_queue_full_count": 0,
                    "memory_budget_exceeded_count": 0,
                },
                "memory": {
                    "total_ram_bytes": 1,
                    "free_ram_bytes": 1,
                    "kv_cache_active_bytes": 0,
                    "kv_cache_soft_limit_bytes": 0,
                    "kv_cache_logical_cap_tokens": 0,
                    "kv_cache_resident_cap_tokens": 0,
                    "kv_cache_budget_policy": "test",
                    "mlx_active_bytes": 0,
                    "mlx_cache_bytes": 0,
                    "mlx_peak_bytes": 0,
                    "mlx_memory_limit_bytes": 0,
                },
                "active_kv_offload": {
                    "enabled": False,
                    "status": "disabled",
                    "active": False,
                    "degraded": False,
                    "mode": "disabled",
                    "resident_pages": 0,
                    "offloaded_pages": 0,
                    "loading_pages": 0,
                    "dirty_pages": 0,
                    "parked_requests": 0,
                    "offloaded_bytes": 0,
                    "swap_out_count": 0,
                    "swap_in_count": 0,
                    "swap_error_count": 0,
                    "last_swap_out_us": 0,
                    "last_swap_in_us": 0,
                    "supported_cache_kinds": [],
                    "not_applicable_cache_kinds": [],
                },
                "version": "test",
            })
        elif self.path == "/health":
            self.json_response({"status": "ok"})
        elif self.path == "/admin/api/models/loaded":
            self.json_response(loaded_models)
        else:
            self.json_response({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/work":
            print("helper work started", flush=True)
            time.sleep(30)
            self.json_response({"success": True})
            return
        model = body.get("model") or ""
        if "DiffusionGemma" in model:
            incompatible = [
                "max_cache_cap",
                "mtp_model_dir",
                "mtp_draft_tokens",
                "prompt_lookup",
                "top_p",
                "top_k",
                "repetition_penalty",
            ]
            present = [key for key in incompatible if key in body]
            if present:
                self.json_response({
                    "success": False,
                    "status": "error",
                    "code": "diffusion_gemma_test_incompatible_configuration",
                    "error": "DiffusionGemma request included causal-only fields: " + ",".join(present),
                }, 400)
                return
        if self.path.endswith("/register"):
            self.json_response({
                "success": True,
                "status": "registered",
                "model": body.get("model"),
                "loaded_models": loaded_models,
            })
            return
        if self.path.endswith("/load"):
            candidate = model_info(body)
            for existing in loaded_models:
                existing["default"] = False if candidate["default"] else existing["default"]
            loaded_models[:] = [model for model in loaded_models if model["id"] != candidate["id"]]
            loaded_models.append(candidate)
            self.json_response({
                "success": True,
                "status": "loaded",
                "model": candidate["id"],
                "loaded_models": loaded_models,
            })
            return
        self.json_response({
            "success": True,
            "status": "ok",
            "loaded_models": loaded_models,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"helper ready pid={server.server_address} port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
