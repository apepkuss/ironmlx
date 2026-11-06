#!/usr/bin/env python3
"""Validate Chat parameters through the pinned official SDK against a test server.

The server must have a causal model loaded. This sends short generation requests;
use a dedicated validation server, not an interactive session.
"""

from __future__ import annotations

import argparse
import json

import httpx
import openai


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--default-budget", type=int, default=256)
    args = parser.parse_args()
    results = []
    errors = []
    with httpx.Client(trust_env=False, timeout=120) as http:
        client = openai.OpenAI(
            base_url=args.base_url, api_key="local-validation", http_client=http,
            max_retries=0, _strict_response_validation=True,
        )
        shared = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Count from 1 to 30, separated by commas."}],
            "temperature": 0,
            "extra_body": {"ignore_eos": True},
        }
        for parameters, expected in [
            ({"max_completion_tokens": 7, "max_tokens": 2,
              "frequency_penalty": 0.5, "presence_penalty": -0.25}, 7),
            ({}, args.default_budget),
            ({"max_completion_tokens": None, "max_tokens": 5,
              "frequency_penalty": None, "presence_penalty": 0}, 5),
        ]:
            response = client.chat.completions.create(**shared, **parameters)
            assert response.usage.completion_tokens == expected, response
            assert response.choices[0].finish_reason == "length", response
            results.append({"parameters": parameters, "completion_tokens": expected})
        chunks = list(client.chat.completions.create(
            **shared, max_completion_tokens=9, max_tokens=3,
            frequency_penalty=-0.5, presence_penalty=0.25,
            stream=True, stream_options={"include_usage": True},
        ))
        assert chunks[-1].usage.completion_tokens == 9, chunks
        assert any(chunk.choices and chunk.choices[0].finish_reason == "length" for chunk in chunks)
        for parameters, status in [
            ({"max_completion_tokens": 0, "max_tokens": 8}, 400),
            ({"frequency_penalty": 2.01}, 400),
            ({"max_completion_tokens": 2**32, "max_tokens": 1}, 413),
        ]:
            try:
                client.chat.completions.create(**shared, **parameters)
            except openai.APIStatusError as error:
                assert error.status_code == status, error
                errors.append({"parameters": parameters, "status": status, "body": error.body})
            else:
                raise AssertionError(f"request unexpectedly succeeded: {parameters}")
    print(json.dumps({"sdk": openai.__version__, "status": "passed", "valid_unary": results,
                      "stream_completion_tokens": 9, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
