#!/usr/bin/env python3
"""Pinned official-SDK black-box contract tests for IronMLX public APIs."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import anthropic
import openai


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_FIXTURES = REPO_ROOT / "ironmlx/tests/fixtures/api_contract_sdk"
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def response_object(
    model: str,
    *,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": "resp_contract",
        "object": "response",
        "created_at": 0,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 64,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": "low", "summary": "auto"},
        "service_tier": "default",
        "store": False,
        "temperature": 0.2,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.9,
        "truncation": "disabled",
        "usage": usage,
    }


def response_usage() -> dict[str, Any]:
    return {
        "input_tokens": 8,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 12,
    }


def response_message(text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_contract",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": text,
            }
        ],
    }


def sse(event: str | None, payload: dict[str, Any] | str) -> str:
    prefix = "" if event is None else f"event: {event}\n"
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"{prefix}data: {data}\n\n"


class ContractHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        body = json.loads(raw)
        self.requests.append(
            {
                "path": self.path,
                "body": body,
                "headers": {key.lower(): value for key, value in self.headers.items()},
            }
        )
        model = body.get("model", "")
        if model.endswith("-400"):
            self._error(400, "invalid_request", "contract bad request")
            return
        if model.endswith("-413"):
            self._error(413, "request_token_capacity_exceeded", "contract request too large")
            return
        if model.endswith("-503"):
            self._error(503, "scheduler_queue_full", "contract overloaded", retry_after="5")
            return

        if self.path == "/v1/chat/completions":
            self._chat(body)
        elif self.path == "/v1/responses":
            self._responses(body)
        elif self.path == "/v1/messages":
            self._messages(body)
        else:
            self._json(404, {"error": {"message": "not found"}})

    def _json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _event_stream(self, frames: list[str]) -> None:
        encoded = "".join(frames).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, code: str, message: str, retry_after: str | None = None) -> None:
        headers = {"request-id": "req_contract"}
        if retry_after is not None:
            headers["retry-after"] = retry_after
        if self.path == "/v1/messages":
            kind = {
                400: "invalid_request_error",
                413: "request_too_large",
                503: "overloaded_error",
            }[status]
            payload = {
                "type": "error",
                "error": {"type": kind, "message": message, "code": code},
                "request_id": "req_contract",
            }
        else:
            payload = {
                "error": {
                    "message": message,
                    "type": "server_error" if status == 503 else "invalid_request_error",
                    "param": None,
                    "code": code,
                }
            }
        self._json(status, payload, headers=headers)

    def _chat(self, body: dict[str, Any]) -> None:
        model = body["model"]
        if body.get("stream"):
            base = {
                "id": "chatcmpl-contract",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
            }
            frames = [
                sse(
                    None,
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    },
                ),
                sse(
                    None,
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "hello"},
                                "finish_reason": None,
                            }
                        ],
                    },
                ),
                sse(
                    None,
                    {
                        **base,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ),
                sse(None, "[DONE]"),
            ]
            self._event_stream(frames)
            return
        if model == "contract-chat-tool":
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_contract",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": '{"answer":"sunny"}'}
            finish_reason = "stop"
        self._json(
            200,
            {
                "id": "chatcmpl-contract",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    def _responses(self, body: dict[str, Any]) -> None:
        model = body["model"]
        if body.get("stream"):
            created = response_object(model, status="in_progress", output=[], usage=None)
            message = response_message("", status="in_progress")
            completed_message = response_message("hello")
            completed = response_object(
                model,
                status="completed",
                output=[response_message("hello")],
                usage=response_usage(),
            )
            frames = [
                sse(
                    "response.created",
                    {"type": "response.created", "sequence_number": 0, "response": created},
                ),
                sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": message,
                    },
                ),
                sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "sequence_number": 2,
                        "output_index": 0,
                        "item_id": "msg_contract",
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": "",
                        },
                    },
                ),
                sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 3,
                        "output_index": 0,
                        "item_id": "msg_contract",
                        "content_index": 0,
                        "logprobs": [],
                        "delta": "hello",
                    },
                ),
                sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "sequence_number": 4,
                        "output_index": 0,
                        "item_id": "msg_contract",
                        "content_index": 0,
                        "logprobs": [],
                        "text": "hello",
                    },
                ),
                sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "sequence_number": 5,
                        "output_index": 0,
                        "item_id": "msg_contract",
                        "content_index": 0,
                        "part": completed_message["content"][0],
                    },
                ),
                sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "sequence_number": 6,
                        "output_index": 0,
                        "item": completed_message,
                    },
                ),
                sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "sequence_number": 7,
                        "response": completed,
                    },
                ),
            ]
            self._event_stream(frames)
            return
        if model == "contract-responses-tool":
            output = [
                {
                    "type": "function_call",
                    "id": "fc_contract",
                    "status": "completed",
                    "arguments": '{"city":"Tokyo"}',
                    "call_id": "call_contract",
                    "name": "get_weather",
                }
            ]
        else:
            output = [
                {
                    "type": "reasoning",
                    "id": "rs_contract",
                    "summary": [{"type": "summary_text", "text": "checked"}],
                    "content": [{"type": "reasoning_text", "text": "brief"}],
                },
                response_message('{"answer":"sunny"}'),
            ]
        self._json(
            200,
            response_object(
                model,
                status="completed",
                output=output,
                usage=response_usage(),
            ),
        )

    def _messages(self, body: dict[str, Any]) -> None:
        model = body["model"]
        if body.get("stream"):
            frames = [
                sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_contract",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 8, "output_tokens": 0},
                        },
                    },
                ),
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hello"},
                    },
                ),
                sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                ),
                sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 4},
                    },
                ),
                sse("message_stop", {"type": "message_stop"}),
            ]
            self._event_stream(frames)
            return
        if model == "contract-messages-tool":
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_contract",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                }
            ]
            stop_reason = "tool_use"
            output_tokens_details = None
        else:
            content = [
                {
                    "type": "thinking",
                    "thinking": "brief",
                    "signature": "ironmlx-contract",
                },
                {"type": "text", "text": '{"answer":"sunny"}'},
            ]
            stop_reason = "end_turn"
            output_tokens_details = {"thinking_tokens": 1}
        usage: dict[str, Any] = {"input_tokens": 8, "output_tokens": 4}
        if output_tokens_details is not None:
            usage["output_tokens_details"] = output_tokens_details
        self._json(
            200,
            {
                "id": "msg_contract",
                "type": "message",
                "role": "assistant",
                "content": content,
                "model": model,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": usage,
            },
            headers={"request-id": "req_contract"},
        )


class FixtureServer:
    def __enter__(self) -> tuple[str, list[dict[str, Any]]]:
        ContractHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ContractHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}", ContractHandler.requests

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def read_fixture(name: str) -> dict[str, Any]:
    return json.loads((REQUEST_FIXTURES / name).read_text())


def request_for(requests: list[dict[str, Any]], model: str) -> dict[str, Any]:
    matches = [request for request in requests if request["body"].get("model") == model]
    assert len(matches) == 1, f"expected one request for {model}, got {len(matches)}"
    return matches[0]


def assert_error(call: Callable[[], object], status: int, code: str, retry_after: str | None) -> None:
    try:
        call()
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        assert error.status_code == status, error
        assert isinstance(error.body, dict), error.body
        body = error.body
        if "error" in body and isinstance(body["error"], dict):
            parsed_code = body["error"].get("code")
        else:
            parsed_code = body.get("code")
        assert parsed_code == code, body
        assert error.response.headers.get("retry-after") == retry_after
    else:
        raise AssertionError(f"expected HTTP {status}")


def run_contract(base_url: str, requests: list[dict[str, Any]]) -> None:
    openai_client = openai.OpenAI(
        api_key="contract-openai-key",
        base_url=f"{base_url}/v1",
        max_retries=0,
    )
    anthropic_client = anthropic.Anthropic(
        api_key="contract-anthropic-key",
        base_url=base_url,
        max_retries=0,
    )

    chat = openai_client.chat.completions.create(
        model="contract-chat",
        messages=[{"role": "user", "content": "Return the weather as JSON or call the tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": TOOL_SCHEMA,
                    "strict": True,
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "weather_answer", "schema": WEATHER_SCHEMA, "strict": True},
        },
        max_tokens=64,
        temperature=0.2,
        top_p=0.9,
        stream=False,
    )
    assert chat.choices[0].message.content == '{"answer":"sunny"}'
    assert chat.usage is not None and chat.usage.total_tokens == 12

    chat_tool = openai_client.chat.completions.create(
        model="contract-chat-tool",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": TOOL_SCHEMA},
            }
        ],
    )
    call = chat_tool.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Tokyo"}

    chat_stream = openai_client.chat.completions.create(
        model="contract-chat-stream",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    assert "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chat_stream
        if chunk.choices
    ) == "hello"

    response = openai_client.responses.create(
        model="contract-responses",
        input="Return the weather as JSON or call the tool.",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": TOOL_SCHEMA,
                "strict": True,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "weather_answer",
                "schema": WEATHER_SCHEMA,
                "strict": True,
            }
        },
        reasoning={"effort": "low", "summary": "auto"},
        max_output_tokens=64,
        temperature=0.2,
        top_p=0.9,
        store=False,
        stream=False,
    )
    assert response.output_text == '{"answer":"sunny"}'
    assert response.output[0].type == "reasoning"

    response_tool = openai_client.responses.create(
        model="contract-responses-tool",
        input="weather?",
        tools=[{"type": "function", "name": "get_weather", "parameters": TOOL_SCHEMA}],
    )
    assert response_tool.output[0].type == "function_call"
    assert response_tool.output[0].name == "get_weather"
    assert json.loads(response_tool.output[0].arguments) == {"city": "Tokyo"}

    response_stream = openai_client.responses.create(
        model="contract-responses-stream",
        input="hello",
        stream=True,
    )
    deltas = [event.delta for event in response_stream if event.type == "response.output_text.delta"]
    assert "".join(deltas) == "hello"

    message = anthropic_client.messages.create(
        model="contract-messages",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": "Return the weather as JSON or call the tool."}
        ],
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": TOOL_SCHEMA,
                "strict": True,
            }
        ],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        output_config={
            "format": {"type": "json_schema", "schema": WEATHER_SCHEMA},
            "effort": "low",
        },
        thinking={"type": "adaptive", "display": "summarized"},
        temperature=0.2,
        top_p=0.9,
        top_k=32,
        stream=False,
    )
    assert message.content[0].type == "thinking"
    assert message.content[1].type == "text"
    assert message.content[1].text == '{"answer":"sunny"}'

    message_tool = anthropic_client.messages.create(
        model="contract-messages-tool",
        max_tokens=64,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"name": "get_weather", "input_schema": TOOL_SCHEMA}],
    )
    assert message_tool.content[0].type == "tool_use"
    assert message_tool.content[0].name == "get_weather"
    assert message_tool.content[0].input == {"city": "Tokyo"}

    with anthropic_client.messages.stream(
        model="contract-messages-stream",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
    ) as stream:
        assert "".join(stream.text_stream) == "hello"
        final_message = stream.get_final_message()
    assert final_message.stop_reason == "end_turn"

    assert_error(
        lambda: openai_client.chat.completions.create(
            model="contract-chat-400",
            messages=[{"role": "user", "content": "x"}],
        ),
        400,
        "invalid_request",
        None,
    )
    assert_error(
        lambda: openai_client.responses.create(model="contract-responses-413", input="x"),
        413,
        "request_token_capacity_exceeded",
        None,
    )
    assert_error(
        lambda: openai_client.responses.create(model="contract-responses-503", input="x"),
        503,
        "scheduler_queue_full",
        "5",
    )
    assert_error(
        lambda: anthropic_client.messages.create(
            model="contract-messages-400",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        ),
        400,
        "invalid_request",
        None,
    )
    assert_error(
        lambda: anthropic_client.messages.create(
            model="contract-messages-413",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        ),
        413,
        "request_token_capacity_exceeded",
        None,
    )
    assert_error(
        lambda: anthropic_client.messages.create(
            model="contract-messages-503",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        ),
        503,
        "scheduler_queue_full",
        "5",
    )

    for model, fixture, path in [
        ("contract-chat", "chat_request.json", "/v1/chat/completions"),
        ("contract-responses", "responses_request.json", "/v1/responses"),
        ("contract-messages", "messages_request.json", "/v1/messages"),
    ]:
        request = request_for(requests, model)
        assert request["path"] == path
        assert request["body"] == read_fixture(fixture), (
            fixture,
            request["body"],
            read_fixture(fixture),
        )

    assert request_for(requests, "contract-chat")["headers"]["authorization"] == (
        "Bearer contract-openai-key"
    )
    anthropic_headers = request_for(requests, "contract-messages")["headers"]
    assert anthropic_headers["x-api-key"] == "contract-anthropic-key"
    assert "anthropic-version" in anthropic_headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="run against the deterministic in-process contract server",
    )
    args = parser.parse_args()
    if not args.fixture:
        parser.error("--fixture is required; live-model acceptance is a separate release gate")
    with FixtureServer() as (base_url, requests):
        run_contract(base_url, requests)
    print(
        "API SDK contract passed: "
        f"openai={openai.__version__}, anthropic={anthropic.__version__}, requests={len(requests)}"
    )


if __name__ == "__main__":
    main()
