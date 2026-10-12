#!/usr/bin/env python3
"""Official OpenAI Python SDK checks for `/v1/decisions` against a live server.

The server must have aac6fef/laya-multilingual-mlx available; results come
from real Laya inference. Uses its own pinned SDK (decisions need openai>=3.26),
separate from ../requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import openai
from openai.types.decision import (
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceScore,
    Decision,
)

MODEL = "aac6fef/laya-multilingual-mlx"
EVIDENCE = '{"message": "发票被重复扣款，请退款。"}'
QUESTIONS: list[dict[str, Any]] = [
    {
        "type": "score",
        "name": "urgency",
        "instructions": "How urgent is this?",
        "levels": [{"label": "can wait"}, {"label": "soon"}, {"label": "today"}],
    },
    {
        "type": "choice",
        "name": "department",
        "instructions": "Which team should handle this?",
        "choices": [
            {"value": "billing", "description": "refunds"},
            {"value": "technical", "description": "bugs"},
        ],
    },
    {"type": "predicate", "name": "refund", "instructions": "Is a refund requested?"},
]
SYSTEM_ONE = {
    "model": MODEL,
    "state": {"message": "发票被重复扣款，请退款。"},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "refunds", "technical": "bugs"},
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "soon", "today"],
        },
    },
}


def client(args: argparse.Namespace, api_key: str) -> openai.OpenAI:
    http_client = openai.DefaultHttpxClient(verify=False) if args.insecure else None
    return openai.OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        max_retries=0,
        timeout=120,
        http_client=http_client,
        _strict_response_validation=True,
    )


def post_system_one(args: argparse.Namespace) -> dict[str, Any]:
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/systemone",
        data=json.dumps(SYSTEM_ONE).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
    )
    context = ssl._create_unverified_context() if args.insecure else None
    with urllib.request.urlopen(request, timeout=120, context=context) as response:
        return json.load(response)


def expect_error(kind: type[openai.APIStatusError], code: str, param: str | None, call) -> dict[str, Any]:
    try:
        call()
    except kind as error:
        assert error.code == code, (error.code, error.body)
        assert error.param == param, (error.param, error.body)
        return {"status": error.status_code, "type": error.type, "code": error.code, "param": error.param}
    raise AssertionError(f"expected {kind.__name__} {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9068/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--wrong-api-key", help="also expect 401 for this key")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed LAN)")
    parser.add_argument("--skip-system-one", action="store_true")
    parser.add_argument("--evidence", help="write responses to this JSON file")
    args = parser.parse_args()
    sdk = client(args, args.api_key)
    evidence: dict[str, Any] = {"sdk": openai.__version__, "base_url": args.base_url}

    decision = sdk.decisions.create(model=MODEL, input=EVIDENCE, questions=QUESTIONS)
    assert isinstance(decision, Decision)
    score, choice, predicate = decision.answers
    assert isinstance(score, AnswerAnswerResourceScore) and score.name == "urgency"
    assert [p.value for p in score.probabilities] == [0, 1, 2]
    assert [p.label for p in score.probabilities] == ["can wait", "soon", "today"]
    assert 0.0 <= score.score <= 2.0
    assert isinstance(choice, AnswerAnswerResourceChoice) and choice.choice in ("billing", "technical")
    assert isinstance(predicate, AnswerAnswerResourcePredicate) and 0.0 <= predicate.probability <= 1.0
    usage = decision.usage
    assert usage.input_tokens > 0 and usage.output_tokens == 0
    assert usage.total_tokens == usage.input_tokens
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.input_tokens_details.cache_write_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0
    evidence["mixed"] = decision.model_dump()

    if not args.skip_system_one:
        system_one = post_system_one(args)
        answers = system_one["answers"]
        assert usage.input_tokens == system_one["usage"]["input_tokens"]
        assert score.score == answers["urgency"]["score"]
        assert score.confidence == answers["urgency"]["confidence"]
        assert [p.probability for p in score.probabilities] == [
            answers["urgency"]["probabilities"][str(i)] for i in range(3)
        ]
        assert choice.choice == answers["department"]["choice"]
        assert choice.confidence == answers["department"]["confidence"]
        assert [p.probability for p in choice.probabilities] == list(
            answers["department"]["probabilities"].values()
        )
        assert predicate.probability == answers["refund"]["noul"]
        evidence["system_one"] = system_one

    typed = sdk.decisions.create(
        model=MODEL,
        input=[
            {"role": "user", "content": "The customer wrote:"},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Yes, please confirm my order."}],
            },
        ],
        questions=[
            {"type": "predicate", "instructions": "Did the customer confirm?"},
            {
                "type": "choice",
                "name": "dup",
                "instructions": "Which value fits?",
                "choices": [
                    {"value": True, "description": "confirmed"},
                    {"value": "true", "description": "the literal word true"},
                    {"value": "", "description": "no decision"},
                    {"value": False},
                ],
            },
            {
                "type": "score",
                "name": "dup",
                "instructions": "How positive is the customer?",
                "levels": [
                    {"label": "negative", "description": "unhappy"},
                    {"label": "neutral"},
                    {"label": "positive", "description": "happy"},
                ],
            },
        ],
        safety_identifier="contract-user",
    )
    unnamed, typed_choice, typed_score = typed.answers
    assert unnamed.type == "predicate" and unnamed.name is None
    assert typed_choice.name == "dup" and typed_score.name == "dup"
    values = [p.value for p in typed_choice.probabilities]
    assert values == [True, "true", "", False]
    assert [type(v) for v in values] == [bool, str, str, bool]
    best = max(typed_choice.probabilities, key=lambda p: p.probability)
    assert typed_choice.choice == best.value and type(typed_choice.choice) is type(best.value)
    assert abs(sum(p.probability for p in typed_choice.probabilities) - 1.0) < 0.001
    evidence["typed"] = typed.model_dump()

    with ThreadPoolExecutor(max_workers=4) as pool:
        repeated = list(
            pool.map(
                lambda _: sdk.decisions.create(model=MODEL, input=EVIDENCE, questions=QUESTIONS).model_dump(),
                range(4),
            )
        )
    assert all(item == evidence["mixed"] for item in repeated)
    evidence["concurrent_identical"] = len(repeated)

    errors = {
        "image": expect_error(
            openai.BadRequestError,
            "unsupported_value",
            "input[0].content[0].type",
            lambda: sdk.decisions.create(
                model=MODEL,
                input=[{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
                questions=[{"type": "predicate", "instructions": "?"}],
            ),
        ),
        "empty_questions": expect_error(
            openai.BadRequestError,
            "invalid_value",
            "questions",
            lambda: sdk.decisions.create(model=MODEL, input="x", questions=[]),
        ),
        "duplicate_choice": expect_error(
            openai.BadRequestError,
            "invalid_value",
            "questions[0].choices[1].value",
            lambda: sdk.decisions.create(
                model=MODEL,
                input="x",
                questions=[{"type": "choice", "instructions": "?", "choices": [{"value": "a"}, {"value": "a"}]}],
            ),
        ),
        "unknown_parameter": expect_error(
            openai.BadRequestError,
            "unknown_parameter",
            "store",
            lambda: sdk.decisions.create(
                model=MODEL,
                input="x",
                questions=[{"type": "predicate", "instructions": "?"}],
                extra_body={"store": True},
            ),
        ),
        "model": expect_error(
            openai.NotFoundError,
            "model_not_found",
            "model",
            lambda: sdk.decisions.create(
                model="gpt-6-luna", input="x", questions=[{"type": "predicate", "instructions": "?"}]
            ),
        ),
    }
    if args.wrong_api_key:
        errors["auth"] = expect_error(
            openai.AuthenticationError,
            "auth_invalid",
            None,
            lambda: client(args, args.wrong_api_key).decisions.create(
                model=MODEL, input="x", questions=[{"type": "predicate", "instructions": "?"}]
            ),
        )
    evidence["errors"] = errors

    if args.evidence:
        with open(args.evidence, "w", encoding="utf-8") as file:
            json.dump(evidence, file, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, "sdk": openai.__version__, "base_url": args.base_url,
                      "input_tokens": usage.input_tokens, "errors": sorted(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
