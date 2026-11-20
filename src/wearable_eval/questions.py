"""Load and validate the evaluation question set (eval/questions.yaml).

The file format is documented in eval/README.md. Validation here is structural: ids,
categories, users, tool names and argument names. Whether the answer paths exist in the
tool outputs is checked when the ground truth is built (ground_truth.py), because that
needs the tools to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wearable.registry import TOOL_SPECS

SCHEMA_VERSION = 1
CATEGORIES = ("activity", "sleep", "heart_rate", "trend", "comparison", "unanswerable")
_ARGUMENTS = {s["name"]: set(s["input_schema"]["properties"]) for s in TOOL_SPECS}
_REQUIRED = {s["name"]: set(s["input_schema"]["required"]) for s in TOOL_SPECS}


@dataclass(frozen=True)
class ExpectedCall:
    tool: str
    args: dict[str, Any]
    answer: list[str]      # output paths whose numbers the answer must state
    match_on: list[str]    # output paths a model's call must reproduce to count as this call


@dataclass(frozen=True)
class Question:
    id: str
    user_id: str
    category: str
    question: str
    expected_calls: list[ExpectedCall] = field(default_factory=list)
    expect_refusal: bool = False
    note: str | None = None


def _fail(qid: str, message: str) -> None:
    raise ValueError(f"Question {qid}: {message}")


def _call(qid: str, raw: dict[str, Any]) -> ExpectedCall:
    unknown = set(raw) - {"tool", "args", "answer", "match_on"}
    if unknown:
        _fail(qid, f"unknown key(s) in expected call: {', '.join(sorted(unknown))}")
    tool = raw.get("tool")
    if tool not in _ARGUMENTS:
        _fail(qid, f"unknown tool {tool!r}")
    args = raw.get("args") or {}
    bad = set(args) - _ARGUMENTS[tool]
    if bad:
        _fail(qid, f"unknown argument(s) for {tool}: {', '.join(sorted(bad))}")
    missing = _REQUIRED[tool] - set(args)
    if missing:
        _fail(qid, f"missing required argument(s) for {tool}: {', '.join(sorted(missing))}")
    answer = list(raw.get("answer") or [])
    match_on = list(raw.get("match_on") or answer)
    if not match_on:
        _fail(qid, f"expected call {tool} needs answer or match_on paths")
    return ExpectedCall(tool, dict(args), answer, match_on)


def parse_questions(data: dict[str, Any], users: set[str] | None = None) -> list[Question]:
    """Turn the loaded YAML into Question objects, raising ValueError on any problem."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    out: list[Question] = []
    seen: set[str] = set()
    for raw in data.get("questions") or []:
        qid = str(raw.get("id"))
        if qid in seen:
            _fail(qid, "duplicate id")
        seen.add(qid)
        unknown = set(raw) - {"id", "user_id", "category", "question", "expected_calls",
                              "expect_refusal", "note"}
        if unknown:
            _fail(qid, f"unknown key(s): {', '.join(sorted(unknown))}")
        if raw.get("category") not in CATEGORIES:
            _fail(qid, f"category must be one of {', '.join(CATEGORIES)}")
        user_id = str(raw.get("user_id"))
        if users is not None and user_id not in users:
            _fail(qid, f"user {user_id} is not in the demo data")
        if not str(raw.get("question") or "").strip():
            _fail(qid, "empty question")
        calls = [_call(qid, c) for c in raw.get("expected_calls") or []]
        refusal = raw.get("expect_refusal", False)
        if not isinstance(refusal, bool):
            _fail(qid, "expect_refusal must be true or false")
        if refusal and any(c.answer for c in calls):
            _fail(qid, "a refusal question cannot require numbers in the answer")
        out.append(Question(qid, user_id, raw["category"], raw["question"].strip(), calls,
                            refusal, raw.get("note")))
    if not out:
        raise ValueError("The question set is empty.")
    return out


def load_questions(path: Path, users: set[str] | None = None) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return parse_questions(yaml.safe_load(f), users)
