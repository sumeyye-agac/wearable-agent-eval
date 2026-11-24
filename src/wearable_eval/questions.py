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

import re

import yaml

from wearable.registry import TOOL_SPECS

SCHEMA_VERSION = 1
CATEGORIES = ("activity", "sleep", "heart_rate", "trend", "comparison", "unanswerable")
_ARGUMENTS = {s["name"]: set(s["input_schema"]["properties"]) for s in TOOL_SPECS}
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REQUIRED = {s["name"]: set(s["input_schema"]["required"]) for s in TOOL_SPECS}
_VERSION = re.compile(r"v[1-9]\d*(?:\.[1-9]\d*)?")


@dataclass(frozen=True)
class ExpectedCall:
    tool: str
    args: dict[str, Any]
    answer: list[str]      # output paths whose numbers the answer must state
    match_on: list[str]    # output paths a model's call must reproduce to count as this call
    coverage: list[str] = field(default_factory=list)  # paths of day counts the answer must mention
    answer_dates: list[str] = field(default_factory=list)  # paths of dates the answer must state


@dataclass(frozen=True)
class Question:
    id: str
    user_id: str
    category: str
    question: str
    reference_date: str    # the pinned "today": "this week" is the 7 days ending on it
    expected_calls: list[ExpectedCall] = field(default_factory=list)
    expect_refusal: bool = False
    note: str | None = None
    tags: list[str] = field(default_factory=list)  # free labels for analysis, e.g. missing_data
    # Other call sets that answer the question equally well (each a list of calls, all
    # needed). expected_calls is the first route; these are the others.
    alternative_routes: list[list[ExpectedCall]] = field(default_factory=list)


def _fail(qid: str, message: str) -> None:
    raise ValueError(f"Question {qid}: {message}")


def _call(qid: str, raw: dict[str, Any]) -> ExpectedCall:
    unknown = set(raw) - {"tool", "args", "answer", "match_on", "coverage", "answer_dates"}
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
    coverage = raw.get("coverage") or []
    if isinstance(coverage, str):
        coverage = [coverage]
    if not isinstance(coverage, list) or not all(isinstance(c, str) and c for c in coverage):
        _fail(qid, f"coverage in expected call {tool} must be an output path or a list of them")
    answer_dates = raw.get("answer_dates") or []
    if not isinstance(answer_dates, list) or not all(isinstance(d, str) and d for d in answer_dates):
        _fail(qid, f"answer_dates in expected call {tool} must be a list of output paths")
    return ExpectedCall(tool, dict(args), answer, match_on, list(coverage), list(answer_dates))


def question_set_version(data: dict[str, Any]) -> str:
    """The version of the question set ("v1", "v1.1", "v2", ...). Any change to the questions,
    their expected calls or the reference dates needs a new version, so results from
    different versions are never compared as if they were the same test."""
    version = data.get("version")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"version must be v1, v1.1, v2, ...; got {version!r}.")
    return version


def parse_questions(data: dict[str, Any], users: set[str] | None = None) -> list[Question]:
    """Turn the loaded YAML into Question objects, raising ValueError on any problem."""
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    question_set_version(data)
    reference_dates = {str(k): str(v) for k, v in (data.get("reference_dates") or {}).items()}
    for uid, day in reference_dates.items():
        if not _ISO_DATE.fullmatch(day):
            raise ValueError(f"reference_dates: {day!r} for user {uid} is not YYYY-MM-DD.")
    out: list[Question] = []
    seen: set[str] = set()
    for raw in data.get("questions") or []:
        qid = str(raw.get("id"))
        if qid in seen:
            _fail(qid, "duplicate id")
        seen.add(qid)
        unknown = set(raw) - {"id", "user_id", "category", "question", "expected_calls",
                              "expect_refusal", "note", "tags", "alternative_routes"}
        if unknown:
            _fail(qid, f"unknown key(s): {', '.join(sorted(unknown))}")
        if raw.get("category") not in CATEGORIES:
            _fail(qid, f"category must be one of {', '.join(CATEGORIES)}")
        user_id = str(raw.get("user_id"))
        if users is not None and user_id not in users:
            _fail(qid, f"user {user_id} is not in the demo data")
        if user_id not in reference_dates:
            _fail(qid, f"user {user_id} has no entry in reference_dates")
        if not str(raw.get("question") or "").strip():
            _fail(qid, "empty question")
        calls = [_call(qid, c) for c in raw.get("expected_calls") or []]
        routes_raw = raw.get("alternative_routes") or []
        if not isinstance(routes_raw, list) or not all(isinstance(r, list) and r
                                                       for r in routes_raw):
            _fail(qid, "alternative_routes must be a list of non-empty lists of calls")
        if routes_raw and not calls:
            _fail(qid, "alternative_routes need expected_calls as the first route")
        routes = [[_call(qid, c) for c in r] for r in routes_raw]
        every_call = calls + [c for r in routes for c in r]
        refusal = raw.get("expect_refusal", False)
        if not isinstance(refusal, bool):
            _fail(qid, "expect_refusal must be true or false")
        if refusal and any(c.answer or c.answer_dates for c in every_call):
            _fail(qid, "a refusal question cannot require numbers or dates in the answer")
        tags = raw.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
            _fail(qid, "tags must be a list of non-empty strings")
        if refusal and any(c.coverage for c in every_call):
            _fail(qid, "a refusal question cannot require a day count")
        out.append(Question(qid, user_id, raw["category"], raw["question"].strip(),
                            reference_dates[user_id], calls, refusal, raw.get("note"),
                            list(tags), routes))
    if not out:
        raise ValueError("The question set is empty.")
    return out


def load_questions(path: Path, users: set[str] | None = None) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return parse_questions(yaml.safe_load(f), users)


def load_version(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return question_set_version(yaml.safe_load(f))
