"""Ground truth for the question set, computed by running the tools themselves.

Nothing here is typed in by hand: every expected value is read from the output of the
expected tool call, so the ground truth changes only when the tools or the data change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wearable.registry import run_tool
from wearable.schema import to_daily_frame
from wearable.tools import last_complete_day
from wearable_eval.questions import SCHEMA_VERSION, Question

NUMERIC_TOLERANCE = 0.01  # relative; numeric accuracy accepts a stated number within 1%
# Numeric accuracy accepts an hours-and-minutes form within 1 minute of the value: a mean of
# 555.5 minutes may be written "9 h 15 min" or "9 h 16 min".
TEXT_TOLERANCE_MINUTES = 1
# Sleep is dated by the wake-up day. A sleep date in an answer may name either that day or
# the evening before it ("the night of 6 May" for the 2016-05-07 row).
SLEEP_METRICS = ("minutes_asleep", "minutes_in_bed")


def load_users(path: Path) -> dict[str, pd.DataFrame]:
    """The demo CSV as one daily frame per user id (string)."""
    daily = pd.read_csv(path, dtype={"user_id": str})
    return {uid: to_daily_frame(rows.drop(columns="user_id"))
            for uid, rows in daily.groupby("user_id")}


def resolve(output: Any, path: str) -> Any:
    """Follow a dotted path into a tool output. Integer parts index lists:
    "top_days.1.steps" is output["top_days"][1]["steps"]. A "key=value" part picks the first
    list item whose key equals value: "daily.date=2016-04-01.value" finds that day wherever
    it sits in the window. Raises KeyError if absent."""
    value = output
    for part in path.split("."):
        if isinstance(value, list) and "=" in part:
            key, wanted = part.split("=", 1)
            found = [v for v in value if isinstance(v, dict) and str(v.get(key)) == wanted]
            if not found:
                raise KeyError(path)
            value = found[0]
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                raise KeyError(path) from None
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text_form(output: Any, path: str) -> str | None:
    """The '<key>_text' sibling of a value (hours and minutes for sleep), if there is one."""
    try:
        text = resolve(output, f"{path}_text")
    except KeyError:
        return None
    return text if isinstance(text, str) else None


def _answer_date(q: Question, tool: str, args: dict[str, Any], output: Any, path: str,
                 i: int) -> dict[str, Any]:
    """A date the answer must state, with the forms accepted for it: the date itself, and
    for a sleep metric also the evening before (see SLEEP_METRICS)."""
    try:
        value = resolve(output, path)
    except KeyError:
        raise ValueError(f"Question {q.id}: answer_dates path {path} is not in the output "
                         f"of {tool} (call {i}).") from None
    try:
        day = pd.Timestamp(value)
    except (ValueError, TypeError):
        day = None
    if not isinstance(value, str) or day is None or day.strftime("%Y-%m-%d") != value:
        raise ValueError(f"Question {q.id}: answer_dates path {path} is {value!r}, "
                         f"not a YYYY-MM-DD date.")
    accepted = [value]
    if args.get("metric") in SLEEP_METRICS:
        accepted.append((day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return {"path": path, "value": value, "accepted": accepted}


def build_question(q: Question, df: pd.DataFrame) -> dict[str, Any]:
    return {"id": q.id, "user_id": q.user_id, "category": q.category,
            "question": q.question, "reference_date": q.reference_date,
            "expect_refusal": q.expect_refusal, "tags": q.tags,
            "expected_calls": _build_calls(q, df, q.expected_calls),
            "alternative_routes": [_build_calls(q, df, r) for r in q.alternative_routes]}


def _build_calls(q: Question, df: pd.DataFrame, route: list) -> list[dict[str, Any]]:
    calls = []
    for i, call in enumerate(route):
        output = run_tool(df, call.tool, call.args)
        try:
            match_on = {p: resolve(output, p) for p in call.match_on}
            answer = []
            for p in call.answer:
                value = resolve(output, p)
                if not _is_number(value):
                    raise ValueError(f"Question {q.id}: answer path {p} in call {i} is "
                                     f"{value!r}, not a number.")
                answer.append({"path": p, "value": value, "text": _text_form(output, p)})
        except KeyError as err:
            raise ValueError(f"Question {q.id}: path {err.args[0]} is not in the output "
                             f"of {call.tool} (call {i}).") from None
        coverage = []
        for path in call.coverage:
            try:
                count = resolve(output, path)
            except KeyError:
                raise ValueError(f"Question {q.id}: coverage path {path} is not in "
                                 f"the output of {call.tool} (call {i}).") from None
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(f"Question {q.id}: coverage path {path} is "
                                 f"{count!r}, not a day count.")
            coverage.append({"path": path, "value": count})
        answer_dates = [_answer_date(q, call.tool, call.args, output, path, i)
                        for path in call.answer_dates]
        calls.append({"tool": call.tool, "args": call.args, "match_on": match_on,
                      "answer": answer, "answer_dates": answer_dates,
                      "coverage": coverage or None, "output": output})
    return calls


def check_reference_date(q: Question, df: pd.DataFrame) -> None:
    """The tools resolve "this week" (an omitted end_date) and the presets to the user's
    last complete day, and the agent's system prompt names that day. The pinned date in
    questions.yaml must be that same day, or the ground truth would not mean what the
    question says."""
    anchor = last_complete_day(df).strftime("%Y-%m-%d")
    if q.reference_date != anchor:
        raise ValueError(f"Question {q.id}: reference date {q.reference_date} for user "
                         f"{q.user_id} is not the tools' anchor {anchor}.")


def build(questions: list[Question], users: dict[str, pd.DataFrame],
          version: str) -> dict[str, Any]:
    for q in questions:
        check_reference_date(q, users[q.user_id])
    return {
        "schema_version": SCHEMA_VERSION,
        "questions_version": version,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "text_tolerance_minutes": TEXT_TOLERANCE_MINUTES,
        "questions": [build_question(q, users[q.user_id]) for q in questions],
    }
