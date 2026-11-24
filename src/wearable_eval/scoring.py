"""Score a saved run against the ground truth. Reads files only; never calls a model.

Per answer (one question, one repeat):
- tool_result_correct: every expected call is matched by a model call to the same tool whose
  recorded output has the ground-truth values at the match_on paths. Arguments may differ.
  None when the question has no expected calls.
- numbers / unsupported_numbers: the numbers in the answer, and those that no tool result of
  this same answer supports (rules at answer_numbers and is_supported).
- dates / unsupported_dates: the dates in the answer (dates.py), and those that are not in
  this answer's tool outputs, the question or the reference date, or carry a wrong weekday.
  Digits inside a date are part of the date and are not counted as numbers.
- refusal_correct: for expect_refusal questions only. The answer says the data cannot
  answer (REFUSAL phrases) and has no unsupported number.
- coverage_mentioned: for expected calls with a coverage path only. The answer states
  that day count (metrics.coverage_mentioned).
Numeric accuracy is not scored yet.

Every rate is reported with its count: {"k": passed or flagged, "n": out of, "rate": k/n}.
"""

from __future__ import annotations

import re
from datetime import date
from statistics import median
from typing import Any, NamedTuple

from wearable_eval import dates
from wearable_eval.ground_truth import NUMERIC_TOLERANCE, resolve
from wearable_eval.metrics import coverage_mentioned, supported_numbers

METRICS = ("tool_result_correct", "refusal_correct", "coverage_mentioned")

# The windows the tools define, in days: weekly_summary and weekly_active_minutes use 7 days,
# compare_periods' last_4_weeks_vs_previous_4_weeks uses 28 (src/wearable/tools.py).
TOOL_WINDOW_DAYS = frozenset({7, 28})

_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
# Digits not glued to a letter ("SpO2", "E2B"), with optional thousands commas and decimals,
# and an optional sign. A sign counts only when it does not follow a word or a digit, so
# "-5%" and "(+5%)" are signed but the "-" in "7-9 hours" is a range, not a minus.
_NUMBER = re.compile(r"(?:(?<![\w.,)])(?P<sign>[+\-−])(?=\d))?"
                     r"(?P<digits>(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?)")
# A period named by its length: "the last 7 days", "previous 4 weeks", "a 28-day window",
# "the 4 weeks before". Only the lengths in TOOL_WINDOW_DAYS are supported this way.
_PERIOD = re.compile(
    r"\b(?:last|past|previous|prior|preceding|recent|following|first|next|other|those"
    r"|these|the|a|an)\s+(?P<n>\d+)[-\s](?P<unit>days?|nights?|weeks?)\b"
    r"|\b(?P<n2>\d+)\s+(?P<unit2>days?|nights?|weeks?)\s+(?:before|after|earlier|prior)\b",
    re.IGNORECASE)

# Ways of saying the data cannot answer. Matched case-insensitively anywhere in the answer.
REFUSAL = re.compile("|".join([
    r"\bnot (?:in|part of|included in|available in|recorded in|covered by|tracked in)"
    r" (?:the|your|this) (?:data|dataset|table)",
    r"\b(?:does not|doesn't|do not|don't|did not|didn't) (?:contain|include|have|record|track"
    r"|measure|cover|go)",
    r"\bno (?:\w+ )?(?:data|measurements?|records?|readings?|tool)\b",
    r"\bnot enough (?:data|days|nights)\b",
    r"\b(?:cannot|can't|can not|unable to|not possible to) (?:be )?(?:answer|tell|say|determine"
    r"|settle|settled|compare|compared|judge|judged|know|give|provide|compute|calculate)",
    r"\b(?:is|was|week is|data is|it's) (?:incomplete|missing)\b",
    r"\bnot available\b",
]), re.IGNORECASE)


class Number(NamedTuple):
    value: float          # negative only when written with a minus
    signed: bool          # written with an explicit + or -
    text: str             # as written, for the report
    window_days: int | None = None   # set when the number names a period ("last 4 weeks")


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def answer_numbers(text: str | None, reference: date) -> list[Number]:
    """The numbers a text states. Dates are taken out first (their digits are not numbers),
    then list markers ("1.")."""
    if not text:
        return []
    text = _blank(text, [(m.start, m.end) for m in dates.find_dates(text, reference)])
    text = _LIST_MARKER.sub(lambda m: " " * len(m.group()), text)
    periods = {}
    for m in _PERIOD.finditer(text):
        n, unit, group = ((m.group("n"), m.group("unit"), "n") if m.group("n")
                          else (m.group("n2"), m.group("unit2"), "n2"))
        periods[m.start(group)] = int(n) * (7 if unit.lower().startswith("week") else 1)
    out = []
    for m in _NUMBER.finditer(text):
        sign, digits = m.group("sign") or "", m.group("digits")
        value = float(digits.replace(",", ""))
        out.append(Number(-value if sign in ("-", "−") else value, bool(sign), sign + digits,
                          periods.get(m.start("digits"))))
    return out


def _strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _strings(v, out)


def supported_values(tool_calls: list[dict[str, Any]], question: str,
                     reference: date) -> set[float]:
    """Numbers an answer may state: metrics.supported_numbers over the answer's own
    successful tool outputs, plus numbers written inside those outputs' text fields
    ("7 h 31 min", guideline text), plus numbers in the question itself ("10,000 steps").
    Signs are kept."""
    outputs = [c["output"] for c in tool_calls if not c["is_error"]]
    values = set(supported_numbers(outputs))
    texts: list[str] = [question]
    _strings(outputs, texts)
    for t in texts:
        values.update(n.value for n in answer_numbers(t, reference))
    return values


def _close(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= NUMERIC_TOLERANCE * abs(b)


def is_supported(number: Number, supported: set[float]) -> bool:
    """A signed number ("-5%", "+5%") must match a supported value with the same sign. An
    unsigned one ("5% lower") matches either sign, since the direction is then in words,
    which this check does not read. A number naming a period ("last 4 weeks") is also
    supported when that period is one the tools define."""
    if number.window_days in TOOL_WINDOW_DAYS:
        return True
    if number.signed:
        return any(_close(number.value, s) for s in supported)
    return any(_close(number.value, abs(s)) for s in supported)


def _call_matches(expected: dict[str, Any], call: dict[str, Any]) -> bool:
    if call["name"] != expected["tool"] or call["is_error"]:
        return False
    for path, value in expected["match_on"].items():
        try:
            if resolve(call["output"], path) != value:
                return False
        except (KeyError, IndexError, TypeError, ValueError):
            return False
    return True


def score_answer(record: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    answer, calls = record["answer"], record["tool_calls"]
    expected = truth["expected_calls"]
    reference = date.fromisoformat(truth["reference_date"])
    question = truth["question"]

    numbers = answer_numbers(answer, reference)
    supported = supported_values(calls, question, reference)
    unsupported = [n.text for n in numbers if not is_supported(n, supported)]

    mentions = dates.find_dates(answer, reference)
    outputs = [c["output"] for c in calls if not c["is_error"]]
    unsupported_dates = dates.check(
        mentions, dates.supported_days(outputs, question, reference),
        dates.question_months(question, reference))

    tool_result_correct = (all(any(_call_matches(e, c) for c in calls) for e in expected)
                           if expected else None)
    refusal = None
    if truth["expect_refusal"]:
        refusal = bool(answer) and REFUSAL.search(answer) is not None and not unsupported
    coverage = [coverage_mentioned(answer, c["value"])
                for e in expected for c in e.get("coverage") or []]
    return {
        "question_id": record["question_id"], "repeat": record["repeat"],
        "category": truth["category"], "status": record["status"],
        "tool_result_correct": tool_result_correct,
        "refusal_correct": refusal,
        "coverage_mentioned": all(coverage) if coverage else None,
        "numbers": len(numbers), "unsupported_numbers": unsupported,
        "dates": dates.count_items(mentions), "unsupported_dates": unsupported_dates,
        "tools_called": [c["name"] for c in calls],
    }


def _count(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def _passed(values: list[bool | None]) -> dict[str, Any]:
    kept = [v for v in values if v is not None]
    return _count(sum(kept), len(kept))


def _spread(per_repeat: list[dict[str, Any]], pooled: dict[str, Any]) -> dict[str, Any]:
    rates = [c["rate"] for c in per_repeat if c["rate"] is not None]
    return {"per_repeat": per_repeat, "all": pooled,
            "min_rate": min(rates) if rates else None,
            "max_rate": max(rates) if rates else None}


def _measures(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every metric over a set of scored answers, as counts."""
    out = {m: _passed([r[m] for r in rows]) for m in METRICS}
    out["unsupported_numbers"] = _count(sum(len(r["unsupported_numbers"]) for r in rows),
                                        sum(r["numbers"] for r in rows))
    out["answers_with_unsupported_number"] = _count(
        sum(bool(r["unsupported_numbers"]) for r in rows), len(rows))
    out["unsupported_dates"] = _count(sum(len(r["unsupported_dates"]) for r in rows),
                                      sum(r["dates"] for r in rows))
    out["answers_with_unsupported_date"] = _count(
        sum(bool(r["unsupported_dates"]) for r in rows), len(rows))
    return out


SUMMARY_ROWS = (*METRICS, "unsupported_numbers", "answers_with_unsupported_number",
                "unsupported_dates", "answers_with_unsupported_date")


def latest_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per (question, repeat). A resumed run can hold an API-error line and a
    later retry for the same pair; the last line that did not end in an error wins, and
    an error line stays only when no retry finished."""
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in lines:
        key = (r["question_id"], r["repeat"])
        if key not in best or r["status"] != "error" or best[key]["status"] == "error":
            best[key] = r
    return list(best.values())


def score_run(config: dict[str, Any], attempts: list[dict[str, Any]],
              ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Scores one line per (question, repeat); usage and cost count every attempt."""
    truth = {q["id"]: q for q in ground_truth["questions"]}
    lines = latest_lines(attempts)
    rows = [score_answer(r, truth[r["question_id"]]) for r in lines]
    repeats = sorted({r["repeat"] for r in rows})

    per_repeat = {k: _measures([r for r in rows if r["repeat"] == k]) for k in repeats}
    pooled = _measures(rows)
    summary = {m: _spread([per_repeat[k][m] for k in repeats], pooled[m])
               for m in SUMMARY_ROWS}
    for m in METRICS:
        summary[m]["n_questions"] = len({r["question_id"] for r in rows if r[m] is not None})

    by_category = {cat: _measures([r for r in rows if r["category"] == cat])
                   for cat in sorted({r["category"] for r in rows})}

    per_question = {}
    for qid in config["question_ids"]:
        mine = sorted((r for r in rows if r["question_id"] == qid), key=lambda r: r["repeat"])
        entry: dict[str, Any] = {"category": truth[qid]["category"],
                                 "status": [r["status"] for r in mine]}
        for m in METRICS:
            values = [r[m] for r in mine]
            if any(v is not None for v in values):
                entry[m] = values
        entry["unsupported_numbers"] = [r["unsupported_numbers"] for r in mine]
        entry["unsupported_dates"] = [r["unsupported_dates"] for r in mine]
        entry["tools_called"] = [r["tools_called"] for r in mine]
        results = [tuple(r[m] for m in METRICS)
                   + (bool(r["unsupported_numbers"]), bool(r["unsupported_dates"]))
                   for r in mine]
        entry["consistent"] = len(set(results)) == 1
        per_question[qid] = entry

    statuses: dict[str, int] = {}
    for r in lines:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    tokens: dict[str, int] = {}
    for r in attempts:
        for k, v in r["tokens"].items():
            tokens[k] = tokens.get(k, 0) + v
    consistent = sum(e["consistent"] for e in per_question.values())
    return {
        "model": config["model"], "questions_version": config["questions_version"],
        "answers": len(lines), "repeats": repeats, "statuses": statuses,
        "summary": summary, "by_category": by_category,
        "consistency": _count(consistent, len(per_question)),
        "usage": {"attempts": len(attempts),
                  "cost_usd": round(sum(r["cost_usd"] for r in attempts), 4),
                  "requests": sum(r["steps"] for r in attempts), **tokens,
                  "seconds_per_answer_median": median(r["seconds"] for r in lines)
                  if lines else None},
        "per_question": per_question,
        "answers_scored": rows,
    }
