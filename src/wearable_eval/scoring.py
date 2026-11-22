"""Score a saved run against the ground truth. Reads files only; never calls a model.

Per answer (one question, one repeat):
- tool_result_correct: every expected call is matched by a model call to the same tool whose
  recorded output has the ground-truth values at the match_on paths. Arguments may differ.
  None when the question has no expected calls.
- numbers / unsupported: the numbers in the answer, and those that no tool result of this
  same answer supports (rules at answer_numbers and supported_values).
- refusal_correct: for expect_refusal questions only. The answer says the data cannot
  answer (REFUSAL phrases) and has no unsupported number.
- coverage_mentioned: for expected calls with a coverage path only. The answer states
  that day count (metrics.coverage_mentioned).
Numeric accuracy is not scored yet.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any, NamedTuple

from wearable_eval.ground_truth import NUMERIC_TOLERANCE, resolve
from wearable_eval.metrics import coverage_mentioned, supported_numbers

METRICS = ("tool_result_correct", "refusal_correct", "coverage_mentioned")

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_YEAR = r"(?:,?\s+\d{4})?"
# Dates are not claims about the data's values, so they are removed before numbers are
# read: 2016-05-11, May 11, May 11th 2016, 11 May 2016, 11th of May, May 2016, 5/11/2016.
_DATES = re.compile("|".join([
    r"\b\d{4}-\d{2}-\d{2}\b",
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b{_YEAR}",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}{_YEAR}",
    rf"\b{_MONTH}\s+\d{{4}}\b",
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
]), re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
# Digits not glued to a letter ("SpO2", "E2B"), with optional thousands commas and decimals,
# and an optional sign. A sign counts only when it does not follow a word or a digit, so
# "-5%" and "(+5%)" are signed but the "-" in "7-9 hours" is a range, not a minus.
_NUMBER = re.compile(r"(?:(?<![\w.,)])([+\-−])(?=\d))?"
                     r"((?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?)")


class Number(NamedTuple):
    value: float   # negative only when written with a minus
    signed: bool   # written with an explicit + or -
    text: str      # as written, for the report

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


def answer_numbers(text: str | None) -> list[Number]:
    """The numbers a text states, after removing dates and list markers."""
    if not text:
        return []
    text = _LIST_MARKER.sub(" ", _DATES.sub(" ", text))
    out = []
    for sign, digits in _NUMBER.findall(text):
        value = float(digits.replace(",", ""))
        out.append(Number(-value if sign in "-−" and sign else value, bool(sign),
                          sign + digits))
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


def supported_values(tool_calls: list[dict[str, Any]], question: str) -> set[float]:
    """Numbers an answer may state: metrics.supported_numbers over the answer's own
    successful tool outputs, plus numbers written inside those outputs' text fields
    ("7 h 31 min", guideline text), plus numbers in the question itself ("10,000 steps").
    Signs are kept."""
    outputs = [c["output"] for c in tool_calls if not c["is_error"]]
    values = set(supported_numbers(outputs))
    texts: list[str] = [question]
    _strings(outputs, texts)
    for t in texts:
        values.update(n.value for n in answer_numbers(t))
    return values


def _close(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= NUMERIC_TOLERANCE * abs(b)


def is_supported(number: Number, supported: set[float]) -> bool:
    """A signed number ("-5%", "+5%") must match a supported value with the same sign. An
    unsigned one ("5% lower") matches either sign, since the direction is then in words,
    which this check does not read."""
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
    supported = supported_values(calls, truth["question"])
    numbers = answer_numbers(answer)
    unsupported = [n.text for n in numbers if not is_supported(n, supported)]

    tool_result_correct = (all(any(_call_matches(e, c) for c in calls) for e in expected)
                      if expected else None)
    refusal = None
    if truth["expect_refusal"]:
        refusal = bool(answer) and REFUSAL.search(answer) is not None and not unsupported
    coverage = [coverage_mentioned(answer, e["coverage"]["value"])
                for e in expected if e.get("coverage")]
    return {
        "question_id": record["question_id"], "repeat": record["repeat"],
        "category": truth["category"], "status": record["status"],
        "tool_result_correct": tool_result_correct,
        "refusal_correct": refusal,
        "coverage_mentioned": all(coverage) if coverage else None,
        "numbers": len(numbers), "unsupported": unsupported,
        "tools_called": [c["name"] for c in calls],
    }


def _rate(values: list[bool | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def _spread(per_repeat: list[float | None]) -> dict[str, Any]:
    kept = [v for v in per_repeat if v is not None]
    return {"per_repeat": per_repeat,
            "mean": round(sum(kept) / len(kept), 4) if kept else None,
            "min": min(kept) if kept else None, "max": max(kept) if kept else None}


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

    summary: dict[str, Any] = {}
    for m in METRICS:
        summary[m] = _spread([_rate([r[m] for r in rows if r["repeat"] == k]) for k in repeats])
        summary[m]["n_questions"] = sum(1 for q in config["question_ids"]
                                        if _rate([r[m] for r in rows if r["question_id"] == q])
                                        is not None)
    numbers = [sum(r["numbers"] for r in rows if r["repeat"] == k) for k in repeats]
    unsupported = [sum(len(r["unsupported"]) for r in rows if r["repeat"] == k) for k in repeats]
    summary["unsupported_number_rate"] = _spread(
        [round(u / n, 4) if n else None for u, n in zip(unsupported, numbers)])
    summary["unsupported_number_rate"].update(numbers=numbers, unsupported=unsupported)
    summary["answers_with_unsupported"] = _spread(
        [_rate([bool(r["unsupported"]) for r in rows if r["repeat"] == k]) for k in repeats])

    by_category: dict[str, dict[str, float | None]] = {}
    for cat in sorted({r["category"] for r in rows}):
        in_cat = [r for r in rows if r["category"] == cat]
        by_category[cat] = {m: _rate([r[m] for r in in_cat]) for m in METRICS}
        by_category[cat]["answers_with_unsupported"] = _rate(
            [bool(r["unsupported"]) for r in in_cat])

    per_question = {}
    for qid in config["question_ids"]:
        mine = sorted((r for r in rows if r["question_id"] == qid), key=lambda r: r["repeat"])
        entry: dict[str, Any] = {"category": truth[qid]["category"],
                                 "status": [r["status"] for r in mine]}
        for m in METRICS:
            values = [r[m] for r in mine]
            if any(v is not None for v in values):
                entry[m] = values
        entry["unsupported"] = [r["unsupported"] for r in mine]
        entry["tools_called"] = [r["tools_called"] for r in mine]
        results = [tuple(r[m] for m in METRICS) + (bool(r["unsupported"]),) for r in mine]
        entry["consistent"] = len(set(results)) == 1
        per_question[qid] = entry

    statuses: dict[str, int] = {}
    for r in lines:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    tokens: dict[str, int] = {}
    for r in attempts:
        for k, v in r["tokens"].items():
            tokens[k] = tokens.get(k, 0) + v
    return {
        "model": config["model"], "questions_version": config["questions_version"],
        "answers": len(lines), "repeats": repeats, "statuses": statuses,
        "summary": summary, "by_category": by_category,
        "consistency": {"consistent_questions": sum(e["consistent"]
                                                    for e in per_question.values()),
                        "questions": len(per_question)},
        "usage": {"attempts": len(attempts),
                  "cost_usd": round(sum(r["cost_usd"] for r in attempts), 4),
                  "requests": sum(r["steps"] for r in attempts), **tokens,
                  "seconds_per_answer_median": median(r["seconds"] for r in lines)
                  if lines else None},
        "per_question": per_question,
        "answers_scored": rows,
    }
