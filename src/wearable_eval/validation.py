"""Validation of the scorer without any model: fake agents, a null baseline and a mutation
test, all run through the real scorer (scoring.score_answer).

- Fake agents: oracle (the expected calls of each question, and an answer written from
  their outputs by a template) must pass everything; empty (no calls, empty answer) must
  fail everything; always-refuse (no calls, "I don't have that data.") is reported as is.
- Null baseline: each real answer is scored against another question's sources (its tool
  calls, question text and user). Whatever then counts as supported is supported by
  coincidence, so this measures how loose the support check is.
- Mutation test: correct answers (end_to_end True) are changed in one place each, the way
  a wrong answer would differ, and should then fail end_to_end.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd

from wearable.registry import run_tool
from wearable_eval import dates
from wearable_eval.scoring import answer_numbers, score_answer

ORACLE_REFUSAL = "The data cannot answer this question."
ALWAYS_REFUSE = "I don't have that data."


# --- fake agents ------------------------------------------------------------------

def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def oracle_record(truth: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """The expected calls of route A, run on the real tools, and an answer that states
    exactly the ground-truth values, dates and day counts, or refuses."""
    calls = []
    for e in truth["expected_calls"]:
        calls.append({"step": 1, "name": e["tool"], "input": dict(e["args"]),
                      "output": run_tool(df, e["tool"], dict(e["args"])), "is_error": False})
    if truth["expect_refusal"]:
        answer = ORACLE_REFUSAL
    else:
        parts = [_fmt(a["value"]) for e in truth["expected_calls"] for a in e["answer"]]
        parts += [d["value"] for e in truth["expected_calls"] for d in e["answer_dates"]]
        answer = "Result: " + "; ".join(parts) + "."
        counts = [_fmt(c["value"]) for e in truth["expected_calls"]
                  for c in e.get("coverage") or []]
        if counts:
            answer += " Days with data: " + " and ".join(counts) + "."
    return _record(truth, answer, calls)


def _record(truth: dict[str, Any], answer: str, calls: list[dict[str, Any]]
            ) -> dict[str, Any]:
    return {"question_id": truth["id"], "repeat": 1, "status": "answered",
            "answer": answer, "tool_calls": calls}


def empty_record(truth: dict[str, Any]) -> dict[str, Any]:
    return _record(truth, "", [])


def refuse_record(truth: dict[str, Any]) -> dict[str, Any]:
    return _record(truth, ALWAYS_REFUSE, [])


# --- null baseline -----------------------------------------------------------------

def _outputs(record: dict[str, Any]) -> list[Any]:
    return [c["output"] for c in record["tool_calls"] if not c["is_error"]]


def null_baseline(records: list[dict[str, Any]], truths: dict[str, dict[str, Any]],
                  prompt: dict[str, set[date]]) -> dict[str, Any]:
    """Score every answer against every other question's sources in the same run and
    repeat. Pairs whose calls share an output are left out: there the support is real,
    not coincidence."""
    out: dict[str, dict[str, int]] = {
        kind: {"pairs": 0, "numbers": 0, "supported": 0, "small_numbers": 0,
               "small_supported": 0, "dates": 0, "dates_supported": 0,
               "answers_with_numbers": 0, "answers_all_supported": 0}
        for kind in ("same_user", "cross_user")}
    skipped = 0
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_repeat.setdefault(r["repeat"], []).append(r)
    for group in by_repeat.values():
        for a in group:
            if not a["answer"]:
                continue
            ta = truths[a["question_id"]]
            plain = [n for n in answer_numbers(a["answer"],
                                               date.fromisoformat(ta["reference_date"]))]
            for b in group:
                if b is a:
                    continue
                if any(x == y for x in _outputs(a) for y in _outputs(b)):
                    skipped += 1
                    continue
                tb = truths[b["question_id"]]
                kind = "same_user" if ta["user_id"] == tb["user_id"] else "cross_user"
                s = score_answer({**a, "tool_calls": b["tool_calls"]}, tb,
                                 prompt[tb["user_id"]])
                small = [n for n in plain if n.kind == "number"
                         and float(n.value).is_integer() and abs(n.value) < 13]
                c = out[kind]
                c["pairs"] += 1
                c["numbers"] += s["numbers"] - len(small)
                c["supported"] += s["numbers"] - len(small) - len(s["unsupported_numbers"])
                c["small_numbers"] += len(small)
                c["small_supported"] += len(small) - len(s["low_confidence_numbers"])
                c["dates"] += s["dates"]
                c["dates_supported"] += s["dates"] - len(s["unsupported_dates"])
                if s["numbers"] - len(small):
                    c["answers_with_numbers"] += 1
                    c["answers_all_supported"] += not s["unsupported_numbers"] \
                        and not s["unsupported_dates"]
    return {"skipped_pairs_sharing_an_output": skipped, **out}


# --- mutation test -------------------------------------------------------------------

_ISO_RANGE = re.compile(r"(?P<a>\d{4}-\d{2}-\d{2})(?P<sep>\s*(?:to|–|-|through)\s*)"
                        r"(?P<b>\d{4}-\d{2}-\d{2})")
_ISO = re.compile(r"(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-])")


def _shift(iso: str, days: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days)).isoformat()


def _format_like(original: str, value: float) -> str:
    digits = original.lstrip("+-−")
    sign = original[: len(original) - len(digits)]
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    body = f"{abs(value):,.{decimals}f}" if "," in digits else f"{abs(value):.{decimals}f}"
    return sign + body


def mutants(answer: str, reference: date, rng: random.Random
            ) -> list[tuple[str, str, str]]:
    """(type, mutated answer, original text) for every place in the answer one mutation
    type applies."""
    out: list[tuple[str, str, str]] = []
    for n in answer_numbers(answer, reference):
        if n.kind != "number" or n.window_days:
            continue
        start = n.end - len(n.text)
        if float(n.value).is_integer() and abs(n.value) < 13:
            new = n.value + rng.choice((-1, 1))
            if new < 0 <= n.value:
                new = n.value + 1
            out.append(("small_int_pm1", answer[:start] + _format_like(n.text, new)
                        + answer[n.end:], n.text))
        elif abs(n.value) >= 13:
            factor = 1 + rng.choice((-1, 1)) * rng.uniform(0.10, 0.30)
            text = _format_like(n.text, n.value * factor)
            if text != n.text:
                out.append(("large_number_10_30pct", answer[:start] + text + answer[n.end:],
                            n.text))
    ranges = list(_ISO_RANGE.finditer(answer))
    in_range = {m.start("a") for m in ranges} | {m.start("b") for m in ranges}
    for m in _ISO.finditer(answer):
        if m.start() in in_range:
            continue
        out.append(("date_pm1", answer[:m.start()] + _shift(m.group(), rng.choice((-1, 1)))
                    + answer[m.end():], m.group()))
    for m in ranges:
        days = rng.choice((-1, 1))
        out.append(("range_shift_1day", answer[:m.start()] + _shift(m["a"], days) + m["sep"]
                    + _shift(m["b"], days) + answer[m.end():], m.group()))
    return out


def refusal_mutant(rng: random.Random) -> str:
    """A refusal replaced by a made-up figure."""
    return f"It was {rng.randint(13, 9999):,}."


def _is_answer_item(truth: dict[str, Any], original: str) -> bool:
    """Whether the mutated text is one of the values or dates the answer must state."""
    routes = [truth["expected_calls"], *truth.get("alternative_routes", [])]
    calls = [c for r in routes for c in r]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", original):
        return any(original in d["accepted"] for c in calls for d in c["answer_dates"])
    try:
        value = abs(float(original.lstrip("+-−").replace(",", "")))
    except ValueError:
        return False
    return any(abs(abs(a["value"]) - value) <= 0.01 * abs(a["value"])
               for c in calls for a in c["answer"])


def mutation_test(records: list[dict[str, Any]], truths: dict[str, dict[str, Any]],
                  prompt: dict[str, set[date]], seed: int = 0) -> dict[str, Any]:
    """Detection rate per mutation type, split into mutants of an answer value or date
    (checked by answer_correct) and of any other number or date (checked only by
    support)."""
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {}
    missed: dict[str, list[str]] = {}

    def add(kind: str, detected: bool, example: str, answer_item: bool = False) -> None:
        c = counts.setdefault(kind, {"mutants": 0, "detected": 0, "low_confidence_only": 0,
                                     "answer_item_mutants": 0, "answer_item_detected": 0})
        c["mutants"] += 1
        c["detected"] += detected
        c["answer_item_mutants"] += answer_item
        c["answer_item_detected"] += answer_item and detected
        if not detected:
            missed.setdefault(kind, []).append(example)

    for r in records:
        t = truths[r["question_id"]]
        p = prompt[t["user_id"]]
        if not r["answer"] or not score_answer(r, t, p)["end_to_end"]:
            continue
        if t["expect_refusal"]:
            s = score_answer({**r, "answer": refusal_mutant(rng)}, t, p)
            add("refusal_to_number", not s["end_to_end"], r["question_id"])
            continue
        reference = date.fromisoformat(t["reference_date"])
        for kind, text, original in mutants(r["answer"], reference, rng):
            s = score_answer({**r, "answer": text}, t, p)
            detected = not s["end_to_end"]
            add(kind, detected, f"{r['question_id']} r{r['repeat']}: {original}",
                _is_answer_item(t, original))
            if not detected and s["low_confidence_numbers"]:
                counts[kind]["low_confidence_only"] += 1
    return {"seed": seed, "types": counts,
            "missed_examples": {k: v[:10] for k, v in missed.items()}}
