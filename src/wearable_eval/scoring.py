"""Score a saved run against the ground truth (scorer v1). Reads files only; never calls a
model.

Per answer (one question, one repeat):
- tool_result_correct: some accepted route (expected_calls, or one of alternative_routes)
  is matched in full: each of its calls by a model call to the same tool whose recorded
  output has the ground-truth values at the match_on paths. Arguments may differ. None
  when the question has no expected calls.
- answer_correct: the answer states every answer value of a matched route (within 1%, or
  as hours and minutes within 1 minute, or as decimal hours) and one accepted form of
  every answer date. None when the question has no answer values or dates. A negative r
  must be written with its minus sign. For a "how many <noun>" question with a small
  count, the first count written next to that noun must be the answer value.
- verdict_correct: for yes/no and comparison questions with a verdict or direction in the
  ground truth. False when the answer starts with the wrong "Yes"/"No", or uses words of
  the wrong direction ("more", "fewer", ...) and none of the right one. None otherwise.
- numbers / unsupported_numbers: the numbers in the answer, and those no source supports
  (rules at number_sources and support). Unsupported whole numbers below 13 are kept apart
  as low_confidence_numbers: they are counted, but not in unsupported_numbers or the
  headline.
- dates / unsupported_dates: the dates in the answer (dates.py), and those that are in no
  tool output, the question or the system prompt, or carry a wrong weekday.
- number_sources / date_sources: every supported number and date with its source.
- model_computed: the fallback rules that fired (a unit conversion or an evening-before
  date the tools already give). Accepted as supported, but counted: each is a number or
  date the model worked out itself.
- refusal_correct: for expect_refusal questions only. The answer says the data cannot
  answer (REFUSAL phrases) and has no unsupported number, small whole numbers included.
- coverage_mentioned: for expected calls with a coverage path only. The answer states
  that day count (metrics.coverage_mentioned).
- call_validity: one label per tool call (tool_calls.py); no_call: the question has
  expected calls and the model made none.
- end_to_end (headline): answered, and every check above that applies passes, with no
  unsupported number or date.

Every rate is reported with its count: {"k": passed or flagged, "n": out of, "rate": k/n}.
"""

from __future__ import annotations

import re
from datetime import date
from statistics import median
from typing import Any, NamedTuple

import pandas as pd

from wearable.tools import data_range, last_complete_day
from wearable_eval import dates
from wearable_eval.ground_truth import NUMERIC_TOLERANCE, TEXT_TOLERANCE_MINUTES, resolve
from wearable_eval.metrics import coverage_mentioned
from wearable_eval.tool_calls import LABELS, classify

SCORER_VERSION = "v1.1"

METRICS = ("tool_result_correct", "answer_correct", "verdict_correct", "refusal_correct",
           "coverage_mentioned", "end_to_end")

# Unsupported whole numbers below this are low-confidence: "2 vigorous minutes", "1 of 7"
# and list words are too often harmless to count against the headline.
SMALL_INTEGER = 13

# The windows the tools define, in days: weekly_summary and weekly_active_minutes use 7 days,
# compare_periods' last_4_weeks_vs_previous_4_weeks uses 28 (src/wearable/tools.py).
TOOL_WINDOW_DAYS = frozenset({7, 28})

# Guideline numbers are supported only as these (value, unit) pairs, and only when the
# answer's own tool calls returned that guideline. The numbers inside get_guideline's text
# support nothing else, so "60" from "Adults 18-60 years" does not support "60 bpm".
_HOURS = r"hours?|hrs?|h"
_MINUTES = r"minutes?|mins?|min"
_YEARS = r"years?|yrs?|year-olds?"
GUIDELINE_NUMBERS: dict[str, tuple[tuple[float, str], ...]] = {
    "sleep_duration": ((7, _HOURS), (8, _HOURS), (9, _HOURS), (18, _YEARS), (60, _YEARS),
                       (61, _YEARS), (64, _YEARS), (65, _YEARS)),
    "physical_activity": ((150, _MINUTES), (75, _MINUTES), (1, _MINUTES), (2, _MINUTES),
                          (2, r"days?"), (3, r"METs?"), (5.9, r"METs?"), (6, r"METs?")),
}

_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
# Digits not glued to a letter ("SpO2", "E2B"), with optional thousands commas and decimals,
# and an optional sign. A sign counts only when it does not follow a word or a digit, so
# "-5%" and "(+5%)" are signed but the "-" in "7-9 hours" is a range, not a minus.
_NUMBER = re.compile(r"(?:(?<![\w.,)])(?P<sign>[+\-−])(?=\d))?"
                     r"(?P<digits>(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?)")
# Durations read as one value: "7 h 31 min", "7 hours and 31 minutes" (in minutes), and
# decimal hours "7.25 hours", "7.25 h" (in hours).
_HM = re.compile(r"(?<![\w.,])(?P<h>\d{1,2})\s*(?:h|hrs?|hours?)\.?,?\s*(?:and\s+)?"
                 r"(?P<m>\d{1,2})\s*(?:min|mins|minutes?)\b")
_DECIMAL_HOURS = re.compile(r"(?<![\w.,])(?P<v>\d{1,2}\.\d{1,2})\s*(?:h|hrs?|hours?)\b")
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
    value: float          # negative only when written with a minus; minutes for "hm"
    signed: bool          # written with an explicit + or -
    text: str             # as written, for the report
    window_days: int | None = None   # set when the number names a period ("last 4 weeks")
    kind: str = "number"  # "number", "hm" (hours and minutes, value in minutes) or "hours"
    end: int = 0          # where it ends in the text, to read the unit after it


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def answer_numbers(text: str | None, reference: date) -> list[Number]:
    """The numbers a text states, in reading order. Dates are taken out first (their digits
    are not numbers), then list markers ("1."), then durations, each read as one value."""
    if not text:
        return []
    text = _blank(text, [(m.start, m.end) for m in dates.find_dates(text, reference)])
    text = _LIST_MARKER.sub(lambda m: " " * len(m.group()), text)
    found: list[tuple[int, Number]] = []
    for m in _HM.finditer(text):
        found.append((m.start(), Number(int(m["h"]) * 60 + int(m["m"]), False, m.group(),
                                        kind="hm", end=m.end())))
    for m in _DECIMAL_HOURS.finditer(text):
        if not any(s <= m.start() < n.end for s, n in found):
            found.append((m.start(), Number(float(m["v"]), False, m.group(), kind="hours",
                                            end=m.end())))
    text = _blank(text, [(s, n.end) for s, n in found])
    periods = {}
    for m in _PERIOD.finditer(text):
        n, unit, group = ((m.group("n"), m.group("unit"), "n") if m.group("n")
                          else (m.group("n2"), m.group("unit2"), "n2"))
        periods[m.start(group)] = int(n) * (7 if unit.lower().startswith("week") else 1)
    for m in _NUMBER.finditer(text):
        sign, digits = m.group("sign") or "", m.group("digits")
        value = float(digits.replace(",", ""))
        found.append((m.start(), Number(-value if sign in ("-", "−") else value, bool(sign),
                                        sign + digits, periods.get(m.start("digits")),
                                        end=m.end())))
    return [n for _, n in sorted(found, key=lambda x: x[0])]


# --- sources ------------------------------------------------------------------

class Source(NamedTuple):
    value: float
    source: str           # "<tool>:<path>", "<tool>:<path>(count|days|text)", "question"
    minutes_text: bool = False   # a duration read from a text field ("7 h 31 min")


def _hm_minutes(text: str) -> list[int]:
    return [int(m["h"]) * 60 + int(m["m"]) for m in _HM.finditer(text)]


def _walk(value: Any, path: str, tool: str, out: list[Source], reference: date,
          text_numbers: bool) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.append(Source(float(value), f"{tool}:{path}"))
    elif isinstance(value, str):
        out.extend(Source(float(m), f"{tool}:{path}(text)", True) for m in _hm_minutes(value))
        if text_numbers:
            out.extend(Source(n.value, f"{tool}:{path}(text)")
                       for n in answer_numbers(value, reference) if n.kind == "number")
    elif isinstance(value, list):
        if value and all(isinstance(v, str) and dates._ISO.match(v) for v in value):
            out.append(Source(float(len(value)), f"{tool}:{path}(count)"))
        for i, item in enumerate(value):
            _walk(item, f"{path}.{i}" if path else str(i), tool, out, reference, text_numbers)
    elif isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
        if isinstance(start, str) and isinstance(end, str) and dates._ISO.match(start) \
                and dates._ISO.match(end):
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
            out.append(Source(float(days), f"{tool}:{path or '.'}(days)"))
        for key, item in value.items():
            _walk(item, f"{path}.{key}" if path else key, tool, out, reference, text_numbers)


def number_sources(tool_calls: list[dict[str, Any]], question: str,
                   reference: date) -> list[Source]:
    """Every number an answer may state, with its source: each numeric value in the
    answer's own successful tool outputs; the length of each list of dates ("2 days
    missing") and of each start-to-end window ("of 7 days"); numbers written in the
    outputs' text fields, except get_guideline's (see GUIDELINE_NUMBERS); durations in
    text fields ("7 h 31 min"); and the numbers in the question."""
    out: list[Source] = []
    for call in tool_calls:
        if not call["is_error"]:
            _walk(call["output"], "", call["name"], out, reference,
                  text_numbers=call["name"] != "get_guideline")
    out.extend(Source(n.value, "question") for n in answer_numbers(question, reference)
               if n.kind == "number")
    out.extend(Source(float(m), "question", True) for m in _hm_minutes(question))
    return out


def supported_values(tool_calls: list[dict[str, Any]], question: str,
                     reference: date) -> set[float]:
    """The values of number_sources, as a set."""
    return {s.value for s in number_sources(tool_calls, question, reference)}


def guideline_topics(tool_calls: list[dict[str, Any]]) -> set[str]:
    """The guidelines the answer's tool calls returned. weekly_active_minutes carries the
    activity recommendation and its conversion rule."""
    topics = set()
    for call in tool_calls:
        if call["is_error"]:
            continue
        if call["name"] == "get_guideline" and isinstance(call.get("input"), dict):
            topics.add(call["input"].get("topic"))
        elif call["name"] == "weekly_active_minutes":
            topics.add("physical_activity")
    return topics & set(GUIDELINE_NUMBERS)


def _close(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= NUMERIC_TOLERANCE * abs(b)


def _match(number: Number, s: float) -> bool:
    """A signed number must match with the same sign; an unsigned one matches either sign,
    since the direction is then in words, which this check does not read."""
    return _close(number.value, s) if number.signed else _close(number.value, abs(s))


def is_supported(number: Number, supported: set[float]) -> bool:
    """A plain number against a set of values (no sources): for tests and quick checks."""
    if number.window_days in TOOL_WINDOW_DAYS:
        return True
    return any(_match(number, s) for s in supported)


def _hours_close(hours: float, minutes: float) -> bool:
    """Decimal hours as written ("7.6", "7.64") against a value in minutes, at the number
    of decimals written."""
    decimals = len(str(hours).split(".")[1]) if "." in str(hours) else 0
    return abs(hours - abs(minutes) / 60) <= 0.5 * 10 ** -decimals + 1e-9


def support(number: Number, sources: list[Source], text: str,
            topics: set[str]) -> tuple[str | None, bool]:
    """(source, fallback) for one number from the answer; (None, False) if unsupported.
    fallback is True when only a conversion the model made supports it."""
    if number.kind == "hm":
        for s in sources:
            if s.minutes_text and abs(number.value - s.value) <= TEXT_TOLERANCE_MINUTES:
                return s.source, False
        for s in sources:   # minutes the tools gave as a number, converted by the model
            if not s.minutes_text and abs(number.value - abs(s.value)) <= TEXT_TOLERANCE_MINUTES:
                return f"fallback:{s.source}", True
        return None, False
    if number.kind == "hours":
        for s in sources:
            if s.source.endswith("_hours") and _close(number.value, abs(s.value)):
                return s.source, False
        for s in sources:
            if not s.minutes_text and _hours_close(number.value, s.value):
                return f"fallback:{s.source}", True
        return None, False
    if number.window_days in TOOL_WINDOW_DAYS:
        return "tool_window", False
    for s in sources:
        if not s.minutes_text and _match(number, s.value):
            return s.source, False
    after = text[number.end:number.end + 30]
    for topic in topics:
        for value, unit in GUIDELINE_NUMBERS[topic]:
            if number.value == value and re.match(
                    rf"^(?:\W*(?:\w+\W+){{0,3}}?)(?:{unit})\b", after, re.IGNORECASE):
                return f"guideline:{topic}", False
    return None, False


def _is_small_integer(number: Number) -> bool:
    return number.kind == "number" and float(number.value).is_integer() \
        and abs(number.value) < SMALL_INTEGER


# --- answer values -------------------------------------------------------------

_MINUTES_AFTER = re.compile(rf"^\s*(?:{_MINUTES})\b", re.IGNORECASE)


def _states_value(numbers: list[Number], value: float, has_text: bool, answer: str,
                  signed: bool = False) -> bool:
    """Whether the answer states one answer value. has_text: the value is in minutes and
    the tool also gave it as hours and minutes, so the minute forms ("7 h 31 min",
    "14 minutes") are accepted within TEXT_TOLERANCE_MINUTES, and decimal hours at the
    decimals written. A whole count up to ten may be written as a word ("four of the
    seven nights"), as in coverage_mentioned. signed: the sign is the meaning (r), so a
    negative value must be written with its minus sign."""
    if signed and value < 0:
        return any(n.kind == "number" and n.signed and _close(n.value, value) for n in numbers)
    for n in numbers:
        if n.kind == "number" and _close(abs(n.value), abs(value)):
            return True
        if has_text and n.kind == "hm" and abs(n.value - abs(value)) <= TEXT_TOLERANCE_MINUTES:
            return True
        if has_text and n.kind == "number" and _MINUTES_AFTER.match(answer[n.end:]) \
                and abs(abs(n.value) - abs(value)) <= TEXT_TOLERANCE_MINUTES:
            return True
        if has_text and n.kind == "hours" and _hours_close(n.value, value):
            return True
    return float(value).is_integer() and 0 <= value <= 10 \
        and coverage_mentioned(answer, int(value))


def _states_date(mentions: list[dates.Mention], accepted: list[str]) -> bool:
    wanted = {date.fromisoformat(a) for a in accepted}
    return any(item.day in wanted for m in mentions for item in m.items)


_COUNT_WORDS = {w: i for i, w in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
     "eleven", "twelve"))}
# A count followed by one of these counts something else ("7 hours on 3 nights").
_OTHER_UNITS = {"hour", "hours", "h", "hr", "hrs", "minute", "minutes", "min", "mins",
                "step", "steps", "bpm", "percent"}
_HOW_MANY = re.compile(r"\bhow many (\w+)", re.IGNORECASE)


def first_count(answer: str, noun: str, reference: date) -> int | None:
    """The first count the answer gives for a noun ("nights"): at the first mention of the
    noun that has one, the earliest whole number or number word in the 4 words before it,
    skipping numbers of another unit. Dates are taken out first."""
    text = _blank(answer, [(m.start, m.end) for m in dates.find_dates(answer, reference)])
    tokens = re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", text.lower())
    singular = noun.lower().rstrip("s")
    for i, token in enumerate(tokens):
        if token.rstrip("s") != singular:
            continue
        for j in range(max(0, i - 4), i):
            value = int(tokens[j]) if tokens[j].isdigit() else _COUNT_WORDS.get(tokens[j])
            if value is not None and tokens[j + 1] not in _OTHER_UNITS:
                return value
    return None


def _route_stated(route: list[dict[str, Any]], numbers: list[Number],
                  mentions: list[dates.Mention], answer: str, question: str,
                  reference: date) -> bool | None:
    answers = [a for call in route for a in call["answer"]]
    answer_dates = [d for call in route for d in call.get("answer_dates") or []]
    if not answers and not answer_dates:
        return None
    how_many = _HOW_MANY.search(question)

    def stated(a: dict[str, Any]) -> bool:
        value = a["value"]
        if how_many and float(value).is_integer() and 0 <= value < SMALL_INTEGER:
            return first_count(answer, how_many.group(1), reference) == value
        signed = a["path"] == "r" or a["path"].endswith(".r")
        return _states_value(numbers, value, a["text"] is not None, answer, signed)

    return (all(stated(a) for a in answers)
            and all(_states_date(mentions, d["accepted"]) for d in answer_dates))


_UP = re.compile(r"\b(?:more|higher|increase[ds]?|increasing|rose|risen|up|gain(?:ed|s)?)\b",
                 re.IGNORECASE)
_DOWN = re.compile(r"\b(?:less|fewer|lower|decrease[ds]?|decreasing|fell|fallen|down|drop"
                   r"(?:ped|s)?|decline[ds]?|reduced?|reduction)\b", re.IGNORECASE)
_YES_NO = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


def verdict_correct(answer: str | None, truth: dict[str, Any]) -> bool | None:
    """The yes/no and the direction of an answer against the ground truth's verdict and
    direction. Only clear contradictions count: a leading "Yes"/"No" that is wrong, or
    words of the wrong direction with none of the right one. None when the question has
    neither."""
    verdict, direction = truth.get("verdict"), truth.get("direction")
    if verdict is None and direction is None:
        return None
    text = answer or ""
    said = _YES_NO.match(text)
    if verdict and said and said.group(1).lower() != verdict:
        return False
    if direction:
        right, wrong = (_UP, _DOWN) if direction == "up" else (_DOWN, _UP)
        if wrong.search(text) and not right.search(text):
            return False
    return True


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


def _routes(truth: dict[str, Any]) -> list[list[dict[str, Any]]]:
    routes = [truth["expected_calls"], *truth.get("alternative_routes", [])]
    return [r for r in routes if r]


# --- one answer ----------------------------------------------------------------

def prompt_days_for(users: dict[str, pd.DataFrame]) -> dict[str, set[date]]:
    """Per user, the days the system prompt states (dates.prompt_days)."""
    out = {}
    for uid, df in users.items():
        first, last = data_range(df)
        out[uid] = dates.prompt_days(first.date(), last.date(), last_complete_day(df).date())
    return out


def score_answer(record: dict[str, Any], truth: dict[str, Any],
                 prompt: set[date] | None = None) -> dict[str, Any]:
    answer, calls = record["answer"], record["tool_calls"]
    reference = date.fromisoformat(truth["reference_date"])
    question = truth["question"]
    routes = _routes(truth)
    ok_calls = [c for c in calls if not c["is_error"]]

    numbers = answer_numbers(answer, reference)
    sources = number_sources(calls, question, reference)
    topics = guideline_topics(calls)
    unsupported, low_confidence, number_tags, computed = [], [], [], []
    for n in numbers:
        source, fallback = support(n, sources, answer or "", topics)
        if source is None:
            (low_confidence if _is_small_integer(n) else unsupported).append(n.text)
        else:
            number_tags.append({"text": n.text, "source": source})
            if fallback:
                computed.append(n.text)

    mentions = dates.find_dates(answer, reference)
    day_sources = dates.day_sources(ok_calls, question, reference, prompt or set())
    unsupported_dates, date_tags = dates.check_sourced(
        mentions, day_sources, dates.question_months(question, reference))
    computed += [t["text"] for t in date_tags if t["source"] == dates.FALLBACK]

    matched = [r for r in routes if all(any(_call_matches(e, c) for c in calls) for e in r)]
    tool_result_correct = bool(matched) if routes else None
    stated = [_route_stated(r, numbers, mentions, answer or "", question, reference)
              for r in (matched or routes)]
    answer_correct = None if all(s is None for s in stated) else any(s for s in stated)

    refusal = None
    if truth["expect_refusal"]:
        refusal = (bool(answer) and REFUSAL.search(answer) is not None and not unsupported
                   and not low_confidence)
    coverage = [coverage_mentioned(answer, c["value"])
                for e in truth["expected_calls"] for c in e.get("coverage") or []]
    coverage_ok = all(coverage) if coverage else None

    route_tools = {c["tool"] for r in routes for c in r}
    validity = [classify(c, route_tools) for c in calls]
    verdict = verdict_correct(answer, truth)
    end_to_end = (record["status"] == "answered" and not unsupported and not unsupported_dates
                  and all(v is not False for v in (tool_result_correct, answer_correct,
                                                    verdict, refusal, coverage_ok)))
    return {
        "question_id": record["question_id"], "repeat": record["repeat"],
        "category": truth["category"], "status": record["status"],
        "tool_result_correct": tool_result_correct,
        "answer_correct": answer_correct,
        "verdict_correct": verdict,
        "refusal_correct": refusal,
        "coverage_mentioned": coverage_ok,
        "end_to_end": end_to_end,
        "numbers": len(numbers), "unsupported_numbers": unsupported,
        "low_confidence_numbers": low_confidence,
        "dates": dates.count_items(mentions), "unsupported_dates": unsupported_dates,
        "model_computed": computed,
        "number_sources": number_tags, "date_sources": date_tags,
        "tools_called": [c["name"] for c in calls],
        "call_validity": validity,
        "no_call": (not calls) if routes else None,
    }


# --- a run ---------------------------------------------------------------------

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
    n_numbers = sum(r["numbers"] for r in rows)
    out["unsupported_numbers"] = _count(sum(len(r["unsupported_numbers"]) for r in rows),
                                        n_numbers)
    out["answers_with_unsupported_number"] = _count(
        sum(bool(r["unsupported_numbers"]) for r in rows), len(rows))
    out["unsupported_dates"] = _count(sum(len(r["unsupported_dates"]) for r in rows),
                                      sum(r["dates"] for r in rows))
    out["answers_with_unsupported_date"] = _count(
        sum(bool(r["unsupported_dates"]) for r in rows), len(rows))
    out["low_confidence_numbers"] = _count(
        sum(len(r["low_confidence_numbers"]) for r in rows), n_numbers)
    out["model_computed"] = _count(sum(len(r["model_computed"]) for r in rows),
                                   n_numbers + sum(r["dates"] for r in rows))
    out["answers_with_model_computed"] = _count(sum(bool(r["model_computed"]) for r in rows),
                                                len(rows))
    out["no_call"] = _passed([r["no_call"] for r in rows])
    return out


SUMMARY_ROWS = (*METRICS, "unsupported_numbers", "answers_with_unsupported_number",
                "unsupported_dates", "answers_with_unsupported_date", "low_confidence_numbers",
                "model_computed", "answers_with_model_computed", "no_call")


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
              ground_truth: dict[str, Any],
              prompt_days: dict[str, set[date]] | None = None) -> dict[str, Any]:
    """Scores one line per (question, repeat); usage and cost count every attempt.
    prompt_days: per user, the days the system prompt states (prompt_days_for)."""
    truth = {q["id"]: q for q in ground_truth["questions"]}
    lines = latest_lines(attempts)
    rows = [score_answer(r, truth[r["question_id"]],
                         (prompt_days or {}).get(truth[r["question_id"]]["user_id"]))
            for r in lines]
    repeats = sorted({r["repeat"] for r in rows})

    per_repeat = {k: _measures([r for r in rows if r["repeat"] == k]) for k in repeats}
    pooled = _measures(rows)
    summary = {m: _spread([per_repeat[k][m] for k in repeats], pooled[m])
               for m in SUMMARY_ROWS}
    for m in METRICS:
        summary[m]["n_questions"] = len({r["question_id"] for r in rows if r[m] is not None})

    by_category = {cat: _measures([r for r in rows if r["category"] == cat])
                   for cat in sorted({r["category"] for r in rows})}

    calls = [label for r in rows for label in r["call_validity"]]
    call_validity = {label: _count(calls.count(label), len(calls)) for label in LABELS}

    per_question = {}
    for qid in config["question_ids"]:
        mine = sorted((r for r in rows if r["question_id"] == qid), key=lambda r: r["repeat"])
        entry: dict[str, Any] = {"category": truth[qid]["category"],
                                 "status": [r["status"] for r in mine]}
        for m in METRICS:
            values = [r[m] for r in mine]
            if any(v is not None for v in values):
                entry[m] = values
        passes = [r["end_to_end"] for r in mine]
        entry["pass_rate"] = _count(sum(passes), len(passes))
        entry["pass_all_repeats"] = bool(passes) and all(passes)
        for key in ("unsupported_numbers", "low_confidence_numbers", "unsupported_dates",
                    "model_computed", "tools_called", "call_validity"):
            entry[key] = [r[key] for r in mine]
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
    pass_all = sum(e["pass_all_repeats"] for e in per_question.values())
    return {
        "scorer_version": SCORER_VERSION,
        "model": config["model"], "questions_version": config["questions_version"],
        "answers": len(lines), "repeats": repeats, "statuses": statuses,
        "summary": summary, "by_category": by_category,
        # pass^k: questions whose every repeat passes end to end (k = number of repeats).
        "pass_all_repeats": {**_count(pass_all, len(per_question)), "repeats": len(repeats)},
        "call_validity": call_validity,
        "consistency": _count(consistent, len(per_question)),
        "usage": {"attempts": len(attempts),
                  "cost_usd": round(sum(r["cost_usd"] for r in attempts), 4),
                  "requests": sum(r["steps"] for r in attempts), **tokens,
                  "seconds_per_answer_median": median(r["seconds"] for r in lines)
                  if lines else None},
        "per_question": per_question,
        "answers_scored": rows,
    }
