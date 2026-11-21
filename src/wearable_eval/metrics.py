"""Answer metrics that are plain text checks: regular expressions, no LLM judge.

coverage_mentioned and supported_numbers (the reference set for the unsupported-claim
check) are here so far. The scoring itself follows when it is built.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable

_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten")


def _count_pattern(count: int) -> re.Pattern[str]:
    # The digits on their own: not inside a longer number ("15", "5,000", "0.5") or a date
    # ("2016-05-09" holds "05", not "5").
    digits = rf"(?<![\d.,]){count}(?!\d|[.,]\d)"
    if 0 <= count < len(_WORDS):
        return re.compile(rf"{digits}|\b{_WORDS[count]}\b", re.IGNORECASE)
    return re.compile(digits)


def coverage_mentioned(answer: str | None, day_count: int) -> bool:
    """True if the answer states the number of days the result rests on, as digits ("5")
    or, up to ten, as an English word ("five"). It checks that the count is stated, not
    that it is attached to the right noun: "5 days" and "5 of 7 nights" both pass."""
    if not answer:
        return False
    return _count_pattern(day_count).search(answer) is not None


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _is_date(value: Any) -> bool:
    return isinstance(value, str) and _ISO_DATE.fullmatch(value) is not None


def _collect(value: Any, out: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.add(float(value))
    elif isinstance(value, list):
        if value and all(_is_date(v) for v in value):
            out.add(float(len(value)))          # "2 days missing" from a list of 2 dates
        for item in value:
            _collect(item, out)
    elif isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
        if _is_date(start) and _is_date(end):  # "of 7 days" from a 7-day window
            out.add(float((date.fromisoformat(end) - date.fromisoformat(start)).days + 1))
        for item in value.values():
            _collect(item, out)


def supported_numbers(outputs: Iterable[Any]) -> set[float]:
    """Every number an answer may state on the strength of these tool outputs.

    That is every numeric value in them, plus two counts the tools imply without stating:
    the length of each list of dates (missing_dates with 2 entries supports "2 days
    missing") and the length in days, inclusive, of each window that has a start and an
    end date (2016-05-05 to 2016-05-11 supports "of 7 days"). Error messages (strings) add
    nothing. Agreed 2026-11-21; the unsupported-claim check compares the answer's numbers
    against this set.
    """
    out: set[float] = set()
    for output in outputs:
        _collect(output, out)
    return out
