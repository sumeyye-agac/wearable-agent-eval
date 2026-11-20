"""Answer metrics that are plain text checks: regular expressions, no LLM judge.

Only coverage_mentioned is here so far. Tool-selection accuracy, numeric accuracy,
unsupported-claim rate and refusal accuracy follow when scoring is built.
"""

from __future__ import annotations

import re

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
