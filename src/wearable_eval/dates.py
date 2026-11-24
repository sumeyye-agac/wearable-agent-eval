"""Find the dates an answer states, resolve them to calendar days, and check them.

Written as general rules, not for one model's wording. Recognised forms:
- ISO: 2016-05-11, optionally after a weekday ("Thursday, 2016-03-17")
- day before month: 17 March, 17th of March 2016, Tue 15 Mar
- lists and ranges of days before a month: 11–17 April, from 16 to 22 March,
  16, 17, 18, 21 and 22 March, Saturday 19 and Sunday 20 March, Monday 21 to Friday 25 March
- month before day: March 17, 2016, Thursday, March 17, May 5th, March 12–31, 2016
- month and year: June 2016 (checked as a month)
- numeric with a year: 5/11/2016 (month/day or day/month; supported if either reading is).
  Without a year "5/7" is read as a fraction ("5/7 nights"), not a date.
- weekday and day with no month: "Thursday 17, Friday 18" takes the month of the closest
  date before it in the same text, or the reference date's month.
Month names must be capitalised ("May", not "may"). Years default to the reference year.

A day counts as supported when it appears in a tool output (any ISO date, or any day of a
start-to-end window), in the question (any day of a range it names), or in the system
prompt: the data range's first and last day, the reference date (last complete day), and
the "this week" / "last week" windows. A month counts as supported when a supported day
falls in it or the question names it. A weekday written next to a date must be that date's
weekday.

Fallback (counted, not silent): the evening before a day of a sleep result (the day a
night began) is accepted when no source has it. The tools give it as night_of, so a firing
means the model worked the date out itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

_MONTHS = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
           "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
           "December": 12, "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Jun": 6, "Jul": 7,
           "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}
_WEEKDAYS = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
             "Saturday": 5, "Sunday": 6, "Mon": 0, "Tue": 1, "Tues": 1, "Wed": 2, "Thu": 3,
             "Thur": 3, "Thurs": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _alt(names: dict[str, int]) -> str:
    return "|".join(sorted(names, key=len, reverse=True))


MONTH = rf"(?:{_alt(_MONTHS)})\b\.?"
WEEKDAY = rf"(?:{_alt(_WEEKDAYS)})\b\.?"
DAY = r"(?<!\d)(?:3[01]|[12]\d|0?[1-9])(?:st|nd|rd|th)?(?!\d)"
_WD = rf"(?:{WEEKDAY},?\s+)"
_ITEM = rf"{_WD}?(?:the\s+)?{DAY}"
# Between items: a range (–, -, to, through, until) or a list (",", "and", "or"). The em
# dash is left out: in English it sets off part of a sentence ("3 — Sunday 20 March").
_SEP = r"(?:\s*[–-]\s*|\s*,\s*(?:and\s+|or\s+)?|\s+(?:to|through|until|and|or)\s+)"
_RANGE_SEP = re.compile(r"^\s*(?:[–-]|to|through|until)\s*$")
_YEAR = r"(?:,?\s+(?P<year>\d{4}))?(?!\d)"

_PATTERNS = [  # tried in this order; a later pattern never takes text an earlier one took
    ("iso", re.compile(rf"(?:(?P<wd>{WEEKDAY}),?\s+)?(?<!\d)(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})(?!\d)")),
    ("slash", re.compile(r"(?<![\d/])(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})(?![\d/])")),
    ("day_month", re.compile(rf"(?<![\w.,]){_ITEM}(?:{_SEP}{_ITEM})*\s+(?:of\s+)?"
                             rf"(?P<month>{MONTH}){_YEAR}")),
    ("month_day", re.compile(rf"(?:(?P<wd>{WEEKDAY}),?\s+)?(?P<month>{MONTH})\s+{DAY}"
                             rf"(?:{_SEP}{DAY})*{_YEAR}")),
    ("month_year", re.compile(rf"(?P<month>{MONTH})\s+(?P<year>\d{{4}})(?!\d)")),
    # "(?![.,]\d)": the start of a longer number is not a day ("Friday, 23,071 steps").
    ("weekday_day", re.compile(rf"(?P<wd>{WEEKDAY}),?\s+(?:the\s+)?(?P<day>{DAY})(?![.,]\d)"
                               r"(?!\s*(?:h|hr|hrs|hours?|min|minutes?|steps|bpm|%)\b)")),
]
_ITEM_RE = re.compile(rf"(?:(?P<wd>{WEEKDAY}),?\s+)?(?:the\s+)?(?P<day>{DAY})")


@dataclass
class DateItem:
    day: date | None           # None for a month mention or an impossible date
    month: tuple[int, int] | None = None   # (year, month) for a month mention
    weekday: str | None = None             # as written next to the day
    problem: str | None = None             # why it cannot be a real date


@dataclass
class Mention:
    text: str
    start: int
    end: int
    items: list[DateItem] = field(default_factory=list)
    either: bool = False   # numeric date: items are alternative readings, not several days


def _day_number(token: str) -> int:
    return int(re.match(r"\d+", token).group())


def _make(year: int, month: int, day: int, weekday: str | None) -> DateItem:
    try:
        return DateItem(date(year, month, day), weekday=weekday)
    except ValueError:
        return DateItem(None, weekday=weekday, problem=f"no such date {year}-{month:02d}-{day:02d}")


def _weekday_word(token: str | None) -> str | None:
    return token.rstrip(".,") if token else None


def find_dates(text: str | None, reference: date) -> list[Mention]:
    """Every date mention in the text, in reading order."""
    if not text:
        return []
    taken: list[tuple[int, int]] = []
    found: list[tuple[str, re.Match[str]]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((kind, m))
    found.sort(key=lambda km: km[1].start())

    mentions: list[Mention] = []
    last_month = (reference.year, reference.month)
    for kind, m in found:
        g = m.groupdict()
        year = int(g["year"]) if g.get("year") else reference.year
        mention = Mention(m.group(0), m.start(), m.end())
        if kind == "iso":
            y, mo, d = map(int, g["iso"].split("-"))
            mention.items.append(_make(y, mo, d, _weekday_word(g.get("wd"))))
        elif kind == "slash":
            a, b, y = int(g["a"]), int(g["b"]), int(g["y"])
            readings = [r for r in (_make(y, a, b, None), _make(y, b, a, None)) if r.day]
            mention.either = True
            mention.items = readings or [_make(y, a, b, None)]
        elif kind == "day_month":
            month = _MONTHS[g["month"].rstrip(".")]
            body = m.group(0)[: m.start("month") - m.start()]
            found_items = list(_ITEM_RE.finditer(body))
            # A list gives every day a weekday or none ("Saturday 19 and Sunday 20"). When
            # only some have one, the days before the first weekday are not part of the date.
            with_wd = [i for i, it in enumerate(found_items) if it.group("wd")]
            if with_wd and len(with_wd) < len(found_items):
                cut = found_items[with_wd[0]].start()
                mention.start += cut
                mention.text = mention.text[cut:]
                found_items = found_items[with_wd[0]:]
            for it in found_items:
                mention.items.append(_make(year, month, _day_number(it.group("day")),
                                           _weekday_word(it.group("wd"))))
            last_month = (year, month)
        elif kind == "month_day":
            month = _MONTHS[g["month"].rstrip(".")]
            body = m.group(0)[m.end("month") - m.start():]
            if g.get("year"):
                body = body[: m.start("year") - m.end("month")]
            days = [_day_number(d.group()) for d in re.finditer(DAY, body)]
            for i, d in enumerate(days):
                mention.items.append(_make(year, month, d,
                                           _weekday_word(g.get("wd")) if i == 0 else None))
            last_month = (year, month)
        elif kind == "month_year":
            month = _MONTHS[g["month"].rstrip(".")]
            mention.items.append(DateItem(None, month=(year, month)))
            last_month = (year, month)
        elif kind == "weekday_day":
            y, mo = last_month
            mention.items.append(_make(y, mo, _day_number(g["day"]), _weekday_word(g["wd"])))
        if kind == "iso" and mention.items[0].day:
            d = mention.items[0].day
            last_month = (d.year, d.month)
        mentions.append(mention)
    return mentions


def _span(a: date, b: date) -> set[date]:
    a, b = sorted((a, b))
    return {a + timedelta(n) for n in range((b - a).days + 1)}


def range_days(text: str, mentions: list[Mention]) -> set[date]:
    """Every day inside the ranges a text names: "from 5 to 9 May" gives 5 to 9 May, and
    "30 April to 4 May" (two mentions joined by a range word) gives 30 April to 4 May."""
    days: set[date] = set()
    for mention in mentions:
        for item in mention.items:
            if item.day:
                days.add(item.day)
        if (len(mention.items) == 2 and all(i.day for i in mention.items)
                and _is_range(mention)):
            days |= _span(mention.items[0].day, mention.items[1].day)
    for left, right in zip(mentions, mentions[1:]):
        if (_RANGE_SEP.match(text[left.end:right.start]) and left.items and right.items
                and left.items[-1].day and right.items[0].day):
            days |= _span(left.items[-1].day, right.items[0].day)
    return days


def _is_range(mention: Mention) -> bool:
    return bool(re.search(r"[–-]|\b(?:to|through|until)\b", mention.text))


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _collect(value: Any, out: set[date]) -> None:
    if isinstance(value, str) and _ISO.match(value):
        try:
            out.add(date.fromisoformat(value))
        except ValueError:
            pass
    elif isinstance(value, list):
        for v in value:
            _collect(v, out)
    elif isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
        if isinstance(start, str) and isinstance(end, str) and _ISO.match(start) \
                and _ISO.match(end):
            a, b = date.fromisoformat(start), date.fromisoformat(end)
            out.update(a + timedelta(n) for n in range((b - a).days + 1))
        for v in value.values():
            _collect(v, out)


def prompt_weeks(reference: date) -> set[date]:
    """"This week" and "last week" as the agent's system prompt defines them: the 7 days
    ending on the reference date, and the 7 days before those."""
    return _span(reference - timedelta(13), reference)


def supported_days(outputs: list[Any], question: str, reference: date) -> set[date]:
    days: set[date] = {reference} | prompt_weeks(reference)
    _collect(outputs, days)
    days |= range_days(question, find_dates(question, reference))
    return days


def prompt_days(first: date, last: date, reference: date) -> set[date]:
    """The days the system prompt states: the data range's ends, the reference date and
    the "this week" / "last week" windows."""
    return {first, last, reference} | prompt_weeks(reference)


SLEEP_METRICS = ("minutes_asleep", "minutes_in_bed")
FALLBACK = "fallback:evening_before"


def _sleep_days(call: dict[str, Any]) -> set[date]:
    """The days of a call's result that are sleep rows (dated by the wake-up day)."""
    days: set[date] = set()
    args = call.get("input") if isinstance(call.get("input"), dict) else {}
    if args.get("metric") in SLEEP_METRICS:
        _collect(call["output"], days)
    elif call["name"] == "missing_data_report" and isinstance(call["output"], dict):
        for metric in SLEEP_METRICS:
            _collect((call["output"].get("metrics") or {}).get(metric), days)
    return days


def day_sources(calls: list[dict[str, Any]], question: str, reference: date,
                prompt: set[date]) -> dict[date, str]:
    """Every day an answer may state, with where it comes from: "tool:<name>", "question",
    "prompt", or the counted fallback (see FALLBACK). The first source found wins."""
    sources: dict[date, str] = {}
    for call in calls:
        if call["is_error"]:
            continue
        days: set[date] = set()
        _collect(call["output"], days)
        for d in days:
            sources.setdefault(d, f"tool:{call['name']}")
    for d in range_days(question, find_dates(question, reference)):
        sources.setdefault(d, "question")
    for d in prompt | {reference} | prompt_weeks(reference):
        sources.setdefault(d, "prompt")
    for call in calls:
        if not call["is_error"]:
            for d in _sleep_days(call):
                sources.setdefault(d - timedelta(1), FALLBACK)
    return sources


def check(mentions: list[Mention], supported: set[date], question_months: set[tuple[int, int]]
          ) -> list[str]:
    """The unsupported date items, each as '<text as written>: <reason>'."""
    return check_sourced(mentions, dict.fromkeys(supported, ""), question_months)[0]


def check_sourced(mentions: list[Mention], sources: dict[date, str],
                  question_months: set[tuple[int, int]]
                  ) -> tuple[list[str], list[dict[str, str]]]:
    """The unsupported date items ('<text as written>: <reason>'), and the supported ones
    with their source ({"text", "source"}). A month's source is "month"."""
    months = {(d.year, d.month) for d in sources} | question_months
    problems: list[str] = []
    tagged: list[dict[str, str]] = []
    for mention in mentions:
        if mention.either and all(i.day for i in mention.items):
            found = [sources[i.day] for i in mention.items if i.day in sources]
            if found:
                tagged.append({"text": mention.text, "source": found[0]})
            else:
                problems.append(f"{mention.text}: not in the tool outputs or the question")
            continue
        for item in mention.items:
            if item.problem:
                problems.append(f"{mention.text}: {item.problem}")
            elif item.month:
                if item.month in months:
                    tagged.append({"text": mention.text, "source": "month"})
                else:
                    problems.append(f"{mention.text}: month not in the tool outputs or the question")
            elif item.day not in sources:
                problems.append(f"{mention.text}: {item.day} not in the tool outputs or the question")
            elif item.weekday and _WEEKDAYS[item.weekday] != item.day.weekday():
                problems.append(f"{mention.text}: {item.day} is a {_NAMES[item.day.weekday()]}, "
                                f"not {item.weekday}")
            else:
                tagged.append({"text": mention.text, "source": sources[item.day]})
    return problems, tagged


def question_months(question: str, reference: date) -> set[tuple[int, int]]:
    months = set()
    for mention in find_dates(question, reference):
        for item in mention.items:
            if item.month:
                months.add(item.month)
            elif item.day:
                months.add((item.day.year, item.day.month))
    return months


def count_items(mentions: list[Mention]) -> int:
    """Dates stated: one per day or month written; a numeric date counts once."""
    return sum(1 if m.either else len(m.items) for m in mentions)
