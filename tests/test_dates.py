"""Date parsing and checking. Cases are general forms seen in real answers (run 1, Haiku
5.5), plus other common wordings, so the rules do not depend on one model's phrasing."""

from datetime import date

import pytest

from wearable_eval.dates import check, find_dates, question_months, supported_days
from wearable_eval.scoring import answer_numbers, is_supported

REF = date(2016, 5, 11)


def days(text, reference=REF):
    return [[i.day for i in m.items] for m in find_dates(text, reference)]


D = date


@pytest.mark.parametrize("text, expected", [
    # single dates
    ("on 2016-03-17", [[D(2016, 3, 17)]]),
    ("Thursday, 2016-03-17", [[D(2016, 3, 17)]]),
    ("on 17 March", [[D(2016, 3, 17)]]),
    ("the 17th of March 2016", [[D(2016, 3, 17)]]),
    ("March 17, 2016", [[D(2016, 3, 17)]]),
    ("Thursday, March 17", [[D(2016, 3, 17)]]),
    ("Tue 15 Mar", [[D(2016, 3, 15)]]),
    ("Mar. 15th", [[D(2016, 3, 15)]]),
    # ranges, written several ways
    ("the week of 11–17 April 2016", [[D(2016, 4, 11), D(2016, 4, 17)]]),
    ("from 11 - 17 April", [[D(2016, 4, 11), D(2016, 4, 17)]]),
    ("from 16 to 22 March", [[D(2016, 3, 16), D(2016, 3, 22)]]),
    ("March 12–31, 2016", [[D(2016, 3, 12), D(2016, 3, 31)]]),
    ("April 4 through 10", [[D(2016, 4, 4), D(2016, 4, 10)]]),
    ("Monday 21 to Friday 25 March", [[D(2016, 3, 21), D(2016, 3, 25)]]),
    ("30 April to 4 May 2016", [[D(2016, 4, 30)], [D(2016, 5, 4)]]),
    # lists
    ("16, 17, 18, 21 and 22 March",
     [[D(2016, 3, d) for d in (16, 17, 18, 21, 22)]]),
    ("Saturday 19 and Sunday 20 March", [[D(2016, 3, 19), D(2016, 3, 20)]]),
    ("May 5, 6 and 7", [[D(2016, 5, 5), D(2016, 5, 6), D(2016, 5, 7)]]),
    # weekday and day without a month take the month from the text before them
    ("On 16 March you were active. Thursday 17 and Friday 18 had none.",
     [[D(2016, 3, 16)], [D(2016, 3, 17)], [D(2016, 3, 18)]]),
    # numeric dates need a year; both readings are kept
    ("on 5/11/2016", [[D(2016, 5, 11), D(2016, 11, 5)]]),
    # months
    ("in June 2016", [[None]]),
])
def test_dates_are_found_and_resolved(text, expected):
    assert days(text) == expected


@pytest.mark.parametrize("text", [
    "4/7 nights", "26,100 steps", "you may 5 times", "Sunday 9 h 29 min",
    "7 h 31 min on average", "about 3 — or 4", "between 5 and 6 hours",
])
def test_not_dates(text):
    assert find_dates(text, REF) == []


def test_an_em_dash_after_a_count_is_not_a_range():
    # "3 — Sunday 20 March": 3 is a count, the date is Sunday 20 March.
    (mention,) = find_dates("Nights at or above 7 h: 3 — Sunday 20 March", REF)
    assert mention.text == "Sunday 20 March"
    assert [i.day for i in mention.items] == [D(2016, 3, 20)]


def test_a_list_keeps_only_the_days_from_the_first_weekday():
    (mention,) = find_dates("Below 7 h: 3 – Saturday 19 and Thursday 24 March", REF)
    assert [i.day for i in mention.items] == [D(2016, 3, 19), D(2016, 3, 24)]


def flags(answer, outputs=(), question="How did I sleep?", reference=REF):
    mentions = find_dates(answer, reference)
    return check(mentions, supported_days(list(outputs), question, reference),
                 question_months(question, reference))


WINDOW = {"start": "2016-03-16", "end": "2016-03-22", "missing_dates": ["2016-03-19"]}


def test_days_inside_a_tool_window_are_supported():
    assert flags("Monday 21 to Tuesday 22 March", [WINDOW]) == []
    assert flags("16, 17, 18, 21 and 22 March", [WINDOW]) == []


def test_a_day_outside_every_source_is_flagged():
    (problem,) = flags("On 23 March you walked more.", [WINDOW])
    assert "2016-03-23" in problem


def test_a_wrong_weekday_is_flagged():
    # 2016-03-17 was a Thursday.
    (problem,) = flags("Wednesday 17 March", [WINDOW | {"end": "2016-03-22"}])
    assert "Thursday, not Wednesday" in problem
    assert flags("Thursday 17 March", [WINDOW]) == []


def test_dates_in_the_question_and_the_prompt_weeks_are_supported():
    question = "Did I sleep more from 5 to 9 May 2016 than in the five nights before?"
    assert flags("Between 6 and 8 May you slept more.", question=question) == []
    # "this week" and "last week" as the system prompt defines them from the reference date
    assert flags("last week (2016-04-28 to 2016-05-04)") == []
    (problem,) = flags("last week (2016-04-27 to 2016-05-04)")
    assert "2016-04-27" in problem


def test_months():
    assert flags("I have no data for June 2016.", question="How much did I sleep in June 2016?") == []
    (problem,) = flags("I have no data for June 2016.", [WINDOW])
    assert "month" in problem


def test_digits_inside_dates_are_not_numbers():
    text = "In the week of 11–17 April 2016 you averaged 8,384 steps; on Saturday 19 and " \
           "Sunday 20 March nothing was recorded."
    assert [n.value for n in answer_numbers(text, REF)] == [8384]


@pytest.mark.parametrize("text, supported", [
    ("the last 7 days", True), ("the last 4 weeks", True), ("the previous 28 days", True),
    ("the 4 weeks before them", True), ("a 7-day window", True),
    ("the last 5 weeks", False), ("the past 10 days", False),
])
def test_period_lengths_are_supported_only_for_tool_windows(text, supported):
    (number,) = answer_numbers(text, REF)
    assert is_supported(number, set()) == supported
