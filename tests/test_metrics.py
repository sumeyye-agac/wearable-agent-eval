import pytest

from wearable_eval.metrics import coverage_mentioned


@pytest.mark.parametrize("answer", [
    "You averaged 9 h 16 min, but only 2 of 7 nights have data.",
    "Based on two nights (5 to 9 May are missing).",
    "Two nights were recorded.",
    "The total covers 2 days.",
])
def test_coverage_mentioned_finds_the_count(answer):
    assert coverage_mentioned(answer, 2)


@pytest.mark.parametrize("answer", [
    "You took 25,000 steps on 2016-05-02.",       # 2 only inside a number and a date
    "You slept 9.2 hours on average.",
    "Your sleep averaged 9 h 16 min this week.",  # no day count at all
    "",
    None,
])
def test_coverage_mentioned_ignores_digits_inside_other_numbers(answer):
    assert not coverage_mentioned(answer, 2)


def test_coverage_mentioned_above_ten_uses_digits_only():
    assert coverage_mentioned("over 23 nights", 23)
    assert not coverage_mentioned("over 123 nights", 23)


def test_supported_numbers_adds_date_list_lengths_and_window_lengths():
    from wearable_eval.metrics import supported_numbers
    output = {"start": "2016-05-05", "end": "2016-05-11", "days_with_data": 5,
              "mean": 451.3, "missing_dates": ["2016-05-07", "2016-05-08"],
              "complete_week": False,
              "period_a": {"start": "2016-03-01", "end": "2016-03-31", "mean": 82}}
    numbers = supported_numbers([output])
    assert {5.0, 451.3, 82.0} <= numbers   # values as given
    assert 2.0 in numbers                   # two missing dates
    assert 7.0 in numbers                   # the 7-day window
    assert 31.0 in numbers                  # the nested period's window
    assert 0.0 not in numbers and 1.0 not in numbers  # False is not a number


def test_supported_numbers_on_real_caveat_questions():
    # The case that raised the rule: "5 of 7 days, 2 days missing" for act-06.
    import json
    from pathlib import Path
    from wearable_eval.metrics import supported_numbers
    gt = json.loads((Path(__file__).resolve().parents[1] / "eval/ground_truth.json")
                    .read_text(encoding="utf-8"))
    out = {q["id"]: q for q in gt["questions"]}["act-06"]["expected_calls"][0]["output"]
    assert {5.0, 7.0, 2.0, 34328.0} <= supported_numbers([out])


def test_supported_numbers_ignores_error_strings():
    from wearable_eval.metrics import supported_numbers
    assert supported_numbers(["No data between 2016-06-01 and 2016-06-30."]) == set()
