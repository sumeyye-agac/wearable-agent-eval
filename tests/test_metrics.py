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
