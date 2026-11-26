"""The README results block: Wilson interval and no hand-typed numbers."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import make_results_table as mrt  # noqa: E402


@pytest.mark.parametrize(("k", "n", "lo", "hi"), [
    (5, 10, 0.2366, 0.7634),   # textbook value, symmetric around 0.5
    (0, 10, 0.0, 0.2775),      # z^2 / (n + z^2)
    (10, 10, 0.7225, 1.0),
])
def test_wilson_known_values(k, n, lo, hi):
    assert mrt.wilson(k, n) == pytest.approx((lo, hi), abs=1e-4)


def test_cell_mean_over_runs_uses_mean_k_and_n():
    one = mrt.cell([(29, 34), (33, 34), (33, 34)])
    assert one.startswith("mean 93.1%, runs 85.3%–97.1%")
    lo, hi = mrt.wilson(95 / 3, 34)
    assert f"CI {100 * lo:.1f}%–{100 * hi:.1f}%" in one


def test_kind_of_takes_first_failed_check():
    base = {"end_to_end": False, "refusal_correct": None, "tool_result_correct": True,
            "coverage_mentioned": True}
    assert mrt.kind_of({**base, "end_to_end": True}) is None
    assert mrt.kind_of({**base, "refusal_correct": False,
                        "tool_result_correct": False}) == "refusal_failed"
    assert mrt.kind_of({**base, "tool_result_correct": False,
                        "coverage_mentioned": False}) == "wrong_tool_result"
    assert mrt.kind_of({**base, "coverage_mentioned": False}) == "missing_coverage"


def test_readme_block_matches_saved_runs():
    readme = mrt.README.read_text(encoding="utf-8")
    assert mrt.replace_block(readme, mrt.build_block()) == readme


def test_why_names_the_failed_check():
    q = {"expected_calls": [{"tool": "most_active_day",
                             "match_on": {"top_days.0.date": "2016-04-14"},
                             "coverage": [{"path": "days_with_data", "value": 5}]}]}
    a = {"unsupported_numbers": ["73"], "low_confidence_numbers": []}
    call = {"name": "most_active_day", "is_error": False,
            "output": {"top_days": [{"date": "2016-03-14"}]}}
    record = {"answer": "You did 73 minutes.", "tool_calls": [call]}
    assert mrt.why("refusal_failed", a, q, record) == "no refusal wording in the answer"
    assert mrt.why("wrong_tool_result", a, q, record) == (
        '`top_days.0.date` expected "2016-04-14", got "2016-03-14"')
    assert mrt.why("wrong_tool_result", a, q, {**record, "tool_calls": []}) == (
        "no successful `most_active_day` call")
    assert mrt.why("missing_coverage", a, q, record).startswith("the tool result covers 5 days")


def test_a_parse_error_counts_as_an_invalid_call():
    scores = {"repeats": [1], "answers_scored": [
        {"repeat": 1, "call_validity": ["valid", "off_route"]},
        {"repeat": 1, "call_validity": []}]}
    unread = {"parse_errors": [{"what": "LiteRT-LM could not parse the tool call"}]}
    transcript = [
        {"question_id": "q1", "repeat": 1, "status": "answered",
         "requests": [{"response": {"litert": {"parse_errors": []}}}]},
        {"question_id": "q2", "repeat": 1, "status": "answered",
         "requests": [{"response": {"litert": unread}}]},
        {"question_id": "q3", "repeat": 1, "status": "answered",   # Haiku: no litert
         "requests": [{"response": {}}]},
    ]
    parse_errors = mrt.parse_errors_by_repeat(transcript)
    assert parse_errors == {1: 1}
    assert mrt.per_repeat(scores, "valid_calls", parse_errors) == [(1, 3)]
    assert mrt.per_repeat(scores, "valid_calls") == [(1, 2)]
