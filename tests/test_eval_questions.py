import json
import sys
from pathlib import Path

import pytest

from wearable_eval.ground_truth import build, load_users, resolve
from wearable_eval.questions import CATEGORIES, load_questions, parse_questions

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_ground_truth  # noqa: E402


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


@pytest.fixture(scope="module")
def questions(users):
    return load_questions(REPO / "eval/questions.yaml", set(users))


def test_every_category_is_used(questions):
    assert {q.category for q in questions} == set(CATEGORIES)


def test_answerable_questions_require_numbers(questions):
    for q in questions:
        if not q.expect_refusal:
            assert q.expected_calls and any(c.answer for c in q.expected_calls), q.id


def test_committed_ground_truth_is_up_to_date():
    committed = (REPO / "eval/ground_truth.json").read_text(encoding="utf-8")
    assert committed == build_ground_truth.render(), "rerun scripts/build_ground_truth.py"


def test_ground_truth_matches_known_values(questions, users):
    # Spot checks against figures read straight from the data, so a change in a tool that
    # moves them shows up here and not only as a regenerated file.
    gt = {q["id"]: q for q in build(questions, users)["questions"]}
    act01 = gt["act-01"]["expected_calls"][0]["match_on"]
    assert act01["most_active.date"] == "2016-03-17"
    assert act01["most_active.steps"] == 26100
    assert act01["most_active.steps_above_second"] == 26100 - 23071
    # The example answer in the ask-your-wearable README: 17,022 steps on 2016-04-14.
    assert gt["act-05"]["expected_calls"][0]["output"]["top_days"][0]["date"] == "2016-04-14"
    assert gt["act-05"]["expected_calls"][0]["answer"][0]["value"] == 17022
    assert gt["act-02"]["expected_calls"][0]["answer"][0]["value"] == 64766
    slp01 = gt["slp-01"]["expected_calls"][0]["answer"][0]
    assert (slp01["value"], slp01["text"]) == (451.3, "7 h 31 min")
    una04 = gt["una-04"]["expected_calls"][0]["match_on"]
    assert una04["combined.reaches_recommendation"] is None


def test_resolve_follows_lists_and_rejects_missing_paths():
    output = {"top_days": [{"steps": 5}, {"steps": 3}]}
    assert resolve(output, "top_days.1.steps") == 3
    with pytest.raises(KeyError):
        resolve(output, "top_days.2.steps")
    with pytest.raises(KeyError):
        resolve(output, "mean")


def _one(**overrides):
    q = {"id": "x-01", "user_id": "1", "category": "sleep", "question": "How did I sleep?",
         "expected_calls": [{"tool": "weekly_summary", "args": {"metric": "minutes_asleep"},
                             "answer": ["mean"]}]}
    q.update(overrides)
    return {"schema_version": 1, "questions": [q]}


@pytest.mark.parametrize("overrides, message", [
    ({"category": "mood"}, "category"),
    ({"user_id": "999"}, "not in the demo data"),
    ({"expected_calls": [{"tool": "nope", "answer": ["mean"]}]}, "unknown tool"),
    ({"expected_calls": [{"tool": "weekly_summary", "args": {"metric": "steps", "x": 1},
                          "answer": ["mean"]}]}, "unknown argument"),
    ({"expected_calls": [{"tool": "weekly_summary", "answer": ["mean"]}]}, "missing required"),
    ({"expected_calls": [{"tool": "weekly_summary", "args": {"metric": "steps"}}]},
     "needs answer or match_on"),
    ({"expect_refusal": True}, "refusal question cannot require numbers"),
])
def test_parse_rejects_invalid_questions(overrides, message):
    with pytest.raises(ValueError, match=message):
        parse_questions(_one(**overrides), users={"1"})


def test_parse_rejects_duplicate_ids():
    data = _one()
    data["questions"].append(dict(data["questions"][0]))
    with pytest.raises(ValueError, match="duplicate id"):
        parse_questions(data, users={"1"})


def test_build_rejects_a_path_not_in_the_output(users):
    data = _one(user_id="6962181067")
    data["questions"][0]["expected_calls"][0]["answer"] = ["not_a_field"]
    with pytest.raises(ValueError, match="not_a_field"):
        build(parse_questions(data), users)


def test_ground_truth_file_is_valid_json():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    assert data["numeric_tolerance"] == 0.01
