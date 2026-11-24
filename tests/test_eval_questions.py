import json
import sys
from datetime import date
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
    gt = {q["id"]: q for q in build(questions, users, "v1")["questions"]}
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


def test_missing_data_cases_really_miss_data(questions, users):
    # A missing_data question must rest on a tool result that shows the gap; if the data or
    # a tool changed so the gap disappeared, the question would test nothing.
    gt = {q["id"]: q for q in build(questions, users, "v1")["questions"]}
    tagged = [q for q in questions if "missing_data" in q.tags]
    assert len(tagged) >= 10
    def short_period(period):
        # a compare_periods period with fewer days of data than days in it
        if not period:
            return False
        days = (date.fromisoformat(period["end"]) - date.fromisoformat(period["start"])).days + 1
        return period["days_with_data"] < days

    for q in tagged:
        out = gt[q.id]["expected_calls"][0]["output"]
        gap = (out.get("days_with_data", 7) < 7
               or out.get("enough_data") is False
               or short_period(out.get("period_a")) or short_period(out.get("period_b"))
               or out.get("n", 99) < 10)
        assert gap, q.id
    assert gt["slp-05"]["expected_calls"][0]["output"]["days_with_data"] == 2
    assert gt["una-06"]["expected_calls"][0]["output"]["enough_data"] is False
    assert gt["una-07"]["expected_calls"][0]["output"]["n"] == 0
    assert gt["una-08"]["expected_calls"][0]["output"]["mean"] is None


def test_resolve_follows_lists_and_rejects_missing_paths():
    output = {"top_days": [{"steps": 5}, {"steps": 3}]}
    assert resolve(output, "top_days.1.steps") == 3
    with pytest.raises(KeyError):
        resolve(output, "top_days.2.steps")
    with pytest.raises(KeyError):
        resolve(output, "mean")


def test_resolve_picks_a_list_item_by_key():
    output = {"daily": [{"date": "2016-03-31", "value": 412.0},
                        {"date": "2016-04-01", "value": 365.0}]}
    assert resolve(output, "daily.date=2016-04-01.value") == 365.0
    with pytest.raises(KeyError):
        resolve(output, "daily.date=2016-04-02.value")


def test_alternative_routes_are_built_like_expected_calls(questions, users):
    # cmp-03 can be answered by one compare_periods call or by two weekly_summary calls.
    gt = {q["id"]: q for q in build(questions, users, "v1.1")["questions"]}
    (route,) = gt["cmp-03"]["alternative_routes"]
    assert [c["match_on"]["total"] for c in route] == [74092, 61947]
    assert gt["cmp-03"]["expected_calls"][0]["match_on"]["difference_b_minus_a"] == -1735
    assert gt["act-01"]["alternative_routes"] == []


def test_answer_dates_accept_the_evening_before_for_sleep_only(users):
    data = _one(user_id="6962181067")
    call = data["questions"][0]["expected_calls"][0]
    call["answer_dates"] = ["lowest.date"]
    (date_,) = build(parse_questions(data), users, "v1")["questions"][0]["expected_calls"][0][
        "answer_dates"]
    assert date_["value"] == "2016-05-07"
    assert date_["accepted"] == ["2016-05-07", "2016-05-06"]
    call["args"] = {"metric": "steps"}
    (date_,) = build(parse_questions(data), users, "v1")["questions"][0]["expected_calls"][0][
        "answer_dates"]
    assert date_["accepted"] == [date_["value"]]
    call["answer_dates"] = ["mean"]
    with pytest.raises(ValueError, match="not a YYYY-MM-DD date"):
        build(parse_questions(data), users, "v1")


def _one(**overrides):
    q = {"id": "x-01", "user_id": "1", "category": "sleep", "question": "How did I sleep?",
         "expected_calls": [{"tool": "weekly_summary", "args": {"metric": "minutes_asleep"},
                             "answer": ["mean"]}]}
    q.update(overrides)
    return {"schema_version": 1, "version": "v1", "questions": [q],
            "reference_dates": {"1": "2024-01-14", "999": "2024-01-14",
                                "6962181067": "2016-05-11"}}


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
    ({"tags": "missing_data"}, "tags must be a list"),
    ({"alternative_routes": [[]]}, "alternative_routes must be a list of non-empty lists"),
    ({"alternative_routes": [{"tool": "weekly_summary"}]}, "alternative_routes must be"),
    ({"expected_calls": [], "alternative_routes": [[
        {"tool": "weekly_summary", "args": {"metric": "steps"}, "answer": ["total"]}]]},
     "need expected_calls"),
    ({"expect_refusal": True, "expected_calls": [
        {"tool": "weekly_summary", "args": {"metric": "steps"}, "match_on": ["total"]}],
      "alternative_routes": [[
        {"tool": "weekly_summary", "args": {"metric": "steps"}, "answer": ["total"]}]]},
     "refusal question cannot require numbers"),
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
        build(parse_questions(data), users, "v1")


def test_ground_truth_file_is_valid_json():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    assert data["numeric_tolerance"] == 0.01
    assert data["text_tolerance_minutes"] == 1
    assert data["questions_version"] == "v1.3"


@pytest.mark.parametrize("version", [None, 1, "1", "v0", "v1.0", "v1.1.1", "V1"])
def test_parse_requires_a_question_set_version(version):
    data = _one()
    if version is None:
        del data["version"]
    else:
        data["version"] = version
    with pytest.raises(ValueError, match="version must be v1, v1.1, v2"):
        parse_questions(data, users={"1"})


@pytest.mark.parametrize("version", ["v1", "v1.1", "v2", "v12.3"])
def test_parse_accepts_minor_versions(version):
    data = _one()
    data["version"] = version
    parse_questions(data, users={"1"})


def test_reference_dates_are_the_tools_anchor_and_in_the_prompt(questions, users):
    # The pinned "today" must be what the tools and the agent's system prompt use.
    from wearable.agent import system_prompt
    for q in questions:
        assert f"anchored on the last complete day, {q.reference_date}" in \
            system_prompt(users[q.user_id]), q.id


def test_build_rejects_a_wrong_reference_date(users):
    data = _one(user_id="6962181067")
    data["reference_dates"] = {"6962181067": "2016-05-12"}
    with pytest.raises(ValueError, match="not the tools' anchor 2016-05-11"):
        build(parse_questions(data), users, "v1")


def test_parse_requires_a_reference_date_for_every_user():
    data = _one()
    data["reference_dates"] = {}
    with pytest.raises(ValueError, match="no entry in reference_dates"):
        parse_questions(data, users={"1"})


def test_coverage_questions_carry_the_day_count_in_the_tool_output(questions, users):
    # The day count is a field of the tool output, so a number like "5 days" in an answer
    # is supported by the tool result, not invented.
    gt = {q["id"]: q for q in build(questions, users, "v1")["questions"]}
    expected = {"act-03": [5], "act-06": [5], "act-07": [5], "act-08": [6], "slp-05": [2],
                "slp-06": [6], "trd-02": [27, 23], "cmp-02": [20, 29]}
    with_coverage = {q["id"]: [x["value"] for x in c["coverage"]] for q in gt.values()
                     for c in q["expected_calls"] if c["coverage"]}
    assert with_coverage == expected
    for qid, counts in expected.items():
        call = gt[qid]["expected_calls"][0]
        assert [resolve(call["output"], x["path"]) for x in call["coverage"]] == counts
