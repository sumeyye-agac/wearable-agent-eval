"""The scorer's validation tools: fake agents and mutations (no model, no network)."""

import json
import random
from datetime import date
from pathlib import Path

import pytest

from wearable_eval.ground_truth import load_users
from wearable_eval.scoring import prompt_days_for, score_answer
from wearable_eval.validation import (empty_record, mutants, null_baseline, oracle_record,
                                      refuse_record)

REPO = Path(__file__).resolve().parents[1]
REF = date(2016, 5, 11)


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


@pytest.fixture(scope="module")
def truths():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}


@pytest.fixture(scope="module")
def prompt(users):
    return prompt_days_for(users)


def test_the_oracle_passes_every_question(users, truths, prompt):
    for t in truths.values():
        s = score_answer(oracle_record(t, users[t["user_id"]]), t, prompt[t["user_id"]])
        assert s["end_to_end"], (t["id"], s)


def test_the_empty_agent_fails_every_question(truths, prompt):
    for t in truths.values():
        assert not score_answer(empty_record(t), t, prompt[t["user_id"]])["end_to_end"]


def test_always_refusing_passes_only_refusals_without_expected_calls(truths, prompt):
    passed = {t["id"] for t in truths.values()
              if score_answer(refuse_record(t), t, prompt[t["user_id"]])["end_to_end"]}
    assert passed == {t["id"] for t in truths.values()
                      if t["expect_refusal"] and not t["expected_calls"]}


def test_mutants_change_one_place_each():
    answer = "On 2016-05-10 you walked 16,315 steps, 7 days in 2016-05-05 to 2016-05-11."
    found = mutants(answer, REF, random.Random(0))
    assert sorted(k for k, _, _ in found) == ["date_pm1", "large_number_10_30pct",
                                              "range_shift_1day", "small_int_pm1"]
    for kind, text, original in found:
        assert text != answer and original in answer
        assert len(text.split()) == len(answer.split())
    (big,) = [t for k, t, _ in found if k == "large_number_10_30pct"]
    value = int(big.split(" steps")[0].rsplit(" ", 1)[1].replace(",", ""))
    assert 0.69 * 16315 <= value <= 1.31 * 16315 and abs(value - 16315) >= 0.09 * 16315


def test_the_null_baseline_skips_pairs_that_share_a_tool_output(truths, prompt):
    out = {"step": 1, "name": "most_active_day", "input": {}, "is_error": False,
           "output": {"most_active": {"steps": 26100}}}
    a = {"question_id": "act-01", "repeat": 1, "status": "answered",
         "answer": "26,100 steps.", "tool_calls": [out]}
    b = {**a, "question_id": "act-07"}
    result = null_baseline([a, b], truths, prompt)
    assert result["skipped_pairs_sharing_an_output"] == 2
    assert result["same_user"]["pairs"] == result["cross_user"]["pairs"] == 0
