"""Scorer v1.1: six wrong answers that scorer v1 passed end to end (audit 2026-11-25), each
next to a correct answer that must still pass. The tool calls are each question's expected
calls, so only the answer text differs."""

import json
from pathlib import Path

import pytest

from wearable_eval.ground_truth import load_users
from wearable_eval.scoring import prompt_days_for, score_answer
from wearable_eval.validation import oracle_record

REPO = Path(__file__).resolve().parents[1]

CASES = [
    # (question, wrong answer, what is wrong, correct answer)
    ("cmp-01", "Yes, you walked more: 5,025 more steps per day in the week ending 17 April.",
     "verdict and direction",
     "No, you walked less: 5,025 fewer steps per day in the week ending 17 April."),
    ("trd-01", "Your steps went up by 4,228 per day in the second half.",
     "direction",
     "Yes, your steps went down by 4,228 per day in the second half."),
    ("trd-03", "No, sleeping less doesn't reduce walking; r = 0.517, so you walk more after "
               "short nights.",
     "sign of r (it is -0.517)",
     "No, sleeping less doesn't reduce walking; r = -0.517, so you walk more after short "
     "nights."),
    ("act-03", "No, you were not active enough: only 592 moderate-equivalent minutes on 5 days.",
     "verdict",
     "Yes, you were active enough: 592 moderate-equivalent minutes on 5 days."),
    ("slp-02", "You slept at least 7 hours on 3 nights; 4 nights fell short.",
     "the count next to its noun",
     "You slept at least 7 hours on 4 nights; 3 nights fell short."),
    ("una-03", "There is no missing data: you took 0 steps on 7 May 2016.",
     "a number in a refusal",
     "There is no step data for 7 May 2016."),
]


@pytest.fixture(scope="module")
def setup():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    users = load_users(REPO / "data/demo/demo_daily.csv")
    return {q["id"]: q for q in data["questions"]}, users, prompt_days_for(users)


def scored(setup, qid, answer):
    truths, users, prompt = setup
    t = truths[qid]
    record = {**oracle_record(t, users[t["user_id"]]), "answer": answer}
    return score_answer(record, t, prompt[t["user_id"]])


@pytest.mark.parametrize("qid, wrong, what, right", CASES, ids=[c[0] for c in CASES])
def test_the_wrong_answer_fails_end_to_end(setup, qid, wrong, what, right):
    assert scored(setup, qid, wrong)["end_to_end"] is False, what


@pytest.mark.parametrize("qid, wrong, what, right", CASES, ids=[c[0] for c in CASES])
def test_the_correct_answer_still_passes(setup, qid, wrong, what, right):
    assert scored(setup, qid, right)["end_to_end"] is True


def test_the_failures_come_from_the_new_checks(setup):
    """Which check catches each wrong answer (scorer v1.1 fields)."""
    assert scored(setup, "cmp-01", CASES[0][1])["verdict_correct"] is False
    assert scored(setup, "trd-01", CASES[1][1])["verdict_correct"] is False
    assert scored(setup, "trd-03", CASES[2][1])["answer_correct"] is False
    assert scored(setup, "act-03", CASES[3][1])["verdict_correct"] is False
    assert scored(setup, "slp-02", CASES[4][1])["answer_correct"] is False
    assert scored(setup, "una-03", CASES[5][1])["refusal_correct"] is False
