"""The runner and the scorer, driven by a scripted stand-in for the API (no network)."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wearable_eval.ground_truth import load_users
from wearable_eval.runs import append_line, read_run, run_anthropic, token_totals, write_json
from wearable_eval.scoring import (REFUSAL, answer_numbers, is_supported, score_answer,
                                   score_run, supported_values)

REPO = Path(__file__).resolve().parents[1]


def usage(i=1000, o=100, thinking=40):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=0,
              cache_read_input_tokens=0, output_tokens_details={"thinking_tokens": thinking})


def reply(*content, stop_reason="tool_use"):
    return NS(content=list(content), stop_reason=stop_reason, usage=usage())


def tool_use(name, arguments):
    return NS(type="tool_use", id="t1", name=name, input=arguments)


def final(text):
    return reply(NS(type="thinking", thinking=""), NS(type="text", text=text),
                 stop_reason="end_turn")


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


@pytest.fixture(scope="module")
def truth():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}


def act01_record(users, answer, args=None):
    client = FakeClient(reply(tool_use("most_active_day", args or {})), final(answer))
    record = run_anthropic(client, users["6962181067"], "Which day was I most active?",
                           "claude-haiku-5-5")
    return {"question_id": "act-01", "repeat": 1, **record}


def test_runner_records_steps_tokens_and_tool_results(users):
    record = act01_record(users, "March 17, 2016: 26,100 steps.")
    assert record["status"] == "answered" and record["steps"] == 2
    assert [r["stop_reason"] for r in record["requests"]] == ["tool_use", "end_turn"]
    assert record["tokens"]["thinking_tokens"] == 80 and record["tokens"]["input_tokens"] == 2000
    assert record["tool_calls"][0]["output"]["most_active"]["steps"] == 26100
    assert record["cost_usd"] > 0
    json.dumps(record)  # every field can be written to transcripts.jsonl


def test_runner_sends_its_own_max_tokens(users):
    client = FakeClient(final("ok"))
    record = run_anthropic(client, users["6962181067"], "hi", "claude-haiku-5-5")
    assert client.sent[0]["max_tokens"] == 8192 == record["requests"][0]["max_tokens"]
    assert "temperature" not in client.sent[0]


def test_tool_result_correct_matches_results_not_arguments(users, truth):
    good = act01_record(users, "x", {"start_date": "2016-03-12", "end_date": "2016-05-11"})
    assert score_answer(good, truth["act-01"])["tool_result_correct"] is True
    narrow = act01_record(users, "x", {"start_date": "2016-04-01"})
    assert score_answer(narrow, truth["act-01"])["tool_result_correct"] is False


def test_unsupported_numbers_skip_dates_and_accept_tool_values(users, truth):
    record = act01_record(users, "Your most active day was Thursday, March 17, 2016 "
                                 "(2016-03-17), with 26,100 steps, 3,029 more than the next "
                                 "day. That is about 7.5 miles.")
    scored = score_answer(record, truth["act-01"])
    assert scored["unsupported"] == ["7.5"]
    assert scored["numbers"] == 3


def values(text):
    return [n.value for n in answer_numbers(text)]


def test_answer_numbers():
    assert values("On 11 May 2016 and May 5th you slept 7 h 31 min, 1,234.5 in "
                  "total; SpO2 and E2B are not numbers.") == [7, 31, 1234.5]
    assert values("1. First\n2. Second: 12%") == [12]
    assert answer_numbers(None) == []


def test_signs_are_kept_but_ranges_are_not_minus_signs():
    assert [(n.value, n.signed) for n in answer_numbers("-5% (+3.2) and 7-9 hours, −4")] == \
        [(-5, True), (3.2, True), (7, False), (9, False), (-4, True)]


@pytest.mark.parametrize("text, ok", [
    ("-12.3%", True), ("12.3% lower", True), ("+12.3%", False), ("-12.3 vs +4", True),
])
def test_an_explicit_sign_must_match_the_tool(text, ok):
    supported = {-12.3, 4}
    assert all(is_supported(n, supported) for n in answer_numbers(text)) == ok


def test_text_fields_and_the_question_support_numbers():
    calls = [{"output": {"mean": 451.3, "mean_text": "7 h 31 min"}, "is_error": False},
             {"output": "end_date 2030-01-01 is after the data; use 1 to 10", "is_error": True}]
    values = supported_values(calls, "Did I reach 10,000 steps?")
    assert {451.3, 7, 31, 10000} <= values
    assert 10 not in values  # numbers in an error message support nothing


@pytest.mark.parametrize("text, refuses", [
    ("Resting heart rate is not in the data.", True),
    ("The data doesn't include resting heart rate.", True),
    ("There is no sleep data for June 2016.", True),
    ("The week is incomplete, so this cannot be settled.", True),
    ("I can't tell from these days.", True),
    ("Your average was 64 bpm.", False),
])
def test_refusal_phrases(text, refuses):
    assert (REFUSAL.search(text) is not None) == refuses


def test_score_run_reports_spread_and_consistency(users, truth, tmp_path):
    lines = []
    for repeat, answer in [(1, "26,100 steps on March 17."), (2, "About 26,000 steps."),
                           (3, "26,100 steps, roughly 7.5 miles.")]:
        lines.append({**act01_record(users, answer), "repeat": repeat})
    config = {"model": "claude-haiku-5-5", "questions_version": "v1",
              "question_ids": ["act-01"]}
    write_json(tmp_path / "config.json", config)
    for line in lines:
        append_line(tmp_path / "transcripts.jsonl", line)
    config, lines = read_run(tmp_path)
    scores = score_run(config, lines, {"questions": list(truth.values())})
    assert scores["summary"]["tool_result_correct"]["per_repeat"] == [1.0, 1.0, 1.0]
    assert scores["summary"]["answers_with_unsupported"]["per_repeat"] == [0.0, 0.0, 1.0]
    assert scores["per_question"]["act-01"]["consistent"] is False
    assert scores["usage"]["thinking_tokens"] == 3 * 80


def test_token_totals_handle_a_missing_thinking_count():
    requests = [{"usage": {"input_tokens": 5, "output_tokens": 2}}]
    assert token_totals(requests)["thinking_tokens"] == 0
