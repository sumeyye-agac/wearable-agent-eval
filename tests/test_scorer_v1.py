"""Scorer v1 rules: call validity, sources, fallbacks, low-confidence numbers, answer
values, alternative routes, end-to-end pass and pass^k."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from wearable.registry import run_tool
from wearable_eval import dates, prompt_v1
from wearable_eval.ground_truth import load_users
from wearable_eval.scoring import prompt_days_for, score_answer, score_run
from wearable_eval.tool_calls import classify, schema_problems

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


@pytest.fixture(scope="module")
def truth():
    data = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}


@pytest.fixture(scope="module")
def prompt(users):
    return prompt_days_for(users)


def call(users, uid, name, args):
    return {"step": 1, "name": name, "input": args, "is_error": False,
            "output": run_tool(users[uid], name, args)}


def record(qid, answer, calls, repeat=1, status="answered"):
    return {"question_id": qid, "repeat": repeat, "status": status, "answer": answer,
            "tool_calls": calls}


def scored(truth, prompt, qid, answer, calls):
    q = truth[qid]
    return score_answer(record(qid, answer, calls), q, prompt[q["user_id"]])


# --- tool-call validity ----------------------------------------------------------

@pytest.mark.parametrize("name, args, label", [
    ("weekly_summary", {"metric": "steps"}, "valid"),
    ("weekly_summary", '{"metric": "steps"', "malformed_args"),
    ("weekly_summary", None, "malformed_args"),
    ("weekly_summary", {}, "schema_invalid"),                       # missing metric
    ("weekly_summary", {"metric": "stress"}, "schema_invalid"),     # not in the enum
    ("weekly_summary", {"metric": "steps", "end_date": "11 May"}, "schema_invalid"),
    ("weekly_summary", {"metric": "steps", "days": 7}, "schema_invalid"),  # unknown argument
    ("most_active_day", {"top_n": True}, "schema_invalid"),         # bool is not an integer
    ("get_sleep", {}, "schema_invalid"),                            # no such tool
    ("get_guideline", {"topic": "sleep_duration"}, "off_route"),   # valid, not on a route
])
def test_call_validity_labels(name, args, label):
    assert classify({"name": name, "input": args}, {"weekly_summary", "most_active_day"}) \
        == label


def test_without_expected_calls_no_tool_is_wrong():
    assert classify({"name": "get_guideline", "input": {"topic": "sleep_duration"}}, set()) \
        == "valid"
    assert schema_problems("correlation", {"metric_x": "steps", "metric_y": "mean_hr"}) == []


def test_no_call_is_flagged_only_when_calls_are_expected(truth, prompt):
    assert scored(truth, prompt, "act-01", "I don't know.", [])["no_call"] is True
    assert scored(truth, prompt, "una-01", "Resting heart rate is not in the data.",
                  [])["no_call"] is None


# --- the "Weekday, 23,071" parse fix and prompt dates ------------------------------

def test_a_weekday_before_a_large_number_is_not_a_date():
    text = "2016-03-18 (Friday, 23,071 steps) and 2016-03-19 (Saturday, 20,424 steps)"
    found = dates.find_dates(text, date(2016, 5, 11))
    assert [m.text for m in found] == ["2016-03-18", "2016-03-19"]
    # A weekday with a real day still parses.
    (m,) = dates.find_dates("on Friday, 18 and", date(2016, 3, 1))
    assert m.items[0].day == date(2016, 3, 18)


def test_prompt_days_are_every_date_the_system_prompt_states(users, prompt):
    for uid, df in users.items():
        stated = {date.fromisoformat(d)
                  for d in re.findall(r"\d{4}-\d{2}-\d{2}", prompt_v1.system_prompt(df))}
        assert stated <= prompt[uid]
        # Besides those: the days inside "this week" and "last week" (14 days in all).
        assert len(prompt[uid] - stated) <= 14 - 4


def test_the_data_range_from_the_prompt_is_a_supported_date(truth, prompt):
    out = scored(truth, prompt, "una-02",
                 "There is no data for June 2016; it covers 2016-03-12 to 2016-05-12.", [])
    assert out["unsupported_dates"] == []
    assert {"text": "2016-03-12", "source": "prompt"} in out["date_sources"]


# --- sources, fallbacks and low-confidence numbers -----------------------------------

def test_every_supported_number_carries_its_source(users, truth, prompt):
    out = scored(truth, prompt, "act-01",
                 "Thursday, 2016-03-17: 26,100 steps, 3,029 more than the next day, "
                 "over 61 days with data.",
                 [call(users, "6962181067", "most_active_day", {})])
    sources = {t["text"]: t["source"] for t in out["number_sources"]}
    assert sources["26,100"] == "most_active_day:most_active.steps"
    assert sources["3,029"] == "most_active_day:most_active.steps_above_second"
    assert sources["61"] == "most_active_day:days_with_data"
    assert {"text": "Thursday, 2016-03-17", "source": "tool:most_active_day"} \
        in out["date_sources"]


SLEEP_WEEK = ("5553957443", "weekly_summary", {"metric": "minutes_asleep",
                                              "end_date": "2016-04-01"})


def test_minutes_text_within_one_minute_and_decimal_hours_from_the_tool(users, truth, prompt):
    # The week's mean is 384.9 min: "6 h 25 min" in the tool, 6.42 h in mean_hours.
    c = call(users, *SLEEP_WEEK)
    out = scored(truth, prompt, "slp-07", "Mean 6 h 24 min, or 6.42 hours.", [c])
    assert out["unsupported_numbers"] == [] and out["model_computed"] == []
    assert {t["source"] for t in out["number_sources"]} == \
        {"weekly_summary:mean_text(text)", "weekly_summary:mean_hours"}


def test_a_conversion_the_tool_did_not_give_is_supported_but_counted(users, truth, prompt):
    # Per-day rows have no decimal hours: 412 min (2016-03-31) as "6.87 hours" is the
    # model's own conversion. (The summary values do have one: 6.08 is median_hours.)
    c = call(users, *SLEEP_WEEK)
    out = scored(truth, prompt, "slp-07", "That night you slept 6.87 hours.", [c])
    assert out["unsupported_numbers"] == [] and out["model_computed"] == ["6.87 hours"]
    assert out["number_sources"][0]["source"] == "fallback:weekly_summary:daily.5.value"
    wrong = scored(truth, prompt, "slp-07", "That night you slept 6.2 hours.", [c])
    assert wrong["unsupported_numbers"] == ["6.2 hours"]


def test_night_of_is_a_tool_date_and_the_evening_before_otherwise_is_counted(users, truth,
                                                                           prompt):
    week = call(users, *SLEEP_WEEK)
    out = scored(truth, prompt, "slp-07", "The night of 2016-03-31: 6 h 05 min.", [week])
    assert out["unsupported_dates"] == [] and out["model_computed"] == []
    assert {"text": "2016-03-31", "source": "tool:weekly_summary"} in out["date_sources"]
    # compare_periods gives no night_of: the evening before its first day (2016-03-25)
    # is the model's own date, accepted and counted.
    periods = {"period_a_start": "2016-03-26", "period_a_end": "2016-04-01",
               "period_b_start": "2016-04-02", "period_b_end": "2016-04-08"}
    sleep = call(users, "5553957443", "compare_periods", {"metric": "minutes_asleep",
                                                          **periods})
    out = scored(truth, prompt, "slp-07", "Nights from 2016-03-25 on.", [sleep])
    assert out["unsupported_dates"] == [] and out["model_computed"] == ["2016-03-25"]
    assert {"text": "2016-03-25", "source": dates.FALLBACK} in out["date_sources"]
    # The evening before is a fallback for sleep results only.
    steps = call(users, "5553957443", "compare_periods", {"metric": "steps", **periods})
    assert scored(truth, prompt, "slp-07", "From 2016-03-25 on.", [steps])["unsupported_dates"]


def test_small_unsupported_integers_are_low_confidence_not_headline(users, truth, prompt):
    c = call(users, "6962181067", "most_active_day", {})
    out = scored(truth, prompt, "act-01",
                 "Thursday, 2016-03-17: 26,100 steps, 3,029 more than the next day; "
                 "about 5 more than usual, and 14.5 km.", [c])
    assert out["low_confidence_numbers"] == ["5"]
    assert out["unsupported_numbers"] == ["14.5"]
    assert out["end_to_end"] is False
    out = scored(truth, prompt, "act-01",
                 "Thursday, 2016-03-17: 26,100 steps, 3,029 more than the next day; "
                 "about 5 more than usual.", [c])
    assert out["low_confidence_numbers"] == ["5"] and out["end_to_end"] is True


def test_guideline_numbers_need_their_unit_and_the_guideline_call(users, truth, prompt):
    g = call(users, "5553957443", "get_guideline", {"topic": "sleep_duration"})
    out = scored(truth, prompt, "slp-07",
                 "Adults 18-60 years are advised 7 or more hours; 65 bpm is unrelated.", [g])
    assert out["unsupported_numbers"] == ["65"]
    assert {t["text"]: t["source"] for t in out["number_sources"]}["18"] == \
        "guideline:sleep_duration"
    without = scored(truth, prompt, "slp-07", "Adults 18-60 years are advised 7 hours.", [])
    assert without["unsupported_numbers"] == ["18", "60"]
    assert without["low_confidence_numbers"] == ["7"]


# --- answer values, routes, end to end ------------------------------------------------

def test_answer_values_in_minute_forms_and_count_words(users, truth, prompt):
    week = call(users, "6962181067", "weekly_summary", {"metric": "minutes_asleep",
                                                        "threshold": 420})
    out = scored(truth, prompt, "slp-02",
                 "Four of the seven nights (2016-05-05 to 2016-05-11) reached 7 hours; "
                 "all 7 nights have data.", [week])
    assert out["answer_correct"] is True
    trd = truth["trd-02"]["expected_calls"][0]
    c = call(users, truth["trd-02"]["user_id"], trd["tool"], trd["args"])
    out = scored(truth, prompt, "trd-02", "Yes, by 14 minutes per night, over 27 and 23 "
                 "nights.", [c])
    assert out["answer_correct"] is True   # 13.8 min, given by the tool as "14 min"
    out = scored(truth, prompt, "trd-02", "Yes, by 16 minutes per night, over 27 and 23 "
                 "nights.", [c])
    assert out["answer_correct"] is False


def test_an_answer_date_may_name_the_evening_before(users, truth, prompt):
    q = truth["slp-04"]
    (e,) = q["expected_calls"]
    c = call(users, q["user_id"], e["tool"], e["args"])
    value = e["answer"][0]["value"]
    for named in e["answer_dates"][0]["accepted"]:
        answer = f"Your shortest night was {named}: {value:g} minutes, over 2 nights."
        assert scored(truth, prompt, "slp-04", answer, [c])["answer_correct"] is True


def test_an_alternative_route_counts_as_a_correct_tool_result(users, truth, prompt):
    weeks = [call(users, "6962181067", "weekly_summary", {"metric": "steps",
                                                          "end_date": d})
             for d in ("2016-04-10", "2016-04-17")]
    out = scored(truth, prompt, "cmp-01", "No: 58,689 steps in the week of 2016-04-11 to "
                 "2016-04-17 against 93,860 in 2016-04-04 to 2016-04-10.", weeks)
    assert out["tool_result_correct"] is True and out["answer_correct"] is True
    assert out["end_to_end"] is True
    only_one = scored(truth, prompt, "cmp-01", "58,689 steps.", weeks[1:])
    assert only_one["tool_result_correct"] is False and only_one["end_to_end"] is False


def test_end_to_end_and_pass_all_repeats(users, truth, prompt):
    c = call(users, "6962181067", "most_active_day", {})
    good = "Thursday, 2016-03-17: 26,100 steps, 3,029 more than the next day."
    lines = [{**record("act-01", a, [c], repeat=k), "steps": 2, "seconds": 1.0,
              "cost_usd": 0.0, "tokens": {}}
             for k, a in [(1, good), (2, good), (3, "Your top day had 26,100 steps.")]]
    config = {"model": "m", "questions_version": "v1.2", "question_ids": ["act-01"]}
    scores = score_run(config, lines, {"questions": list(truth.values())}, prompt)
    entry = scores["per_question"]["act-01"]
    assert entry["end_to_end"] == [True, True, False]   # r3 does not name the day
    assert entry["pass_rate"] == {"k": 2, "n": 3, "rate": 0.6667}
    assert entry["pass_all_repeats"] is False
    assert scores["pass_all_repeats"] == {"k": 0, "n": 1, "rate": 0.0, "repeats": 3}
    assert scores["summary"]["end_to_end"]["all"] == {"k": 2, "n": 3, "rate": 0.6667}
    assert scores["call_validity"]["valid"] == {"k": 3, "n": 3, "rate": 1.0}
    assert scores["scorer_version"] == "v1.1"


def test_a_refusal_with_an_invented_number_fails_end_to_end(truth, prompt):
    ok = scored(truth, prompt, "una-01", "Resting heart rate is not in the data.", [])
    assert ok["refusal_correct"] is True and ok["end_to_end"] is True
    bad = scored(truth, prompt, "una-01", "Your resting heart rate was 58 bpm.", [])
    assert bad["refusal_correct"] is False and bad["end_to_end"] is False
