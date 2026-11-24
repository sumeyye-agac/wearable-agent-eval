"""Prompt v1: the frozen system prompt and tool schemas for every model in the eval."""

import copy
from pathlib import Path

import pytest

from wearable.registry import TOOL_SPECS
from wearable_eval import prompt_v1
from wearable_eval.ground_truth import load_users

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


def _without_descriptions(specs):
    specs = copy.deepcopy(specs)
    for spec in specs:
        spec.pop("description")
        for prop in spec["input_schema"]["properties"].values():
            prop.pop("description", None)
    return specs


def test_only_descriptions_differ_from_the_registry():
    # Names, arguments, types, enums and required lists stay those of the tools, so
    # run_tool runs every call prompt v1 allows.
    assert _without_descriptions(prompt_v1.TOOL_SPECS) == _without_descriptions(TOOL_SPECS)


def test_tool_specs_are_shorter():
    import json
    assert len(json.dumps(prompt_v1.TOOL_SPECS)) < len(json.dumps(TOOL_SPECS))


def test_date_arguments_say_the_format_in_text():
    # Gemma's chat template drops "format": "date" and integer enums, so the description
    # carries the date format and the allowed range.
    for spec in prompt_v1.TOOL_SPECS:
        for name, prop in spec["input_schema"]["properties"].items():
            if prop.get("format") == "date":
                assert "YYYY-MM-DD" in prop["description"], (spec["name"], name)
            if prop.get("type") == "integer":
                assert " to " in prop["description"], (spec["name"], name)


def test_this_week_and_last_week_are_given_as_dates(users):
    prompt = " ".join(prompt_v1.system_prompt(users["6962181067"]).split())
    assert "anchored on the last complete day, 2016-05-11 (Wednesday)" in prompt
    assert '"This week" is 2016-05-05 to 2016-05-11; "last week" is 2016-04-28 to 2016-05-04.' \
        in prompt
    # 6962181067 has a partial day after the anchor; the prompt says it is outside this week
    assert "Data after 2016-05-11 is from a day that was not recorded in full" in prompt


@pytest.mark.parametrize("rule", [
    "If a question names no time window, use all the data",
    "without end_date they cover this week",
    "Every number you state must come from a tool result in this conversation or from the question",
    "You may write minutes as hours, including decimal hours",
    "Every date you state must come from a tool result, the question or this prompt",
    "When you name a weekday, copy it from that field",
    "Say which period your answer covers",
    "for a comparison, for each period",
    "When the data cannot answer a question, say so plainly",
    "it never means zero",
])
def test_the_conventions_are_stated(users, rule):
    assert rule in " ".join(prompt_v1.system_prompt(users["5553957443"]).split())


def test_no_recommendation_numbers_in_the_prompt(users):
    # Recommendations come only from get_guideline; the old metric list said "7-9 h".
    assert "7-9" not in prompt_v1.system_prompt(users["5553957443"])
