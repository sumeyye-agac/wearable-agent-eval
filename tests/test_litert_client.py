"""The LiteRT-LM client's message conversion, with hand-made library replies (no model)."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wearable import agent
from wearable_eval import litert_client as lc
from wearable_eval import prompt_v1
from wearable_eval.ground_truth import load_users
from wearable_eval.runs import RecordingClient, run_anthropic, run_litert

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def users():
    return load_users(REPO / "data/demo/demo_daily.csv")


VALID = {"role": "assistant", "tool_calls": [
    {"type": "function", "function": {"name": "most_active_day",
                                      "arguments": {"start_date": "2016-03-12"}}}]}
MALFORMED = {"role": "assistant", "tool_calls": [
    {"type": "function", "function": {"name": "most_active_day",
                                      "arguments": '{"start_date": "2016-03-12"'}}]}
TEXT = {"role": "assistant", "content": [{"type": "text", "text": "26,100 steps."}]}


def test_a_valid_tool_call_becomes_a_tool_use_block():
    blocks, errors = lc.from_litert_reply(VALID, turn=1)
    assert errors == []
    (b,) = blocks
    assert (b.type, b.id, b.name, b.input) == ("tool_use", "toolu_01_00", "most_active_day",
                                                {"start_date": "2016-03-12"})
    assert lc.stop_reason(blocks, decode_tokens=30, max_output_tokens=8192) == "tool_use"
    # Arguments given as a JSON string holding an object are the same call.
    as_string = {"tool_calls": [{"function": {"name": "most_active_day",
                                              "arguments": '{"top_n": 3}'}}]}
    assert lc.from_litert_reply(as_string, 2)[0][0].input == {"top_n": 3}


@pytest.mark.parametrize("call", [
    MALFORMED["tool_calls"][0],
    {"function": {"name": "most_active_day", "arguments": "[1, 2]"}},
    {"function": {"arguments": {}}},
    "most_active_day()",
])
def test_a_malformed_call_is_returned_as_text_with_a_parse_error(call):
    blocks, errors = lc.from_litert_reply({"tool_calls": [call]}, turn=1)
    (b,) = blocks
    assert b.type == "text" and b.text == json.dumps(call)   # as returned, not repaired
    (e,) = errors
    assert e["raw"] == call and e["what"]
    assert lc.stop_reason(blocks, 30, 8192) == "end_turn"


def test_a_plain_text_answer_and_a_cut_off_one():
    blocks, errors = lc.from_litert_reply(TEXT, turn=2)
    assert errors == [] and [(b.type, b.text) for b in blocks] == [("text", "26,100 steps.")]
    assert lc.stop_reason(blocks, 30, 8192) == "end_turn"
    assert lc.stop_reason(blocks, 8192, 8192) == "max_tokens"


def test_a_tool_result_round_trip():
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": [NS(type="tool_use", id="toolu_01_00",
                                                    name="most_active_day", input={}),
                                                 NS(type="tool_use", id="toolu_01_01",
                                                    name="weekly_summary", input={})]}]
    results = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01_00", "is_error": False,
         "content": json.dumps({"most_active": {"steps": 26100}})},
        {"type": "tool_result", "tool_use_id": "toolu_01_01", "is_error": True,
         "content": "Unknown metric 'stress'."}]}
    out = lc.to_litert_message(results, lc.tool_names_by_id(history))
    assert out == {"role": "tool", "content": [
        {"type": "tool_response", "name": "most_active_day",
         "response": {"most_active": {"steps": 26100}}},
        {"type": "tool_response", "name": "weekly_summary",
         "response": "Unknown metric 'stress'."}]}
    assert lc.to_litert_message({"role": "user", "content": "q"}, {}) == "q"


def test_lost_in_rendering_names_what_the_prompt_does_not_show():
    spec = next(s for s in prompt_v1.TOOL_SPECS if s["name"] == "most_active_day")
    rendered = ('declaration:most_active_day{description:<|"|>x<|"|>,parameters:{properties:'
                '{end_date:{description:<|"|>d<|"|>,type:<|"|>STRING<|"|>},start_date:'
                '{description:<|"|>d<|"|>,type:<|"|>STRING<|"|>},top_n:{description:'
                '<|"|>n<|"|>,type:<|"|>INTEGER<|"|>}},type:<|"|>OBJECT<|"|>}')
    lost = lc.lost_in_rendering([spec], rendered)
    assert "most_active_day: strict" in lost
    assert "most_active_day: additionalProperties" in lost
    assert "most_active_day.start_date: format date" in lost
    assert "most_active_day.top_n: enum (integer)" in lost
    assert lc.lost_in_rendering([spec], "") == ["most_active_day: whole declaration"]


# --- the client inside the real agent loop, with a fake engine -------------------------

class FakeConversation:
    def __init__(self, replies):
        self.replies, self.sent, self.closed = list(replies), [], False

    def render_message_to_string(self, message):
        return f"<rendered {message}>"

    def get_benchmark_info(self):
        return NS(time_to_first_token_in_second=1.0, last_prefill_token_count=100,
                  last_prefill_tokens_per_second=50.0, last_decode_token_count=20,
                  last_decode_tokens_per_second=10.0)

    token_count = 120

    def close(self):
        self.closed = True


def fake_client(monkeypatch, replies):
    conversation = FakeConversation(replies)
    client = object.__new__(lc.LiteRTClient)
    client.settings = lc.LiteRTSettings(model_file="m.litertlm")
    client._litert = NS(interfaces=NS(SamplerConfig=dict, ThinkingConfig=dict,
                                      ConstrainedDecodingConfig=dict))
    client._SpecTool = lambda spec: spec
    client._engine = NS(create_conversation=lambda **kw: conversation)
    client._conversation, client._history = None, []
    client.rendered_first_turn = None
    client.messages = NS(create=client._create)

    def send_raw(conv, message, max_output_tokens):
        conv.sent.append(message)
        return conv.replies.pop(0)

    monkeypatch.setattr(lc, "_send_raw", send_raw)
    return client, conversation


def test_ask_runs_unchanged_on_the_client(monkeypatch, users):
    client, conversation = fake_client(monkeypatch, [json.dumps(VALID), json.dumps(TEXT)])
    record = run_litert(client, users["6962181067"], "Which day was I most active?",
                        lc.MODEL_NAME)
    assert record["status"] == "answered" and record["answer"] == "26,100 steps."
    assert [c["name"] for c in record["tool_calls"]] == ["most_active_day"]
    # The second message sent is the tool result, named and parsed back from its JSON.
    sent = conversation.sent[1]
    assert sent["role"] == "tool" and sent["content"][0]["name"] == "most_active_day"
    assert sent["content"][0]["response"]["most_active"]["steps"] == 26100
    first = record["requests"][0]["response"]
    assert first["litert"]["raw_reply"] == json.dumps(VALID)
    assert first["litert"]["prefill_seconds"] == 2.0 and first["litert"]["decode_seconds"] == 2.0
    assert record["cost_usd"] == 0 and record["tokens"]["input_tokens"] == 200


def test_a_malformed_call_reaches_the_record_as_is(monkeypatch, users):
    client, _ = fake_client(monkeypatch, [json.dumps(MALFORMED)])
    record = run_litert(client, users["6962181067"], "Which day was I most active?",
                        lc.MODEL_NAME)
    # Not repaired and not retried: the call is the answer text, and no tool ran.
    assert record["status"] == "answered" and record["tool_calls"] == []
    assert record["answer"] == json.dumps(MALFORMED["tool_calls"][0])
    (error,) = record["requests"][0]["response"]["litert"]["parse_errors"]
    assert error["what"].startswith("arguments are not valid JSON")


# The error litert-lm 0.19.2 raised on hr-02 with constrained decoding off (shortened).
REJECTED_CALL = "<|tool_call>call:correlation{lag_days:0,metric_x:steps,metric_y:mean_hr}<tool_call|>"
LIBRARY_ERROR = (
    "litert_lm_conversation_send_message failed with status INVALID_ARGUMENT (3): "
    "INVALID_ARGUMENT: Failed to parse tool calls from code block: "
    "call:correlation{lag_days:0,metric_x:steps,metric_y:mean_hr}\n"
    f"full response: {REJECTED_CALL}\n"
    "error: Failed to parse FC tool calls: FallThrough(ParseCancelledError(...))")


def test_a_call_the_library_cannot_parse_reaches_the_record_as_text(monkeypatch, users):
    client, _ = fake_client(monkeypatch, [])

    def send_raw(conv, message, max_output_tokens):
        raise RuntimeError(LIBRARY_ERROR)

    monkeypatch.setattr(lc, "_send_raw", send_raw)
    record = run_litert(client, users["6962181067"], "Do I walk more on days with a "
                        "higher heart rate?", lc.MODEL_NAME)
    # Same rule as a malformed call: answered, not repaired, no tool ran.
    assert record["status"] == "answered" and record["tool_calls"] == []
    assert record["answer"] == REJECTED_CALL
    litert = record["requests"][0]["response"]["litert"]
    (error,) = litert["parse_errors"]
    assert error["raw"] == REJECTED_CALL and error["library_error"] == LIBRARY_ERROR
    assert litert["benchmark"]["last_decode_token_count"] is None


def test_other_library_errors_are_not_caught(monkeypatch):
    client, _ = fake_client(monkeypatch, [])

    def send_raw(conv, message, max_output_tokens):
        raise RuntimeError("litert_lm_conversation_send_message failed: out of memory")

    monkeypatch.setattr(lc, "_send_raw", send_raw)
    with pytest.raises(RuntimeError, match="out of memory"):
        client.messages.create(model="m", max_tokens=10, system="s", tools=[],
                               messages=[{"role": "user", "content": "q"}])


def test_rejected_reply_without_a_quoted_reply_keeps_the_whole_error():
    assert lc.rejected_reply("Failed to parse tool calls: x") == "Failed to parse tool calls: x"


def test_a_changed_history_is_refused(monkeypatch):
    client, _ = fake_client(monkeypatch, [json.dumps(VALID)])
    client.messages.create(model="m", max_tokens=10, system="s", tools=[],
                           messages=[{"role": "user", "content": "q"}])
    with pytest.raises(RuntimeError, match="history differs"):
        client.messages.create(model="m", max_tokens=10, system="s", tools=[], messages=[
            {"role": "user", "content": "q"}, {"role": "assistant", "content": []},
            {"role": "user", "content": []}])


# --- Haiku's thinking option -------------------------------------------------------------

class FakeAnthropic:
    def __init__(self):
        self.sent = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return NS(content=[NS(type="text", text="ok")], stop_reason="end_turn",
                  usage=NS(input_tokens=10, output_tokens=2, cache_creation_input_tokens=0,
                           cache_read_input_tokens=0))


def test_thinking_off_sends_disabled_and_on_sends_nothing(users):
    off, on = FakeAnthropic(), FakeAnthropic()
    run_anthropic(off, users["6962181067"], "hi", agent.DEFAULT_MODEL, thinking="off")
    run_anthropic(on, users["6962181067"], "hi", agent.DEFAULT_MODEL)
    assert off.sent[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in on.sent[0]
    assert RecordingClient(on)._extra == {}
