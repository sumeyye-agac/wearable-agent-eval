"""A LiteRT-LM client shaped like the Anthropic client, so wearable.agent.ask() runs Gemma
through the same loop as Claude: same step cap, same error handling, same tool-result
format.

It implements only what ask() and runs.RecordingClient use:
- client.messages.create(model=, max_tokens=, system=, tools=, messages=, output_config=)
- response.content: blocks with .type "text" (.text) or "tool_use" (.id, .name, .input)
- response.stop_reason: "tool_use", "end_turn" or "max_tokens"
- response.usage: .input_tokens, .output_tokens (cache fields are 0)
Everything else on the response (the library's raw reply, parse errors, timings) is extra
and is saved by RecordingClient with the request.

Conversion rules, in both directions, are format conversion only. A tool call whose
arguments are not a JSON object is not repaired: it comes back as a text block holding the
call as the library returned it, with a parse_error. The library's own Message.from_json is
not used, because it turns arguments that do not parse into {} without saying so.

LiteRT-LM keeps the conversation in its KV cache, while the Anthropic API is stateless and
gets the whole history every time. So the client sends only the new message of each
request, after checking that the rest of the history is what it already sent.

Built against litert-lm-api 0.19.2 (the Python docs do not cover tool calls without
automatic execution, constrained decoding or the raw reply; the behaviour used here is read
from the package source). Another version is refused.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODEL_NAME = "gemma-4-E2B-it"
LITERT_LM_VERSION = "0.19.2"


@dataclass(frozen=True)
class LiteRTSettings:
    """The fixed generation settings of a Gemma run. All of them go into the run config."""
    model_file: str
    constrained_decoding: bool = True   # off only for the ablation
    temperature: float = 0.0
    top_k: int = 1
    thinking: bool = False
    max_num_tokens: int = 8192          # context; the default 4096 is below our prompt
    backend: str = "cpu"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- tool schemas ---------------------------------------------------------------------

def to_litert_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """One tool spec as given to Claude, in the OpenAPI shape LiteRT-LM takes. Nothing is
    dropped here; what the chat template leaves out is measured by lost_in_rendering."""
    function = {"name": spec["name"], "description": spec["description"],
                "parameters": spec["input_schema"]}
    if "strict" in spec:
        function["strict"] = spec["strict"]
    return {"type": "function", "function": function}


def _declaration(rendered: str, name: str) -> str:
    start = rendered.find(f"declaration:{name}{{")
    if start < 0:
        return ""
    end = rendered.find("declaration:", start + 1)
    return rendered[start:end if end > 0 else len(rendered)]


def lost_in_rendering(specs: list[dict[str, Any]], rendered: str) -> list[str]:
    """Schema keys of the tool specs that do not reach the prompt the model reads, found by
    looking for each one in the rendered first turn. Description and type are expected to
    be there and are checked too."""
    lost = []
    for spec in specs:
        name = spec["name"]
        block = _declaration(rendered, name)
        if not block:
            lost.append(f"{name}: whole declaration")
            continue
        if "strict" in spec and "strict" not in block:
            lost.append(f"{name}: strict")
        schema = spec["input_schema"]
        if "additionalProperties" in schema and "additionalProperties" not in block:
            lost.append(f"{name}: additionalProperties")
        for prop, value in schema["properties"].items():
            if f"{prop}:{{" not in block:
                lost.append(f"{name}.{prop}: whole property")
                continue
            if "format" in value and "format:" not in block:
                lost.append(f"{name}.{prop}: format {value['format']}")
            if "enum" in value and not all(str(v) in block for v in value["enum"]):
                lost.append(f"{name}.{prop}: enum ({value['type']})")
    return lost


# --- messages in -----------------------------------------------------------------------

def tool_names_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    """tool_use id -> tool name, from the assistant turns in the history."""
    names = {}
    for m in messages:
        if m["role"] == "assistant" and not isinstance(m["content"], str):
            for block in m["content"]:
                if _get(block, "type") == "tool_use":
                    names[_get(block, "id")] = _get(block, "name")
    return names


def _get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key)


def to_litert_message(message: dict[str, Any], names: dict[str, str]) -> Any:
    """One Anthropic user message as LiteRT-LM input: the question as text, or tool
    results as a "tool" message. A result's content is the JSON string ask() made from the
    tool output (parsed back to the same object) or the error message."""
    if message["role"] != "user":
        raise ValueError(f"expected a user message, got {message['role']!r}")
    content = message["content"]
    if isinstance(content, str):
        return content
    results = []
    for block in content:
        if _get(block, "type") != "tool_result":
            raise ValueError(f"unexpected block in a user message: {_get(block, 'type')!r}")
        raw = _get(block, "content")
        response = raw if _get(block, "is_error") else json.loads(raw)
        results.append({"type": "tool_response", "name": names[_get(block, "tool_use_id")],
                        "response": response})
    return {"role": "tool", "content": results}


# --- response out ----------------------------------------------------------------------

def from_litert_reply(reply: dict[str, Any], turn: int) -> tuple[list[SimpleNamespace],
                                                                   list[dict[str, Any]]]:
    """The library's reply (the JSON it returns, unparsed by Message.from_json) as content
    blocks, and the parse errors. Text items become text blocks. A tool call becomes a
    tool_use block if its name is a string and its arguments a JSON object (or a string
    holding exactly one); otherwise the call is returned as a text block with the call as
    the library gave it, and a parse error. Thinking ("channels") is not part of the
    content; it is kept in the raw reply."""
    blocks: list[SimpleNamespace] = []
    errors: list[dict[str, Any]] = []
    content = reply.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text":
            if item.get("text"):
                blocks.append(SimpleNamespace(type="text", text=item["text"]))
        else:
            errors.append({"what": "content item that is not text", "raw": item})
            blocks.append(SimpleNamespace(type="text", text=json.dumps(item)))
    for i, call in enumerate(reply.get("tool_calls") or []):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        args = function.get("arguments") if isinstance(function, dict) else None
        problem = None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as err:
                problem = f"arguments are not valid JSON: {err}"
        if problem is None and not isinstance(name, str):
            problem = "tool call without a name"
        elif problem is None and not isinstance(args, dict):
            problem = f"arguments are {type(args).__name__}, not an object"
        if problem:
            errors.append({"what": problem, "raw": call})
            blocks.append(SimpleNamespace(type="text", text=json.dumps(call)))
        else:
            blocks.append(SimpleNamespace(type="tool_use", id=f"toolu_{turn:02d}_{i:02d}",
                                          name=name, input=args))
    return blocks, errors


def stop_reason(blocks: list[SimpleNamespace], decode_tokens: int,
                max_output_tokens: int) -> str:
    if any(b.type == "tool_use" for b in blocks):
        return "tool_use"
    if decode_tokens >= max_output_tokens:
        return "max_tokens"
    return "end_turn"


# --- the client --------------------------------------------------------------------------

def _spec_tool_class() -> type:
    """The adapter handed to LiteRT-LM for one tool spec. Never executed: automatic tool
    calling is off. Built lazily because it subclasses litert_lm.interfaces.Tool."""
    from litert_lm import interfaces

    class SpecTool(interfaces.Tool):
        def __init__(self, spec: dict[str, Any]):
            self.spec = spec

        def get_tool_description(self) -> dict[str, Any]:
            return to_litert_tool(self.spec)

        def execute(self, param: Any) -> Any:  # pragma: no cover
            raise RuntimeError("tools are run by the agent loop, not by LiteRT-LM")

    return SpecTool


def _send_raw(conversation: Any, message: Any, max_output_tokens: int) -> str:
    """send_message() of litert-lm 0.19.2 without automatic tool calling, stopped before
    Message.from_json: returns the reply JSON exactly as the library produced it."""
    from litert_lm._ffi import create_checked, get_checked
    from litert_lm._messages import normalize_message
    import ctypes

    current = normalize_message(message)
    args_ptr = conversation._create_optional_args(max_output_tokens=max_output_tokens,
                                                  current_message=current)
    try:
        resp_ptr = create_checked(conversation._lib, "litert_lm_conversation_send_message",
                                  conversation._ptr, json.dumps(current),
                                  json.dumps(getattr(conversation, "extra_context", {})),
                                  args_ptr)
        try:
            raw = get_checked(conversation._lib, "litert_lm_json_response_get_string",
                              ctypes.c_char_p, resp_ptr)
        finally:
            conversation._lib.litert_lm_json_response_delete(resp_ptr)
    finally:
        conversation._delete_optional_args(args_ptr)
    return raw.decode("utf-8") if raw else ""


class LiteRTClient:
    """client.messages.create(...) on a local Gemma model. One engine for the whole run; a
    new conversation for each question (each request whose history is one message)."""

    def __init__(self, settings: LiteRTSettings):
        import importlib.metadata

        import litert_lm

        version = importlib.metadata.version("litert-lm-api")
        if version != LITERT_LM_VERSION:
            raise RuntimeError(f"litert-lm-api {version} is installed; this client is "
                               f"written against {LITERT_LM_VERSION}")
        if settings.backend != "cpu":
            raise ValueError("only the CPU backend is set up")
        self.settings = settings
        self._litert = litert_lm
        self._SpecTool = _spec_tool_class()
        start = time.perf_counter()
        self._engine = litert_lm.Engine(settings.model_file, backend=litert_lm.Backend.CPU(),
                                        max_num_tokens=settings.max_num_tokens,
                                        enable_benchmark=True)
        self.engine_load_seconds = round(time.perf_counter() - start, 3)
        self._conversation: Any = None
        self._history: list[dict[str, Any]] = []
        self.rendered_first_turn: str | None = None
        self.messages = SimpleNamespace(create=self._create)

    def close(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
        self._engine.close()

    def _new_conversation(self, system: str, tools: list[dict[str, Any]]) -> Any:
        lm, s = self._litert, self.settings
        kwargs: dict[str, Any] = dict(
            tools=[self._SpecTool(spec) for spec in tools], automatic_tool_calling=False,
            system_message=system,
            sampler_config=lm.interfaces.SamplerConfig(temperature=s.temperature,
                                                       top_k=s.top_k),
            thinking_config=lm.interfaces.ThinkingConfig(enable_thinking=s.thinking))
        if s.constrained_decoding:
            kwargs["constrained_decoding_config"] = \
                lm.interfaces.ConstrainedDecodingConfig(enable=True)
        return self._engine.create_conversation(**kwargs)

    def _create(self, *, model: str, max_tokens: int, system: str,
                tools: list[dict[str, Any]], messages: list[dict[str, Any]],
                **ignored: Any) -> SimpleNamespace:
        """ignored: Anthropic-only fields (output_config with the effort level)."""
        if len(messages) == 1:
            if self._conversation is not None:
                self._conversation.close()
            self._conversation = self._new_conversation(system, tools)
            self._history = []
            self.rendered_first_turn = self._conversation.render_message_to_string(
                to_litert_message(messages[0], {}))
        elif messages[:-1] != self._history:
            raise RuntimeError("the history differs from what this conversation was sent; "
                               "LiteRT-LM keeps its own state, so it cannot be replayed")
        message = to_litert_message(messages[-1], tool_names_by_id(messages))
        turn = len(self._history) // 2 + 1
        start = time.perf_counter()
        raw = _send_raw(self._conversation, message, max_tokens)
        seconds = round(time.perf_counter() - start, 3)
        b = self._conversation.get_benchmark_info()
        bench = {k: getattr(b, k) for k in (
            "time_to_first_token_in_second", "last_prefill_token_count",
            "last_prefill_tokens_per_second", "last_decode_token_count",
            "last_decode_tokens_per_second")}
        try:
            reply = json.loads(raw) if raw else {}
            if not isinstance(reply, dict):
                raise ValueError(f"reply is {type(reply).__name__}, not an object")
            blocks, errors = from_litert_reply(reply, turn)
        except ValueError as err:  # the library's own JSON did not parse: keep it as text
            blocks = [SimpleNamespace(type="text", text=raw)]
            errors = [{"what": f"reply is not a JSON object: {err}", "raw": raw}]
        decode = bench["last_decode_token_count"] or 0
        response = SimpleNamespace(
            content=blocks,
            stop_reason=stop_reason(blocks, decode, max_tokens),
            usage=SimpleNamespace(input_tokens=bench["last_prefill_token_count"] or 0,
                                  output_tokens=decode, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0),
            litert={"raw_reply": raw, "parse_errors": errors, "wall_seconds": seconds,
                    "benchmark": bench, "token_count": self._conversation.token_count,
                    "prefill_seconds": _seconds(bench["last_prefill_token_count"],
                                                bench["last_prefill_tokens_per_second"]),
                    "decode_seconds": _seconds(bench["last_decode_token_count"],
                                               bench["last_decode_tokens_per_second"])})
        # ask() appends this content as the assistant turn; the next request must match.
        self._history = [*messages, {"role": "assistant", "content": blocks}]
        return response

    def render_first_turn(self, system: str, tools: list[dict[str, Any]],
                          question: str) -> str:
        """The first turn as the model reads it (system prompt, tool declarations and the
        question), from a throwaway conversation."""
        conversation = self._new_conversation(system, tools)
        try:
            return conversation.render_message_to_string(question)
        finally:
            conversation.close()

    def run_config(self) -> dict[str, Any]:
        return {**asdict(self.settings), "litert_lm_api": LITERT_LM_VERSION,
                "model_name": MODEL_NAME}


def _seconds(tokens: int | None, per_second: float | None) -> float | None:
    if not tokens or not per_second:
        return None
    return round(tokens / per_second, 3)
