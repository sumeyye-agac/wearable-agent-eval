"""Run the question set through a model and save every answer to disk.

A run is a folder runs/<dev|final>/<date>_<model>/ holding
- config.json: what was run (model, settings, hashes of the inputs, git commit)
- transcripts.jsonl: one line per (question, repeat) with the tool calls, the tool results,
  the answer, and the time and token usage of every model request.
Scoring reads these files only (scoring.py) and never calls a model.

Only the Anthropic backend exists so far. A backend takes (client, df, question, model)
and returns the fields of one transcript line, so a LiteRT-LM backend can sit next to it
without changing the runner or the scorer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from wearable import agent
from wearable.registry import TOOL_SPECS

# Per request; thinking counts towards it. agent.py sends 4096; the runner's client wrapper
# raises it to leave room for adaptive thinking, without changing agent.py.
MAX_TOKENS = 8192
TOKEN_FIELDS = ("input_tokens", "output_tokens", "thinking_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json(value: Any) -> Any:
    """SDK objects (pydantic models), stand-ins in tests and plain values, as JSON data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, SimpleNamespace):
        return to_json(vars(value))
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def _thinking_tokens(usage: dict[str, Any]) -> int:
    # Thinking is billed as output tokens; the API also reports it on its own here.
    return ((usage.get("output_tokens_details") or {}).get("thinking_tokens")) or 0


class RecordingClient:
    """Wraps an Anthropic client. Passes every request through with max_tokens set to the
    runner's value, and records per request:
    - request: every field sent except the long fixed ones (system and tools are stored as
      sha256, messages as the one new message: the question, or the tool results exactly
      as sent back, with tool_use_id, the JSON string and is_error), plus the field names;
    - response: the full response as returned (every content block: thinking with its
      signature, text, tool_use with id, name and input), stop_reason and usage;
    - seconds: wall-clock time of the request."""

    def __init__(self, client: Any, max_tokens: int = MAX_TOKENS):
        self._client = client
        self._max_tokens = max_tokens
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        kwargs["max_tokens"] = self._max_tokens
        messages = kwargs["messages"]
        sent = {k: to_json(v) for k, v in kwargs.items()
                if k not in ("system", "tools", "messages")}
        sent.update(fields=sorted(kwargs), n_messages=len(messages),
                    new_message=to_json(messages[-1]),
                    system_sha256=sha256_text(kwargs.get("system", "")),
                    tools_sha256=sha256_text(json.dumps(kwargs.get("tools"), sort_keys=True)))
        start = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        seconds = round(time.perf_counter() - start, 3)
        body = to_json(response)
        self.requests.append({
            "step": len(self.requests) + 1,
            "seconds": seconds,
            "stop_reason": body["stop_reason"],
            "usage": body["usage"],
            "request": sent,
            "response": body,
        })
        return response


def token_totals(requests: list[dict[str, Any]]) -> dict[str, int]:
    totals = dict.fromkeys(TOKEN_FIELDS, 0)
    for r in requests:
        u = r["usage"]
        for key in TOKEN_FIELDS:
            totals[key] += _thinking_tokens(u) if key == "thinking_tokens" else (u.get(key) or 0)
    return totals


def run_anthropic(client: Any, df: pd.DataFrame, question: str, model: str) -> dict[str, Any]:
    """One question through the existing agent loop (src/wearable/agent.py), unchanged.

    API errors are recorded in the line instead of stopping the run. The cost is summed
    from the recorded usage, so a question that failed half way still counts what it spent.
    """
    import anthropic

    recorder = RecordingClient(client)
    start = time.perf_counter()
    result, error = None, None
    try:
        result = agent.ask(recorder, df, question, model=model)
    except anthropic.APIError as err:
        error = f"{type(err).__name__}: {err}"
    seconds = round(time.perf_counter() - start, 3)
    cost = sum(agent.request_cost(SimpleNamespace(**r["usage"]), model)
               for r in recorder.requests)
    return {
        "status": result.status if result else "error",
        "answer": result.answer if result else None,
        "steps": len(recorder.requests),
        "tool_calls": [asdict(c) for c in result.tool_calls] if result else [],
        "requests": recorder.requests,
        "seconds": seconds,
        "tokens": token_totals(recorder.requests),
        "cost_usd": round(cost, 6),
        "error": error,
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build_config(*, model: str, repeats: int, questions_version: str, question_ids: list[str],
                 questions_path: Path, ground_truth_path: Path, data_path: Path,
                 users: dict[str, pd.DataFrame]) -> dict[str, Any]:
    import anthropic

    return {
        "model": model,
        "backend": "anthropic",
        "temperature": None,  # Claude Haiku 5.5 rejects a non-default temperature; none is sent
        "thinking": "adaptive (API default, not sent)",
        "effort": agent.EFFORT,
        "max_tokens": MAX_TOKENS,  # set by the runner; agent.py's own value is 4096
        "max_steps": agent.MAX_STEPS,
        "max_usd_per_question": agent.MAX_USD_PER_QUESTION,
        "repeats": repeats,
        "questions_version": questions_version,
        "question_ids": question_ids,
        "sha256": {
            "questions.yaml": sha256_file(questions_path),
            "ground_truth.json": sha256_file(ground_truth_path),
            "demo_daily.csv": sha256_file(data_path),
            "tool_specs": sha256_text(json.dumps(TOOL_SPECS, sort_keys=True)),
            "system_prompts": {uid: sha256_text(agent.system_prompt(df))
                               for uid, df in sorted(users.items())},
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "versions": {"python": platform.python_version(), "anthropic": anthropic.__version__},
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_at": None,
    }


def new_run_dir(root: Path, kind: str, model: str, day: str) -> Path:
    """runs/<kind>/<day>_<model>, with -2, -3, ... added if that folder already exists."""
    base = root / "runs" / kind / f"{day}_{model}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    path = run_dir / "transcripts.jsonl"
    lines = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    return config, lines


def repair_tail(path: Path) -> int:
    """Cut a last line that an interruption left half written. Returns the bytes removed."""
    if not path.exists():
        return 0
    data = path.read_bytes()
    keep = data[:data.rfind(b"\n") + 1]
    if len(keep) != len(data):
        path.write_bytes(keep)
    return len(data) - len(keep)


def finished_pairs(lines: list[dict[str, Any]]) -> set[tuple[str, int]]:
    """(question id, repeat) pairs with a line that did not end in an API error."""
    return {(r["question_id"], r["repeat"]) for r in lines if r["status"] != "error"}


RESUME_KEYS = ("model", "backend", "temperature", "thinking", "effort", "max_tokens",
               "max_steps", "max_usd_per_question", "repeats", "questions_version",
               "question_ids", "sha256")


def resume_mismatches(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Settings or inputs that differ between a run's config and the current state. A run
    may only be resumed when this is empty, so all its lines answer the same test."""
    return [k for k in RESUME_KEYS if old.get(k) != new.get(k)]
