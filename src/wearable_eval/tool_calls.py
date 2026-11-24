"""Tool-call validity: is each call the model made a well-formed call to a real tool?

Checked against the tool schemas (wearable.registry.TOOL_SPECS; prompt v1.1 has the same
names, arguments, types, enums and required lists). One label per call:

- malformed_args: the arguments are not a JSON object (for example text that did not parse).
- schema_invalid: unknown tool, missing required argument, unknown argument, wrong JSON
  type, value outside an enum, or a date argument that is not YYYY-MM-DD.
- off_route: a valid call to a tool outside every accepted route of the question (an
  extra lookup, for example). Not an error: it is reported next to valid, apart from the
  two error labels. Only for questions with expected calls.
- valid: a valid call to a tool on an accepted route.

A valid call can still fail in the tool (a date outside the data); that is a tool error,
not a validity problem. "No call" is a property of an answer, not of a call: see scoring.
"""

from __future__ import annotations

import re
from typing import Any

from wearable.registry import TOOL_SPECS

LABELS = ("valid", "off_route", "schema_invalid", "malformed_args")

_SPECS = {s["name"]: s for s in TOOL_SPECS}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": (int, float),
                          "boolean": bool, "object": dict, "array": list}


def schema_problems(name: str, args: Any) -> list[str]:
    """What is wrong with a call against the tool's schema; empty if nothing."""
    if name not in _SPECS:
        return [f"unknown tool {name!r}"]
    if not isinstance(args, dict):
        return [f"arguments are {type(args).__name__}, not an object"]
    schema = _SPECS[name]["input_schema"]
    props = schema.get("properties", {})
    problems = [f"missing required {k!r}" for k in schema.get("required", []) if k not in args]
    if schema.get("additionalProperties") is False:
        problems += [f"unknown argument {k!r}" for k in args if k not in props]
    for key, value in args.items():
        prop = props.get(key)
        if not prop:
            continue
        kind = prop.get("type")
        # bool is an int in Python, but not a JSON integer or number.
        if kind and (not isinstance(value, _TYPES[kind])
                     or (kind in ("integer", "number") and isinstance(value, bool))):
            problems.append(f"{key}={value!r} is not {kind}")
            continue
        if "enum" in prop and value not in prop["enum"]:
            problems.append(f"{key}={value!r} is not one of {prop['enum']}")
        if prop.get("format") == "date" and not (isinstance(value, str)
                                                 and _ISO.fullmatch(value)):
            problems.append(f"{key}={value!r} is not YYYY-MM-DD")
    return problems


def classify(call: dict[str, Any], route_tools: set[str]) -> str:
    """The label of one recorded call (name, input). route_tools: the tools of every
    accepted route of the question, empty when the question has no expected calls."""
    args = call.get("input")
    if not isinstance(args, dict):
        return "malformed_args"
    if schema_problems(call["name"], args):
        return "schema_invalid"
    if route_tools and call["name"] not in route_tools:
        return "off_route"
    return "valid"
