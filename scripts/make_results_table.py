"""Write the results table and 3 failure examples into README.md from the saved runs.

Reads only runs/final/*/scores.json, transcripts.jsonl and eval/ground_truth.json; never calls
a model. Edit the README block only by rerunning this script.

    .venv/bin/python scripts/make_results_table.py           # rewrite the README block
    .venv/bin/python scripts/make_results_table.py --check   # exit 1 if the README is stale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from wearable_eval.ground_truth import resolve
from wearable_eval.scoring import REFUSAL, latest_lines

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
GROUND_TRUTH = REPO / "eval/ground_truth.json"
START = "<!-- results:start -->"
END = "<!-- results:end -->"

# Rows in table order. The Gemma determinism rerun (gemma-4-E2B-it-2, 3 questions) is left out.
RUNS = (
    ("Claude Haiku 5.5, thinking off", "runs/final/2026-11-25_claude-haiku-5-5"),
    ("Gemma-4-E2B-it, Mac CPU, constrained decoding", "runs/final/2026-11-25_gemma-4-E2B-it"),
    ("Claude Haiku 5.5, thinking on (side row)", "runs/final/2026-11-25_claude-haiku-5-5-2"),
)
EXAMPLES_RUN = "runs/final/2026-11-25_gemma-4-E2B-it"

# Failure kinds, in the order an answer is put into one: the first check that failed.
KINDS = (
    ("refusal_failed", "Failed refusal: answered a question it should have refused"),
    ("wrong_tool_result", "Wrong tool result: wrong tool, wrong arguments or wrong route"),
    ("missing_coverage", "Missing coverage note: does not say how many days of data it used"),
)
EXCERPT_CHARS = 300


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n. k may be a mean (not whole)."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_errors_by_repeat(transcript: list[dict[str, Any]]) -> dict[int, int]:
    """LiteRT parse errors per repeat: calls the client could not read, so they are not in
    tool_calls and the scorer does not see them. Counted on the lines the scorer kept."""
    out: dict[int, int] = {}
    for record in latest_lines(transcript):
        n = sum(len(((r.get("response") or {}).get("litert") or {}).get("parse_errors") or [])
                for r in record["requests"])
        out[record["repeat"]] = out.get(record["repeat"], 0) + n
    return out


def per_repeat(scores: dict[str, Any], metric: str,
               parse_errors: dict[int, int] | None = None) -> list[tuple[int, int]]:
    """(k, n) for each repeat of a run. Valid tool calls: a parse error counts as one
    invalid call, as in decision rule 2."""
    if metric == "valid_calls":
        out = []
        for repeat in scores["repeats"]:
            labels = [label for a in scores["answers_scored"] if a["repeat"] == repeat
                      for label in a["call_validity"]]
            unread = (parse_errors or {}).get(repeat, 0)
            out.append((labels.count("valid"), len(labels) + unread))
        return out
    return [(c["k"], c["n"]) for c in scores["summary"][metric]["per_repeat"]]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cell(counts: list[tuple[int, int]]) -> str:
    """One run: "k/n rate, CI". Several: mean rate, min-max over runs, CI on mean k / mean n."""
    if len(counts) == 1:
        k, n = counts[0]
        lo, hi = wilson(k, n)
        return f"{k}/{n} = {pct(k / n)}<br>CI {pct(lo)}–{pct(hi)}"
    rates = [k / n for k, n in counts]
    mean_rate = sum(rates) / len(rates)
    mean_n = sum(n for _, n in counts) / len(counts)
    lo, hi = wilson(mean_rate * mean_n, mean_n)
    return (f"mean {pct(mean_rate)}, runs {pct(min(rates))}–{pct(max(rates))}"
            f"<br>CI {pct(lo)}–{pct(hi)}")


def interval(counts: list[tuple[int, int]]) -> tuple[float, float]:
    """The CI shown in a cell (see cell())."""
    mean_n = sum(n for _, n in counts) / len(counts)
    mean_rate = sum(k / n for k, n in counts) / len(counts)
    return wilson(mean_rate * mean_n, mean_n)


def refusal_note(runs: list[tuple[str, dict[str, Any], dict[int, int]]],
                 n_refusal: int) -> str:
    """Whether the refusal intervals of the first two rows (Haiku off, Gemma) overlap."""
    (lo_a, hi_a), (lo_b, hi_b) = (interval(per_repeat(s, "refusal_correct")) for _, s, _ in runs[:2])
    if lo_a <= hi_b and lo_b <= hi_a:
        return (f"- Refusal has only {n_refusal} questions per run, so its intervals are wide; "
                "the Haiku (thinking off) and Gemma intervals overlap, so the refusal gap is "
                "not clear yet.")
    return (f"- Refusal has only {n_refusal} questions per run; the Haiku (thinking off) and "
            "Gemma intervals do not overlap.")


def table(runs: list[tuple[str, dict[str, Any], dict[int, int]]]) -> str:
    lines = ["| Run | Repeats | End-to-end pass | Refusal correct | Valid tool calls |",
             "| --- | --- | --- | --- | --- |"]
    for label, scores, unread in runs:
        cells = [cell(per_repeat(scores, m, unread))
                 for m in ("end_to_end", "refusal_correct", "valid_calls")]
        lines.append(f"| {label} | {len(scores['repeats'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kind_of(answer: dict[str, Any]) -> str | None:
    """The failure kind of a scored answer, or None if it passed or fits no kind."""
    if answer["end_to_end"]:
        return None
    if answer["refusal_correct"] is False:
        return "refusal_failed"
    if answer["tool_result_correct"] is False:
        return "wrong_tool_result"
    if answer["coverage_mentioned"] is False:
        return "missing_coverage"
    return None


def calls_text(calls: list[dict[str, Any]], tool_key: str, args_key: str) -> str:
    if not calls:
        return "none"
    return ", ".join(f"`{c[tool_key]}({json.dumps(c[args_key], sort_keys=True)})`"
                     for c in calls)


def value_at(output: Any, path: str) -> Any:
    try:
        return resolve(output, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return "missing"


def why(kind: str, a: dict[str, Any], q: dict[str, Any], record: dict[str, Any]) -> str:
    """Which check failed, with the values the scorer compared."""
    if kind == "refusal_failed":
        if not REFUSAL.search(record["answer"] or ""):
            return "no refusal wording in the answer"
        numbers = a["unsupported_numbers"] + a["low_confidence_numbers"]
        return ("refusal wording found, but the answer states numbers the tools do not "
                "support: " + ", ".join(numbers))
    if kind == "missing_coverage":
        days = [c["value"] for e in q["expected_calls"] for c in e.get("coverage") or []]
        return ("the tool result covers " + ", ".join(map(str, days))
                + " days with data; the answer does not say so")
    reasons = []
    calls = [c for c in record["tool_calls"] if not c["is_error"]]
    for e in q["expected_calls"]:
        same = [c for c in calls if c["name"] == e["tool"]]
        if not same:
            reasons.append(f"no successful `{e['tool']}` call")
            continue
        got = same[0]["output"]
        for path, value in e["match_on"].items():
            if value_at(got, path) != value:
                reasons.append(f"`{path}` expected {json.dumps(value)}, "
                               f"got {json.dumps(value_at(got, path))}")
    return "; ".join(reasons[:3]) + (f" (+{len(reasons) - 3} more)" if len(reasons) > 3 else "")


def excerpt(text: str | None) -> str:
    text = (text or "").strip()
    if len(text) > EXCERPT_CHARS:
        text = text[:EXCERPT_CHARS].rstrip() + " …"
    return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())


def examples(scores: dict[str, Any], transcript: list[dict[str, Any]],
             truth: dict[str, Any]) -> str:
    """The first failing answer by question id for each failure kind."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for a in scores["answers_scored"]:
        kind = kind_of(a)
        if kind:
            by_kind.setdefault(kind, []).append(a)
    records = {(r["question_id"], r["repeat"]): r for r in transcript}
    parts = []
    for kind, title in KINDS:
        found = sorted(by_kind.get(kind, []), key=lambda a: (a["question_id"], a["repeat"]))
        if not found:
            continue
        a = found[0]
        q = truth["questions"][a["question_id"]]
        record = records[(a["question_id"], a["repeat"])]
        expected = ("refuse" + (", after " if q["expected_calls"] else "")
                    if q["expect_refusal"] else "")
        if q["expected_calls"]:
            expected += calls_text(q["expected_calls"], "tool", "args")
        parts.append("\n".join([
            f"**{title}** ({len(found)} of {scores['answers']} answers). "
            f"Example: `{a['question_id']}`, \"{q['question']}\"",
            "",
            f"- Expected: {expected}",
            f"- Gemma called: {calls_text(record['tool_calls'], 'name', 'input')}",
            f"- Why it failed: {why(kind, a, q, record)}",
            "- Gemma answered:",
            "",
            excerpt(record["answer"]),
        ]))
    return "\n\n".join(parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_block() -> str:
    runs = [(label, json.loads((REPO / d / "scores.json").read_text(encoding="utf-8")),
             parse_errors_by_repeat(read_jsonl(REPO / d / "transcripts.jsonl")))
            for label, d in RUNS]
    versions = {(s["scorer_version"], s["questions_version"]) for _, s, _ in runs}
    n_parse_errors = sum(n for _, _, unread in runs for n in unread.values())
    if len(versions) != 1:
        raise SystemExit(f"runs were scored with different scorer/question versions: {versions}")
    scorer, questions = versions.pop()
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    truth["questions"] = {q["id"]: q for q in truth["questions"]}
    n_questions = len(truth["questions"])
    n_refusal = sum(q["expect_refusal"] for q in truth["questions"].values())
    gemma = json.loads((REPO / EXAMPLES_RUN / "scores.json").read_text(encoding="utf-8"))
    transcript = read_jsonl(REPO / EXAMPLES_RUN / "transcripts.jsonl")
    return "\n".join([
        START,
        f"*Draft: scorer {scorer}, before blind labeling. Questions {questions}: "
        f"{n_questions} questions, {n_refusal} of them should be refused. "
        "Written by `scripts/make_results_table.py` from `runs/final/`.*",
        "",
        table(runs),
        "",
        "- CI is a 95% Wilson interval. For one run it uses that run's k/n. For Haiku "
        "thinking off it uses the mean over the 3 runs with n = 34 questions (8 for refusal), "
        "because the 3 runs repeat the same questions and are not independent; pooling them "
        "as 102 answers would make the interval too narrow.",
        "- Valid tool calls count calls, not questions (n differs per run); the Haiku CI uses "
        "the mean number of calls per run. Calls are not independent either: one answer can "
        "make several. A call the client could not read (a parse error) counts as one "
        f"invalid call; there are {n_parse_errors} in these runs.",
        refusal_note(runs, n_refusal),
        "- Gemma ran once at temperature 0; a rerun of 3 questions gave identical calls and "
        "answers.",
        "",
        "**Gemma failure examples.** Each failed answer is put in the first kind it fits, in "
        "this order; the example is the first such answer by question id.",
        "",
        examples(gemma, transcript, truth),
        END,
    ])


def replace_block(readme: str, block: str) -> str:
    start, end = readme.index(START), readme.index(END) + len(END)
    return readme[:start] + block + readme[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if README.md differs from what this script writes")
    check = parser.parse_args().check

    readme = README.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        print(f"README.md has no {START} ... {END} block.")
        return 1
    new = replace_block(readme, build_block())
    if check:
        if new != readme:
            print("README.md results block is stale; run scripts/make_results_table.py.")
            return 1
        print("README.md results block is up to date.")
        return 0
    README.write_text(new, encoding="utf-8")
    print("README.md results block written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
