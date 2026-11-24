"""Score a saved run folder and write scores.json into it. Never calls a model.

    .venv/bin/python scripts/score_run.py runs/dev/2026-11-22_claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wearable_eval.ground_truth import load_users
from wearable_eval.runs import read_run, sha256_file, write_json
from wearable_eval.scoring import SUMMARY_ROWS, prompt_days_for, score_run

REPO = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO / "eval/ground_truth.json"
DATA = REPO / "data/demo/demo_daily.csv"


def fmt(c: dict) -> str:
    """A count as "k/n rate", e.g. "3/8 37.5%"."""
    if not c["n"]:
        return "-"
    return f"{c['k']}/{c['n']} {100 * c['rate']:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir

    config, lines = read_run(run_dir)
    if config["sha256"]["ground_truth.json"] != sha256_file(GROUND_TRUTH):
        print("eval/ground_truth.json changed since this run; scores would not be comparable.")
        return 1
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    scores = score_run(config, lines, ground_truth, prompt_days_for(load_users(DATA)))
    write_json(run_dir / "scores.json", scores)

    s = scores["summary"]
    print(f"scorer {scores['scorer_version']}, {scores['model']}, "
          f"questions {scores['questions_version']}, "
          f"{scores['answers']} answers, statuses {scores['statuses']}")
    cols = [f"r{k}" for k in scores["repeats"]] + ["all"]
    print(f"{'metric':32}" + "".join(f"{c:>18}" for c in cols))
    for m in SUMMARY_ROWS:
        cells = [fmt(c) for c in s[m]["per_repeat"]] + [fmt(s[m]["all"])]
        print(f"{m:32}" + "".join(f"{c:>18}" for c in cells))
    p = scores["pass_all_repeats"]
    print(f"pass^{p['repeats']} (every repeat passes end to end): {fmt(p)} of questions")
    print("tool calls: " + ", ".join(f"{k} {fmt(v)}" for k, v in scores["call_validity"].items()))
    print(f"consistent over repeats: {fmt(scores['consistency'])} of questions")
    print(f"usage: {scores['usage']}")
    print(f"Wrote {run_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
