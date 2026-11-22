"""Score a saved run folder and write scores.json into it. Never calls a model.

    .venv/bin/python scripts/score_run.py runs/dev/2026-11-22_claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wearable_eval.runs import read_run, sha256_file, write_json
from wearable_eval.scoring import METRICS, score_run

REPO = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO / "eval/ground_truth.json"


def _pct(v: float | None) -> str:
    return "  -  " if v is None else f"{100 * v:5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir

    config, lines = read_run(run_dir)
    if config["sha256"]["ground_truth.json"] != sha256_file(GROUND_TRUTH):
        print("eval/ground_truth.json changed since this run; scores would not be comparable.")
        return 1
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    scores = score_run(config, lines, ground_truth)
    write_json(run_dir / "scores.json", scores)

    s = scores["summary"]
    print(f"{scores['model']}, questions {scores['questions_version']}, "
          f"{scores['answers']} answers, statuses {scores['statuses']}")
    print(f"{'metric':26} {'n':>3}  " + "  ".join(f"  r{k}  " for k in scores["repeats"])
          + "   mean")
    for m in (*METRICS, "unsupported_number_rate", "answers_with_unsupported"):
        n = s[m].get("n_questions", "")
        print(f"{m:26} {n:>3}  " + "  ".join(_pct(v) for v in s[m]["per_repeat"])
              + f"  {_pct(s[m]['mean'])}")
    c = scores["consistency"]
    print(f"consistent over repeats: {c['consistent_questions']}/{c['questions']} questions")
    print(f"usage: {scores['usage']}")
    print(f"Wrote {run_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
