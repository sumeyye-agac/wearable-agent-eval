"""Compute eval/ground_truth.json from eval/questions.yaml by running the tools.

    .venv/bin/python scripts/build_ground_truth.py          # write the file
    .venv/bin/python scripts/build_ground_truth.py --check  # fail if it is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wearable_eval.ground_truth import build, load_users
from wearable_eval.questions import load_questions, load_version

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data/demo/demo_daily.csv"
QUESTIONS = REPO / "eval/questions.yaml"
GROUND_TRUTH = REPO / "eval/ground_truth.json"


def render() -> str:
    users = load_users(DEMO)
    questions = load_questions(QUESTIONS, set(users))
    data = build(questions, users, load_version(QUESTIONS))
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if ground_truth.json differs from a fresh build")
    text = render()
    if parser.parse_args().check:
        current = GROUND_TRUTH.read_text(encoding="utf-8") if GROUND_TRUTH.exists() else ""
        if current != text:
            print(f"{GROUND_TRUTH.relative_to(REPO)} is out of date; rerun without --check.")
            return 1
        print("Ground truth is up to date.")
        return 0
    GROUND_TRUTH.write_text(text, encoding="utf-8")
    n = len(json.loads(text)["questions"])
    print(f"Wrote {n} questions to {GROUND_TRUTH.relative_to(REPO)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
