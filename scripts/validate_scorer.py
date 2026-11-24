"""Validate the scorer without calling any model: fake agents, null baseline, mutation test.

    .venv/bin/python scripts/validate_scorer.py runs/dev/<run> [runs/dev/<run> ...] \
        --out runs/dev/scorer_validation.json

The runs give the real answers for the null baseline and the mutation test; they are scored
against the current ground truth. Nothing is written into the run folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wearable_eval.ground_truth import load_users
from wearable_eval.runs import read_run, write_json
from wearable_eval.scoring import latest_lines, prompt_days_for, score_answer
from wearable_eval.validation import (empty_record, mutation_test, null_baseline,
                                      oracle_record, refuse_record)

REPO = Path(__file__).resolve().parents[1]


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = json.loads((REPO / "eval/ground_truth.json").read_text(encoding="utf-8"))
    truths = {q["id"]: q for q in ground_truth["questions"]}
    users = load_users(REPO / "data/demo/demo_daily.csv")
    prompt = prompt_days_for(users)
    report: dict = {"questions_version": ground_truth["questions_version"], "agents": {}}

    print(f"Questions {ground_truth['questions_version']}, {len(truths)} questions")
    checks = ("end_to_end", "tool_result_correct", "answer_correct", "refusal_correct",
              "coverage_mentioned")
    for name, make in (("oracle", lambda t: oracle_record(t, users[t["user_id"]])),
                       ("empty", empty_record), ("always_refuse", refuse_record)):
        rows = [score_answer(make(t), t, prompt[t["user_id"]]) for t in truths.values()]
        result = {}
        for m in checks:
            kept = [r[m] for r in rows if r[m] is not None]
            result[m] = {"passed": sum(kept), "n": len(kept)}
        result["passed_end_to_end"] = [r["question_id"] for r in rows if r["end_to_end"]]
        result["failed_end_to_end"] = [r["question_id"] for r in rows if not r["end_to_end"]]
        report["agents"][name] = result
        print(f"\n{name}: " + ", ".join(f"{m} {pct(v['passed'], v['n'])}"
                                       for m, v in result.items() if m in checks))
        shown = result["failed_end_to_end" if name == "oracle" else "passed_end_to_end"]
        print(f"  {'failed' if name == 'oracle' else 'passed'} end to end: {shown or 'none'}")

    report["runs"] = {}
    for run_dir in args.runs:
        config, lines = read_run(run_dir)
        records = [r for r in latest_lines(lines) if r["question_id"] in truths]
        null = null_baseline(records, truths, prompt)
        mutation = mutation_test(records, truths, prompt)
        report["runs"][run_dir.name] = {"answers": len(records), "null_baseline": null,
                                        "mutation_test": mutation}
        print(f"\n== {run_dir.name}: {len(records)} answers")
        print(f"null baseline (pairs sharing an output skipped: "
              f"{null['skipped_pairs_sharing_an_output']})")
        for kind in ("same_user", "cross_user"):
            c = null[kind]
            print(f"  {kind:10} pairs {c['pairs']}: numbers supported "
                  f"{pct(c['supported'], c['numbers'])}, small integers "
                  f"{pct(c['small_supported'], c['small_numbers'])}, dates "
                  f"{pct(c['dates_supported'], c['dates'])}, answers passing the support "
                  f"check {pct(c['answers_all_supported'], c['answers_with_numbers'])}")
        print("mutation test (detected = end_to_end turns false)")
        for kind, c in mutation["types"].items():
            extra = f", low-confidence only {c['low_confidence_only']}" \
                if c["low_confidence_only"] else ""
            print(f"  {kind:22} {pct(c['detected'], c['mutants'])}{extra}; of these, "
                  f"answer values/dates {pct(c['answer_item_detected'], c['answer_item_mutants'])}")
    write_json(args.out, report)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
