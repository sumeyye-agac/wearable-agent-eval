"""Run every question in eval/questions.yaml through a model and save the run to runs/.

    .venv/bin/python scripts/run_eval.py                    # 3 repeats, runs/dev/
    .venv/bin/python scripts/run_eval.py --ids act-01,una-03 --repeats 1
    .venv/bin/python scripts/run_eval.py --final            # runs/final/ (committed)
    .venv/bin/python scripts/run_eval.py --resume runs/dev/2026-11-22_claude-haiku-5-5

This calls the Anthropic API and costs money. The key comes from ANTHROPIC_API_KEY
(a git-ignored .env file works). Score the result with scripts/score_run.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from wearable.agent import DEFAULT_MODEL
from wearable_eval.ground_truth import load_users
from wearable_eval.questions import load_questions, load_version
from wearable_eval.runs import (append_line, build_config, finished_pairs, new_run_dir,
                                read_run, repair_tail, resume_mismatches, run_anthropic,
                                write_json)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/demo/demo_daily.csv"
QUESTIONS = REPO / "eval/questions.yaml"
GROUND_TRUTH = REPO / "eval/ground_truth.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", help="comma-separated question ids (default: all)")
    parser.add_argument("--final", action="store_true",
                        help="write to runs/final/ (committed) instead of runs/dev/")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="continue an interrupted run: skip (question, repeat) pairs "
                             "that finished, retry those that ended in an API error")
    args = parser.parse_args()

    import anthropic
    from dotenv import load_dotenv

    users = load_users(DATA)
    questions = load_questions(QUESTIONS, set(users))
    if args.ids:
        wanted = args.ids.split(",")
        unknown = set(wanted) - {q.id for q in questions}
        if unknown:
            parser.error(f"unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in wanted]

    def config_for(model: str, repeats: int, ids: list[str]) -> dict:
        return build_config(model=model, repeats=repeats,
                            questions_version=load_version(QUESTIONS), question_ids=ids,
                            questions_path=QUESTIONS, ground_truth_path=GROUND_TRUTH,
                            data_path=DATA, users=users)

    done: set[tuple[str, int]] = set()
    if args.resume:
        run_dir = args.resume
        config, lines = read_run(run_dir)
        current = config_for(config["model"], config["repeats"], config["question_ids"])
        mismatches = resume_mismatches(config, current)
        if mismatches:
            parser.error(f"cannot resume: {', '.join(mismatches)} changed since the run started")
        cut = repair_tail(run_dir / "transcripts.jsonl")
        done = finished_pairs(read_run(run_dir)[1])
        config.setdefault("resumes", []).append(
            {"at": current["started_at"], "git_commit": current["git_commit"],
             "git_dirty": current["git_dirty"], "pairs_done": len(done),
             "bytes_cut_from_tail": cut})
        args.model, args.repeats = config["model"], config["repeats"]
        questions = [q for q in load_questions(QUESTIONS, set(users))
                     if q.id in config["question_ids"]]
        print(f"Resuming {run_dir}: {len(done)} pairs done")
    else:
        run_dir = new_run_dir(REPO, "final" if args.final else "dev", args.model,
                              date.today().isoformat())
        config = config_for(args.model, args.repeats, [q.id for q in questions])
        print(f"Writing {run_dir.relative_to(REPO)}")
    write_json(run_dir / "config.json", config)

    load_dotenv(REPO / ".env")
    client = anthropic.Anthropic()
    total = 0.0
    # Repeats are the outer loop, so each repeat is a full pass over the set.
    for repeat in range(1, args.repeats + 1):
        for q in questions:
            if (q.id, repeat) in done:
                continue
            record = {"question_id": q.id, "user_id": q.user_id, "repeat": repeat,
                      "model": args.model,
                      **run_anthropic(client, users[q.user_id], q.question, args.model)}
            append_line(run_dir / "transcripts.jsonl", record)
            total += record["cost_usd"]
            print(f"r{repeat} {q.id:7} {record['status']:9} steps={record['steps']} "
                  f"{record['seconds']:6.1f}s ${record['cost_usd']:.4f} (total ${total:.3f})"
                  + (f"  {record['error']}" if record["error"] else ""))

    config["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "config.json", config)
    print(f"Done. Score it: .venv/bin/python scripts/score_run.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
