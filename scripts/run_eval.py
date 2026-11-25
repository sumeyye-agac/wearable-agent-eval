"""Run every question in eval/questions.yaml through a model and save the run to runs/.

    .venv/bin/python scripts/run_eval.py                    # 3 repeats, runs/dev/
    .venv/bin/python scripts/run_eval.py --ids act-01,una-03 --repeats 1
    .venv/bin/python scripts/run_eval.py --final            # runs/final/ (committed)
    .venv/bin/python scripts/run_eval.py --resume runs/dev/2026-11-22_claude-haiku-5-5
    .venv/bin/python scripts/run_eval.py --thinking off      # Haiku without thinking
    .venv/bin/python scripts/run_eval.py --backend litert --repeats 1 \
        --model-file local/spike-litert/models/gemma-4-E2B-it.litertlm

The Anthropic backend calls the API and costs money; the key comes from ANTHROPIC_API_KEY
(a git-ignored .env file works). The LiteRT-LM backend runs Gemma on this machine's CPU.
Score the result with scripts/score_run.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from wearable.agent import DEFAULT_MODEL
from wearable_eval import prompt_v1
from wearable_eval.ground_truth import load_users
from wearable_eval.questions import load_questions, load_version
from wearable_eval.runs import (append_line, build_config, finished_pairs, new_run_dir,
                                read_run, repair_tail, resume_mismatches, run_anthropic,
                                run_litert, sha256_text, write_json, THINKING_CONFIG)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/demo/demo_daily.csv"
QUESTIONS = REPO / "eval/questions.yaml"
GROUND_TRUTH = REPO / "eval/ground_truth.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["anthropic", "litert"], default="anthropic")
    parser.add_argument("--model", help=f"default: {DEFAULT_MODEL}, or Gemma for litert")
    parser.add_argument("--thinking", choices=["on", "off"], default="on",
                        help="Anthropic only: on (API default, nothing sent) or off")
    parser.add_argument("--constrained", choices=["on", "off"], default="on",
                        help="LiteRT-LM only: constrained decoding (off for the ablation)")
    parser.add_argument("--model-file", type=Path,
                        help="LiteRT-LM only: the .litertlm model file")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", help="comma-separated question ids (default: all)")
    parser.add_argument("--final", action="store_true",
                        help="write to runs/final/ (committed) instead of runs/dev/")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="continue an interrupted run: skip (question, repeat) pairs "
                             "that finished, retry those that ended in an API error")
    args = parser.parse_args()

    users = load_users(DATA)
    questions = load_questions(QUESTIONS, set(users))
    if args.ids:
        wanted = args.ids.split(",")
        unknown = set(wanted) - {q.id for q in questions}
        if unknown:
            parser.error(f"unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q.id in wanted]

    litert_client, litert = None, None
    if args.resume:
        old = read_run(args.resume)[0]
        args.backend = old["backend"]
        if args.backend == "anthropic":
            args.thinking = {v: k for k, v in THINKING_CONFIG.items()}[old["thinking"]]
        else:
            args.constrained = "on" if old["litert"]["constrained_decoding"] else "off"
            args.model_file = args.model_file or Path(old["litert"]["model_file"])
    if args.backend == "litert":
        from wearable_eval.litert_client import MODEL_NAME, LiteRTClient, LiteRTSettings
        from wearable_eval.litert_client import lost_in_rendering, sha256_file

        if not args.model_file:
            parser.error("--backend litert needs --model-file")
        args.model = args.model or MODEL_NAME
        settings = LiteRTSettings(model_file=str(args.model_file),
                                  constrained_decoding=args.constrained == "on")
        print(f"Loading {args.model_file} ...")
        litert_client = LiteRTClient(settings)
        # The run's first question: on resume, the one the run started with.
        first_id = old["question_ids"][0] if args.resume else questions[0].id
        first = next(q for q in load_questions(QUESTIONS, set(users)) if q.id == first_id)
        rendered = litert_client.render_first_turn(
            prompt_v1.system_prompt(users[first.user_id]), prompt_v1.TOOL_SPECS,
            first.question)
        litert = {**litert_client.run_config(),
                  "model_file_sha256": sha256_file(args.model_file),
                  "lost_in_chat_template": lost_in_rendering(prompt_v1.TOOL_SPECS, rendered),
                  "first_turn_sha256": sha256_text(rendered)}
    args.model = args.model or DEFAULT_MODEL

    def config_for(model: str, repeats: int, ids: list[str]) -> dict:
        return build_config(model=model, repeats=repeats,
                            questions_version=load_version(QUESTIONS), question_ids=ids,
                            questions_path=QUESTIONS, ground_truth_path=GROUND_TRUTH,
                            data_path=DATA, users=users, backend=args.backend,
                            thinking=args.thinking, litert=litert)

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

    if args.backend == "anthropic":
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
        client = anthropic.Anthropic()

        def run_one(df, question):
            return run_anthropic(client, df, question, args.model, thinking=args.thinking)
    else:
        def run_one(df, question):
            return run_litert(litert_client, df, question, args.model)
    total = 0.0
    # Repeats are the outer loop, so each repeat is a full pass over the set.
    for repeat in range(1, args.repeats + 1):
        for q in questions:
            if (q.id, repeat) in done:
                continue
            record = {"question_id": q.id, "user_id": q.user_id, "repeat": repeat,
                      "model": args.model,
                      **run_one(users[q.user_id], q.question)}
            append_line(run_dir / "transcripts.jsonl", record)
            total += record["cost_usd"]
            print(f"r{repeat} {q.id:7} {record['status']:9} steps={record['steps']} "
                  f"{record['seconds']:6.1f}s ${record['cost_usd']:.4f} (total ${total:.3f})"
                  + (f"  {record['error']}" if record["error"] else ""))

    if litert_client is not None:
        litert_client.close()  # releases the last conversation before the engine
    config["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "config.json", config)
    print(f"Done. Score it: .venv/bin/python scripts/score_run.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
