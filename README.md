# wearable-agent-eval

> **Status: in progress** (updated 2026-11-24)
>
> ✅ **Last done:** a Claude Haiku 5.5 pilot run on questions v1 (not reported as a result: the scorer is not validated yet), and a LiteRT-LM spike on my Mac (go: valid tool calls, deterministic at temperature 0).
>
> 🔜 **Next:** question set v1.1.

<details>
<summary><b>Plan and progress</b></summary>

**Done so far**
- [x] Set up the repo and copied the tools and demo data from [ask-your-wearable](https://github.com/sumeyye-agac/ask-your-wearable), with its tests passing
- [x] Gemma-4-E2B-it runs on my Android phone (Google AI Edge Gallery): ~10 vs ~6.5 tok/s decode on GPU vs CPU, ~3.0 vs ~2.2 GB RAM; time-to-first-token with the full agent prompt (5,145 tokens, Gemma tokenizer): 29.6 s on GPU
- [x] A 28-question pilot set (questions v1); every expected answer is computed by the tools, none written by hand
- [x] A runner that saves every run to `runs/`, and a scorer that reads only the saved runs
- [x] A Claude Haiku 5.5 pilot run on questions v1 (not reported as a result: the scorer is not validated yet)
- [x] A LiteRT-LM spike on my Mac: Gemma-4-E2B-it makes valid tool calls and is deterministic at temperature 0

**A. Freeze the inputs**
- [ ] Question set v1.1
- [ ] Prompt v1
- [ ] Scorer v1 with validation: null baseline, mutation test, labeling policy, and decision rules written before any Gemma output is read

**B. Runs on the frozen inputs**
- [ ] Claude Haiku 5.5 with thinking on and with thinking off
- [ ] Gemma-4-E2B-it on my Mac
- [ ] Blind labeling of 60 answers
- [ ] First comparison table and failure taxonomy

**C. Phone and questions v2**
- [ ] KV cache for the fixed part of the prompt
- [ ] Latency and RAM on the phone
- [ ] 20-question parity between the phone and the Mac runs
- [ ] Demo video
- [ ] Question set v2

</details>

## How it works
![System diagram](docs/architecture.png)

How accurate is a small on-device model, compared with a cloud model, when both answer
health-data questions with the same tested tools? And what does it gain in latency, memory
and battery?

The agent and the tools come from
[ask-your-wearable](https://github.com/sumeyye-agac/ask-your-wearable). The copy here is
from commit `a725a58`. The cloud reference is Claude Haiku 5.5, and the on-device model is
Gemma-4-E2B-it.

## Layout

| Path | Contents |
| --- | --- |
| `src/wearable/` | Tools, schema and agent loop, copied unchanged from ask-your-wearable |
| `src/wearable_eval/` | Question loading, ground truth, runner and scoring |
| `runs/` | Saved runs: `final/` is committed, `dev/` stays local |
| `eval/` | Question set and generated ground truth ([format](eval/README.md)) |
| `data/demo/` | Daily table for 3 demo users |
| `android/` | On-device app (not started) |

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/build_ground_truth.py

# Needs ANTHROPIC_API_KEY (a git-ignored .env file works); costs money
.venv/bin/python scripts/run_eval.py                       # 3 repeats into runs/dev/
.venv/bin/python scripts/run_eval.py --resume runs/dev/<run>  # continue an interrupted run
.venv/bin/python scripts/score_run.py runs/dev/<run>          # never calls a model
```

## Data

Furberg, Brinton, Keating & Ortiz (RTI International), *Crowd-sourced Fitbit datasets
03.12.2016–05.12.2016*, Zenodo, 2016, [doi:10.5281/zenodo.53894](https://doi.org/10.5281/zenodo.53894),
CC BY 4.0. The files in `data/demo/` are derived from it under the same licence.
Wellness information only; nothing here diagnoses anything.
