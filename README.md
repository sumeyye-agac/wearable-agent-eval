# wearable-agent-eval

> **Status: in progress** (updated 2026-11-22)
>
> ✅ **Last done:** runner and scorer: every run is saved to `runs/` and scored from the saved files only.
>
> 🔜 **Next:** run and score the Claude Haiku 5.5 baseline.

<details>
<summary><b>Plan and progress</b></summary>

**Phase 0: Can a small model run on my phone at all?**
- [x] Installed Gemma-4-E2B-it on my Android phone with Google AI Edge Gallery
- [x] Measured speed on GPU and CPU: ~10 vs ~6.5 tok/s decode, ~3.0 vs ~2.2 GB RAM
- [x] Sanity check: one tool call came back as valid JSON with the right tool and date
- [x] Sanity check: asked for data no tool provides (blood oxygen); the model said it had no tool for it instead of inventing one
- [x] Measured time-to-first-token with the full agent prompt: 5,145 tokens (Gemma tokenizer), 29.6 s on GPU

**Phase 1: Evaluation harness**
- [x] Set up the repo and copied the tools and demo data from [ask-your-wearable](https://github.com/sumeyye-agac/ask-your-wearable), with its tests passing
- [x] Wrote a 28-question pilot set; every expected answer is computed by the tools, none written by hand
- [x] Added questions built on real gaps in the data, where the right answer is a refusal or must say how many days it covers
- [x] Metrics as regex checks, no LLM judge: coverage mentioned, and the set of numbers the tools support (the unsupported-number check that uses it is not built yet)
- [ ] Build the scoring: tool selection, numeric accuracy, unsupported numbers, refusal
- [x] Save every run to `runs/` (config, full transcripts, timings) and score from saved runs only
- [ ] Run each question 3 times and report the spread
- [ ] Run the Claude Haiku 5.5 baseline on the pilot set
- [ ] Grow the set to 100–150 questions

**Phase 2: Small model on my Mac**
- [ ] Per-model format adapter (Anthropic tool use, LiteRT-LM tool format), one shared loop and the same tools
- [ ] Run Gemma-4-E2B-it through the same harness
- [ ] Check whether Gemma corrects itself after a tool error (e.g. invalid dates)
- [ ] Optional second small model (Qwen3-1.7B)
- [ ] Results table and error analysis: wrong tool, invented numbers, broken format

**Phase 3: Android app**
- [ ] Minimal app with the model and the tools running on the phone
- [ ] Port the tools to Kotlin and check them against the Python outputs
- [ ] Measure latency with and without caching the fixed part of the prompt
- [ ] Run a 20-question subset on the phone and compare answers with the Mac runs
- [ ] One-minute demo video

**Phase 4: Write-up**
- [ ] Results table at the top of this README, with limits

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
