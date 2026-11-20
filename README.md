# wearable-agent-eval

How accurate is a small on-device model, compared with a cloud model, when both answer
health-data questions with the same tested tools? And what does it gain in latency, memory
and battery?

The agent and the tools come from
[ask-your-wearable](https://github.com/sumeyye-agac/ask-your-wearable). The copy here is
from commit `a725a58` (recorded in `docs/.source_commit`). The cloud reference is Claude
Haiku 5.5, and the on-device model is Gemma 4 E2B.

Work in progress. No results yet.

## Layout

| Path | Contents |
| --- | --- |
| `src/wearable/` | Tools, schema and agent loop, copied unchanged from ask-your-wearable |
| `src/wearable_eval/` | Question loading and ground truth |
| `eval/` | Question set and generated ground truth ([format](eval/README.md)) |
| `data/demo/` | Daily table for 3 demo users |
| `android/` | On-device app (not started) |

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/build_ground_truth.py
```

## Data

Furberg, Brinton, Keating & Ortiz (RTI International), *Crowd-sourced Fitbit datasets
03.12.2016–05.12.2016*, Zenodo, 2016, [doi:10.5281/zenodo.53894](https://doi.org/10.5281/zenodo.53894),
CC BY 4.0. The files in `data/demo/` are derived from it under the same licence.
Wellness information only; nothing here diagnoses anything.
