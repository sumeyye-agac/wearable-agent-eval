# Decision rules

Written before any Gemma output is read. Each rule says what happens for a given result,
so the result cannot shape the rule.

## 1. Scorer v2

Scorer v2 is required if, on the blind labels (`docs/labeling_policy.md`), any of these
holds:

- precision or recall below 0.9 (point estimate) for `end_to_end`, `refusal_correct`,
  `unsupported_numbers` or `unsupported_dates`, where that precision or recall rests on
  n >= 10. Below n = 10 it is reported with its interval but does not trigger scorer v2;
- for any one of `end_to_end`, `refusal_correct`, `unsupported_numbers` or
  `unsupported_dates`: 2 or more errors I label that the scorer misses.

If scorer v2 is built, both models are scored again with it. The scorer v1 numbers stay in
the repo, next to the v2 numbers.

Limit: the samples are small, so the intervals are wide. With n = 40 and a point estimate
of 0.9, the Wilson 95% interval is 0.77 to 0.96. Precision and recall are often measured on
fewer than 40 items (only the flagged errors, or only the labeled errors), and then the
interval is wider: 0.70 to 0.97 at n = 20.

## 2. Enum-date ablation (Gemma)

An ablation with dates as enums in the tool schemas is run if, in the Gemma main run:

- 20% or more of the tool calls are `malformed_args`, `schema_invalid` or a parse error (a
  call the client could not read, recorded as `parse_errors` in the run record), counting
  each parse error as one call, or
- those three together are the most common failure type.

## 3. Runs and repeats

- Haiku main run: thinking off, 3 repeats; `pass_all_repeats` (every repeat passes
  `end_to_end`) is reported.
- Haiku with thinking on: 1 run, as a side comparison.
- Gemma main run: 1 run (temperature 0, top_k 1). Gemma's determinism is checked by running
  3 questions again, and reported; it is not assumed for the other questions (a stated
  limit).

## 4. Freeze

After the first Gemma output is read, the questions (v1.3), the prompt (v1.1), the tools
and scorer v1 do not change because of what a model answered. Any change makes a new
version: both models are run again, and the old results are kept.

## 5. Reporting

Every run is reported, including ablations and bad results.
