# Labeling policy

Written before any Gemma output exists. It does not change after the first label is made.

## 1. Purpose

I label a sample of answers by hand, blind, to measure how often scorer v1 agrees with a
human (precision and recall) before any result is reported. The scorer is regular
expressions and rules, not an LLM judge, so it needs a human check.

## 2. Sample

Frozen now, before any Gemma output exists.

- 60 answers: 40 from Gemma-4-E2B-it (constrained decoding on, main run, 1 run) and 20
  from Claude Haiku 5.5 (thinking off, main run, 3 repeats).
- Every question appears at least once: the 40 Gemma answers cover each of the 34
  questions once, plus 6 more questions drawn at random. The 20 Haiku answers are 20
  different questions drawn at random.
- Gemma has one answer per question, so there is no repeat to draw. For each sampled Haiku
  question, one of its 3 repeats is drawn at random.
- Seed: `20261125`. All draws (questions, repeats, order) use Python's
  `random.Random(20261125)`, in the order listed here.

## 3. Blinding

- The labeling sheet shows only: the question text, the tool calls with their results, and
  the final answer.
- Hidden: model name, run id, latency, token counts and every scorer verdict.
- The 60 answers are shuffled with the same seed. The file that maps each sheet row to its
  model and run lives in `local/` (never committed) and is opened only after all 60 are
  labeled.
- Known limit: the style of an answer may give the model away.

## 4. Labels per answer

- **end_to_end:** correct / wrong, defined exactly as scorer v1 decides it. Correct means
  all of these hold:
  - the model gave an answer;
  - where the question has expected tool calls, it called the tools of an accepted route
    and got the expected results;
  - the answer states what the question asks (its values and dates);
  - for a refusal question, it says the data cannot answer; where a coverage check
    applies, it states the day count;
  - no number or date in it is invented or self_computed, with three exceptions that do not
    make an answer wrong on their own: a unit conversion of a value the tools gave
    (minutes as hours and minutes, or as decimal hours), the evening before a wake-up date
    the tools gave, and a whole number below 13.

  The number-level labels below still mark those exceptions as self_computed or invented.
- **refusal_correct:** only for questions where the right answer says the data cannot
  answer (`expect_refusal: true`): correct / wrong.
- **coverage_mentioned:** only for questions with a coverage check (`coverage` in
  `eval/questions.yaml`): yes / no, the answer states how many days the result rests on.
- **Each number and each date** in the answer gets one label:
  - **supported:** a number equals a value in a tool result or in the question, or a count
    the tools imply (the number of dates in a list, the days in a start-to-end window); a
    guideline number counts only with its unit and only when that guideline was returned.
    A date is in a tool result or the question, or is one the system prompt states (the
    data range's first and last day, the last complete day, "this week" and "last week").
    Numbers in the system prompt are not a source. Rounding and unit display that the tools
    give count as supported (for example "7 h 31 min" for 451.3 minutes). A weekday named
    next to a date must be that date's weekday; a wrong one makes the date invented.
  - **self_computed:** the model did arithmetic or a conversion itself (a difference, a sum,
    minutes turned into hours, the evening before a wake-up date).
  - **invented:** no source at all.

  self_computed counts as an error even when the arithmetic is right. The rule is that the
  model never computes; the tools return every number an answer needs.

## 5. Scorer agreement

- The positive class is an error or failure: an answer that fails, a number or date that is
  not supported.
- **Precision:** the share of errors flagged by the scorer that I also label as errors.
- **Recall:** the share of errors I label that the scorer flags.
- Answer-level metrics (`end_to_end`, `refusal_correct`, `coverage_mentioned`) are measured
  per answer.
- `unsupported_numbers` and `unsupported_dates` are measured per number and per date. An
  error is a number or date I label self_computed or invented. The scorer flags a number
  when it is in `unsupported_numbers` or `model_computed`, and a date when it is in
  `unsupported_dates` or `model_computed`. A number in `low_confidence_numbers` is not
  flagged, so an error I label there counts as a recall miss.
- A confusion table is reported next to them: my label (supported / self_computed /
  invented) against the scorer's bucket (supported / model_computed / unsupported /
  low_confidence; dates have no low_confidence bucket).
- Answer-level and number/date-level results are reported separately, never pooled.
- Each precision and recall is reported with its point estimate, its Wilson 95% interval
  and n.
