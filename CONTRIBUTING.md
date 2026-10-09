# Contributing to ABM

## Setup and tests

```bash
pip install -e ".[replicate]"
python -m pytest            # reference/ and bsm/tests
```

Work on the `master` branch; open an issue before large changes.

## New claims need a preregistration

ABM is in a research phase: every new claim about accuracy, capacity or
comparison with a baseline is tested by a preregistration.

1. Write the preregistration in [`docs/preregistration/`](docs/preregistration/):
   hypothesis, prediction with its threshold, dataset, seeds, analysis, and what
   would count as a failure.
2. Commit and **push the preregistration before running** the experiment, so the
   timestamp precedes the result. Then run
   `tools/stamp_prereg.sh docs/preregistration/<name>.md examples/<harness>.py`:
   it refuses unpushed or uncommitted files and writes an OpenTimestamps proof
   (`<file>.ots`, anchored in Bitcoin, free; only the hash leaves the machine).
   Commit the `.ots` files. Anyone can check later with `ots verify <file>.ots`.
   Preregistrations 1–19 predate this rule: their order is attested only by
   local commit dates (see the paper, "Anteriority").
3. Run, save the raw results as JSON in `results/`, report against the
   preregistered threshold.

Negative results are recorded like positive ones (in BENCHMARKS.md and in the
preregistration); a failed prediction is never removed or rewritten.

## Code

- `reference/` is the executable specification: behaviour changes there need a
  version bump and a CHANGELOG entry.
- Tests for new code; numpy is the only runtime dependency.
