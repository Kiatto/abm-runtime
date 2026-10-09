# Changelog

## 1.1.0 (2026-10-09) — abm

- Input validation across `Memory`, `exact` and the CLI (`--dim 0` no longer loops;
  empty memories and out-of-range targets raise clear errors)
- Reads no longer mutate the codebook: an unknown symbol is not added by `query`/`chain`
  (changes test 5 under `--code current` by ≤ 3.3 points; published results unchanged)
- `import abm` no longer edits `sys.path`; tests moved out of the wheel
- CLI and `abm demo` print the exact-model contract; Law IV only with `--law-iv`
- README rewritten after tests 18–19: no density or algebra claims

## 1.0.0rc1 (2026-07-03)

- Unified `BSM` entry point (encode, observe, recall, predict, route, sleep)
- Three encoders: Hash, Projection, Learned
- Memory Store with POPCOUNT Hamming search
- Router with prototype-based classification
- Lifecycle: sleep (consolidate/forget), health checks
- Metrics: info(), health(), metrics()
- Persistence: save() / load() (`.bsm-store.npz` format)
- BSM-Bench: `bsm-bench` CLI for reproducible benchmarks
- Formal Specification v1.0 (`docs/SPECIFICATION.md`)
- 6 initial RFCs (`docs/rfc/RFC-0001` through `RFC-0006`)
- Foundation 1.0 experiments sealed in `bsm/core/`
