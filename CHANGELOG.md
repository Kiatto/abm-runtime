# Changelog

## 1.1.0 (2026-10-09) — abm

- Input validation across `Memory`, `exact` and the CLI (`--dim 0` no longer loops;
  empty memories and out-of-range targets raise clear errors)
- Reads no longer mutate the codebook: an unknown symbol is not added by `query`/`chain`
  (changes test 5 under `--code current` by ≤ 3.3 points; published results unchanged)
- `import abm` no longer edits `sys.path`; tests moved out of the wheel
- CLI and `abm demo` print the exact-model contract; Law IV only with `--law-iv`
- README rewritten after tests 18–19: no density or algebra claims

## 1.0.1 (2026-07-18) — abm

Reconstructed from git log (commit 13538cc): examples in `examples/sdk/`, the
`abm` CLI (`abm inspect`, `abm demo`), docs and API polish. Later additions
shipped under 1.0.1 without a version bump: `abm.exact` (c2d7816, 2026-09-28,
bugs fixed in 2a41f74, 2026-09-30), `margin_z` and `Memory.query_z` (e963234,
2026-10-05).

## 1.0.0 (2026-07-18) — abm

Reconstructed from git log (commit d500b3b): first `abm-runtime` package,
`import abm` from the frozen reference implementation (`reference/abm.py`,
published 2026-07-14 in 5b63f08) with the Inspector (Law IV contract,
f5b79ae and b6645f7).

## Note on versions up to 1.0.x

Until 1.0.0rc1 the version numbers of this repository covered the **BSM** line
(the research codebase in `bsm/`, then called "BSM Foundation"). The `abm`
package starts at 1.0.0; the entry below describes BSM, not `abm`.

## 1.0.0rc1 (2026-07-03) — BSM, not abm

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
