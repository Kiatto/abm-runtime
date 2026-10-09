# ABM — Algebraic Binary Memory

**ABM is a deterministic binary memory for symbolic facts whose accuracy can be
computed before it is built: from the dimension and the facts it will hold, with
no fitted parameter.** That is what it offers, and the only thing measured in
its favour.

It is not dense and its algebra buys nothing: at equal bits an exact store, a
Bloom filter or a `dict` + join beat it in every cell measured (preregistered
tests 18 and 19), and in Python it is at least 2 600× slower than a `dict`. The
computed accuracy is a mean over queries, not a per-query guarantee, and an exact
store's accuracy is computable too, more simply. ABM is of use only where a
distributed binary representation is required anyway. Read
[Limits](#limits) before using it.

The LLM is the compiler: it reads documents and emits triples. ABM stores them in
one holographic binary trace and answers single- and multi-hop queries, with the
expected accuracy computed in advance.

```
Documents → LLM compiler → triples → ABM → exact contract (expected accuracy)
                                       ↓
                                    queries
```

## Quickstart

ABM is not published on PyPI. Install it from GitHub:

```bash
pip install git+https://github.com/Kiatto/abm-runtime
abm demo
abm inspect examples/triples.json --dim 8192 --grounding 0.93
```

`abm inspect` reads a JSON file holding a non-empty array of
`[subject, relation, object]` triples (each element is converted to a string),
the natural output of an LLM extractor; see
[`examples/triples.json`](examples/triples.json). `--grounding` is the audited
precision of your extractor, in [0, 1].

Developer guide (no theory required): [docs/SDK.md](docs/SDK.md).
What is measured, what failed, and how to rerun all of it:
[BENCHMARKS.md](BENCHMARKS.md).
The runtime is numpy-only: `abm.py` (the specification), `exact.py` (the
model), `inspector.py` (the superseded Law IV contract) and `cli.py`, about
1 300 lines.

### Supported configurations

Every row below was executed, not inferred, with the full suite
(`python -m pytest`: about 300 tests, `reference/` and `bsm/tests`).

| Configuration | Supported | Performance |
|---|---|---|
| Python 3.10 – 3.13 | ✓ | — |
| Python ≤ 3.9 | ✗ | `int.bit_count()` requires 3.10 |
| NumPy 1.24 – 1.26 | ✓ | **reduced**: cleanup ~12x slower |
| NumPy ≥ 2.0 | ✓ | nominal |

NumPy 1.x is correct and passes every test bit for bit; it lacks
`np.bitwise_count`, so the popcount falls back to a byte lookup table. Full
matrix, measured penalties and what the matrix does *not* cover (single BLAS,
macOS x86-64, untested on big-endian): [COMPATIBILITY.md](COMPATIBILITY.md).
CI runs the full suite on Linux x86-64, macOS arm64 and Windows x86-64, and
checks that codewords are bit-identical across all three.

### Sizing a memory before storing anything

`abm.exact` computes the accuracy of a memory from its dimension and from the
facts it will hold — with no fitted parameter, counting symmetric relations
(which the encoding fuses into one fact) and the aliases they create. It is the
model tested by the paper's preregistrations (`docs/preregistration/`).

```python
from abm import exact

triples = [("payment_service", "requires", "auth_service"), ...]
exact.contract_for(triples, dim=4096)   # expected accuracy, alias ceiling, twins
exact.min_dimension(triples, 0.8)       # smallest D predicted to reach 80%
```

With symmetric relations, aliases cap the accuracy of some queries whatever the
dimension: `contract_for` reports that ceiling, and `min_dimension` returns
`None` for targets above it. The older contract in `inspector` uses the
asymptotic Law IV, which on graphs with symmetric relations promises more than
it delivers (preregistration 10).

Duplicates are not removed: a triple stored twice is one fact of weight 2, in the
memory and in the model; `(s, r, o)` and `(o, r, s)` are the same vector, and all
self-loops `(s, r, s)` of a relation are one vector that makes every subject its
own alias. Deduplicate extractor output first if that is not what you want.

```python
from abm import Memory, exact
from abm.inspector import stats, contract, report

mem = Memory(dim=8192)
mem.store("payment_service", "requires", "auth_service")
mem.store("auth_service", "writes_to", "session_store")

# elementary query: (answer, confidence). The confidence is a logistic of the
# observed margin: it ranks answers, 0.5 = noise, but it is NOT calibrated.
mem.query("payment_service", "requires")     # → ("auth_service", 0.99…)
# a subject never stored is not added to the codebook: the answer is noise,
# recognizable by a margin near 0 (mem.query_z)

# multi-hop reasoning
mem.chain("payment_service", ["requires", "writes_to"])  # → ("session_store", 0.9…)
# the confidence of a chain is the product of the per-hop confidences

# algebraic truth oracle: one Hamming distance
mem.member("payment_service", "requires", "auth_service")  # → True

# the exact contract, computed BEFORE storing anything
exact.contract_for([("payment_service", "requires", "auth_service"),
                    ("auth_service", "writes_to", "session_store")], dim=8192)

# the older asymptotic Law IV contract (superseded, kept for compatibility)
print(contract(mem, grounding=0.93))
```

`abm inspect` and `abm demo` print the exact contract; `abm inspect --law-iv`
adds the Law IV report, labelled as superseded. Output of `abm demo`:

```
MEMORY CONTRACT (exact model, abm.exact.contract_for)
  Facts           =  3 (D=4096, codebook=7)
  Expected accuracy = 100.0% (mean over the stored (subject, relation) queries)
  Alias ceiling   =  100.0% (whatever D)
  Symmetric twins =  0.0% of triples; queries with aliases 0.0%
  Grounding       =  93.0% (projected single query 93.0%)
  A mean over queries, not a per-query guarantee.
```

## Why trust the contract

The exact model was tested by nineteen preregistrations; failures are kept in
the record ([BENCHMARKS.md](BENCHMARKS.md)):

- **Exact model**: 0.27 points of error at a dimension never used before
  (D = 16 384), 0.6–1.1 points on dense subgraphs of two real knowledge
  graphs, zero fitted parameters; choosing D in advance kept the promise on 80
  unseen subgraphs (test 10), where the asymptotic Law IV fell 2.8–6.2 points
  short on WN18RR.
- **Capacity law** (asymptotic, superseded by the exact model)
  N\* = k·2D/(π·z_G(M)²), k = 0.92 ± 0.03 measured.
- **Hop composition** Acc(h) = p^h — an approximation: hops on the same
  trace are not independent; the exact two-hop dependence is
  `exact.two_hop_joint`, longer chains `exact.chain_accuracy_mc`.
- **Resource Composition Law** Acc = E_q[Pg] × Pr(N_eff) — your
  extractor's audited precision composes multiplicatively with the
  memory's predicted accuracy; corroborated against i.i.d., clustered
  and systematic error structures.
- **End-to-end pilot**: documents → compiler → 1000 queries; contract
  issued pre-query with a declared CI; theory error 1.7%.
- **Deterministic**: same inputs, same bits, same answers.

Full record: [the paper](docs/paper.md) ·
[FORMALISM v2.1](docs/FORMALISM.md) (normative, frozen) ·
[experiment reports](docs/) · raw results as JSON in [`results/`](results/).

## Repository layout

- [`reference/abm.py`](reference/abm.py) — the executable specification
  (frozen against FORMALISM, now v1.1.0: input validation and read-only
  queries; changes require a version bump)
- [`reference/inspector.py`](reference/inspector.py) — the asymptotic Law IV
  contract (superseded by `exact`): `stats()`, `contract()`, `report()`,
  `aliasing()`
- [`reference/exact.py`](reference/exact.py) — the exact finite-dimension
  model (`contract_for`, `min_dimension`, `predict_queries`)
- [`reference/cli.py`](reference/cli.py) — the `abm` command
- [`reference/tests/`](reference/tests/) — property tests against the axioms,
  the model against exhaustive enumeration and Monte Carlo, CLI smoke tests
  (not shipped in the wheel)
- [`examples/`](examples/) — every experiment behind every number above,
  reproducible; `industrial_pilot.py --extraction your_llm_output.json`
  plugs in a real LLM compiler
- [`bsm/`](bsm/) — the research codebase the theory grew out of
  (encoders, RAG integration, reasoning engine)

## Limits

Stated plainly; details and numbers in
[BENCHMARKS.md, Limits](BENCHMARKS.md#limits-stated-plainly).

- **Not dense, not a compressor.** At equal bits an idealised exact store
  recalls more in 14/14 cells and a Bloom filter makes fewer membership errors in
  every cell (test 18); ABM uses 3.8–46× the Fano minimum.
- **The algebra gives no advantage.** On two-hop chains and compiled
  compositions a `dict` + join or a path table of the same bits is at least as
  accurate in all 16 cells per task (test 19). Inverse queries are not "free":
  the symmetric encoding turns them into aliases that cap accuracy.
- **Slow.** In Python, ABM is 2 600–229 725× slower per question than a `dict`
  (test 19).
- **What remains** is accuracy computable before the memory is built, as a mean
  over queries. An exact store's is computable too, more simply: this matters
  only where a distributed binary representation is required anyway.
- The model assumes independent codewords and facts; even cycles (four facts on
  a rectangle) break it, by up to 6 points on a loaded biclique. Tiny codebooks
  are out of scope.
- `confidence` is not calibrated; the Law IV contract in `abm.inspector` is
  optimistic on graphs with symmetric relations.

## What ABM is not

Not an embedding store, not a vector-DB replacement, not "a better RAG", and
not denser or faster than an exact store. It is a binary memory with a
parameter-free model of its own accuracy.

## License

MIT
