# ABM — Algebraic Binary Memory

**ABM is a deterministic binary memory for symbolic facts whose accuracy can be
computed before it is built: from the dimension and the facts it will hold, with
no fitted parameter.** That is what it offers, and the only thing measured in
its favour.

It is not dense and its algebra buys nothing: at equal bits an exact store, a
Bloom filter or a `dict` + join beat it in every cell measured (preregistered
tests 18 and 19), a store with polar or BCH codes still beats it when stored
bits flip at rates up to 20% (test 20), and in Python it is at least 2 600×
slower than a `dict`. The
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
  Alias ceiling   =  100.0% (expected, random insertion order)
  Symmetric twins =  0.0% of triples; queries with aliases 0.0%
  Grounding       =  93.0% (projected single query 93.0%)
  A mean over queries, not a per-query guarantee.
```

### Sizing a VSA memory

`abm size` asks the model of `abm.sizing` which of six VSA families (MAP-B, the
reference; MAP-I, FHRR, HRR, BSDC-OR, BSDC-S) reaches a target mean accuracy at
the fewest bits, for *your* triples, and optionally measures it on real memories:

```
abm size examples/size_demo.json --target 0.9 --verify
```

Input: `.json` (a list of `[s, r, o]`, as for `abm inspect`), or `.tsv` / `.csv`
with three columns and no header (blank lines and `#` lines skipped). Names are
only stripped, never lowercased: names equal up to case are reported, because
MAP-B gives them the same atom. Options: `--families MAP-B,FHRR`, `--budget 64KB`
or `512000bits` (binary units, trace bits only), `--queries FILE` (`[s, r]`
pairs), `--seeds N` and `--tol 0.01` for `--verify`, `--max-mem 1GB` (dimensions
whose estimated peak memory exceeds it are not built), `--json` (schema
`abm.size/1`, versioned in the output), `--export torchhd [--out FILE]`.
Exit codes: 0 at least one recommendation is `validated` (and, with `--verify`,
verified), 3 none is, 2 usage error, 1 input error (including a target above the
alias ceiling). Real output on the 126-triple demo graph (a symmetric relation
and a pair `{s, o}` linked by two relations, which puts MAP-B under caution):

```
GRAPH  examples/size_demo.json (json)
  facts N = 126 (weighted; 126 distinct), symbols M = 116, queries = 126, regime = uniform, alias ceiling = 1.0000
  statistic                 value   validated range (preregs 21-22)
  N                           126   [100, 1600]
  alias (queries)          0.0000   [0, 0.094]
  twins (vectors)          0.0080   [0, 0.144]
  pair multi-relation      0.0159   [0, 0.3514]
  multi-object             0.0000   [0, 0.1506]
  hub subject max          0.0159   [0.001437, 0.284]
  hub subject p99          0.0159   [0.0009741, 0.2132]
  hub object max           0.1111   [0.0095, 0.7993]
  hub object p99           0.1079   [0.002095, 0.6604]
  relation max             0.4762   [0.06175, 0.804]
  relation p99             0.4619   [0.05351, 0.7512]

VERIFY COST (before running): 16 memories per family (8 selection + 8 report seeds); time is a crude estimate (one pilot memory × count; more if D is corrected), memory from families.estimate_bytes
  BSDC-OR  D=1792    peak ≈ 1.7 MiB (≈ 12.8 MiB at 8·D)  time ~0.1 s or more
  MAP-B    D=3136    peak ≈ 8.1 MiB (≈ 37.7 MiB at 8·D)  time ~3.2 s or more
  MAP-I    D=2048    peak ≈ 7.9 MiB (≈ 38.6 MiB at 8·D)  time ~2.4 s or more
  BSDC-S   D=2560    peak ≈ 2.4 MiB (≈ 16.3 MiB at 8·D)  time ~0.4 s or more
  FHRR     D=1024    peak ≈ 5.4 MiB (≈ 43.6 MiB at 8·D)  time ~1.7 s or more
  HRR      D=2944    peak ≈ 12.7 MiB (≈ 31.5 MiB at 8·D)  time ~3.9 s or more

RECOMMENDATIONS  target = 0.9 (mean accuracy over queries), sorted by bits
  family         D        KB  predicted          measured [95% CI]     q10  verified  reliability 
  BSDC-OR     1792      0.22     0.9290    0.9435 [0.9220, 0.9649]  0.9159       yes  validated   
  MAP-B       3200      0.39     0.9064    0.9117 [0.8906, 0.9328]  0.8873       yes  caution     
  MAP-I       2112      1.80     0.9184    0.9296 [0.9152, 0.9439]  0.9103       yes  validated   
  BSDC-S      2816      2.41     0.9400    0.9375 [0.9240, 0.9510]  0.9183       yes  validated   
  FHRR        1024      4.00     0.9096    0.9187 [0.8990, 0.9383]  0.8865       yes  validated   
  HRR         2944     11.50     0.9010    0.9157 [0.8944, 0.9369]  0.8921       yes  validated   

WARNINGS AND BENCHMARK ERROR
  BSDC-OR: observed benchmark error (uniform) mean 0.72 pt, worst 1.83 pt over 20 cells
  MAP-B: observed benchmark error (uniform) mean 0.74 pt, worst 1.79 pt over 20 cells
    - MAP-B: 1.6% of the facts share their unordered pair {s, o} with a fact of another relation (e.g. (s, r1, o) + (o, r2, s)); such facts share s and o, so in the MAP-B majority they cancel wherever ρ(r1) ≠ ρ(r2); the model treats them as independent and is optimistic (YAGO3-10 dense; tens of points on synthetic graphs). Use verify=True
    - verification: at D = 3136 the selection measure rejected the target (predicted 0.9003, measured 0.8998, model optimistic by 0.0 points; 95% CI lower bound 0.8832 < target − tol = 0.8900); D increased from 3136 to 3200
  MAP-I: observed benchmark error (uniform) mean 0.81 pt, worst 2.98 pt over 20 cells
    - verification: at D = 2048 the selection measure rejected the target (predicted 0.9096, measured 0.9008, model optimistic by 0.9 points; 95% CI lower bound 0.8831 < target − tol = 0.8900); D increased from 2048 to 2112
  BSDC-S: observed benchmark error (uniform) mean 0.65 pt, worst 2.9 pt over 12 cells
    - BSDC-S: residual bias of the exact-ties model, signed error up to +0.83 points per group (prereg 22)
    - verification: at D = 2560 the selection measure rejected the target (predicted 0.9084, measured 0.8889, model optimistic by 2.0 points; 95% CI lower bound 0.8522 < target − tol = 0.8900); D increased from 2560 to 2816
  FHRR: observed benchmark error (uniform) mean 0.5 pt, worst 1.89 pt over 20 cells
  HRR: observed benchmark error (uniform) mean 0.71 pt, worst 1.39 pt over 12 cells

VERDICT  at least one recommendation is 'validated' and verified; exit 0

NOTE  Predicted = model (abm.sizing), computed from the structure of these triples, no memory built. Measured = accuracy of real memories of our encoding on fresh seeds (only with --verify; a finite sample, CI over seeds). 'validated' only means the graph's structure lies inside the ranges tested in preregistrations 21-22; the observed-error bands are errors seen on those benchmark subgraphs, not guarantees for this graph. All accuracies are means over the (subject, relation) queries.
```

Limits. The predictions are a model, tested only on subgraphs of six public
knowledge graphs (preregistrations 21–22, N 100–1600, 1 hop); "validated" means
only that the graph's structural statistics fall inside the ranges seen there,
and the observed-error bands are benchmark history, not bounds for your graph.
`--verify` measures a finite number of memories of *our* encoding (CI over
seeds; q10 is a crude per-memory quantile), so it is evidence, not a guarantee.
The pre-verification time is a crude estimate (one pilot memory). With
`--export torchhd` the mapping (MAP-B→BSC, MAP-I→MAP, FHRR→FHRR, HRR→HRR, BSDC:
no equivalent, BSBC bundles differently) was checked against torchhd's source,
not run; where torchhd's encoding or precision differs (FHRR float16, BSDC
bundling, storage sizes) the numbers do not transfer.

## Why trust the contract

The exact model is part of the twenty-two preregistrations reported in the paper
(v1.17), whose subject is now the a-priori accuracy estimate for six
vector-symbolic families, with ABM as the case studied most; failures are kept
in the record ([BENCHMARKS.md](BENCHMARKS.md)):

- **Exact model**: 0.27 points of error at a dimension never used before
  (D = 16 384), 0.6–1.1 points on dense subgraphs of two real knowledge
  graphs, zero fitted parameters — but it **failed on dense YAGO3-10**
  (test 22): mean absolute error 4.02 points, signed error −3.5 (optimistic),
  worst cell 7.2, and the dimension chosen in advance missed its target in 4
  of 12 cells; choosing D in advance kept the promise on 80
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
- **Noisy storage does not help.** Under i.i.d. bit flips up to ε = 0.2 a
  store with polar or BCH codes recalled more in 110 of 112 cells, the other two
  being exact ties (test 20).
- **The model fails on at least one dense graph.** On dense YAGO3-10 it was 3.5
  points optimistic on average, 7.2 in the worst cell, and `min_dimension`-style
  sizing missed its target in 4 of 12 cells (test 22); on dense FB15k-237,
  WN18RR and CoDEx-M its mean error was 0.4–0.6 points. Two-hop chain
  predictions on real graphs were falsified (test 22).
- The model assumes independent codewords and facts; even cycles (four facts on
  a rectangle) break it, by up to 6 points on a loaded biclique, a candidate,
  not verified, for the YAGO3-10 failure. Tiny codebooks are out of scope.
- `confidence` is not calibrated; the Law IV contract in `abm.inspector` is
  optimistic on graphs with symmetric relations.

## What ABM is not

Not an embedding store, not a vector-DB replacement, not "a better RAG", and
not denser or faster than an exact store. It is a binary memory with a
parameter-free model of its own accuracy.

## License

MIT
