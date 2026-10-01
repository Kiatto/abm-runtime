# ABM benchmarks

Every number on this page comes from a committed result file, which a committed
script produced. Eleven of the tests were **preregistered**: their predictions
and pass criteria were committed before any data was looked at
(`docs/preregistration/`). Two of the eleven failed. They are kept below with
their causes.

What is measured is not task accuracy against other systems. It is how well the
model **predicts, in advance, the accuracy of the memory**: predicted minus
measured, in percentage points. A small error means the contract can be trusted
before the memory is deployed.

All of it reruns from a clean clone with one command (see
[Reproducing](#reproducing)). On 2026-10-01 every entry on this page reproduced:
twelve byte for byte, and two (tests 4 and 10) up to the last digit of one float.

---

## Headline

| | result | test |
|---|---|---|
| Prediction error at a dimension never used before (D = 16 384) | **0.27 points** (signed −0.09) | 2 |
| Dense subgraphs of two real knowledge graphs, symmetric twins counted | **0.6–1.1 points** | 2, 4 |
| Under grounding errors of four kinds, up to 50%, no calibration | **0.75–1.89 points** | 5 |
| Choosing D in advance on 80 unseen subgraphs | promise kept in every group (measured − target: +0.01 to +1.68) | 10 |
| Same, with the asymptotic Law IV that the package used to expose | WN18RR: 2.8–6.2 points below target, up to 12/20 subgraphs more than 6 below | 10 |
| Questions written by people (SimpleQuestions ∩ FB15k-237), memory level | 0.807 measured against 0.802 predicted (+0.3 SE) | 11 |
| Fitted parameters in the model | **0** | — |

The model is exact under stated idealisations: independent random codewords, and
facts independent over GF(2). Where those fail, so does the model. See
[Limits](#limits-stated-plainly).

---

## The eleven preregistered tests

| # | file | what it tests | outcome | result file | `replicate.py` name |
|---|---|---|---|---|---|
| 1 | `fb15k237.md` | Law IV (k = 0.92) on uniform FB15k-237 samples | H1 supported (1.05 points); H2 undecided | `fb15k237_prereg_results.json` | `fb15k237` |
| 2 | `exact_contract.md` | the exact model, 0 parameters, D = 16 384, several answers, dense FB15k-237 | supported **in part**: dense D = 2048 bias −2.61, outside ±2 | `exact_prereg_results.json` | `exact_contract` |
| 3 | `dependence.md` | correlation between two hops | supported; hop independence rejected at 6.0 SE | `dependence_prereg_results.json` | `dependence` |
| 4 | `twins.md` | symmetric twins as weight-2 facts; WN18RR; exact Law VII | all five supported | `twins_prereg_results.json` | `twins` |
| 5 | `composition.md` | grounding errors × reasoning, no calibration | all supported | `composition_prereg_results.json` | `composition` |
| 6 | `asymmetric.md` | symmetric vs asymmetric encoding | all supported; sign right in 6/6 | `asymmetric_prereg_results.json` | `asymmetric` |
| 7 | `deepchain.md` | hop dependence at depth, tiny codebook | **falsified**: off-path recovery, not modelled | `deepchain_prereg_results.json` | `deepchain` |
| 8 | `deepchain2.md` | the same with 1 000 distractors | **falsified** at one hop: the tie rule | `deepchain2_prereg_results.json` | `deepchain2` |
| 9 | `deepchain3.md` | the same with the exact tie rule | supported; p^h rejected at 4.3–8.5 SE | `deepchain3_prereg_results.json` | `deepchain3` |
| 10 | `sizing.md` | choosing D in advance; the alias ceiling | all supported; Law IV misses on WN18RR | `sizing_prereg_results.json` | `sizing` |
| 11 | `human_questions.md` | human-written questions through a local LLM front-end | memory level supported; end-to-end **in part** (+7.2 points, inside its interval, beyond the 5 set) | `human_questions_prereg_results.json` | `human_questions` |

Result files are in `results/`, harnesses in `examples/<file stem>_prereg.py`.
Summary errors with confidence intervals: `results/prereg_summary_results.json`,
from `examples/prereg_summary.py`.

### Selected numbers

**Several true answers per query (test 2).** With g = 1, 2 and 4 true objects the
exact model erred by 0.58, 0.86 and 0.32 points. Law IV, which assumes a single
target, erred by 0.6, 13 and 21.

**Symmetric twins (test 4).** On dense WN18RR subgraphs, where a quarter of the
triples have their reverse twin in the sample, counting twins as weight 2 gives
0.57 and 0.60 points. Ignoring them gives −2.1 and −3.8 on average, and −8.9 in
one cell. On uniform samples, where twins are rare, the two versions agree.

**Hop dependence (test 3).** Per-bit correlation predicted −0.0645 / −0.0213 /
−0.0071 at N = 10 / 30 / 90, measured −0.0634 / −0.0212 / −0.0072, all within
1.5 SE.

**Composition (test 5).** Errors of 0.75 (missing facts), 1.89 (wrong relation),
1.01 (wrong entity) and 1.18 (spurious facts) points. For missing facts, the
calibrated predictor of earlier versions erred by 10.2.

**Human questions (test 11).** The contract issued before the 643 test questions
predicted 0.289 [0.219, 0.367] end to end. The measurement was 0.361. The memory
held its prediction; the front end was underestimated by the audit (0.36 against
0.41 on test), and front end and memory are not independent.

---

## Other results with scripts

| what | result | script | result file | `replicate.py` name |
|---|---|---|---|---|
| Capacity N\* at 50% accuracy, D = 512–4096, 10 seeds | exact model inside every interval | `capacity_seed10.py` | `capacity_seed10_rerun.json` | `seed10` |
| Comparison with the bound of Clarkson et al. | see paper §7 | `clarkson_comparison.py` | `clarkson_comparison_results.json` | `clarkson` |
| ProofWriter, parsable subset, 150 problems per depth | 92–100% at depths 0, 1, 2, 3 and 5 | `proofwriter_eval.py 150` | `proofwriter_results.json` | `proofwriter` |
| Self-loops, effect of the 2026-09-30 model fix (exploratory) | see below | `selfloop_impact.py` | `selfloop_impact_results.json` | — |

**ProofWriter, against an exact store.** On the open-world attribute fragment, the
forward chainer whose only truth oracle is a Hamming distance to the trace reaches
99.8%, 99.1% and 92.4% at depths 0, 2 and 5 (10 seeds). The same chainer with a
Python set in place of the trace scores **100% at every depth**. ABM loses to an
exact store, and the trace takes 512 bytes where a minimal exact encoding takes
8–23. Only 35–55% of the problems parse, so the subset is selected.

**Self-loops.** FB15k-237 train has 1 625 self-loops (0.6%), which an earlier
audit had counted as zero. The model has handled them since 2026-09-30, after
tests 4, 6, 10 and 11 had been computed. Recomputed on the same samples, the mean
bias of test 4 moves by under 0.1 points. One WN18RR cell had been predicted at
97.6% against 80.0% measured; the corrected model gives 80.2%. No dimension chosen
in test 10 changes. Test 6 has **not** been recomputed.

---

## Limits, stated plainly

- **ABM is not a compressor.** An exact store beats it on ProofWriter, and on
  bytes per fact.
- **Even cycles break the independence assumption.** Four facts on a rectangle
  XOR to the identity. On a loaded biclique the model is up to 6 points
  optimistic. This is not modelled yet.
- **Tiny codebooks are out of scope.** Off-path recovery dominates there (test 7,
  falsified).
- **Dense real graphs at low D and high load:** the model was pessimistic by up
  to 8 points before twins were counted, and residual bias remains on hubs.
- **Small samples.** Cells use at most 200 queries. The minimum detectable bias is
  about 3–4 points in the noisy cells (audit 2026-09-30), and pass criteria of
  ±2–3 points are only 1.5–3 times that floor.
- **No external replication.** Every number here was produced by the author.
  `replicate.py` makes a rerun easy, but nobody else has run it yet.
- **No real documents.** Facts are synthetic or come from knowledge-graph dumps.
  None was extracted by an LLM reading real documents; test 11 uses an LLM only
  to choose the relation of a question.

---

## Reproducing

```bash
git clone https://github.com/Kiatto/abm-runtime && cd abm-runtime
pip install -e ".[replicate]"            # numpy, plus pyarrow for ProofWriter
python examples/replicate.py --list      # names and indicative times
python examples/replicate.py fb15k237 seed10 clarkson proofwriter human_questions
python examples/replicate.py --all       # everything, about four hours on 12 cores
```

The script downloads the public data (FB15k-237, WN18RR, ProofWriter,
SimpleQuestions v2) and checks its sha256. It then runs each harness **on the
commit that recorded its results**, writes the rerun to `results/replica/`, and
compares it with the published file. It reports `IDENTICA`, `IDENTICA a meno di
1e-12 relativo` (identical up to float rounding) or `DIVERSA`. `--code current`
runs today's code instead. Since the self-loop fix, tests 4, 6 and 10 crash
under it, and test 11 moves by 0.13 points.

Test 11 needs a local language model. `replicate.py` reruns only its deterministic
part: the exact-model prediction, and the memory's answer to the true
(subject, relation).

Speed is not benchmarked here. Measurements of the runtime on one machine, with
their spreads, are in [PERFORMANCE.md](PERFORMANCE.md).
