# A Parameter-Free, Finite-Dimension Resource Theory for Binary Holographic Memory, with Preregistered Tests

*(Algebraic Binary Memory — ABM)*

**Preprint v1.6 — September 2026**
*Normative specification: [FORMALISM.md](FORMALISM.md) (frozen, v2.1).
Reference implementation: [`reference/abm.py`](../reference/abm.py); exact
theory: [`bsm/memory/exact_contract.py`](../bsm/memory/exact_contract.py).
Every number is produced by a script in `examples/` with results committed as
JSON; the five preregistrations are in `docs/preregistration/`.*

## Abstract

We study Algebraic Binary Memory (ABM), a binary vector-symbolic model of the MAP-B family: facts are XOR-bound triples in one majority-vote trace, and reasoning alternates unbinding with cleanup onto a codebook. The O(N log M) scaling of such traces is known; asymptotic laws predict accuracy only up to a fitted constant. We compute cleanup accuracy at finite dimension from the axioms, with no parameter, and extend it to several true answers, aliases, weighted facts, the reference tie rule and the dependence between hops on one trace. Nine preregistered tests, with predictions committed before any data: seven supported their primary hypotheses and two failed — one through the tie rule, which the model had approximated and a later test confirmed; one through off-path recovery in tiny codebooks, which remains unmodelled. At an unmeasured dimension the error was 0.27 points; on dense subgraphs of two real knowledge graphs, 0.6–1.1 points once symmetric relations are recognised as one fact vector of weight 2; under grounding errors, 0.8–1.9 points without calibration. The theory predicted a failure of our own earlier law: hops on one trace are negatively correlated, by −ρ²/(1−ρ²) per bit, and chains fall below p^h, by up to 8.5 standard errors at six hops. It also predicted that the symmetric encoding costs up to 7 points at low load and gains up to 8 at high load, and where the sign changes. The known bound over-provisions dimension about sixfold. The limits are measured too: an exact store beats the trace on ProofWriter, and the trace is smaller than a minimal exact encoding only below about 75–80% accuracy. ABM is not a compressor; it offers accuracy that can be stated, and checked, before deployment.

## 1. Introduction

Retrieval-augmented systems treat memory as an index: a store whose only
operation is similarity search, with all composition delegated to a language
model [@lewis2020rag]. We study the opposite division of labour — a memory whose
*operations themselves* compose knowledge — and ask how much such a memory can
hold and how reliably it answers, as a function of its resources: dimension D,
load N, codebook size M, and reasoning depth h.

In the vector-symbolic taxonomy ABM is a MAP-B architecture
[@clarkson2023capacity], and the order of growth of its capacity is known. What
a deployment needs is not an order of growth but a number: the accuracy of *this*
memory at *this* load. This paper supplies that number without fitted
parameters, and tests it the way such a claim should be tested.

**Contributions.**

1. An **exact finite-D accuracy** for cleanup, derived from the axioms: exact
   majority agreement and binomial distances, extended to several true answers,
   aliases, weighted facts and symmetric twins (§3). It replaces an asymptotic
   law whose single constant, k = 0.92, turns out to be nothing but the error of
   its own approximations.
2. The **exact dependence between hops** on one trace (§4). The independence
   assumed by the composition law Acc(h) = p^h is false; the violation is
   derived and then measured.
3. **Nine preregistered tests** (§6), with predictions, criteria and harnesses
   committed before any run, on synthetic data at an unmeasured dimension, on
   two real knowledge graphs, on grounding errors, on two encodings and on deep
   chains. Two failed; both failures are reported with their causes.
4. An account of **what the model is not** (§7): not a compressor, not better
   than an exact store at small scale, and not free of the faults we found in
   our own earlier versions (§8).

**Methodology.** No claim enters without a quantitative prediction and a test
designed to falsify it. Since v1.4 every new claim is **preregistered**: the
prediction, the success and falsification criteria and the harness are committed
before the run, and the outcome is appended to the same file, whatever it is.
Retired laws stay in the record with the data that killed them (§8).

## 2. The model

**States.** x ∈ 𝔹^D with 𝔹 = {−1,+1}; d_H the Hamming distance; the null
distribution between independent states is Binomial(D, ½).

**Operators** (with their axioms):

- **Binding** x ⊕ y: elementwise product (≡ XOR).
  *A1: an isometric involution* — (x⊕y)⊕y = x, distances preserved.
- **Bundling** ⊞(x₁…x_n): bitwise majority vote, ties broken by a fixed
  pseudo-random vector.
  *A2: majority superposition* — a member agrees with the bundle on each bit
  with probability p_agree(n), independently across bits.
- **Cleanup** over codebook C: argmin_{c∈C} d_H(x, c).
  *A3: idempotent projection* onto C.

**Facts and traces.** A fact (s, r, o) is encoded as f = c_s ⊕ ρ(c_r) ⊕ c_o with
ρ a cyclic shift; a memory is the single trace T = maj(f₁…f_N);
query(s, r) = cleanup(T ⊕ c_s ⊕ ρ(c_r)). Controllers (a Horn chainer, a planner,
an LLM) are programs over this machine; the model is separated from the
pipeline.

**Conventions.** ± is a 95% confidence half-width across seeds unless the text
says SD or range. Differences between accuracies are in percentage points.

### 2.1 The operators are necessary and independent

Ablating each operator kills a disjoint capability set (D = 1024, N = 50,
10 seeds; an earlier 3-seed run gave 85% and 67% for the first two entries of
the full model, with the same zeros):

| variant | relational recall | holographic O(D) | 3-hop | compose | symbols |
|---|---|---|---|---|---|
| full | 82% | ✓ | 80% | ✓ | ✓ |
| −binding | **0%** (content membership survives at 73%) | ✓ | 0% | 0% | ✓ |
| −cleanup | 0% | ✓ | 0% | ✓ (signal 3.6σ, never a symbol) | **0%** |
| −bundling | **100%** | **✗ (50× space)** | 100% | ✓ | ✓ |

Binding carries relational structure; cleanup carries symbolization and depth;
bundling carries superposition — and is the only operator that *costs*
accuracy.

## 3. Capacity

### 3.1 The laws, at a glance

| law | statement | status |
|---|---|---|
| I | null distance Binomial(D, ½) | exact |
| IV | asymptotic capacity N\* = k·2D/(π·z_G(M)²) | scaling correct; the constant k is the error of its approximations (§3.2); superseded for prediction |
| IV-exact | cleanup accuracy from exact majority agreement and binomial distances (§3.2–3.3) | **preregistered** (tests 2–5): every primary hypothesis supported, except dense FB15k-237 at D = 2048 before twins were counted (in part) (§6) |
| V | hops compose as Acc(h) = p^h | **falsified as an exact law**: hops on one trace are negatively correlated, by a derived amount that grows with depth (§4) |
| VI | failure grows with out-degree | **retired**: a load artifact (§8) |
| VI′ | topological neutrality: only load matters | corroborated on synthetic data; on real hubs, residual +3.3 points, neither supported nor falsified |
| VII | redundancy: a fact of weight w counts w² (N_eff = Σw²) | approximate; the exact weighted form is preregistered and removes its saturation error (§3.3) |
| VIII | every end-to-end failure is attributable to one level | principle; supported by the composition test (§6) and by pilots |

### 3.2 From an asymptotic law to an exact one

**Law IV (asymptotic).** The member-trace correlation gives a signal z-score
√(2D/(πN)); retrieval fails when the minimum of M − 1 null distances crosses
it, which for large M sits at the second-order Gumbel threshold
z_G(M) = √(2 ln M) − (ln ln M + ln 4π)/(2√(2 ln M)). The 50% load is then

  N\* = k · 2D / (π · z_G(M)²),  with k = 1 derived.

Measured, k = 0.92 ± 0.03 (range over two sweeps, `examples/k_from_results.py`):
0.943 ± 0.005 (SD, 10 seeds) over a 36× codebook range at D = 2048, and
0.914 ± 0.025 over D ∈ [512, 4096], lower at small D. The scaling was posed as a
prediction and held (a linear model N\* = cD is rejected: c drifts from 0.098
to 0.068 over D ∈ [512, 4096]). But k is a fitted constant, and it drifts.

**The exact accuracy.** Both approximations in Law IV — a Gaussian signal and a
Gumbel extreme — can be removed. By A2, each bit of the query agrees with the
codeword of a stored object with probability exactly

  p_agree(N) = P( 1 + Σ_{j=2}^{N} x_j > 0 ) + ½·P( 1 + Σ_{j=2}^{N} x_j = 0 ),

the x_j independent Rademacher variables of the other facts. The distance to the
true codeword is therefore Binomial(D, 1 − p_agree(N)), each of the M − 1 null
distances is Binomial(D, ½), and the probability that cleanup returns the true
object is a finite sum over these distributions. There is no parameter.

What "exact" means here, precisely: for independent random codewords and facts,
the probability is computed from the exact discrete distributions, with no
Gaussian or extreme-value approximation. Ties need care. The reference returns
the *first* codeword inserted among those at minimal distance, so a target with
n_b null codewords inserted before it and n_a after wins iff

  P(win | d) = P(null > d)^{n_b} · P(null ≥ d)^{n_a},

which is exact given independent null distances (`win_ordered`). Tests 1–6 used
an even split of ties instead — its average over random positions — which is
accurate where ties are rare or positions mixed; test 8 found where it is not
(§6). The extensions of §3.3 add one assumption each, stated there: that
candidates at equal signal have independent distances.

Against the capacity data behind k, the exact N\* is 50.5 / 87.5 / 155.5 / 277.5
at D = 512 / 1024 / 2048 / 4096, measured 50.2 ± 4.5 / 87.1 ± 3.1 / 158.8 ± 5.3 /
280.0 ± 6.4: inside every interval, including the small D where k drifted. **k =
0.92 is the error of the Gaussian and Gumbel approximations, not a property of
the memory.** Against a Monte Carlo of the reference implementation at D = 256,
N = 20, the exact model gives 0.722 for a measured 0.714, where Law IV gives
0.792; the test suite requires the exact model within 3 standard errors *and*
Law IV outside them. These checks use data we had already seen; the tests of §6
use data we had not (Fig. 1).

![Capacity: measured 50%-accuracy load N\* (10 seeds, 95% CI) against the exact
model (no parameter) and Law IV with k = 1 and with the fitted k = 0.92. The
codebook grows with the load (M = 2N + 11).](figures/fig1_capacity.png){width=60%}

### 3.3 Several answers, aliases, weights and twins

The same construction extends without new parameters.

- **Several true objects.** If (s, r) has g stored objects, each has the signal
  of a stored fact, and the query is correct if cleanup returns any of them. The
  exact model takes the minimum over g signal distances, treated as independent.
- **Aliases.** The encoding is symmetric in s and o, so a stored (x, r, s) gives x
  the same signal as a true answer to (s, r). Aliases join the signal candidates
  and win their share of ties.
- **Weights (Law VII, exact).** A fact written w times weighs w in the vote. The
  agreement of a fact of weight w is P(w + Σ_j w_j x_j > 0) + ½P(… = 0), computed
  exactly by convolving the weighted sums; Law VII's N_eff = Σw² is its Gaussian
  approximation.
- **Symmetric twins.** Because the encoding is symmetric, (s, r, o) and (o, r, s)
  are **the same vector**. A symmetric relation stored in both directions is one
  fact of weight 2, not two facts. In FB15k-237 this concerns 12.5% of the
  triples, in WN18RR 34.2%.

Each extension was tested before being used for a claim (§6).

**Encoding symmetry is a trade-off, not a flaw.** An asymmetric encoding,
s ⊕ ρ(r) ⊕ ρ²(o), has neither aliases nor twins, at the cost of the inverse
queries the symmetric one answers for free. The model predicts both without
parameters, and predicts that neither wins: at low load the aliases of the
symmetric encoding cost accuracy, at high load its twins — two facts fused into
one of weight 2 — reduce the noise and gain it. Test 6 confirmed this on dense
WN18RR, with the sign of the difference right in 6 cells of 6, from −7.0 to +7.8
points (Fig. 2). The model can say, for a given graph, which encoding to choose
at a given load, before storing anything.

![Symmetric minus asymmetric encoding, dense subgraphs (preregistered): measured
(points) against the prediction (lines). Circles: WN18RR; triangles:
FB15k-237.](figures/fig10_asymmetric.png){width=60%}

### 3.4 Other capacity results

**Typed projection (P3).** Restricting cleanup to a typed sub-codebook S raises
capacity; with the exact model the gain is the ratio of the two exact N\*. The
original experiment is lost (§8); its reconstruction was preregistered (§6).

**Robustness asymmetry.** Flipping a fraction ε of trace bits degrades gracefully
(86% → 42% at ε = 20%); corrupting the codebook is catastrophic (ε = 5% halves
accuracy), because key construction, cleanup targets and stored content corrupt
together. The item memory is the trusted base of the model.

**Cleanup cannot be sublinear in this regime.** A metric-tree index visits
4 487 of 5 000 nodes at D = 2048: a query sits at 0.446–0.483·D from its *own*
codeword and at 0.500·D from every other, a gap of 34–111 bits that no triangle
bound or banded LSH can exploit. Cleanup is linear with a low constant (6.1 ms at
M = 10⁵, bandwidth-bound).

**Acceptance must be an extreme-value test in M.** A fixed confidence threshold
of 0.75 accepted 0 of 30 correct answers on a memory predicted at 1.00. Cleanup
reports a maximum of M z-scores, so the threshold must grow with M: accepting iff
z ≥ Φ⁻¹((1 − α)^{1/M}) caps false accepts at α (measured 30/30 true accepts, 3/300
false at α = 0.01).

## 4. Composition

### 4.1 Hops on one trace are not independent

Law V states that a chain of h hops, with cleanup between them, succeeds with
probability p^h. Its argument needs the hops' successes to be independent, which
earlier versions of this paper measured (φ = +0.014 ± 0.024) but did not prove.
They are not independent. For two facts f₁, f₂ queried by consecutive hops, on
every bit

  (T·f₁)·(T·f₂) = f₁·f₂,  because T² = 1,

and f₁·f₂ is a fair bit independent of the vote. The agreements of the query with
the two targets therefore have covariance −ρ², where ρ = 2·p_agree(N) − 1, and
correlation

  **−ρ² / (1 − ρ²) ≈ −2/(πN)**.

The identity is elementary, and it may well be known in the analysis of
Boolean functions, where the majority function is a central object
[@odonnell2014boolean]; we have not found it stated for the decoding of
vector-symbolic bundles, and earlier versions of this paper assumed the
opposite.

Conditioning on the number of bits where f₁ = f₂ gives the exact joint
distribution of the two distances, and hence the exact probability that both hops
succeed (`two_hop_joint_fast`; the null distances of the two hops are treated as
independent, their correlation being a sum of random signs of order 1/√D). The
prediction: P(both) < p², by an amount that matters only at small N. The old
measurement could not see it — its error on the per-bit correlation was ±0.033,
the effect at its N = 90 is −0.007. The preregistered test (§6) could, and did:
Law V is falsified as an exact law, by the amount derived (Fig. 3). As an
approximation it remains good at large N: in the depth experiment below
(N = 120), |Acc − p^h| = 0.007 over h ≤ 24.

![Per-bit correlation between the agreements of one trace with two queried facts:
exact −ρ²/(1−ρ²) (steps at small N are the parity of the majority vote: N and
N − 1 agree when N is even) against measurement (±3 SE), and the independence
Law V assumes.](figures/fig9_dependence.png){width=60%}

**Deep chains.** For h hops the chain's fact bits are still independent fair
signs (the entities are distinct), the other facts sum to a binomial, and the
trace is the sign of the total; `chain_accuracy_mc` samples this per-bit model
directly, without memories or codewords, and matches the exact two-hop formula.
It predicts that the shortfall below p^h *grows with depth*. Testing that took
three preregistrations. The first failed: with a codebook of about twenty
entries, a failed hop often lands on the right entity by chance and the chain
recovers, a term (of order h/M) that the prediction ignored. The second, with
1 000 distractors to make recovery negligible, failed at a single hop: the
distractors, inserted after the target, lose every tie to it under the
reference's tie rule, which the model had approximated by an even split. The
third, with the exact tie rule and new data at two dimensions, held: all ten
cells within 3 standard errors of the model, and p^h rejected at 4.3–8.5 SE at
four and six hops (Fig. 4).

![Deep chains (preregistered): chain accuracy minus p^h, measured (points)
against the per-bit model with the exact tie rule (lines), N = 12, 1 000
distractors.](figures/fig11_deepchain.png){width=60%}

### 4.2 The Memory Calculus

Terms e ::= a | 1 | e⊕e | ρ(e) | ⊞(e…) | cleanup(e), in an *exact fragment*
(⊕, ρ) and a *probabilistic fragment* (⊞, cleanup).

- **Normal form (exact fragment).** (E, ⊕, 1) is the free Boolean group — a
  vector space over GF(2) — on ρ-stratified atoms, so every exact term has the
  unique normal form "atoms of odd multiplicity". The fact is elementary; the
  calculus is built on it.
- **Compose is not a rule.** For facts sharing a bridge,
  nf(f₁ ⊕ f₂) = c_A ⊕ ρr₁ ⊕ ρr₂ ⊕ c_C: the two-hop fact is generated by
  normalization, and since c_B ⊕ c_B = 1 it contains no information about the
  bridge (**Bridge Elimination**).
- **No-go.** T ⊕ T = 1: reusing a raw decode, which contains T, inside a later key
  on the same trace collapses deterministically. Depth requires symbolization.
- **Cost semantics.** Exact steps are free and certain; ⊞ spends capacity (§3);
  cleanup spends reliability. Composing reliabilities multiplicatively is sound
  only up to the hop dependence of §4.1, which is now a known term rather than an
  assumption.

## 5. Predictions made before measurement

**P1 — depth is cheap.** From Laws IV and V, D_min(h) = Θ(N ln M) + Θ(N ln h).
At constant load N = 120, target chain accuracy 95%, 10 seeds, geometric grid in
10% steps:

| h | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| D_min | 3846 | 4402 | 5277 | 5528 | 6248 | 7017 | 7697 |

Growth 1 → 64 hops: 2.0× (logarithmic fit R² = 0.992, linear 0.798). The
prediction is modest — logarithmic growth follows from p^h and a Gaussian tail —
and the measurement adds that the constant is small. Load, not depth, is the
limiting resource (Fig. 5).

![P1. Minimum D for chain accuracy ≥ 95% at N = 120 (mean ± SD, 10 seeds, 10%
grid).](figures/fig2_depth.png){width=60%}

**P2 — the compiler.** Composing facts exactly into a second trace before
decoding turns two-cleanup queries into one-cleanup queries at the compiled
trace's load. Measured: 99% vs 82% (40 chains), 89% vs 25% (80), 44% vs 3%
(160); the interpreted path tracks p² (Fig. 6).

![P2. Two-hop queries answered by two cleanups (interpreted) or one cleanup on a
compiled trace T₂; dotted: p².](figures/fig4_compiler.png){width=60%}

**P3 — typed projection.** Reconstructed and preregistered (§6): measured gains
1.84 / 2.21 / 2.63× over 2 000 / 8 000 / 32 000 distractors, exact prediction
1.86 / 2.26 / 2.75×. The Gumbel-ratio prediction of earlier versions
(1.94 / 2.37 / 2.82×) overestimated more; the "consistent 5% bias" we reported
then was that approximation's error.

**Compose cost.** Composition by two independent cleanups costs p²: measured
0.77 / 0.23 / 0.01 against 0.77 / 0.21 / 0.02 at three loads (10 seeds).

## 6. Preregistered tests

Each test below was committed — prediction, success and falsification criteria,
harness — before its run; the commit hashes are in the files. Outcomes are
evaluated only with those criteria, and appended to the same file.

| # | file | what is tested | outcome |
|---|---|---|---|
| 1 | `fb15k237.md` | Law IV (k = 0.92) on FB15k-237 [@toutanova2015observed], uniform samples | primary **supported**: 1.05 points (signed −0.47); hubs neither supported nor falsified (+5.5) |
| 2 | `exact_contract.md` | exact model: D = 16 384; several true objects; FB15k-237 dense subgraphs | 5 hypotheses: 3 **supported**, 1 **in part**, 1 neither |
| 3 | `dependence.md` | the hop dependence of §4.1; P3 reconstructed | all **supported**; Law V rejected at 6 SE |
| 4 | `twins.md` | symmetric twins; WN18RR; exact Law VII | all five **supported** |
| 5 | `composition.md` | grounding × reasoning without calibration | all **supported** |
| 6 | `asymmetric.md` | symmetric against asymmetric encoding | all **supported**; sign of the difference right in 6/6 |
| 7 | `deepchain.md` | hop dependence at depth, tiny codebook | **falsified**: off-path recovery, not modelled |
| 8 | `deepchain2.md` | the same with 1 000 distractors | **falsified** at one hop: the tie rule |
| 9 | `deepchain3.md` | the same with the exact tie rule, two dimensions | all **supported**; p^h rejected at 4.3–8.5 SE |

**Test 2 — the exact model on new configurations.** At D = 16 384, a dimension no
experiment had used, the mean absolute error over six loads was **0.27 points**
(signed −0.09; criterion ≤ 2). With g = 1, 2 and 4 true objects per query it was
0.58, 0.86 and 0.32 points, where Law IV, which assumes one target, erred by 0.6,
13 and 21. On dense subgraphs of FB15k-237 (a breadth-first sample, so that
entities recur) it was 1.06 points at D = 8192, and 2.88 at D = 2048 with a
signed bias of −2.61, outside the ±2 we had set: **supported in part**. The exact
model was *pessimistic* on dense real graphs, most at high load (−8.4 at
N = 400).

**Test 4 — symmetric twins.** Looking for the cause, after test 2 and on its data,
we found that in the dense samples 6–14% of the triples had their symmetric twin
in the sample — two directions of one relation, and so one vector of weight 2
(§3.3) — against 0% in uniform samples. Counting twins as weight 2 removed the
bias on those cells. Because that fit was made after seeing the data, it was
tested on data we had not seen: FB15k-237 with new seeds, and a second graph,
WN18RR, where a quarter of the triples in dense samples are twins. With twins the
error was **0.57 and 0.60 points** on WN18RR (signed −0.16 and −0.36); without
them, the model was pessimistic by −2.1 and −3.8 on average and by up to −8.9 in
a cell; on uniform samples, where twins are rare (0.1–0.3%), both versions agree
(0.9 and 1.1 points), so the term does not improve everything by chance (Fig. 7).
The exact weighted form of Law VII, on six synthetic weight configurations, erred
by at most 1.4 points; with two facts of weight 14 the N_eff approximation erred
by 2.7 and the exact form by 0.5.

![Symmetric twins on dense WN18RR subgraphs: measurement against the exact model
counting twins as one fact of weight 2 (solid) and not counting them
(dotted).](figures/fig8_twins.png){width=60%}

**Test 3 — the hop dependence.** Per-bit correlation between two hops, predicted
−0.0645 / −0.0213 / −0.0071 at N = 10 / 30 / 90, measured −0.0634 / −0.0212 /
−0.0072, within 1.5 SE and 31–91 SE away from zero. On events, the correlation
between the successes of two chained hops was φ = −0.030 at N = 10 (predicted
−0.032) and −0.016 at N = 30 (predicted −0.011), within 3 SE; independence is
rejected at 6.0 SE.

**Test 5 — composition without calibration.** Grounding errors of four kinds, at
rates up to 50%, on two-hop chains: the prediction — the share of intact chains
times the exact two-hop probability at the load actually stored — erred by 0.75
(missing facts), 1.89 (wrong relation), 1.01 (wrong entity) and 1.18 (spurious)
points. For missing facts the calibrated predictor of earlier versions erred by
10.2 points; a lighter trace helps more than (1 − ε)² accounts for, and the exact
model captures it by computing at the load that remains (Fig. 8). Under three
error structures with the same mean rate the error was 1.1–1.2 points, and the
predicted order held at ε = 0.4: errors concentrated on whole chains (28.3%) cost
less than independent ones (18.0%), and errors concentrated on one hop (10.0%)
cost more.

![Composition under four kinds of extraction error (preregistered): measurement
(points) against the prediction without calibration
(lines).](figures/fig6_robustness.png){width=65%}

**Tests 6–9** are described in §3.3 (encoding symmetry) and §4.1 (deep chains).
Tests 7 and 8 are the two failures of this paper's own predictions; each is kept
with its data in `docs/preregistration/`, and each changed the model: test 8
replaced the even split of ties with the exact rule, test 7 marks a regime — tiny
codebooks — where the model does not yet apply.

**Test 1**, the earliest, used Law IV with k = 0.92 on uniform samples of
FB15k-237 and passed (Fig. 9); applied afterwards to the same data, the exact
model gives 0.88 points.

![Test 1, FB15k-237 uniform samples: measurement against the preregistered Law IV
prediction (dashed).](figures/fig7_fb15k237.png){width=60%}

**How much margin.** The criteria compare mean errors with thresholds; they do
not say how far below the threshold a result sits, or whether a mean error is
distinguishable from sampling noise. `examples/prereg_summary.py` adds both,
resampling seeds within cells (2 000 bootstrap draws). Two findings. First, most
errors of the final predictor are **at or below the noise floor** — the mean
|error| a perfect model would show, √(2/π) times each cell's standard error: at
D = 16 384, 0.27 against a floor of 0.70; on dense WN18RR with twins, 0.57 and
0.60 against 1.31; for composition, 0.75–1.24 against 1.2–1.4. There the model
cannot be told apart from the truth with these data. Second, **four of the
seventeen final summaries carry a signed bias whose 95% interval excludes zero**:
dense FB15k-237 at D = 2048 (−0.81), uniform WN18RR at D = 2048 (−0.67) and 8192
(+1.11), and wrong-relation errors (+1.34). All are below 1.4 points, and the two
WN18RR biases have opposite signs, so they do not point to one missing term; the
wrong-relation bias is unexplained.

**What the tests leave open.** The exact model treats candidates at equal signal,
and the null distances of different hops, as independent; test 2 puts the cost
of the first below one point. On real hubs a residual of +3.3 points remains
(test 2), and dense samples favour the neighbourhood of their starting entity.
All tests use queries that are either synthetic or the triples themselves; none
uses questions written by people.

## 7. What the model is, and is not

**Against an exact store.** On ProofWriter [@tafjord2021proofwriter]
(open-world attribute fragment, 10 seeds), a forward chainer whose only truth
oracle is one Hamming distance to the trace reaches 99.8% ± 0.3, 99.1% ± 0.3 and
92.4% ± 1.4 at depths 0, 2 and 5 on the problems four grammatical patterns parse
(35–55% of them, so the subset is selected; majority baseline 42%). The same
chainer with a Python set in place of the trace scores **100% at every depth**:
the parser and the chaining are sound, and the whole loss is the oracle's. The
trace takes 512 bytes where a minimal exact encoding takes 8–23 (Fig. 10). The
oracle's z ≥ 3 threshold is the M = 1 case of §3.4 and controls false accepts per
test, not per proof.

![ProofWriter, parsable subset: accuracy by depth (10 seeds, 95% CI) against the
majority baseline.](figures/fig3_proofwriter.png){width=60%}

**Not a compressor.** Holding N facts at single-query accuracy a needs about
π·(z_G(M) + Φ⁻¹(a))²/(2k) bits per fact, while a minimal exact encoding of
(s, r, o) needs 2·log₂V + log₂R. The trace is the smaller of the two only below
a = 74% (M = 100) to 81% (M = 10⁵); at 95% it is 1.5–1.8 times larger. What it
offers instead is a fixed size, membership by one distance, exact algebraic
composition (§4.2) and — the subject of this paper — a degradation that can be
computed before deployment.

**The known bound over-provisions.** Clarkson, Ubaru and Yang's Theorem 16
[@clarkson2023capacity] gives, from its proof, a sufficient dimension
m_C = 56·n·ln(2d/δ) for a threshold membership test over d items. On their own
task (d = 500, δ = 0.1; `examples/clarkson_comparison.py`) m_C exceeds the
measured minimum 5.7–6.3 times, and 5.7–6.9 times with exactly their threshold,
while exact constants match the measurement within a 10% grid. Their theorem
proves a scaling and is correct as stated; a contract needs the dimension itself.

**Capacity contract.** At 1 KB (D = 8192), with 10 fresh seeds, the exact model is
within 0.4 points of measurement on average, Law IV with k = 0.92 within 0.3, and
Law IV with k = 1 off by 3.2, always optimistic. The line "up to 300 facts at
≥ 85% in 1 KB" that an earlier version printed fails its own measurement (84.6%);
a minimal exact encoding of the same facts fits about 360 of them in that kilobyte, at 100% (Fig. 11).

![Capacity contract at 1 KB (D = 8192), 10 seeds: exact model, Law IV with
k = 0.92 and with k = 1, against measurement.](figures/fig5_contract.png){width=60%}

**Pilots.** On HotpotQA [@yang2018hotpotqa] a regex grounding layer never engaged
the algebra (1.9 triples per 42 sentences), and multi-hop heuristics subtracted
value (7% against a 13% single-hop baseline): a failure of grounding, not of the
algebra. An LLM extractor restored 10/10 on the same questions, but that
extractor had seen them (n = 10; a feasibility sample only). A synthetic
end-to-end pilot (500 sentences written by us, a template extractor with
deliberate gaps, D = 16 384) violated its first contract by 47 points because the
audit ignored the s/o symmetry; the audit was changed after the violation, and
the second contract, issued before any query, was 9.8 points off, inside its
±15. On one real document (a six-page Italian contract, 14 facts, a local 4B
model) level ablation placed every failure in extraction or planning, never in
the algebra; at that load, under 5% of capacity, any model predicts
near-perfect recall, so the pilot supports Law VIII and not the capacity theory.
It is a case, not a sample.

## 8. Falsifications and corrections kept on record

| retired or corrected claim | what showed it |
|---|---|
| Law VI: failure grows with out-degree | at constant load, accuracy is flat (64–73%) for out-degree 1–24; the effect was load |
| geometric decay of raw chaining, z_eff = √D·ρ^h | two-trace raw chaining measures 0% where it predicts ~40%; T ⊕ T = 1 is the sharp form |
| **Law V as an exact law** (hops independent) | preregistered test 3: correlation −ρ²/(1−ρ²) per bit, independence rejected at 6 SE |
| k = 0.92 as a constant of the memory | the exact model reproduces the capacity data with no constant (§3.2) |
| P3's "consistent 5% bias, unexplained" | the Gumbel-ratio approximation; the exact model predicts the reconstructed gains within 1–4% |
| the composition law as parameter-free (v1.4 and before) | its script took the clean accuracy from the measurements and calibrated the spurious-fact curve; replaced by a preregistered prediction without calibration (test 5) |
| "~3% mean deviation" for the composition law | the script's own data gave 4.3 (3 seeds) and 3.6 (10), and 10.2 for missing facts |
| sublinear cleanup; a fixed acceptance threshold | §3.4 |
| our deep-chain prediction with a tiny codebook | preregistered test 7: off-path recovery dominates, up to +18.9 SE |
| the even split of ties | preregistered test 8: with 1 000 distractors after the target, it misses a single hop by 1.8 points (13.8 SE); replaced by the exact rule |

**Lost scripts.** Four result files had no script that produced them:
`independence_results.json` and `projection_results.json` (replaced by test 3),
`conjecture7_results.json` (Law VII; replaced by test 4) and
`composition_stress_results.json` (replaced by test 5). The claims of earlier
versions that rested on them are superseded by the preregistered tests, whose
harnesses are committed.

**Not yet re-tested.** The compiler-ranking dry run (three simulated extractors
ranked by a per-query contract before any query, 2/2 resolvable pairs at 95% CI)
and the cross-domain invariance results (four synthetic topologies, mean
deviation 2.4%; an exact 1/g aliasing factor for single-relation chains,
corroborated at g ∈ {2, 3, 4, 8}) predate the exact model and were computed with
Law IV; they are consistent with it but have not been re-run.

## 9. Related work

Vector-symbolic architectures and hyperdimensional computing
[@kanerva1988sdm; @kanerva2009hd; @gayler2003jackendoff; @kleyko2022survey]
supply the operators: Plate's holographic reduced representations
[@plate1995hrr], Gallant and Okaywe's matrix binding [@gallant2013objects],
Rachkovskij and Kussul's context-dependent thinning [@rachkovskij2001thinning].
Capacity of the D/ln M form is classical. For MAP-B, Clarkson, Ubaru and Yang
[@clarkson2023capacity] prove O(n log(d/δ)) bounds for membership in majority
bundles, including bundles of bindings; Thomas, Dasgupta and Rosing
[@thomas2021theoretical] give a broader theory of hyperdimensional computing;
Frady, Kleyko and Sommer [@frady2018sequence] derive retrieval accuracy from
crosstalk noise in VSA-coded recurrent networks. Our contribution relative to
them is narrower than a new scaling: exact finite-D accuracy without parameters,
its extension to several answers, weights and hop dependence, and preregistered
tests on real graphs. Clarkson et al. also prove that the reliability of nested
bundling decays with depth (their Lemma 17); the hop dependence of §4.1 concerns
a different composition — cleanup between hops on one flat bundle.

Hopfield networks [@hopfield1982] share the interference-plus-extreme-value
mechanism, with the 0.138·N regime of Amit, Gutfreund and Sompolinsky
[@amit1985storing]. Holographic embeddings of knowledge graphs
[@nickel2016hole] store (s, r, o) triples in holographic vectors but learn them
for link prediction; ABM learns nothing and stores the facts themselves.
ProofWriter [@tafjord2021proofwriter] and RuleTaker [@clark2020ruletaker] study
soft theorem proving with transformers; our use of ProofWriter keeps the chaining
symbolic and makes only the truth oracle algebraic. The rewriting results of §4.2
are standard [@baader1998term].

## 10. Open problems

1. **Real hubs.** A residual of +3.3 points on high-degree subjects (test 2) is
   not explained by twins alone.
2. **Joint independence of candidates.** The exact model treats candidates at
   equal signal, and the null distances of consecutive hops, as independent;
   test 2 bounds the first cost below a point, the second is unmeasured.
3. **Off-path recovery.** In tiny codebooks a failed hop can land on the right
   entity by chance; a uniform 1/M recovery overestimates it (test 7). No clean
   model yet.
4. **Questions written by people.** Every test here queries stored triples or
   synthetic chains.
5. **Negation and quantifiers** in the algebraic truth oracle.
6. **Distributional confluence** for nested probabilistic redexes.
7. **The ABM complexity class** (polynomial D, O(1) controller).

## 11. Conclusion

ABM is not a better retriever, not a compressor, and not better than an exact
store at the scales we measured, and we say so with measurements. It is a memory
whose accuracy can be computed, exactly and without fitted parameters, from its
dimension, its load, its codebook and the structure of what it stores — including
structure that surprised us, like symmetric relations collapsing into single
facts. Nine preregistered tests put that claim at risk, on synthetic data, on
two real knowledge graphs, under grounding errors, on two encodings and on deep
chains; seven supported it, two failed and changed the model, and one falsified a
law of our own earlier versions by the amount the theory predicted. The strongest
evidence for the theory is not that it fits: it is that its predictions were
fixed before the data, and that where it failed, the failure was found, explained
and tested again.

---

*Reproducibility: every number is produced by a script in `examples/` with
results committed as JSON; results from 10-seed reruns sit beside the originals
with an `_s10` suffix; the five preregistrations, with their outcomes, are in
`docs/preregistration/`. The frozen reference implementation
(`reference/abm.py`, 257 lines, numpy-only, deterministic) and the exact theory
(`bsm/memory/exact_contract.py`, numpy-only) are covered by the test suite, which
runs in continuous integration on Linux x86-64 (Python 3.10–3.13, NumPy
1.24–2.5), macOS arm64 and Windows x86-64, with codewords checked bit-identical
across the three. Figures: `examples/make_figures.py`.*

## References
