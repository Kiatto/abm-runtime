# A Parameter-Free, Finite-Dimension Resource Theory for Binary Holographic Memory, with Preregistered Tests

*(Algebraic Binary Memory — ABM)*

**Preprint v1.13 — October 2026**
*Normative specification: [FORMALISM.md](FORMALISM.md) (frozen, v2.1).
Reference implementation: [`reference/abm.py`](../reference/abm.py); exact
theory, shipped as `abm.exact`: [`reference/exact.py`](../reference/exact.py).
Every number is produced by a script in `examples/` with results committed as
JSON; the nineteen preregistrations are in `docs/preregistration/`.*

## Abstract

We study Algebraic Binary Memory (ABM), a MAP-B vector-symbolic memory: facts are XOR-bound triples in one majority-vote trace, and reasoning alternates unbinding with cleanup. We predict cleanup accuracy at finite dimension with no fitted parameter, under stated idealisations (independent random codewords, facts independent over GF(2)). The readout probability is that of Frady, Kleyko and Sommer; what we add is the accounting: the reference tie rule, several answers, aliases, weighted facts and the dependence between hops. Nineteen preregistered tests: thirteen supported their primary hypotheses, three in part, three failed. At an unmeasured dimension the error was 0.27 points; on dense subgraphs of two real graphs, 0.6–1.1 (test 4); under grounding errors, 0.8–1.9 without calibration. Choosing the dimension in advance on 80 unseen subgraphs, the model kept its promise on average, where the asymptotic law missed by over 6 points in up to 12 of 20 on WN18RR. Hops on one trace are negatively correlated (about −2/(πN) per bit); chains fall below p^h. The idealisation of independent facts fails on even cycles: on a biclique the model is up to 6 points optimistic. At equal bits ABM loses everywhere: an idealised exact store recalls more (14 of 14 cells) and a Bloom filter makes fewer membership errors, with ABM at 3.8–46× the Fano minimum, as the model predicted within 1 point (test 18). Nor does its algebra help: on two-hop chains and compiled compositions on two real graphs a dictionary with a join or a path table of the same bits did as well or better in 16 of 16 cells, as predicted within 2 points (test 19). Three front-end tests on questions written by people show the bottleneck is outside the memory: an embedding shortlist lifts a small model's answers by 20–23 points; linking by name loses a quarter. What remains is an accuracy computed before the memory is built, checked on average.

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

1. A **finite-D accuracy without fitted parameters** for cleanup, a
   specialisation of the finite-size theory of Frady, Kleyko and Sommer
   [@frady2018sequence] and of Kleyko et al. [@kleyko2023perceptron] to binary majority traces, with the
   classical majority-agreement probability [@kanerva1988sdm; @kleyko2022survey].
   What is new is the accounting: the reference tie rule, several true answers,
   aliases, weighted facts and symmetric twins (§3). It replaces an asymptotic
   law whose single constant, k = 0.92, turns out to be the error of its own
   approximations. The binomial distances themselves add nothing measurable at
   the dimensions we use: the FKS finite-M integral, with the same p_agree,
   agrees within 0.1 points on every tested configuration, and within 0.005 on
   test 2 (§3.2).
2. The **dependence between hops** on one trace, a one-line lemma (§4). The independence
   assumed by the composition law Acc(h) = p^h is false; the violation is
   derived and then measured.
3. **Nineteen preregistered tests** (§6), with predictions, criteria and harnesses
   committed before any run, on synthetic data at an unmeasured dimension, on
   two real knowledge graphs, on grounding errors, on two encodings, on deep
   chains, on the use the paper argues for: choosing a memory's dimension
   before storing anything, and questions written by people, and against an
   exact store and a Bloom filter with the same number of bits, and two-hop
   chains and compiled compositions against a dictionary with a join. Three failed
   their primary hypothesis; all are reported with their causes.
4. An account of **what the model is not** (§7): not a compressor, not better
   than an exact store or a Bloom filter at equal bits in any cell we measured
   (test 18), not better on two-hop chains or compiled compositions either
   (test 19), and not free of the faults we found in
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
| I | null distance Binomial(D, ½) | exact for a uniform random codeword independent of the probe (§3.2) |
| IV | asymptotic capacity N\* = k·2D/(π·z_G(M)²) | scaling correct; the constant k is the error of its approximations (§3.2); superseded for prediction |
| IV-exact | cleanup accuracy from exact majority agreement and binomial distances (§3.2–3.3) | **preregistered** (tests 2–6, 9, 10): every primary hypothesis supported, except dense FB15k-237 at D = 2048 before twins were counted (in part) (§6) |
| V | hops compose as Acc(h) = p^h | **falsified as an exact law**: hops on one trace are negatively correlated, by a derived amount that grows with depth (§4) |
| VI | failure grows with out-degree | **retired**: a load artifact (§8) |
| VI′ | topological neutrality: only load matters | corroborated on synthetic data; on real hubs the +3.3-point residual of test 2 was the missing twin term (−1.6 [−3.5, +0.3] with twins, exploratory) |
| VII | redundancy: a fact of weight w counts w² (N_eff = Σw²) | approximate; the exact weighted form is preregistered and removes its saturation error (§3.3) |
| VIII | every end-to-end failure is attributable to one level | design principle; not tested as a law |

### 3.2 From an asymptotic law to an exact one

**Law IV (asymptotic).** The member-trace correlation gives a signal z-score
√(2D/(πN)); retrieval fails when the minimum of M − 1 null distances crosses
it, which for large M sits at the second-order Gumbel threshold
z_G(M) = √(2 ln M) − (ln ln M + ln 4π)/(2√(2 ln M)). The 50% load is then

  N\* = k · 2D / (π · z_G(M)²),  with k = 1 derived.

Measured, k = 0.92 ± 0.03 (range over two sweeps, `examples/k_from_results.py`):
0.936 ± 0.009 (SD across codebook sizes) over a 36× codebook range at D = 2048, and
0.914 ± 0.025 over D ∈ [512, 4096], lower at small D. The scaling was posed as a
prediction and held (a linear model N\* = cD is rejected: c drifts from 0.098
to 0.068 over D ∈ [512, 4096]). But k is a fitted constant, and it drifts.

**The exact accuracy.** Both approximations in Law IV — a Gaussian signal and a
Gumbel extreme — can be removed, although only the second matters in practice
(see below). By A2, each bit of the query agrees with the
codeword of a stored object with the classical majority-agreement probability
[@kanerva1988sdm; @kanerva2009hd; @kleyko2022survey], exactly

  p_agree(N) = P( 1 + Σ_{j=2}^{N} x_j > 0 ) + ½·P( 1 + Σ_{j=2}^{N} x_j = 0 ),

the x_j independent Rademacher variables of the other facts. The distance to the
true codeword is therefore Binomial(D, 1 − p_agree(N)), each of the M − 1 null
distances is Binomial(D, ½), and the probability that cleanup returns the true
object is a finite sum over these distributions. There is no parameter.

What "exact" means here, precisely: for independent random codewords, and for
facts whose vectors are independent — which they are not always, see below — the
probability is computed from the exact discrete distributions, with no
Gaussian or extreme-value approximation. Ties need care. The reference returns
the *first* codeword inserted among those at minimal distance, so a target with
n_b null codewords inserted before it and n_a after wins iff

  P(win | d) = P(null > d)^{n_b} · P(null ≥ d)^{n_a},

which is exact given independent null distances (`win_ordered`). Tests 1–8 used
an even split of ties instead, truncated to first order: a tie with one null
codeword counts one half, and ties with two or more are dropped. The exact
average over random positions is (P(null ≥ d)^{n+1} − P(null > d)^{n+1}) /
((n+1)·P(null = d)), with n = n_b + n_a. For one cleanup at N = 12 with 1 000
distractors, the truncation lies 0.13 points below it at D = 320 (test 8's
dimension) and 0.23 at D = 256. The truncation is accurate where ties are rare
or positions mixed; test 8 found where it is not
(§6). The extensions of §3.3 add one assumption each, stated there: that
candidates at equal signal have independent distances.

**Idealisations, all in one place.** The model is exact only under three
assumptions, and every use of the word in this paper carries them: (i) the null
distances of the M − 1 wrong codewords are independent; (ii) candidates at equal
signal (§3.3), and the null distances of consecutive hops (§4), are independent;
(iii) the fact vectors are independent over GF(2). The third fails on even
cycles of the fact graph: in a rectangle (a,r,b), (a,r,c), (d,r,b), (d,r,c) the
four vectors multiply to the identity on every bit, so they are not four
independent Rademacher variables. By exhaustive counting, p_agree for each is
0.625, not 0.6875. On a K₄,₅ biclique with 20 facts at D = 64 the model predicts
0.930 against a measured 0.887 ± 0.008, and a model that keeps the linear
dependence in the vote (but not in the cleanup) predicts 0.882; under load, with
distractor facts, the independent model is 1–6 points optimistic and the linear
one 3–4 points pessimistic (`examples/cycles_probe.py`, exploratory, not
preregistered). Knowledge graphs do contain such dependencies. In the dense
FB15k-237 samples of test 4, 14% of distinct fact vectors lie in a 4-dependency
(four facts whose XOR is the identity) at D = 2048 and 27% at D = 8192; the
GF(2) rank deficit, which also counts longer even cycles, is 5–16% across the
dense samples of tests 2 and 4, and zero in uniform ones. At this sample size it
shows no detectable association with the per-cell error: taking deviations from
the mean of each (graph, D, N) group, the correlation of rank deficit with
predicted − measured accuracy is +0.13 [−0.12, 0.36] for dense FB15k-237 at
D = 2048 and +0.03 [−0.15, 0.20] over all 240 dense cells (bootstrap over
cells; `examples/cycles_residuals.py`, exploratory, on published data). Since
cycles make the model optimistic, they cannot be the main cause of the
pessimistic dense bias (−0.81 at D = 2048); a small optimistic share masked by
another effect is not excluded.

Against the capacity data behind k, the exact N\* is 50.5 / 87.5 / 155.5 / 277.5
at D = 512 / 1024 / 2048 / 4096, measured 50.2 ± 4.5 / 87.1 ± 3.1 / 158.8 ± 5.3 /
280.0 ± 6.4: inside every interval, including the small D where k drifted. The script behind these measurements was lost; a rerun with the protocol they
imply (`examples/capacity_seed10.py`, 10 new seeds) gives 50.6 ± 1.7 / 86.8 ± 2.2 /
156.2 ± 3.4 / 279.6 ± 5.5, again with the exact model, 50.2 / 88.7 / 155.3 /
277.4 on that script's grid, inside every interval. **k = 0.92 is the error of
the asymptotic treatment of M and of the approximate p_agree, not a property of
the memory.**

*Relation to Frady, Kleyko and Sommer.* Their finite-M readout, ∫φ(x)Φ(·)^{M−1}dx
with Gaussian crosstalk, is not an asymptotic law of the Law IV kind. To measure
what our binomial distances add, we instantiated it for the majority trace with
our exact p_agree(N) (signal distance N(Dq, Dq(1−q)), null distances N(D/2, D/4),
g true answers; cells with aliases not covered; `examples/fks_gaussian_vs_exact.py`,
exploratory). On every configuration above the two agree: N\* 50.2 / 88.7 /
155.3 / 277.4 for both (gaps under 0.03 facts), mean absolute error 0.27 and 0.59
points on test 2 parts A and B for both, and 0.723 against 0.722 in the D = 256
cell. A continuity correction makes the Gaussian worse. Among the sampled points
of a sweep over D and M, they differ by more than half a point only at D = 64
with M ≥ 4 096 and at D = 256 with M = 65 536, by at most 2.8 points. So the
gain over Law IV comes from treating M finitely with the exact p_agree, which the
FKS integral does once given that p_agree; we did not test the p_agree
approximation of the original papers. p_agree itself is classical [@kanerva1988sdm; @kleyko2022survey]. What this
paper adds is the accounting (ordered tie rule, several answers, aliases,
weights and twins, hop dependence) and the preregistered validation. Against a Monte Carlo of the reference implementation at D = 256,
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
  They are not: by the identity of §4.1, the agreements of two true objects of
  one query are anti-correlated, by −μ²/(1−μ²) per bit. This spreads the minimum,
  so the model should underpredict. An exploratory run (not preregistered) at
  g = 5, D = 256 and N = 20–40 found a bias of at most +0.2 points, within
  two standard errors of zero in every cell.
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

**Aliases set a ceiling.** Under the symmetric encoding, a query with a aliases at
equal signal besides its g true answers cannot exceed g/(g + a), whatever the
dimension. The ceiling of a set of facts is therefore computable before storing
them; on dense WN18RR subgraphs it is about 0.93–0.95, on FB15k-237 about 0.99. A
contract that does not state it promises accuracy no dimension can deliver.
Test 10 confirmed it: at D = 16 384, 38 of 40 subgraphs stayed within 2 points of
their ceiling.

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

**No practical sublinear speedup for cleanup in this regime.** A metric-tree index visits
4 487 of 5 000 nodes at D = 2048: a query sits at 0.446–0.483·D from its *own*
codeword and at 0.500·D from every other, a gap of 34–111 bits. Triangle bounds
cannot exploit it. Bit-sampling LSH is sublinear only in name: its exponent is
ε = ln(1 − 0.446…0.483)/ln(0.5) ≈ 0.85–0.95 (the usual LSH ρ, renamed here),
under the 1/c ≈ 0.89–0.97 bound, so a query costs about M^ε with M^ε tables. Cleanup is linear with a low constant (6.1 ms at
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
the two targets therefore have covariance −μ², where μ = 2·p_agree(N) − 1, and
correlation

  **−μ² / (1 − μ²) ≈ −2/(πN)**.

This is a one-line lemma. μ is the level-1 Fourier coefficient of majority on
each input, and the level-1 Fourier weight of majority tends to 2/π
[@odonnell2014boolean], which gives the approximation. We have not found the
lemma stated for the decoding of vector-symbolic bundles, and earlier versions
of this paper assumed the opposite.

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
exact −μ²/(1−μ²) (steps at small N are the parity of the majority vote: N and
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
four and six hops (Fig. 4). In absolute terms the shortfall is 1.5–1.8 points at
N = 12. The per-bit correlation behind it falls as about 2/(πN); we did not
measure the shortfall at other N.

![Deep chains (preregistered): chain accuracy minus p^h, measured (points)
against the per-bit model with the exact tie rule (lines), N = 12, 1 000
distractors.](figures/fig11_deepchain.png){width=60%}

### 4.2 The Memory Calculus

Terms e ::= a | 1 | e⊕e | ρ(e) | ⊞(e…) | cleanup(e), in an *exact fragment*
(⊕, ρ) and a *probabilistic fragment* (⊞, cleanup).

- **Normal form (exact fragment).** (E, ⊕, 1) is the free Boolean group — a
  vector space over GF(2) — on ρ-stratified atoms, so every exact term has the
  unique normal form "atoms of odd multiplicity". The fact is elementary; the
  calculus is built on it. Freeness and uniqueness are syntactic: they hold for
  terms over formal atoms. Realised by random codewords, two distinct normal
  forms give the same vector only if the atoms' codewords are linearly
  dependent over GF(2), which is rare but not excluded. Nor does the normal form
  say that fact vectors are statistically independent: the four facts of a
  rectangle normalize to 1, and the vote sees them as dependent (§3.2).
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

## 5. Earlier (not preregistered) results

Only P3 was later preregistered (§6); P1, P2 and the compose cost were not.

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
The tally in the abstract counts each test by its primary hypothesis. Test 2's primary hypothesis (H1, D = 16 384) was supported, though a secondary one was only in part. Test 11's primary hypothesis is end-to-end accuracy, and it was supported in part. Test 12's primary hypothesis was supported; those of tests 13 and 14 in part. Test 15's primary hypothesis (H1, the relation chosen) was falsified, though its end-to-end H2 was supported; it is counted as failed. Test 17's primary hypotheses (H1, H2) were supported; H3 was in part and H4 undecided. Tests 16, 18 and 19 were supported in full. In all: thirteen supported, three in part, three failed.

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
| 10 | `sizing.md` | the contract in use: choosing D in advance; the alias ceiling | all **supported**; the asymptotic law misses its promise on WN18RR |
| 11 | `human_questions.md` | questions written by people, through a local LLM front-end | memory level **supported** (+0.3 SE); end-to-end **in part** (+7.2 points, inside its interval) |
| 12 | `escalation.md` | the contract as the criterion for passing questions to a larger model | all three **supported**; but the larger path won almost always |
| 13 | `escalation2.md` | the same, one memory for both paths; a confidence index for abstaining | routing against random **supported**; the rest **in part** |
| 14 | `escalation3.md` | exact front-end confidence; the observed margin of an answer | the observed margin **supported**; routing against random **supported**; the rest **in part** |
| 15 | `memory_preview.md` | the small front-end sees the memory's answer for each candidate relation | relation (H1) **falsified** (−0.3 [−2.6, +2.0]); answer (H2) **supported** (+2.8 [+1.0, +4.6]) |
| 16 | `shortlist.md` | an embedding shortlist of k = 3 relations for the small front-end | both **supported**: relation +26.0 [+21.7, +30.3], answer +19.9 [+15.9, +24.1] |
| 17 | `webqsp.md` | the same front-ends on 515 natural questions (WebQuestionsSP) | H1, H2 **supported**; H3 (within 5 points of Qwen3-4B) **in part**; H4 **undecided** |
| 18 | `equal_bits.md` | ABM against an ideal exact store and a Bloom filter with the same bits | all three **supported**: ABM loses in every cell, as predicted |
| 19 | `algebra.md` | two-hop chains and compiled compositions against a dictionary + join / path table with the same bits; time | all four **supported**: no ABM advantage in 0 of 16 cells per task; model within 0.32–2.01 points; ABM ≥ 2 600× slower |

**Test 2 — the exact model on new configurations.** At D = 16 384, a dimension no
experiment had used, the mean absolute error over six loads was **0.27 points**
(signed −0.09; criterion ≤ 2). With g = 1, 2 and 4 true objects per query it was
0.58, 0.86 and 0.32 points, where Law IV, which assumes one target, erred by 0.6,
13 and 21. On dense subgraphs of FB15k-237 (a breadth-first sample, so that
entities recur) it was 1.06 points at D = 8192, and 2.88 at D = 2048 with a
signed bias of −2.61, outside the ±2 we had set: **supported in part**. The exact
model was *pessimistic* on dense real graphs, most at high load (−8.4 at
N = 400).

**Test 4 — symmetric twins.** Looking for the cause, after test 2 and on its data:
in the dense samples 6–14% of the triples had their symmetric twin
in the sample — two directions of one relation, and so one vector of weight 2
(§3.3) — against 0% in uniform samples. Counting twins as weight 2 reduced the
bias on those cells. Because that fit was made after seeing the data, it was
tested on data we had not seen: FB15k-237 with new seeds, and a second graph,
WN18RR. WN18RR was chosen knowing its twin share: before any measurement we had
looked at its structure, where 34.2% of the triples have a symmetric twin; in
dense samples a quarter do. On FB15k-237 the effect did not replicate. On seeds
10–19 at D = 2048 the model without twins had a signed bias of −1.29, against
−0.81 with twins, where test 2 had −2.61 (at D = 8192, −0.65 against −0.23).
Both versions meet H1, so H1 does not separate the two models. The evidence for
the twin mechanism rests on WN18RR (H3). With twins the
error was **0.57 and 0.60 points** on WN18RR (signed −0.16 and −0.36); without
them, the model was pessimistic by −2.1 and −3.8 on average and by up to −8.9 in
a cell; on uniform samples, where twins are rare (0.1–0.3%), both versions agree
(0.9 and 1.1 points). This is a check that the term does no harm where it should not act, not evidence that it helps. Both uniform cells also carry a signed bias whose 95% interval excludes zero, −0.67 at D = 2048 and +1.11 at D = 8192. The bias has opposite signs in the two cells, twins are too rare there to cause it, and it is unexplained (Fig. 7).
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

**Test 10 — the contract in use.** The paper's thesis is practical, so the last
test uses the model as an engineer would. For 80 unseen dense subgraphs of
FB15k-237 and WN18RR (N = 150 and 300), the model chose the smallest dimension it
predicted to reach a target accuracy T ∈ {0.70, 0.80}; accuracy was then measured
at that dimension. With the exact model the promise held in every group: measured
minus target averaged +0.01 to +1.68 points, and at most 2 subgraphs of 20 fell
more than 6 points short (observed 0 to 2 per group; expected under a perfect model, 0.4 to 2.2). That criterion was miscalibrated. If each measurement is binomial around its prediction, with each subgraph's own number of queries, a perfect model passes it in all eight groups with probability 0.39 (`sizing.md`, errata). This makes the test harsher, not laxer; a failure would have said little. The promise holds on average, not per subgraph: 11 to 14
of 20 met the target in each group. The cap of 200 queries per subgraph was often
not reached (median 92 on FB15k-237 at N = 150, minimum 10; at least 85 elsewhere),
so the 6-point threshold is about 1.2–1.4 standard errors where a subgraph has
around 90 queries, less where it has fewer, and about 2 where it has 200. With the asymptotic Law IV — the contract the reference
implementation had exposed — the promise held on FB15k-237, where aliases and twins
are rare (measured minus target +0.25 to +2.43 points, against +0.01 to +1.68 for the exact model, so the exact model did not beat it there), and failed on WN18RR: on average 2.8 to 6.2 points below target, and in
up to 12 of 20 subgraphs more than 6 points below. The exact model is now shipped
as `abm.exact`, with `contract_for` and `min_dimension`; the latter refuses targets
above the alias ceiling.

*Post hoc, not preregistered.* The Law IV arm above has neither aliases nor twins,
and on FB15k-237 it did as well as the exact model. To see what the WN18RR gap
comes from, we repeated the choice on the same 80 subgraphs with two stronger
asymptotic laws (`examples/sizing_lawiv_posthoc.py`). With Law IV × g/(g+a) per
query, WN18RR fell short by 0.15 to 2.79 points on average, against 2.82 to 6.21
for bare Law IV, and at most 6 of 20 subgraphs were more than 6 points short,
against 12. Adding twins as a signal proportional to their weight made things
worse: 3.40 to 5.65 points short, with up to 12 of 20. So aliases explain most of
the gap, and this naive twin term does not stand in for the exact one (§3.3). The
exact model was still the only one with every group at or above target on average.

**Test 11 — questions written by people.** The 743 questions of SimpleQuestions
v2 whose Freebase triple is in FB15k-237 were answered by a realistic pipeline:
link the subject by name, let a local 2-billion-parameter model (Gemma 4 E2B,
2-bit, llama.cpp, temperature 0) pick the relation among those stored for it, and
query 28 memories of 26–2 394 triples at D = 16 384. The contract was issued after
an audit of 100 questions and before the 643 test questions: front-end accuracy
0.36 [0.27, 0.46] times the memory accuracy predicted by `abm.exact`, 0.802, gave
0.289 [0.219, 0.367]. Measured: **0.361**, inside the interval but 7.2 points
above, beyond the 5 we had set — supported in part. The threshold was tight. The audit's sampling error, 0.802·√(0.36·0.64/100) = 3.85 points, and the binomial error of 643 questions, 1.79 points, give a predictive SD of 4.24 points. A correct contract would miss by more than 5 points with probability 0.24, and +7.2 is 1.7 SD: "in part" is compatible with audit noise alone. Scored strictly, counting an
answer only when the relation was also right, the result is 0.345 (+5.7 points),
still above the 5. Questions are clustered in memories, so we resample the 28
memories (10 000 draws): 95% intervals [0.318, 0.402] end-to-end and [0.305,
0.385] strict. The memory level agreed with the model: given the true (s, r),
0.807 against 0.803 recomputed per question (0.802 in the contract), +0.4 points
with a cluster-bootstrap SD of 1.4 points. By memory size: 0.877 against 0.878
on the 20 memories of 500–999 triples (464 questions), 0.624 against 0.607 on the
7 of 1 000 or more (178 questions). The gap is the
front-end's: the audit underestimated it (0.36 against 0.41 on the test), and
front-end and memory are not independent — on questions the front-end gets right
the memory answers 85%, against 78% on the others (z ≈ 2.3 treating questions as independent; resampling the 28 memories, z = 2.16 and 95% interval of the difference [0.010, 0.142]; exploratory, not
preregistered) — the first limit the preregistration had declared.

**Tests 12–14 — the contract as a router.** Tests 2–11 ask whether the
prediction is right; these ask whether it is useful for a decision. A small
front-end (Gemma 4 E2B) chooses the relation and the memory answers; for some
questions the system pays for a larger one (Qwen3-4B, 4 billion parameters) to
choose instead. Same 643–743 questions as test 11, all models local, choices at
temperature 0. Which questions to pass is decided per question, before the answer
is known, by the contract, by the small model's own confidence, or at random.
In test 12 the larger path also answered from an exact store; it was right where
the small one was wrong in 400 questions and the reverse in 9, so passing every
question was best and no router was needed. Even so, ranking by the contract
beat random choice by 4.8 points [3.4, 5.9] (mean over the shares 20–50%) and the
small model's confidence by 1.9 [0.15, 3.5]. In tests 13 and 14 both paths use
the same memory at D = 16 384, so passing a question can fix only the front-end.
There the value p̂_mem × (1 − confidence) — pass a question when the memory will
answer it and the small model is unsure — beat random choice by 2.7 [1.5, 3.8]
and 3.4 [2.4, 4.5] points, and the small model's confidence only at the higher
shares (in part). As an index for answering or abstaining, the contract added to
the small model's confidence in the expected direction but within noise, while
the **observed margin** of the returned answer, z = (D/2 − d)/(√D/2), added 2.6
points [0.9, 4.1] (test 14; exposed as `Memory.query_z`). The signal is real but
small, because most of the error is the small front-end's: it picks the right
relation in 41% of questions. The choices turned out not to be fully
deterministic across reruns (2 of 643 for the small model, 6 for the larger).

**Tests 15–17 — the front-end.** Tests 12–14 located most of the error in the
small front-end, so these three tests change only the front-end; the memory
(D = 16 384) and its contract are unchanged, and none of them tests the capacity
theory. All use a cluster bootstrap over the memories (10 000 draws). In test 15
each candidate relation was shown with the answer the memory returns for it. The
small model did not pick the right relation more often (H1, −0.3 points
[−2.6, +2.0]: **falsified**; we had predicted +5 to +10), but it answered right
more often (H2, 0.354 → 0.382, +2.8 [+1.0, +4.6]: supported): exploratory
per-question counts suggest it moved toward options where the memory returns a
plausible answer. In test 16 a 33-million-parameter embedding model
(bge-small-en-v1.5) kept the k = 3 relations closest to the question (k chosen on
the 100 audit questions; recall 97.5% on the 643 test questions). Relation accuracy
rose from 0.409 to 0.669 (+26.0 [+21.7, +30.3]) and answer accuracy from 0.393
to 0.593 (+19.9 [+15.9, +24.1]); both supported, about twice the predicted gain.
The embedding's top choice alone, reported but not a hypothesis, answered 0.647.
SimpleQuestions was written by people who saw the triple, so test 17 repeated the
comparison on 515 one-relation questions of WebQuestionsSP [@yih2016webqsp],
taken from search queries (27 memories). Answer accuracy was 0.381 for the small
model with all options, 0.598 for the embedding alone, 0.608 for the shortlist
with the small model and 0.656 for Qwen3-4B with all options: embedding against
the small model +21.7 [+16.4, +27.3] and shortlist against it +22.7
[+17.8, +28.1], both supported; embedding against Qwen3-4B −5.8 [−8.9, −3.2], so
non-inferiority within 5 points was only in part; embedding against shortlist
−1.0 [−3.2, +1.2], undecided. The gain of test 16 was therefore not an artefact
of the dataset. The next bottleneck is not the relation: linking the entity by
name was right in 75% of the WebQuestionsSP questions, the same for every
front-end, so it loses a quarter of them before any choice is made.

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
resampling seeds within cells (2 000 bootstrap draws), so its intervals are conditional on the cells. Of its 31 rows we count
the 25 that score the final predictor. Six are left out: test 1 (two rows, which
tested Law IV, not the exact model), test 2 on dense FB15k-237 (two rows,
superseded by test 4) and dense WN18RR without twins (two rows). Two findings. First, most
errors of the final predictor are **at or below the noise floor** — the mean
|error| a perfect model would show, √(2/π) times each cell's standard error: at
D = 16 384, 0.27 against a floor of 0.70; on dense WN18RR with twins, 0.57 and
0.60 against 1.31; for composition, 0.75–1.24 against 1.2–1.4. There the model
cannot be told apart from the truth with these data; the same holds for both
encodings in test 6. The floor is not a validation. The smallest signed bias these
data detect (about 2.8 SE of the summary: a two-sided 5% test with 80% power) is
0.8–2.1 points for tests 2–5 and 1.4–4.9 for test 6. In a single cell it is
1.1–8 points, and 3–5 points in the noisy dense and composition cells. The
criteria |err| ≤ 2 and ≤ 3 sit only about 1.5–3 times above the floor in those
cells, so passing them rules out large errors, not errors of a point or two.
Second, **four of the twenty-five final summaries carry a
signed bias whose 95% interval excludes zero**:
dense FB15k-237 at D = 2048 (−0.81), uniform WN18RR at D = 2048 (−0.67) and 8192
(+1.11), and wrong-relation errors (+1.34). All are below 1.4 points, and the two
WN18RR biases have opposite signs, so they do not point to one missing term; the
wrong-relation bias is unexplained. The summaries overlap (they share data and
cells), so we do not attach a chance probability to the count of four. Each bias
is near or below the smallest bias its test detects.

**What the tests leave open.** The exact model treats candidates at equal signal,
and the null distances of different hops, as independent; test 2 puts the cost
of the first below one point. A later exploratory run at g = 5 found it
below 0.3 points and not distinguishable from zero (§3.3). The hub residual of test 2
(+3.28 points, hub minus rest, measured − predicted) was computed before twins
were counted. On the same queries it is −1.64 [−3.49, +0.26] with twins, and
−1.81 [−3.69, +0.06] with twins and self-loops (`examples/cycles_residuals.py`):
the residual was the missing twin term, and what remains is compatible with
zero. Hub queries are three times as often on a 4-dependency (48% against 16%),
but the per-cell hub–rest gap in dependence does not track the gap in error
(r = +0.07 [−0.04, +0.22]). Dense samples favour the neighbourhood of their starting entity.
Tests 1–10, 18 and 19 query stored triples or synthetic chains; tests 11–17 use questions
written by people, through a deliberately simple front-end. All tests are closed-world: the codebook holds only the stored items, and every query asks for a stored (s, r) pair. The abstention threshold of §3.4 rests on one synthetic check (3/300 false accepts); it has not been tested on absent queries over a full-vocabulary codebook.

## 7. What the model is, and is not

**Against an exact store.** On ProofWriter [@tafjord2021proofwriter]
(open-world attribute fragment, 10 seeds), a forward chainer whose only truth
oracle is one Hamming distance to the trace reaches 99.8% ± 0.3, 99.1% ± 0.3 and
92.4% ± 1.4 at depths 0, 2 and 5 on the problems four grammatical patterns parse
(35–56% of them, so the subset is selected; majority baseline 42%). The seed
runs use the first 100 parsable problems per depth (88 at depth 2, the only
parsable ones among its first 300); the same problems appear in
every run, so the ± covers encoding seeds only, not problem sampling. The same
chainer with a Python set in place of the trace is deterministic; scored once on
150 problems per depth, it reaches **100% at every depth**:
the parser and the chaining are sound, and the whole loss is the oracle's. The
trace takes 512 bytes where a minimal exact encoding takes 8–23 (Fig. 10). The
oracle's z ≥ 3 threshold is the M = 1 case of §3.4 and controls false accepts per
test, not per proof.

![ProofWriter, parsable subset: accuracy by depth (10 seeds, 95% CI) against the
majority baseline.](figures/fig3_proofwriter.png){width=60%}

**Not a compressor.** Holding N facts at single-query accuracy a needs about
π·(z_G(M) + Φ⁻¹(a))²/(2k) bits per fact (Law IV), while a minimal exact encoding
of (s, r, o) needs 2·log₂V + log₂R. At high accuracy the trace is the larger of
the two; the accuracy at which they cross depends on V, R and M, and an earlier
version printed a crossover range that no committed script reproduces, so we
withdraw it (§8).

**At equal bits (test 18).** We then measured it. Facts (s_i, r_{i mod 13}, o_i)
over 2N entities, three seeds per cell, D = 2048 with N = 50–600 and D = 8192 with
N = 200–2 400. An *idealised* exact store spends 2·⌈log₂ 2N⌉ + ⌈log₂ 13⌉ bits per
fact with no overhead and no stored keys, which no real table achieves, so the
comparison favours the store; with D bits it keeps ⌊D/b⌋ facts and loses the
rest. ABM recalled fewer facts in **14 of 14 cells**, including every overloaded
one (for example 0.660 against 0.853 at D = 8192, N = 400; 0.015 against 0.114 at
N = 2 400): there is no crossover, as the exact model had predicted, and it
predicted ABM's accuracy with a mean error of 0.99 points at D = 2048 and 0.54 at
D = 8192. For membership, a Bloom filter of D bits made fewer errors (false
negatives plus false positives on 1 000 unstored facts) than `Memory.member`
(z ≥ 3) in every cell: the Bloom filter had no false negatives, while
`Memory.member`, with almost no false positives, missed 26–93% of the stored
facts above the lightest loads. Measured against the Fano minimum for answering N
queries among 2N objects at the accuracy it reached, ABM used 3.8 to 46 times the
bits. All three hypotheses were supported, which here means ABM lost. One synthetic
protocol, no aliases or twins, and only single-hop recall and membership were
compared. **ABM is not dense**, and no claim of this paper rests on density or
compactness. What it offers, if anything, is a degradation that can be computed
before deployment — test 18 shows that too, since the unfavourable result was
predicted within a point — and an algebra: binding, unbinding and composition on
one fixed-size vector (§4.2). Test 19 measured the second.

**The algebra at equal bits (test 19).** On FB15k-237 and WN18RR we sampled N
facts as two-hop paths (x, r₁, y), (y, r₂, z) and asked the paths that are
functional in the sample (up to 200 per cell and seed, 3 seeds; D = 2048 with
N = 100–800, D = 8192 with N = 400–3 200; 16 cells per task). The store was
realisable, not idealised: implicit keys, as in a static-function filter, at
1.23·⌈log₂ E⌉ bits per fact, keeping a random subset when D is short. *Chain:*
`Memory.chain` with cleanup per hop against two dictionary lookups (a join). ABM
was below the store in every cell (for example 0.101 against 0.848 at D = 2048,
N = 200 on FB15k-237; 0.057 against 0.718 at D = 8192, N = 800): a chain pays p²
where the store pays (c/N)². *Compiled composition:* the queried paths bound into
one trace with the bridge cancelled, one cleanup, against a path table
(s, r₁, r₂) → z of the same bits. Composition came closest (0.91–0.99 at
D = 8192), but the table stayed at 1.0 there, and at D = 2048, N ≥ 400 it held
0.76–0.96 against ABM's 0.26–0.38. ABM was ahead by more than 2 SE in **0 of 16
cells** for either task. The exact model, including the new translation of a
compiled composition into a single-hop query, predicted ABM with a mean error of
0.32–2.01 points per (graph, D, task). For one two-hop question, ABM with a
vectorised cleanup took 2 600 to 230 000 times as long as a Python `dict`; the
store's time is near the timer's resolution, so the ratio gives only the order of
magnitude. All four hypotheses were supported, which again means ABM lost. What
remains unmeasured is richer queries (conjunctions, analogies, role filling) and
a bitpacked runtime; on what was measured, the algebra is **not an advantage**,
and the only claim left is that ABM's accuracy can be computed before the memory
is built.

**Sufficient bounds and sizing constants.** Clarkson, Ubaru and Yang's Theorem 16
[@clarkson2023capacity] gives, from its proof, a sufficient dimension
m_C = 56·n·ln(2d/δ) for a threshold membership test over d items. On their own
task (d = 500, δ = 0.1; `examples/clarkson_comparison.py`) m_C exceeds the
measured minimum 5.7–6.3 times, and 5.7–6.9 times with exactly their threshold,
while a Gaussian-tail estimate, 2π·n·z², matches the measurement within a 10%
grid. This compares a constant chosen for a proof with a measurement, and says
nothing against the theorem, which proves a scaling and is correct as stated; it
only shows that sizing a memory needs a different number. The comparison uses the
2023 preprint's constant.

**Capacity contract.** At 1 KB (D = 8192), with 10 fresh seeds, the exact model is
within 0.4 points of measurement on average, Law IV with k = 0.92 within 0.3, and
Law IV with k = 1 off by 3.2, always optimistic. The line "up to 300 facts at
≥ 85% in 1 KB" that an earlier version printed fails its own measurement (84.6%);
a minimal exact encoding of the same facts fits about 360 of them in that kilobyte, at 100% (Fig. 11). That count uses the test's own vocabulary, 2N entities and 13 relations (2·log₂(2N) + log₂13 ≈ 22.7 bits per fact at N = 361); with the vocabulary of FB15k-237 (about 14 500 entities, 237 relations, 35.5 bits per fact) it is about 230.

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
near-perfect recall, so the pilot illustrates the level decomposition of Law VIII and tests nothing in the capacity theory.
It is a case, not a sample.

## 8. Falsifications and corrections kept on record

| retired or corrected claim | what showed it |
|---|---|
| Law VI: failure grows with out-degree | at constant load, accuracy is flat (64–73%) for out-degree 1–24; the effect was load |
| geometric decay of raw chaining, z_eff = √D·ρ^h | two-trace raw chaining measures 0% where it predicts ~40%; T ⊕ T = 1 is the sharp form |
| **Law V as an exact law** (hops independent) | preregistered test 3: correlation −μ²/(1−μ²) per bit, independence rejected at 6 SE |
| k = 0.92 as a constant of the memory | the exact model reproduces the capacity data with no constant (§3.2) |
| P3's "consistent 5% bias, unexplained" | the Gumbel-ratio approximation; the exact model predicts the reconstructed gains within 1–4% |
| the composition law as parameter-free (v1.4 and before) | its script took the clean accuracy from the measurements and calibrated the spurious-fact curve; replaced by a preregistered prediction without calibration (test 5) |
| "~3% mean deviation" for the composition law | the script's own data gave 4.3 (3 seeds) and 3.6 (10), and 10.2 for missing facts |
| sublinear cleanup; a fixed acceptance threshold | §3.4 |
| our deep-chain prediction with a tiny codebook | preregistered test 7: off-path recovery dominates, up to +18.9 SE |
| the bits-per-fact crossover (74–81%, v1.8) | no committed script reproduces it; withdrawn (audit 2026-09-30) |
| "the known bound over-provisions dimension sixfold" (v1.8 abstract) | a proof constant against a measurement; kept in §7 as a sizing remark, out of the abstract |
| "exact" without the GF(2) caveat | facts on even cycles are dependent; several points on a loaded biclique (§3.2) |
| "a fixed size, membership by one distance" as properties of the design (v1.11 §7, untested) | preregistered test 18: at equal bits an idealised exact store recalls more and a Bloom filter makes fewer membership errors in every cell |
| "an algebra: binding, unbinding and composition" as the residual value (v1.12 abstract and §7, unmeasured) | preregistered test 19: on two-hop chains and compiled compositions a dictionary + join or a path table of the same bits is at least as accurate in every cell, and orders of magnitude faster |
| the even split of ties | preregistered test 8: with 1 000 distractors after the target, it misses a single hop by 1.8 points (13.8 SE); replaced by the exact rule |

**Lost scripts.** Four result files had no script that produced them:
`independence_results.json` and `projection_results.json` (replaced by test 3),
`conjecture7_results.json` (Law VII; replaced by test 4) and
`composition_stress_results.json` (replaced by test 5). The claims of earlier
versions that rested on them are superseded by the preregistered tests, whose
harnesses are committed.

**Self-loops.** Since 2026-09-30 `abm.exact` models self-loops (s, r, s): each
one is ρ(c_r) whatever s is, so all those of a relation are one vector, and every
query (x, r) sees x as an alias. FB15k-237 train has 1 625 of them (0.6%), not
none as the audit stated; WN18RR has 7. Tests 4, 6, 10 and 11 were computed
before the fix. An exploratory recomputation on the same samples
(`examples/selfloop_impact.py`, not preregistered) moves the mean bias of test 4
by under 0.1 points on either graph. One WN18RR cell (uniform, D = 8192,
N = 200) was predicted at 97.6% against 80.0% measured; the corrected model
predicts 80.2%. No dimension chosen in test 10 changes, so no promise flips. In
test 11 the memory prediction moves from 80.19% to 80.32%. Re-evaluated with
their preregistered criteria, every hypothesis of tests 4 and 6 is still
supported; no cell of test 6 moves by more than 0.06 points. The frozen harnesses of tests 4, 6 and 10 no longer run on the
current model; `replicate.py` runs each harness on the commit that recorded its
results.

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
Frady, Kleyko and Sommer [@frady2018sequence] derive finite-size retrieval
accuracy — the probability that the winner-take-all readout returns the right
item — from the crosstalk distribution in VSA-coded memories, and Kleyko et al.
[@kleyko2023perceptron] extend this theory to predict accuracy across VSA
models; Plate [@plate1995hrr] and Gallant and Okaywe [@gallant2013objects] give
the earlier capacity analyses, and Schlegel, Neubert and Protzel
[@schlegel2022comparison] and Mirus, Stewart and Conradt [@mirus2020capacity]
compare capacity across architectures empirically. Our §3 is a refinement of
that line, not an alternative: the same readout probability, computed with the
classical majority-agreement probability of a binary majority trace (discrete
and Gaussian versions compared in §3.2), and extended to the
reference tie rule, several answers, aliases, weights, symmetric twins and hop
dependence, with preregistered tests on real graphs. Resonator networks
[@frady2020resonator] factor a bound vector by iterating cleanup on all factors
at once; §4 concerns the simpler hop-by-hop cleanup, whose errors are
correlated through the shared trace. Clarkson et al. also prove that the reliability of nested
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

1. **Cycles.** Facts that close even cycles are dependent over GF(2) (§3.2);
   on a loaded biclique the model is several points optimistic, and keeping the
   dependence only in the vote overcorrects. A model that keeps it in the cleanup
   too is open. In the real dense samples 5–16% of the rank is lost to such
   dependencies, with no detectable association with the per-cell error at
   this sample size (§3.2).
2. **Real hubs.** The +3.3-point residual of test 2 disappears once twins are
   counted (§6); that recomputation is exploratory, and a preregistered test on
   new hubs has not been run.
3. **Joint independence of candidates.** The exact model treats candidates at
   equal signal, and the null distances of consecutive hops, as independent;
   the first cost is measured (test 2 and an exploratory run, §3.3): at most
   about 0.2 points at g = 5, not distinguishable from zero, although the
   expected direction is underprediction; the second is unmeasured.
4. **Off-path recovery.** In tiny codebooks a failed hop can land on the right
   entity by chance; a uniform 1/M recovery overestimates it (test 7). No clean
   model yet.
5. **Two-level contracts.** On questions written by people (test 11) the memory
   level is predicted exactly, but the end-to-end contract errs by 7 points,
   through the audit's sampling error and a correlation between front-end and
   memory success that the product formula ignores. As a router (tests 12–14)
   the contract helps, but its share of the decision is bounded by the share of
   the error that is the memory's. With an embedding shortlist (tests 16–17) the
   largest measured loss is entity linking by name, a quarter of the
   WebQuestionsSP questions; it is outside the memory and untested here.
6. **The value of the algebra.** At equal bits ABM loses single-hop recall and
   membership (test 18), and two-hop chains and compiled compositions (test 19),
   to exact stores of the same size. Richer queries (conjunctions, analogies,
   role filling) and a bitpacked runtime, where the time gap might shrink, are
   unmeasured; nothing measured so far suggests they would reverse the result.
7. **Negation and quantifiers** in the algebraic truth oracle.
8. **Distributional confluence** for nested probabilistic redexes.
9. **The ABM complexity class** (polynomial D, O(1) controller).

## 11. Conclusion

ABM is not a better retriever, not a compressor, and not better than an exact
store or a Bloom filter with the same number of bits in any cell we measured,
not better than a dictionary and a join on two-hop chains or compiled
compositions, and we say so with measurements. It is a memory
whose accuracy can be computed without fitted parameters, under stated
idealisations, from its
dimension, its load, its codebook and the structure of what it stores — including
structure such as symmetric relations collapsing into single facts. Nineteen
preregistered tests put that claim and its uses at risk, on synthetic data, on
two real knowledge graphs, under grounding errors, on two encodings, on deep
chains, in use sizing memories in advance, on questions written by people, as
a router to a larger model, with three front-ends, at equal bits against an
exact store and a Bloom filter, and on its algebra against a dictionary and a
join; thirteen supported their primary hypotheses, three
in part, and three failed — two changed the model, one (test 15) was a wrong
prediction about the front-end — and one falsified a law of our own earlier
versions by the amount the theory predicted. The test at equal bits went against
ABM, as the model said it would, and so did the test of its algebra. What is
left is the one claim this paper can support: the accuracy is computable before
the memory is built. The predictions were fixed before the data; the
outcomes are in §6, the failures and the changes they caused in §8 and in the
preregistration files.

---

*Reproducibility: every number is produced by a script in `examples/` with
results committed as JSON; results from 10-seed reruns sit beside the originals
with an `_s10` suffix; the nineteen preregistrations, with their outcomes, are in
`docs/preregistration/`. The frozen reference implementation
(`reference/abm.py`, under 300 lines, numpy-only, deterministic) and the exact theory
(`reference/exact.py`, shipped as `abm.exact`, numpy-only) are covered by the test
suite, which
runs in continuous integration on Linux x86-64 (Python 3.10–3.13, NumPy
1.24–2.5), macOS arm64 and Windows x86-64, with codewords checked bit-identical
across the three. Figures: `examples/make_figures.py`. To replicate any
preregistration from a clean clone: `python examples/replicate.py <name>`, which
downloads the data with checksums, runs each harness on the commit that recorded
its results and compares the rerun with them (test 11 only in its deterministic
part, without the language model; tests 12–14 only their analysis, from the
committed model answers). On 2026-10-01 the first eleven tests, the seed-10
capacity data, the Clarkson comparison and ProofWriter reproduced: twelve
byte-identical, two (tests 4 and 10) up to the last digit of one float; on
2026-10-05 the analyses of tests 12–14 reproduced up to float rounding. Tests
15–19 are not yet wired into `replicate.py`; their harnesses
(`examples/memory_preview_prereg.py`, `shortlist_prereg.py`, `webqsp_prereg.py`,
`equal_bits_prereg.py`, `algebra_prereg.py`) and model answers are committed.*

## References
