# Sizing a Vector-Symbolic Memory before Building It: An A-Priori Accuracy Estimate and Twenty-Two Preregistered Tests

*(with Algebraic Binary Memory, a dominated MAP-B design, as the case studied most)*

**Preprint v1.16 — October 2026**
*Normative specification of ABM: [FORMALISM.md](FORMALISM.md) (frozen, v2.1).
Reference implementation: [`reference/abm.py`](../reference/abm.py); exact
MAP-B theory, shipped as `abm.exact`: [`reference/exact.py`](../reference/exact.py);
the estimates for the other families are in the harnesses of tests 21–22
(`examples/vsa_families_prereg.py`, `examples/generalization_prereg.py`).
Every number is produced by a script in `examples/` with results committed as
JSON; the twenty-two preregistrations are in `docs/preregistration/`.*

## Abstract

At one hop, a parameter-free estimate of cleanup accuracy, computed from the triples before the memory is built, held within about one point on average for five vector-symbolic families (MAP-I, FHRR, HRR and two sparse block codes) on subgraphs of two knowledge graphs it had never seen, CoDEx-M and YAGO3-10. It failed elsewhere: for MAP-B, the binary majority memory (ABM) this work started from, it was 3.5 points optimistic on dense YAGO3-10 (worst cell 7.2) and the dimension it chose missed its target in 4 of 12 cells; its two-hop chain predictions were falsified; and at equal bits ABM lost to exact stores in every cell, also under bit flips up to 20% against practical error-correcting codes. The readout model for MAP, FHRR and HRR is that of Frady, Kleyko and Sommer. Ours are the accounting of graph structure (symmetric twins as weight-2 vectors, aliases, several answers, codebook), atom multiplicities for HRR, exact ties for a summing block code and MAP-B's exact majority model. The accounting cut 2–8 points from the naive estimate on dense WN18RR; it is untested out of sample, since neither new graph was dense enough to be admitted. All 22 tests were preregistered; from test 20 on, before each run, the files were pushed and time-stamped. By primary hypothesis: theory of MAP-B (tests 1–10) 8 supported, 2 failed; front-end (11–17, not tests of the estimate) 3 supported, 3 in part, 1 failed; baselines at equal bits (18–20) ABM lost each time, as predicted (2 supported, 1 hypothesis of an ABM win falsified); generalization (21–22) 1 supported, 1 in part. Per hypothesis, 60 of 83 supported, 11 in part, 8 falsified, 4 undecided or untestable; for nine tests the primary was named after the outcomes. Errors of a point or two are not excluded.

## 1. Introduction

**What this paper is.** It proposes an estimator and reports how it fared. The
estimator gives, before a vector-symbolic memory is built, the accuracy of its
cleanup — the probability that unbinding a key and taking the nearest codeword
returns a true answer — from the dimension and the triples it will hold, with no
fitted parameter. It covers six families: MAP-B (binary, majority vote), MAP-I
(integer sum), FHRR (phasors), HRR (real Gaussian vectors, circular
convolution) and two sparse block codes, BSDC-OR and BSDC-S (§2). Its new part
is the **accounting of graph structure**: symmetric twins fused into one vector
of weight 2, aliases created by the symmetric encoding, several true answers,
the codebook. It was put at risk twenty-two times, with predictions, criteria
and harnesses committed before each run and every outcome appended to the same
file.

**What held and what did not.** At one hop the estimate held, within about one
point on average, for five families on two graphs it had never seen (test 22,
§5). It failed for MAP-B, the family this project started from, on dense
YAGO3-10; its two-hop chain predictions were falsified; and the accounting has
one test, on dense WN18RR (test 21).

**The memory studied most.** Algebraic Binary Memory (ABM) is a MAP-B
vector-symbolic architecture [@clarkson2023capacity]: facts bound by XOR into one
majority-vote trace, read by unbinding and cleanup. Most of the record (tests
1–20) concerns it. At equal bits an exact store, a Bloom filter, a dictionary
with a join and a path table dominate it on every task we measured (tests
18–19), and under independent bit flips up to ε = 0.2 an exact store protected
by polar or BCH codes still recalls more in every cell where the two differ
(test 20, §4). Within the families of test 21, a partitioned Bloom filter
(BSDC-OR) given MAP-B's bits sat near its ceiling (descriptive, §5). An exact
store's accuracy is also computable in advance, and more simply. Predicting a
vector-symbolic memory's accuracy is of interest only to someone who must use a
distributed representation anyway — vector-symbolic or neuromorphic hardware,
in-memory computing, a network whose interface requires bundled vectors — and
we claim no reader beyond that perimeter.

**Contributions, with what is not ours.**

1. **The accounting of graph structure** inside known readout formulas (§2.3,
   §2.5): twins as weight-2 vectors (noise S2 = Σw², not N), aliases as
   candidates with the true object's signal, several true objects, codebook
   M = entities + relations. The Gaussian readout for MAP, FHRR and HRR is that
   of Frady, Kleyko and Sommer [@frady2018sequence; @kleyko2023perceptron]; the
   BSDC-OR formula is Bloom-filter occupancy [@bloom1970space]; the families were
   compared empirically by Schlegel et al. [@schlegel2022comparison]. **No
   capacity theory here is new.** The accounting was tested once, on dense
   WN18RR (test 21, H6–H7); test 22 admitted no graph to test it again.
2. **Two family-specific models.** Atom multiplicities for HRR (§2.5), built on
   Plate's variance analysis [@plate1995hrr]: without them HRR is predicted 8–20
   points too optimistic (test 22, H4). Exact ties for BSDC-S, which replaced an
   approximation after test 21 had shown its bias; the search began from that
   bias (§2.5, §8).
3. **MAP-B's exact model** (§2.2–2.4): the classical majority-agreement
   probability [@kanerva1988sdm; @kleyko2022survey] with the reference tie
   rule, weights, and the dependence between hops on one trace, a one-line
   consequence of the level-1 Fourier weight of majority [@odonnell2014boolean].
   The finite-M readout of FKS with the same p_agree agrees with it within 0.1
   points (Appendix H): what is ours is the bookkeeping, not the mathematics.
4. **A preregistration record** of twenty-two tests (§3–§6), with the failures
   and the corrections they caused (§7), a power analysis (§3.2) and the limits
   in one place (§8).
5. An account of **what ABM is not** (§4): not a compressor, not better than an
   exact store or a Bloom filter at equal bits, not better on two-hop chains or
   compiled compositions, not rescued by noisy storage.

### 1.1 Method and anteriority

Since v1.4 every new claim is preregistered; retired laws stay in the record
with the data that killed them (§7). Two limits. First, **anteriority**. Up to
test 19 the preregistrations were local git commits, pushed in batches after
the runs, with commit times set by the author: for test 19 the result file was
written 3 min 21 s after the preregistration commit, which had not been pushed
when the run took place, and nothing but the author's word shows the harness
was not run earlier. From test 20 on, the preregistration, the frozen
predictions and the harness are pushed to a public repository and stamped with
OpenTimestamps before the run (`tools/stamp_prereg.sh`; `.ots` files next to
each file; commits `78a3234`, `a96da4c`, `5463e7f`). This makes the existence of
those exact files before the run checkable by a third party: the `.ots` files in
the repository are upgraded to Bitcoin block attestations (e.g. block 971093 for
test 20) and check with `ots verify`. The `.md` files had their outcome appended
after the run, so one verifies the version of the stamping commit, not the
current one: e.g. `git show 2e20676:docs/preregistration/generalization.md > /tmp/f
&& ots verify -f /tmp/f docs/preregistration/generalization.md.ots` (without a
local Bitcoin node, `ots --no-bitcoin verify ...` checks the file against the
stamp and names the block whose merkle root to compare). The harnesses and
prediction files are unchanged since stamping and verify directly. It does not show
that no unrecorded run preceded them. Second, which hypothesis of a test is
"primary" was in nine tests (2, 11, 13, 14, 15, 17, 20, 21, 22) decided when
the tally was written, after the outcomes (§3.1); we therefore give the count
per hypothesis beside the count per test.

## 2. The model and the estimate

### 2.1 The memory

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
pipeline. Each operator is necessary: ablating any one kills a disjoint set of
capabilities (Appendix A).

**On the word "exact".** An "exact" model here is a computation from the
discrete distributions with no Gaussian or extreme-value step, *under the
idealisations of §2.2*; for real data it is an estimate, not an exact law.

**Conventions.** ± is a 95% confidence half-width across seeds unless the text
says SD or range. Differences between accuracies are in percentage points.

| law | statement | status |
|---|---|---|
| I | null distance Binomial(D, ½) | exact for a uniform random codeword independent of the probe (§2.2) |
| IV | asymptotic capacity N\* = k·2D/(π·z_G(M)²) | scaling correct; the constant k is the error of its approximations (§2.2); superseded for prediction |
| IV-exact | cleanup accuracy from exact majority agreement and binomial distances (§2.2–2.3) | **preregistered** (tests 2–6, 9, 10): every primary hypothesis supported, except dense FB15k-237 at D = 2048 before twins were counted (in part) (§3); held as a control on FB15k-237 and WN18RR (test 21) and on CoDEx-M, **failed on dense YAGO3-10** (test 22, §5) |
| V | hops compose as Acc(h) = p^h | **falsified as an exact law**: hops on one trace are negatively correlated, by a derived amount that grows with depth (§2.4) |
| VI | failure grows with out-degree | **retired**: a load artifact (§7) |
| VI′ | topological neutrality: only load matters | corroborated on synthetic data; on real hubs the +3.3-point residual of test 2 was the missing twin term (−1.6 [−3.5, +0.3] with twins, exploratory) |
| VII | redundancy: a fact of weight w counts w² (N_eff = Σw²) | approximate; the exact weighted form is preregistered and removes its saturation error (§2.3) |
| VIII | every end-to-end failure is attributable to one level | design principle; not tested as a law |

### 2.2 From an asymptotic law to an exact one

**From Law IV to an exact count.** Earlier versions used an asymptotic law,
Law IV: N\* = k·2D/(π·z_G(M)²) for the 50% load, with a Gaussian signal and a
Gumbel extreme in M. Measured, k = 0.92 ± 0.03, a fitted constant that drifts
with D (Appendix H). Removing both approximations gives the exact accuracy below.

**The exact accuracy.** By A2, each bit of the query agrees with the codeword of
a stored object with the classical majority-agreement probability
[@kanerva1988sdm; @kanerva2009hd; @kleyko2022survey], exactly

  p_agree(N) = P( 1 + Σ_{j=2}^{N} x_j > 0 ) + ½·P( 1 + Σ_{j=2}^{N} x_j = 0 ),

the x_j independent Rademacher variables of the other facts. The distance to the
true codeword is therefore Binomial(D, 1 − p_agree(N)), each of the M − 1 null
distances is Binomial(D, ½), and the probability that cleanup returns the true
object is a finite sum over these distributions. There is no parameter.

Ties need care. The reference returns the *first* codeword inserted among those
at minimal distance, so a target with n_b null codewords inserted before it and
n_a after wins iff

  P(win | d) = P(null > d)^{n_b} · P(null ≥ d)^{n_a},

which is exact given independent null distances (`win_ordered`). Tests 1–8
used an approximate even split of ties instead (Appendix H); test 8 found where
it fails (§3.3).

**Idealisations, all in one place.** The model is exact only under three
assumptions, and every use of the word in this paper carries them: (i) the null
distances of the M − 1 wrong codewords are independent; (ii) candidates at equal
signal (§2.3), and the null distances of consecutive hops (§2.4), are independent;
(iii) the fact vectors are independent over GF(2). The third fails on even
cycles of the fact graph: in a rectangle (a,r,b), (a,r,c), (d,r,b), (d,r,c) the
four vectors multiply to the identity on every bit, so they are not four
independent Rademacher variables. On a loaded biclique the independent model is
1–6 points optimistic; real dense samples lose 5–16% of the GF(2) rank to such
dependencies, with no detectable association with the per-cell error
(Appendix H).

**What the exact model reproduces.** On the capacity data behind k, on a
10-seed rerun and against a Monte Carlo of the reference, the exact model needs
no constant; the FKS readout with the same p_agree agrees with it within 0.1
points (Appendix H, Fig. 5).

### 2.3 Several answers, aliases, weights and twins

The same construction extends without new parameters; each extension adds one
assumption, that candidates at equal signal have independent distances, and
each was tested before being used for a claim (§3).

- **Several true objects.** If (s, r) has g stored objects, each has the signal
  of a stored fact, and the query is correct if cleanup returns any of them. The
  exact model takes the minimum over g signal distances, treated as independent.
  They are not: by the identity of §2.4, the agreements of two true objects of
  one query are anti-correlated, by −μ²/(1−μ²) per bit. This spreads the minimum,
  so the model should underpredict. An exploratory run at g = 5, D = 256 and
  N = 20–40 found a bias of at most +0.2 points, within two standard errors of
  zero in every cell.
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

**Encoding symmetry is a trade-off.** An asymmetric encoding, s ⊕ ρ(r) ⊕ ρ²(o),
has neither aliases nor twins, at the cost of inverse queries; the model
predicted which encoding wins at a given load on dense WN18RR, with the sign
right in 6 of 6 cells (test 6, Appendix M).

**Aliases set an expected ceiling.** Under the symmetric encoding, a query with
a aliases at equal signal besides its g true answers is answered correctly with
probability g/(g + a) at infinite dimension *if the insertion order of the
candidates is random*. The reference breaks ties by insertion order, so for a
given memory the outcome is deterministic and a set of queries can land above
or below that expectation. The expected ceiling of a set of facts is computable
before storing them (about 0.93–0.95 on dense WN18RR, 0.99 on FB15k-237). It is
a mean, not a bound: single subgraphs exceeded it by up to +4.3 (WN18RR) and +8.2
(FB15k-237) points in test 10 (Appendix M).

### 2.4 Hops on one trace are not independent

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
[@odonnell2014boolean], which gives the approximation. The tests use the exact form (its accuracy at
small N is in Appendix H).

Conditioning on the number of bits where f₁ = f₂ gives the exact joint
distribution of the two distances, and hence the exact probability that both hops
succeed (`two_hop_joint_fast`; the null distances of the two hops are treated as
independent, their correlation being a sum of random signs of order 1/√D). The
prediction: P(both) < p², by an amount that matters only at small N
(Appendix H).

**Deep chains.** For h hops the chain's fact bits are still independent fair
signs (the entities are distinct), the other facts sum to a binomial, and the
trace is the sign of the total; `chain_accuracy_mc` samples this per-bit model
directly, without memories or codewords, and matches the exact two-hop formula.
It predicts that the shortfall below p^h *grows with depth* (tests 7–9, §3.3).

### 2.5 Other families: MAP-I, FHRR, HRR and two block codes

§2.1–2.4 concern MAP-B. Tests 21–22 apply the same procedure — size a memory
from its triples before building it — to five more families. All use the same
encoding f = s ∘ ρ(r) ∘ o (∘ the family's binding, ρ a cyclic shift, of blocks
for BSDC), symmetric in s and o, unbinding by s ∘ ρ(r) and cleanup over the M
symbols of the codebook. A query is right if it returns one of its g true
objects. Let S2 = Σ_j w_j² over the distinct fact vectors (a twin pair counts
once, with w = 2), and a the number of aliases of the query.

| family | atoms | bundling | cleanup | bits per component (as stored) |
|---|---|---|---|---|
| MAP-B (ABM) | ±1 | majority | min Hamming | 1 |
| MAP-I | ±1 | integer sum | argmax ⟨y, c⟩ | ⌈log₂(N+1)⌉ |
| FHRR | phasors e^{iθ} | complex sum (float16) | argmax Re⟨y, c̄⟩ | 32 |
| HRR | N(0, 1/D) reals, circular convolution | real sum (float32) | argmax ⟨y, c⟩ | 32 |
| BSDC-OR | B blocks of L, one 1 per block | OR | blocks hit | 1 |
| BSDC-S | as BSDC-OR | sum (counts) | sum of counts read | ⌈log₂(N+1)⌉ |

**MAP-I and FHRR (FKS).** Scores are treated as independent Gaussians: a
candidate with signal weight w (a true object or an alias) scores
N(w·D, v·D·(S2 − w²)), a null N(0, v·D·S2), with v = 1 for MAP-I and ½ for
FHRR; the accuracy is the probability that the maximum is a true object, by
numerical integration. This is the readout of Frady, Kleyko and Sommer
[@frady2018sequence]; only S2, g, a and M come from the graph.

**HRR: atom multiplicities.** With atoms a ~ N(0, 1/D)^D, a fact
f = s ⊛ ρ(r) ⊛ o and unbinding by involution, write the score in Fourier
components, which are circular complex Gaussians with E|X|^{2k} = k!. The
contribution of fact j to the variance of a candidate c's score is then

  w_j² · Π_a k_a! / D,

with k_a the multiplicity of atom a in the multiset {s, r, c} ∪ atoms(f_j). A
fact sharing no atom with the query and the candidate contributes 1/D, as in
MAP-I; one sharing s, r or c contributes 2/D; facts with the same key (the
other true objects and the aliases) 4/D; a signal candidate adds 14·w²/D of
its own. Plate [@plate1995hrr] gives the variance of convolution products; the
per-fact multiplicities on a real graph are our addition. Without them, HRR
on real subgraphs is predicted 8–20 points too optimistic (test 22, H4).
Declared approximation: the common fluctuation of |K_ν|² across the
candidates of one query is ignored.

**BSDC-OR** (a partitioned Bloom filter). A signal candidate hits all B blocks;
a null hits all of them with probability π = Π_b K_b/L, K_b the occupied bins of
block b, whose exact distribution is P(K = k) = S(n, k)·L!/((L − k)!·L^n). Then

  Acc = E_π[ g · E[ 1/(g + a + X) ] ],  X | π ~ Bin(M − g − a, π).

**BSDC-S.** A null's score is Σ_b Σ_w w·Bin(n_w, 1/L), whose pmf is exact by
convolution; a signal candidate of weight w scores w·B plus the same noise
without its own vector. Integer scores tie often. Test 21 shared a tie in
proportion to each candidate's hazard f/F, an approximation that made the model
pessimistic by +1.7 to +2.1 points. The corrected model (`bsdc_s_exact`) keeps
the same independence assumptions and resolves ties exactly: for each value x
of the maximum, a dynamic programme over the numbers t_g of true objects and
t_a of aliases at x, with every null ≤ x with probability F(x)^n and
J ~ Bin(n, f(x)/F(x)) nulls at x, gives the share t_g/(t_g + t_a + J). It was
derived, not fitted, but sought after the bias of test 21 was known (§8). A
synthetic diagnosis (test 22, step 1) rejected the reviewer's hypothesis of
correlated counts: a model conditioned on the realised trace agreed with the
independent one within 0.3 points.

**The naive comparator.** In every cell the same formula is also computed with
N facts of weight 1, one true object and no aliases (for MAP-B,
`exact.cleanup_accuracy(N, D, M)`): the FKS estimate without the accounting.
Where the two agree the test replicates FKS and says nothing about the
accounting.

**Chains.** For h hops the families other than MAP-B are predicted by the product
of the one-hop predictions of the queries on the path; MAP-B multiplies that
product by the hop-dependence factor of §2.4, computed for uniform facts. This
is the prediction test 22 falsified (§5).

## 3. The record: tally, power, and the tests of MAP-B (tests 1–10)

### 3.1 Tally

Each test was committed — prediction, success and falsification criteria,
harness — before its run; the commit hashes are in the files. Outcomes are
evaluated only with those criteria, and appended to the same file. We count
four groups separately, because they test different things.

| group | tests | what is at risk | primary hypothesis |
|---|---|---|---|
| theory | 1–10 | the MAP-B estimate and its extensions | 8 supported, 2 failed (7, 8) |
| front-end | 11–17 | pipelines around the memory; not the estimate | 3 supported (12, 16, 17), 3 in part (11, 13, 14), 1 failed (15) |
| baselines | 18–20 | ABM against exact structures at equal bits | 18–19 supported (ABM loses); 20 failed (H1, an ABM win, falsified) — all three as predicted |
| generalization | 21–22 | the estimate on five or six families and on new graphs | 21 supported (H1–H9); 22 in part |
| all | 1–22 | | 14 supported, 4 in part, 4 failed |

Test 2's primary hypothesis (H1, D = 16 384) was supported, though a secondary one
was only in part. Test 11's primary is end-to-end accuracy (in part). Test 15's
primary (H1, the relation chosen) was falsified, though its end-to-end H2 was
supported; it is counted as failed. Test 17's primaries (H1, H2) were supported;
H3 was in part and H4 undecided. Test 20 stated its first hypothesis in the
direction we expected to fail ("ABM beats the best code somewhere"); it was
falsified, as predicted, and its model hypotheses H2–H4 were supported. Test 22
counts as in part: at one hop H1, H3, H4 and H7 held and H2 held in part (MAP-B on
dense YAGO3-10), while H6 (chains) and H8 (out-of-sample dimension, MAP-B on
YAGO3-10) were falsified. This count depends on the rule: for tests 2, 11, 13,
14, 15, 17, 20, 21 and 22 the primary hypothesis was not named in the
preregistration; we chose it when writing this tally, after the outcomes (for
test 21 the choice does not matter, every hypothesis held).

Counted per hypothesis, as read from the outcome tables of the twenty-two files,
there are 83: **60 supported, 11 in part, 8 falsified, 4 undecided or
untestable**. Through test 19 the count was 61: 44 supported, 9 in part, 5
falsified, 3 neither or undecided. Test 20 adds 4 (H2–H4 supported, H1
falsified; its H0 is a control and is not counted), test 21 adds 9 (all
supported), test 22 adds 9 (H1, H3, H4, H7 supported; H2, H9 in part; H6, H8
falsified; H5 untestable, since no graph was admitted to it). Three-hop chains,
listed in test 22 as untestable for lack of cells, are not counted.

### 3.2 Outcomes and power

The criteria compare mean errors with thresholds; they do not say how large an
error a test could have seen. The last column gives it. For the error summaries
it is the smallest signed bias detectable by a two-sided 5% test with 80% power,
about 2.8 standard errors of the summary, from `examples/prereg_summary.py`
(2 000 bootstrap draws over seeds within cells, so conditional on the cells). For
differences between arms it is about 1.43 times the reported 95% half-width.

| # | file | what is tested | primary outcome | what the test could detect |
|---|---|---|---|---|
| 1 | `fb15k237.md` | Law IV (k = 0.92) on FB15k-237 [@toutanova2015observed], uniform samples | **supported**: 1.05 points (signed −0.47); hubs neither (+5.5) | bias 0.8–1.0 points |
| 2 | `exact_contract.md` | exact model: D = 16 384; several true objects; dense FB15k-237 | **supported** (H1); 5 hypotheses: 3 supported, 1 in part, 1 neither | bias 0.8–1.3 points; 1.2–1.6 dense |
| 3 | `dependence.md` | the hop dependence of §2.4; P3 reconstructed | **supported**; Law V rejected at 6 SE | correlation within 1.5 SE of prediction, 31–91 SE from zero |
| 4 | `twins.md` | symmetric twins; WN18RR; exact Law VII | all five **supported** | bias 0.8–1.2 points; H1 does not separate the models (Appendix M) |
| 5 | `composition.md` | grounding × reasoning without calibration | all **supported** | bias 1.4–2.1 points |
| 6 | `asymmetric.md` | symmetric against asymmetric encoding | all **supported**; sign right in 6/6 | bias 1.4–4.9 points |
| 7 | `deepchain.md` | hop dependence at depth, tiny codebook | **falsified**: off-path recovery, not modelled | departure up to 18.9 SE |
| 8 | `deepchain2.md` | the same with 1 000 distractors | **falsified** at one hop: the tie rule | 1.8 points at 13.8 SE |
| 9 | `deepchain3.md` | the same with the exact tie rule, two dimensions | all **supported**; p^h rejected at 4.3–8.5 SE | shortfall 1.5–1.8 points seen at ≥ 4.3 SE |
| 10 | `sizing.md` | choosing D in advance; the alias ceiling | all **supported**; Law IV misses on WN18RR | a perfect model passes H1 with P = 0.39; threshold ≈ 1.2–2 SE per subgraph |
| 11 | `human_questions.md` | human-written questions, local LLM front-end | end-to-end **in part** (+7.2 points); memory level supported | a correct contract misses by > 5 points with P = 0.24 |
| 12 | `escalation.md` | the contract as the criterion for passing questions to a larger model | all three **supported** | ≈ 1.8 points |
| 13 | `escalation2.md` | the same, one memory for both paths | routing against random **supported**; the rest **in part** | ≈ 1.6 points |
| 14 | `escalation3.md` | front-end confidence; the observed margin | margin and routing **supported**; the rest **in part** | ≈ 1.5–2.3 points |
| 15 | `memory_preview.md` | the small front-end sees the memory's answer per relation | relation (H1) **falsified**; answer (H2) supported | ≈ 3.3 points (H1), 2.6 (H2) |
| 16 | `shortlist.md` | an embedding shortlist of k = 3 relations | both **supported** | ≈ 6 points |
| 17 | `webqsp.md` | the same front-ends on WebQuestionsSP | H1, H2 **supported**; H3 in part; H4 undecided | ≈ 3–8 points |
| 18 | `equal_bits.md` | ABM against an ideal exact store and a Bloom filter with the same bits | all three **supported** (ABM loses, as predicted) | not computed; expected outcome |
| 19 | `algebra.md` | chains and compiled compositions against dictionary + join / path table | all four **supported** (ABM loses, as predicted) | not computed; expected outcome |
| 20 | `bit_errors.md` | ABM against an exact store with practical codes (polar SC, BCH, concatenated) at equal bits, under i.i.d. bit flips ε ≤ 0.2 | H1 (an ABM win somewhere) **falsified**, as predicted; H2–H4 (the models) **supported** | cell MDE 1.6–13 points (model-based) |
| 21 | `vsa_families.md` | the estimate on five families; FB15k-237 and WN18RR, uniform and dense | all nine **supported** | cell MDE 2.2–5.8 points (lower bound) |
| 22 | `generalization.md` | six families on CoDEx-M and YAGO3-10; two-hop chains | one hop: four supported, H2 in part (MAP-B); H6 (chains) and H8 (MAP-B's dimension) **falsified**; H9 in part | group MDE 0.9–1.0 points at one hop, 1.5–4.0 per hop at two (lower bounds) |

In a single cell the detectable bias is 1.1–8 points. Most mean errors of the
final MAP-B predictor are at or below the noise floor, so passing the criteria
rules out large errors, not errors of a point or two; four of 25 summaries carry
a signed bias whose interval excludes zero, all under 1.4 points (Appendix I).

### 3.3 The tests of MAP-B in brief (tests 1–10)

The full account of each test, with its figures, is in Appendix M; here are the
outcomes and the numbers the rest of the paper uses.

- **Test 1** (Law IV with k = 0.92, uniform FB15k-237 [@toutanova2015observed])
  passed, 1.05 points; the exact model, applied afterwards, gives 0.88.
- **Test 2** (the exact model on new configurations): **0.27 points** at
  D = 16 384, a dimension not used before (signed −0.09); 0.58, 0.86 and 0.32
  with g = 1, 2 and 4 true objects, where Law IV erred by 0.6, 13 and 21; on
  dense FB15k-237, 1.06 at D = 8192 and 2.88 at D = 2048 (signed −2.61, up to
  −8.4 in a cell, pessimistic): that hypothesis **in part**, the primary
  (H1) supported.
- **Test 3** (hop dependence): per-bit correlation −0.0634 / −0.0212 / −0.0072
  measured against −0.0645 / −0.0213 / −0.0071 predicted at N = 10 / 30 / 90;
  independence rejected at 6.0 SE; Law V falsified as an exact law.
- **Test 4** (symmetric twins as weight 2): **0.57 and 0.60 points** on dense
  WN18RR, against −2.1 and −3.8 on average (−8.9 in a cell) without twins; on
  FB15k-237 the twin effect did not replicate.
- **Test 5** (grounding errors, no calibration): 0.75, 1.89, 1.01 and 1.18 points
  for four kinds of error; 10.2 for the calibrated predictor it replaced.
- **Test 6** (symmetric against asymmetric encoding): sign right in 6 of 6
  dense WN18RR cells, −7.0 to +7.8 points (§2.3).
- **Tests 7–9** (deep chains): 7 failed (off-path recovery in a tiny codebook,
  up to +18.9 SE), 8 failed at one hop (the even split of ties, 1.8 points at
  13.8 SE), 9 held with the exact tie rule (p^h rejected at 4.3–8.5 SE; a
  shortfall of 1.5–1.8 points at N = 12).
- **Test 10** (choosing D in advance on 80 unseen dense subgraphs): the promise
  held in every group (measured − target +0.01 to +1.68 points), on average and
  not per subgraph; with Law IV, WN18RR fell 2.8 to 6.2 points short.

## 4. Baselines at equal bits (tests 18–20)

**Tests 18–19: recall, membership, chains and compositions** (full account in
Appendix O). Against an *idealised* exact store whose accuracy is computed, not
run, ABM recalled less in **14 of 14 cells** (for example 0.660 against 0.853 at
D = 8192, N = 400; in the lightest cell, 0.993 against 1.000, the gap is under
one standard error), and a Bloom filter of the same bits made fewer membership
errors in every cell; ABM used 3.8 to 46 times the Fano minimum. The exact model
predicted ABM with mean errors of 0.99 (D = 2048) and 0.54 points (D = 8192),
worst cell 1.97. On FB15k-237 and WN18RR, a realisable store with implicit keys
(1.23·⌈log₂ E⌉ bits per fact) answered two-hop chains by a join and compiled
compositions by a path table: ABM was ahead by more than 2 SE in **0 of 16
cells** for either task (for example 0.101 against 0.848 at D = 2048, N = 200),
and 2 600 to 229 725 times slower than a Python `dict`. The model predicted ABM
with a mean error of 0.32–2.01 points per (graph, D, task), worst cell 7.4. All
seven hypotheses were supported, which here means ABM lost; both outcomes were
expected before the run, confirmations of the obvious with a number attached.
**ABM is not dense**, and no claim of this paper rests on density, compactness
or the algebra.

**Test 20: noisy storage.** The classical argument for distributed codes is that
they degrade gracefully, while an exact store loses a whole key to one wrong
bit. Test 20 measured that argument where it should matter, in memories that
flip stored bits (analog or in-memory hardware), and let the adversary choose
its code. On FB15k-237 and WN18RR (D = 2048 with N = 50–400, D = 8192 with
N = 200–1 600; ε ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 0.03, 0.1, 0.2}; 5 seeds, 20 noise
draws per memory; 112 cells), every stored bit flipped independently with
probability ε: ABM's trace, and every bit of the store, redundancy included.
The store was an XOR filter or a minimal perfect hash with values, protected by
the code a fixed rule predicted best — none, mixed repetition, BCH, polar codes
with successive-cancellation decoding, or repetition inside BCH or polar — with
fewer keys allowed in exchange for a stronger code. ABM's exact model under
noise replaces p by ε + p(1 − 2ε) for every signal candidate.

ABM was below the chosen store in **110 of 112 cells**; the other two were
exact ties at 1.000 (WN18RR, D = 2048, N = 50, ε = 0 and 10⁻⁴), which the
harness reported as wins of ABM because it turned t = 0/0 into p = 0 (§7). At
ε ≥ 0.1 the smallest gap per series was 4.0–12.3 points (at ε = 0.2, D = 2048,
N = 50 on FB15k-237: 0.676 against 0.847), and the normal-approximation
ceiling of Polyanskiy, Poor and Verdú for the store was higher still. The model
predicted ABM within 0.38–0.73 points per (graph, D) group, 1.48 in the worst
cell, also at ε = 0.2; the store's model was within 0.13–0.39 points of an
independent re-estimate. H1 (an ABM win somewhere) was falsified, as predicted,
and H2–H4 supported. The first version of the design used only repetition and
Hamming codes and predicted 21 ABM wins; a hostile review of the design, before
any data, asked for a modern adversary, and with it the model predicted none.
Those 21 wins were wins over weak codes. Choices that favour ABM: SC
decoding, not list decoding or LDPC; a simplified hash layout; ABM's codebook
kept noise-free in the main analysis (corrupting it is catastrophic,
Appendix B). Untested: bursts (descriptive only), read errors, drift. "Degrades
gracefully" holds against an unprotected store, not against a coded one.

## 5. Generalization: five and six families, two new graphs (tests 21–22)

### 5.1 Test 21: five families on FB15k-237 and WN18RR

**Design.** MAP-B, MAP-I, FHRR, BSDC-OR and BSDC-S (§2.5) on subgraphs of
FB15k-237 and WN18RR with N ∈ {100, 400} facts, *uniform* (sampled at random)
or *dense* (breadth-first from a random entity), 10 seeds, all distinct (s, r)
queries up to 200 per seed. For each (graph, sample, N, family) the harness
chose, from the prediction averaged over seeds 0–4, the smallest D reaching
t × the alias ceiling, t ∈ {0.50, 0.85}, then built the memories and measured
all ten seeds. A group is (family, sample, graph), four cells. The uniform
cells replicate FKS — aliases and twins are under 0.5% there and the estimate
with and without accounting differ by 0.1–0.2 points — so the test of the
accounting is the dense groups (H6) and the ablation against the naive
comparator on dense WN18RR, where twins are 12–14% of the vectors and the two
estimates differ by 4.4–8.3 points (H7).

**Outcome.** All nine hypotheses were supported. Mean absolute error in points
(worst cell in brackets):

| group | MAP-B | MAP-I | FHRR | BSDC-OR | BSDC-S |
|---|---|---|---|---|---|
| uniform FB15k-237 | 0.97 (1.8) | 1.06 (3.0) | 0.83 (1.9) | 0.54 (1.5) | 1.88 (2.7) |
| uniform WN18RR | 1.03 (1.3) | 0.97 (1.5) | 0.64 (1.1) | 1.01 (1.8) | 2.08 (3.6) |
| dense FB15k-237 | 0.42 (0.6) | 1.71 (3.9) | 0.64 (1.2) | 1.09 (1.6) | 1.74 (3.3) |
| dense WN18RR | 0.57 (1.2) | 0.56 (1.4) | 0.34 (0.5) | 0.77 (1.8) | 1.81 (3.1) |
| naive FKS, dense WN18RR | 5.19 | 5.92 | 5.38 | 8.51 | 3.90 |

On dense WN18RR the accounting removed 4.6, 5.4, 5.0, 7.7 and 2.1 points of
error (criterion ≥ 2 in all five: BSDC-S at the limit). Cells with |t| > 3.25:
0, 0, 0, 0 and 2 of 16 per family (criterion ≤ 2). With D chosen on seeds 0–4,
seeds 5–9 reached the target within 3 points in 16, 16, 16, 15 and 16 of 16
cells. **BSDC-S was biased**: +1.7 to +2.1 points in all four groups, with
cluster intervals excluding zero; the hypotheses held, the model did not
entirely. Descriptively, at the bits MAP-B spent for its 0.85 cell, the order
of the families was predicted in 73 of 73 pairs differing by ≥ 5 points, and
BSDC-OR — a partitioned Bloom filter — measured 0.94–1.00.

**What it does not show.** MAP-I and FHRR follow FKS; replicating their readout
on real graphs is a check, not a new theory. The MAP-B atoms were derived from
the names alone, identical in all ten seeds (the design table says so; the
consequence was seen only after the run): MAP-B's seeds are not independent,
so its cluster standard errors, and its t in H8, are underestimated. Test 22
salts the atoms with the seed.

### 5.2 Test 22: two new graphs, HRR, the corrected BSDC-S, chains

**Design.** Six families (test 21's, with BSDC-S corrected, plus HRR) on
CoDEx-M [@safavi2020codex] and YAGO3-10 [@mahdisoltani2015yago3], the latter
in the split of Dettmers et al. [@dettmers2018conve], train triples; N ∈ {100,
400, 1 600} (N = 3 200 was dropped by a time rule written before the
predictions of YAGO3-10), uniform and dense, 20 seeds; D chosen on seeds 0–9,
out-of-sample check on seeds 10–19 (H8). Two-hop chains, on functional paths,
measured on the same memories where at least ten seeds had ten chains: only
four configurations qualified, and no three-hop cell. YAGO3-10 entered by a
revision of the design, before its predictions, with an **admission rule** for
the test of the accounting: a graph is admitted if the predicted gap between
the estimate and the naive comparator, in the dense one-hop cells, is ≥ 2
points in at least three families. Neither graph was admitted (gaps 0.8–1.6
points, HRR's 18–21 points coming from multiplicities, not twins): on both,
H1–H2 replicate FKS (for HRR, they test the multiplicities), and H5, the
ablation of the accounting, was untestable.

**Outcome at one hop.** Mean absolute error in points per group (worst cell in
brackets for MAP-B on dense YAGO3-10):

| group | MAP-B | MAP-I | FHRR | HRR | BSDC-OR | BSDC-S |
|---|---|---|---|---|---|---|
| CoDEx-M uniform | 0.34 | 0.61 | 0.24 | 0.57 | 0.80 | 0.49 |
| CoDEx-M dense | 0.36 | 0.62 | 0.64 | 0.70 | 0.80 | 0.98 |
| YAGO3-10 uniform | 0.78 | 0.73 | 0.45 | 0.85 | 0.56 | 0.81 |
| YAGO3-10 dense | **4.02** (7.2) | 0.58 | 0.29 | 0.61 | 0.58 | 0.69 |

In 23 of 24 groups the error was 0.24–0.98 points. HRR's multiplicities held
(H4): the naive Gaussian erred by 8.2, 20.7, 10.8 and 17.5 points, a gain of
7.6, 20.0, 10.0 and 16.9 against thresholds 4.0, 10.0, 5.5 and 8.5. The
corrected BSDC-S had signed errors +0.83, +0.49, −0.07 and +0.18 (H3,
|signed| ≤ 1); its postdiction on test 21's cells had moved the bias from
+1.7…+2.1 to +0.1…+0.6 (`generalization_postdiction_bsdc_s.json`, a
postdiction, not a confirmation). At most one cell per family and graph had
|t| > 2.861 (H7). **MAP-B failed on dense YAGO3-10**: the model was optimistic
by 3.5 points on average, by 3.8, 5.3, 5.8 and 7.2 in the four cells with
N ≥ 400, and the dimension chosen in advance missed its target by more than 3
points in 4 of 12 cells (H8 falsified; every other family and graph met it in
11 or 12 of 12). MAP-I and FHRR on the same subgraphs did not fail, so the
defect is specific to the binary model. Our candidate, not verified, is the
GF(2) dependence of facts on even cycles (§2.2), which makes the model
optimistic by up to 6 points on a loaded biclique.

**Chains.** Of 18 chain groups, scored per hop (accuracy^(1/2)), 13 had a mean
error ≤ 3 points, three 3.3–4.8, and two exceeded 5 (CoDEx-M dense, BSDC-S 5.56
and MAP-B 5.33), each a single cell whose minimum detectable error, 3.6–4.0
points, was above the support threshold. The criterion was "≤ 5 in every
group": H6 is **falsified**. Against a comparator keeping each query's
signature but no twins or aliases, the estimate gained at least its threshold
in 14 of 18 groups and lost in 2 (H9 in part). Three-hop chains were not
tested.

**What it shows.** At one hop, the estimate generalised to two graphs never
used before for five families of six, including the two models that are ours
(HRR's multiplicities, BSDC-S's ties). It did not generalise for MAP-B on the
denser graph, it did not predict chains, and it said nothing new about the
accounting, which remains tested only by test 21.

## 6. Front-end tests (tests 11–17), in brief

These tests change the pipeline around the memory — a language model choosing the
relation, a router, an embedding shortlist — while the memory (D = 16 384) and its
contract stay fixed. Apart from the memory-level check of test 11, **none of them
tests the capacity theory**. Details, numbers and caveats are in Appendix F.

In brief: at the memory level of test 11 the model agreed (0.807 measured
against 0.803, +0.4 points), while the end-to-end contract was 7.2 points low,
beyond the 5 set (**in part**); as a router the contract beat random choice by
2.7–4.8 points (tests 12–14); showing the front-end the memory's answers did not
improve its choice of relation (test 15, H1 **falsified**, −0.3 [−2.6, +2.0]);
an embedding shortlist added +19.9 and +22.7 points of answer accuracy (tests
16–17), after which entity linking by name loses a quarter of the
WebQuestionsSP questions for every front-end.

## 7. Falsifications, corrections and retractions kept on record

| retired or corrected claim | what showed it |
|---|---|
| Law VI: failure grows with out-degree | at constant load, accuracy is flat (64–73%) for out-degree 1–24; the effect was load |
| geometric decay of raw chaining, z_eff = √D·ρ^h | two-trace raw chaining measures 0% where it predicts ~40%; T ⊕ T = 1 is the sharp form |
| **Law V as an exact law** (hops independent) | preregistered test 3: correlation −μ²/(1−μ²) per bit, independence rejected at 6 SE |
| k = 0.92 as a constant of the memory | the exact model reproduces the capacity data with no constant (§2.2) |
| P3's "consistent 5% bias, unexplained" | the Gumbel-ratio approximation; the exact model predicts the reconstructed gains within 1–4% |
| the composition law as parameter-free (v1.4 and before) | its script took the clean accuracy from the measurements and calibrated the spurious-fact curve; replaced by a preregistered prediction without calibration (test 5) |
| "~3% mean deviation" for the composition law | the script's own data gave 4.3 (3 seeds) and 3.6 (10), and 10.2 for missing facts |
| sublinear cleanup; a fixed acceptance threshold | Appendix B |
| our deep-chain prediction with a tiny codebook | preregistered test 7: off-path recovery dominates, up to +18.9 SE |
| the even split of ties | preregistered test 8: with 1 000 distractors after the target, it misses a single hop by 1.8 points (13.8 SE); replaced by the exact rule |
| the bits-per-fact crossover (74–81%, v1.8) | no committed script reproduces it; withdrawn (audit 2026-09-30) |
| "the known bound over-provisions dimension sixfold" (v1.8 abstract) | a proof constant against a measurement; kept in Appendix E as a sizing remark, out of the abstract |
| "exact" without the GF(2) caveat | facts on even cycles are dependent; several points on a loaded biclique (§2.2) |
| "a fixed size, membership by one distance" as properties of the design (v1.11 §7, untested) | preregistered test 18: at equal bits an idealised exact store (computed, not run) recalls more and a Bloom filter makes fewer membership errors in every cell |
| "an algebra: binding, unbinding and composition" as the residual value (v1.12 abstract and §7, unmeasured) | preregistered test 19: on two-hop chains and compiled compositions a dictionary + join or a path table of the same bits is at least as accurate in every cell, and orders of magnitude faster |
| "aliases set a ceiling no dimension can exceed" (v1.14 §3.3) | recomputed from test 10's data: the ceiling is an expectation under random insertion order; single subgraphs exceed it by up to +4.3 (WN18RR) and +8.2 (FB15k-237) points |
| "noisy hardware may favour ABM: it degrades gracefully" (first design of test 20, `99ccc03`: 21 predicted ABM wins against repetition and Hamming codes) | hostile review of the design, before any data: against polar and BCH codes the model predicted no win; test 20 observed none (110 losses, 2 exact ties) |
| test 20's harness: t = 0/0 converted to p = 0 | two exact ties at 1.000 reported as ABM wins; the criterion's text ("beats") prevails; the frozen output was not changed |
| BSDC-S with ties shared by hazard (test 21) | pessimistic by +1.7 to +2.1 points in all four groups; replaced by exact ties (postdiction +0.1…+0.6, then test 22 H3 supported) |
| MAP-B atoms derived from names alone (test 21) | identical in all seeds, so MAP-B's standard errors are underestimated; salted with the seed in test 22 |
| the MAP-B estimate carries over to new graphs (implicit since test 10) | test 22: 3.5 points optimistic on dense YAGO3-10 (worst cell 7.2), D short of target in 4 of 12 cells |
| two-hop chains as the product of one-hop estimates (MAP-B: times the dependence factor) | test 22, H6 falsified: two groups above 5 points per hop |

**Self-loops** (s, r, s), modelled since 2026-09-30, and four result files whose
scripts were lost are documented in Appendix J; neither changes an outcome.
Appendix J also lists the implementation changes made after the tests, which
alter what today's code computes for tests 4, 5, 6, 10 and 11 (the reference
implementation is not frozen; the specification is).

## 8. Limits

Stated once, with where each comes from.

- **MAP-B on dense graphs.** The estimate for the family this project built was
  3.5 points optimistic on dense YAGO3-10, up to 7.2 in a cell, and its chosen
  dimension fell short in 4 of 12 cells (test 22). The candidate cause, GF(2)
  dependence on even cycles (§2.2), is not verified; on dense FB15k-237 and
  WN18RR the same model was within 0.4–0.6 points (test 21).
- **Chains.** Two-hop predictions on real graphs were falsified (test 22, H6),
  with few chains per cell; three hops were never tested on a real graph. The
  MAP-B dependence factor is computed for uniform facts.
- **The accounting is tested once.** Its only test is dense WN18RR in test 21;
  neither new graph of test 22 was admitted. On uniform samples, and on both new
  graphs, the estimate is a replication of FKS.
- **BSDC-S's correction was sought after its bias was seen** (test 21), on
  synthetic data; it held out of sample (test 22, H3), but the path depended on
  the outcome, and a residual of up to +1.4 points remains on synthetic data at
  small B, unexplained.
- **Harness defects found after runs.** Test 20 read t = 0/0 as p = 0 (two ties
  reported as wins; the criterion's text prevails). Test 21 used the same MAP-B
  atoms in every seed, which understates MAP-B's standard errors there.
- **Power.** Minimum detectable biases: 0.8–4.9 points for the summaries of
  tests 1–6 (bootstrap over seeds); 2.2–5.8 per cell in test 21, 0.9–1.0 per
  group at one hop and 1.5–4.0 per hop at two in test 22 (lower bounds, from a
  model of the standard error). Errors of a point or two are not excluded anywhere.
- **Scale and setting.** N ≤ 1 600 (≤ 400 in test 21); one level of binding;
  atoms i.i.d.; closed-world queries over stored (s, r) pairs; bits per
  component are storage formats, not entropy, and the codebook is not counted.
  Cells of one (graph, sample, N) share subgraphs, so groups are not independent.
- **ABM is dominated** at equal bits on every task measured (tests 18–20); the
  estimate is of use only where a distributed representation is required anyway.
- **Anteriority** is checkable only from test 20 on, and only for the stamped
  files (§1.1).
- **No external replication**; tests 20–22 are not yet wired into
  `replicate.py` as named tests (their data are). Every number was produced by
  the author.

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
compare capacity across architectures empirically. Our §2 is a refinement of
that line, not an alternative: the same readout probability, computed with the
classical majority-agreement probability of a binary majority trace (discrete
and Gaussian versions compared in §2.2), and extended to the reference tie rule,
several answers, aliases, weights, symmetric twins and hop dependence — and,
for the other families, HRR's multiplicities and BSDC-S's exact ties — with
preregistered tests on real graphs. Resonator networks [@frady2020resonator]
factor a bound vector by iterating cleanup on all factors at once; §2.4 concerns
the simpler hop-by-hop cleanup, whose errors are correlated through the shared
trace. Clarkson et al. also prove that the reliability of nested bundling decays
with depth (their Lemma 17); the hop dependence of §2.4 concerns a different
composition — cleanup between hops on one flat bundle. Hopfield networks,
holographic embeddings, retrieval augmentation, soft theorem proving and exact
and approximate membership (Bloom, Bloomier and XOR filters, the Fano minimum)
are discussed in Appendix K.

*Families and data of tests 21–22.* MAP-I, FHRR and HRR follow the taxonomy of
the surveys [@kleyko2022survey; @schlegel2022comparison]; HRR is Plate's
[@plate1995hrr], whose variance analysis of convolution products the
multiplicities of §2.5 apply fact by fact. The block codes are sparse binary
distributed codes in the line of Rachkovskij and Kussul
[@rachkovskij2001thinning], without their thinning; BSDC-OR is a partitioned
Bloom filter [@bloom1970space]. The graphs are FB15k-237
[@toutanova2015observed], WN18RR [@dettmers2018conve], CoDEx-M
[@safavi2020codex] and YAGO3-10 [@mahdisoltani2015yago3; @dettmers2018conve].

## 10. Open problems

1. **A model of MAP-B on dense graphs** (next). Test 22 located the failure but
   not its cause; the candidate is the GF(2) dependence of facts on even cycles
   (§2.2, Appendix H), which a model would have to keep in the cleanup as well
   as in the vote. It should be preregistered on YAGO3-10 dense before anything
   else is claimed for MAP-B.
2. **A dense graph admitted for the accounting.** The accounting has one test
   (WN18RR, test 21). A second needs a graph that passes test 22's admission
   rule (gap ≥ 2 points in at least three families), chosen before its
   predictions are seen.
3. **Chains on real graphs.** Enough chains per cell to have power at two hops,
   a model for paths whose hops have several true objects, and any test at
   three hops.
4. **Real hubs.** The hub residual of test 2 vanishes once twins are counted
   (Appendix I), but only in an exploratory recomputation; new hubs are untested.
5. **Joint independence of candidates.** The cost of treating candidates at
   equal signal as independent is at most about 0.2 points at g = 5 (§2.3),
   though the expected direction is underprediction; that of treating the null
   distances of consecutive hops as independent is unmeasured.
6. **Off-path recovery.** In tiny codebooks a uniform 1/M recovery
   overestimates chance recovery (test 7); no clean model yet.
7. **Two-level contracts.** On questions written by people the memory level is
   predicted within half a point, but the end-to-end contract errs by 7 points,
   through the audit's sampling error and a correlation between front-end and
   memory success that the product formula ignores (test 11). As a router the
   contract's share of a decision is bounded by the memory's share of the error
   (tests 12–14); with a shortlist the largest loss is entity linking by name
   (tests 16–17), outside the memory.
8. **The value of the algebra.** Richer queries (conjunctions, analogies, role
   filling) and a bitpacked runtime are unmeasured; nothing measured so far
   suggests they would reverse the results of tests 18–20.

Three further theoretical questions are listed in Appendix G.

## 11. Conclusion

An estimate of cleanup accuracy computed from the triples, before the memory
exists, held at one hop within about a point on average for five
vector-symbolic families on two graphs it had never seen. Its readout is that of
Frady, Kleyko and Sommer; what is ours — the accounting of twins, aliases,
several answers and the codebook, HRR's atom multiplicities, BSDC-S's exact
ties, MAP-B's exact majority model — was tested by preregistration, the
accounting only once. It failed where the project began: for MAP-B on a dense
new graph, and for two-hop chains.

ABM, the MAP-B memory studied most, is a dominated design. It is not a better
retriever, not a compressor, and not better than an exact store or a Bloom
filter with the same number of bits in any cell we measured, nor than a
dictionary and a join on two-hop chains or compiled compositions, nor than a
coded exact store when the stored bits are noisy. Its accuracy can be computed
without fitted parameters, under stated idealisations, from its dimension, its
load, its codebook and the structure of what it stores — on average and not per
cell, with errors of a point or two not excluded by our data, and not on dense
YAGO3-10.

Of the ten tests of the MAP-B theory, eight supported their primary
hypothesis. Two failed: test 7 showed that chains recover by chance in tiny
codebooks, which the model does not cover; test 8 showed that splitting ties
evenly is wrong when distractors follow the target, and the exact tie rule
replaced it. Test 3 was supported, and in being supported withdrew a law of our
own earlier versions (independent hops) by the amount the theory predicted. The
three baseline tests (18–20) went against ABM, as the model said they would. Of
the seven front-end tests, which do not test the estimate, one failed: test 15,
a wrong prediction about the front-end. Of the two generalization tests, 21 was
supported and 22 in part. The predictions were fixed before the data; the
outcomes are in §3–§6, the failures and the changes they caused in §7 and in the
preregistration files. Reproduction instructions are in Appendix L.
---

## Appendix A. The operators are necessary and independent

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

## Appendix B. Other capacity results

**Typed projection (P3).** Restricting cleanup to a typed sub-codebook S raises
capacity; with the exact model the gain is the ratio of the two exact N\*. The
original experiment is lost (Appendix J); its reconstruction was preregistered (test 3).

**Robustness asymmetry.** Flipping a fraction ε of trace bits degrades gracefully
(86% → 42% at ε = 20%); corrupting the codebook is catastrophic (ε = 5% halves
accuracy), because key construction, cleanup targets and stored content corrupt
together. The item memory is the trusted base of the model.

**No practical sublinear speedup for cleanup in this regime.** A metric-tree
index visits 4 487 of 5 000 nodes at D = 2048: a query sits at 0.446–0.483·D from
its *own* codeword and at 0.500·D from every other, a gap of 34–111 bits.
Triangle bounds cannot exploit it. Bit-sampling LSH is sublinear only in name:
its exponent is ε = ln(1 − 0.446…0.483)/ln(0.5) ≈ 0.85–0.95 (the usual LSH ρ,
renamed here), under the 1/c ≈ 0.89–0.97 bound, so a query costs about M^ε with
M^ε tables. Cleanup is linear with a low constant (6.1 ms at M = 10⁵,
bandwidth-bound).

**Acceptance must be an extreme-value test in M.** A fixed confidence threshold
of 0.75 accepted 0 of 30 correct answers on a memory predicted at 1.00. Cleanup
reports a maximum of M z-scores, so the threshold must grow with M: accepting iff
z ≥ Φ⁻¹((1 − α)^{1/M}) caps false accepts at α (measured 30/30 true accepts, 3/300
false at α = 0.01).

## Appendix C. The Memory Calculus

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
  rectangle normalize to 1, and the vote sees them as dependent (§2.2).
- **Compose is not a rule.** For facts sharing a bridge,
  nf(f₁ ⊕ f₂) = c_A ⊕ ρr₁ ⊕ ρr₂ ⊕ c_C: the two-hop fact is generated by
  normalization, and since c_B ⊕ c_B = 1 it contains no information about the
  bridge (**Bridge Elimination**).
- **No-go.** T ⊕ T = 1: reusing a raw decode, which contains T, inside a later key
  on the same trace collapses deterministically. Depth requires symbolization.
- **Cost semantics.** Exact steps are free and certain; ⊞ spends capacity (§2);
  cleanup spends reliability. Composing reliabilities multiplicatively is sound
  only up to the hop dependence of §2.4, which is now a known term rather than an
  assumption.

## Appendix D. Earlier (not preregistered) results

Only P3 was later preregistered (test 3); P1, P2 and the compose cost were not.

**P1 — depth is cheap.** From Laws IV and V, D_min(h) = Θ(N ln M) + Θ(N ln h).
At constant load N = 120, target chain accuracy 95%, 10 seeds, geometric grid in
10% steps:

| h | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| D_min | 3846 | 4402 | 5277 | 5528 | 6248 | 7017 | 7697 |

Growth 1 → 64 hops: 2.0× (logarithmic fit R² = 0.992, linear 0.798). The
prediction is modest — logarithmic growth follows from p^h and a Gaussian tail —
and the measurement adds that the constant is small. Load, not depth, is the
limiting resource (Fig. 1).

![P1. Minimum D for chain accuracy ≥ 95% at N = 120 (mean ± SD, 10 seeds, 10%
grid).](figures/fig2_depth.png){width=60%}

**P2 — the compiler.** Composing facts exactly into a second trace before
decoding turns two-cleanup queries into one-cleanup queries at the compiled
trace's load. Measured: 99% vs 82% (40 chains), 89% vs 25% (80), 44% vs 3%
(160); the interpreted path tracks p² (Fig. 2). This compares ABM with itself;
against a path table of the same bits (test 19, §4) the compiled trace is never
ahead, so P2 is superseded as evidence of value.

![P2. Two-hop queries answered by two cleanups (interpreted) or one cleanup on a
compiled trace T₂; dotted: p².](figures/fig4_compiler.png){width=60%}

**P3 — typed projection.** Reconstructed and preregistered (test 3): measured
gains 1.84 / 2.21 / 2.63× over 2 000 / 8 000 / 32 000 distractors, exact
prediction 1.86 / 2.26 / 2.75×. The Gumbel-ratio prediction of earlier versions
(1.94 / 2.37 / 2.82×) overestimated more; the "consistent 5% bias" we reported
then was that approximation's error.

**Compose cost.** Composition by two independent cleanups costs p²: measured
0.77 / 0.23 / 0.01 against 0.77 / 0.21 / 0.02 at three loads (10 seeds).

## Appendix E. ProofWriter, sizing constants, the capacity contract and pilots

**Against an exact store on ProofWriter.** On ProofWriter
[@tafjord2021proofwriter] (open-world attribute fragment, 10 seeds), a forward
chainer whose only truth oracle is one Hamming distance to the trace reaches
99.8% ± 0.3, 99.1% ± 0.3 and 92.4% ± 1.4 at depths 0, 2 and 5 on the problems
four grammatical patterns parse (35–56% of them, so the subset is selected;
majority baseline 42%). The seed runs use the first 100 parsable problems per
depth (88 at depth 2, the only parsable ones among its first 300); the same
problems appear in every run, so the ± covers encoding seeds only, not problem
sampling. The same chainer with a Python set in place of the trace is
deterministic; scored once on 150 problems per depth, it reaches **100% at every
depth**: the parser and the chaining are sound, and the whole loss is the
oracle's. The trace takes 512 bytes where a minimal exact encoding takes 8–23
(Fig. 3). The oracle's z ≥ 3 threshold is the M = 1 case of Appendix B and
controls false accepts per test, not per proof.

![ProofWriter, parsable subset: accuracy by depth (10 seeds, 95% CI) against the
majority baseline.](figures/fig3_proofwriter.png){width=60%}

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
a minimal exact encoding of the same facts fits about 360 of them in that
kilobyte, at 100% (Fig. 4). That count uses the test's own vocabulary, 2N
entities and 13 relations (2·log₂(2N) + log₂13 ≈ 22.7 bits per fact at N = 361);
with the vocabulary of FB15k-237 (about 14 500 entities, 237 relations, 35.5 bits
per fact) it is about 230.

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
near-perfect recall, so the pilot illustrates the level decomposition of Law
VIII and tests nothing in the capacity theory. It is a case, not a sample.

**Not yet re-tested.** The compiler-ranking dry run (three simulated extractors
ranked by a per-query contract before any query, 2/2 resolvable pairs at 95% CI)
and the cross-domain invariance results (four synthetic topologies, mean
deviation 2.4%; an exact 1/g aliasing factor for single-relation chains,
corroborated at g ∈ {2, 3, 4, 8}) predate the exact model and were computed with
Law IV; they are consistent with it but have not been re-run.

## Appendix F. The front-end tests in detail

**Summary, as printed in the body of v1.15.**

- **Test 11** (743 SimpleQuestions v2 questions whose triple is in FB15k-237, a
  2-billion-parameter local front-end): at the memory level the model agreed
  (0.807 measured against 0.803 recomputed per question, +0.4 points); end to
  end, 0.361 against a contract of 0.289 [0.219, 0.367], inside the interval but
  7.2 points above, beyond the 5 we had set: **in part**.
- **Tests 12–14** (the contract as a router to a larger model): ranking by the
  contract beat random choice by 2.7–4.8 points; the observed margin of an
  answer added 2.6 points [0.9, 4.1] to the small model's confidence. Most of the
  error is the small front-end's.
- **Test 15** (the front-end sees the memory's answer per candidate relation):
  relation choice did not improve (H1 **falsified**, −0.3 [−2.6, +2.0]; we had
  predicted +5 to +10); answers did (+2.8 [+1.0, +4.6]).
- **Tests 16–17** (an embedding shortlist of 3 relations): answer accuracy +19.9
  points on SimpleQuestions and +22.7 on WebQuestionsSP. Entity linking by name
  then loses a quarter of the WebQuestionsSP questions for every front-end.

The baseline of each comparison is not the same: in test 15 it is the small
model with all candidate relations and no preview (the answers recorded in test
14), on 743 questions (0.402 relation, 0.354 answer); in test 16 it is the small
model with all candidates *and* the preview, i.e. the preview arm of test 15
from an earlier execution, restricted to the 643 test questions (0.409, 0.393);
in test 17 it is the same preview prompt run within the test, on 515
WebQuestionsSP questions (0.383, 0.381).

**Caveats for tests 11–17.** *Multiplicity:* no correction was preregistered.
Within its own test, Bonferroni leaves test 15's H2 (+2.8 [+1.0, +4.6], z ≈ 3.0,
two hypotheses) and test 14's observed margin (+2.6 [+0.9, +4.1], z ≈ 3.2, five
hypotheses) significant (lower bounds about +0.7 and +0.5); across the 61
hypotheses of tests 1–19 neither would survive. *Contamination:* SimpleQuestions and
WebQuestionsSP are public since 2015–2016 and may be in the pretraining data of
Gemma, Qwen and bge; we did not test for it, and it would favour the front-ends
equally rather than the memory. *Determinism:* the language models ran at
temperature 0, but about 0.3% of the small model's choices (2 of 643) and 0.9%
of the larger one's (6 of 643) changed across reruns, and neither the llama.cpp
build nor a seed was recorded. In test 17 a quarter of the questions are lost to
entity linking before any front-end acts, which scores 0 for every arm and
dilutes each difference by about a quarter.

**Test 11 — questions written by people.** The 743 questions of SimpleQuestions
v2 whose Freebase triple is in FB15k-237 were answered by a realistic pipeline:
link the subject by name, let a local 2-billion-parameter model (Gemma 4 E2B,
2-bit, llama.cpp, temperature 0) pick the relation among those stored for it, and
query 28 memories of 26–2 394 triples at D = 16 384. The contract was issued after
an audit of 100 questions and before the 643 test questions: front-end accuracy
0.36 [0.27, 0.46] times the memory accuracy predicted by `abm.exact`, 0.802, gave
0.289 [0.219, 0.367]. Measured: **0.361**, inside the interval but 7.2 points
above, beyond the 5 we had set — supported in part. The threshold was tight. The
audit's sampling error, 0.802·√(0.36·0.64/100) = 3.85 points, and the binomial
error of 643 questions, 1.79 points, give a predictive SD of 4.24 points. A
correct contract would miss by more than 5 points with probability 0.24, and
+7.2 is 1.7 SD: "in part" is compatible with audit noise alone. Scored strictly,
counting an answer only when the relation was also right, the result is 0.345
(+5.7 points), still above the 5. Questions are clustered in memories, so we
resample the 28 memories (10 000 draws): 95% intervals [0.318, 0.402] end-to-end
and [0.305, 0.385] strict. The memory level agreed with the model: given the
true (s, r), 0.807 against 0.803 recomputed per question (0.802 in the
contract), +0.4 points with a cluster-bootstrap SD of 1.4 points. By memory
size: 0.877 against 0.878 on the 20 memories of 500–999 triples (464 questions),
0.624 against 0.607 on the 7 of 1 000 or more (178 questions). The gap is the
front-end's: the audit underestimated it (0.36 against 0.41 on the test), and
front-end and memory are not independent — on questions the front-end gets right
the memory answers 85%, against 78% on the others (z ≈ 2.3 treating questions as
independent; resampling the 28 memories, z = 2.16 and 95% interval of the
difference [0.010, 0.142]; exploratory, not preregistered) — the first limit the
preregistration had declared.

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
relation in 41% of questions.

**Tests 15–17 — the front-end.** Tests 12–14 located most of the error in the
small front-end, so these three tests change only the front-end; the memory
(D = 16 384) and its contract are unchanged. All use a cluster bootstrap over the
memories (10 000 draws). Each baseline is stated with its prompt and its
questions, since they differ.

*Test 15* (743 questions). Each candidate relation was shown with the answer the
memory returns for it (the preview prompt). Baseline: the same small model with
all candidates and no preview, the answers recorded in test 14 (an earlier
execution): relation 0.402, answer 0.354. With the preview the small model did
not pick the right relation more often (H1, 0.402 → 0.400, −0.3 points
[−2.6, +2.0]: **falsified**; we had predicted +5 to +10), but it answered right
more often (H2, 0.354 → 0.382, +2.8 [+1.0, +4.6]: supported); it changed its
choice on 186 questions. Exploratory per-question counts suggest it moved toward
options where the memory returns a plausible answer.

*Test 16* (the 643 test questions; the 100 audit questions excluded). A
33-million-parameter embedding model (bge-small-en-v1.5) kept the k = 3
relations closest to the question (k chosen on the 100 audit questions; recall
97.5% on the 643 test questions), and the small model chose among them with the
preview prompt. Baseline: the small model with all candidates and the preview,
i.e. the preview arm of test 15 from that earlier execution, restricted to these
643 questions: relation 0.409, answer 0.393. Relation accuracy rose to 0.669
(+26.0 [+21.7, +30.3]) and answer accuracy to 0.593 (+19.9 [+15.9, +24.1]); both
supported, about twice the predicted gain. The embedding's top choice alone,
reported but not a hypothesis, answered 0.647 (relation 0.796).

*Test 17* (515 one-relation questions of WebQuestionsSP [@yih2016webqsp], taken
from search queries; 27 memories). SimpleQuestions was written by people who saw
the triple, so test 17 repeated the comparison on natural questions. All arms
were run within the test. Baseline (G): the small model with all candidates and
the preview prompt of test 15: relation 0.383, answer 0.381. Answer accuracy was
0.598 for the embedding alone (E), 0.608 for the shortlist with the small model
(S) and 0.656 for Qwen3-4B with all options (Q): E against G +21.7
[+16.4, +27.3] and S against G +22.7 [+17.8, +28.1], both supported; E against Q
−5.8 [−8.9, −3.2], so non-inferiority within 5 points was only in part; E
against S −1.0 [−3.2, +1.2], undecided. The gain of test 16 was therefore not an
artefact of the dataset. The next bottleneck is not the relation: linking the
entity by name was right in 75% of the WebQuestionsSP questions, the same for
every front-end, so it loses a quarter of them before any choice is made.

## Appendix G. Further open problems

1. **Negation and quantifiers** in the algebraic truth oracle.
2. **Distributional confluence** for nested probabilistic redexes.
3. **The ABM complexity class** (polynomial D, O(1) controller).

## Appendix H. Details of the estimate

**What the exact model reproduces.** Against the capacity data behind k, the
exact N\* falls inside every 95% interval at D = 512–4096, including the small D
where k drifted, on the original data and on a 10-seed rerun: **k = 0.92 is the
error of the asymptotic treatment of M and of the approximate p_agree, not a
property of the memory.** The finite-M readout of Frady, Kleyko and Sommer, given
the same exact p_agree, agrees with our binomial computation within 0.1 points on
every tested configuration (Appendix H); what this paper adds is the accounting
of §2.3–2.4 and the preregistered validation. A Monte Carlo of the reference at
D = 256, N = 20, M = 43 gives 0.714 for an exact prediction of 0.722, where Law IV
gives 0.792. These checks use data we had already seen; the tests of §3 use data
we had not (Fig. 5).

![Capacity: measured 50%-accuracy load N\* (10 seeds, 95% CI) against the exact
model (no parameter) and Law IV with k = 1 and with the fitted k = 0.92. The
codebook grows with the load (M = 2N + 11).](figures/fig1_capacity.png){width=60%}

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

**The hop lemma at small N.** The approximation is
poor at small odd N, where there are no ties: −0.127 against −0.164 at N = 5
(22% low) and −0.212 against −0.333 at N = 3; at even N it is within about 1%
(−0.0637 against −0.0645 at N = 10). The tests use the exact form. We have not
found the lemma stated for the decoding of vector-symbolic bundles, and earlier
versions of this paper assumed the opposite. The prediction: P(both) < p², by an amount that matters only at small N. The old measurement could not see it — its error on the per-bit correlation was ±0.033, the effect at its N = 90 is −0.007. As an approximation p^h remains good at large N: in the depth experiment of Appendix D (N = 120), |Acc − p^h| = 0.007 over h ≤ 24.

**Ties: the even split used in tests 1–8.** Tests 1–8 used an
even split of ties, truncated to first order: a tie with one null
codeword counts one half, and ties with two or more are dropped. The exact
average over random positions is (P(null ≥ d)^{n+1} − P(null > d)^{n+1}) /
((n+1)·P(null = d)), with n = n_b + n_a. For one cleanup at N = 12 with 1 000
distractors, the truncation lies 0.13 points below it at D = 320 (test 8's
dimension) and 0.23 at D = 256. The truncation is accurate where ties are rare
or positions mixed; test 8 found where it is not (Appendix M).

**Even cycles.** In a rectangle (a,r,b), (a,r,c), (d,r,b), (d,r,c) the four
vectors multiply to the identity on every bit. By exhaustive counting, p_agree for each is
0.625, not 0.6875. On a K₄,₅ biclique with 20 facts at D = 64 the model predicts
0.930 against a measured 0.887 ± 0.008, and a model that keeps the linear
dependence in the vote (but not in the cleanup) predicts 0.882; under load, with
distractor facts, the independent model is 1–6 points optimistic and the linear
one 3–4 points pessimistic (`examples/cycles_probe.py`, exploratory). Knowledge
graphs do contain such dependencies. In the dense FB15k-237 samples of test 4,
14% of distinct fact vectors lie in a 4-dependency (four facts whose XOR is the
identity) at D = 2048 and 27% at D = 8192; the GF(2) rank deficit, which also
counts longer even cycles, is 5–16% across the dense samples of tests 2 and 4,
and zero in uniform ones. At this sample size it shows no detectable association
with the per-cell error: taking deviations from the mean of each (graph, D, N)
group, the correlation of rank deficit with predicted − measured accuracy is
+0.13 [−0.12, 0.36] for dense FB15k-237 at D = 2048 and +0.03 [−0.15, 0.20] over
all 240 dense cells (bootstrap over cells; `examples/cycles_residuals.py`,
exploratory, on published data). Since cycles make the model optimistic, they
cannot be the main cause of the pessimistic dense bias (−0.81 at D = 2048); a
small optimistic share masked by another effect is not excluded.

**Capacity data.** Against the capacity data behind k, the exact N\* is 50.5 / 87.5 / 155.5 / 277.5
at D = 512 / 1024 / 2048 / 4096, measured 50.2 ± 4.5 / 87.1 ± 3.1 / 158.8 ± 5.3 /
280.0 ± 6.4: inside every interval, including the small D where k drifted. The
script behind these measurements was lost; a rerun with the protocol they imply
(`examples/capacity_seed10.py`, 10 new seeds) gives 50.6 ± 1.7 / 86.8 ± 2.2 /
156.2 ± 3.4 / 279.6 ± 5.5, again with the exact model, 50.2 / 88.7 / 155.3 /
277.4 on that script's grid, inside every interval. **k = 0.92 is the error of
the asymptotic treatment of M and of the approximate p_agree, not a property of
the memory.**

*Relation to Frady, Kleyko and Sommer.* Their finite-M readout, ∫φ(x)Φ(·)^{M−1}dx
with Gaussian crosstalk, is not an asymptotic law of the Law IV kind. We
instantiated it for the majority trace with our exact p_agree(N) (signal
distance N(Dq, Dq(1−q)), null distances N(D/2, D/4), g true answers; cells with
aliases not covered; `examples/fks_gaussian_vs_exact.py`, exploratory). On every
configuration above the two agree: N\* 50.2 / 88.7 / 155.3 / 277.4 for both
(gaps under 0.03 facts), mean absolute error 0.27 and 0.59 points on test 2
parts A and B for both, and 0.723 against 0.722 in the D = 256 cell: within 0.1
points on every tested configuration, and within 0.005 on test 2. A continuity correction makes the
Gaussian worse. Among the sampled points of a sweep over D and M, they differ by
more than half a point only at D = 64 with M ≥ 4 096 and at D = 256 with
M = 65 536, by at most 2.8 points. So the gain over Law IV comes from treating M
finitely with the exact p_agree, which the FKS integral does once given that
p_agree; we did not test the p_agree approximation of the original papers.
Against a Monte Carlo of the reference implementation at D = 256, N = 20, M = 43
(800 queries), the exact model gives 0.722 for a measured 0.714, where Law IV
gives 0.792; the test suite requires the exact model within 3 standard errors
*and* Law IV outside them.

## Appendix I. Details of the tests

**What tests 1–10 leave open.** The exact model treats candidates at equal signal, and the null distances of
different hops, as independent; test 2 puts the cost of the first below one
point, and an exploratory run at g = 5 below 0.3 points, not distinguishable
from zero (§2.3); the second is unmeasured. The +3.3-point hub residual of test 2
was the missing twin term (exploratory, below). Dense samples favour the neighbourhood of their
starting entity. Tests 1–10, 18 and 19 query stored triples or synthetic chains;
tests 11–17 use questions written by people, through a deliberately simple
front-end.
All tests are closed-world: the codebook holds only the stored items, and every
query asks for a stored (s, r) pair. The abstention threshold of Appendix B rests
on one synthetic check (3/300 false accepts); it has not been tested on absent
queries over a full-vocabulary codebook.

**Noise floor and signed biases (tests 1–10).** In a single cell the detectable bias is 1.1–8 points, and 3–5 points in the
noisy dense and composition cells. Most mean errors of the final predictor are
**at or below the noise floor** — the mean |error| a perfect model would show,
√(2/π) times each cell's standard error: at D = 16 384, 0.27 against a floor of
0.70; on dense WN18RR with twins, 0.57 and 0.60 against 1.31; for composition,
0.75–1.24 against 1.2–1.4; the same holds for both encodings in test 6. There
the model cannot be told apart from the truth with these data. The floor is not
a validation: the criteria |err| ≤ 2 and ≤ 3 sit only about 1.5–3 times above it
in the noisy cells, so passing them rules out large errors, not errors of a
point or two.

Of the 25 summaries that score the final predictor (Appendix I lists the six
left out), **four
carry a signed bias whose 95% interval excludes zero**: dense FB15k-237 at
D = 2048 (−0.81), uniform WN18RR at D = 2048 (−0.67) and 8192 (+1.11), and
wrong-relation errors (+1.34). All are below 1.4 points, and the two WN18RR
biases have opposite signs, so they do not point to one missing term; the
wrong-relation bias is unexplained. The summaries overlap (they share data and
cells), so we do not attach a chance probability to the count of four. Each bias
is near or below the smallest bias its test detects.

**Hub residual of test 2.** The residual (+3.28
points, hub minus rest, measured − predicted) was computed before twins were
counted. On the same queries it is −1.64 [−3.49, +0.26] with twins, and −1.81
[−3.69, +0.06] with twins and self-loops (`examples/cycles_residuals.py`): the
residual was the missing twin term, and what remains is compatible with zero.
Hub queries are three times as often on a 4-dependency (48% against 16%), but
the per-cell hub–rest gap in dependence does not track the gap in error
(r = +0.07 [−0.04, +0.22]).

**Test 10, post hoc (not preregistered).** The Law IV arm of test 10 has neither aliases nor
twins, and on FB15k-237 it did as well as the exact model. To see what the
WN18RR gap comes from, we repeated the choice on the same 80 subgraphs with two
stronger asymptotic laws (`examples/sizing_lawiv_posthoc.py`). With Law IV ×
g/(g+a) per query, WN18RR fell short by 0.15 to 2.79 points on average, against
2.82 to 6.21 for bare Law IV, and at most 6 of 20 subgraphs were more than 6
points short, against 12. Adding twins as a signal proportional to their weight
made things worse: 3.40 to 5.65 points short, with up to 12 of 20. So aliases
explain most of the gap, and this naive twin term does not stand in for the
exact one (§2.3). The exact model was still the only one with every group at or
above target on average.

**Test 10, per subgraph.** The criterion "at most 2 of 20 subgraphs more than 6
points short" was miscalibrated. If each measurement is binomial
around its prediction, with each subgraph's own number of queries, a perfect
model passes it in all eight groups with probability 0.39 (`sizing.md`, errata).
This makes the test harsher, not laxer; a failure would have said little. The
promise holds on average, not per subgraph: 11 to 14 of 20 met the target in
each group. The cap of 200 queries per subgraph was often not reached (median 92
on FB15k-237 at N = 150, minimum 10; at least 85 elsewhere), so the 6-point
threshold is about 1.2–1.4 standard errors where a subgraph has around 90
queries, less where it has fewer, and about 2 where it has 200.

**Test 4, uniform samples.** On uniform WN18RR both cells carry a signed bias whose
95% interval excludes zero, −0.67 at D = 2048 and +1.11 at D = 8192, of opposite
signs, too large for the rare twins to cause, and unexplained.

**Test 18, bits per fact.** Measured against the Fano minimum for answering N queries among 2N
objects at the accuracy it reached, ABM used 3.8 to 46 times the bits, rising
with load: 6.3, 4.1, 4.3, 5.3, 8.1, 11.0, 18.3 at D = 2048 and N = 50–600, and
4.9, 3.8, 4.9, 6.3, 10.8, 19.2, 46.2 at D = 8192 and N = 200–2 400.

**Summaries counted in §3.2.** Of the 31 rows of `prereg_summary.py` we count the 25 that score the final
predictor. Six are left out: test 1 (two rows, which tested Law IV, not the
exact model), test 2 on dense FB15k-237 (two rows, superseded by test 4) and
dense WN18RR without twins (two rows).

## Appendix J. Lost scripts, self-loops and implementation changes

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
supported; no cell of test 6 moves by more than 0.06 points.

**Implementation changes after the tests.** The published results are those of
the commit that recorded each test; `replicate.py` runs each harness on that
commit, and `--code current` on today's code. Two later changes alter what
today's code computes. (i) abm 1.1.0 (2026-10-09, commits `ec0c981` and
`505c2e5`) validates inputs and no longer adds an unknown symbol to the codebook
on a read. Under `--code current`, test 5 changes in 11 of 72 cells, by at most
3.3 points (a missing chain head is no longer a cleanup candidate); its
preregistered outcome is that of the recording commit and was not re-evaluated.
(ii) Commit `2dca851` made the twins harness use the fact key and the self-loop
alias, as `abm.exact.predict_queries` does; before it, the harnesses of tests 4,
6 and 10 crashed under `--code current`. They now run: test 10 is identical;
test 4 changes its twin prediction in 7 of 240 graph cells, by at most 17.4
points (the WN18RR uniform cell at D = 8192, N = 200, seed 7: 97.6% → 80.2%, measured 80.0%); test 6 moves the
symmetric prediction by at most 0.06 points; every hypothesis of tests 4 and 6
stays supported (H5 of test 4 has no self-loops). Test 11 moves by 0.13 points.
On 2026-10-09 every other test runnable under `--code current` (1–3, 7–9,
12–14, 18, 19, the Clarkson comparison, the seed-10 capacity data and
ProofWriter) was identical. The reference implementation is therefore not
frozen: the specification (FORMALISM.md v2.1) is.

## Appendix K. Further related work

*Exact and approximate membership.* What ABM was compared with at equal bits is
classical. Bloom filters [@bloom1970space] and their lower bound
[@carter1978exact] set what membership costs; Bloomier filters
[@chazelle2004bloomier] store a static function without storing the keys,
which is the "implicit keys" store of test 19; its 1.23 bits per key per value
bit is the factor of the peeling construction used by XOR filters
[@graf2020xor]. The Fano minimum used in test 18 is the standard inequality
[@cover2006elements]. Clarkson, Ubaru and Yang already connect the two lines:
their section on sparse binary bundling and Bloom filter analysis
[@clarkson2023capacity, §2.5 and §7 of the current arXiv version] treats a sparse
binary bundle as a Bloom filter. Our tests 18–19 are an empirical, dense-MAP-B
instance of that comparison, with the expected outcome.

Hopfield networks [@hopfield1982] share the interference-plus-extreme-value
mechanism, with the 0.138·N regime of Amit, Gutfreund and Sompolinsky
[@amit1985storing]. Holographic embeddings of knowledge graphs
[@nickel2016hole] store (s, r, o) triples in holographic vectors but learn them
for link prediction; ABM learns nothing and stores the facts themselves.
Retrieval-augmented systems treat memory as an index whose only operation is
similarity search, with composition delegated to a language model
[@lewis2020rag]; ABM was meant as the opposite division of labour, and test 19
measured its cost. ProofWriter [@tafjord2021proofwriter] and RuleTaker
[@clark2020ruletaker] study soft theorem proving with transformers; our use of
ProofWriter (Appendix E) keeps the chaining symbolic and makes only the truth
oracle algebraic. The rewriting results of Appendix C are standard
[@baader1998term].

## Appendix L. Reproducibility

*Reproducibility: every number is produced by a script in `examples/` with
results committed as JSON; results from 10-seed reruns sit beside the originals
with an `_s10` suffix; the twenty-two preregistrations, with their outcomes, are in
`docs/preregistration/`. The reference implementation (`reference/abm.py`, about
340 lines, numpy-only, deterministic; changed after the tests as Appendix J states) and
the exact theory (`reference/exact.py`, shipped as `abm.exact`, numpy-only) are
covered by the test suite, which runs in continuous integration on Linux x86-64
(Python 3.10–3.13, NumPy 1.24–2.5), macOS arm64 and Windows x86-64, with
codewords checked bit-identical across the three. Figures:
`examples/make_figures.py`. To replicate any preregistration from a clean clone:
`python examples/replicate.py <name>`, which downloads the data with checksums,
runs each harness on the commit that recorded its results and compares the rerun
with them (test 11 only in its deterministic part, without the language model;
tests 12–14 only their analysis, from the committed model answers). On
2026-10-01 the first eleven tests, the seed-10 capacity data, the Clarkson
comparison and ProofWriter reproduced: twelve byte-identical, two (tests 4 and
10) up to the last digit of one float; on 2026-10-05 the analyses of tests 12–14
reproduced up to float rounding. On 2026-10-09 test 18 reproduced
byte-identical, and test 19 identical in every field except its wall-clock
timings, which are not deterministic and are excluded from the comparison. Tests
15–17 query a local language model and are not wired into `replicate.py`; their
harnesses (`examples/memory_preview_prereg.py`, `shortlist_prereg.py`,
`webqsp_prereg.py`) and model answers are committed. Tests 20–22
(`examples/bit_errors_prereg.py`, `vsa_families_prereg.py`,
`generalization_prereg.py`; about 98, 14 and 81 minutes) are not yet wired into
`replicate.py` as named tests and have not been rerun; their data (FB15k-237,
WN18RR, and CoDEx-M and YAGO3-10 at pinned commits with sha256) are, and their
preregistrations, frozen predictions and harnesses carry OpenTimestamps
stamps (§1.1). Test 20 needs the optional `galois` dependency for BCH codes.*

## Appendix M. The tests of MAP-B in detail (tests 1–10)

**Test 1**, the earliest, used Law IV with k = 0.92 on uniform samples of
FB15k-237 and passed (Fig. 6); applied afterwards to the same data, the exact
model gives 0.88 points.

![Test 1, FB15k-237 uniform samples: measurement against the preregistered Law IV
prediction (dashed).](figures/fig7_fb15k237.png){width=60%}

**Test 2 — the exact model on new configurations.** At D = 16 384, a dimension
not used before the preregistration, the mean absolute error over six loads was
**0.27 points** (signed −0.09; criterion ≤ 2). With g = 1, 2 and 4 true objects
per query it was 0.58, 0.86 and 0.32 points, where Law IV, which assumes one
target, erred by 0.6, 13 and 21. On dense subgraphs of FB15k-237 (a
breadth-first sample, so that entities recur) it was 1.06 points at D = 8192,
and 2.88 at D = 2048 with a signed bias of −2.61, outside the ±2 we had set:
**supported in part**. The exact model was *pessimistic* on dense real graphs,
most at high load (−8.4 at N = 400). The cost of treating candidates at equal
signal as independent is, by this test, below one point.

**Test 3 — the hop dependence.** Per-bit correlation between two hops, predicted
−0.0645 / −0.0213 / −0.0071 at N = 10 / 30 / 90, measured −0.0634 / −0.0212 /
−0.0072, within 1.5 SE and 31–91 SE away from zero. On events, the correlation
between the successes of two chained hops was φ = −0.030 at N = 10 (predicted
−0.032) and −0.016 at N = 30 (predicted −0.011), within 3 SE; independence is
rejected at 6.0 SE (Fig. 7). Law V is falsified as an exact law, by the amount
derived. The same file preregistered the reconstruction of typed projection
(P3, Appendix D): measured gains 1.84 / 2.21 / 2.63× against the exact
1.86 / 2.26 / 2.75×.

![Per-bit correlation between the agreements of one trace with two queried facts:
exact −μ²/(1−μ²) (steps at small N are the parity of the majority vote: N and
N − 1 agree when N is even) against measurement (±3 SE), and the independence
Law V assumes.](figures/fig9_dependence.png){width=60%}

**Test 4 — symmetric twins.** Looking for the cause of test 2's dense bias,
after test 2 and on its data: in the dense samples 6–14% of the triples had
their symmetric twin in the sample — one vector of weight 2 (§2.3) — against 0%
in uniform samples. Counting twins as weight 2 reduced the bias on those cells.
Because that fit was made after seeing the data, it was tested on data we had
not seen: FB15k-237 with new seeds, and a second graph, WN18RR. WN18RR was
chosen knowing its twin share: before any measurement we had looked at its
structure, where 34.2% of the triples have a symmetric twin; in dense samples a
quarter do. On FB15k-237 the effect did not replicate. On seeds 10–19 at
D = 2048 the model without twins had a signed bias of −1.29, against −0.81 with
twins, where test 2 had −2.61 (at D = 8192, −0.65 against −0.23). Both versions
meet H1, so H1 does not separate the two models. The evidence for the twin
mechanism rests on WN18RR (H3). With twins the error was **0.57 and 0.60
points** on WN18RR (signed −0.16 and −0.36); without them, the model was
pessimistic by −2.1 and −3.8 on average and by up to −8.9 in a cell. On uniform
samples, where twins are rare (0.1–0.3%), both versions agree (0.9 and 1.1
points); this is a check that the term does no harm where it should not act,
not evidence that it helps; their unexplained signed biases are in Appendix I (Fig. 8).
The exact weighted form of Law VII, on six synthetic weight configurations, erred by at
most 1.4 points; with two facts of weight 14 the N_eff approximation erred by 2.7
and the exact form by 0.5.

![Symmetric twins on dense WN18RR subgraphs: measurement against the exact model
counting twins as one fact of weight 2 (solid) and not counting them
(dotted).](figures/fig8_twins.png){width=60%}

**Test 5 — composition without calibration.** Grounding errors of four kinds, at
rates up to 50%, on two-hop chains: the prediction — the share of intact chains
times the exact two-hop probability at the load actually stored — erred by 0.75
(missing facts), 1.89 (wrong relation), 1.01 (wrong entity) and 1.18 (spurious)
points. For missing facts the calibrated predictor of earlier versions erred by
10.2 points; a lighter trace helps more than (1 − ε)² accounts for, and the exact
model captures it by computing at the load that remains (Fig. 9). Under three
error structures with the same mean rate the error was 1.1–1.2 points, and the
predicted order held at ε = 0.4: errors concentrated on whole chains (28.3%) cost
less than independent ones (18.0%), and errors concentrated on one hop (10.0%)
cost more.

![Composition under four kinds of extraction error (preregistered): measurement
(points) against the prediction without calibration
(lines).](figures/fig6_robustness.png){width=65%}

**Encoding symmetry is a trade-off, not a flaw.** An asymmetric encoding,
s ⊕ ρ(r) ⊕ ρ²(o), has neither aliases nor twins, at the cost of the inverse
queries the symmetric one answers for free. The model predicts both without
parameters, and predicts that neither wins: at low load the aliases of the
symmetric encoding cost accuracy, at high load its twins — two facts fused into
one of weight 2 — reduce the noise and gain it. Test 6 confirmed this on dense
WN18RR, with the sign of the difference right in 6 cells of 6, from −7.0 to +7.8
points (Fig. 10). On dense WN18RR (6 cells), then, the model said which encoding
to choose at a given load before storing anything; we have not tested this
choice elsewhere.

![Symmetric minus asymmetric encoding, dense subgraphs (preregistered): measured
(points) against the prediction (lines). Circles: WN18RR; triangles:
FB15k-237.](figures/fig10_asymmetric.png){width=60%}

**Test 6 — encoding symmetry.** Described above (Fig. 10): sign of the
symmetric-minus-asymmetric difference right in 6 of 6 dense WN18RR cells.

**Tests 7–9 — deep chains.** Testing the depth prediction of §2.4 took three
preregistrations. Test 7 failed: with a codebook of about twenty entries, a
failed hop often lands on the right entity by chance and the chain recovers, a
term (of order h/M) that the prediction ignored (up to +18.9 SE); a uniform 1/M
recovery overestimates it, and tiny codebooks remain a regime where the model
does not apply. Test 8, with 1 000 distractors to make recovery negligible,
failed at a single hop: the distractors, inserted after the target, lose every
tie to it under the reference's tie rule, which the model had approximated by an
even split (a miss of 1.8 points at 13.8 SE). The model was changed to the exact
rule (§2.2). Test 9, with that rule and new data at two dimensions, held: all ten
cells within 3 standard errors of the model, and p^h rejected at 4.3–8.5 SE at
four and six hops (Fig. 11). In absolute terms the shortfall is 1.5–1.8 points at
N = 12. The per-bit correlation behind it falls as about 2/(πN); we did not
measure the shortfall at other N. Tests 7 and 8 are kept with their data in
`docs/preregistration/`.

![Deep chains (preregistered): chain accuracy minus p^h, measured (points)
against the per-bit model with the exact tie rule (lines), N = 12, 1 000
distractors.](figures/fig11_deepchain.png){width=60%}

**The alias ceiling in test 10.** The expected ceiling is computable before storing them; on dense WN18RR subgraphs it is about 0.93–0.95, on
FB15k-237 about 0.99. Test 10 checked it at D = 16 384: on WN18RR 38 of 40
subgraphs stayed at or below their ceiling + 2 points, with mean
|measured − predicted| 0.62 points; but 21 of 40 WN18RR subgraphs measured above
their ceiling, two by more than 2 points (+2.3 and +4.3), and on FB15k-237 one
subgraph exceeded it by +8.2 (N = 300; recomputed from
`sizing_prereg_results.json`). It is a mean, not a bound. A contract that
ignores it still promises, on average, accuracy no dimension delivers.

**Test 10 — the contract in use.** For 80 unseen dense subgraphs of FB15k-237 and
WN18RR (N = 150 and 300), the model chose the smallest dimension it predicted to
reach a target accuracy T ∈ {0.70, 0.80}; accuracy was then measured at that
dimension. With the exact model the promise held in every group: measured minus
target averaged +0.01 to +1.68 points, and at most 2 subgraphs of 20 fell more
than 6 points short (observed 0 to 2 per group; expected under a perfect model,
0.4 to 2.2). The criterion was miscalibrated (a perfect model passes it with
P = 0.39) and the promise holds on average, not per subgraph (Appendix I). With
the asymptotic Law IV — the contract the reference implementation had exposed — the
promise held on FB15k-237, where aliases and twins are rare (measured minus
target +0.25 to +2.43 points, against +0.01 to +1.68 for the exact model, so the
exact model did not beat it there), and failed on WN18RR: on average 2.8 to 6.2
points below target, and in up to 12 of 20 subgraphs more than 6 points below.
The alias-ceiling results of this test are given above. The exact model is now
shipped as `abm.exact`, with `contract_for` and `min_dimension`; the latter
refuses targets above the expected alias ceiling, which is conservative for a
given memory, since a memory can exceed its expected ceiling by a few points
(§2.3). A post hoc comparison with stronger asymptotic laws (Appendix I) finds
that aliases explain most of Law IV's gap on WN18RR.


## Appendix N. Details of tests 20–22

**Test 20, per group.** ABM's model error (mean absolute, points) per (graph, D):
FB15k-237 0.38 (D = 2048) and 0.41 (8192), WN18RR 0.73 and 0.55; worst cell
1.48. The store's model against its independent re-estimate: 0.39, 0.14, 0.29,
0.13. ε\* (the noise level at which ABM would overtake the store) was predicted
"never" in every series and observed "never" in all 13 series where it could be
assessed. At ε = 0, ABM was below the store in 15 of 16 series, the sixteenth
being the tie at 1.000. The rule chose an XOR-filter layout almost everywhere,
with no code at ε = 0, BCH at small ε and polar codes at ε ≥ 0.1. The
preregistered minimum detectable difference per cell ran from 1.6 points
(D = 8192, N = 200, small ε) to 13 (D = 2048, N = 50, ε = 0.2).

**Test 21, structure.** Dense FB15k-237: aliases 3.5% (N = 100) and 0.7% (400),
twins 7.9% and 0.9%, expected ceiling 0.985 and 0.996; dense WN18RR: aliases 8.5%
and 9.4%, twins 12.1% and 14.4%, ceiling 0.927 and 0.957; uniform samples:
aliases ≤ 0.4%, twins ≤ 0.1%. Predicted gaps between the estimate and the naive
comparator: uniform 0.1–0.2 points, dense FB15k-237 1.7–4.2, dense WN18RR
4.4–8.3. BSDC-S's model was changed once after a synthetic smoke test (a
Gaussian version was 10–13 points optimistic there); this was declared before
the run.

**Test 22, structure and out-of-sample counts.** Aliases ≤ 1.0% and twins
≤ 3.4% in every (graph, sample, N); ceiling ≥ 0.992. With D chosen on seeds 0–9,
seeds 10–19 reached the target within 3 points in 12 of 12 cells for every
family and graph except HRR (11 of 12 on both graphs) and MAP-B on YAGO3-10
(8 of 12). In MAP-B's YAGO3-10 dense cells the error grows with load: +1.6 and
−0.5 points at N = 100, −3.8 and −5.3 at N = 400, −5.8 and −7.2 at N = 1 600
(measured minus predicted). Chain cells existed only at two hops, in CoDEx-M
uniform N = 1 600 (20 seeds), CoDEx-M dense N = 400 (10) and 1 600 (11) and
YAGO3-10 dense N = 1 600 (14); cells whose per-hop detectable error exceeded 5
points were excluded, as preregistered. Synthetic checks before the run: HRR
at one hop within 0.1–0.9 points of a Monte Carlo on 16 subgraphs, and 0.820
predicted against 0.807 ± 0.016 at N = 3 200; chains within about 2 SE with
≥ 1 600 chains.

## Appendix O. Tests 18–19 in detail

**Test 18: recall and membership.** Facts (s_i, r_{i mod 13}, o_i) over 2N
entities, three seeds per cell, D = 2048 with N = 50–600 and D = 8192 with
N = 200–2 400. An *idealised* exact store spends 2·⌈log₂ 2N⌉ + ⌈log₂ 13⌉ bits per
fact with no overhead and no stored keys, which no real table achieves, so the
comparison favours the store; with D bits it keeps ⌊D/b⌋ facts and loses the
rest. The store was not run: its accuracy is computed, min(1, ⌊D/b⌋/N), so only
ABM's side is a measurement. ABM's accuracy was below that figure in **14 of 14
cells**, including every overloaded one (for example 0.660 against 0.853 at
D = 8192, N = 400; 0.015 against 0.114 at N = 2 400); in the lightest cell
(D = 2048, N = 50, 0.993 against 1.000) the gap is under one standard error and
is not a meaningful loss. There is no crossover, as the exact model had
predicted, and it predicted ABM's accuracy with a mean error of 0.99 points at
D = 2048 and 0.54 at D = 8192 (worst cell 1.97, at D = 2048, N = 100). For
membership, a Bloom filter of D bits made fewer errors (false negatives plus
false positives on 1 000 unstored facts) than `Memory.member` (z ≥ 3) in every
cell: the Bloom filter had no false negatives, while `Memory.member`, with
almost no false positives, missed 26–93% of the stored facts above the lightest
loads. Against the Fano minimum ABM used 3.8 to 46 times the bits, rising with
load (Appendix I). All three hypotheses were supported, which here means ABM lost. The outcome was expected
before the run; the test is a confirmation of the obvious with a number
attached, not a strong test of the theory. One synthetic protocol, no aliases or
twins, and only single-hop recall and membership were compared. **ABM is not
dense**, and no claim of this paper rests on density or compactness. An earlier
version argued that ABM is not a compressor from the asymptotic Law IV, which is
retired; the argument now rests on this test and on ProofWriter (Appendix E),
where the trace takes 512 bytes where a minimal exact encoding takes 8–23.

**Test 19: chains and compiled compositions.** On FB15k-237 and WN18RR we sampled
N facts as two-hop paths (x, r₁, y), (y, r₂, z) and asked the paths that are
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
0.32–2.01 points per (graph, D, task); per cell the error reached 7.4 points
(WN18RR, D = 2048, N = 100, chain), where "within 2 points" would mislead. For
one two-hop question, ABM with a vectorised cleanup took 2 600 to 229 725 times
as long as a Python `dict`; the store's time is near the timer's resolution, so
the ratio gives only the order of magnitude. All four hypotheses were supported,
which again means ABM lost, as we expected before the run. This also supersedes
the compiler result of earlier versions (99% against 82%, Appendix D), which
compared ABM only with itself. What remains unmeasured is richer queries
(conjunctions, analogies, role filling) and a bitpacked runtime; on what was
measured, the algebra is **not an advantage**.

## References
