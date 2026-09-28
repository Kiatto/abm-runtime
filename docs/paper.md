# A Quantitative Theory of Binary Holographic Memory, with a Reference Implementation

*(Algebraic Binary Memory — ABM)*

**Preprint v1.4 — September 2026**
*Normative specification: [FORMALISM.md](FORMALISM.md) (frozen, v2.1).
Reference implementation: [`reference/abm.py`](../reference/abm.py).
All experiments reproducible from `examples/`; raw results in the
repository as JSON.*

## Abstract

We study Algebraic Binary Memory (ABM), a binary vector-symbolic model of the MAP-B family: facts are XOR-bound triples stored in one majority-vote trace in {−1,+1}^D, and reasoning alternates unbinding with cleanup onto a codebook. That the dimension needed for membership in such a trace scales as O(N log M) is known; we make it predictive. The capacity law N\* = k·2D/(π·z_G(M)²), with a second-order Gumbel threshold, is derived with k = 1 and measured at k = 0.92 ± 0.03 (range). On the membership task of the known bound, its explicit constant over-provisions dimension about sixfold, while exact constants match simulation within grid resolution. With k fixed on synthetic data, the accuracy contract predicts a fresh ten-seed rerun within 0.3 points, and, in a preregistered test on the real knowledge graph FB15k-237, within 1.05 points (signed −0.5). Assuming independence across hops, which we measure but do not prove, multi-hop accuracy composes as p^h, and three predictions derived before measurement held. Two of our own models were falsified and are kept with their data. The results also bound the model: on ProofWriter an exact store scores 100% where the trace scores 92% at depth 5, and by Law IV the trace needs fewer bits than a minimal exact encoding only below about 75–80% per-query accuracy. The contribution is not compression but a memory whose capacity, depth and reliability can be stated before deployment, with the record of where it failed.

## 1. Introduction

Modern retrieval-augmented systems treat memory as an index: a store
whose only operation is similarity search, with all composition
delegated to a language model. We investigate the opposite division of
labour: a memory whose *operations themselves* compose knowledge, and
ask a model-theoretic question:

> **Which classes of inference are closed under binding and unbinding
> over discrete states?**

This question is independent of any language model, benchmark, or
implementation. Our contribution is a quantitative treatment: a minimal
axiom set, laws with confidence intervals, derivations with explicit
constants where the literature gives asymptotic bounds, predictions made before measurement — and
falsifications retained on record. The model is deterministic, requires
no training, uses no floating-point tensors in the reasoning path, and
its null distribution is known a priori — which is what makes a
*resource theory* (space D, codebook M, depth h, reliability p)
possible at all. In the vector-symbolic taxonomy ABM is a MAP-B
architecture [@clarkson2023capacity]; §8 separates what is already known
about it from what is new here.

**Methodological rule.** No law enters the formalism without a new
quantitative prediction and a designed falsification experiment.
Retired laws stay in the record with the data that killed them. This
rule is load-bearing: it corrected our capacity law (linear → Gumbel
form), retired one law entirely (§6), and produced the three verified
predictions of §5.

## 2. The model

**States.** x ∈ 𝔹^D with 𝔹 = {−1,+1}; d_H the Hamming distance;
the null distribution between independent states is Binomial(D, ½).

**Operators** (with their axioms):

- **Binding** x ⊕ y: elementwise product (≡ XOR).
  *A1: an isometric involution* — (x⊕y)⊕y = x, distances preserved.
- **Bundling** ⊞(x₁…x_n): bitwise majority vote.
  *A2: Gaussian superposition* — member correlation √(2/(πn)),
  fluctuations 𝒩(0, √D/2) against outsiders.
- **Cleanup** over codebook C: argmin_{c∈C} d_H(x, c).
  *A3: idempotent projection* onto C.

**Facts and traces.** A fact (s, r, o) is encoded as
f = (c_s ⊕ ρ(c_r)) ⊕ c_o with ρ a cyclic shift; a memory is the single
trace T = maj(f₁…f_N). Query(s,r) = cleanup(T ⊕ c_s ⊕ ρ(c_r)).

**The ABM machine** is (𝓜, 𝓐, C, S, I, O): memory configuration,
operators, codebook, an O(1) symbolic controller, and name↔codeword
interfaces. Controllers (Horn chainer, planner, LLM) are programs over
the same machine; the model is separated from the pipeline.

**The potential Φ.** All quantities in the model are expressions of one
scalar field Φ_y(x) = (D/2 − d_H(x,y))/√D: distance, z-scores,
calibrated confidence σ(2Φ/τ), cleanup (argmax Φ), membership tests,
and — non-trivially — bundling itself, which is the variational
maximizer of Σwᵢ·Φ_{xᵢ}(x) (provable bitwise). Binding is the group
action that preserves Φ. Reasoning is a trajectory of Φ-descents with
resets.

**Conventions.** ± is a 95% confidence half-width across seeds unless
the text says SD or range. Differences between accuracies are in
percentage points.

### 2.1 The operators are necessary and independent

Ablating each operator kills a disjoint capability set (measured,
D = 1024, N = 50, 10 seeds; an earlier 3-seed run gave 85% and 67% for the
first two entries of the full model, with the same zeros):

| variant | relational recall | holographic O(D) | 3-hop | compose | symbols |
|---|---|---|---|---|---|
| full | 82% | ✓ | 80% | ✓ | ✓ |
| −binding | **0%** (content membership survives at 73%) | ✓ | 0% | 0% | ✓ |
| −cleanup | 0% | ✓ | 0% | ✓ (signal 3.6σ, never a symbol) | **0%** |
| −bundling | **100%** | **✗ (50× space)** | 100% | ✓ | ✓ |

Binding carries relational structure; cleanup carries symbolization and
depth; bundling carries compression — and is the only operator that
*costs* accuracy: it is the memory operator, the other two are the
computation operators.

## 3. Capacity laws

The laws referred to by number throughout:

| law | statement | status |
|---|---|---|
| I | null distance Binomial(D, ½) ≈ 𝒩(D/2, √D/2); the basis of every threshold | exact |
| IV | capacity: N\* = k·2D/(π·z_G(M)²) | derived k = 1; measured 0.92 ± 0.03; preregistered test on FB15k-237 passed (§7) |
| V | hop composition: Acc(h) = p^h with cleanup between hops | corroborated; the argument assumes independence (§4) |
| VI | failure grows with out-degree | **retired**: a load artifact (§6) |
| VI′ | topological neutrality: only N_eff/D matters | corroborated on synthetic data; on real hubs neither supported nor falsified (§7) |
| VII | redundancy: N_eff = Σw² | corroborated; saturates at w ≳ 10 |
| VIII | every end-to-end failure is attributable to one level | principle; supported by pilots only (§7) |

**Law IV (capacity).** The load at which query accuracy crosses 50% is

  N\* = k · 2D / (π · z_G(M)²),  derived k = 1, measured k = 0.92 ± 0.03 (range)

where z_G(M) = √(2 ln M) − (ln ln M + ln 4π)/(2√(2 ln M)) is the
second-order Gumbel threshold for the minimum of M null distances.
Derivation: member-trace correlation (A2) gives a signal z-score
√(2D/(πN)); retrieval fails when the extreme of M−1 null distances
crosses it. The naive linear model N\* = cD is *rejected by our own
data*: c drifts systematically with D (0.098→0.068, disjoint CIs over
D ∈ [512, 4096]); the Gumbel form is stable within 1.1% over
M ∈ [447, 16 242] (R² = 0.9988 vs 0.979 linear). The ln M dependence
was posed as a prediction and confirmed before the constant was
refined.

The constant is recomputed from the saved results by
`examples/k_from_results.py`. Across the codebook sweep (D = 2048, M from
447 to 16 242) k = 0.943 ± 0.005 (SD, 10 seeds), a 1.1% spread (3 seeds had
given 0.936 ± 0.009, 2.2%); across the dimension
sweep (D from 512 to 4096) k = 0.914 ± 0.025 (SD), lower at D ≤ 1024
(0.89) than above it (0.93–0.94). We quote 0.92 ± 0.03 as the range over
both sweeps; the drift at small D is a finite-size effect we have not
modelled. The O(N log M) form itself is not new (§8); what the law adds is
the constant and the accuracy curve (Fig. 1).

![Law IV. Measured 50%-accuracy load N\* (10 seeds, 95% CI) against the
prediction, solved as a fixed point because the codebook grows with the
load (M = 2N + 11). Dotted: pure theory, k = 1.](figures/fig1_capacity.png){width=60%}

**Law VII (redundancy).** A fact written with multiplicity wᵢ weights
the majority vote; the effective load felt by singletons is
N_eff = Σwⱼ² (participation ratio). Model comparison over six weighted
configurations at constant unique N: mean |error| 6.3% for the Σw²
model versus 28.5% for total-count; the residual at extreme weights
(w ≳ 10) is sign saturation, identified and unmodelled. Frequency is
salience, at quadratic cost to the rest of memory.

**Law VI′ (topological neutrality) — via internal falsification.** An
earlier law ("failure scales with node out-degree", 100%→14% for
B=1→16) was **falsified at constant load**: accuracy is flat (64–73%)
for B ∈ [1, 24] when total facts are fixed. The original effect was
entirely a load artifact. Only N_eff/D matters; graph topology does
not.

**Robustness asymmetry.** Flipping a fraction ε of trace bits degrades
gracefully (signal ∝ 1−2ε; 86%→42% at ε=20%). Corrupting the codebook
is catastrophic (ε=5% halves accuracy): key construction, cleanup
targets and stored content corrupt jointly. The item memory is the
trusted computing base of the paradigm.

### 3.x Two families of results

The theory splits into two families. **Resource laws** answer *how
much it costs*: they depend on (D, N, M, ε, h) — Laws I, IV, V, VII
and the Resource Composition Law. **Structural laws** answer *what is
possible*: they depend on the symmetries and invariances of the
algebra, not on resources — Bridge Elimination, Compose, trace
self-cancellation (the no-go theorem), and edge-symmetry aliasing
(§7). The distinction assigns each operator a conceptual role: cleanup
removes *noise* (a resource phenomenon); typed projection removes
*unwanted symmetries* (a structural phenomenon) — it is a
disambiguation operator, not an optimization. Aliasing candidates sit
at equal signal, so no amount of cleanup can separate them; projection
eliminates them exactly.

## 4. Error composition and the Memory Calculus

**Proposition (hop composition).** With cleanup between hops, and
assuming the hop outcomes are independent,

  P(h-hop chain correct) = p^h + ε, |ε| ≤ h/M·(1+o(1)).

*Argument.* (i) *Noise reset*: cleanup returns an exact codebook
element (A3), so each hop's key is built from noise-free vectors; (ii)
*decorrelation*: noise components of distinct keys are uncorrelated
bitwise (E[k_i k_j] = 0; proven to first order, and measured:
success-event correlation φ = 0.014 ± 0.024, bit-level noise
correlation −0.005 across 1 800 query pairs); (iii) *absorption*:
off-path decodes are near-uniform over C, so return probability is
O(1/M). Step (ii) is proved only to first order: joint independence is
measured, not proved (§9, problem 3), which is why this is a proposition
and not a theorem. Empirically |Acc − p^h| = 0.020 ± 0.008 over 10 seeds. The law
is a structural property of the composition cleanup→bind→cleanup, not
of any benchmark.

**Memory Calculus.** Terms e ::= a | 1 | e⊕e | ρ(e) | ⊞(e…) |
cleanup(e), in two sorts: an *exact fragment* (⊕, ρ) and a
*probabilistic fragment* (⊞, cleanup).

- **Normal form (exact fragment).** (E, ⊕, 1) is the free Boolean group
  — a vector space over GF(2) — on ρ-stratified atoms, so every exact term
  has the unique normal form "atoms of odd multiplicity". The fact is
  elementary; it is stated because the calculus is built on it. Checked on
  200 random reduction orders.
- **Compose is not a rule.** For facts sharing a bridge,
  nf(f₁ ⊕ f₂) = c_A ⊕ ρr₁ ⊕ ρr₂ ⊕ c_C: the two-hop fact is *generated*
  by normalization, and the bridge is *exactly eliminated*
  (**Bridge Elimination**: since c_B ⊕ c_B = 1, the composed fact contains
  no information about c_B).
- **No-go theorem.** T ⊕ T = 1: any algorithm that reuses a raw decode
  (containing T) inside a subsequent key on the same trace collapses
  deterministically. Depth *requires* symbolization (A3). We also
  falsified our own softer model here: geometric signal decay
  z_eff = √D·ρ^h predicts measurable accuracy for two-trace raw
  chaining at h=2; measurement gives 0%. Raw composition is confined
  to h=1; the prediction is retained as falsified.
- **Confluence.** The exact fragment is terminating and confluent
  (Newman); distributional confluence for disjoint probabilistic
  redexes follows from the measured independence; nested redexes are
  open.
- **Cost semantics.** Exact steps are free and certain; ⊞ spends
  capacity (Law IV/VII); cleanup spends reliability (Law V). The hop
  composition proposition is precisely the soundness of this semantics,
  under the same independence assumption.

## 5. Predictions verified in advance

**P1 — Depth is exponentially cheap.** From Laws IV+V:
D_min(h) = Θ(N ln M) + Θ(N ln h). Measured at constant load (N=120),
target 95% chain accuracy:

| h | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| D_min | 3846 | 4402 | 5277 | 5528 | 6248 | 7017 | 7697 |

Growth 1→64 hops: **2.0×**, with a logarithmic fit R² = 0.992 against 0.798
for a linear one (10 seeds, D searched on a geometric grid with 10% steps; SD
300–600). An earlier 3-seed run on a 25% grid gave 2.1× but could not
separate h = 4 from h = 8; the finer run is monotone. The prediction is
modest: logarithmic growth is what Law V implies together with a Gaussian
tail, since keeping p^h ≥ 0.95 needs a per-hop error of order 1/h and hence a
margin that grows only as √(ln h). What the measurement adds is that the
constant is small. In the same run at fixed D = 1024, Law V itself holds to
|Acc − p̂^h| = 0.007 over h ≤ 24. Load, not depth, is the limiting resource
(Fig. 2).

![P1. Minimum D for chain accuracy ≥ 95% at constant load N = 120
(mean ± SD over 10 seeds, geometric grid with 10% steps).](figures/fig2_depth.png){width=60%}

**P2 — The compiler.** The calculus predicts that normalizing before
decoding (composing facts exactly at "sleep time" into a second trace)
converts p² two-hop queries into single-cleanup queries at the
compiled trace's load. Measured: 99% vs 82% (40 chains), **89% vs 25%**
(80), **44% vs 3%** (160); the naive path tracks p² throughout.
Offline consolidation is the strategy the cost semantics prescribes,
not a heuristic borrowed from biology (Fig. 3).

![P2. Accuracy of two-hop queries answered by two cleanups
(interpreted) or by one cleanup on a compiled trace T₂; dotted: p²
from Law V.](figures/fig4_compiler.png){width=60%}

**P3 — Typed projection.** Restricting cleanup to a typed sub-codebook
S buys capacity by exactly z_G(M)²/z_G(|S|)² (no free parameters).
Measured gains 1.84/2.25/2.68× vs predicted 1.94/2.37/2.82× over a 16×
distractor range: all three are about 5% below the prediction, a
consistent bias we have not explained. Typed cleanup is also immune to
codebook inflation.

**Compose cost.** P_compose = p² (two independent cleanups): measured
0.77/0.23/0.01 vs predicted 0.77/0.21/0.02 across three loads (10 seeds).

## 6. Falsifications retained

| retired claim | killed by |
|---|---|
| Law VI: failure ∝ out-degree | constant-load ablation: flat 64–73% for B ∈ [1,24]; original effect was load |
| geometric decay of raw chaining (z_eff = √D·ρ^h for h≥2) | two-trace raw chaining: 0% measured where the model predicts ~40%; T⊕T no-go is the sharp form |

Both remain in the formalism, marked RETIRED, with the data.

### 6.1 Two limits derived at the implementation layer

The bitpacked runtime made two claims that this paper never made, but that
the theory can adjudicate. Both were tested and both fell; the model
explains why, after the fact.

**Cleanup cannot be sublinear in this regime.** A metric-tree index
(VP-tree) was expected to give O(log M) cleanup. Measured, it visits 4 487
of 5 000 nodes (D = 2048) and gains nothing. The cause is structural and
follows from A2: a query sits at 0.446·D–0.483·D from its *own* codeword
(50–500 stored facts), against 0.500·D for every other codeword — a gap of
only 34–111 bits. Triangle-inequality bounds cannot prune when the target is
almost as far as the distractors, and banded LSH fails for the same reason
(per-band collision probability 0.53^r for r bits). The correct claim is not
"sublinear" but "linear with a low constant": 6.1 ms at M = 10⁵,
memory-bandwidth-bound.

**Acceptance must be an extreme-value test in M.** A fixed confidence
threshold (0.75) accepted 0/30 correct answers on a memory whose predicted
accuracy was 1.00. The failure was the opposite of the one first
hypothesized (an excess of false accepts), but the defect was in the form,
not the value: cleanup reports a *maximum* of M z-scores, so its null
distribution depends on M, while a fixed threshold does not. The replacement
follows from the same extreme-value reasoning as Law IV — accept iff
z ≥ Φ⁻¹((1−α)^{1/M}) — which caps false accepts at α for any codebook size
(measured: 30/30 true accepts, 3/300 false accepts at α = 0.01; the
threshold is 3.09 at M = 10 and 6.00 at M = 10⁷). The ProofWriter oracle of
§7 is the M = 1 case: each membership test is a single comparison, and
z ≥ 3 corresponds to α ≈ 0.0014 per test. A proof runs many such tests,
however, and the oracle does not correct for their number, so its
false-accept rate per proof grows with it. That is a limitation of the
present oracle, not a confirmation of it.

## 7. External validation and error attribution

**ProofWriter** (OWA, attribute fragment, 100 questions/depth, 10
seeds). Facts live only in the holographic trace; forward chaining uses
a single truth oracle: d_H(fact-hv, T) under a z ≥ 3 threshold. Derived
facts are written back (noise grows with depth — part of the test).

| depth | 0 | 2 | 5 | majority baseline |
|---|---|---|---|---|
| accuracy | 99.8% ± 0.3 | 99.1% ± 0.3 | 92.4% ± 1.4 | 42% |

The depth-5 degradation is consistent with the load increase that Law IV
implies; we do not report the predicted value. Declared limits: ~35–55%
grammatical coverage (4 patterns), and accuracy is measured only on the
problems those patterns parse, so the subset is selected; negation is
untested; chaining control is symbolic, the truth oracle is purely
algebraic (Fig. 4).

**Against an exact store.** The same forward chainer on the same questions,
with a Python set in place of the trace, scores **100% at every depth**, so
the parser and the chaining are sound and the whole 8-point loss at depth 5
belongs to the algebraic oracle. Its footprint is also smaller: a 512-byte
trace against 8–23 bytes for a minimal exact encoding of the same facts. At
this scale the trace loses on both axes.

The comparison generalises. By Law IV, holding N facts at single-query
accuracy a needs D/N = π·(z_G(M) + Φ⁻¹(a))²/(2k) bits per fact, while a
minimal exact encoding of (s, r, o) needs 2·log₂V + log₂R. The trace is the
smaller of the two only below a = 74% (M = 100) to 81% (M = 10⁵); at 95% it
is 1.5–1.8 times larger. A holographic trace is therefore not a compressor.
What it offers instead is a fixed size, membership by one Hamming distance,
exact algebraic composition (§4), and a degradation that can be stated in
advance (§7, capacity contract). Whether those are worth the bits is a
question of use, not of this paper.

![ProofWriter, parsable subset: accuracy by inference depth (10 seeds,
95% CI) against the majority baseline.](figures/fig3_proofwriter.png){width=60%}

**HotpotQA [@yang2018hotpotqa] (negative result).** With a regex grounding layer, the
algebra is never engaged (1.9 triples per 42 sentences; 0.5% of
queries planned) and demo-tuned multi-hop heuristics *subtract* value
(7% vs a 13% single-hop baseline; oracle 100%, chance 4.9%). This
falsifies the grounding layer, not the algebra — and motivates the
error-conservation principle (Law VIII): every failure is attributable
to exactly one level (grounding / capacity / cleanup / controller) by
level ablation.

**Attribution pilot.** Replacing the regex layer with an
LLM extractor (all 10 contexts read, gold + distractors; generic
35-relation schema; question-blind protocol with auto-inverses;
chains up to 3 hops) restores **10/10** on the same task with
calibrated confidences and per-answer provenance — and the lower
confidences of the blind protocol (0.28–0.64 vs 0.64–0.85) are the
Law IV effect of the doubled load. Declared caveats: n=10 feasibility
sample; extractor previously exposed to the questions; production
numbers require a blind extractor at n ≥ 100.

**Capacity contract.** At a fixed 1 KB budget (D = 8192), the accuracy
predicted by Law IV from pure theory (k = 1, no fitted parameter) is
within 4.2 points of measurement on average over N ∈ [100, 600]. The error
has one sign, however: the contract is never pessimistic, and for N ≥ 300
it is optimistic by 3.9–8.5 points. That is the same ≈ 8% that k = 0.92
measures. The spec-sheet line an earlier version printed, *"up to 300 facts
at ≥ 85% accuracy in 1 KB"*, fails its own measurement: at N = 300 the
measured accuracy is 84.6%.

With k = 0.92 in the forward formula, the bias disappears. The constant
comes from other experiments (§3) and was fixed before a fresh rerun with
10 seeds and up to 200 queries per seed: the calibrated contract is within
**0.3 points** of measurement on average (signed −0.1), inside the sampling
error (≈ 1 point) at every load, where pure theory is off by 3.2 (signed
+3.2). This is an out-of-sample prediction, and an extrapolation: k was
measured at D ≤ 4096, the contract runs at D = 8192 (Fig. 5).

![Capacity contract at 1 KB (D = 8192), fresh 10-seed rerun: pure theory
(k = 1) and the calibrated contract (k = 0.92, measured on other
experiments) against measurement.](figures/fig5_contract.png){width=60%}

**Synthetic end-to-end pilot.** 500 sentences written by us, a template
extractor with deliberate gaps (one phrasing in four uncovered, one trap
that inverts the arguments), 379 triples, D = 16 384 chosen by Law IV, and
1 000 queries. The first contract, built on a string-match audit of the
triples, predicted 10% and was violated by 47 points (measured 56.8%):
inverted triples still answer correctly, because the encoding is symmetric
in s and o (see *Cross-domain invariance* below), and the audit did not
account for it. We then changed the audit to test membership up to that
symmetry — a change made *after* seeing the violation. The second contract,
issued before any query from an audit of 40 chains, predicted 47% ± 15%;
measured accuracy was 56.8%, 9.8 points off and inside the interval.
Recomputing the audit term on the full gold set, possible only in a dry
run, brings the prediction to 58.5%, a 1.7-point residual; the rest is the
audit's sampling error. The pilot is synthetic, the protocol change was
post hoc, and the audit has n = 40.

**Resource Composition Law (empirically corroborated, per-query form).**
Injecting four extraction-error types at rates ε ∈ [0, 0.5] (2-hop
chains, D=2048), end-to-end accuracy follows Acc = E_q[Pg(q)] ·
Pr(N_eff(ε)): grounding and reasoning compose multiplicatively, with
the reasoning factor evaluated at the load the grounding actually
leaves (missing facts also *lighten* the trace). Over 40 cells and 10
seeds the mean absolute deviation is 3.6 points (an earlier version printed
"~3%"; its own 3-seed data gave 4.3). The average hides where the error is:
for wrong relations, wrong entities and spurious facts the prediction is
within 1 point on average, but for **missing** facts it is pessimistic by
10.2 points. A lighter trace helps more than N_eff(ε) accounts for. Stress-tested against non-i.i.d. structures: the
per-query form survives (|dev| 2.3–4.8%) while the mean-precision form
Acc = p̄^k · Pr breaks exactly where predicted — cluster-correlated
errors outperform i.i.d. errors of equal mean rate (52% vs 30% at
ε = 0.4) because the damage concentrates on fewer queries. One
prediction was falsified and is retained: chain confidence does *not*
detect grounding errors (the confident-wrong signal exists per hop but
dilutes in the product), so grounding must be audited at its own level (Fig. 6).

![Resource composition: end-to-end accuracy under four kinds of injected
extraction error at rate ε (2-hop chains, D = 2048); dotted: prediction.](figures/fig6_robustness.png){width=65%}

**Compiler ranking (dry run).** Three simulated extractors with equal
apparent quality but different error structure (uniform 8%, whole-chain
clusters at 20%, recall-tuned with 30% spurious facts) were compiled
into memories and ranked by the per-query contract *before any query*.
On the statistically resolvable pairs the predicted ranking matches the
observed one (2/2 at 95% CI, 10 seeds); on the cluster extractor the
mean-precision form errs by 11 points where the per-query form errs
by 4. Corollary: at equal mean precision, an extractor that fails in
clusters is preferable to one that fails uniformly — a selection
criterion no precision/recall metric expresses. Real-LLM replication
is the natural next step.

**Cross-domain invariance (with one exact exception).** Four
deliberately different graph topologies (dense relation reuse,
single-relation sequences, subject hubs, object hubs) under one
contract formula with no fitted per-domain parameters: mean deviation
2.4%. The single-relation domain initially *falsified* the naive
domain-independence claim (37% vs 79%) through an exact algebraic
mechanism, not noise: the encoding s⊕ρ(r)⊕o is symmetric in s and o —
every fact is an undirected edge — so with equal relations on
consecutive hops the predecessor aliases the successor at equal signal
(verified: 21/19/0 on clean hops). The derived 1/g aliasing factor
restores the contract (|dev| 2.6%), and the typed projection already
in the model (cleanup over the unvisited subset) eliminates the alias
entirely (80% measured vs 79% full contract). Structure enters the law
only through an algebraic term computable from the query plan before
any query — never through content. The symmetry is also a feature:
inverse queries come for free.

**Aliasing Factor Hypothesis.** The derived correction Acc = p · Π 1/gᵢ
(gᵢ = equal-signal candidates at hop i, computable from the query plan)
is corroborated at g ∈ {2, 3, 4, 8} (mean |dev| 4.2%, max 7.5%), and
the counter-check separates the two operator roles: guided projection
restores accuracy to p at *every* g, confirming that aliasing is
symmetry, not noise. It remains a hypothesis pending mixed multi-hop
plans and real corpora.

**Preregistered test on a real knowledge graph.** Every law above was
measured on synthetic facts: random entities and relations, used
uniformly. To test Law IV on real structure — hubs, long-tailed degrees,
one-to-many relations — we preregistered predictions and success criteria
on FB15k-237 [@toutanova2015observed] and committed them, with the harness,
before any run (`docs/preregistration/fb15k237.md`). Ten seeds per cell
sample N triples uniformly from the training split; each stored (s, r) is
queried, and an answer counts as correct if it is any true object (35% of
pairs have several). The prediction is Law IV with k = 0.92, never measured
on this dataset, times the aliasing factor of §7 computed from the stored
facts before querying.

The primary hypothesis passed its criteria (mean absolute error ≤ 5, signed
within ±3): over 12 cells at D = 2048 and D = 8192 the mean absolute error is
**1.05 points**, signed −0.47, and the same in each D separately (Fig. 7).
With k = 1 the error is 2.5 points and optimistic again. Two limits: the
aliasing factor is 0.994–1.000 in every cell, because FB15k-237 was built
without inverse relations, so that term was not really tested; and
uniform sampling makes the stored graph sparse, since few of the 14 505
entities recur in a sample of at most 1 600 triples. The secondary
hypothesis, that hub subjects are predicted as well as the rest (Law VI′),
was neither supported nor falsified: hubs were answered *better* than
predicted by 5.5 points, between the preregistered thresholds of 5 and 10.
A possible cause, identified only after the run and untested, is that hub
queries have more true objects and the prediction assumes one.

![Preregistered test on FB15k-237: measured accuracy (points, 10 seeds)
against the prediction fixed before the run (dashed), at two
dimensions.](figures/fig7_fb15k237.png){width=60%}

**Real-document pilot (error attribution, not capacity).** Two sessions on
real documents (a product catalogue and a six-page Italian employment
contract) first produced 0 useful answers out of 4. The contract was then
re-run through a fully local pipeline (a 4B-parameter instruction model on
CPU, temperature 0), and level ablation located every failure outside the
algebra: in extraction, 11 of 14 triples were bound to the employer instead
of the employee, because formal Italian leaves the subject implicit; in the
planner, 0 of 3 generated retrieval plans were correct. With both corrected,
the pipeline answered 8 of 10 questions, and the two remaining errors are
relation-classification confusions between near-synonyms. The ABM level was
never the cause: its contract predicted 100% and measured 100%.

That last figure is *not* evidence for Law IV. The memory held about 14
facts at D = 2048, under 5% of the predicted N\* (≈ 270–410 for a codebook
of 20–50 entries), a load at which any model predicts near-perfect recall.
What the pilot supports is Law VIII — each failure was attributable to
exactly one level — and only on one document, with questions written by
the author, who knew the extracted vocabulary, and with single-hop queries.
It is a case, not a sample.

## 8. Related work

Vector Symbolic Architectures and hyperdimensional computing
[@kanerva1988sdm; @kanerva2009hd; @gayler2003jackendoff; @kleyko2022survey]
supply the operator vocabulary: Plate's holographic reduced representations
[@plate1995hrr], Gallant and Okaywe's matrix binding [@gallant2013objects],
and Rachkovskij and Kussul's context-dependent thinning
[@rachkovskij2001thinning]. Superposition capacity of the D/ln M form is
classical. For the architecture ABM belongs to, MAP-B, Clarkson, Ubaru and
Yang [@clarkson2023capacity] prove that membership in a majority bundle of
n items — including bundles of bindings, which is what an ABM trace is —
needs dimension O(n log(d/δ)) for failure probability δ over d items.
Thomas, Dasgupta and Rosing [@thomas2021theoretical] give a broader
theoretical treatment of hyperdimensional computing, and Frady, Kleyko and
Sommer [@frady2018sequence] derive retrieval accuracy from crosstalk noise
in VSA-coded recurrent networks.

Law IV therefore does not establish the scaling. What it adds is narrower:
explicit constants — the 2/π of majority correlation and a second-order
Gumbel threshold on the codebook — that predict the *accuracy curve* and the
50% collapse point instead of bounding a failure probability, with one
measured constant (0.92 ± 0.03 against a derived 1).

The difference is quantitative, not only of kind. From the proof of their
Theorem 16 the sufficient dimension is m_C = 56·n·ln(2d/δ): the proof uses a
lower bound of 1/√(7n) on the per-coordinate signal, where the true value is
√(2/(πn)), and Hoeffding's inequality in place of a Gaussian tail. On their
own task — a threshold membership test over d = 500 items, all correct with
probability 1 − δ = 0.9 — we measured the minimum dimension by simulation
(`examples/clarkson_comparison.py`, 200 trials per point, 10% grid). For
n = 10, 25 and 50, m_C exceeds the measured minimum 5.7–6.3 times, with the
threshold at half the expected signal, and 5.7–6.9 times with exactly their
threshold; the same rule with exact constants, m = 2πn·z² with
z = Φ⁻¹(1 − δ/2d), matches the measurement within the grid. This is not a
flaw in their theorem, which proves a scaling and is sufficient as stated;
it is why a contract, which needs the dimension itself, needs the constants.
Those bounds also stop at a single bundle; the composition results of §4–§5 (multi-hop chaining,
offline compilation, typed projection) and the contracts built on them
are, to our knowledge, not covered there. Holographic embeddings of
knowledge graphs [@nickel2016hole] also store (s, r, o) triples in
compositional holographic vectors, but learn the embeddings for link
prediction; ABM learns nothing and stores the facts themselves in one
trace. Hopfield networks [@hopfield1982] share the
interference-plus-extreme-value mechanism (our measured c ≈ 0.07–0.09 at
M ≈ 2N recalls the 0.138·N regime of Amit, Gutfreund and Sompolinsky
[@amit1985storing]). ProofWriter [@tafjord2021proofwriter] and RuleTaker
[@clark2020ruletaker] study soft theorem proving with transformers; our use
of ProofWriter inverts the setting, keeping the chaining symbolic and making
the *truth oracle* algebraic. Retrieval-augmented generation
[@lewis2020rag] is the design we contrast with in §1: memory as an index
with composition delegated to the language model. The confluence results of
§4 are standard rewriting theory [@baader1998term] applied to a new algebra.

Our contribution relative to this literature is the *resource theory*: laws
with confidence intervals and derivations, a calculus whose cost semantics
is sound under the independence assumption of §4, predictions issued before
measurement,
and an axiomatization justified by operator-ablation.

## 9. Open problems

1. Distributional confluence for nested probabilistic redexes.
2. The ABM complexity class (polynomial D, O(1) controller); working
   conjecture: the neighbourhood of bounded-width branching programs.
3. Exact finite-D independence in the hop-composition lemma (first-order
   decorrelation proved; joint independence measured but assumed).
4. Sign-saturation correction to Law VII at extreme weights.
5. Negation and quantifiers in the algebraic truth oracle.
6. Φ-descent dynamics off-codebook (Hopfield-style multi-step cleanup).
7. An observed residual in the compiler-ranking experiment suggests
   that correlated grounding errors may reduce effective memory load
   on unaffected queries beyond the first-order model (+4 pt on the
   clustered extractor, consistent across seeds). Only one clustering
   type has been tested; characterization is left for future work.
8. The drift of k with D (0.89 at D ≤ 1024, 0.93–0.94 above): a finite-size
   correction that, if derived, would predict k instead of measuring it.
9. Missing facts help more than N_eff(ε) predicts (+10 points); and on
   real hubs the prediction is 5.5 points pessimistic, possibly because
   queries with several true objects have several targets.
10. A dense-subgraph version of the FB15k-237 test, preregistered, to test
    Law IV where entities recur.

## 10. Conclusion

ABM is not proposed as a better retriever — on generic semantic search
it is not competitive, and we say so with measurements. It is proposed
as a *computational model of associative memory with predictable
resources*: three necessary and independent operators, laws that
survived deliberate falsification attempts, a calculus in which
composition and abstraction are normal-form phenomena, and contracts
(capacity, depth, reliability) that can be issued before the system runs,
provided their measured bias is declared with them. The theory's strongest evidence is that it corrected and retired
its own laws — and that its predictions, from depth-scaling to
compilation to typed projection, were confirmed *after* being derived.

---

*Reproducibility: every number in this paper is produced by a script in
`examples/` with results committed as JSON; the formal specification is
frozen as FORMALISM.md v2.1; the reference implementation
(`reference/abm.py`, 257 lines, numpy-only, deterministic) passes the
property tests derived from the axioms. The full test suite runs in
continuous integration on Linux x86-64 (Python 3.10–3.13, NumPy 1.24–2.5),
macOS arm64 and Windows x86-64, with codewords checked bit-identical across
the three. The constant k is recomputed from the saved results by
`examples/k_from_results.py`; the figures by `examples/make_figures.py`.
Results from 10-seed reruns are stored beside the originals with an `_s10`
suffix, and the preregistration of §7 is `docs/preregistration/fb15k237.md`,
committed before its run.*


## References
