"""abm.sizing — il dimensionatore: quale famiglia VSA, a quale dimensione, per queste triple.

Fase 2 di `abm size`: solo le previsioni a priori, nessuna memoria costruita (la verifica
empirica è la fase 3). I modelli sono quelli validati dalle preregistrazioni 21
(docs/preregistration/vsa_families.md) e 22 (docs/preregistration/generalization.md),
copiati e ripuliti dagli harness congelati `examples/vsa_families_prereg.py` e
`examples/generalization_prereg.py` (che restano la fonte: questo modulo deve dare le
stesse previsioni, vedi tests/test_sizing.py). Solo numpy.

Famiglie (stesso encoding f = s ∘ ρ(r) ∘ o; `dim` = numero di componenti):
- MAP-B    la reference ABM (bipolare, maggioranza): `abm.exact.predict_queries`.
- MAP-I    bipolare, somma intera: gaussiana FKS con la contabilità (v = 1).
- FHRR     fasori, somma complessa salvata in float16: gaussiana FKS (v = 1/2).
- HRR      gaussiani, convoluzione circolare, float32: gaussiana con le molteplicità
           degli atomi (Plate), varianze eterogenee.
- BSDC-OR  blocchi sparsi, traccia OR: occupazione realizzata. dim = B·L.
- BSDC-S   blocchi sparsi, traccia a conteggi: pmf esatta del rumore, pareggi esatti
           (`bsdc_s_exact` della 22). dim = B·L.

Contabilità (per tutte): (s, r, o) e (o, r, s) sono un vettore di peso 2 (gemelli),
x con (x, r, s) memorizzato è un alias di (s, r), più oggetti veri sono più bersagli,
codebook = soggetti + relazioni + oggetti distinti. `target` è un'accuratezza assoluta
(media sulle query), non una frazione del tetto degli alias.

Affidabilità (`_reliability`): solo strutturale. Ogni statistica di STRUCT_RANGES (quote di
alias, gemelli, coppie {s, o} con più relazioni, (s, r) con più oggetti, frequenza massima
e p99 di soggetti, oggetti e relazioni) fuori dall'intervallo delle medie di cella delle
preregistrazioni 21–22 → "unvalidated"; MAP-B con coppie a più relazioni o su grafo denso
→ almeno "caution". OBSERVED_ERROR è l'errore osservato nei benchmark (quali grafi, quante
celle: vedi ObservedError.context), mostrato come contesto e mai usato per decidere
"validated". "validated" non è una garanzia sull'errore.

Verifica (fase 3, `recommend(..., verify=True)`). Due insiemi DISGIUNTI di semi derivati da
`rng_seed`: S_sel (selezione di D) e S_rep (report), `seeds` ciascuno (default 8). Se le
query (passate, o tutte le coppie (s, r) presenti) sono più di `verify_queries`, anche le
query si dividono: k = min(verify_queries, Q // 2) per la selezione e altre k, disgiunte,
per il report; altrimenti entrambe usano tutte le query. Una risposta è corretta se è uno
degli oggetti veri di (s, r).
Criterio: D è accettato se il LIMITE INFERIORE dell'IC 95% misurato su S_sel è ≥ target −
tol (tol dichiarata, default 0.01 = 1 punto). Si parte dal D del modello d0; se rifiutato,
raddoppio (multipli del passo) fino al primo D accettato o a verify_d_max (default
min(8·d0, d_max)), poi bisezione fra l'ultimo rifiutato e il primo accettato. Al D finale
measured, measured_ci e per_memory_q10 si misurano su S_rep (e sulle query di report):
non sono la stessa misura usata per scegliere D, quindi non ne ereditano la selezione.
Se la MEDIA del report è < target − tol: avviso e verified = False (non l'IC_inf: con il
modello esatto, μ ≈ target, l'IC_inf del report cadrebbe sotto target − tol circa una
volta su due, e "non verificato" non direbbe nulla).
Potenza. Con vero valore medio μ = target − δ e δ ≥ tol, P(accetta) = P(x̄ − t·s/√n ≥
target − tol) ≤ P(T_{n−1} ≥ t_{0.975}) = 0.025 per OGNI n e ogni sd (i semi i.i.d., le
medie per seme circa normali): un ottimismo ≥ tol si rileva con potenza ≥ 0.975, e uno di
2 punti con potenza maggiore. Il numero di semi non serve alla potenza ma al costo dei
falsi rifiuti (D gonfiato quando il modello è esatto): con μ = target si accetta d0 solo se
x̄ − μ ≥ t·s/√n − tol. Misura (grafo uniforme N = 200, ~190 query, MAP-B, 40 semi,
sottoinsiemi casuali; sd per seme 2.1 punti): P(accetta d0) = 0.13 con 3 semi, 0.40 con
8, 0.61 con 12. Con ottimismo 0.7 / 2.0 / 4.8 / 10.2 punti (grafi "club" con 20/30/45/60
giocatori, il caso del revisore) P(accetta) con 8 semi = 0.037 / 0 / 0 / 0 (il vecchio
criterio, IC_sup ≥ target con 3 semi: 0.95 / 0.84 / 0.69 / 0.09). Default 8 semi: il
costo di un falso rifiuto è un D con ~1 punto di margine, non un errore.
per_memory_q10: quantile 10% (empirico, interpolato) delle accuratezze delle singole
memorie di S_rep: l'utente usa UNA memoria, e la media non dice quanto può andare male
la sua; con 8 memorie è una stima grezza (fra la peggiore e la seconda peggiore).
IC: con seeds ≥ 2, media dei semi ± t(seeds − 1, 0.975)·sd/√seeds (i semi sono i cluster);
con seeds = 1, IC binomiale di Wilson sulle query (ignora la correlazione fra query della
stessa memoria). Memoria: prima di costruire, la memoria di picco è stimata
(`families.estimate_bytes` se c'è, altrimenti `_fallback_bytes`); un D oltre
`max_mem_bytes` (default 1 GiB) non viene costruito e la verifica si ferma lì con un avviso.

Bit: `bits` = traccia, con N = fatti con peso (triple memorizzate, duplicati e gemelli
contati: `bits_per_component`); con count_codebook=True si aggiungono gli atomi
memorizzati esplicitamente (`explicit_codebook_bits`; default False perché le famiglie li
rigenerano da seme e nomi). Da non confondere con `families.Memory.codebook_bits()`, che
conta solo l'INDICE simbolo → riga (M·⌈log2 M⌉ bit), assumendo gli atomi rigenerati:
sono due convenzioni diverse (atomi espliciti vs indice), con nomi diversi apposta.

API: predict, min_dim, recommend (→ list[Recommendation]), FAMILIES, OBSERVED_ERROR,
STRUCT_RANGES, bits_per_component, explicit_codebook_bits.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import exact
from . import families as _families

FAMILIES = ("MAP-B", "MAP-I", "FHRR", "HRR", "BSDC-OR", "BSDC-S")
STEP = 64                    # granularità di D per MAP-B, MAP-I, FHRR, HRR (come gli harness)
D_MAX = 1 << 18              # come la preregistrazione 22
FLOAT_BITS = {"FHRR": 32, "HRR": 32}   # FHRR: 2 × float16; HRR: float32
PI_DRAWS = 2000              # estrazioni dell'occupazione realizzata (BSDC-OR)
PI_SEED = 21_000_021         # seme FISSO delle estrazioni di BSDC-OR: quello della 21, così
                             # le previsioni coincidono con quelle congelate

# ---- errore osservato nei benchmark ------------------------------------------
# |misura − previsione| in punti percentuali, a 1 hop, per famiglia e regime, sulle celle
# "calib" della 21 (FB15k-237, WN18RR; N 100, 400; target 0.50 e 0.85 del tetto; 10 seed)
# e sulle celle a 1 hop della 22 (CoDEx-M, YAGO3-10; N 100, 400, 1600; 20 seed), riunite.
# Fonte: results/vsa_families_prereg_results.json (rows, block=calib) e
# results/generalization_prereg_results.json (rows, hops=1), |meas − pred| per cella.
# BSDC-S e HRR: solo la 22 (la 21 usava per BSDC-S il modello con i pareggi approssimati,
# distorto di +1.7–2.1 punti, ora sostituito; HRR non c'era).
# È un dato STORICO su quei sottografi (sei grafi pubblici, campionati uniform o a BFS):
# non è un intervallo di errore per un grafo qualsiasi, e non entra nella regola di
# affidabilità (che guarda solo la struttura, vedi _reliability).
# (mae, cella peggiore, celle, cella peggiore: provenienza)
BENCH_CONTEXT = {
    "21": "prereg 21: FB15k-237 + WN18RR subgraphs, N 100/400, 10 seeds, calib cells",
    "22": "prereg 22: CoDEx-M + YAGO3-10 subgraphs, N 100/400/1600, 20 seeds, 1-hop cells",
}
OBSERVED_ERROR = {
    ("MAP-B", "uniform"): (0.74, 1.79, 20, "21: FB15k-237 N=400 t=0.50"),
    ("MAP-B", "dense"): (1.51, 7.22, 20, "22: YAGO3-10 N=1600 t=0.85 (ottimista)"),
    ("MAP-I", "uniform"): (0.81, 2.98, 20, "21: FB15k-237 N=100 t=0.50"),
    ("MAP-I", "dense"): (0.81, 3.92, 20, "21: FB15k-237 N=100 t=0.50"),
    ("FHRR", "uniform"): (0.50, 1.89, 20, "21: FB15k-237 N=100 t=0.50"),
    ("FHRR", "dense"): (0.47, 1.21, 20, "22: CoDEx-M N=100 t=0.50"),
    ("HRR", "uniform"): (0.71, 1.39, 12, "22: CoDEx-M N=400 t=0.50"),
    ("HRR", "dense"): (0.66, 1.74, 12, "22: CoDEx-M N=100 t=0.50"),
    ("BSDC-OR", "uniform"): (0.72, 1.83, 20, "21: WN18RR N=100 t=0.50"),
    ("BSDC-OR", "dense"): (0.78, 3.23, 20, "22: CoDEx-M N=100 t=0.50"),
    ("BSDC-S", "uniform"): (0.65, 2.90, 12, "22: YAGO3-10 N=100 t=0.50"),
    ("BSDC-S", "dense"): (0.84, 2.55, 12, "22: CoDEx-M N=100 t=0.50"),
}
# Range testato. N: 100–1600 (N = 3200 fu tolto dalla griglia della 22 per il budget di
# tempo, regola scritta nel documento: non è stato misurato).
N_MIN_TESTED, N_MAX_TESTED = 100, 1600
# Intervalli strutturali [min, max] delle MEDIE DI CELLA (cella = dataset × schema × N; 20
# celle, 320 sottografi) ricostruite con gli harness congelati come fa la regressione
# (examples/vsa_families_prereg.py: SeedSequence([SEED_BASE, di, si, N, k]).spawn(2), seed
# 0–9; examples/generalization_prereg.py: .spawn(3), seed 0–19; stesse query, Q_MAX = 200)
# e misurate con _Graph (alias_share, twin_share) e _structure. Script di ricostruzione:
# tests/test_sizing.py::test_struct_ranges_reproduce (salta senza i dati). Le quote di
# alias e gemelli coincidono con le costanti precedenti (0.094, 0.144). Una statistica
# fuori da [min, max] → "unvalidated". Gli intervalli sono larghi perché le celle dense
# (BFS) contengono hub e relazioni multiple: stare dentro NON vuol dire stare in un regime
# dove il modello ha retto (YAGO3-10 dense, dove MAP-B ha fallito, è il massimo di
# pair_multirel_share); per questo MAP-B ha una regola propria (vedi _reliability).
STRUCT_RANGES = {
    # nome: (min, max, cella del min, cella del max)
    "alias_share": (0.0, 0.094, "22 CoDEx-M dense 100", "21 WN18RR dense 400"),
    "twin_share": (0.0, 0.143966, "21 FB15k-237 uniform 100", "21 WN18RR dense 400"),
    "pair_multirel_share": (0.0, 0.351375, "21 FB15k-237 uniform 100",
                            "22 YAGO3-10 dense 400"),
    "multiobj_share": (0.0, 0.150564, "22 YAGO3-10 uniform 100", "21 FB15k-237 dense 100"),
    "s_max_share": (0.0014375, 0.284, "22 YAGO3-10 uniform 1600", "21 FB15k-237 dense 100"),
    "s_p99_share": (0.000974063, 0.21324, "22 YAGO3-10 uniform 1600",
                    "21 FB15k-237 dense 100"),
    "o_max_share": (0.0095, 0.79925, "21 WN18RR uniform 400", "22 CoDEx-M dense 400"),
    "o_p99_share": (0.00209531, 0.660379, "22 YAGO3-10 uniform 1600", "22 CoDEx-M dense 400"),
    "r_max_share": (0.06175, 0.804, "21 FB15k-237 uniform 400", "22 CoDEx-M dense 400"),
    "r_p99_share": (0.05351, 0.75119, "21 FB15k-237 uniform 400", "22 CoDEx-M dense 400"),
}
# Regime uniform/dense (sceglie solo quale errore osservato mostrare). Quote: massimo delle
# medie di cella uniform (alias 0.0085 della 22 CoDEx-M 1600; gemelli 0.00126 della 21
# WN18RR 400). Conteggi: massimo per SINGOLO sottografo uniform (alias 6 query, gemelli 2
# vettori), stessa ricostruzione. "dense" solo se conteggio E quota superano entrambi.
ALIAS_MAX_UNIFORM = 0.0085
TWIN_MAX_UNIFORM = 0.00126
UNIFORM_MAX = {"alias_share": ALIAS_MAX_UNIFORM, "twin_share": TWIN_MAX_UNIFORM,
               "alias_count": 6, "twin_count": 2}
ALIAS_MAX_TESTED = STRUCT_RANGES["alias_share"][1]
TWIN_MAX_TESTED = STRUCT_RANGES["twin_share"][1]
MAPB_DENSE_SIGNED = -3.5     # 22, MAP-B YAGO3-10 dense: errore con segno medio (punti)
BSDC_S_SIGNED_MAX = 0.83     # 22, BSDC-S: |errore con segno| massimo per gruppo (punti)

# ---- verifica -------------------------------------------------------------------
VERIFY_SEEDS = 8             # semi di default, per la selezione e (altrettanti) per il report
VERIFY_TOL = 0.01            # tolleranza dichiarata: accetta se IC_inf ≥ target − tol
MAX_MEM_BYTES = 1 << 30      # memoria di picco stimata oltre la quale la verifica salta un D


@dataclass(frozen=True)
class ObservedError:
    """Errore |misura − previsione| osservato nei benchmark 21–22 per famiglia e regime:
    storico, su quei sottografi; non un limite all'errore sul grafo passato."""
    regime: str          # "uniform" | "dense"
    mae: float           # punti percentuali, errore medio osservato sulle celle
    worst: float         # punti percentuali, cella peggiore osservata
    cells: int           # celle del benchmark su cui è misurato
    worst_cell: str      # provenienza della cella peggiore
    context: str         # quali grafi, carichi e semi

    def __repr__(self):
        return (f"ObservedError(benchmark error, not a bound for this graph: regime={self.regime}, "
                f"mae={self.mae} pt, worst={self.worst} pt [{self.worst_cell}], "
                f"over {self.cells} cells; {self.context})")


def _observed_error(family, regime):
    mae, worst, cells, src = OBSERVED_ERROR[(family, regime)]
    ctx = "; ".join(BENCH_CONTEXT[k] for k in ("21", "22")
                    if family not in ("HRR", "BSDC-S") or k == "22")
    return ObservedError(regime, mae, worst, cells, src, ctx)


@dataclass
class Recommendation:
    family: str
    dim: int             # componenti (BSDC: B·L)
    bits: int
    kbytes: float
    predicted: float     # accuratezza media prevista sulle query
    observed_error: ObservedError
    reliability: str     # "validated" | "caution" | "unvalidated"
    warnings: list = field(default_factory=list)
    # verifica (recommend(..., verify=True)); senza verifica restano None/False
    measured: float = None       # REPORT: media sui semi freschi S_rep al dim finale
    measured_ci: tuple = None    # REPORT: IC 95% (a cluster per seme; Wilson se seeds = 1)
    per_memory_q10: float = None  # REPORT: quantile 10% dell'accuratezza della SINGOLA memoria
    selection_measured: float = None  # SELEZIONE: media sui semi S_sel al dim finale
    selection_ci: tuple = None   # SELEZIONE: IC 95%
    verified: bool = False       # selezione: IC_inf ≥ target − tol; report: media ≥ target − tol
    tol: float = None            # tolleranza usata (accuratezza assoluta)
    dim_predicted: int = None    # il dim del modello, prima della verifica
    adjusted: bool = False       # True se la verifica ha cambiato dim
    codebook_bits: int = 0       # explicit_codebook_bits inclusi in bits (solo count_codebook=True)
    over_budget: bool = False    # True se la correzione ha portato bits oltre budget_bits


# ---- struttura delle triple --------------------------------------------------

class _Graph:
    """Triple, query e contabilità (come la Cell degli harness, senza campionamento)."""

    def __init__(self, triples, queries=None):
        triples = [tuple(t) for t in triples]
        if not triples:
            raise ValueError("no triples")
        for t in triples:
            if len(t) != 3:
                raise ValueError(f"a triple must be (subject, relation, object), got {t!r}")
        self.triples = triples
        self.n = len(triples)
        objects, into = defaultdict(set), defaultdict(set)
        for s, r, o in triples:
            objects[(s, r)].add(o)
            into[(o, r)].add(s)
        self.objects, self.into = objects, into
        self.queries = sorted(objects) if queries is None else [tuple(q) for q in queries]
        if not self.queries:
            raise ValueError("no queries")
        exact._check_queries(triples, self.queries, "raise")
        self.self_loops = any(s == o for s, _r, o in triples)
        self.symbols = sorted({x for t in triples for x in t})
        self.m = len(self.symbols)
        self.weights = exact.fact_weights(triples)
        self.s2 = float(sum(w * w for w in self.weights.values()))
        self.n_vec = len(self.weights)
        self.wcounts = tuple(sorted(Counter(self.weights.values()).items()))
        self.sigs = [self._sig(s, r) for s, r in self.queries]
        self.alias_share = float(np.mean([len(a) > 0 for _g, a in self.sigs]))
        self.twin_share = float(np.mean([w == 2 for w in self.weights.values()]))
        self.ceiling = exact.ceiling(triples, self.queries)
        self.n_alias_q = int(sum(len(a) > 0 for _g, a in self.sigs))
        self.n_twins = int(sum(w == 2 for w in self.weights.values()))
        self.stats = _structure(triples, objects)
        self._cache = {}
        self._hrr = None

    def _sig(self, s, r):
        good = self.objects.get((s, r), set())
        bad = self.into.get((s, r), set()) - good
        gw = tuple(sorted(self.weights[exact.fact_key(s, r, o)] for o in good))
        aw = tuple(sorted(self.weights[exact.fact_key(x, r, s)] for x in bad))
        return gw, aw

    @property
    def regime(self):
        """"dense" se alias o gemelli superano INSIEME il massimo assoluto e la quota
        massima viste nelle celle uniform delle 21-22 (vedi UNIFORM_MAX). Solo la quota
        farebbe scattare "dense" per un solo gemello su un grafo piccolo (1/199 > 0.00126);
        solo il conteggio ignorerebbe la scala."""
        dense_alias = (self.n_alias_q > UNIFORM_MAX["alias_count"]
                       and self.alias_share > UNIFORM_MAX["alias_share"])
        dense_twin = (self.n_twins > UNIFORM_MAX["twin_count"]
                      and self.twin_share > UNIFORM_MAX["twin_share"])
        return "dense" if dense_alias or dense_twin else "uniform"


def _structure(triples, objects):
    """Statistiche strutturali del grafo (sulle triple distinte, non sulle query):
    - pair_multirel_share: quota di fatti la cui coppia NON ordinata {s, o} compare con
      ≥ 2 relazioni distinte (inverse con nome diverso incluse: (s, r1, o) + (o, r2, s));
      in MAP-B questi fatti condividono s e o e si annullano nella maggioranza;
    - multiobj_share: quota delle coppie (s, r) con più oggetti;
    - {s,o,r}_max_share, {s,o,r}_p99_share: numero di fatti del soggetto (dell'oggetto,
      della relazione) più frequente, e il percentile 99 dei conteggi, come quota di N
      (i conteggi assoluti crescono con N; le quote si confrontano fra carichi)."""
    uniq = sorted(set(triples))
    n = len(uniq)
    rels_of = defaultdict(set)
    cnt = {"s": Counter(), "o": Counter(), "r": Counter()}
    for s, r, o in uniq:
        rels_of[frozenset((s, o))].add(r)
        cnt["s"][s] += 1
        cnt["o"][o] += 1
        cnt["r"][r] += 1
    out = {"pair_multirel_share":
           sum(len(rels_of[frozenset((s, o))]) >= 2 for s, _r, o in uniq) / n,
           "multiobj_share": float(np.mean([len(v) > 1 for v in objects.values()]))}
    for k, c in cnt.items():
        v = np.array(list(c.values()), float)
        out[f"{k}_max_share"] = float(v.max() / n)
        out[f"{k}_p99_share"] = float(np.percentile(v, 99) / n)
    return out


# ---- MAP-I, FHRR: gaussiana FKS con la contabilità (dalla 21) -----------------

_ERFC = np.vectorize(math.erfc)


def _log_cdf_exact(z):
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(0.5 * _ERFC(-z / math.sqrt(2.0)), 1e-300))


def _gaussian_max(cands, null_mu, null_sd, n_null):
    """P(il massimo è un candidato 'ok'); cands = [(media, sd, ok)], nulli i.i.d."""
    mus = [(m - null_mu) / null_sd for m, _s, _ok in cands]
    sds = [max(s, 1e-9) / null_sd for _m, s, _ok in cands]
    x = np.linspace(-9.0, max(mus) + 9.0, 6001)
    dx = x[1] - x[0]
    logc = [_log_cdf_exact((x - m) / s) for m, s in zip(mus, sds)]
    tot = np.sum(logc, axis=0) + n_null * _log_cdf_exact(x)
    acc = 0.0
    for i, (_m, _s, ok) in enumerate(cands):
        if ok:
            z = (x - mus[i]) / sds[i]
            dens = np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sds[i])
            acc += float(np.sum(dens * np.exp(tot - logc[i])) * dx)
    return min(acc, 1.0)


def _gaussian_dense_acc(gw, aw, s2, n_null, dim, v):
    """MAP-I (v = 1), FHRR (v = 1/2): segnale w·D, varianza v·D·(S2 − w²); nullo v·D·S2."""
    cands = [(w * dim, math.sqrt(v * dim * (s2 - w * w)), ok)
             for ws, ok in ((gw, True), (aw, False)) for w in ws]
    return _gaussian_max(cands, 0.0, math.sqrt(v * dim * s2), n_null)


# ---- BSDC ---------------------------------------------------------------------

def bsdc_length(n):
    """L: potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà)."""
    return int(2 ** round(math.log2(n / math.log(2))))


def _binom(n, p):
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    return np.exp(lg[n] - lg - lg[::-1] + k * math.log(p) + (n - k) * math.log1p(-p))


_SUMPMF = {}


def _bsdc_s_noise(wcounts, blocks, length):
    """pmf esatta di Σ_b Σ_w w·Bin(n_w, 1/L): punteggio di un nullo in BSDC-S."""
    key = (wcounts, blocks, length)
    if key not in _SUMPMF:
        per = np.array([1.0])
        for w, n_w in wcounts:
            b = _binom(n_w, 1.0 / length)
            part = np.zeros(w * n_w + 1)
            part[::w] = b
            per = np.convolve(per, part)
        per = per[: max(1, np.flatnonzero(per > 1e-15).max() + 1)]
        tot, base, k = np.array([1.0]), per, blocks
        while k:                                   # potenza per quadrati
            if k & 1:
                tot = np.convolve(tot, base)
            base = np.convolve(base, base)
            k >>= 1
            tot = tot[: np.flatnonzero(tot > 1e-15).max() + 1]
            base = base[: np.flatnonzero(base > 1e-15).max() + 1]
        _SUMPMF[key] = tot / tot.sum()
    return _SUMPMF[key]


def _bsdc_s_exact(gw, aw, wcounts, n_null, blocks, length):
    """BSDC-S (22): pmf marginale esatta del rumore, punteggi indipendenti, pareggi
    divisi uniformemente a caso (come la misura)."""
    if not gw:
        return 0.0
    noise = _bsdc_s_noise(wcounts, blocks, length)
    top = len(noise) + blocks * (max(gw + aw) + 1)
    null = np.zeros(top)
    null[: len(noise)] = noise
    sig = []
    for w in gw + aw:
        others = tuple((v, c - (v == w)) for v, c in wcounts if c - (v == w) > 0)
        nz = _bsdc_s_noise(others, blocks, length)
        f = np.zeros(top)
        f[w * blocks: w * blocks + len(nz)] = nz[: top - w * blocks]
        sig.append(f)
    sig = np.asarray(sig)
    cdf_prev = np.cumsum(sig, axis=1) - sig                      # P(< x)
    F = np.cumsum(null)
    g, k = len(gw), len(gw) + len(aw)
    acc = 0.0
    for x in np.flatnonzero(sig.sum(axis=0) > 1e-15):
        if F[x] <= 0:
            continue
        poly = np.zeros((g + 1, k - g + 1))
        poly[0, 0] = 1.0
        for i in range(k):
            a, b = sig[i, x], cdf_prev[i, x]
            new = poly * b
            if i < g:
                new[1:, :] += poly[:-1, :] * a
            else:
                new[:, 1:] += poly[:, :-1] * a
            poly = new
        q = min(null[x] / F[x], 1.0)
        j = np.arange(n_null + 1)
        pj = _binom(n_null, q) if 0 < q < 1 else (j == (n_null if q >= 1 else 0)) * 1.0
        base = F[x] ** n_null
        for tg in range(1, g + 1):
            for ta in range(k - g + 1):
                if poly[tg, ta] > 0:
                    acc += poly[tg, ta] * base * float(np.sum(pj * tg / (tg + ta + j)))
    return float(min(acc, 1.0))


_OCC = {}


def _occupancy_pmf(n, length):
    """P(K = k) bin occupati dopo n palline uniformi in L bin (esatta, per ricorrenza)."""
    key = (n, length)
    if key not in _OCC:
        p = np.zeros(length + 1)
        p[0] = 1.0
        k = np.arange(length + 1)
        for _ in range(n):
            q = p * k / length
            q[1:] += p[:-1] * (length - k[:-1]) / length
            p = q
        _OCC[key] = p
    return _OCC[key]


_PI = {}


def _pi_draws(n, length, blocks):
    """Estrazioni di π = Π_b K_b/L con K_b dall'occupazione esatta (seme fisso)."""
    key = (n, length, blocks)
    if key not in _PI:
        pmf = _occupancy_pmf(n, length)
        rng = np.random.default_rng([PI_SEED, n, length, blocks])
        k = rng.choice(length + 1, size=(PI_DRAWS, blocks), p=pmf / pmf.sum())
        _PI[key] = np.exp(np.sum(np.log(np.maximum(k, 1) / length), axis=1))
    return _PI[key]


_T = np.linspace(0.0, 1.0, 2001)
_PGRID = np.concatenate(([0.0], np.logspace(-12, 0, 600)))
_F = {}
_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2


def _bsdc_or_acc(gw, aw, n_vec, n_null, blocks, length):
    """g · E[1/(g + a + X)], X | π ~ Bin(n_null, π), π con occupazione realizzata."""
    if not gw:
        return 0.0
    g, c = len(gw), len(gw) + len(aw)
    key = (c, n_null)
    if key not in _F:
        pg = _PGRID[:, None]
        _F[key] = _trapezoid(_T[None, :] ** (c - 1) * (1 - pg + pg * _T[None, :]) ** n_null,
                             _T, axis=1)
    return float(g * np.mean(np.interp(_pi_draws(n_vec, length, blocks), _PGRID, _F[key])))


# ---- HRR: molteplicità di Plate (dalla 22) ------------------------------------

_FTAB = np.array([math.factorial(k) for k in range(8)], float)
_ZT = np.linspace(-40.0, 40.0, 160001)
_LT = None


def _log_cdf_table(z):
    """log Φ per interpolazione su una tabella (passo 5e-4) di log Φ esatta."""
    global _LT
    if _LT is None:
        _LT = _log_cdf_exact(_ZT)
    return np.interp(z, _ZT, _LT)


def _hrr_struct(g):
    idx = {x: i for i, x in enumerate(g.symbols)}
    atoms, w, fact_of = [], [], {}
    for j, k in enumerate(g.weights):
        pair, r = k
        a, b = sorted(pair)
        atoms.append((idx[a], idx[r], idx[b]))
        w.append(g.weights[k])
        fact_of[(pair, r)] = j
    w = np.asarray(w, float)
    return {"atoms": np.asarray(atoms), "w": w, "w2": w * w, "id": idx, "m": len(g.symbols),
            "fact_of": fact_of, "mom": {}}


def _hrr_query_moments(hc, s, r, cand_good, cand_alias):
    """(medie e varianze ×D dei candidati a segnale, varianze ×D dei nulli)."""
    atoms, w2, w = hc["atoms"], hc["w2"], hc["w"]
    key = (hc["id"][s], hc["id"][r])
    m0 = np.ones(len(w))
    for col in range(3):
        a = atoms[:, col]
        first = np.ones(len(w), bool)
        for c2 in range(col):
            first &= atoms[:, c2] != a
        tot = (atoms == a[:, None]).sum(axis=1) + (a == key[0]) + (a == key[1])
        m0 *= np.where(first, _FTAB[tot], 1.0)
    for k in set(key):
        kk = (key[0] == k) + (key[1] == k)
        absent = (atoms == k).sum(axis=1) == 0
        m0 *= np.where(absent, math.factorial(kk), 1.0)
    base = float(np.sum(w2 * m0))
    extra = np.zeros(hc["m"])
    np.add.at(extra, atoms.ravel(), np.repeat(w2 * m0, 3))
    nkey = np.zeros(hc["m"])
    nkey[key[0]] += 1
    nkey[key[1]] += 1
    var = base * (nkey + 1) + extra
    sig = []
    for names, ok in ((cand_good, True), (cand_alias, False)):
        for c in names:
            ci = hc["id"][c]
            j = hc["fact_of"][frozenset((s, c)), r]
            v = var[ci] - w2[j] * m0[j] * (np.sum(atoms[j] == ci) + nkey[ci] + 1) \
                + 14.0 * w2[j]
            sig.append((float(w[j]), float(v), ok, ci))
    mask = np.ones(hc["m"], bool)
    mask[list({x[3] for x in sig})] = False
    return sig, var[mask]


def _gaussian_max_het(cands, null_var, dim):
    """Come _gaussian_max, con varianze dei nulli eterogenee (raggruppate a 1e-3)."""
    sd_n = np.sqrt(np.asarray(null_var) / dim)
    vals, counts = np.unique(np.exp(np.round(np.log(sd_n), 3)), return_counts=True)
    mus = np.array([c[0] for c in cands])
    sds = np.array([math.sqrt(max(c[1], 1e-12) / dim) for c in cands])
    lo = min(-9 * (vals.max() if len(vals) else 1), float(np.min(mus - 9 * sds)))
    hi = max(float(np.max(mus + 9 * sds)), 9 * (vals.max() if len(vals) else 0))
    x = np.linspace(lo, hi, 4001)
    dx = x[1] - x[0]
    tot = np.zeros_like(x)
    for v, n in zip(vals, counts):
        tot += n * _log_cdf_table(x / v)
    logc = [_log_cdf_table((x - m) / s) for m, s in zip(mus, sds)]
    tot += np.sum(logc, axis=0)
    acc = 0.0
    for i, c in enumerate(cands):
        if c[2]:
            z = (x - mus[i]) / sds[i]
            dens = np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sds[i])
            acc += float(np.sum(dens * np.exp(tot - logc[i])) * dx)
    return min(acc, 1.0)


def _hrr_moments(g):
    """Momenti per query, indipendenti da D (calcolati una volta per grafo): medie e
    varianze ×D dei candidati, maschera 'ok', varianze ×D dei nulli. None = senza oggetti."""
    if g._hrr is None:
        g._hrr = _hrr_struct(g)
    hc = g._hrr
    mc = hc["mom"]
    out = []
    for s, r in g.queries:
        if (s, r) not in mc:
            good = sorted(g.objects.get((s, r), ()))
            bad = sorted(g.into.get((s, r), set()) - set(good))
            if good:
                sig, nv = _hrr_query_moments(hc, s, r, good, bad)
                mc[(s, r)] = (np.array([c[0] for c in sig], float),
                              np.array([c[1] for c in sig], float),
                              np.array([c[2] for c in sig], bool), np.asarray(nv, float))
            else:
                mc[(s, r)] = None
        out.append(mc[(s, r)])
    return out


_LT_DIFF = None


def _log_cdf_uniform(z):
    """Come _log_cdf_table (np.interp sulla stessa tabella), ma con l'indice calcolato
    direttamente: la griglia è uniforme, niente ricerca binaria. Stessi valori a meno
    dell'arrotondamento (~1e-16 relativo)."""
    global _LT_DIFF
    if _LT_DIFF is None:
        _log_cdf_table(0.0)
        _LT_DIFF = np.append(np.diff(_LT), 0.0)
    n = len(_ZT) - 1
    t = z - _ZT[0]
    t *= n / (_ZT[-1] - _ZT[0])
    np.clip(t, 0.0, float(n), out=t)
    j = t.astype(np.intp)
    t -= j
    t *= _LT_DIFF[j]
    t += _LT[j]
    return t


_HRR_CHUNK = 1 << 22         # elementi per blocco di query nella versione vettorizzata


def _hrr_accs(g, dim):
    """_gaussian_max_het per tutte le query, vettorizzata a blocchi di query. I momenti
    sono precalcolati (_hrr_moments); per D restano solo scala (÷D), raggruppamento dei
    nulli (arrotondato in log, quindi da rifare per D: non si fattorizza esattamente) e
    l'integrale sulla griglia."""
    moms = _hrr_moments(g)
    per = np.zeros(len(moms))
    live = [i for i, m in enumerate(moms) if m is not None]
    rows = []
    for i in live:
        mus, v, ok, nv = moms[i]
        sd_n = np.sqrt(nv / dim)
        vals, counts = np.unique(np.exp(np.round(np.log(sd_n), 3)), return_counts=True)
        sds = np.sqrt(np.maximum(v, 1e-12) / dim)
        vmax = vals.max() if len(vals) else None
        lo = min(-9 * (vmax if vmax is not None else 1), float(np.min(mus - 9 * sds)))
        hi = max(float(np.max(mus + 9 * sds)), 9 * (vmax if vmax is not None else 0))
        rows.append((i, mus, sds, ok, vals, counts, lo, hi))
    nx = 4001
    a = 0
    while a < len(rows):
        b, width = a, 0
        gmax = cmax = 0
        while b < len(rows):
            gm, cm = max(gmax, len(rows[b][4])), max(cmax, len(rows[b][1]))
            if b > a and (b - a + 1) * (gm + cm) * nx > _HRR_CHUNK:
                break
            gmax, cmax, b = gm, cm, b + 1
        blk = rows[a:b]
        q = len(blk)
        x = np.stack([np.linspace(r[6], r[7], nx) for r in blk])          # (q, nx)
        dx = x[:, 1] - x[:, 0]
        vals = np.ones((q, max(gmax, 1)))
        cnt = np.zeros((q, max(gmax, 1)))
        mus = np.zeros((q, cmax))
        sds = np.ones((q, cmax))
        cm = np.zeros((q, cmax), bool)
        okm = np.zeros((q, cmax), bool)
        for k, r in enumerate(blk):
            vals[k, :len(r[4])] = r[4]
            cnt[k, :len(r[5])] = r[5]
            mus[k, :len(r[1])] = r[1]
            sds[k, :len(r[2])] = r[2]
            cm[k, :len(r[1])] = True
            okm[k, :len(r[3])] = r[3]
        tot = np.einsum("qg,qgx->qx", cnt, _log_cdf_uniform(x[:, None, :] / vals[:, :, None]))
        zc = (x[:, None, :] - mus[:, :, None]) / sds[:, :, None]
        logc = np.where(cm[:, :, None], _log_cdf_uniform(zc), 0.0)
        tot = tot + logc.sum(axis=1)
        dens = np.exp(-0.5 * zc * zc) / (math.sqrt(2 * math.pi) * sds[:, :, None])
        acc = np.where(okm, np.sum(dens * np.exp(tot[:, None, :] - logc), axis=2)
                       * dx[:, None], 0.0).sum(axis=1)
        per[[r[0] for r in blk]] = np.minimum(acc, 1.0)
        a = b
    return per


def _hrr_acc(g, s, r, dim):
    """Una query, percorso originale (non vettorizzato): riferimento per i test."""
    if g._hrr is None:
        g._hrr = _hrr_struct(g)
    hc = g._hrr
    good = sorted(g.objects.get((s, r), ()))
    bad = sorted(g.into.get((s, r), set()) - set(good))
    if not good:
        return 0.0
    sig, nv = _hrr_query_moments(hc, s, r, good, bad)
    return _gaussian_max_het([(m, v, ok) for m, v, ok, _ in sig], nv, dim)


# ---- previsione ----------------------------------------------------------------

def _check_family(family):
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _step(g, family):
    return bsdc_length(g.n) if family.startswith("BSDC") else STEP


def _graph(triples, queries):
    return triples if isinstance(triples, _Graph) else _Graph(triples, queries)


def _predict_graph(g, family, dim):
    key = (family, dim)
    if key in g._cache:
        return g._cache[key]
    if g.self_loops and family != "MAP-B":
        raise ValueError(f"self-loops (s, r, s) are modelled only for MAP-B, not {family}")
    if family == "MAP-B":
        if g.self_loops:
            per = exact.predict_queries(g.triples, dim, g.queries)
        else:
            # exact dipende dalla query solo per la firma: un rappresentante per firma
            rep = {}
            for q, sg in zip(g.queries, g.sigs):
                rep.setdefault(sg, q)
            sgs = list(rep)
            vals = dict(zip(sgs, exact.predict_queries(g.triples, dim, [rep[s] for s in sgs])))
            per = [vals[sg] for sg in g.sigs]
    elif family == "HRR":
        per = _hrr_accs(g, dim)
    else:
        length = bsdc_length(g.n)
        if family.startswith("BSDC") and dim < length:
            raise ValueError(f"{family}: dim must be at least L = {length} (one block)")
        cache, per = {}, []
        for gw, aw in g.sigs:
            if (gw, aw) not in cache:
                n_null = g.m - len(gw) - len(aw)
                if not gw:
                    v = 0.0
                elif family == "BSDC-OR":
                    v = _bsdc_or_acc(gw, aw, g.n_vec, n_null, dim // length, length)
                elif family == "BSDC-S":
                    v = _bsdc_s_exact(gw, aw, g.wcounts, n_null, dim // length, length)
                else:
                    v = _gaussian_dense_acc(gw, aw, g.s2, n_null, dim,
                                            1.0 if family == "MAP-I" else 0.5)
                cache[(gw, aw)] = v
            per.append(cache[(gw, aw)])
    acc = float(np.mean(per))
    g._cache[key] = acc
    return acc


def predict(triples, family, dim, queries=None) -> float:
    """Accuratezza media prevista del cleanup sulle query (s, r) (default: tutte le coppie
    presenti), per la famiglia a `dim` componenti (BSDC: dim = B·L, B = dim // L)."""
    _check_family(family)
    dim = _check_positive_int("dim", dim)
    return _predict_graph(_graph(triples, queries), family, dim)


def min_dim(triples, family, target, queries=None, d_max=D_MAX):
    """Il D minimo (multiplo di 64; BSDC: di L) con accuratezza prevista ≥ target, o None
    se non raggiungibile entro d_max (per esempio sopra il tetto degli alias)."""
    _check_family(family)
    if isinstance(target, bool) or not isinstance(target, (int, float, np.integer,
                                                          np.floating)) \
            or not 0.0 < target <= 1.0:
        raise ValueError(f"target must be an accuracy in (0, 1], got {target!r}")
    d_max = _check_positive_int("d_max", d_max)
    g = _graph(triples, queries)
    step = _step(g, family)
    top = d_max // step
    if top < 1:
        raise ValueError(f"d_max must be at least the step {step} of {family}")
    if _predict_graph(g, family, step * top) < target:
        return None
    lo, hi = 0, top
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _predict_graph(g, family, step * mid) >= target:
            hi = mid
        else:
            lo = mid
    return step * hi


def bits_per_component(family, n):
    """Bit per componente della TRACCIA. MAP-B, BSDC-OR: 1. FHRR: 2 × float16. HRR: float32.
    MAP-I, BSDC-S: interi con N + 1 valori, ceil(log2(N + 1)) bit (convenzione degli
    harness). N = fatti CON PESO, cioè le triple memorizzate contando i duplicati (un
    gemello (s, r, o) + (o, r, s) conta 2: sono due addendi della somma), come
    `families.Memory.bits()`; è len(triples) passato al dimensionatore."""
    if family in ("MAP-B", "BSDC-OR"):
        return 1
    if family in FLOAT_BITS:
        return FLOAT_BITS[family]
    return math.ceil(math.log2(n + 1))


def explicit_codebook_bits(family, dim, m, n):
    """Bit per memorizzare ESPLICITAMENTE il codebook (M atomi interi), se non lo si rigenera dal
    seme e dai nomi (che è il default delle famiglie: per questo recommend non li conta se
    non con count_codebook=True). MAP-B, MAP-I: D bit per atomo (±1). FHRR, HRR: D × 32
    (stessa precisione della traccia). BSDC: B·ceil(log2 L) per atomo (un indice per blocco)."""
    if family in ("MAP-B", "MAP-I"):
        return m * dim
    if family in FLOAT_BITS:
        return m * dim * FLOAT_BITS[family]
    length = bsdc_length(n)
    return m * (dim // length) * max(1, math.ceil(math.log2(length)))


def _range_value(g, name):
    if name in ("alias_share", "twin_share"):
        return getattr(g, name)
    return g.stats[name]


def _reliability(g, family):
    """Regole di affidabilità. Dipendono SOLO dalla struttura del grafo, mai dall'errore
    osservato nei benchmark (OBSERVED_ERROR, che è solo mostrato):
    - N fuori da [100, 1600] → "unvalidated" sopra (3200 era previsto ma tolto dalla
      griglia), "caution" sotto;
    - qualunque statistica di STRUCT_RANGES fuori dall'intervallo [min, max] delle medie di
      cella delle preregistrazioni 21–22 → "unvalidated";
    - self-loop → "unvalidated" (nessun sottografo testato ne aveva);
    - MAP-B con pair_multirel_share > 0 → almeno "caution": fatti con la stessa coppia
      {s, o} e relazioni diverse condividono s e o e nella maggioranza si annullano dove
      ρ(r1) ≠ ρ(r2); il modello li tratta come indipendenti ed è ottimista;
    - MAP-B su grafo denso → "caution": fallito su YAGO3-10 dense (ottimista di 3.5 punti
      in media, fino a 7.2; D a priori sotto il target in 4 celle su 12);
    - BSDC-S → avviso del bias residuo (≤ 0.83 punti con segno nella 22).
    "validated" vuol dire soltanto: dentro gli intervalli strutturali testati e senza
    regole di cautela attive. Non è una garanzia sull'errore."""
    level, warns = 0, []
    names = ("validated", "caution", "unvalidated")
    if g.n > N_MAX_TESTED:
        level = 2
        warns.append(f"N = {g.n} > {N_MAX_TESTED}, the largest load tested (preregs 21-22; "
                     "N = 3200 was planned but dropped from the grid)")
    if g.n < N_MIN_TESTED:
        level = max(level, 1)
        warns.append(f"N = {g.n} < {N_MIN_TESTED}, the smallest load tested")
    for name, (lo, hi, _clo, chi) in STRUCT_RANGES.items():
        v = _range_value(g, name)
        if v > hi + 1e-12 or v < lo - 1e-12:
            level = 2
            side = f"> {hi}, the largest" if v > hi else f"< {lo}, the smallest"
            warns.append(f"{name.replace('_', ' ')} {v:.4f} {side} cell mean tested "
                         f"(preregs 21-22): outside the validated structure")
    if g.self_loops:
        level = 2
        warns.append("self-loops: never present in the tested subgraphs")
    if family == "MAP-B" and g.stats["pair_multirel_share"] > 0:
        level = max(level, 1)
        warns.append(
            f"MAP-B: {g.stats['pair_multirel_share']:.1%} of the facts share their unordered "
            "pair {s, o} with a fact of another relation (e.g. (s, r1, o) + (o, r2, s)); "
            "such facts share s and o, so in the MAP-B majority they cancel wherever "
            "ρ(r1) ≠ ρ(r2); the model treats them as independent and is optimistic "
            "(YAGO3-10 dense; tens of points on synthetic graphs). Use verify=True")
    if family == "MAP-B" and g.regime == "dense":
        level = max(level, 1)
        warns.append(f"MAP-B on a dense graph: the model failed on YAGO3-10 dense (prereg 22), "
                     f"optimistic by {-MAPB_DENSE_SIGNED} points on average and up to "
                     f"{OBSERVED_ERROR[('MAP-B', 'dense')][1]}; the a-priori D missed the "
                     "target in 4 of 12 cells")
    if family == "BSDC-S":
        warns.append(f"BSDC-S: residual bias of the exact-ties model, signed error up to "
                     f"+{BSDC_S_SIGNED_MAX} points per group (prereg 22)")
    return names[level], warns


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def _t975(df):
    k = max(x for x in _T975 if x <= df)
    return _T975[k] if df <= 30 else 1.96


def _fallback_bytes(family, dim, n, m, q):
    """Stima di picco (byte) se families.estimate_bytes non c'è: atomi, traccia e i
    temporanei principali della costruzione (non a blocchi) e di un blocco di query."""
    if family.startswith("BSDC"):
        length = bsdc_length(n)
        b = -(-dim // length)
        return 16 * m * b + 8 * b * length + 24 * n * b + 16 * min(q, 4096) * m * b
    if family == "HRR":                       # float32 + 2 rfft complex128 (M·(D/2+1)·16)
        return 4 * m * dim + 2 * m * (dim // 2 + 1) * 16 + 8 * m * dim + 16 * dim
    if family == "FHRR":                      # θ float64 + atomi e ρ complex64
        return 24 * m * dim + 16 * dim
    if family == "MAP-I":                     # atomi e ρ float64 + prodotti N × D
        return 16 * m * dim + 24 * n * dim
    return 5 * m * dim + 16 * n * dim         # MAP-B: int8 + float32, prodotti int64


def _peak_bytes(g, family, dim, n_queries):
    est = getattr(_families, "estimate_bytes", None)
    if est is not None:
        try:
            return int(est(family, dim, n_facts=g.n, n_symbols=g.m, n_queries=n_queries))
        except TypeError:
            pass
    return int(_fallback_bytes(family, dim, g.n, g.m, n_queries))


def _measure(g, family, dim, sample, seeds):
    """Accuratezza su `sample` per ciascuna memoria (un seme ciascuna, `seeds` è la lista
    dei semi): media, IC 95%, quantile 10% per memoria, accuratezze per seme."""
    accs = []
    for seed in seeds:
        mem = _families.make(family, dim, int(seed))
        mem.store(g.triples)
        ans = mem.query_many(sample)
        ok = sum(o in g.objects.get(q, ()) for q, (o, _c) in zip(sample, ans))
        accs.append(ok / len(sample))
    mean = float(np.mean(accs))
    if len(seeds) == 1:                             # Wilson sulle query
        n, z = len(sample), 1.96
        c = (mean + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        ci = (max(0.0, c - h), min(1.0, c + h))
    else:
        h = _t975(len(seeds) - 1) * float(np.std(accs, ddof=1)) / math.sqrt(len(seeds))
        ci = (max(0.0, mean - h), min(1.0, mean + h))
    return {"mean": mean, "ci": ci, "q10": float(np.quantile(accs, 0.10)), "accs": accs}


def _split(g, rng_seed, seeds, verify_queries):
    """Semi e query di selezione e di report, disgiunti, derivati da rng_seed."""
    seq_seeds, seq_q = np.random.SeedSequence([0x51_2E, rng_seed]).spawn(2)
    vals = np.random.default_rng(seq_seeds).choice(1 << 31, 2 * seeds, replace=False)
    sel_seeds = [int(x) for x in vals[:seeds]]
    rep_seeds = [int(x) for x in vals[seeds:]]
    qs = list(g.queries)
    if len(qs) > verify_queries:
        k = min(verify_queries, len(qs) // 2)
        perm = np.random.default_rng(seq_q).permutation(len(qs))
        sel_q = [qs[i] for i in sorted(perm[:k])]
        rep_q = [qs[i] for i in sorted(perm[k:2 * k])]
    else:
        sel_q = rep_q = qs
    return sel_seeds, rep_seeds, sel_q, rep_q


def _verify(g, family, d0, target, tol, top, sample, seeds, max_mem, skipped):
    """Selezione di D: (dim, misura al dim o None, accettato, misura a d0 o None).
    Accetta D se IC_inf ≥ target − tol; raddoppio fino a `top`, poi bisezione al passo.
    I D la cui memoria stimata supera max_mem non si costruiscono (aggiunti a skipped)."""
    step = _step(g, family)
    memo = {}

    def run(d):
        if d not in memo:
            if _peak_bytes(g, family, d, len(sample)) > max_mem:
                memo[d] = None
                skipped.append(d)
            else:
                memo[d] = _measure(g, family, d, sample, seeds)
        return memo[d]

    def accept(r):
        return r is not None and r["ci"][0] >= target - tol

    r0 = run(d0)
    if r0 is None:
        return d0, None, False, None
    if accept(r0):
        return d0, r0, True, r0
    lo, hi = d0, None
    while lo < top:
        d = min(2 * lo, top)
        r = run(d)
        if r is None:
            break
        if accept(r):
            hi = d
            break
        lo = d
    if hi is None:
        return lo, run(lo), False, r0
    while hi - lo > step:
        mid = (lo // step + hi // step) // 2 * step
        if accept(run(mid)):
            hi = mid
        else:
            lo = mid
    return hi, run(hi), True, r0


def _set_dim(g, rec, d, count_codebook):
    f = rec.family
    rec.dim = d
    rec.codebook_bits = explicit_codebook_bits(f, d, g.m, g.n) if count_codebook else 0
    rec.bits = d * bits_per_component(f, g.n) + rec.codebook_bits
    rec.kbytes = rec.bits / 8 / 1024


def _verify_rec(g, rec, target, tol, top, split, budget_bits, max_mem, count_codebook):
    f, d0 = rec.family, rec.dim
    sel_seeds, rep_seeds, sel_q, rep_q = split
    rec.tol = tol
    skipped = []
    d, rsel, ok, r0 = _verify(g, f, d0, target, tol, top, sel_q, sel_seeds, max_mem, skipped)
    if skipped:
        rec.warnings.append(
            f"verification: D = {', '.join(map(str, sorted(set(skipped))))} not built: "
            f"estimated peak memory over max_mem_bytes = {max_mem}")
    if rsel is None:
        rec.warnings.append("verification: nothing measured; not verified")
        return
    rec.selection_measured, rec.selection_ci = rsel["mean"], rsel["ci"]
    if d != d0:
        rec.adjusted = True
        _set_dim(g, rec, d, count_codebook)
        rec.predicted = _predict_graph(g, f, d)
        p0 = _predict_graph(g, f, d0)
        rec.warnings.append(
            f"verification: at D = {d0} the selection measure rejected the target "
            f"(predicted {p0:.4f}, measured {r0['mean']:.4f}, model optimistic by "
            f"{100 * (p0 - r0['mean']):.1f} points; 95% CI lower bound "
            f"{r0['ci'][0]:.4f} < target − tol = {target - tol:.4f}); D increased from "
            f"{d0} to {d}")
    if not ok:
        rec.warnings.append(
            f"verification: the CI lower bound stays below target − tol = {target - tol:.4f} "
            f"up to D = {d} (verification d_max {top}); not verified")
    rep = rsel if (rep_seeds == sel_seeds and rep_q == sel_q) else \
        _measure(g, f, d, rep_q, rep_seeds)
    rec.measured, rec.measured_ci, rec.per_memory_q10 = rep["mean"], rep["ci"], rep["q10"]
    rec.verified = ok and rep["mean"] >= target - tol
    if ok and not rec.verified:
        rec.warnings.append(
            f"verification: on fresh seeds/queries (report) the mean accuracy "
            f"{rep['mean']:.4f} < target − tol = {target - tol:.4f} (95% CI "
            f"{rep['ci'][0]:.4f}–{rep['ci'][1]:.4f}): the selected D does not replicate; "
            "not verified")
    if budget_bits is not None and rec.bits > budget_bits:
        rec.over_budget = True
        rec.warnings.append(f"verification: the corrected D needs {rec.bits} bits, over the "
                            f"budget of {budget_bits}")


def _check_target(target):
    if isinstance(target, bool) or not isinstance(target, (int, float, np.integer,
                                                          np.floating)) \
            or not 0.0 < target <= 1.0:
        raise ValueError(f"target must be an accuracy in (0, 1], got {target!r}")


def recommend(triples, target, families=FAMILIES, budget_bits=None, queries=None,
              d_max=D_MAX, verify=False, verify_queries=2000, seeds=VERIFY_SEEDS,
              rng_seed=0, tol=VERIFY_TOL, verify_d_max=None, max_mem_bytes=MAX_MEM_BYTES,
              count_codebook=False):
    """Per ogni famiglia il D minimo che raggiunge `target` (accuratezza assoluta media
    sulle query) SECONDO IL MODELLO, ordinate per bit crescenti; esclude le famiglie che non
    lo raggiungono entro d_max o che superano budget_bits. Se il target supera il tetto
    degli alias (nessuna dimensione lo raggiunge, in nessuna famiglia) solleva ValueError.
    Nessun risultato è una garanzia: la previsione è un modello, la verifica una misura
    finita (vedi il docstring del modulo).

    count_codebook=False: bits conta solo la traccia (gli atomi si rigenerano da seme e
    nomi); True aggiunge explicit_codebook_bits (atomi memorizzati esplicitamente; NON è
    families.Memory.codebook_bits(), che conta solo l'indice M·⌈log2 M⌉).

    verify=True: misura ogni raccomandazione su memorie vere (vedi il docstring del modulo:
    `seeds` semi di selezione + `seeds` di report, disgiunti; accetta se il limite
    inferiore dell'IC 95% ≥ target − tol). verify_d_max: il D massimo della correzione,
    default min(8·d0, d_max). max_mem_bytes: un D con memoria di picco stimata superiore
    non viene costruito (avviso). Una famiglia corretta oltre budget_bits resta nella lista
    con over_budget=True; una non verificata resta con verified=False e un avviso."""
    _check_target(target)
    families = tuple(families)
    if not families:
        raise ValueError("no families")
    for f in families:
        _check_family(f)
    if budget_bits is not None:
        budget_bits = _check_positive_int("budget_bits", budget_bits)
    if verify:
        verify_queries = _check_positive_int("verify_queries", verify_queries)
        seeds = _check_positive_int("seeds", seeds)
        max_mem_bytes = _check_positive_int("max_mem_bytes", max_mem_bytes)
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, (int, np.integer)) \
                or rng_seed < 0:
            raise ValueError(f"rng_seed must be a non-negative integer, got {rng_seed!r}")
        if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating))\
                or not 0.0 <= tol < 1.0:
            raise ValueError(f"tol must be in [0, 1), got {tol!r}")
        if verify_d_max is not None:
            verify_d_max = _check_positive_int("verify_d_max", verify_d_max)
    g = _graph(triples, queries)
    if target > g.ceiling:
        raise ValueError(
            f"target {target} is above the alias ceiling {g.ceiling:.4f}: queries with "
            "aliases (x with (x, r, s) stored, which encodes like (s, r, x)) get the same "
            "signal as the true object, so no family and no dimension can exceed the "
            "ceiling on average; lower the target or remove the aliased facts")
    out, split = [], None
    for f in families:
        d = min_dim(g, f, target, d_max=d_max)
        if d is None:
            continue
        rel, warns = _reliability(g, f)
        rec = Recommendation(
            family=f, dim=d, bits=0, kbytes=0.0, predicted=_predict_graph(g, f, d),
            observed_error=_observed_error(f, g.regime),
            reliability=rel, warnings=warns, dim_predicted=d)
        _set_dim(g, rec, d, count_codebook)
        if budget_bits is not None and rec.bits > budget_bits:
            continue
        if verify:
            if split is None:
                split = _split(g, int(rng_seed), seeds, verify_queries)
            step = _step(g, f)
            cap = min(8 * d, d_max) if verify_d_max is None else verify_d_max
            top = max(d, (cap // step) * step)
            _verify_rec(g, rec, target, tol, top, split, budget_bits, max_mem_bytes,
                        count_codebook)
        out.append(rec)
    out.sort(key=lambda r: (r.bits, FAMILIES.index(r.family)))
    return out
