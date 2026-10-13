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

Verifica (fase 3, `recommend(..., verify=True)`): per ogni raccomandazione costruisce la
memoria vera (`abm.families.make`) al dim previsto su `seeds` semi (0 … seeds − 1), interroga
le query reali (quelle passate, o tutte le coppie (s, r) presenti; al più `verify_queries`,
campionate senza reimmissione con `rng_seed`, lo stesso campione per tutti i dim e i semi)
e conta corretta una risposta che è uno degli oggetti veri di (s, r).
Criterio (due tempi). (1) Accettazione del dim del modello: il LIMITE SUPERIORE dell'IC
95% misurato ≥ target, cioè si corregge solo se la misura rifiuta il target (il modello è
ottimista oltre il rumore della misura). Con la sola media si correggerebbe il rumore: su
un grafo uniforme a N = 200 (200 query) la media di 3 semi oscilla di ±2 punti e tre
famiglie su sei venivano "corrette" senza che il modello sbagliasse. Con il limite
inferiore ancora peggio: D crescerebbe sempre di un margine pari alla larghezza dell'IC.
(2) Se rifiutato, il D corretto è il più piccolo (nella ricerca) con MEDIA misurata ≥
target, la stessa definizione di target di min_dim: raddoppio di D (multipli del passo)
fino al primo dim che passa o a d_max, poi bisezione fra l'ultimo fallito e il primo
riuscito. verified = True se il dim finale soddisfa il suo criterio; un d0 accettato
può avere measured < target (dentro l'IC): measured e measured_ci lo dicono.
IC: con seeds ≥ 2, media dei semi ± t(seeds − 1, 0.975)·sd/√seeds (i semi sono i cluster:
le query dello stesso seme condividono gli atomi); con seeds = 1, IC binomiale di Wilson
sulle query (che però ignora la correlazione fra query della stessa memoria).

API: predict, min_dim, recommend (→ list[Recommendation]), FAMILIES, ERROR_BANDS.
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

# ---- tabella di validazione -------------------------------------------------
# Errore |misura − previsione| in punti percentuali, a 1 hop, per famiglia e regime,
# sulle celle "calib" della 21 (FB15k-237, WN18RR; N 100, 400; target 0.50 e 0.85 del
# tetto; 10 seed) e sulle celle a 1 hop della 22 (CoDEx-M, YAGO3-10; N 100, 400, 1600;
# 20 seed), riunite. Fonte: results/vsa_families_prereg_results.json (rows, block=calib)
# e results/generalization_prereg_results.json (rows, hops=1), |meas − pred| per cella;
# esiti in coda a docs/preregistration/vsa_families.md e generalization.md.
# BSDC-S e HRR: solo la 22 (la 21 usava per BSDC-S il modello con i pareggi
# approssimati, distorto di +1.7–2.1 punti, ora sostituito; HRR non c'era).
# (mae, cella peggiore, celle, cella peggiore: provenienza)
ERROR_BANDS = {
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
# tempo, regola scritta nel documento: non è stato misurato). Quote (definite come negli
# harness: alias_share = quota di query con ≥ 1 alias; twin_share = quota di vettori
# distinti di peso 2), medie di cella sui seed:
N_MIN_TESTED, N_MAX_TESTED = 100, 1600
ALIAS_MAX_UNIFORM = 0.0085   # 22, YAGO3-10/CoDEx-M uniform (21: 0.0035)
TWIN_MAX_UNIFORM = 0.00126   # 21, uniform (22: 0.00013)
ALIAS_MAX_TESTED = 0.094     # 21, WN18RR dense
TWIN_MAX_TESTED = 0.144      # 21, WN18RR dense
MAPB_DENSE_SIGNED = -3.5     # 22, MAP-B YAGO3-10 dense: errore con segno medio (punti)
BSDC_S_SIGNED_MAX = 0.83     # 22, BSDC-S: |errore con segno| massimo per gruppo (punti)


@dataclass(frozen=True)
class ErrorBand:
    regime: str          # "uniform" | "dense"
    mae: float           # punti percentuali, MAE tipico misurato
    worst: float         # punti percentuali, cella peggiore misurata
    cells: int
    source: str


@dataclass
class Recommendation:
    family: str
    dim: int             # componenti (BSDC: B·L)
    bits: int
    kbytes: float
    predicted: float     # accuratezza media prevista sulle query
    error_band: ErrorBand
    reliability: str     # "validated" | "caution" | "unvalidated"
    warnings: list = field(default_factory=list)
    # verifica (recommend(..., verify=True)); senza verifica restano None/False
    measured: float = None       # accuratezza media misurata (media dei seed) al dim finale
    measured_ci: tuple = None    # IC 95% (a cluster per seed; binomiale se seeds = 1)
    verified: bool = False       # True se la misura al dim finale raggiunge il target
    dim_predicted: int = None    # il dim del modello, prima della verifica
    adjusted: bool = False       # True se la verifica ha cambiato dim
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
        return ("uniform" if self.alias_share <= ALIAS_MAX_UNIFORM
                and self.twin_share <= TWIN_MAX_UNIFORM else "dense")


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
    """MAP-B, BSDC-OR: 1. FHRR: 2 × float16. HRR: float32. MAP-I, BSDC-S: interi con N + 1
    valori, ceil(log2(N + 1)) bit (convenzione degli harness)."""
    if family in ("MAP-B", "BSDC-OR"):
        return 1
    if family in FLOAT_BITS:
        return FLOAT_BITS[family]
    return math.ceil(math.log2(n + 1))


def _reliability(g, family):
    """Regole di affidabilità (documentate nel modulo e nel report):
    - N fuori da [100, 1600] o quote di alias/gemelli oltre il massimo visto nei test
      21–22 → "unvalidated" (oltre 1600: 3200 era previsto ma tolto dalla griglia);
    - MAP-B su grafo denso → "caution": fallito su YAGO3-10 dense (ottimista di 3.5 punti
      in media, fino a 7.2; D a priori sotto il target in 4 celle su 12);
    - BSDC-S → avviso del bias residuo (≤ 0.83 punti con segno nella 22);
    - self-loop → "unvalidated" (nessun sottografo testato ne aveva)."""
    level, warns = 0, []
    names = ("validated", "caution", "unvalidated")
    if g.n > N_MAX_TESTED:
        level = 2
        warns.append(f"N = {g.n} > {N_MAX_TESTED}, the largest load tested (preregs 21-22; "
                     "N = 3200 was planned but dropped from the grid)")
    if g.n < N_MIN_TESTED:
        level = max(level, 1)
        warns.append(f"N = {g.n} < {N_MIN_TESTED}, the smallest load tested")
    if g.alias_share > ALIAS_MAX_TESTED:
        level = 2
        warns.append(f"alias share {g.alias_share:.3f} > {ALIAS_MAX_TESTED}, the largest tested")
    if g.twin_share > TWIN_MAX_TESTED:
        level = 2
        warns.append(f"twin share {g.twin_share:.3f} > {TWIN_MAX_TESTED}, the largest tested")
    if g.self_loops:
        level = 2
        warns.append("self-loops: never present in the tested subgraphs")
    if family == "MAP-B" and g.regime == "dense":
        level = max(level, 1)
        warns.append(f"MAP-B on a dense graph: the model failed on YAGO3-10 dense (prereg 22), "
                     f"optimistic by {-MAPB_DENSE_SIGNED} points on average and up to "
                     f"{ERROR_BANDS[('MAP-B', 'dense')][1]}; the a-priori D missed the target "
                     "in 4 of 12 cells")
    if family == "BSDC-S":
        warns.append(f"BSDC-S: residual bias of the exact-ties model, signed error up to "
                     f"+{BSDC_S_SIGNED_MAX} points per group (prereg 22)")
    return names[level], warns


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def _t975(df):
    k = max(x for x in _T975 if x <= df)
    return _T975[k] if df <= 30 else 1.96


def _measure(g, family, dim, sample, seeds):
    """(media, IC 95%) dell'accuratezza misurata sulle query `sample`, su `seeds` memorie."""
    accs, hits = [], 0
    for seed in range(seeds):
        mem = _families.make(family, dim, seed)
        mem.store(g.triples)
        ans = mem.query_many(sample)
        ok = sum(o in g.objects.get(q, ()) for q, (o, _c) in zip(sample, ans))
        hits += ok
        accs.append(ok / len(sample))
    mean = float(np.mean(accs))
    if seeds == 1:                                  # Wilson sulle query
        n, z = len(sample), 1.96
        c = (mean + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        ci = (max(0.0, c - h), min(1.0, c + h))
    else:
        h = _t975(seeds - 1) * float(np.std(accs, ddof=1)) / math.sqrt(seeds)
        ci = (max(0.0, mean - h), min(1.0, mean + h))
    return mean, ci


def _verify(g, family, d0, target, d_max, sample, seeds):
    """(dim, media, IC, raggiunto, media a d0) dopo la verifica: raddoppio poi bisezione."""
    step = _step(g, family)
    top = max(d0, (d_max // step) * step)
    memo = {}

    def run(d):
        if d not in memo:
            memo[d] = _measure(g, family, d, sample, seeds)
        return memo[d]

    if run(d0)[1][1] >= target:                 # la misura non rifiuta il target
        return d0, *run(d0), True, run(d0)[0]
    lo, hi = d0, None
    while lo < top:
        d = min(2 * lo, top)
        if run(d)[0] >= target:
            hi = d
            break
        lo = d
    if hi is None:
        return lo, *run(lo), False, run(d0)[0]
    while hi - lo > step:
        mid = (lo // step + hi // step) // 2 * step
        if run(mid)[0] >= target:
            hi = mid
        else:
            lo = mid
    return hi, *run(hi), True, run(d0)[0]


def _verify_rec(g, rec, target, d_max, sample, seeds, budget_bits):
    f, d0 = rec.family, rec.dim
    d, mean, ci, ok, m0 = _verify(g, f, d0, target, d_max, sample, seeds)
    rec.measured, rec.measured_ci, rec.verified = mean, ci, ok
    if d != d0:
        rec.adjusted = True
        rec.dim = d
        rec.bits = d * bits_per_component(f, g.n)
        rec.kbytes = rec.bits / 8 / 1024
        rec.predicted = _predict_graph(g, f, d)
        gap = 100 * (_predict_graph(g, f, d0) - m0)
        rec.warnings.append(
            f"verification: the model was optimistic by {gap:.1f} points at D = {d0} "
            f"(predicted {_predict_graph(g, f, d0):.4f}, measured {m0:.4f}); "
            f"D increased from {d0} to {d}")
    if not ok:
        rec.warnings.append(
            f"verification: measured {mean:.4f} < target {target} even at D = {d} "
            f"(d_max); not verified")
    if budget_bits is not None and rec.bits > budget_bits:
        rec.over_budget = True
        rec.warnings.append(f"verification: the corrected D needs {rec.bits} bits, over the "
                            f"budget of {budget_bits}")


def recommend(triples, target, families=FAMILIES, budget_bits=None, queries=None,
              d_max=D_MAX, verify=False, verify_queries=2000, seeds=3, rng_seed=0):
    """Per ogni famiglia il D minimo che raggiunge `target` (accuratezza assoluta media
    sulle query), ordinate per bit crescenti; esclude le famiglie che non lo raggiungono
    entro d_max o che superano budget_bits. Se il target supera il tetto degli alias
    (nessuna dimensione lo raggiunge, in nessuna famiglia) solleva ValueError.

    verify=True: misura ogni raccomandazione su memorie vere e, se la media misurata è
    sotto il target, aumenta dim (vedi il docstring del modulo). Una famiglia corretta
    oltre budget_bits resta nella lista con over_budget=True e un avviso; una che non
    raggiunge il target nemmeno a d_max resta con verified=False e un avviso."""
    if isinstance(target, bool) or not isinstance(target, (int, float, np.integer,
                                                          np.floating)) \
            or not 0.0 < target <= 1.0:
        raise ValueError(f"target must be an accuracy in (0, 1], got {target!r}")
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
    g = _graph(triples, queries)
    if target > g.ceiling:
        raise ValueError(
            f"target {target} is above the alias ceiling {g.ceiling:.4f}: queries with "
            "aliases (x with (x, r, s) stored, which encodes like (s, r, x)) get the same "
            "signal as the true object, so no family and no dimension can exceed the "
            "ceiling on average; lower the target or remove the aliased facts")
    out, sample = [], None
    for f in families:
        d = min_dim(g, f, target, d_max=d_max)
        if d is None:
            continue
        bits = d * bits_per_component(f, g.n)
        if budget_bits is not None and bits > budget_bits:
            continue
        mae, worst, cells, src = ERROR_BANDS[(f, g.regime)]
        rel, warns = _reliability(g, f)
        rec = Recommendation(
            family=f, dim=d, bits=bits, kbytes=bits / 8 / 1024,
            predicted=_predict_graph(g, f, d),
            error_band=ErrorBand(g.regime, mae, worst, cells, src),
            reliability=rel, warnings=warns, dim_predicted=d)
        if verify:
            if sample is None:
                qs = list(g.queries)
                if len(qs) > verify_queries:
                    pick = np.random.default_rng(rng_seed).choice(len(qs), verify_queries,
                                                                  replace=False)
                    qs = [qs[i] for i in sorted(pick)]
                sample = qs
            _verify_rec(g, rec, target, d_max, sample, seeds, budget_bits)
        out.append(rec)
    out.sort(key=lambda r: (r.bits, FAMILIES.index(r.family)))
    return out
