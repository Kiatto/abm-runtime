"""generalization_prereg.py — preregistrazione 22: la stima a priori del cleanup generalizza?

docs/preregistration/generalization.md. Riusa, importandolo e senza modificarlo, il codice
della preregistrazione 21 (examples/vsa_families_prereg.py). Nuovo rispetto alla 21:

- BSDC-S con il rumore condizionato alla traccia realizzata (diagnosi del bias della 21,
  passo 1, `--diagnose`, solo sintetico);
- HRR: vettori gaussiani, binding per convoluzione circolare (FFT), unbinding per
  involuzione; varianza del rumore con le molteplicità degli atomi (passo 2, `--verify`);
- catene a 2 e 3 hop, cleanup a ogni hop come `Memory.chain` (passo 3, `--verify`);
- dati nuovi: CoDEx-M (train), carichi fino a N = 3200, 20 seed.

Uso, dalla root:
    python examples/generalization_prereg.py --diagnose   # passo 1, sintetico
    python examples/generalization_prereg.py --verify     # passi 2-3, sintetico
    python examples/generalization_prereg.py --smoke      # impianto intero, grafo sintetico
    python examples/generalization_prereg.py --predict    # CoDEx-M: D e previsioni, nessuna memoria
    python examples/generalization_prereg.py              # il run preregistrato (NON eseguito qui)
"""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import vsa_families_prereg as v21  # noqa: E402  (aggiunge anche reference/ al path)
import abm  # noqa: E402
import exact  # noqa: E402

DATA = ROOT / "data" / "external" / "codex" / "codex-m_train.txt"
DATA_SHA = "d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556"
OUT = ROOT / "results" / "generalization_prereg_results.json"
PRED_OUT = ROOT / "results" / "generalization_prereg_predictions.json"
DIAG_SEEDS = 16
VERIFY_SEEDS = 16
DIAG_OUT = ROOT / "results" / "generalization_diagnosis.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
SCHEMES = ("uniform", "dense")
LOADS = (100, 400, 1600, 3200)
TARGETS = (0.50, 0.85)                 # frazione del tetto degli alias (regola della 21)
FAMILIES = ("MAP-B", "MAP-I", "FHRR", "HRR", "BSDC-OR", "BSDC-S")
HOPS = (1, 2, 3)
SEEDS = 20
FIT_SEEDS = 10                         # D scelto sui seed 0-9, misurato anche su 10-19
Q_MAX = 200                            # query a 1 hop per seed (come la 21)
Q_CHAIN = 100                          # catene per seed e per h ∈ {2, 3}
CHAIN_MIN = 10                         # un seed entra in una cella catena se ha ≥ 10 catene
CHAIN_SEEDS = 10                       # e la cella esiste se questi seed sono ≥ 10
STEP = 64
D_MAX = 1 << 18
HRR_BITS = 32                          # float32
BSDC_S_DRAWS = 400                     # tracce estratte dal modello (BSDC-S condizionato)
MC3_TRIALS = 20000                     # exact.chain_accuracy_mc per MAP-B a 3 hop
T_CRIT = 2.861                         # t di Student, 19 gdl, bilaterale 0.01
T_95, T_80 = 2.093, 0.861              # IC al 95 % e potenza 0.8, 19 gdl
SEED_BASE = 22_000_022
# -----------------------------------------------------------------------------

v21.Q_MAX = Q_MAX                      # Cell della 21 legge la costante di modulo


def load():
    if not DATA.exists():
        raise SystemExit(f"{DATA} mancante: scaricalo come in examples/replicate.py (DATASETS)")
    got = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if got != DATA_SHA:
        raise SystemExit(f"{DATA.name}: sha256 {got}, atteso {DATA_SHA}")
    return [tuple(line.rstrip("\n").split("\t")) for line in DATA.open()]


# =============================================================================
# Passo 1 — BSDC-S: rumore condizionato alla traccia realizzata
# =============================================================================
#
# Il modello della 21 tratta i punteggi dei candidati nulli come i.i.d. dalla pmf
# MARGINALE del rumore. Ma dato la traccia (gli istogrammi dei B blocchi), il punteggio
# di un nullo è Σ_b hist_b[U_b] con U_b uniformi e indipendenti: i nulli sono i.i.d.
# CONDIZIONATAMENTE alla traccia, che però è la stessa per tutti. Quindi
#     P(vince) = E_traccia[ Π F_traccia(x) ... ]  ≥  Π E_traccia[F](x)   (Jensen)
# e il modello marginale è pessimista. Qui la media sulla traccia si fa estraendo
# tracce dal modello generativo (posizioni uniformi dei vettori con i loro pesi), non
# dai dati: nessun parametro tarato. Pareggi esatti (divisione uniforme a caso).


def _weights_list(wcounts):
    return np.repeat([w for w, _c in wcounts], [c for _w, c in wcounts]).astype(np.int64)


def _tie_win(n_null, f_s, cdf_s, t_good, t_all):
    """P(un oggetto vero vince) dato che il massimo dei candidati a segnale vale S, con
    t_good veri e t_all candidati a segnale pari a S; n_null nulli i.i.d. con
    P(= S) = f_s, P(≤ S) = cdf_s; pareggi divisi uniformemente."""
    if t_good == 0 or cdf_s <= 0:
        return 0.0
    if n_null == 0:
        return t_good / t_all
    q = min(f_s / cdf_s, 1.0)
    j = np.arange(n_null + 1)
    pj = v21._binom(n_null, q) if 0 < q < 1 else (j == (n_null if q >= 1 else 0)).astype(float)
    return float(cdf_s ** n_null * np.sum(pj * t_good / (t_all + j)))


_BS = {}


def bsdc_s_cond_acc(gw, aw, wcounts, n_null, blocks, length, draws=BSDC_S_DRAWS):
    """BSDC-S, rumore condizionato alla traccia (vedi sopra), per Monte Carlo sul modello."""
    key = (gw, aw, wcounts, n_null, blocks, length, draws)
    if key in _BS:
        return _BS[key]
    if not gw:
        return 0.0
    w_all = _weights_list(wcounts)
    n_vec = len(w_all)
    # i candidati a segnale sono vettori della traccia con il loro peso
    used, sig_idx = set(), []
    for w in gw + aw:
        i = next(k for k in range(n_vec) if w_all[k] == w and k not in used)
        used.add(i)
        sig_idx.append(i)
    good = np.array([True] * len(gw) + [False] * len(aw))
    h = hashlib.sha256(repr(key).encode()).digest()
    rng = np.random.default_rng([SEED_BASE, int.from_bytes(h[:8], "little")])
    off = (np.arange(blocks) * length)[None, :]
    acc = 0.0
    for _ in range(draws):
        pos = rng.integers(length, size=(n_vec, blocks))
        hist = np.bincount((pos + off).ravel(), weights=np.repeat(w_all, blocks),
                           minlength=blocks * length).astype(np.int64).reshape(blocks, length)
        sig = hist[np.arange(blocks)[None, :], pos[sig_idx]].sum(axis=1)
        top = int(sig.max())
        at = sig == top
        pmf = np.array([1.0])
        for b in range(blocks):
            pmf = np.convolve(pmf, np.bincount(hist[b]) / length)
        cdf = np.cumsum(pmf)
        f_s = pmf[top] if top < len(pmf) else 0.0
        c_s = cdf[min(top, len(cdf) - 1)]
        acc += _tie_win(n_null, f_s, c_s, int(np.sum(at & good)), int(np.sum(at)))
    _BS[key] = acc / draws
    return _BS[key]


def bsdc_s_exact(gw, aw, wcounts, n_null, blocks, length):
    """BSDC-S: le ipotesi della 21 (pmf marginale esatta del rumore, punteggi indipendenti)
    con i pareggi ESATTI (divisione uniforme a caso, come la misura) invece della quota
    proporzionale al rischio. Per ogni valore x del massimo: programmazione dinamica sul
    numero di oggetti veri (tg) e alias (ta) pari a x, poi nulli: tutti ≤ x, J ~ Bin(n,
    f(x)/F(x)) pari a x, e quota tg/(tg + ta + J)."""
    if not gw:
        return 0.0
    noise = v21._bsdc_s_noise(wcounts, blocks, length)
    top = len(noise) + blocks * (max(gw + aw) + 1)
    null = np.zeros(top)
    null[: len(noise)] = noise
    sig = []
    for w in gw + aw:
        others = tuple((v, c - (v == w)) for v, c in wcounts if c - (v == w) > 0)
        nz = v21._bsdc_s_noise(others, blocks, length)
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
        pj = v21._binom(n_null, q) if 0 < q < 1 else (j == (n_null if q >= 1 else 0)) * 1.0
        base = F[x] ** n_null
        for tg in range(1, g + 1):
            for ta in range(k - g + 1):
                if poly[tg, ta] > 0:
                    acc += poly[tg, ta] * base * float(np.sum(pj * tg / (tg + ta + j)))
    return float(min(acc, 1.0))


def bsdc_s_iid_mc(gw, aw, wcounts, n_null, blocks, length, trials, rng):
    """Diagnosi: le ipotesi del modello della 21 (rumore marginale, candidati
    indipendenti) campionate esattamente, con pareggi esatti. Isola l'approssimazione
    dei pareggi dall'ipotesi d'indipendenza."""
    noise = v21._bsdc_s_noise(wcounts, blocks, length)
    win = 0
    for _ in range(trials):
        nulls = rng.choice(len(noise), size=n_null, p=noise) if n_null else np.array([], int)
        sc = []
        for w in gw + aw:
            others = tuple((v, c - (v == w)) for v, c in wcounts if c - (v == w) > 0)
            nz = v21._bsdc_s_noise(others, blocks, length)
            sc.append(w * blocks + rng.choice(len(nz), p=nz))
        allsc = np.concatenate((sc, nulls))
        best = np.flatnonzero(allsc == allsc.max())
        win += int(best[rng.integers(len(best))] < len(gw))
    return win / trials


# =============================================================================
# Passo 2 — HRR
# =============================================================================
#
# Atomi a ~ N(0, 1/D)^D, reali. Fatto f = s ⊛ ρ(r) ⊛ o (⊛ convoluzione circolare,
# ρ = shift ciclico, che non cambia |DFT|). Traccia T = Σ_j w_j f_j. Unbinding per
# involuzione: y = k* ⊛ T, k = s ⊛ ρ(r), k* l'involuzione (DFT coniugata). Punteggio del
# candidato c: ⟨y, c⟩ = (1/D) Σ_ν conj(K_ν) T_ν conj(C_ν). Con E|A_ν|² = 1 e le
# componenti di Fourier complesse gaussiane circolari (a parte ν = 0, D/2), per ogni
# vettore j il termine Z_ν = conj(S R C)_ν · (A1 A2 A3)_ν ha:
#   - se il multinsieme {s, r, c} = {atomi di f_j} (segnale: oggetto vero o alias)
#     Z = |S|²|R|²|C|² reale, E Z = 1, E Z² = 8: contributo w_j alla media e
#     2·7·w_j²/D alla varianza (la somma su ν è fatta di D/2 coppie coniugate);
#   - altrimenti E Z = 0, E Z² = 0 e E|Z|² = Π_a k_a! (k_a = molteplicità dell'atomo a
#     in {s, r, c} ∪ {atomi di f_j}; E|X|^{2k} = k!): contributo w_j²·Π k_a!/D.
# I termini di vettori diversi sono scorrelati. Previsione: punteggi gaussiani
# indipendenti con queste medie e varianze, argmax (come FKS, ma con le molteplicità:
# per MAP-I ed FHRR |X| = 1 e le molteplicità non contano). Approssimazioni: si ignora
# la fluttuazione comune |K_ν|² condivisa da tutti i candidati di una query (ordine
# √(3/D) relativo sulla varianza) e le frequenze reali ν = 0, D/2.

_FACT = math.factorial
_FTAB = np.array([math.factorial(k) for k in range(8)], float)


def hrr_query_moments(hc, s, r, cand_good, cand_alias):
    """(medie e varianze ×D dei candidati a segnale, varianze ×D dei nulli) per la query
    (s, r). hc: struttura precalcolata del sottografo (hrr_struct)."""
    atoms, w2, w = hc["atoms"], hc["w2"], hc["w"]
    key = (hc["id"][s], hc["id"][r])
    nk = (atoms == key[0]).sum(axis=1) + (atoms == key[1]).sum(axis=1)   # in-key count/fact
    # m0_j = Π_a (n_fact(a) + n_key(a))! senza candidato
    m0 = np.ones(len(w))
    for col in range(3):
        a = atoms[:, col]
        first = np.ones(len(w), bool)
        for c2 in range(col):
            first &= atoms[:, c2] != a
        tot = (atoms == a[:, None]).sum(axis=1) + (a == key[0]) + (a == key[1])
        m0 *= np.where(first, _FTAB[tot], 1.0)
    # atomi della chiave che non compaiono nel fatto
    for k in set(key):
        kk = (key[0] == k) + (key[1] == k)
        absent = (atoms == k).sum(axis=1) == 0
        m0 *= np.where(absent, _FACT(kk), 1.0)
    base = float(np.sum(w2 * m0))
    # var(c) = Σ_j w_j² m0_j (n_fact_j(c) + n_key(c) + 1)  [fattore (k+1)!/k!]
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
    sig_ids = {x[3] for x in sig}
    mask = np.ones(hc["m"], bool)
    mask[list(sig_ids)] = False
    return sig, var[mask]


def hrr_struct(cell):
    sym = cell.symbols
    idx = {x: i for i, x in enumerate(sym)}
    keys = list(cell.weights)
    atoms, w, fact_of = [], [], {}
    for j, k in enumerate(keys):
        pair, r = k
        a, b = sorted(pair)
        atoms.append((idx[a], idx[r], idx[b]))
        w.append(cell.weights[k])
        fact_of[(pair, r)] = j
    w = np.asarray(w, float)
    return {"atoms": np.asarray(atoms), "w": w, "w2": w * w, "id": idx, "m": len(sym),
            "fact_of": fact_of}


_ZT = np.linspace(-40.0, 40.0, 160001)
_LT = v21._log_cdf(_ZT)


def _log_cdf(z):
    """log Φ per interpolazione su una tabella (passo 5e-4) di v21._log_cdf."""
    return np.interp(z, _ZT, _LT)


def gaussian_max_het(cands, null_var, dim):
    """P(il massimo è un candidato 'ok'); cands = [(media, var×D, ok)], nulli N(0, var×D / D)
    indipendenti, varianze eterogenee (raggruppate)."""
    sd_n = np.sqrt(np.asarray(null_var) / dim)
    vals, counts = np.unique(np.exp(np.round(np.log(sd_n), 3)), return_counts=True)  # rel. 1e-3
    mus = np.array([c[0] for c in cands])
    sds = np.array([math.sqrt(max(c[1], 1e-12) / dim) for c in cands])
    lo = min(-9 * (vals.max() if len(vals) else 1), float(np.min(mus - 9 * sds)))
    hi = max(float(np.max(mus + 9 * sds)), 9 * (vals.max() if len(vals) else 0))
    x = np.linspace(lo, hi, 4001)
    dx = x[1] - x[0]
    tot = np.zeros_like(x)
    for v, n in zip(vals, counts):
        tot += n * _log_cdf(x / v)
    logc = [_log_cdf((x - m) / s) for m, s in zip(mus, sds)]
    tot += np.sum(logc, axis=0)
    acc = 0.0
    for i, c in enumerate(cands):
        if c[2]:
            z = (x - mus[i]) / sds[i]
            dens = np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sds[i])
            acc += float(np.sum(dens * np.exp(tot - logc[i])) * dx)
    return min(acc, 1.0)


def hrr_acc(cell, hc, s, r, dim):
    mc = hc.setdefault("_mom", {})
    if (s, r) not in mc:
        good = sorted(cell.objects.get((s, r), ()))
        bad = sorted(cell.into.get((s, r), set()) - set(good))
        if not good:
            mc[(s, r)] = None
        else:
            mc[(s, r)] = hrr_query_moments(hc, s, r, good, bad)
    if mc[(s, r)] is None:
        return 0.0
    sig, nv = mc[(s, r)]
    return gaussian_max_het([(m, v, ok) for m, v, ok, _ in sig], nv, dim)


# =============================================================================
# Previsioni per query (1 hop) per tutte le famiglie
# =============================================================================

def sig_of(cell, s, r):
    good = cell.objects.get((s, r), set())
    bad = cell.into.get((s, r), set()) - good
    gw = tuple(sorted(cell.weights[exact.fact_key(s, r, o)] for o in good))
    aw = tuple(sorted(cell.weights[exact.fact_key(x, r, s)] for x in bad))
    return gw, aw


def _like(cell):
    return (cell.m, cell.s2, cell.n_vec,
            tuple(sorted(Counter(cell.weights.values()).items())), len(cell.triples))


def predict_queries(cell, family, dim, queries, naive=False):
    """Previsione per query (s, r) a 1 hop. naive: comparatore FKS ingenuo della 21
    (stessa per tutte le query)."""
    n = len(cell.triples)
    if naive:
        if family == "HRR":
            # FKS ingenuo per HRR: la gaussiana di MAP-I (v = 1), nessuna molteplicità
            return [v21._sig_acc((cell.m, float(n), n, ((1, n),), n), "MAP-I", dim, (1,), ())] \
                * len(queries)
        if family == "BSDC-S":
            return [_bsdc_s_pred((cell.m, float(n), n, ((1, n),), n), dim, (1,), ())] * len(queries)
        return [v21.predict_naive(cell, family, dim)] * len(queries)
    if family == "MAP-B":
        # exact.predict_queries dipende dalla query solo per la firma (pesi dei veri e
        # degli alias): una query rappresentante per firma (verificato nello smoke)
        rep = {}
        for q in queries:
            rep.setdefault(sig_of(cell, *q), q)
        cache = cell.__dict__.setdefault("_pb", {})
        todo = [q for sg, q in rep.items() if (dim, sg) not in cache]
        if todo:
            for q, v in zip(todo, exact.predict_queries(cell.triples, dim, todo)):
                cache[(dim, sig_of(cell, *q))] = v
        return [cache[(dim, sig_of(cell, *q))] for q in queries]
    if family == "HRR":
        hc = cell.__dict__.setdefault("_hrr", hrr_struct(cell))
        return [hrr_acc(cell, hc, s, r, dim) for s, r in queries]
    like = _like(cell)
    cache, out = cell.__dict__.setdefault("_pc", {}), []
    for q in queries:
        sig = sig_of(cell, *q)
        k = (family, dim, sig)
        if k not in cache:
            if not sig[0]:
                cache[k] = 0.0
            elif family == "BSDC-S":
                cache[k] = _bsdc_s_pred(like, dim, *sig)
            else:
                cache[k] = v21._sig_acc(like, family, dim, *sig)
        out.append(cache[k])
    return out


def _bsdc_s_pred(like, dim, gw, aw):
    m, _s2, _nv, wcounts, n = like
    length = v21.bsdc_length(n)
    return bsdc_s_exact(gw, aw, wcounts, m - len(gw) - len(aw), dim // length, length)


# =============================================================================
# Passo 3 — catene
# =============================================================================
#
# Protocollo di Memory.chain: a ogni hop query (nodo, r_i) e cleanup sull'intero
# codebook; il nodo restituito è la chiave dell'hop dopo. Catene campionate dal
# sottografo: percorsi s0 -r1-> e1 -r2-> ... con ogni (e_i, r_{i+1}) a un solo oggetto
# vero (percorso funzionale) e nodi distinti. Corretta = nodo finale = e_h.
# Previsione: Π_i p(e_{i-1}, r_i) (hop indipendenti) per MAP-I, FHRR, HRR, BSDC-OR,
# BSDC-S; per MAP-B la stessa per il fattore di dipendenza fra hop della reference
#   dep_2 = both/single²   (exact.two_hop_joint_fast(N, D, M))
#   dep_3 = chain_accuracy_mc(N, D, M, 3)/single³
# con N = numero di vettori distinti. Perché l'indipendenza per le altre: il solo termine
# di rumore condiviso fra l'hop i e l'hop i+1 è ⟨f_i, f_{i+1}⟩ (il fatto dell'altro hop
# visto come rumore), una covarianza w_i w_{i+1}/D su varianze S2/D: correlazione
# ≈ 1/S2 ≤ 1 % a N ≥ 100 (verificato sul sintetico, `--verify`). In MAP-B il voto di
# maggioranza crea invece la covarianza negativa per bit -ρ² della reference.
# Recupero per caso (hop sbagliato, nodo finale giusto) ignorato: probabilità ≲ 1/M.


def sample_chains(cell, h, rng, q=Q_CHAIN):
    """Fino a q percorsi funzionali distinti di h hop, per passeggiate a caso."""
    out_rel = defaultdict(list)
    for (s, r), objs in cell.objects.items():
        if len(objs) == 1:
            out_rel[s].append((r, next(iter(objs))))
    starts = sorted(out_rel)
    for s in starts:
        out_rel[s].sort()
    found, seen = [], set()
    if not starts:
        return found
    for _ in range(40 * q):
        node = starts[int(rng.integers(len(starts)))]
        path, nodes = [], [node]
        for _i in range(h):
            opts = [x for x in out_rel.get(node, []) if x[1] not in nodes]
            if not opts:
                break
            r, o = opts[int(rng.integers(len(opts)))]
            path.append(r)
            nodes.append(o)
            node = o
        if len(path) == h:
            k = (nodes[0], tuple(path))
            if k not in seen:
                seen.add(k)
                found.append((tuple(nodes), tuple(path)))
                if len(found) == q:
                    break
    return found


_DEP = {}


def mapb_dep(n_vec, dim, m, h):
    if h == 1:
        return 1.0
    key = (n_vec, dim, m, h)
    if key not in _DEP:
        if h == 2:
            single, both, sq = exact.two_hop_joint_fast(n_vec, dim, m)
            _DEP[key] = both / sq
        else:
            single = float(exact.cleanup_accuracy(n_vec, dim, m))
            mc = exact.chain_accuracy_mc(n_vec, dim, m, h, trials=MC3_TRIALS,
                                         seed=SEED_BASE % (2 ** 31))
            _DEP[key] = mc / single ** h
    return _DEP[key]


def predict_chains(cell, family, dim, chains, naive=False, dep=None):
    """dep: fattore di dipendenza MAP-B già calcolato (nelle celle preregistrate, uno per
    cella con N e M medi sui seed); None = calcolato su questo sottografo."""
    if not chains:
        return []
    hop_q = sorted({(nodes[i], path[i]) for nodes, path in chains for i in range(len(path))})
    p = dict(zip(hop_q, predict_queries(cell, family, dim, hop_q, naive=naive)))
    h = len(chains[0][1])
    if family != "MAP-B" or naive:
        dep = 1.0
    elif dep is None:
        dep = mapb_dep(cell.n_vec, dim, cell.m, h)
    return [dep * float(np.prod([p[(nodes[i], path[i])] for i in range(h)]))
            for nodes, path in chains]


# =============================================================================
# Misure (memorie vere; batch di query per hop)
# =============================================================================

CACHE_BYTES = 768 << 20                # sopra, gli atomi si rigenerano a blocchi


class Engine:
    """Memoria di una famiglia su un sottografo; answer(s_idx, r_idx) -> indici.

    Gli atomi densi sono generati per simbolo (MAP-B: abm.random_hv del nome; gli altri:
    un generatore per simbolo derivato da rng), così il codebook si può rigenerare a
    blocchi senza tenerlo in memoria quando M·D è grande."""

    def __init__(self, cell, family, dim, rng):
        self.f, self.dim, self.rng = family, dim, rng
        self.sym = cell.symbols
        self.idx = {x: i for i, x in enumerate(self.sym)}
        self.m = len(self.sym)
        self.base = int(rng.integers(2 ** 62))
        n = len(cell.triples)
        T = [(self.idx[s], self.idx[r], self.idx[o]) for s, r, o in cell.triples]
        if family in ("BSDC-OR", "BSDC-S"):
            L = v21.bsdc_length(n)
            B = dim // L
            self.L, self.B = L, B
            self.A = rng.integers(L, size=(self.m, B))
            self.rho = np.roll(self.A, 1, axis=1)
            mem = np.zeros((B, L), np.int64)
            ar = np.arange(B)
            for s, r, o in T:
                np.add.at(mem, (ar, (self.A[s] + self.rho[r] + self.A[o]) % L), 1)
            self.mem = (mem > 0).astype(np.int64) if family == "BSDC-OR" else mem
            return
        width = {"MAP-B": 1, "MAP-I": 1, "FHRR": 8, "HRR": 4}[family]
        self.full = None
        if self.m * dim * width <= CACHE_BYTES:
            self.full = self._gen(np.arange(self.m))
        if family == "HRR":
            tr = np.zeros(dim // 2 + 1, np.complex128)
            for s, r, o in T:
                a = self.rows([s, r, o]).astype(np.float64)
                a[1] = np.roll(a[1], 1)
                F = np.fft.rfft(a, axis=1)
                tr += F[0] * F[1] * F[2]
            self.trace = np.fft.irfft(tr, n=dim).astype(np.float32)   # traccia salvata, float32
            self.Ft = np.fft.rfft(self.trace.astype(np.float64))
            return
        acc = np.zeros(dim, np.complex128 if family == "FHRR" else np.int64)
        for s, r, o in T:
            a = self.rows([s, r, o])
            if family == "FHRR":
                acc += a[0].astype(np.complex128) * np.roll(a[1], 1) * a[2]
            else:
                acc += a[0].astype(np.int64) * np.roll(a[1], 1) * a[2]
        if family == "MAP-B":
            if n % 2 == 0:
                acc = acc * 2 + abm.random_hv("__tie__", dim)
            self.trace = np.where(acc >= 0, 1, -1).astype(np.float32)
        elif family == "MAP-I":
            self.trace = acc.astype(np.float64)
        else:
            self.trace = (acc.real.astype(np.float16).astype(np.float32)
                          + 1j * acc.imag.astype(np.float16).astype(np.float32))

    def _gen(self, ix):
        d = self.dim
        if self.f == "MAP-B":
            return np.stack([abm.random_hv(self.sym[i], d) for i in ix]).astype(np.int8)
        out = []
        for i in ix:
            g = np.random.default_rng([self.base, int(i)])
            if self.f == "MAP-I":
                out.append((2 * g.integers(2, size=d) - 1).astype(np.int8))
            elif self.f == "FHRR":
                out.append(np.exp(1j * g.uniform(0, 2 * np.pi, size=d)).astype(np.complex64))
            else:
                out.append((g.standard_normal(d) / math.sqrt(d)).astype(np.float32))
        return np.stack(out)

    def rows(self, ix):
        ix = np.asarray(ix)
        return self.full[ix] if self.full is not None else self._gen(ix)

    def _argmax(self, score):
        if self.f == "MAP-B":
            return np.argmax(score, axis=1)          # primo a distanza minima (ItemMemory)
        out = np.empty(len(score), int)
        for i, row in enumerate(score):
            best = np.flatnonzero(row == row.max())
            out[i] = best[self.rng.integers(len(best))]
        return out

    def _keys(self, ss, rr):
        """Vettori da confrontare col codebook, per un blocco di query."""
        S, R = self.rows(ss), np.roll(self.rows(rr), 1, axis=1)
        if self.f == "HRR":
            K = np.fft.rfft(S.astype(np.float64), axis=1) * np.fft.rfft(R.astype(np.float64), axis=1)
            return np.fft.irfft(np.conj(K) * self.Ft[None, :], n=self.dim).astype(np.float32)
        if self.f == "FHRR":
            return (self.trace[None, :] * np.conj(S) * np.conj(R)).astype(np.complex64)
        return (self.trace[None, :] * S * R).astype(np.float32)

    def answer(self, s, r, chunk=256):
        s, r = np.asarray(s), np.asarray(r)
        res = []
        for a in range(0, len(s), chunk):
            ss, rr = s[a:a + chunk], r[a:a + chunk]
            if self.f in ("BSDC-OR", "BSDC-S"):
                ar = np.arange(self.B)
                sc = np.stack([self.mem[ar[None, :], (self.A + self.A[x] + self.rho[y]) % self.L]
                               .sum(axis=1) for x, y in zip(ss, rr)])
            else:
                Y = self._keys(ss, rr)
                step = max(1, (256 << 20) // (self.dim * 8))
                parts = []
                for c0 in range(0, self.m, step):
                    C = self.rows(np.arange(c0, min(self.m, c0 + step)))
                    if self.f == "FHRR":
                        parts.append((Y @ np.conj(C).T).real)
                    else:
                        parts.append(Y @ C.T.astype(np.float32))
                sc = np.concatenate(parts, axis=1)
            res.append(self._argmax(sc))
        return np.concatenate(res) if res else np.array([], int)


def measure(cell, family, dim, rng, queries, chains_by_h):
    """Accuratezze per query (1 hop) e per catena (h = 2, 3) sulla stessa memoria."""
    eng = Engine(cell, family, dim, rng)
    ix = eng.idx
    out = {}
    got = eng.answer([ix[s] for s, _ in queries], [ix[r] for _, r in queries])
    out[1] = [eng.sym[g] in cell.objects[q] for g, q in zip(got, queries)]
    for h, chains in chains_by_h.items():
        if not chains:
            out[h] = []
            continue
        node = np.array([ix[nodes[0]] for nodes, _ in chains])
        for i in range(h):
            node = eng.answer(node, [ix[path[i]] for _, path in chains])
        out[h] = [eng.sym[g] == nodes[-1] for g, (nodes, _) in zip(node, chains)]
    return out


# =============================================================================
# Diagnosi (passo 1) e verifiche sintetiche (passi 2-3)
# =============================================================================

def _syn_cells(n, scheme, seeds, base, n_ent=300, n_rel=6, n_facts=3000, gseed=0):
    g = v21.Graph(v21.synthetic(n_ent=n_ent, n_rel=n_rel, n_facts=n_facts, seed=gseed))
    out = []
    for k in range(seeds):
        a, b = np.random.SeedSequence([base, n, len(scheme), k]).spawn(2)
        r = np.random.default_rng(a)
        out.append(v21.Cell(v21.sample(g, n, scheme, r), r, b))
    return out


def diagnose():
    """Passo 1. Sintetico soltanto (seed 22_100_xxx). Per ogni impianto: misura, modello
    della 21, ipotesi della 21 campionate con pareggi esatti (MC i.i.d.), modello
    condizionato alla traccia."""
    rows = []
    base = SEED_BASE + 100
    for scheme in ("uniform", "dense"):
        for n, mult in ((100, (8, 12, 16)), (400, (12, 18))):
            cells = _syn_cells(n, scheme, DIAG_SEEDS, base)
            L = v21.bsdc_length(n)
            for b in mult:
                dim = L * b
                meas, old, iid, cond, ex = [], [], [], [], []
                for k, c in enumerate(cells):
                    rng = np.random.default_rng([base, n, b, k])
                    accs = [np.mean(measure(c, "BSDC-S", dim, rng, c.queries, {})[1])
                            for _ in range(5)]
                    meas.append(float(np.mean(accs)))
                    old.append(float(np.mean(v21.predict_cell(c, "BSDC-S", dim))))
                    cond.append(float(np.mean(predict_queries(c, "BSDC-S", dim, c.queries))))
                    like = _like(c)
                    sigs = Counter(sig_of(c, *q) for q in c.queries)
                    tot = te = 0.0
                    for (gw, aw), cnt in sigs.items():
                        if gw:
                            nn = c.m - len(gw) - len(aw)
                            tot += cnt * bsdc_s_iid_mc(gw, aw, like[3], nn, b, L, 300, rng)
                            te += cnt * bsdc_s_exact(gw, aw, like[3], nn, b, L)
                    iid.append(tot / len(c.queries))
                    ex.append(te / len(c.queries))
                r = {"scheme": scheme, "N": n, "B": b, "L": L,
                     "meas": float(np.mean(meas)), "pred21": float(np.mean(old)),
                     "iid_exact_ties": float(np.mean(iid)), "cond": float(np.mean(cond)),
                     "exact_ties": float(np.mean(ex)),
                     "se_meas": float(np.std(meas, ddof=1) / math.sqrt(len(meas)))}
                rows.append(r)
                print(f"{scheme:7s} N={n:3d} B={b:2d} mis {r['meas']:.3f}±{r['se_meas']:.3f} "
                      f"21 {r['pred21']:.3f} iid-pareggi-esatti {r['iid_exact_ties']:.3f} "
                      f"condizionato {r['cond']:.3f} pareggi-esatti-analitico {r['exact_ties']:.3f}",
                      flush=True)
    return rows


def verify():
    """Passi 2-3: HRR e catene, Monte Carlo sintetico (seed 22_200_xxx)."""
    rows = []
    base = SEED_BASE + 200
    for scheme in ("uniform", "dense"):
        for n in (100, 400):
            cells = _syn_cells(n, scheme, VERIFY_SEEDS, base)
            chains = [{h: sample_chains(c, h, np.random.default_rng([base, n, k, h]))
                       for h in (2, 3)} for k, c in enumerate(cells)]
            for fam in FAMILIES:
                dim = choose_dim(cells[:4], fam, 0.85 * float(np.mean([c.ceiling for c in cells])), n)
                if dim is None:
                    continue
                p = {h: [] for h in HOPS}
                mm = {h: [] for h in HOPS}
                for k, c in enumerate(cells):
                    rng = np.random.default_rng([base, n, k, len(fam)])
                    res = measure(c, fam, dim, rng, c.queries, chains[k])
                    p[1].append(float(np.mean(predict_queries(c, fam, dim, c.queries))))
                    mm[1].append(float(np.mean(res[1])))
                    for h in (2, 3):
                        if chains[k][h]:
                            p[h].append(float(np.mean(predict_chains(c, fam, dim, chains[k][h]))))
                            mm[h].append(float(np.mean(res[h])))
                r = {"scheme": scheme, "N": n, "family": fam, "D": dim}
                for h in HOPS:
                    if p[h]:
                        e = np.asarray(mm[h]) - np.asarray(p[h])
                        r[f"pred{h}"], r[f"meas{h}"] = float(np.mean(p[h])), float(np.mean(mm[h]))
                        r[f"se{h}"] = float(np.std(e, ddof=1) / math.sqrt(len(e)))
                        r[f"n{h}"] = (len(c.queries) * len(cells) if h == 1 else
                                      int(sum(len(ch[h]) for ch in chains)))
                rows.append(r)
                print(f"{scheme:7s} N={n:3d} {fam:7s} D={dim:6d} " + " ".join(
                    f"h{h}: prev {r[f'pred{h}']:.3f} mis {r[f'meas{h}']:.3f}±{r[f'se{h}']:.3f} (n={r[f'n{h}']})"
                    for h in HOPS if f"pred{h}" in r), flush=True)
    return rows


# =============================================================================
# Impianto
# =============================================================================

def bits_per_component(family, n):
    return HRR_BITS if family == "HRR" else v21.bits_per_component(family, n)


def cell_acc(cell, family, dim):
    return float(np.mean(predict_queries(cell, family, dim, cell.queries)))


def choose_dim(cells, family, target_abs, n):
    step = v21.bsdc_length(n) if family.startswith("BSDC") else STEP

    def acc(d):
        return float(np.mean([cell_acc(c, family, d) for c in cells]))
    top = D_MAX // step
    if acc(step * top) < target_abs:
        return None
    lo, hi = 0, top
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if acc(step * mid) >= target_abs:
            hi = mid
        else:
            lo = mid
    return step * hi


def build(graph, loads, seeds=SEEDS):
    cells = {}
    for si, scheme in enumerate(SCHEMES):
        for n in loads:
            cs = []
            for k in range(seeds):
                samp, atoms, ch = np.random.SeedSequence([SEED_BASE, si, n, k]).spawn(3)
                rng = np.random.default_rng(samp)
                c = v21.Cell(v21.sample(graph, n, scheme, rng), rng, atoms)
                crng = np.random.default_rng(ch)
                c.chains = {h: sample_chains(c, h, crng) for h in HOPS if h > 1}
                cs.append(c)
            cells[(scheme, n)] = cs
    return cells


def se_model(preds_by_seed):
    v = [np.sum(np.asarray(p) * (1 - np.asarray(p))) / len(p) ** 2 for p in preds_by_seed if len(p)]
    return float(math.sqrt(np.sum(v)) / len(v))


def plan(cells):
    rows = []
    for (scheme, n), cs in cells.items():
        ceil = float(np.mean([c.ceiling for c in cs[:FIT_SEEDS]]))
        chain_ok = {h: len(chain_seeds(cs, h)) >= CHAIN_SEEDS for h in (2, 3)}
        for t in TARGETS:
            for fam in FAMILIES:
                t0 = time.time()
                d = choose_dim(cs[:FIT_SEEDS], fam, t * ceil, n)
                for h in HOPS:
                    if h > 1 and not chain_ok[h]:
                        continue
                    rows.append(_row(cs, scheme, n, t, fam, d, ceil, h))
                print(f"  {scheme} N={n} t={t} {fam} D={d} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def chain_seeds(cs, h):
    return [k for k, c in enumerate(cs) if len(c.chains[h]) >= CHAIN_MIN]


def _row(cs_all, scheme, n, t, fam, d, ceil, h):
    seeds = list(range(len(cs_all))) if h == 1 else chain_seeds(cs_all, h)
    cs = [cs_all[k] for k in seeds]
    r = {"scheme": scheme, "N": n, "target": t, "target_abs": t * ceil, "family": fam,
         "hops": h, "D": d, "ceiling": ceil, "seeds": seeds,
         "bits": None if d is None else d * bits_per_component(fam, n),
         "M": float(np.mean([c.m for c in cs])),
         "queries": int(sum(len(c.queries) if h == 1 else len(c.chains[h]) for c in cs)),
         "alias_share": float(np.mean([np.mean([len(a) > 0 for _g, a in c.sigs]) for c in cs])),
         "twin_share": float(np.mean([np.mean([w == 2 for w in c.weights.values()]) for c in cs]))}
    if d is None:
        r.update(pred=None, pred_naive=None, se_model=None, mde=None)
        return r
    if h == 1:
        preds = [predict_queries(c, fam, d, c.queries) for c in cs]
        naive = [predict_queries(c, fam, d, c.queries[:1], naive=True)[0] for c in cs]
    else:
        dep = None
        if fam == "MAP-B":                         # uno per cella: N e M medi sui 20 seed
            dep = mapb_dep(int(round(np.mean([c.n_vec for c in cs]))), d,
                           int(round(np.mean([c.m for c in cs]))), h)
            r["dep"] = dep
        preds = [predict_chains(c, fam, d, c.chains[h], dep=dep) for c in cs]
        naive = [float(np.mean(predict_chains(c, fam, d, c.chains[h], naive=True))) for c in cs]
    r["pred_seed"] = [float(np.mean(p)) for p in preds]
    r["pred"] = float(np.mean(r["pred_seed"]))
    oos = [p for k, p in zip(seeds, r["pred_seed"]) if k >= FIT_SEEDS]
    r["pred_oos"] = float(np.mean(oos)) if oos else None
    r["pred_naive_seed"] = [float(x) for x in naive]
    r["pred_naive"] = float(np.mean(naive))
    if fam == "BSDC-S":                            # modello della 21 (pareggi approssimati)
        p21 = []
        for c in cs:
            if h == 1:
                p21.append(v21.predict_cell(c, "BSDC-S", d))
                continue
            hq = sorted({(nd[i], pa[i]) for nd, pa in c.chains[h] for i in range(h)})
            pc = dict(zip(hq, v21.predict_cell(Cell21View(c, hq), "BSDC-S", d)))
            p21.append([float(np.prod([pc[(nd[i], pa[i])] for i in range(h)]))
                        for nd, pa in c.chains[h]])
        r["pred_v21_seed"] = [float(np.mean(p)) for p in p21]
        r["pred_v21"] = float(np.mean(r["pred_v21_seed"]))
    r["se_model"] = se_model(preds)
    r["mde"] = (T_95 + T_80) * r["se_model"]
    return r


class Cell21View:
    """La cella della 21 con un'altra lista di query (per v21.predict_cell)."""

    def __init__(self, cell, queries):
        self.__dict__.update({k: cell.__dict__[k] for k in
                              ("triples", "weights", "s2", "n_vec", "m", "objects", "into")})
        self.queries = list(queries)
        self.sigs = [sig_of(cell, *q) for q in self.queries]


def run(cells, rows):
    """Una memoria per (cella, famiglia, D, seed): 1 hop e catene sulla stessa."""
    done = {}
    for i, r in enumerate(rows):
        if r["pred"] is None:
            continue
        key = (r["scheme"], r["N"], r["target"], r["family"])
        cs = cells[(r["scheme"], r["N"])]
        if key not in done:
            done[key] = []
            for k, c in enumerate(cs):
                rng = np.random.default_rng(np.random.SeedSequence(
                    c.atom_seq.entropy, spawn_key=c.atom_seq.spawn_key
                    + (SCHEMES.index(r["scheme"]), FAMILIES.index(r["family"]),
                       TARGETS.index(r["target"]))))
                done[key].append(measure(c, r["family"], r["D"], rng, c.queries,
                                         {h: c.chains[h] for h in HOPS if h > 1}))
        accs = [float(np.mean(done[key][k][r["hops"]])) for k in r["seeds"]]
        e = np.asarray(accs) - np.asarray(r["pred_seed"])
        r["meas_seed"] = accs
        r["meas"] = float(np.mean(accs))
        oos = [a for k, a in zip(r["seeds"], accs) if k >= FIT_SEEDS]
        r["meas_oos"] = float(np.mean(oos)) if oos else None
        r["err"] = r["meas"] - r["pred"]
        r["err_naive"] = r["meas"] - r["pred_naive"]
        r["se_cluster"] = float(np.std(e, ddof=1) / math.sqrt(len(e)))
        if r["se_cluster"] > 0:
            r["t"] = r["err"] / r["se_cluster"]
        else:                          # 0/0 = nessuna differenza; d ≠ 0 con SE = 0: infinito
            r["t"] = 0.0 if r["err"] == 0 else math.copysign(math.inf, r["err"])
        print(f"{r['scheme']:7s} N={r['N']:4d} t={r['target']} {r['family']:7s} h={r['hops']} "
              f"D={r['D']:6d} prev {r['pred']:.3f} ingenuo {r['pred_naive']:.3f} "
              f"mis {r['meas']:.3f}", flush=True)
    return rows


def summarize(rows):
    """Gruppo = (famiglia, campione, h): celle dei 4 carichi × 2 target misurabili."""
    meas = [r for r in rows if r.get("err") is not None]
    out = {}
    for fam in FAMILIES:
        for scheme in SCHEMES:
            for h in HOPS:
                sel = [r for r in meas if (r["family"], r["scheme"], r["hops"]) == (fam, scheme, h)]
                if not sel:
                    continue
                if h == 1:       # stessi 20 seed in ogni cella: errore per seed, IC a cluster
                    e = np.mean([np.asarray(r["meas_seed"]) - np.asarray(r["pred_seed"])
                                 for r in sel], axis=0)
                    ci = 100 * T_95 * float(np.std(e, ddof=1)) / math.sqrt(len(e))
                else:            # catene: seed diversi per cella; media degli errori di cella
                    e, ci = np.array([r["err"] for r in sel]), None
                out[f"{fam}|{scheme}|h{h}"] = {
                    "mae": 100 * float(np.mean([abs(r["err"]) for r in sel])),
                    "max_cell": 100 * float(np.max([abs(r["err"]) for r in sel])),
                    "mae_naive": 100 * float(np.mean([abs(r["err_naive"]) for r in sel])),
                    "signed": 100 * float(np.mean(e)),
                    "ci95_half": ci,
                    "oos_ok": int(sum(r["meas_oos"] >= r["target_abs"] - 0.03 for r in sel
                                      if h == 1)),
                    "t_over": int(sum(abs(r["t"]) > T_CRIT for r in sel if r["hops"] == 1)),
                    "cells": len(sel)}
    return out


def synthetic_graph():
    return v21.Graph(v21.synthetic(n_ent=400, n_rel=8, n_facts=4000, seed=SEED_BASE % 1000))


def check_mapb(cell, dim):
    """Smoke: l'Engine MAP-B risponde come abm.Memory.query (stessa traccia, stessi pareggi)."""
    mem = abm.Memory(dim)
    for x in cell.symbols:
        mem.items.add(x)
    for t in cell.triples:
        mem._facts.append(mem.fact_hv(*t))
    mem._trace = abm.bundle(mem._facts)
    eng = Engine(cell, "MAP-B", dim, np.random.default_rng(0))
    got = eng.answer([eng.idx[s] for s, _ in cell.queries], [eng.idx[r] for _, r in cell.queries])
    ref = [mem.query(s, r)[0] for s, r in cell.queries]
    assert [eng.sym[g] for g in got] == ref, "Engine MAP-B diverge da Memory.query"


def timing(rows):
    """Stima del tempo del run: una memoria per famiglia al D più grande previsto, su un
    grafo SINTETICO con lo stesso N (nessuna misura sui dati reali)."""
    g = synthetic_graph()
    total = 0.0
    for (scheme, n) in {(r["scheme"], r["N"]) for r in rows}:
        c = _syn_cells(n, scheme, 1, SEED_BASE + 300, n_ent=max(400, n), n_rel=8,
                       n_facts=max(4000, 3 * n))[0]
        c.chains = {h: sample_chains(c, h, np.random.default_rng(h)) for h in (2, 3)}
        for fam in FAMILIES:
            for t in TARGETS:
                ds = [r["D"] for r in rows if (r["scheme"], r["N"], r["family"], r["target"])
                      == (scheme, n, fam, t) and r["D"]]
                if not ds:
                    continue
                t0 = time.time()
                measure(c, fam, ds[0], np.random.default_rng(0), c.queries, c.chains)
                dt = time.time() - t0
                total += dt * SEEDS
                print(f"  tempo {scheme} N={n} {fam} t={t} D={ds[0]}: {dt:.1f} s/seed", flush=True)
    del g
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="solo D e previsioni, nessuna memoria")
    ap.add_argument("--smoke", action="store_true", help="impianto intero su grafo sintetico")
    ap.add_argument("--diagnose", action="store_true", help="passo 1, BSDC-S, sintetico")
    ap.add_argument("--verify", action="store_true", help="passi 2-3, HRR e catene, sintetico")
    ap.add_argument("--timing", action="store_true", help="stima del tempo del run, sintetico")
    a = ap.parse_args()
    t0 = time.time()
    if a.diagnose or a.verify:
        res = {}
        if DIAG_OUT.exists():
            res = json.loads(DIAG_OUT.read_text())
        if a.diagnose:
            res["bsdc_s"] = diagnose()
        if a.verify:
            res["verify"] = verify()
        DIAG_OUT.write_text(json.dumps(res, indent=1))
        print(f"{time.time() - t0:.0f} s")
        return
    if a.timing:
        rows = json.loads(PRED_OUT.read_text())
        tot = timing(rows)
        print(f"run stimato: {tot / 60:.0f} min")
        return
    if a.smoke:
        cells = build(synthetic_graph(), (40, 80))
        c0 = cells[("dense", 40)][0]
        check_mapb(c0, 512)
        assert np.allclose(predict_queries(c0, "MAP-B", 512, c0.queries),
                           exact.predict_queries(c0.triples, 512, c0.queries)), "firma MAP-B"
    else:
        cells = build(v21.Graph(load()), LOADS)
    rows = plan(cells)
    if a.predict:
        for r in rows:
            p = "—" if r["pred"] is None else f"{r['pred']:.3f} ingenuo={r['pred_naive']:.3f}"
            mde = "—" if r["mde"] is None else f"{100 * r['mde']:.1f}"
            print(f"{r['scheme']:7s} N={r['N']:4d} t={r['target']} {r['family']:7s} h={r['hops']} "
                  f"D={r['D']} bit={r['bits']} tetto={r['ceiling']:.3f} prev={p} MDE={mde}pt "
                  f"q={r['queries']}")
        if not a.smoke:
            PRED_OUT.write_text(json.dumps(rows, indent=1))
        print(f"{time.time() - t0:.0f} s")
        return
    rows = run(cells, rows)
    summ = summarize(rows)
    for k, v in summ.items():
        print(k, v)
    if not a.smoke:
        OUT.write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
