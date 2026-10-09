"""vsa_families_prereg.py — la stima a priori dell'accuratezza del cleanup su cinque famiglie VSA.

Preregistrazione 21: docs/preregistration/vsa_families.md. Committato insieme a quel
file e PRIMA di misurare (revisione del disegno prima dei dati inclusa). Prima del
commit è stato eseguito solo con `--smoke` (grafo sintetico generato qui, mai
FB15k-237 o WN18RR) e con `--predict` (legge i dati, campiona i sottografi e calcola D
e previsioni; non costruisce nessuna memoria).

Famiglie (tutte con lo stesso encoding di un fatto, f = s ∘ ρ(r) ∘ o, simmetrico in s, o):
- MAP-B    la reference ABM congelata (bipolare, bundling a maggioranza). Controllo:
           la previsione è `exact.predict_queries`, già testata (prereg. 10, 18, 19).
           Atomi dall'md5 del nome: gli stessi in tutti i seed.
- MAP-I    atomi bipolari, binding prodotto, bundling per somma intera NON binarizzata,
           cleanup argmax del prodotto scalare.
- FHRR     atomi fasori e^{iθ}; binding prodotto complesso, unbinding per coniugato,
           bundling per somma, cleanup argmax di Re⟨y, c̄⟩. Traccia salvata in float16.
- BSDC-OR  codici sparsi a blocchi (≈ Bloom partizionato): B blocchi di lunghezza L, un 1
           per blocco; binding = somma degli indici mod L; bundling per OR (B·L bit);
           cleanup = numero di blocchi colpiti.
- BSDC-S   come BSDC-OR ma traccia a conteggi (bundling per somma); cleanup = somma dei
           conteggi letti nei B blocchi.

Previsioni con la contabilità di ABM (gemelli = un vettore di peso 2, alias, più oggetti
veri, codebook M = entità + relazioni). Comparatore "FKS ingenuo": stessa formula con
S2 = N, nessun alias, un solo oggetto vero di peso 1.

Uso, dalla root:  python examples/vsa_families_prereg.py [--predict | --smoke]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

EXT = ROOT / "data" / "external"
FILES = {
    "fb15k237": ("fb15k237_train.txt",
                 "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"),
    "wn18rr": ("wn18rr_train.csv",
               "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"),
}
OUT = ROOT / "results" / "vsa_families_prereg_results.json"
PRED_OUT = ROOT / "results" / "vsa_families_prereg_predictions.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DATASETS = ("fb15k237", "wn18rr")
SCHEMES = ("uniform", "dense")
LOADS = (100, 400)
TARGETS = (0.50, 0.85)           # frazione del tetto degli alias
FAMILIES = ("MAP-B", "MAP-I", "FHRR", "BSDC-OR", "BSDC-S")
SEEDS = 10
FIT_SEEDS = 5                    # D scelto sui seed 0–4; validazione fuori campione su 5–9
Q_MAX = 200
STEP = 64                        # granularità di D (MAP-B, MAP-I, FHRR)
D_MAX = 1 << 16
FHRR_BITS = 32                   # float16 reale + float16 immaginaria
PI_DRAWS = 2000                  # estrazioni dell'occupazione realizzata (BSDC-OR)
T_CRIT = 3.250                   # t di Student, 9 gdl, bilaterale 0.01 (H8)
T_95, T_80 = 2.262, 0.883        # per IC al 95 % e MDE (potenza 0.8), 9 gdl
SEED_BASE = 21_000_021
# -----------------------------------------------------------------------------


def load(name):
    fn, sha = FILES[name]
    path = EXT / fn
    if not path.exists():
        raise SystemExit(f"{path} mancante: scaricalo come in examples/replicate.py (DATASETS)")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"{fn}: sha256 {got}, atteso {sha}")
    if name == "wn18rr":
        rows = list(csv.reader(path.open()))[1:]
        return [(f"wn{h}", r, f"wn{t}") for h, r, t in rows]
    return [tuple(line.rstrip("\n").split("\t")) for line in path.open()]


def synthetic(n_ent=60, n_rel=4, n_facts=400, seed=0):
    """Grafo minuscolo per lo smoke, con simmetrie (gemelli) e alias."""
    rng = np.random.default_rng(seed)
    fs = set()
    while len(fs) < n_facts:
        s, o = rng.integers(n_ent, size=2)
        if s != o:
            r = int(rng.integers(n_rel))
            fs.add((f"e{s}", f"r{r}", f"e{o}"))
            if r == 0:
                fs.add((f"e{o}", f"r{r}", f"e{s}"))
    return sorted(fs)


class Graph:
    def __init__(self, triples):
        self.triples = sorted({t for t in triples if t[0] != t[2]})   # niente self-loop
        self.adj = defaultdict(list)
        for i, (s, _r, o) in enumerate(self.triples):
            self.adj[s].append(i)
            self.adj[o].append(i)
        self.entities = sorted(self.adj)


def sample(g, n, scheme, rng):
    """N triple distinte. uniform: a caso. dense: BFS non orientata da un'entità a caso
    (vicinati interi, quindi gemelli, alias e query a più oggetti)."""
    if scheme == "uniform":
        return [g.triples[i] for i in sorted(rng.choice(len(g.triples), n, replace=False))]
    chosen, seen_e, taken = [], set(), set()
    while len(chosen) < n:
        start = g.entities[int(rng.integers(len(g.entities)))]
        if start in seen_e:
            continue
        frontier = [start]
        seen_e.add(start)
        while frontier and len(chosen) < n:
            e = frontier.pop(0)
            idx = list(g.adj[e])
            rng.shuffle(idx)
            for i in idx:
                if i in taken:
                    continue
                taken.add(i)
                chosen.append(g.triples[i])
                if len(chosen) == n:
                    break
                s, _r, o = g.triples[i]
                for x in (s, o):
                    if x not in seen_e:
                        seen_e.add(x)
                        frontier.append(x)
    return chosen


class Cell:
    """Un sottografo con le sue query e la contabilità per la previsione."""

    def __init__(self, triples, rng, atom_seq):
        self.triples = triples
        self.atom_seq = atom_seq             # flusso degli atomi, separato dal campionamento
        objects, into = defaultdict(set), defaultdict(set)
        for s, r, o in triples:
            objects[(s, r)].add(o)
            into[(o, r)].add(s)
        keys = sorted(objects)
        if len(keys) > Q_MAX:
            keys = [keys[i] for i in sorted(rng.choice(len(keys), Q_MAX, replace=False))]
        self.objects, self.into, self.queries = objects, into, keys
        self.symbols = sorted({x for t in triples for x in t})
        self.weights = exact.fact_weights(triples)
        self.s2 = float(sum(w * w for w in self.weights.values()))
        self.n_vec = len(self.weights)
        self.m = len(self.symbols)
        self.sigs = []                       # (pesi oggetti veri, pesi alias) per query
        for s, r in self.queries:
            good = objects[(s, r)]
            bad = into[(s, r)] - good
            gw = tuple(sorted(self.weights[exact.fact_key(s, r, o)] for o in good))
            aw = tuple(sorted(self.weights[exact.fact_key(x, r, s)] for x in bad))
            self.sigs.append((gw, aw))
        self.ceiling = float(np.mean([len(g) / (len(g) + len(a)) for g, a in self.sigs]))

    def atoms_rng(self, row):
        return np.random.default_rng(np.random.SeedSequence(
            self.atom_seq.entropy, spawn_key=self.atom_seq.spawn_key + (row,)))


# ---- previsioni -------------------------------------------------------------

_ERFC = np.vectorize(math.erfc)


def _log_cdf(z):
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(0.5 * _ERFC(-z / math.sqrt(2.0)), 1e-300))


def gaussian_max(cands, null_mu, null_sd, n_null):
    """P(il massimo è un candidato 'ok'); cands = [(media, sd, ok)], nulli N(null_mu, null_sd²)
    i.i.d., tutti indipendenti."""
    mus = [(m - null_mu) / null_sd for m, _s, _ok in cands]
    sds = [max(s, 1e-9) / null_sd for _m, s, _ok in cands]
    x = np.linspace(-9.0, max(mus) + 9.0, 6001)
    dx = x[1] - x[0]
    logc = [_log_cdf((x - m) / s) for m, s in zip(mus, sds)]
    tot = np.sum(logc, axis=0) + n_null * _log_cdf(x)
    acc = 0.0
    for i, (_m, _s, ok) in enumerate(cands):
        if ok:
            z = (x - mus[i]) / sds[i]
            dens = np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sds[i])
            acc += float(np.sum(dens * np.exp(tot - logc[i])) * dx)
    return min(acc, 1.0)


def gaussian_dense_acc(gw, aw, s2, n_null, dim, v):
    """MAP-I (v = 1), FHRR (v = 1/2): segnale w·D, varianza v·D·(S2 − w²); nullo v·D·S2."""
    cands = [(w * dim, math.sqrt(v * dim * (s2 - w * w)), ok)
             for ws, ok in ((gw, True), (aw, False)) for w in ws]
    return gaussian_max(cands, 0.0, math.sqrt(v * dim * s2), n_null)


_SUMPMF = {}


def _bsdc_s_noise(wcounts, blocks, length):
    """pmf esatta di Σ_b Σ_w w·Bin(n_w, 1/L): il punteggio di un nullo (somma dei conteggi
    letti nei B blocchi), con n_w vettori distinti di peso w."""
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


def _binom(n, p):
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    return np.exp(lg[n] - lg - lg[::-1] + k * math.log(p) + (n - k) * math.log1p(-p))


def bsdc_s_acc(gw, aw, wcounts, n_null, blocks, length):
    """BSDC-S, discreto: nullo ~ pmf esatta del rumore; candidato a segnale di peso w ~
    w·B + rumore degli altri vettori (il proprio è tolto). Pareggi
    divisi in proporzione alle densità relative dei candidati al massimo."""
    noise = _bsdc_s_noise(wcounts, blocks, length)
    top = len(noise) + blocks * (max(gw + aw) + 1)
    null = np.zeros(top)
    null[: len(noise)] = noise
    cands = []
    for ws, ok in ((gw, True), (aw, False)):
        for w in ws:
            others = tuple((v, c - (v == w)) for v, c in wcounts if c - (v == w) > 0)
            nz = _bsdc_s_noise(others, blocks, length)
            f = np.zeros(top)
            f[w * blocks: w * blocks + len(nz)] = nz[: top - w * blocks]
            cands.append((f, ok))
    def cdfs(f):
        c = np.cumsum(f)
        return c, np.concatenate(([0.0], c[:-1]))
    log_now, log_prev = np.zeros(top), np.zeros(top)
    hz_all, hz_ok = np.zeros(top), np.zeros(top)
    with np.errstate(divide="ignore", invalid="ignore"):
        for f, ok, cnt in [(null, False, n_null)] + [(f, ok, 1) for f, ok in cands]:
            c, cp = cdfs(f)
            log_now += cnt * np.log(np.maximum(c, 1e-300))
            log_prev += cnt * np.log(np.maximum(cp, 1e-300))
            h = np.where(c > 0, cnt * f / np.maximum(c, 1e-300), 0.0)
            hz_all += h
            if ok:
                hz_ok += h
        p_max = np.exp(log_now) - np.exp(log_prev)
        share = np.where(hz_all > 0, hz_ok / np.maximum(hz_all, 1e-300), 0.0)
    return float(np.clip(np.sum(p_max * share), 0.0, 1.0))


_OCC = {}


def occupancy_pmf(n, length):
    """P(K = k) bin occupati dopo n palline uniformi in L bin (esatta, per ricorrenza;
    equivale a S(n,k)·L!/((L−k)!·L^n))."""
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


def pi_draws(n, length, blocks):
    """Estrazioni di π = Π_b K_b/L con K_b dall'occupazione esatta (seme fisso)."""
    key = (n, length, blocks)
    if key not in _PI:
        pmf = occupancy_pmf(n, length)
        rng = np.random.default_rng([SEED_BASE, n, length, blocks])
        k = rng.choice(length + 1, size=(PI_DRAWS, blocks), p=pmf / pmf.sum())
        _PI[key] = np.exp(np.sum(np.log(np.maximum(k, 1) / length), axis=1))
    return _PI[key]


_T = np.linspace(0.0, 1.0, 2001)


_F = {}
_PGRID = np.concatenate(([0.0], np.logspace(-12, 0, 600)))


def bsdc_or_acc(gw, aw, n_vec, n_null, blocks, length):
    """g · E[1/(g + a + X)], X | π ~ Bin(n_null, π), π con occupazione realizzata;
    E[1/(c+X)] = ∫₀¹ t^{c−1} (1 − π + π t)^{n_null} dt, calcolato su una griglia di π
    (601 punti log-spaziati) e interpolato sulle estrazioni."""
    g, c = len(gw), len(gw) + len(aw)
    key = (c, n_null)
    if key not in _F:
        pg = _PGRID[:, None]
        _F[key] = np.trapezoid(_T[None, :] ** (c - 1) * (1 - pg + pg * _T[None, :]) ** n_null,
                               _T, axis=1)
    f = _F[key]
    return float(g * np.mean(np.interp(pi_draws(n_vec, length, blocks), _PGRID, f)))


def bsdc_length(n):
    """L: potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà)."""
    return int(2 ** round(math.log2(n / math.log(2))))


def _sig_acc(cell_like, family, dim, gw, aw):
    m, s2, n_vec, wcounts, n = cell_like
    n_null = m - len(gw) - len(aw)
    if family in ("BSDC-OR", "BSDC-S"):
        length = bsdc_length(n)
        if family == "BSDC-OR":
            return bsdc_or_acc(gw, aw, n_vec, n_null, dim // length, length)
        return bsdc_s_acc(gw, aw, wcounts, n_null, dim // length, length)
    return gaussian_dense_acc(gw, aw, s2, n_null, dim, 1.0 if family == "MAP-I" else 0.5)


def predict_cell(cell, family, dim):
    """Accuratezza prevista per query, con la contabilità. dim = componenti (BSDC: B·L)."""
    if family == "MAP-B":
        return exact.predict_queries(cell.triples, dim, cell.queries)
    n = len(cell.triples)
    from collections import Counter
    like = (cell.m, cell.s2, cell.n_vec,
            tuple(sorted(Counter(cell.weights.values()).items())), n)
    cache, out = {}, []
    for sig in cell.sigs:
        if sig not in cache:
            cache[sig] = _sig_acc(like, family, dim, *sig)
        out.append(cache[sig])
    return out


def predict_naive(cell, family, dim):
    """Comparatore FKS ingenuo: N fatti distinti di peso 1, un oggetto vero, nessun alias."""
    n = len(cell.triples)
    if family == "MAP-B":
        return float(exact.cleanup_accuracy(n, dim, cell.m))
    return _sig_acc((cell.m, float(n), n, ((1, n),), n), family, dim, (1,), ())


def bits_per_component(family, n):
    if family in ("MAP-B", "BSDC-OR"):
        return 1
    if family == "FHRR":
        return FHRR_BITS
    # MAP-I: T_d ≡ N (mod 2), N + 1 valori. BSDC-S: conteggi 0..N, N + 1 valori.
    return math.ceil(math.log2(n + 1))


def choose_dim(cells, family, target_abs, n):
    """Il D minimo (multiplo di STEP; BSDC: di L) con previsione media sui seed di stima
    ≥ target_abs; None se non raggiungibile entro D_MAX."""
    step = bsdc_length(n) if family.startswith("BSDC") else STEP

    def acc(d):
        return float(np.mean([np.mean(predict_cell(c, family, d)) for c in cells]))
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


def se_model(preds_by_seed):
    """SE del modello per la media di cella (media dei seed), query indipendenti:
    limite inferiore dell'SE a cluster, usato solo per l'MDE preregistrato."""
    v = [np.sum(np.asarray(p) * (1 - np.asarray(p))) / len(p) ** 2 for p in preds_by_seed]
    return float(math.sqrt(np.sum(v)) / len(v))


# ---- misure ------------------------------------------------------------------

def measure(cell, family, dim, rng):
    if family == "MAP-B":
        mem = abm.Memory(dim)
        for s in cell.symbols:
            mem.items.add(s)
        for s, r, o in cell.triples:
            mem._facts.append(mem.fact_hv(s, r, o))
        mem._trace = abm.bundle(mem._facts)
        return [mem.query(s, r)[0] in cell.objects[(s, r)] for s, r in cell.queries]
    idx = {s: i for i, s in enumerate(cell.symbols)}
    m = cell.m
    if family.startswith("BSDC"):
        length = bsdc_length(len(cell.triples))
        blocks = dim // length
        atoms = rng.integers(length, size=(m, blocks))
        rho = np.roll(atoms, 1, axis=1)
        mem = np.zeros((blocks, length), dtype=np.int64)
        ar = np.arange(blocks)
        for s, r, o in cell.triples:
            np.add.at(mem, (ar, (atoms[idx[s]] + rho[idx[r]] + atoms[idx[o]]) % length), 1)
        if family == "BSDC-OR":
            mem = (mem > 0).astype(np.int64)
        out = []
        for s, r in cell.queries:
            pos = (atoms + atoms[idx[s]] + rho[idx[r]]) % length
            score = mem[ar[None, :], pos].sum(axis=1)
            out.append(_pick(score, rng) in _ids(cell, idx, s, r))
        return out
    if family == "MAP-I":
        atoms = (2 * rng.integers(2, size=(m, dim)) - 1).astype(np.float64)
        rho = np.roll(atoms, 1, axis=1)
        trace = np.zeros(dim)
        for s, r, o in cell.triples:
            trace += atoms[idx[s]] * rho[idx[r]] * atoms[idx[o]]
        n = len(cell.triples)
        assert np.all(np.abs(trace) <= n) and np.all((trace + n) % 2 == 0)
        out = []
        for s, r in cell.queries:
            y = trace * atoms[idx[s]] * rho[idx[r]]
            out.append(_pick(atoms @ y, rng) in _ids(cell, idx, s, r))
        return out
    # FHRR
    theta = rng.uniform(0, 2 * np.pi, size=(m, dim))
    atoms = np.exp(1j * theta).astype(np.complex64)
    rho = np.roll(atoms, 1, axis=1)
    trace = np.zeros(dim, dtype=np.complex128)
    for s, r, o in cell.triples:
        trace += atoms[idx[s]] * rho[idx[r]] * atoms[idx[o]]
    stored = (trace.real.astype(np.float16).astype(np.float32)
              + 1j * trace.imag.astype(np.float16).astype(np.float32))
    conj = np.conj(atoms)
    out = []
    for s, r in cell.queries:
        y = stored * conj[idx[s]] * np.conj(rho[idx[r]])
        out.append(_pick((conj @ y).real, rng) in _ids(cell, idx, s, r))
    return out


def _ids(cell, idx, s, r):
    return {idx[o] for o in cell.objects[(s, r)]}


def _pick(score, rng):
    """argmax con pareggi divisi a caso."""
    best = np.flatnonzero(score == score.max())
    return int(best[rng.integers(len(best))])


def check_mapb_equivalence(cell, dim):
    """Smoke: la traccia costruita in `measure` è quella di Memory.store."""
    a = abm.Memory(dim)
    for t in cell.triples:
        a.store(*t)
    b = abm.Memory(dim)
    for t in cell.triples:
        b._facts.append(b.fact_hv(*t))
    b._trace = abm.bundle(b._facts)
    assert np.array_equal(a._trace, b._trace), "measure MAP-B diverge da Memory.store"


# ---- impianto ---------------------------------------------------------------

def build_cells(graphs, datasets, loads):
    cells = {}
    for di, name in enumerate(datasets):
        for si, scheme in enumerate(SCHEMES):
            for n in loads:
                cs = []
                for k in range(SEEDS):
                    samp, atoms = np.random.SeedSequence([SEED_BASE, di, si, n, k]).spawn(2)
                    rng = np.random.default_rng(samp)
                    cs.append(Cell(sample(graphs[name], n, scheme, rng), rng, atoms))
                cells[(name, scheme, n)] = cs
    return cells


def plan(cells):
    """Tutte le celle con D scelto e previsioni, calcolate senza costruire memorie."""
    rows = []
    for (name, scheme, n), cs in cells.items():
        ceil = float(np.mean([c.ceiling for c in cs[:FIT_SEEDS]]))
        d_abm_hi = None
        for t in TARGETS:
            for fam in FAMILIES:
                d = choose_dim(cs[:FIT_SEEDS], fam, t * ceil, n)
                if fam == "MAP-B" and t == max(TARGETS):
                    d_abm_hi = d
                rows.append(_row(cs, name, scheme, n, "calib", t, fam, d, ceil))
        budget = d_abm_hi                          # bit di MAP-B al target alto
        for fam in FAMILIES:
            if fam == "MAP-B" or budget is None:
                continue
            bpc = bits_per_component(fam, n)
            if fam.startswith("BSDC"):
                length = bsdc_length(n)
                d = (budget // bpc // length) * length
            else:
                d = budget // bpc
            rows.append(_row(cs, name, scheme, n, "equal_bits", None, fam, d, ceil, budget))
    return rows


def _row(cs, name, scheme, n, block, t, fam, d, ceil, budget=None):
    r = {"dataset": name, "scheme": scheme, "N": n, "block": block, "target": t,
         "target_abs": None if t is None else t * ceil,
         "family": fam, "D": d, "ceiling": ceil,
         "bits": None if d is None else d * bits_per_component(fam, n), "budget": budget,
         "M": float(np.mean([c.m for c in cs])),
         "queries": int(sum(len(c.queries) for c in cs)),
         "alias_share": float(np.mean([np.mean([len(a) > 0 for _g, a in c.sigs]) for c in cs])),
         "twin_share": float(np.mean([np.mean([w == 2 for w in c.weights.values()]) for c in cs]))}
    if d is None or d <= 0:
        r.update(pred=None, pred_naive=None, se_model=None, mde=None)
        return r
    preds = [predict_cell(c, fam, d) for c in cs]
    r["pred_seed"] = [float(np.mean(p)) for p in preds]
    r["pred"] = float(np.mean(r["pred_seed"]))
    r["pred_oos"] = float(np.mean(r["pred_seed"][FIT_SEEDS:]))
    r["pred_naive_seed"] = [predict_naive(c, fam, d) for c in cs]
    r["pred_naive"] = float(np.mean(r["pred_naive_seed"]))
    r["se_model"] = se_model(preds)
    r["mde"] = (T_95 + T_80) * r["se_model"]      # α = 0.05 bilaterale, potenza 0.8, 9 gdl
    return r


def run(cells, rows):
    for i, r in enumerate(rows):
        if r["pred"] is None:
            continue
        cs = cells[(r["dataset"], r["scheme"], r["N"])]
        accs = [float(np.mean(measure(c, r["family"], r["D"], c.atoms_rng(i)))) for c in cs]
        e = np.asarray(accs) - np.asarray(r["pred_seed"])
        r["meas_seed"] = accs
        r["meas"] = float(np.mean(accs))
        r["meas_oos"] = float(np.mean(accs[FIT_SEEDS:]))
        r["err"] = r["meas"] - r["pred"]
        r["err_naive"] = r["meas"] - r["pred_naive"]
        r["se_cluster"] = float(np.std(e, ddof=1) / math.sqrt(len(e)))
        r["t"] = r["err"] / r["se_cluster"] if r["se_cluster"] > 0 else None
        print(f"{r['dataset']:8s} {r['scheme']:7s} N={r['N']:4d} {r['block']:10s} "
              f"{str(r['target']):5s} {r['family']:7s} D={r['D']:6d} "
              f"prev {r['pred']:.3f} ingenuo {r['pred_naive']:.3f} mis {r['meas']:.3f}", flush=True)
    return rows


def summarize(rows):
    calib = [r for r in rows if r["block"] == "calib" and r.get("err") is not None]
    out = {}
    for fam in FAMILIES:
        for scheme in SCHEMES:
            for ds in DATASETS:
                sel = [r for r in calib if r["family"] == fam and r["scheme"] == scheme
                       and r["dataset"] == ds]
                if not sel:
                    continue
                # errore con segno per seed, mediato sulle 4 celle (che condividono i sottografi)
                e = np.mean([np.asarray(r["meas_seed"]) - np.asarray(r["pred_seed"])
                             for r in sel], axis=0)
                half = T_95 * float(np.std(e, ddof=1)) / math.sqrt(len(e))
                out[f"{fam}|{scheme}|{ds}"] = {
                    "mae": 100 * float(np.mean([abs(r["err"]) for r in sel])),
                    "max_cell": 100 * float(np.max([abs(r["err"]) for r in sel])),
                    "mae_naive": 100 * float(np.mean([abs(r["err_naive"]) for r in sel])),
                    "signed": 100 * float(np.mean(e)), "ci95_half": 100 * half,
                    "oos_ok": sum(r["meas_oos"] >= r["target_abs"] - 0.03 for r in sel),
                    "cells": len(sel)}
        ts = [r["t"] for r in calib if r["family"] == fam and r.get("t") is not None]
        out[f"{fam}|t>{T_CRIT}"] = (sum(abs(t) > T_CRIT for t in ts), len(ts))
    # descrittivo: ordine fra famiglie a pari bit (MAP-B nella sua cella calib alta)
    eq = [r for r in rows if r["block"] == "equal_bits" and r.get("meas") is not None]
    agree, total = 0, 0
    for key in {(r["dataset"], r["scheme"], r["N"]) for r in eq}:
        grp = [r for r in eq if (r["dataset"], r["scheme"], r["N"]) == key]
        grp += [r for r in calib if (r["dataset"], r["scheme"], r["N"]) == key
                and r["family"] == "MAP-B" and r["target"] == max(TARGETS)]
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                if abs(a["pred"] - b["pred"]) >= 0.05:
                    total += 1
                    agree += (a["pred"] > b["pred"]) == (a["meas"] > b["meas"])
    out["descr_equal_bits_order"] = (agree, total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true", help="solo D e previsioni, nessuna memoria")
    ap.add_argument("--smoke", action="store_true", help="grafo sintetico minuscolo")
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        graphs = {"fb15k237": Graph(synthetic(seed=1)), "wn18rr": Graph(synthetic(seed=2))}
        cells = build_cells(graphs, DATASETS, (40,))
        check_mapb_equivalence(next(iter(cells.values()))[0], 512)
    else:
        graphs = {n: Graph(load(n)) for n in DATASETS}
        cells = build_cells(graphs, DATASETS, LOADS)
    rows = plan(cells)
    if a.predict:
        for r in rows:
            p = "—" if r["pred"] is None else f"{r['pred']:.3f} ingenuo={r['pred_naive']:.3f}"
            mde = "—" if r["mde"] is None else f"{100 * r['mde']:.1f}"
            print(f"{r['dataset']:8s} {r['scheme']:7s} N={r['N']:4d} {r['block']:10s} "
                  f"t={str(r['target']):5s} {r['family']:7s} D={r['D']} bits={r['bits']} "
                  f"tetto={r['ceiling']:.3f} prev={p} MDE={mde}pt")
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
