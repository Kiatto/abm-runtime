"""vsa_families_prereg.py — la stima a priori dell'accuratezza del cleanup su quattro famiglie VSA.

Preregistrazione 21: docs/preregistration/vsa_families.md. Committato insieme a quel
file e PRIMA di misurare. Prima del commit è stato eseguito solo con `--smoke` (grafo
sintetico generato qui, mai FB15k-237 o WN18RR) e con `--predict` (legge i dati,
campiona i sottografi e calcola D e previsioni; non costruisce nessuna memoria).

Famiglie (tutte con lo stesso encoding di un fatto, f = s ∘ ρ(r) ∘ o, simmetrico in s, o):
- MAP-B  la reference ABM congelata (bipolare, bundling a maggioranza). Controllo:
         la previsione è `exact.predict_queries`, già testata (prereg. 10, 18, 19).
- MAP-I  atomi bipolari, binding prodotto elemento per elemento, bundling per somma
         intera SENZA binarizzare, cleanup argmax del prodotto scalare.
- FHRR   atomi fasori e^{iθ}, θ uniforme; binding prodotto complesso, unbinding per
         coniugato, bundling per somma, cleanup argmax di Re⟨y, c̄⟩. Traccia salvata
         in float16 (parte reale e immaginaria): 32 bit per componente.
- BSDC   codici sparsi a blocchi (BSDC-SEG): B blocchi di lunghezza L, un 1 per blocco;
         binding = somma degli indici mod L blocco per blocco; bundling per OR (la
         traccia è B·L bit); cleanup = numero di blocchi in cui il candidato colpisce.

Previsioni, prima di costruire la memoria, con la contabilità di ABM (gemelli (s,r,o)
e (o,r,s) come un vettore di peso 2, alias x con (x,r,s) memorizzato, più oggetti
veri come più bersagli, codebook M = entità + relazioni del sottografo):
- MAP-I, FHRR: approssimazione gaussiana di Frady-Kleyko-Sommer (2018), candidati
  indipendenti: oggetto vero/alias di peso w ~ N(w·D, v·D·(S2 − w²)), nullo ~
  N(0, v·D·S2), S2 = Σ w_j², v = 1 (MAP-I), 1/2 (FHRR). Corretto = il massimo è un
  oggetto vero.
- BSDC: un candidato a segnale colpisce tutti i B blocchi; un nullo li colpisce tutti
  con probabilità π = (1 − (1 − 1/L)^n)^B (n vettori distinti); pareggi divisi a caso:
  acc = E[g / (g + a + X)], X ~ Binomiale(M − g − a, π).

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
FAMILIES = ("MAP-B", "MAP-I", "FHRR", "BSDC")
SEEDS = 10
Q_MAX = 200
STEP = 64                        # granularità di D (MAP-B, MAP-I, FHRR)
D_MAX = 1 << 16
FHRR_BITS = 32                   # float16 reale + float16 immaginaria
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
    rng = np.random.RandomState(seed)
    fs = set()
    while len(fs) < n_facts:
        s, o = rng.randint(n_ent, size=2)
        if s != o:
            r = rng.randint(n_rel)
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
    chosen, seen_e = [], set()
    taken = set()
    while len(chosen) < n:
        start = g.entities[rng.randint(len(g.entities))]
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


def structure(triples, rng):
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in triples:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)
    keys = sorted(objects)
    if len(keys) > Q_MAX:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), Q_MAX, replace=False))]
    symbols = sorted({x for t in triples for x in t})
    return objects, into, keys, symbols


class Cell:
    """Un sottografo con le sue query e la contabilità per la previsione."""

    def __init__(self, triples, rng):
        self.triples = triples
        self.objects, self.into, self.queries, self.symbols = structure(triples, rng)
        self.weights = exact.fact_weights(triples)
        self.s2 = float(sum(w * w for w in self.weights.values()))
        self.n_vec = len(self.weights)
        self.m = len(self.symbols)
        self.sigs = []                       # (pesi oggetti veri, pesi alias) per query
        for s, r in self.queries:
            good = self.objects[(s, r)]
            bad = self.into[(s, r)] - good
            gw = tuple(sorted(self.weights[exact.fact_key(s, r, o)] for o in good))
            aw = tuple(sorted(self.weights[exact.fact_key(x, r, s)] for x in bad))
            self.sigs.append((gw, aw))
        self.ceiling = float(np.mean([len(g) / (len(g) + len(a)) for g, a in self.sigs]))


# ---- previsioni -------------------------------------------------------------

_ERFC = np.vectorize(math.erfc)
_X = np.linspace(-9.0, 9.0, 7201)


def _log_cdf(z):
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(0.5 * _ERFC(-z / math.sqrt(2.0)), 1e-300))


def gaussian_acc(gw, aw, s2, n_null, dim, v):
    """P(il massimo dei punteggi è un oggetto vero), candidati gaussiani indipendenti,
    standardizzati sul nullo (media 0, varianza v·D·S2)."""
    sd0 = math.sqrt(v * s2 / dim)
    cands = [(w, True) for w in gw] + [(w, False) for w in aw]
    mus = [w / sd0 for w, _ in cands]
    sds = [math.sqrt(max(s2 - w * w, 1e-12) / s2) for w, _ in cands]
    lo, hi = -9.0, max(mus) + 9.0
    x = np.linspace(lo, hi, 6001)
    dx = x[1] - x[0]
    log_null = n_null * _log_cdf(x)
    logc = [_log_cdf((x - m) / s) for m, s in zip(mus, sds)]
    tot = np.sum(logc, axis=0) + log_null
    acc = 0.0
    for i, (w, ok) in enumerate(cands):
        if not ok:
            continue
        z = (x - mus[i]) / sds[i]
        dens = np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sds[i])
        acc += float(np.sum(dens * np.exp(tot - logc[i])) * dx)
    return min(acc, 1.0)


def _binom_pmf(n, p):
    k = np.arange(n + 1)
    lg = np.vectorize(math.lgamma)
    with np.errstate(divide="ignore"):
        logp = (lg(n + 1) - lg(k + 1) - lg(n - k + 1)
                + k * (math.log(p) if p > 0 else -np.inf)
                + (n - k) * (math.log1p(-p) if p < 1 else -np.inf))
    pmf = np.exp(logp)
    if p == 0:
        pmf = np.zeros(n + 1)
        pmf[0] = 1.0
    return pmf


def bsdc_acc(gw, aw, n_vec, n_null, blocks, length):
    g, a = len(gw), len(aw)
    q = 1.0 - (1.0 - 1.0 / length) ** n_vec
    pi = q ** blocks
    pmf = _binom_pmf(n_null, pi)
    k = np.arange(n_null + 1)
    return float(np.sum(pmf * g / (g + a + k)))


def bsdc_length(n):
    """L: potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà)."""
    return int(2 ** round(math.log2(n / math.log(2))))


def predict_cell(cell, family, dim):
    """Accuratezza prevista per query. dim = numero di componenti (BSDC: B·L)."""
    if family == "MAP-B":
        return exact.predict_queries(cell.triples, dim, cell.queries)
    out = []
    cache = {}
    for gw, aw in cell.sigs:
        key = (gw, aw)
        if key not in cache:
            n_null = cell.m - len(gw) - len(aw)
            if family == "BSDC":
                length = bsdc_length(len(cell.triples))
                cache[key] = bsdc_acc(gw, aw, cell.n_vec, n_null, dim // length, length)
            else:
                v = 1.0 if family == "MAP-I" else 0.5
                cache[key] = gaussian_acc(gw, aw, cell.s2, n_null, dim, v)
        out.append(cache[key])
    return out


def bits_per_component(family, n):
    if family in ("MAP-B", "BSDC"):
        return 1
    if family == "FHRR":
        return FHRR_BITS
    return math.ceil(math.log2(2 * n + 1))   # MAP-I: somma intera senza perdita, |T_d| ≤ N


def choose_dim(cells, family, target_abs, n):
    """Il D minimo (multiplo di STEP; BSDC: multiplo di L) con previsione media sui
    seed ≥ target_abs; None se non raggiungibile entro D_MAX."""
    step = bsdc_length(n) if family == "BSDC" else STEP

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


def se_cell(preds):
    """SE del modello per la media di una cella (query indipendenti, Poisson-binomiale)."""
    p = np.asarray(preds)
    return float(math.sqrt(np.sum(p * (1 - p))) / len(p))


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
    if family == "BSDC":
        length = bsdc_length(len(cell.triples))
        blocks = dim // length
        atoms = rng.randint(length, size=(m, blocks))
        rho = np.roll(atoms, 1, axis=1)
        mem = np.zeros((blocks, length), dtype=bool)
        ar = np.arange(blocks)
        for s, r, o in cell.triples:
            mem[ar, (atoms[idx[s]] + rho[idx[r]] + atoms[idx[o]]) % length] = True
        out = []
        for s, r in cell.queries:
            pos = (atoms + atoms[idx[s]] + rho[idx[r]]) % length
            score = mem[ar[None, :], pos].sum(axis=1)
            out.append(_pick(score, rng) in _ids(cell, idx, s, r))
        return out
    if family == "MAP-I":
        atoms = (2 * rng.randint(2, size=(m, dim)) - 1).astype(np.float64)
        rho = np.roll(atoms, 1, axis=1)
        trace = np.zeros(dim)
        for s, r, o in cell.triples:
            trace += atoms[idx[s]] * rho[idx[r]] * atoms[idx[o]]
        assert np.max(np.abs(trace)) <= len(cell.triples)
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
    return int(best[rng.randint(len(best))])


# ---- impianto ---------------------------------------------------------------

def build_cells(graphs, datasets, loads):
    cells = {}
    for name in datasets:
        for scheme in SCHEMES:
            for n in loads:
                cs = []
                for k in range(SEEDS):
                    seed = SEED_BASE + 1009 * k + 7 * n + (13 if name == "wn18rr" else 0) \
                        + (101 if scheme == "dense" else 0)
                    rng = np.random.RandomState(seed)
                    cs.append(Cell(sample(graphs[name], n, scheme, rng), rng))
                cells[(name, scheme, n)] = cs
    return cells


def plan(cells):
    """Tutte le celle con D scelto e previsione, calcolate senza costruire memorie."""
    rows = []
    for (name, scheme, n), cs in cells.items():
        ceil = float(np.mean([c.ceiling for c in cs]))
        d_abm_hi = None
        for t in TARGETS:
            for fam in FAMILIES:
                d = choose_dim(cs, fam, t * ceil, n)
                if fam == "MAP-B" and t == max(TARGETS):
                    d_abm_hi = d
                rows.append(_row(cs, name, scheme, n, "calib", t, fam, d, ceil))
        budget = d_abm_hi                          # bit di MAP-B al target alto
        for fam in FAMILIES:
            if fam == "MAP-B" or budget is None:
                continue
            bpc = bits_per_component(fam, n)
            if fam == "BSDC":
                d = (budget // bsdc_length(n)) * bsdc_length(n)
            else:
                d = budget // bpc
            rows.append(_row(cs, name, scheme, n, "equal_bits", None, fam, d, ceil, budget))
    return rows


def _row(cs, name, scheme, n, block, t, fam, d, ceil, budget=None):
    r = {"dataset": name, "scheme": scheme, "N": n, "block": block, "target": t,
         "family": fam, "D": d, "ceiling": ceil,
         "bits": None if d is None else d * bits_per_component(fam, n), "budget": budget,
         "M": float(np.mean([c.m for c in cs])),
         "queries": int(sum(len(c.queries) for c in cs)),
         "alias_share": float(np.mean([np.mean([len(a) > 0 for _g, a in c.sigs]) for c in cs])),
         "twin_share": float(np.mean([np.mean([w == 2 for w in c.weights.values()]) for c in cs]))}
    if d is None or d <= 0:
        r.update(pred=None, se=None, mdb=None)
        return r
    preds = [predict_cell(c, fam, d) for c in cs]
    flat = [p for ps in preds for p in ps]
    r["pred_seed"] = [float(np.mean(p)) for p in preds]
    r["pred"] = float(np.mean(r["pred_seed"]))
    r["se"] = se_cell(flat)
    r["mdb"] = 2.8 * r["se"]                      # α = 0.05 bilaterale, potenza 0.8
    return r


def run(cells, rows):
    for i, r in enumerate(rows):
        if r["pred"] is None:
            continue
        cs = cells[(r["dataset"], r["scheme"], r["N"])]
        accs = []
        for k, c in enumerate(cs):
            rng = np.random.RandomState(SEED_BASE + 31 * i + k)
            accs.append(float(np.mean(measure(c, r["family"], r["D"], rng))))
        r["meas_seed"] = accs
        r["meas"] = float(np.mean(accs))
        r["err"] = r["meas"] - r["pred"]
        r["z"] = r["err"] / r["se"] if r["se"] > 0 else None
        print(f"{r['dataset']:8s} {r['scheme']:7s} N={r['N']:4d} {r['block']:10s} "
              f"{str(r['target']):5s} {r['family']:5s} D={r['D']:6d} "
              f"prev {r['pred']:.3f} mis {r['meas']:.3f}", flush=True)
    return rows


def summarize(rows):
    def mae(sel):
        e = [abs(r["err"]) for r in sel if r.get("err") is not None]
        return (float(np.mean(e)) * 100, float(np.max(e)) * 100, len(e)) if e else None
    out = {}
    calib = [r for r in rows if r["block"] == "calib"]
    for fam in FAMILIES:
        for scheme in SCHEMES:
            for ds in DATASETS:
                sel = [r for r in calib if r["family"] == fam and r["scheme"] == scheme
                       and r["dataset"] == ds]
                out[f"{fam}|{scheme}|{ds}"] = mae(sel)
        zs = [r["z"] for r in calib if r["family"] == fam and r.get("z") is not None]
        out[f"{fam}|z>3"] = (sum(abs(z) > 3 for z in zs), len(zs))
    eq = [r for r in rows if r["block"] == "equal_bits" and r.get("meas") is not None]
    # H6: ordine fra famiglie a pari bit (MAP-B misurato nella sua cella calib alta)
    agree, total = 0, 0
    for key in {(r["dataset"], r["scheme"], r["N"]) for r in eq}:
        grp = [r for r in eq if (r["dataset"], r["scheme"], r["N"]) == key]
        grp += [r for r in calib if (r["dataset"], r["scheme"], r["N"]) == key
                and r["family"] == "MAP-B" and r["target"] == max(TARGETS)
                and r.get("meas") is not None]
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                if abs(a["pred"] - b["pred"]) >= 0.05:
                    total += 1
                    agree += (a["pred"] > b["pred"]) == (a["meas"] > b["meas"])
    out["H6_ranking"] = (agree, total)
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
    else:
        graphs = {n: Graph(load(n)) for n in DATASETS}
        cells = build_cells(graphs, DATASETS, LOADS)
    rows = plan(cells)
    if a.predict:
        for r in rows:
            p = "—" if r["pred"] is None else f"{r['pred']:.3f}"
            mdb = "—" if r["mdb"] is None else f"{100 * r['mdb']:.1f}"
            print(f"{r['dataset']:8s} {r['scheme']:7s} N={r['N']:4d} {r['block']:10s} "
                  f"t={str(r['target']):5s} {r['family']:5s} D={r['D']} bits={r['bits']} "
                  f"tetto={r['ceiling']:.3f} alias={r['alias_share']:.2f} "
                  f"gemelli={r['twin_share']:.2f} M={r['M']:.0f} prev={p} MDB={mdb}pt")
        PRED_OUT.parent.mkdir(exist_ok=True)
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
