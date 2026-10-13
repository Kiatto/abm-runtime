"""sizing_tool_prereg.py — preregistrazione 23: il dimensionatore `abm.sizing.recommend`
mantiene la sua promessa su grafi e sottografi mai usati per costruirlo o tararlo?

docs/preregistration/sizing_tool.md. Collauda il PRODOTTO (abm.sizing, abm.families), non
i modelli: non importa gli harness congelati delle 21-22 (ne copia solo campionamento e
caricamento dei dati, sotto) e non usa funzioni private di abm.sizing.

Per ogni sottografo nuovo (semi con offset 50_000, SEED_BASE della 23) e target:
1. `recommend(triples, target, queries=Q, verify=True, seeds=8, tol=0.01, rng_seed=…)`;
2. il "vero" di ogni raccomandazione: accuratezza media su TRUTH_SEEDS = 20 memorie FRESCHE
   (`families.make(f, D, seme)`, semi ≥ 2^31, disgiunti per costruzione da quelli della
   verifica, che `recommend` estrae in [0, 2^31)), sulle stesse query Q, al D finale e al
   D del modello (`dim_predicted`);
3. il D minimo misurato (H4): il più piccolo D sulla griglia della famiglia con media delle
   stesse 20 memorie fresche ≥ target (ricerca: dimezzamenti da D finale, poi bisezione).

Uso, dalla root:
    python examples/sizing_tool_prereg.py --smoke     # impianto intero, solo sintetico
    python examples/sizing_tool_prereg.py --predict   # previsioni dello strumento, NESSUNA memoria
    python examples/sizing_tool_prereg.py --timing    # stima del tempo (memorie solo sintetiche)
    python examples/sizing_tool_prereg.py             # il run preregistrato (NON eseguito qui)
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from abm import exact, families, sizing

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
OUT = ROOT / "results" / "sizing_tool_prereg_results.json"
PRED_OUT = ROOT / "results" / "sizing_tool_prereg_predictions.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
# (file, membro dell'archivio o None, sha256)
DATA = {
    "fb15k237": (EXT / "fb15k237_train.txt", None,
                 "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"),
    "wn18rr": (EXT / "wn18rr_train.csv", None,
               "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"),
    "codex-m": (EXT / "codex" / "codex-m_train.txt", None,
                "d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556"),
    "yago3-10": (EXT / "yago3_10" / "YAGO3-10.tar.gz", "train.txt",
                 "9a3bdd5fd08c064bd4b9ffdbe972f37ba0b8799f4e9f868ff70f5d2f5331fb92"),
    "umls": (EXT / "umls" / "umls.tar.gz", "train.txt",       # grafo nuovo (ConvE f3c0eb2)
             "26223aa91f0ec8db2324e0f489b9a4425e04d1d0df72ce5cfd62af4b91aee6fd"),
}
DATASETS = tuple(DATA)
SCHEMES = ("uniform", "dense")
LOADS = (100, 400, 1600)
TARGETS = (0.70, 0.85, 0.95)        # accuratezza assoluta (la semantica di recommend)
CEIL_MARGIN = 0.02                  # (sottografo, target) scartato se target > tetto − 0.02
FAMILIES = sizing.FAMILIES
K_SUB = 4                           # sottografi per (dataset, schema, N): vedi la riduzione
K_OFFSET = 50_000                   # indice dei sottografi: k = 50_000 … (mai usati)
Q_MAX = 200                         # query per sottografo (come le 21-22)
VERIFY_SEEDS = 8                    # default di recommend
TOL = 0.01                          # default di recommend
TRUTH_SEEDS = 20                    # memorie fresche per il "vero"
MAX_MEM = 1 << 30                   # max_mem_bytes (verifica) e limite delle misure del vero
BAND = 0.02                         # H2: target − 2 punti
ERR_BIG = 0.03                      # H3: |previsione − vero| > 3 punti
SEED_BASE = 23_000_023
RUN_BUDGET_H = 3.0
WORKERS = 6
# riduzione della griglia se la stima supera RUN_BUDGET_H (regola scritta, nel documento):
# passi in ordine, ci si ferma al primo sotto il budget; (K_SUB, target con D_min, N max D_min)
# (K_SUB, target con D_min, N max con D_min, max_mem_bytes per HRR)
REDUCTION = [(4, TARGETS, 1600, MAX_MEM), (4, (0.85,), 1600, MAX_MEM),
             (4, (0.85,), 400, MAX_MEM), (3, (0.85,), 400, MAX_MEM),
             (2, (0.85,), 400, MAX_MEM), (2, (0.85,), 400, 256 << 20),
             (2, (0.85,), 400, 128 << 20), (2, (0.85,), 400, 64 << 20)]
GRID_STEP = 6                       # scelto da --timing (stima 2.98 h; passi 0-5 oltre le 3 h)
DMIN_TARGETS, DMIN_MAX_N, HRR_MEM = TARGETS, 1600, MAX_MEM


def mem_cap(family):
    return HRR_MEM if family == "HRR" else MAX_MEM
# soglie delle ipotesi (vedi il documento)
H1_P, H1_FLOOR = 0.95, 0.90         # p̂ ≥ 0.95 e CP_inf ≥ 0.90; falsa se CP_sup < 0.95
H2_P, H2_FLOOR = 0.90, 0.85
H3_P, H3_FLOOR = 0.80, 0.60         # sensibilità
H3B_P, H3B_FLOOR = 0.50, 0.40       # specificità
H4_MED, H4_CI = 0.15, 0.20          # mediana ≤ 15 % e IC_sup della mediana ≤ 20 %
N_MIN_TEST = {"H1": 30, "H2": 30, "H3": 10, "H3b": 30, "H4": 30}
# -----------------------------------------------------------------------------


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    """Triple del train split; stesse convenzioni delle 21-22 (WN18RR: prefisso 'wn')."""
    path, member, sha = DATA[name]
    if not path.exists():
        raise SystemExit(f"{path} mancante: scaricalo come in examples/replicate.py (DATASETS)")
    got = sha256(path)
    if got != sha:
        raise SystemExit(f"{path.name}: sha256 {got}, atteso {sha}")
    if name == "wn18rr":
        rows = list(csv.reader(path.open()))[1:]
        return [(f"wn{h}", r, f"wn{t}") for h, r, t in rows]
    if member is None:
        lines = path.read_text().splitlines()
    else:
        with tarfile.open(path) as t:
            lines = t.extractfile(member).read().decode().splitlines()
    return [tuple(x.strip().split("\t")) for x in lines if x.strip()]


# ---- campionamento: copia di examples/vsa_families_prereg.py (Graph, sample) ----------

class Graph:
    def __init__(self, triples):
        self.triples = sorted({t for t in triples if t[0] != t[2]})   # niente self-loop
        self.adj = defaultdict(list)
        for i, (s, _r, o) in enumerate(self.triples):
            self.adj[s].append(i)
            self.adj[o].append(i)
        self.entities = sorted(self.adj)


def sample(g, n, scheme, rng):
    """N triple distinte. uniform: a caso. dense: BFS non orientata da un'entità a caso."""
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


def queries_of(triples, rng):
    keys = sorted({(s, r) for s, r, _o in triples})
    if len(keys) > Q_MAX:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), Q_MAX, replace=False))]
    return keys


def subgraphs(name, graph, loads=LOADS, k_sub=K_SUB, di=None):
    """Sottografi nuovi: SeedSequence([SEED_BASE, di, si, N, 50_000 + k]) → (campione e
    query, semi freschi del vero, rng_seed della verifica)."""
    di = DATASETS.index(name) if di is None else di
    out = []
    for si, scheme in enumerate(SCHEMES):
        for n in loads:
            if n > len(graph.triples) // 2:
                continue
            for k in range(K_OFFSET, K_OFFSET + k_sub):
                samp, fresh, ver = np.random.SeedSequence([SEED_BASE, di, si, n, k]).spawn(3)
                rng = np.random.default_rng(samp)
                triples = sample(graph, n, scheme, rng)
                q = queries_of(triples, rng)
                fseeds = [int(x) + (1 << 31) for x in np.random.default_rng(fresh).choice(
                    1 << 31, TRUTH_SEEDS, replace=False)]
                out.append({"dataset": name, "scheme": scheme, "N": n, "k": k,
                            "triples": triples, "queries": q, "fresh": fseeds,
                            "rng_seed": int(ver.generate_state(1)[0])})
    return out


# ---- controlli positivi sintetici (casi del revisore, reference/tests/test_sizing.py) ----

def _base(n_facts, seed=0):
    rng = np.random.default_rng(seed)
    base = set()
    while len(base) < n_facts:
        s, o = rng.integers(5000, size=2)
        if s != o:
            base.add((f"e{s}", f"r{rng.integers(20)}", f"e{o}"))
    return base


def controls():
    """club: ogni giocatore ha due relazioni verso lo stesso club (MAP-B ottimista di ~33
    punti); inverse: 50 coppie (p, a, q) + (q, b, p) su 100 fatti uniformi; three: 40
    giocatori con tre relazioni verso 3 club su 80 fatti uniformi."""
    club = sorted({(f"p{i}", r, f"c{i % 2}") for i in range(100) for r in ("r1", "r2")})
    base = _base(100)
    inverse = base | {t for i in range(50) for t in [(f"p{i}", "a", f"q{i}"),
                                                       (f"q{i}", "b", f"p{i}")]}
    three = set(sorted(base)[:80]) | {(f"p{i}", r, f"c{i % 3}") for i in range(40)
                                      for r in ("a", "b", "c")}
    out = []
    for j, (name, t) in enumerate((("club", club), ("inverse", sorted(inverse)),
                                    ("three", sorted(three)))):
        fresh, ver = np.random.SeedSequence([SEED_BASE, 99, j]).spawn(2)
        q = sorted({(s, r) for s, r, _o in t})
        fseeds = [int(x) + (1 << 31) for x in np.random.default_rng(fresh).choice(
            1 << 31, TRUTH_SEEDS, replace=False)]
        out.append({"dataset": f"control:{name}", "scheme": "synthetic", "N": len(t), "k": 0,
                    "triples": t, "queries": q, "fresh": fseeds,
                    "rng_seed": int(ver.generate_state(1)[0])})
    return out


# ---- previsioni (nessuna memoria) ------------------------------------------------------

def struct(sg):
    g = sizing._Graph(sg["triples"], sg["queries"]) if hasattr(sizing, "_Graph") else None
    ceiling = exact.ceiling(sg["triples"], sg["queries"])
    out = {"M": len({x for t in sg["triples"] for x in t}), "Q": len(sg["queries"]),
           "ceiling": float(ceiling)}
    if g is not None:      # solo descrittivo (regime e statistiche), non usato per decidere
        out.update(regime=g.regime, alias_share=g.alias_share, twin_share=g.twin_share,
                   **g.stats)
    return out


def targets_of(ceiling):
    """Regola scritta: un target entra se target ≤ tetto degli alias − CEIL_MARGIN."""
    return [t for t in TARGETS if t <= ceiling - CEIL_MARGIN]


def predict_sub(sg):
    st = struct(sg)
    rows = []
    for t in TARGETS:
        base = {k: sg[k] for k in ("dataset", "scheme", "N", "k")}
        base.update(target=t, **st)
        if t not in targets_of(st["ceiling"]):
            rows.append({**base, "family": None, "discarded": "ceiling"})
            continue
        recs = sizing.recommend(sg["triples"], t, queries=sg["queries"])
        got = {r.family for r in recs}
        for r in recs:
            rows.append({**base, "family": r.family, "dim_predicted": r.dim, "bits": r.bits,
                         "predicted": r.predicted, "reliability": r.reliability,
                         "n_warnings": len(r.warnings), "discarded": None})
        for f in FAMILIES:
            if f not in got:
                rows.append({**base, "family": f, "discarded": "unreachable (d_max)"})
    return rows


# ---- misure ------------------------------------------------------------------------

def step_of(family, n):
    return families.bsdc_length(n) if family.startswith("BSDC") else sizing.STEP


class Truth:
    """Accuratezze delle memorie fresche per D (memoizzate): il "vero"."""

    def __init__(self, sg):
        self.sg = sg
        self.objects = defaultdict(set)
        for s, r, o in sg["triples"]:
            self.objects[(s, r)].add(o)
        self.m = len({x for t in sg["triples"] for x in t})
        self.memo = {}

    def fits(self, family, d):
        return families.estimate_bytes(family, d, n_facts=len(self.sg["triples"]),
                                       n_symbols=self.m,
                                       n_queries=len(self.sg["queries"])) <= mem_cap(family)

    def __call__(self, family, d, seeds=None):
        key = (family, d)
        if key not in self.memo:
            if not self.fits(family, d):
                self.memo[key] = None
            else:
                q = self.sg["queries"]
                accs = []
                for seed in (seeds or self.sg["fresh"]):
                    mem = families.make(family, d, seed)
                    mem.store(self.sg["triples"])
                    ans = mem.query_many(q)
                    accs.append(sum(o in self.objects[x] for x, (o, _c) in zip(q, ans)) / len(q))
                self.memo[key] = accs
        return self.memo[key]


def mean_or_none(a):
    return None if a is None else float(np.mean(a))


def d_min_measured(truth, family, target, d_hi, n):
    """Il più piccolo D della griglia con media delle memorie fresche ≥ target. Da d_hi:
    se non basta, raddoppio (fino a 8·d_hi o al limite di memoria); poi dimezzamenti fino
    al primo D che non basta; poi bisezione al passo. Monotonia assunta (rumore: 20 memorie)."""
    step = step_of(family, n)
    ok = lambda d: (lambda a: a is not None and np.mean(a) >= target)(truth(family, d))
    hi = d_hi
    while not ok(hi):
        if hi >= 8 * d_hi or truth(family, 2 * hi) is None:
            return None
        hi *= 2
    lo = hi
    while lo > step:
        lo = max(step, (lo // 2) // step * step)
        if not ok(lo):
            break
        hi = lo
    else:
        return hi
    while hi - lo > step:
        mid = (lo // step + hi // step) // 2 * step
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def run_sub(sg, truth_seeds=None, dmin=True):
    """Tutte le misure di un sottografo (o di un controllo)."""
    if truth_seeds is not None:
        sg = {**sg, "fresh": sg["fresh"][:truth_seeds]}
    st = struct(sg)
    truth = Truth(sg)
    rows = []
    for t in targets_of(st["ceiling"]):
        recs = [r for fams in ([f for f in FAMILIES if f != "HRR"], ["HRR"])
                for r in sizing.recommend(sg["triples"], t, families=fams,
                                          queries=sg["queries"], verify=True,
                                          seeds=VERIFY_SEEDS, tol=TOL, rng_seed=sg["rng_seed"],
                                          max_mem_bytes=mem_cap(fams[0]))]
        for r in recs:
            n = len(sg["triples"])
            pred_at_model = sizing.predict(sg["triples"], r.family, r.dim_predicted,
                                           queries=sg["queries"])
            tp = truth(r.family, r.dim_predicted)
            tf = truth(r.family, r.dim)
            row = {k: sg[k] for k in ("dataset", "scheme", "N", "k")}
            row.update(target=t, family=r.family, reliability=r.reliability,
                       warnings=list(r.warnings), dim_predicted=r.dim_predicted, dim=r.dim,
                       adjusted=r.adjusted, verified=r.verified, measured=r.measured,
                       selection_measured=r.selection_measured,
                       predicted_at_model=float(pred_at_model),
                       truth_model=mean_or_none(tp), truth_final=mean_or_none(tf),
                       truth_final_accs=tf, truth_model_accs=tp, **st)
            want = dmin and t in DMIN_TARGETS and n <= DMIN_MAX_N
            row["d_min"] = (d_min_measured(truth, r.family, t, r.dim, n)
                            if want and tf is not None else None)
            rows.append(row)
    return rows


# ---- statistica ---------------------------------------------------------------------

def _logpmf(n, x):
    k = np.arange(n + 1)
    lg = np.array([math.lgamma(j + 1) for j in range(n + 1)])
    with np.errstate(divide="ignore"):
        return lg[n] - lg - lg[::-1] + k * np.log(x) + (n - k) * np.log1p(-x)


def _upper_tail(k, n, x):
    """P(Bin(n, x) ≥ k)."""
    if k <= 0:
        return 1.0
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return float(np.exp(_logpmf(n, x)[k:]).sum())


def _bisect(f, lo=0.0, hi=1.0):
    """x in [lo, hi] con f crescente che attraversa 0."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp(k, n, alpha=0.05):
    """Limiti di Clopper-Pearson UNILATERALI al 95 %: (inferiore, superiore).
    inf: P(Bin(n, inf) ≥ k) = alpha; sup: P(Bin(n, sup) ≤ k) = alpha."""
    lo = 0.0 if k == 0 else _bisect(lambda x: _upper_tail(k, n, x) - alpha)
    hi = 1.0 - _bisect(lambda y: _upper_tail(n - k, n, y) - alpha) if k < n else 1.0
    return lo, hi


def verdict_prop(k, n, p_claim, floor, n_min):
    if n < n_min:
        return {"k": k, "n": n, "verdict": "non testabile"}
    lo, hi = cp(k, n)
    p = k / n
    v = ("sostenuta" if p >= p_claim and lo >= floor else
         "falsificata" if hi < p_claim else "non conclusiva")
    return {"k": k, "n": n, "p": p, "cp_inf": lo, "cp_sup": hi, "verdict": v}


def power_prop(n, p_true, p_claim, floor, alpha=0.05):
    """P(sostenuta) e P(falsificata) esatte (binomiale) per n unità indipendenti.
    CP_inf ≥ floor  ⟺  P(Bin(n, floor) ≥ k) ≤ alpha;  CP_sup < p  ⟺  P(Bin(n, p) ≤ k) < alpha."""
    if n == 0:
        return 0.0, 0.0
    pmf = np.exp(_logpmf(n, p_true))
    up_floor = np.cumsum(np.exp(_logpmf(n, floor))[::-1])[::-1]     # P(X ≥ k)
    low_claim = np.cumsum(np.exp(_logpmf(n, p_claim)))              # P(X ≤ k)
    k = np.arange(n + 1)
    sup_k = (k / n >= p_claim) & (up_floor <= alpha)
    fal_k = low_claim < alpha
    return float(pmf[sup_k].sum()), float(pmf[fal_k & ~sup_k].sum())


def median_ci(x):
    """IC 95 % distribution-free della mediana (statistiche d'ordine, binomiale 1/2)."""
    x = sorted(x)
    n = len(x)
    cdf, j = 0.0, 0
    while j < n:
        cdf += math.comb(n, j) / 2 ** n
        if cdf > 0.025:
            break
        j += 1
    return x[max(j - 1, 0)], x[min(n - j, n - 1)]


def summarize(rows):
    real = [r for r in rows if not r["dataset"].startswith("control:")]
    out = {}
    # H1: validated + verified, vero al D finale ≥ target − tol
    u = [r for r in real if r["reliability"] == "validated" and r["verified"]
         and r["truth_final"] is not None]
    k = sum(r["truth_final"] >= r["target"] - TOL for r in u)
    out["H1"] = verdict_prop(k, len(u), H1_P, H1_FLOOR, N_MIN_TEST["H1"])
    # H1 descrittiva: tutte le verificate, qualunque affidabilità
    u = [r for r in real if r["verified"] and r["truth_final"] is not None]
    out["H1_all_verified"] = {"k": sum(r["truth_final"] >= r["target"] - TOL for r in u),
                              "n": len(u)}
    # H2: validated, D del modello, vero ≥ target − BAND
    u = [r for r in real if r["reliability"] == "validated" and r["truth_model"] is not None]
    k = sum(r["truth_model"] >= r["target"] - BAND for r in u)
    out["H2"] = verdict_prop(k, len(u), H2_P, H2_FLOOR, N_MIN_TEST["H2"])
    # H3: sensibilità e specificità degli avvisi su |errore del modello| > 3 punti
    u = [r for r in real if r["truth_model"] is not None]
    pos = [r for r in u if abs(r["predicted_at_model"] - r["truth_model"]) > ERR_BIG]
    neg = [r for r in u if abs(r["predicted_at_model"] - r["truth_model"]) <= ERR_BIG]
    out["H3"] = verdict_prop(sum(r["reliability"] != "validated" for r in pos), len(pos),
                             H3_P, H3_FLOOR, N_MIN_TEST["H3"])
    out["H3b"] = verdict_prop(sum(r["reliability"] == "validated" for r in neg), len(neg),
                              H3B_P, H3B_FLOOR, N_MIN_TEST["H3b"])
    opt = [r for r in pos if r["predicted_at_model"] > r["truth_model"]]
    out["H3_optimistic_only"] = {"k": sum(r["reliability"] != "validated" for r in opt),
                                 "n": len(opt)}
    # H4: gonfiamento di D delle raccomandazioni verificate rispetto al minimo misurato
    u = [r["dim"] / r["d_min"] - 1 for r in real if r["verified"] and r.get("d_min")]
    if len(u) < N_MIN_TEST["H4"]:
        out["H4"] = {"n": len(u), "verdict": "non testabile"}
    else:
        med = statistics.median(u)
        lo, hi = median_ci(u)
        v = ("sostenuta" if med <= H4_MED and hi <= H4_CI else
             "falsificata" if lo > H4_MED else "non conclusiva")
        out["H4"] = {"n": len(u), "median": med, "ci": [lo, hi], "verdict": v}
    out["H4_model_D"] = [r["dim_predicted"] / r["d_min"] - 1 for r in real if r.get("d_min")]
    if out["H4_model_D"]:
        out["H4_model_D"] = {"n": len(out["H4_model_D"]),
                             "median": statistics.median(out["H4_model_D"])}
    out["controls"] = control_checks([r for r in rows if r["dataset"].startswith("control:")])
    return out


def control_checks(rows):
    """Controlli positivi (non entrano nelle ipotesi): MAP-B mai 'validated'; MAP-B sul club
    con errore del modello > 3 punti e segnalato; ogni raccomandazione verificata raggiunge
    target − tol sulle memorie fresche."""
    out = {}
    for name in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == name]
        mb = [r for r in rs if r["family"] == "MAP-B"]
        out[name] = {
            "mapb_not_validated": all(r["reliability"] != "validated" for r in mb) and bool(mb),
            "mapb_model_error": [r["predicted_at_model"] - r["truth_model"] for r in mb
                                 if r["truth_model"] is not None],
            "verified_reach": [(r["family"], r["target"], r["truth_final"]) for r in rs
                               if r["verified"]],
            "verified_all_reach": all(r["truth_final"] >= r["target"] - TOL for r in rs
                                      if r["verified"] and r["truth_final"] is not None)}
    return out


def grid_rows(pred, step):
    """Le raccomandazioni previste che il run misura con la griglia REDUCTION[step]."""
    ksub, _dts, _dn, hm = REDUCTION[step]
    return [r for r in pred if r["k"] - K_OFFSET < ksub and not (
        r["family"] == "HRR" and not r["discarded"] and families.estimate_bytes(
            "HRR", r["dim_predicted"], n_facts=r["N"], n_symbols=r["M"],
            n_queries=r["Q"]) > hm)]


def power_table(pred):
    """Potenza dalle previsioni (n delle unità note prima dei dati)."""
    ok = [r for r in pred if r["family"] and not r["discarded"]]
    n_val = sum(r["reliability"] == "validated" for r in ok)
    tab = {"n_recs": len(ok), "n_validated": n_val}
    for h, (pc, fl) in {"H1": (H1_P, H1_FLOOR), "H2": (H2_P, H2_FLOOR)}.items():
        tab[h] = {f"p={p}": dict(zip(("P_sostenuta", "P_falsificata"),
                                     power_prop(n_val, p, pc, fl)))
                  for p in (0.99, 0.98, 0.97, 0.95, 0.90, 0.85)}
    tab["H3"] = {f"n_pos={n},sens={s}": dict(zip(("P_sostenuta", "P_falsificata"),
                                                 power_prop(n, s, H3_P, H3_FLOOR)))
                 for n in (10, 20, 40) for s in (0.95, 0.9, 0.7, 0.5)}
    return tab


# ---- tempo (solo memorie sintetiche) ----------------------------------------------

def _syn(n, n_ent, seed=0):
    rng = np.random.default_rng(seed)
    fs = set()
    while len(fs) < n:
        s, o = rng.integers(n_ent, size=2)
        if s != o:
            fs.add((f"e{s}", f"r{rng.integers(20)}", f"e{o}"))
    return sorted(fs)


def timing(pred):
    """Per (N, famiglia): una memoria (store + 200 query) su un grafo SINTETICO con lo
    stesso N e M ≈ mediana dei sottografi reali, a tre D (min, mediana, max previsti);
    interpolazione lineare in D. Memorie attese per raccomandazione (dichiarato):
    16 di verifica + 16 di correzione se non 'validated' + 20 al D del modello + 20 al D
    finale se corretto (contate come 20·(non validated)) + D_min: (2 + log2(D/2·passo))
    valutazioni da 20 a ~0.75·D. Secondi totali / WORKERS."""
    ok = [r for r in pred if r["family"] and not r["discarded"]]
    per_row, by = [], defaultdict(float)
    for n in sorted({r["N"] for r in ok}):
        for f in FAMILIES:
            rs = [r for r in ok if r["N"] == n and r["family"] == f]
            if not rs:
                continue
            m = int(np.median([r["M"] for r in rs]))
            t = _syn(n, max(m // 2, 20))
            q = sorted({(s, r) for s, r, _o in t})[:Q_MAX]
            ds = sorted({r["dim_predicted"] for r in rs})
            pts = sorted({ds[0], ds[len(ds) // 2], ds[-1]})
            tt = []
            for d in pts:
                if families.estimate_bytes(f, d, n_facts=n, n_symbols=m, n_queries=len(q)) \
                        > MAX_MEM:
                    tt.append(None)
                    continue
                t0 = time.time()
                mem = families.make(f, d, 1)
                mem.store(t)
                mem.query_many(q)
                tt.append(time.time() - t0)
            xs = [d for d, s in zip(pts, tt) if s is not None]
            ys = [s for s in tt if s is not None]
            tf = (lambda d: float(np.interp(d, xs, ys))) if xs else (lambda d: 0.0)
            for r in rs:
                d = r["dim_predicted"]
                nv = r["reliability"] != "validated"
                mems = 16 + 16 * nv + 20 + 20 * nv
                evals = 2 + max(0, math.ceil(math.log2(max(d / (2 * step_of(f, n)), 1))))
                base, dm = mems * tf(d), evals * 20 * tf(0.75 * d)
                big = families.estimate_bytes(f, d, n_facts=n, n_symbols=r["M"],
                                              n_queries=r["Q"])
                per_row.append((r["k"] - K_OFFSET, r["target"], n, base, dm, f, big))
                by[(n, f)] += base + dm
            print(f"  N={n} {f:7s} M≈{m} D={pts} s/memoria={['—' if s is None else f'{s:.2f}' for s in tt]}"
                  f" → {by[(n, f)] / 60:.1f} min (seriale)", flush=True)
    steps = []
    for i, (ksub, dts, dn, hm) in enumerate(REDUCTION):
        tot = sum(b + (dm if (t in dts and n <= dn) else 0.0)
                  for k, t, n, b, dm, f, big in per_row
                  if k < ksub and not (f == "HRR" and big > hm)) / WORKERS
        skip = sum(1 for k, t, n, b, dm, f, big in per_row
                   if k < ksub and f == "HRR" and big > hm)
        steps.append((i, ksub, dts, dn, hm, tot))
        print(f"  passo {i}: K={ksub} D_min su target {dts} e N ≤ {dn}, HRR ≤ {hm >> 20} MiB "
              f"({skip} racc. HRR non misurate): {tot / 3600:.2f} h")
    return steps, {f"{n}:{f}": v for (n, f), v in by.items()}


# ---- main ---------------------------------------------------------------------------

def _pool_map(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(workers) as p:
        return p.map(fn, items, chunksize=1)


def _run_full(sg):
    return run_sub(sg)


def smoke():
    """Impianto intero su grafi SINTETICI (nessun dato reale): sottografi, previsioni,
    verifica, vero, D minimo, statistiche, controlli (con meno memorie fresche)."""
    g = Graph(_syn(3000, 400, seed=7))
    sgs = subgraphs("smoke", g, loads=(100, 400), k_sub=1, di=9)
    pred = [r for sg in sgs for r in predict_sub(sg)]
    assert all(r["discarded"] or r["dim_predicted"] for r in pred)
    rows = []
    for sg in sgs[:2] + controls():
        rows += run_sub(sg, truth_seeds=6)
    for r in rows:
        if r["dim"] != r["dim_predicted"]:
            assert r["adjusted"]
        if r["d_min"] is not None:
            assert r["d_min"] % step_of(r["family"], r["N"]) == 0
    # semi freschi disgiunti da quelli della verifica (che recommend estrae in [0, 2^31))
    assert all(s >= 1 << 31 for sg in sgs for s in sg["fresh"])
    assert cp(95, 100)[0] < 0.95 < cp(100, 100)[1] + 1e-12
    summ = summarize(rows)
    for r in rows:
        print(f"{r['dataset']:16s} {r['scheme']:9s} N={r['N']:4d} t={r['target']} "
              f"{r['family']:7s} {r['reliability']:11s} D={r['dim_predicted']}→{r['dim']} "
              f"ver={r['verified']} prev={r['predicted_at_model']:.3f} "
              f"vero(model)={r['truth_model']} vero(fin)={r['truth_final']} Dmin={r['d_min']}")
    print(json.dumps(summ, indent=1, default=str))
    print(json.dumps(power_table(pred), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="impianto intero, solo sintetico")
    ap.add_argument("--predict", action="store_true", help="previsioni, nessuna memoria")
    ap.add_argument("--timing", action="store_true", help="stima del tempo (sintetico)")
    ap.add_argument("--power", action="store_true",
                    help="potenza sulla griglia GRID_STEP, scritta nelle previsioni")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        smoke()
        print(f"{time.time() - t0:.0f} s")
        return
    if a.power:
        d = json.loads(PRED_OUT.read_text())
        d["grid_step"], d["power_grid"] = GRID_STEP, power_table(grid_rows(d["rows"], GRID_STEP))
        PRED_OUT.write_text(json.dumps(d, indent=1))
        print(json.dumps(d["power_grid"], indent=1))
        return
    if a.timing:
        pred = json.loads(PRED_OUT.read_text())["rows"]
        steps, by = timing(pred)
        ok = [st for st in steps if st[-1] <= RUN_BUDGET_H * 3600]
        print(f"budget {RUN_BUDGET_H} h, {WORKERS} processi: passo scelto "
              f"{ok[0][0] if ok else len(steps) - 1}")
        return
    graphs = {n: Graph(load(n)) for n in DATASETS}
    if a.predict:
        sgs = [sg for n, g in graphs.items() for sg in subgraphs(n, g)]
        pred = _pool_map(predict_sub, sgs, a.workers)
        pred = [r for rs in pred for r in rs]
        PRED_OUT.write_text(json.dumps({
            "sha256": {n: DATA[n][2] for n in DATASETS}, "K_SUB": K_SUB, "rows": pred},
            indent=1))
        PRED_OUT.write_text(json.dumps({
            "sha256": {n: DATA[n][2] for n in DATASETS}, "K_SUB": K_SUB, "rows": pred,
            "power": power_table(pred)}, indent=1))
        print(json.dumps(power_table(pred), indent=1))
        print(f"{time.time() - t0:.0f} s")
        return
    global DMIN_TARGETS, DMIN_MAX_N, HRR_MEM
    k_sub, DMIN_TARGETS, DMIN_MAX_N, HRR_MEM = REDUCTION[GRID_STEP]
    sgs = [sg for n, g in graphs.items() for sg in subgraphs(n, g, k_sub=k_sub)]
    rows = [r for rs in _pool_map(_run_full, sgs + controls(), a.workers) for r in rs]
    summ = summarize(rows)
    OUT.write_text(json.dumps({"rows": rows, "summary": summ, "grid_step": GRID_STEP,
                               "reduction": REDUCTION[GRID_STEP]}, indent=1))
    print(json.dumps(summ, indent=1, default=str))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
