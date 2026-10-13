"""sizing_tool_prereg.py — preregistrazione 23: il dimensionatore `abm.sizing.recommend`
mantiene la sua promessa su sottografi e grafi mai usati per costruirlo o tararlo?

docs/preregistration/sizing_tool.md (versione dopo la revisione ostile del disegno).
Collauda il PRODOTTO (abm.sizing, abm.families) con la sola API pubblica (recommend,
predict, bits_per_component, STEP, FAMILIES; families.make, estimate_bytes, bsdc_length).
Eccezioni dichiarate: `sizing._Graph` per statistiche solo descrittive, e
`sizing._predict_graph` sostituito (monkeypatch) SOLO nel controllo "modello rotto" dello
smoke. Non importa gli harness congelati (copia campionamento e caricamento dei dati).

Per ogni sottografo e target (semi derivati da (sottografo, target)):
1. `recommend(triples, target, queries=Q, verify=True, seeds=8, tol=0.01, rng_seed=…)`
   (HRR in una chiamata a parte, con il suo max_mem_bytes);
2. il vero: accuratezza media su 20 memorie FRESCHE su TUTTE le chiavi (s, r) del
   sottografo (primario) e sulle 200 query Q della verifica (secondario), al D finale e al
   D del modello;
3. D minimo misurato (H4): il più piccolo D della griglia accettato con lo STESSO criterio
   della verifica (IC_inf 95 % ≥ target − tol) su altre 20 memorie (semi indipendenti dal vero);
4. baseline empirica (H7): ricerca di D senza modello, stesso criterio e stessi 8+8 semi
   di selezione/report, partendo da D = 256.

Uso, dalla root:
    python examples/sizing_tool_prereg.py --smoke     # impianto intero, solo sintetico
    python examples/sizing_tool_prereg.py --predict   # previsioni dello strumento, NESSUNA memoria
    python examples/sizing_tool_prereg.py --timing    # stima del tempo (memorie sintetiche, 3 ripetizioni)
    python examples/sizing_tool_prereg.py --power     # potenza per cluster dalle previsioni
    python examples/sizing_tool_prereg.py             # il run preregistrato (NON eseguito qui)
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from abm import exact, families, sizing

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
OUT = ROOT / "results" / "sizing_tool_prereg_results.json"
CKPT = ROOT / "results" / "sizing_tool_prereg_results.jsonl"
PRED_OUT = ROOT / "results" / "sizing_tool_prereg_predictions.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DATA = {   # (file, membro dell'archivio o None, sha256)
    "fb15k237": (EXT / "fb15k237_train.txt", None,
                 "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"),
    "wn18rr": (EXT / "wn18rr_train.csv", None,
               "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"),
    "codex-m": (EXT / "codex" / "codex-m_train.txt", None,
                "d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556"),
    "yago3-10": (EXT / "yago3_10" / "YAGO3-10.tar.gz", "train.txt",
                 "9a3bdd5fd08c064bd4b9ffdbe972f37ba0b8799f4e9f868ff70f5d2f5331fb92"),
    "umls": (EXT / "umls" / "umls.tar.gz", "train.txt",          # mai visto (ConvE f3c0eb2)
             "26223aa91f0ec8db2324e0f489b9a4425e04d1d0df72ce5cfd62af4b91aee6fd"),
    "kinship": (EXT / "kinship" / "kinship.tar.gz", "train.txt",  # mai visto (ConvE f3c0eb2)
                "a3cb9ca67213eb38133aebc12901195e283d844ebd59f1a6cbf2d4fa12ced4e6"),
}
DATASETS = tuple(DATA)
IN_DIST = ("fb15k237", "wn18rr", "codex-m", "yago3-10")   # grafi della calibrazione
UNSEEN = ("umls", "kinship")                               # grafi mai visti
SCHEMES = ("uniform", "dense")
LOADS = (100, 400, 1600)
TARGETS = (0.70, 0.85, 0.95)        # accuratezza assoluta (la semantica di recommend)
CEIL_MARGIN = 0.02                  # (sottografo, target) scartato se target > tetto − 0.02
FAMILIES = sizing.FAMILIES
K_OFFSET = 50_000                   # indice dei sottografi: k = 50_000 … (mai usati)
K_PRED = {"in": 8, "unseen": 40, "unseen_1600": 8}   # sottografi con previsioni per cella
Q_MAX = 200                         # query della verifica per sottografo (come le 21-22)
VERIFY_SEEDS = 8                    # default di recommend
TOL = 0.01                          # default di recommend
TRUTH_SEEDS = 20                    # memorie fresche per il vero
DMIN_SEEDS = 20                     # altre memorie fresche per il D minimo (H4)
BASE_D0 = 256                       # baseline empirica (H7): D di partenza
D_MAX = sizing.D_MAX
MAX_MEM = 1 << 30                   # max_mem_bytes per le famiglie non HRR
BAND = 0.02                         # H2: target − 2 punti
ERR_BIG = 0.03                      # H3/H6: |previsione − vero| > 3 punti
SEED_BASE = 23_000_023
BOOT_B, BOOT_SEED = 10_000, 23_023  # bootstrap per cluster (sottografo)
RUN_BUDGET_H = 10.0                 # la stima sceglie il passo (mai tronca)
HARD_CAP_H = 12.0                   # tetto rigido: si analizza il prefisso di cluster interi
TIMING_REPEATS = 3                  # --timing: massimo su 3 ripetizioni
WORKERS = 6
MIN_CLUSTERS = 30                   # H1, H2, H1u, H2u, H8u, H4, H5, H7
# soglie (vedi il documento)
H1 = (0.95, 0.90)                   # p̂ ≥ 0.95 e bootstrap_inf ≥ 0.90; falsa se boot_sup < 0.95
H2 = (0.90, 0.85)
H3 = (0.80, 0.60)                   # sensibilità; minimo 10 positivi in ≥ 5 cluster
H5 = (0.50, 0.40)                   # copertura
H6 = (3.0, 1.5)                     # risk ratio ≥ 3 e boot_inf ≥ 1.5; falsa se boot_sup < 3
H4 = (0.15, 0.20)                   # mediana ≤ 15 %, boot_sup ≤ 20 %; falsa se boot_inf > 15 %
H7 = (0.50, 0.60, 1.10)             # memorie: mediana del rapporto ≤ 0.5 (boot_sup ≤ 0.6);
                                    # bit: mediana del rapporto ≤ 1.10 (boot_sup ≤ 1.10)
# Griglia: passi in ordine, la stima di --timing sceglie il primo sotto RUN_BUDGET_H.
# k_in: sottografi base per cella (grafi di calibrazione); k_val: sottografi con almeno una
# raccomandazione validated prevista per cella, aggiunti in ordine di k fino a k_val;
# k_un: sottografi base per cella dei grafi mai visti; k_val_un: come k_val sui mai visti;
# dmin/h7: (target, N massimo) in cui si misurano D minimo e baseline; hrr: max_mem di HRR.
REDUCTION = [
    dict(k_in=4, k_val=6, k_un=8, k_val_un=30, dmin=(TARGETS, 1600), h7=((0.85,), 400), hrr=1 << 30),
    dict(k_in=4, k_val=6, k_un=8, k_val_un=30, dmin=((0.85,), 400), h7=((0.85,), 400), hrr=1 << 30),
    dict(k_in=4, k_val=6, k_un=8, k_val_un=30, dmin=((0.85,), 400), h7=((0.85,), 100), hrr=256 << 20),
    dict(k_in=3, k_val=4, k_un=6, k_val_un=30, dmin=((0.85,), 400), h7=((0.85,), 100), hrr=256 << 20),
    dict(k_in=3, k_val=4, k_un=6, k_val_un=30, dmin=((0.85,), 400), h7=((0.85,), 100), hrr=128 << 20),
    dict(k_in=2, k_val=4, k_un=4, k_val_un=30, dmin=((0.85,), 400), h7=((0.85,), 100), hrr=128 << 20),
    dict(k_in=2, k_val=3, k_un=4, k_val_un=30, dmin=((0.85,), 100), h7=((0.85,), 100), hrr=64 << 20),
]
GRID_STEP = 2                       # scelto da --timing (stima 8.64 h ≤ 10 h; passi 0-1 oltre)
# -----------------------------------------------------------------------------

CFG = dict(REDUCTION[0])            # configurazione attiva: SOLO via configure()


def configure(step):
    """Applica il passo della griglia (usato da run, smoke, timing, power: stessa via)."""
    CFG.clear()
    CFG.update(REDUCTION[step], step=step)


def mem_cap(family):
    return CFG["hrr"] if family == "HRR" else MAX_MEM


def excluded(family, d, n, m, q):
    """La stessa stima di memoria della verifica di recommend (families.estimate_bytes con
    n_queries = |Q|): True se la memoria a D non si costruisce né nella verifica né nel vero."""
    return families.estimate_bytes(family, d, n_facts=n, n_symbols=m, n_queries=q) > mem_cap(family)


def _jdef(o):
    """numpy → tipi nativi per json."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


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


def keys_of(triples):
    return sorted({(s, r) for s, r, _o in triples})


def seeds_of(seq, n, offset):
    """n semi distinti in [offset, offset + 2^31): insiemi disgiunti per offset diversi
    (verifica di recommend: [0, 2^31); vero: 2^31; D minimo: 2^32; baseline: 2^33)."""
    return [int(x) + offset for x in np.random.default_rng(seq).choice(1 << 31, n, replace=False)]


def make_sub(name, di, si, scheme, n, k, triples, rng=None):
    q = keys_of(triples)
    if rng is not None and len(q) > Q_MAX:
        q = [q[i] for i in sorted(rng.choice(len(q), Q_MAX, replace=False))]
    return {"dataset": name, "di": di, "si": si, "scheme": scheme, "N": n, "k": k,
            "key": f"{name}|{scheme}|{n}|{k}", "triples": triples, "queries": q}


def subgraph(name, graph, si, n, k, di=None):
    di = DATASETS.index(name) if di is None else di
    samp = np.random.SeedSequence([SEED_BASE, di, si, n, k]).spawn(1)[0]
    rng = np.random.default_rng(samp)
    triples = sample(graph, n, SCHEMES[si], rng)
    return make_sub(name, di, si, SCHEMES[si], n, k, triples, rng)


def target_seeds(sg, ti):
    """Semi derivati da (sottografo, target): verifica, vero, D minimo, baseline (sel, rep)."""
    ver, tru, dmn, bas = np.random.SeedSequence(
        [SEED_BASE, sg["di"], sg["si"], sg["N"], sg["k"], 1000 + ti]).spawn(4)
    b = seeds_of(bas, 2 * VERIFY_SEEDS, 1 << 33)
    return {"rng_seed": int(ver.generate_state(1)[0]), "truth": seeds_of(tru, TRUTH_SEEDS, 1 << 31),
            "dmin": seeds_of(dmn, DMIN_SEEDS, 1 << 32), "base_sel": b[:VERIFY_SEEDS],
            "base_rep": b[VERIFY_SEEDS:]}


# ---- controlli sintetici --------------------------------------------------------------

def _uniform(n_facts, n_ent=5000, n_rel=20, seed=0):
    rng = np.random.default_rng(seed)
    base = set()
    while len(base) < n_facts:
        s, o = rng.integers(n_ent, size=2)
        if s != o:
            base.add((f"e{s}", f"r{rng.integers(n_rel)}", f"e{o}"))
    return base


ADV_N = 6


def adversarial_params():
    """Parametri dei controlli avversari, estratti ORA da un seme fisso (registrati nelle
    previsioni): hub Zipf, molti oggetti per (s, r), simmetriche (gemelli e alias),
    multi-relazione 30-60 %."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, 777]))
    out = []
    for j in range(ADV_N):
        out.append({"n_facts": int(rng.integers(150, 401)), "n_ent": int(rng.integers(60, 401)),
                    "n_rel": int(rng.integers(3, 13)), "zipf": float(rng.uniform(1.1, 2.0)),
                    "p_multiobj": float(rng.uniform(0.1, 0.4)),
                    "p_sym": float(rng.uniform(0.05, 0.3)),
                    "p_multirel": float(rng.uniform(0.3, 0.6)), "seed": int(rng.integers(1 << 30))})
    return out


def adversarial_graph(p):
    rng = np.random.default_rng(p["seed"])
    w = 1.0 / np.arange(1, p["n_ent"] + 1) ** p["zipf"]
    w /= w.sum()
    fs = set()
    while len(fs) < p["n_facts"]:
        s, o = rng.choice(p["n_ent"], 2, p=w)
        if s == o:
            continue
        r = int(rng.integers(p["n_rel"]))
        fs.add((f"e{s}", f"r{r}", f"e{o}"))
        if rng.random() < p["p_multirel"]:
            r2 = (r + 1 + int(rng.integers(p["n_rel"] - 1))) % p["n_rel"]
            fs.add((f"e{s}", f"r{r2}", f"e{o}") if rng.random() < 0.5
                   else (f"e{o}", f"r{r2}", f"e{s}"))
        if rng.random() < p["p_sym"]:
            fs.add((f"e{o}", f"r{r}", f"e{s}"))
        if rng.random() < p["p_multiobj"]:
            o2 = int(rng.choice(p["n_ent"], p=w))
            if o2 != s:
                fs.add((f"e{s}", f"r{r}", f"e{o2}"))
    return sorted(fs)


def controls():
    """Positivi (casi del revisore: club, inverse, three), avversari (6, generatore sopra) e
    negativo (uniforme: lo strumento deve dire validated e avere ragione)."""
    club = sorted({(f"p{i}", r, f"c{i % 2}") for i in range(100) for r in ("r1", "r2")})
    base = _uniform(100)
    inverse = base | {t for i in range(50) for t in [(f"p{i}", "a", f"q{i}"),
                                                       (f"q{i}", "b", f"p{i}")]}
    three = set(sorted(base)[:80]) | {(f"p{i}", r, f"c{i % 3}") for i in range(40)
                                      for r in ("a", "b", "c")}
    cases = [("club", club), ("inverse", sorted(inverse)), ("three", sorted(three))]
    cases += [(f"adv{j}", adversarial_graph(p)) for j, p in enumerate(adversarial_params())]
    cases += [("negative", sorted(_uniform(200, seed=23)))]
    return [make_sub(f"control:{name}", 90, j, "synthetic", len(t), 0, t)
            for j, (name, t) in enumerate(cases)]


def control_targets(sg):
    """Target dei controlli: i tre standard più, per gli avversari, uno vicino al tetto
    (tetto − 0.03, per difetto al centesimo)."""
    ts = list(TARGETS)
    if sg["dataset"].startswith("control:adv"):
        c = min(exact.ceiling(sg["triples"], sg["queries"]), exact.ceiling(sg["triples"]))
        t = math.floor((c - 0.03) * 100) / 100
        if 0.0 < t and t not in ts:
            ts.append(t)
    return ts


# ---- previsioni (nessuna memoria) ------------------------------------------------------

def struct(sg):
    allk = keys_of(sg["triples"])
    out = {"M": len({x for t in sg["triples"] for x in t}), "Q": len(sg["queries"]),
           "n_keys": len(allk),
           "ceiling": float(min(exact.ceiling(sg["triples"], sg["queries"]),
                                exact.ceiling(sg["triples"], allk)))}
    if hasattr(sizing, "_Graph"):   # solo descrittivo, mai usato per decidere
        g = sizing._Graph(sg["triples"], sg["queries"])
        out.update(regime=g.regime, alias_share=g.alias_share, twin_share=g.twin_share,
                   **g.stats)
    return out


def targets_of(sg, ceiling):
    """Regola scritta: un target entra se target ≤ min(tetto su Q, tetto su tutte le chiavi)
    − CEIL_MARGIN."""
    ts = control_targets(sg) if sg["dataset"].startswith("control:") else TARGETS
    return [t for t in ts if t <= ceiling - CEIL_MARGIN]


def predict_sub(sg):
    st = struct(sg)
    rows = []
    base = {k: sg[k] for k in ("dataset", "scheme", "N", "k", "key")}
    for t in TARGETS:
        if t not in targets_of(sg, st["ceiling"]):
            rows.append({**base, "target": t, **st, "family": None, "discarded": "ceiling"})
            continue
        recs = sizing.recommend(sg["triples"], t, queries=sg["queries"])
        got = {r.family for r in recs}
        for r in recs:
            rows.append({**base, "target": t, **st, "family": r.family,
                         "dim_predicted": r.dim, "bits": r.bits, "predicted": r.predicted,
                         "predicted_all_keys": float(sizing.predict(sg["triples"], r.family,
                                                                    r.dim)),
                         "reliability": r.reliability, "n_warnings": len(r.warnings),
                         "discarded": None})
        for f in FAMILIES:
            if f not in got:
                rows.append({**base, "target": t, **st, "family": f,
                             "discarded": "unreachable (d_max)"})
    return rows


# ---- selezione dei sottografi (dalle sole previsioni) e ordine di priorità -------------

def _validated_after_cap(rows):
    return any(r["family"] and not r["discarded"] and r["reliability"] == "validated" and not (
        r["family"] == "HRR" and excluded("HRR", r["dim_predicted"], r["N"], r["M"], r["Q"]))
        for r in rows)


def select(pred):
    """Il run set del passo attivo, in ORDINE DI PRIORITÀ FISSO. Per cella (grafo, schema,
    N): i primi k_in (k_un sui mai visti) sottografi sono 'base' (campione non selezionato,
    usato da H3, H5, H6); poi, in ordine di k, i sottografi con almeno una validated prevista
    (dopo l'esclusione di HRR) finché la cella ha k_val (k_val_un) sottografi con validated
    ('extra', usati solo da H1, H2, H1u, H2u, H8u). Ordine: per giro (posizione nella lista
    della cella), poi grafo, schema, N: ogni prefisso è bilanciato fra le celle."""
    by_sub = defaultdict(list)
    for r in pred:
        by_sub[r["key"]].append(r)
    cells = defaultdict(list)
    for key, rs in by_sub.items():
        r0 = rs[0]
        cells[(r0["dataset"], r0["scheme"], r0["N"])].append((r0["k"], key, rs))
    out = []
    for (ds, sc, n), subs in cells.items():
        subs.sort()
        unseen = ds in UNSEEN
        k_base = CFG["k_un"] if unseen else CFG["k_in"]
        k_val = CFG["k_val_un"] if unseen else CFG["k_val"]
        chosen = []
        nval = 0
        for pos, (k, key, rs) in enumerate(subs):
            v = _validated_after_cap(rs)
            if pos < k_base:
                chosen.append((key, True))
                nval += v
            elif v and nval < k_val:
                chosen.append((key, False))
                nval += 1
        for rnd, (key, is_base) in enumerate(chosen):
            out.append((rnd, DATASETS.index(ds) if ds in DATASETS else 99,
                        SCHEMES.index(sc), n, key, is_base))
    out.sort()
    return [(key, is_base) for _r, _d, _s, _n, key, is_base in out]


# ---- misure ------------------------------------------------------------------------

def step_of(family, n):
    return families.bsdc_length(n) if family.startswith("BSDC") else sizing.STEP


_T975 = {5: 2.571, 7: 2.365, 19: 2.093}   # 5: solo lo smoke (6 memorie)


def _ci_low(accs):
    a = np.asarray(accs, float)
    return float(a.mean() - _T975[len(a) - 1] * a.std(ddof=1) / math.sqrt(len(a)))


class Truth:
    """Accuratezze di memorie per (famiglia, D, insieme di semi), memoizzate. Ogni memoria
    risponde su TUTTE le chiavi; l'accuratezza su Q (secondaria) è il sottoinsieme."""

    def __init__(self, sg):
        self.sg = sg
        self.objects = defaultdict(set)
        for s, r, o in sg["triples"]:
            self.objects[(s, r)].add(o)
        self.allk = keys_of(sg["triples"])
        qset = set(sg["queries"])
        self.qmask = np.array([k in qset for k in self.allk])
        self.n, self.m = len(sg["triples"]), len({x for t in sg["triples"] for x in t})
        self.memo = {}
        self.built = Counter()

    def fits(self, family, d):
        return not excluded(family, d, self.n, self.m, len(self.sg["queries"]))

    def __call__(self, family, d, seeds, queries="all"):
        key = (family, d, tuple(seeds))
        if key not in self.memo:
            if not self.fits(family, d):
                self.memo[key] = None
            else:
                acc_all, acc_q = [], []
                for seed in seeds:
                    mem = families.make(family, d, seed)
                    mem.store(self.sg["triples"])
                    ans = mem.query_many(self.allk)
                    ok = np.array([o in self.objects[x] for x, (o, _c) in zip(self.allk, ans)])
                    acc_all.append(float(ok.mean()))
                    acc_q.append(float(ok[self.qmask].mean()))
                    self.built[family] += 1
                self.memo[key] = {"all": acc_all, "q": acc_q}
        r = self.memo[key]
        return None if r is None else r[queries]


def mean_or_none(a):
    return None if a is None else float(np.mean(a))


def d_min_measured(truth, family, target, d_hi, n, seeds):
    """Il più piccolo D della griglia accettato con il criterio della verifica (IC_inf 95 %
    su 20 memorie ≥ target − tol, tutte le chiavi). Da d_hi: raddoppio se non basta (fino a
    8×), poi dimezzamenti fino al primo D rifiutato, poi bisezione al passo."""
    step = step_of(family, n)

    def ok(d):
        a = truth(family, d, seeds)
        return a is not None and _ci_low(a) >= target - TOL

    hi = d_hi
    while not ok(hi):
        if hi >= 8 * d_hi or not truth.fits(family, 2 * hi):
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


def baseline(truth, family, target, n, sel, rep):
    """Ricerca empirica senza modello: stesso criterio di recommend (IC_inf su 8 semi di
    selezione ≥ target − tol, sulle query Q), da D0 = 256 (BSDC: multiplo di L ≥ 256),
    raddoppio fino a D_MAX o al limite di memoria, poi bisezione; report su 8 semi nuovi
    (media ≥ target − tol). Restituisce (D, verificato, memorie costruite, secondi)."""
    t0, b0 = time.time(), sum(truth.built.values())
    step = step_of(family, n)
    ok = lambda d: (lambda a: a is not None and _ci_low(a) >= target - TOL)(
        truth(family, d, sel, "q"))
    d = -(-BASE_D0 // step) * step
    lo = None
    while not ok(d):
        lo = d
        if 2 * d > D_MAX or not truth.fits(family, 2 * d):
            return None, False, sum(truth.built.values()) - b0, time.time() - t0
        d *= 2
    hi = d
    if lo is not None:
        while hi - lo > step:
            mid = (lo // step + hi // step) // 2 * step
            if ok(mid):
                hi = mid
            else:
                lo = mid
    r = truth(family, hi, rep, "q")
    return hi, r is not None and np.mean(r) >= target - TOL, \
        sum(truth.built.values()) - b0, time.time() - t0


class _CountMake:
    """Conta le memorie costruite da recommend (per famiglia), sostituendo families.make
    solo durante la chiamata."""

    def __init__(self):
        self.c = Counter()

    def __enter__(self):
        self.orig = families.make

        def make(family, *a, **kw):
            self.c[family] += 1
            return self.orig(family, *a, **kw)
        families.make = make
        return self

    def __exit__(self, *exc):
        families.make = self.orig


def run_sub(sg, truth_seeds=None):
    """Tutte le misure di un sottografo (o di un controllo), con la configurazione CFG."""
    st = struct(sg)
    truth = Truth(sg)
    n = len(sg["triples"])
    rows = []
    for ti, t in enumerate(targets_of(sg, st["ceiling"])):
        sd = target_seeds(sg, ti)
        tseeds = sd["truth"][:truth_seeds] if truth_seeds else sd["truth"]
        dseeds = sd["dmin"][:truth_seeds] if truth_seeds else sd["dmin"]
        recs, cost = [], {}
        for fams in ([f for f in FAMILIES if f != "HRR"], ["HRR"]):
            t0 = time.time()
            with _CountMake() as cm:
                recs += sizing.recommend(sg["triples"], t, families=fams, queries=sg["queries"],
                                         verify=True, seeds=VERIFY_SEEDS, tol=TOL,
                                         rng_seed=sd["rng_seed"], max_mem_bytes=mem_cap(fams[0]))
            cost.update({f: (cm.c[f], time.time() - t0) for f in fams})
        for r in recs:
            row = {k: sg[k] for k in ("dataset", "scheme", "N", "k", "key")}
            row.update(target=t, family=r.family, reliability=r.reliability,
                       warnings=list(r.warnings), dim_predicted=r.dim_predicted, dim=r.dim,
                       adjusted=r.adjusted, verified=r.verified, measured=r.measured,
                       selection_measured=r.selection_measured, bits=r.bits,
                       tool_memories=cost[r.family][0], tool_seconds_call=cost[r.family][1],
                       pred_all=float(sizing.predict(sg["triples"], r.family, r.dim_predicted)),
                       pred_q=float(sizing.predict(sg["triples"], r.family, r.dim_predicted,
                                                   queries=sg["queries"])),
                       excluded_model=not truth.fits(r.family, r.dim_predicted),
                       excluded_final=not truth.fits(r.family, r.dim), **st)
            for name, d in (("model", r.dim_predicted), ("final", r.dim)):
                row[f"truth_{name}"] = mean_or_none(truth(r.family, d, tseeds))
                row[f"truth_{name}_q"] = mean_or_none(truth(r.family, d, tseeds, "q"))
            row["d_min"] = None
            dts, dn = CFG["dmin"]
            if t in dts and n <= dn and row["truth_final"] is not None:
                row["d_min"] = d_min_measured(truth, r.family, t, r.dim, n, dseeds)
            row["baseline"] = None
            hts, hn = CFG["h7"]
            if t in hts and n <= hn:
                bd, bver, bmem, bsec = baseline(truth, r.family, t, n, sd["base_sel"],
                                                sd["base_rep"])
                row["baseline"] = {"dim": bd, "verified": bver, "memories": bmem,
                                   "seconds": bsec,
                                   "bits": None if bd is None else
                                   bd * sizing.bits_per_component(r.family, n),
                                   "truth": None if bd is None else
                                   mean_or_none(truth(r.family, bd, tseeds))}
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
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp(k, n, alpha=0.05):
    """Limiti di Clopper-Pearson UNILATERALI al 95 %: (inferiore, superiore)."""
    lo = 0.0 if k == 0 else _bisect(lambda x: _upper_tail(k, n, x) - alpha)
    hi = 1.0 - _bisect(lambda y: _upper_tail(n - k, n, y) - alpha) if k < n else 1.0
    return lo, hi


def cluster_boot(units, stat, b=BOOT_B, seed=BOOT_SEED):
    """Bootstrap per cluster: units = {cluster: lista di valori}; stat(lista di valori) →
    numero. Restituisce (stima, quantile 5 %, quantile 95 %): limiti unilaterali al 95 %."""
    keys = sorted(units)
    est = stat([v for k in keys for v in units[k]])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.integers(len(keys), size=len(keys))
        vals.append(stat([v for i in pick for v in units[keys[i]]]))
    vals = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if est is None or not len(vals):          # statistica non definita: nessun verdetto
        return est, float("nan"), float("nan")
    return est, float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def _prop(vals):
    return float(np.mean(vals)) if vals else None


def verdict_boot(units, claim, floor, n_min=MIN_CLUSTERS, b=BOOT_B):
    """Quota con bootstrap per cluster: sostenuta se p̂ ≥ claim e boot_inf ≥ floor;
    falsificata se boot_sup < claim; non testabile con meno di n_min cluster."""
    units = {k: v for k, v in units.items() if v}
    n_units = sum(len(v) for v in units.values())
    if len(units) < n_min:
        return {"clusters": len(units), "n": n_units, "verdict": "non testabile"}
    p, lo, hi = cluster_boot(units, _prop, b)
    v = "sostenuta" if p >= claim and lo >= floor else \
        "falsificata" if hi < claim else "non conclusiva"
    # secondario: esito di cluster (tutte le unità del cluster riescono), CP sui cluster
    allpass = sum(all(x) for x in units.values())
    return {"clusters": len(units), "n": n_units, "p": p, "boot_inf": lo, "boot_sup": hi,
            "verdict": v, "cluster_allpass": allpass,
            "cluster_allpass_cp": cp(allpass, len(units))}


def _by_cluster(rows, value):
    u = defaultdict(list)
    for r in rows:
        u[r["key"]].append(value(r))
    return u


def _err(r):
    return abs(r["pred_all"] - r["truth_model"])


def summarize(rows, base_keys, n_min=None):
    """Ipotesi sui grafi reali (i controlli sono a parte). base_keys: i sottografi 'base'."""
    nm = MIN_CLUSTERS if n_min is None else n_min
    real = [r for r in rows if not r["dataset"].startswith("control:")]
    ind = [r for r in real if r["dataset"] in IN_DIST]
    uns = [r for r in real if r["dataset"] in UNSEEN]
    base = [r for r in real if r["key"] in base_keys]
    out = {}
    vv = lambda rs: [r for r in rs if r["reliability"] == "validated" and r["verified"]
                     and r["truth_final"] is not None]
    val = lambda rs: [r for r in rs if r["reliability"] == "validated"
                      and r["truth_model"] is not None]
    reach = lambda r: r["truth_final"] >= r["target"] - TOL
    band = lambda r: r["truth_model"] >= r["target"] - BAND
    out["H1"] = verdict_boot(_by_cluster(vv(ind), reach), *H1, n_min=nm)
    out["H2"] = verdict_boot(_by_cluster(val(ind), band), *H2, n_min=nm)
    out["H1u"] = verdict_boot(_by_cluster(vv(uns), reach), *H1, n_min=nm)
    out["H2u"] = verdict_boot(_by_cluster(val(uns), band), *H2, n_min=nm)
    ver_u = [r for r in uns if r["verified"] and r["truth_final"] is not None]
    out["H8u"] = verdict_boot(_by_cluster(ver_u, reach), *H1, n_min=nm)
    # secondario: le stesse sulle 200 query della verifica
    out["H1_q"] = verdict_boot(_by_cluster(vv(ind), lambda r: r["truth_final_q"] >= r["target"]
                                           - TOL), *H1, n_min=nm)
    out["H2_q"] = verdict_boot(_by_cluster(val(ind), lambda r: r["truth_model_q"] >= r["target"]
                                           - BAND), *H2, n_min=nm)
    # worst case: le raccomandazioni non misurabili (memoria) contano come fallimenti
    wc = lambda rs, ok: _by_cluster([r for r in rs if r["reliability"] == "validated"],
                                    lambda r: r["truth_final"] is not None and ok(r))
    out["H1_worst_case"] = verdict_boot(wc([r for r in ind if r["verified"] or
                                            r["excluded_final"]], reach), *H1, n_min=nm)
    out["H2_worst_case"] = verdict_boot(_by_cluster(
        [r for r in ind if r["reliability"] == "validated"],
        lambda r: r["truth_model"] is not None and band(r)), *H2, n_min=nm)
    out["excluded"] = sorted({f"{r['key']}|{r['target']}|{r['family']}" for r in real
                              if r["excluded_model"] or r["excluded_final"]})
    # H3 (sensibilità), H6 (discriminazione), H5 (copertura): solo sottografi base
    meas = [r for r in base if r["truth_model"] is not None]
    pos = [r for r in meas if _err(r) > ERR_BIG]
    u = _by_cluster(pos, lambda r: r["reliability"] != "validated")
    if len(pos) < 10 or len(u) < 5:
        out["H3"] = {"n": len(pos), "clusters": len(u), "verdict": "non testabile"}
    else:
        out["H3"] = verdict_boot(u, *H3, n_min=5)
    opt = [r for r in pos if r["pred_all"] > r["truth_model"]]
    out["H3_optimistic_only"] = {"k": sum(r["reliability"] != "validated" for r in opt),
                                 "n": len(opt)}
    out["specificity_descr"] = _prop([r["reliability"] == "validated"
                                      for r in meas if _err(r) <= ERR_BIG])

    def rr(vals):
        v = [x for x in vals if x[0]]
        nv = [x for x in vals if not x[0]]
        if not v or not nv:
            return None
        a = (sum(x[1] for x in nv) + 0.5) / (len(nv) + 1)
        b = (sum(x[1] for x in v) + 0.5) / (len(v) + 1)
        return a / b
    u6 = _by_cluster(meas, lambda r: (r["reliability"] == "validated", _err(r) > ERR_BIG))
    if len(u6) < nm:
        out["H6"] = {"clusters": len(u6), "verdict": "non testabile"}
    else:
        est, lo, hi = cluster_boot(u6, rr)
        v = "sostenuta" if est is not None and est >= H6[0] and lo >= H6[1] else \
            "falsificata" if hi < H6[0] else "non conclusiva"
        out["H6"] = {"clusters": len(u6), "rr": est, "boot_inf": lo, "boot_sup": hi,
                     "verdict": v}
    pairs = defaultdict(list)
    for r in base:
        pairs[(r["key"], r["target"])].append(r["reliability"] == "validated" and r["verified"])
    u5 = defaultdict(list)
    for (key, _t), v in pairs.items():
        u5[key].append(any(v))
    out["H5"] = verdict_boot(u5, *H5, n_min=nm)
    # H4: gonfiamento di D (verificate, nel perimetro del D minimo)
    u4 = _by_cluster([r for r in real if r["verified"] and r.get("d_min")],
                     lambda r: r["dim"] / r["d_min"] - 1)
    med = lambda v: float(np.median(v)) if v else None
    if len(u4) < nm:
        out["H4"] = {"clusters": len(u4), "verdict": "non testabile"}
    else:
        m, lo, hi = cluster_boot(u4, med)
        v = "sostenuta" if m <= H4[0] and hi <= H4[1] else \
            "falsificata" if lo > H4[0] else "non conclusiva"
        out["H4"] = {"clusters": len(u4), "n": sum(map(len, u4.values())), "median": m,
                     "boot_inf": lo, "boot_sup": hi, "verdict": v}
    out["H4_model_D"] = med([r["dim_predicted"] / r["d_min"] - 1 for r in real if r.get("d_min")])
    # H7: contro la baseline empirica (memorie costruite e bit al D finale)
    h7 = [r for r in real if r["baseline"] and r["baseline"]["dim"] and r["verified"]
          and r["baseline"]["verified"]]
    um = _by_cluster(h7, lambda r: r["tool_memories"] / max(r["baseline"]["memories"], 1))
    ub = _by_cluster(h7, lambda r: r["bits"] / r["baseline"]["bits"])
    if len(um) < nm:
        out["H7"] = {"clusters": len(um), "verdict": "non testabile"}
    else:
        m1, lo1, hi1 = cluster_boot(um, med)
        m2, lo2, hi2 = cluster_boot(ub, med)
        sup = m1 <= H7[0] and hi1 <= H7[1] and m2 <= H7[2] and hi2 <= H7[2]
        fal = lo1 > H7[0] or lo2 > H7[2]
        out["H7"] = {"clusters": len(um), "n": len(h7), "mem_ratio": [m1, lo1, hi1],
                     "bits_ratio": [m2, lo2, hi2],
                     "verdict": "sostenuta" if sup else "falsificata" if fal else "non conclusiva"}
    out["H7_descr"] = {
        "baseline_verified": _prop([bool(r["baseline"]["verified"]) for r in real
                                    if r["baseline"]]),
        "baseline_truth_reach": _prop([r["baseline"]["truth"] >= r["target"] - TOL for r in real
                                       if r["baseline"] and r["baseline"]["truth"] is not None])}
    # descrittivi per regime, famiglia, grafo
    desc = {}
    for name, keyf in (("dataset", lambda r: r["dataset"]), ("family", lambda r: r["family"]),
                       ("regime", lambda r: r.get("regime"))):
        d = defaultdict(list)
        for r in real:
            d[keyf(r)].append(r)
        desc[name] = {str(k): {"n": len(v), "adjusted": _prop([x["adjusted"] for x in v]),
                               "verified": _prop([x["verified"] for x in v]),
                               "mae_model_pt": None if not [x for x in v if x["truth_model"]
                                                            is not None] else
                               100 * float(np.mean([_err(x) for x in v
                                                    if x["truth_model"] is not None]))}
                      for k, v in d.items()}
    out["descriptive"] = desc
    out["controls"] = control_checks([r for r in rows if r["dataset"].startswith("control:")])
    return out


def control_checks(rows):
    """Controlli (non entrano nelle ipotesi). Positivi (club, inverse, three): MAP-B mai
    validated; club: errore del modello di MAP-B > 3 punti. Avversari: nessuna validated con
    |errore| > 3 punti. Tutti tranne il negativo: fra le verificate, quota ≥ 0.8 con vero
    (tutte le chiavi) ≥ target − tol e nessuna sotto target − 0.03. Negativo: tutte validated, |errore| ≤ 3 punti, vero al D del modello
    ≥ target − 0.02."""
    out = {}
    for name in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == name]
        c = name.split(":", 1)[1]
        meas = [r for r in rs if r["truth_model"] is not None]
        ver = [r for r in rs if r["verified"] and r["truth_final"] is not None]
        # verificate: quota ≥ 0.8 a target − tol e nessuna sotto target − 0.03 (criterio
        # corretto dopo lo smoke sintetico: "tutte ≥ target − tol" è più severo di H1 stessa)
        reach = (not ver or (_prop([r["truth_final"] >= r["target"] - TOL for r in ver]) >= 0.8
                             and min(r["truth_final"] - r["target"] for r in ver) >= -0.03))
        if c == "negative":
            ok = (all(r["reliability"] == "validated" for r in rs)
                  and all(_err(r) <= ERR_BIG and r["truth_model"] >= r["target"] - BAND
                          for r in meas))
        elif c.startswith("adv"):
            ok = reach and not any(r["reliability"] == "validated" and _err(r) > ERR_BIG
                                   for r in meas)
        else:
            mb = [r for r in rs if r["family"] == "MAP-B"]
            ok = reach and bool(mb) and all(r["reliability"] != "validated" for r in mb)
            if c == "club":
                ok = ok and all(r["pred_all"] - r["truth_model"] > ERR_BIG for r in mb
                                if r["truth_model"] is not None)
        out[name] = {"passed": bool(ok), "n": len(rs),
                     "validated": sum(r["reliability"] == "validated" for r in rs),
                     "verified_reach": reach,
                     "max_abs_err_pt": None if not meas else 100 * max(map(_err, meas))}
    return out


# ---- potenza (per cluster, dalle previsioni) ----------------------------------------

def power_cluster(sizes, p, claim, floor, rho, sims=400, b=1000, seed=7):
    """P(sostenuta), P(falsificata) per la quota con bootstrap per cluster: cluster con le
    dimensioni previste, p_c ~ Beta di media p e correlazione intra-cluster rho."""
    rng = np.random.default_rng(seed)
    sizes = np.array(sizes)
    sup = fal = 0
    for _ in range(sims):
        if rho > 0:
            a = p * (1 - rho) / rho
            pc = rng.beta(a, (1 - p) * (1 - rho) / rho, len(sizes))
        else:
            pc = np.full(len(sizes), p)
        k = rng.binomial(sizes, pc)
        est = k.sum() / sizes.sum()
        pick = rng.integers(len(sizes), size=(b, len(sizes)))
        boot = k[pick].sum(1) / sizes[pick].sum(1)
        lo, hi = np.quantile(boot, 0.05), np.quantile(boot, 0.95)
        sup += est >= claim and lo >= floor
        fal += hi < claim
    return sup / sims, fal / sims


def power_table(pred, sel):
    keys = {k for k, _b in sel}
    base_keys = {k for k, b in sel if b}
    ok = [r for r in pred if r["key"] in keys and r["family"] and not r["discarded"]
          and not (r["family"] == "HRR" and excluded("HRR", r["dim_predicted"], r["N"],
                                                     r["M"], r["Q"]))]
    capped = [r for r in pred if r["key"] in keys and r["family"] == "HRR" and not r["discarded"]
              and excluded("HRR", r["dim_predicted"], r["N"], r["M"], r["Q"])]
    tab = {"subgraphs": len(keys), "base_subgraphs": len(base_keys), "recs": len(ok),
           "reliability": Counter(r["reliability"] for r in ok),
           "hrr_excluded": len(capped),
           "hrr_excluded_validated": sum(r["reliability"] == "validated" for r in capped)}
    for grp, dss in (("in", IN_DIST), ("unseen", UNSEEN)):
        c = Counter(r["key"] for r in ok if r["dataset"] in dss and r["reliability"] == "validated")
        sizes = sorted(c.values())
        tab[f"validated_clusters_{grp}"] = len(sizes)
        tab[f"validated_recs_{grp}"] = int(sum(sizes))
        if len(sizes) >= MIN_CLUSTERS:
            for h, (cl, fl) in (("H1", H1), ("H2", H2)):
                tab[f"{h}_{grp}"] = {f"p={p},rho={rho}": power_cluster(sizes, p, cl, fl, rho)
                                     for p in (0.99, 0.98, 0.97, 0.95, 0.92, 0.90, 0.85)
                                     for rho in (0.0, 0.2)}
    pairs = defaultdict(set)
    for r in pred:
        if r["key"] in base_keys and r["dataset"] in DATASETS and not r["discarded"]:
            pairs[r["key"], r["target"]]
            if r["reliability"] == "validated" and not (
                    r["family"] == "HRR" and excluded("HRR", r["dim_predicted"], r["N"],
                                                      r["M"], r["Q"])):
                pairs[r["key"], r["target"]].add(r["family"])
    tab["H5_predicted_max_coverage"] = _prop([bool(v) for v in pairs.values()])
    tab["H5_pairs"] = len(pairs)
    return tab


# ---- tempo (solo memorie sintetiche) ----------------------------------------------

def _syn(n, n_ent, seed=0):
    return sorted(_uniform(n, n_ent=n_ent, n_rel=20, seed=seed))


def _time_mem(f, d, triples, q):
    t0 = time.time()
    mem = families.make(f, d, 1)
    mem.store(triples)
    mem.query_many(q[:1])            # la traccia si costruisce pigramente alla prima query
    t1 = time.time()
    mem.query_many(q)
    return t1 - t0, (time.time() - t1) / max(len(q), 1)


def timing(pred):
    """Per (N, famiglia): costruzione e query per chiave di una memoria su un grafo
    SINTETICO con lo stesso N e M ≈ mediana reale, a tre D (min, mediana, max previsti),
    massimo su TIMING_REPEATS ripetizioni, interpolazione in D. Memorie attese per
    raccomandazione (dichiarato): verifica 16 (+16 se non validated) su Q; vero 20 al D del
    modello (+20 al D finale se non validated) su tutte le chiavi; D minimo (nel perimetro)
    (2 + log2(D/2·passo)) × 20 a 0.75·D su tutte le chiavi; baseline (nel perimetro)
    (log2(D/256) + log2(D/2·passo) + 2) × 8 a 0.75·D su Q + 20 su tutte le chiavi.
    Una raccomandazione HRR esclusa dal limite di memoria costa 0."""
    ok = [r for r in pred if r["family"] and not r["discarded"]]
    tf = {}
    for n in sorted({r["N"] for r in ok}):
        for f in FAMILIES:
            rs = [r for r in ok if r["N"] == n and r["family"] == f]
            if not rs:
                continue
            m = int(np.median([r["M"] for r in rs]))
            t = _syn(n, max(m // 2, 20))
            q = keys_of(t)[:Q_MAX]
            ds = sorted({r["dim_predicted"] for r in rs})
            pts = sorted({ds[0], ds[len(ds) // 2], ds[-1]})
            xs, yb, yq = [], [], []
            for d in pts:
                if families.estimate_bytes(f, d, n_facts=n, n_symbols=m, n_queries=len(q)) \
                        > MAX_MEM:
                    continue
                meas = [_time_mem(f, d, t, q) for _ in range(TIMING_REPEATS)]
                xs.append(d)
                yb.append(max(b for b, _ in meas))
                yq.append(max(c for _, c in meas))
            tf[(n, f)] = (xs, yb, yq)
            print(f"  N={n} {f:7s} M≈{m} D={xs} costruzione={[round(x, 2) for x in yb]} s "
                  f"query={[round(1e3 * x, 2) for x in yq]} ms/chiave", flush=True)

    def cost(f, n, d, nq):
        xs, yb, yq = tf[(n, f)]
        if not xs:
            return 0.0
        return float(np.interp(d, xs, yb)) + float(np.interp(d, xs, yq)) * nq

    steps = []
    for i in range(len(REDUCTION)):
        configure(i)
        sel = dict(select(pred))
        tot = 0.0
        for r in ok:
            if r["key"] not in sel:
                continue
            f, n, d = r["family"], r["N"], r["dim_predicted"]
            if f == "HRR" and excluded(f, d, n, r["M"], r["Q"]):
                continue
            nv = r["reliability"] != "validated"
            stp = step_of(f, n)
            sec = (16 + 16 * nv) * cost(f, n, d, r["Q"]) + \
                (20 + 20 * nv) * cost(f, n, d, r["n_keys"])
            dts, dn = CFG["dmin"]
            if r["target"] in dts and n <= dn:
                ev = 2 + max(0, math.ceil(math.log2(max(d / (2 * stp), 1))))
                sec += ev * 20 * cost(f, n, 0.75 * d, r["n_keys"])
            hts, hn = CFG["h7"]
            if r["target"] in hts and n <= hn:
                ev = 2 + max(0, math.ceil(math.log2(max(d / BASE_D0, 1)))) + \
                    max(0, math.ceil(math.log2(max(d / (2 * stp), 1))))
                sec += ev * 8 * cost(f, n, 0.75 * d, r["Q"]) + 20 * cost(f, n, d, r["n_keys"])
            tot += sec
        tot /= WORKERS
        steps.append((i, len(sel), tot))
        print(f"  passo {i}: {REDUCTION[i]} → {len(sel)} sottografi, {tot / 3600:.2f} h", flush=True)
    return steps


# ---- main ---------------------------------------------------------------------------

def _pool_map(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(workers) as p:
        return p.map(fn, items, chunksize=1)


def all_subgraphs(graphs, kmax=None):
    out = []
    for name, g in graphs.items():
        for si in range(len(SCHEMES)):
            for n in LOADS:
                kk = kmax or K_PRED["in" if name not in UNSEEN else
                                    "unseen_1600" if n == 1600 else "unseen"]
                if n <= len(g.triples) // 2:
                    out += [subgraph(name, g, si, n, K_OFFSET + j) for j in range(kk)]
    return out


def _smoke_rows(sgs, truth_seeds, tag):
    """Righe dello smoke; con SMOKE_CACHE=<cartella> si salvano/rileggono (solo sintetico)."""
    import os
    d = os.environ.get("SMOKE_CACHE")
    path = Path(d) / f"smoke_{tag}.json" if d else None
    if path and path.exists():
        return json.loads(path.read_text())
    rows = [r for sg in sgs for r in run_sub(sg, truth_seeds)]
    if path:
        path.write_text(json.dumps(rows, default=_jdef))
    return rows


def smoke():
    """Impianto intero su grafi SINTETICI con la configurazione del passo scelto: selezione,
    misure, statistiche, controlli; poi il controllo "modello rotto" (previsione +5 punti):
    H2 e H3 devono fallire."""
    configure(GRID_STEP if GRID_STEP is not None else len(REDUCTION) - 1)
    g = Graph(_syn(3000, 400, seed=7))
    sgs = [make_sub("smoke", 9, 0, "uniform", 100, K_OFFSET + j,
                    sample(g, 100, "uniform", np.random.default_rng(j))) for j in range(3)]
    pred = [r for sg in sgs for r in predict_sub(sg)]
    sel = select(pred)
    assert [k for k, _ in sel] == [sg["key"] for sg in sgs][:len(sel)]
    rows = _smoke_rows(sgs + controls(), 6, "ok")
    for r in rows:
        if r["dim"] != r["dim_predicted"]:
            assert r["adjusted"]
        if r["d_min"] is not None:
            assert r["d_min"] % step_of(r["family"], r["N"]) == 0
    assert cp(95, 100)[0] < 0.95 < cp(100, 100)[1] + 1e-12
    # seme disgiunti per costruzione
    sd = target_seeds(sgs[0], 0)
    assert not (set(sd["truth"]) & set(sd["dmin"]) or set(sd["truth"]) & set(sd["base_sel"]))
    summ = summarize([dict(r, dataset="fb15k237") if r["dataset"] == "smoke" else r
                      for r in rows], {sg["key"] for sg in sgs}, n_min=1)
    for r in rows:
        print(f"{r['dataset']:18s} N={r['N']:4d} t={r['target']} {r['family']:7s} "
              f"{r['reliability']:11s} D={r['dim_predicted']}→{r['dim']} ver={r['verified']} "
              f"prev={r['pred_all']:.3f} vero={r['truth_model']} fin={r['truth_final']} "
              f"Dmin={r['d_min']} base={r['baseline'] and r['baseline']['dim']}")
    print(json.dumps({k: v for k, v in summ.items() if k != "descriptive"}, indent=1,
                     default=_jdef))
    bad = [k for k, v in summ["controls"].items() if not v["passed"]]
    print("controlli non superati:", bad)
    # modello rotto: tutte le previsioni +5 punti (solo qui, monkeypatch)
    orig = sizing._predict_graph
    sizing._predict_graph = lambda g_, f, d: min(1.0, orig(g_, f, d) + 0.05)
    try:
        broken = _smoke_rows(sgs + [c for c in controls() if c["dataset"] == "control:negative"],
                             6, "broken")
    finally:
        sizing._predict_graph = orig
    bs = summarize([dict(r, dataset="fb15k237") for r in broken],
                   {r["key"] for r in broken}, n_min=1)
    print("modello rotto:", json.dumps({k: bs[k] for k in ("H2", "H3")}, default=_jdef))
    val = [r for r in broken if r["reliability"] == "validated" and r["truth_model"] is not None]
    pos = [r for r in broken if r["truth_model"] is not None and _err(r) > ERR_BIG]
    p2 = _prop([r["truth_model"] >= r["target"] - BAND for r in val])
    s3 = _prop([r["reliability"] != "validated" for r in pos])
    print(f"modello rotto: H2 quota {p2} su {len(val)}, H3 sensibilità {s3} su {len(pos)}")
    assert bs["H2"].get("verdict") != "sostenuta" and p2 is not None and p2 < H2[0], \
        "modello rotto: H2 deve fallire"
    assert s3 is not None and s3 < H3[0], "modello rotto: H3 deve fallire"


def run(graphs, workers):
    """Il run preregistrato: checkpoint JSONL per sottografo nell'ordine di priorità, ripresa
    saltando i sottografi fatti, tetto rigido HARD_CAP_H; il riepilogo solo alla fine, sul
    prefisso di cluster interi."""
    if GRID_STEP is None:
        raise SystemExit("GRID_STEP non scelto: esegui --timing e fissalo nel codice")
    configure(GRID_STEP)
    pred = json.loads(PRED_OUT.read_text())["rows"]
    sel = select(pred)
    base_keys = {k for k, b in sel if b}
    subs = {sg["key"]: sg for sg in all_subgraphs(graphs)}
    done = []
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            d = json.loads(line)
            if d["grid_step"] != GRID_STEP:
                raise SystemExit("checkpoint con un altro passo: deviazione da dichiarare")
            done.append(d)
    have = {d["key"] for d in done}
    todo = [subs[k] for k, _b in sel if k not in have] + \
        [c for c in controls() if c["key"] not in have]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool, CKPT.open("a") as fh:
        for sg, rows in zip(todo, pool.imap(run_sub, todo, chunksize=1)):
            fh.write(json.dumps({"key": sg["key"], "grid_step": GRID_STEP, "rows": rows}, default=_jdef) + "\n")
            fh.flush()
            if time.time() - t0 > HARD_CAP_H * 3600:
                pool.terminate()
                print("tetto rigido raggiunto: analisi sul prefisso")
                break
    rows = [r for line in CKPT.read_text().splitlines() for r in json.loads(line)["rows"]]
    summ = summarize(rows, base_keys)
    OUT.write_text(json.dumps({"summary": summ, "grid_step": GRID_STEP,
                               "config": {k: v for k, v in REDUCTION[GRID_STEP].items()},
                               "prefix_subgraphs": len({r["key"] for r in rows})},
                              indent=1, default=_jdef))
    print(json.dumps(summ, indent=1, default=_jdef))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="impianto intero, solo sintetico")
    ap.add_argument("--predict", action="store_true", help="previsioni, nessuna memoria")
    ap.add_argument("--timing", action="store_true", help="stima del tempo (sintetico)")
    ap.add_argument("--power", action="store_true", help="potenza per cluster (previsioni)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        smoke()
    elif a.timing:
        pred = json.loads(PRED_OUT.read_text())["rows"]
        steps = timing(pred)
        ok = [s for s in steps if s[2] <= RUN_BUDGET_H * 3600]
        print(f"budget {RUN_BUDGET_H} h: passo scelto {ok[0][0] if ok else len(steps) - 1}")
    elif a.power:
        d = json.loads(PRED_OUT.read_text())
        configure(GRID_STEP)
        sel = select(d["rows"])
        d["grid_step"], d["selection"] = GRID_STEP, sel
        d["power_grid"] = power_table(d["rows"], sel)
        PRED_OUT.write_text(json.dumps(d, indent=1, default=_jdef))
        print(json.dumps(d["power_grid"], indent=1, default=_jdef))
    else:
        graphs = {n: Graph(load(n)) for n in DATASETS}
        if a.predict:
            sgs = all_subgraphs(graphs)
            pred = [r for rs in _pool_map(predict_sub, sgs, a.workers) for r in rs]
            PRED_OUT.write_text(json.dumps({
                "sha256": {n: DATA[n][2] for n in DATASETS}, "K_PRED": K_PRED,
                "adversarial_params": adversarial_params(), "rows": pred}, indent=1))
        else:
            run(graphs, a.workers)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
