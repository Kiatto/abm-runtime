"""algebra_prereg.py — l'algebra di ABM (catene, composizione) contro uno store esatto + join.

Preregistrazione: docs/preregistration/algebra.md. Committato insieme a quel file e
PRIMA di misurare sui dati reali; l'unica esecuzione fatta prima del commit è
`--smoke`, su un grafo sintetico generato qui, mai su FB15k-237 o WN18RR.

Domanda: le operazioni algebriche di ABM danno un vantaggio misurabile, a pari bit
o a pari tempo, rispetto a uno store esatto con un join? Disegno ostile ad ABM.

Per ogni (dataset, D, N, seed): N fatti campionati dal grafo come percorsi a due hop
(x, r1, y), (y, r2, z); le domande sono i percorsi funzionali nel campione
((x, r1) e (y, r2) hanno un solo oggetto), al più Q_MAX.

A — catena: ABM una traccia di D bit con gli N fatti, `Memory.chain(x, [r1, r2])`.
    Store: D bit, chiavi implicite in una struttura di retrieval statica (tipo XOR /
    Bloomier) a 1.23·⌈log₂ E⌉ bit per fatto; se non bastano tiene un sottoinsieme a
    caso dei fatti; risponde con due lookup (join). Riportato anche lo store a chiavi
    esplicite della preregistrazione 18 (2⌈log₂ E⌉ + ⌈log₂ R⌉ bit per fatto).
B — composizione compilata: ABM una traccia di D bit con i P percorsi composti
    (`compile_pairs`), una sola cleanup (`query_compiled`). Store: D bit, la migliore
    *prevista* fra (i) tabella dei percorsi (s, r1, r2) → z e (ii) fatti base + join.
C — tempo: tempo medio per domanda a due hop, ABM con cleanup vettoriale numpy
    (più veloce della reference, a favore di ABM) contro dict Python + join.

Uso, dalla root:
    python examples/algebra_prereg.py --smoke     # grafo sintetico piccolo: verifica che giri
    python examples/algebra_prereg.py --predict   # solo previsioni (legge i dati, non misura ABM)
    python examples/algebra_prereg.py             # esecuzione completa
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from math import ceil, floor, log2
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
OUT = ROOT / "results" / "algebra_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [100, 200, 400, 800], 8192: [400, 800, 1600, 3200]}
DATASETS = ("fb15k237", "wn18rr")
SEEDS = 3
Q_MAX = 200
RET_OVERHEAD = 1.23          # bit per chiave di un XOR filter / Bloomier, per bit di valore
TIME_Q = 50                  # domande cronometrate per cella
SEED_BASE = 19_000_003
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


def synthetic_graph(n_ent=400, n_rel=12, n_facts=3000, seed=0):
    rng = np.random.RandomState(seed)
    fs = set()
    while len(fs) < n_facts:
        s, o = rng.randint(n_ent, size=2)
        if s != o:
            fs.add((f"e{s}", f"r{rng.randint(n_rel)}", f"e{o}"))
    return sorted(fs)


class Graph:
    def __init__(self, triples):
        self.triples = [t for t in triples if t[0] != t[2]]
        self.out = defaultdict(list)
        for t in self.triples:
            self.out[t[0]].append(t)


def sample_cell(g, n, seed):
    """N fatti distinti come percorsi a due hop; domande = percorsi funzionali."""
    rng = np.random.RandomState(seed)
    facts, order, paths = set(), [], []
    T = g.triples

    def add(t):
        if t not in facts:
            facts.add(t)
            order.append(t)

    tries = 0
    while len(facts) < n - 1 and tries < 50 * n:
        tries += 1
        f1 = T[rng.randint(len(T))]
        nxt = g.out.get(f1[2])
        if not nxt:
            continue
        f2 = nxt[rng.randint(len(nxt))]
        if f2[2] in (f1[0], f1[2]) or f1 == f2:
            continue
        add(f1)
        add(f2)
        paths.append((f1, f2))
    while len(facts) < n:
        add(T[rng.randint(len(T))])
    order = order[:n]
    facts = set(order)
    objs = defaultdict(set)
    for s, r, o in order:
        objs[(s, r)].add(o)
    seen, qs = set(), []
    for f1, f2 in paths:
        if f1 in facts and f2 in facts and objs[f1[:2]] == {f1[2]} and objs[f2[:2]] == {f2[2]}:
            key = (f1[0], f1[1], f2[1])
            if key not in seen:
                seen.add(key)
                qs.append((f1, f2))
    perm = rng.permutation(len(qs))
    return order, [qs[i] for i in perm[:Q_MAX]]


def sizes(facts):
    ents = {x for s, _, o in facts for x in (s, o)}
    rels = {r for _, r, _ in facts}
    return len(ents), len(rels)


def store_bits(facts):
    e, r = sizes(facts)
    le, lr = ceil(log2(max(e, 2))), ceil(log2(max(r, 2)))
    return {"ret_fact": RET_OVERHEAD * le, "exp_fact": 2 * le + lr,
            "ret_pair": RET_OVERHEAD * le, "exp_pair": 2 * le + 2 * lr}


def both_kept(cap, n):
    return 1.0 if cap >= n else cap * (cap - 1) / (n * (n - 1))


def comp_label(r1, r2):
    # ρ(r1) ⊕ ρ(r2) è simmetrico in (r1, r2) e nullo se r1 = r2
    return "∅" if r1 == r2 else "∘".join(sorted((r1, r2)))


def predict_cell(facts, qs, dim):
    n, p = len(facts), len(qs)
    e, r = sizes(facts)
    m = e + r
    h1 = exact.predict_queries(facts, dim, [f1[:2] for f1, _ in qs])
    h2 = exact.predict_queries(facts, dim, [f2[:2] for _, f2 in qs])
    chain = float(np.mean(np.array(h1) * np.array(h2)))
    syn = [(f1[0], comp_label(f1[1], f2[1]), f2[2]) for f1, f2 in qs]
    m_syn = len({x for t in syn for x in t})
    comp = float(np.mean(exact.predict_queries(syn, dim, [t[:2] for t in syn],
                                               codebook=max(m, m_syn))))
    b = store_bits(facts)
    cap = {k: floor(dim / v) for k, v in b.items()}
    a_ret, a_exp = both_kept(cap["ret_fact"], n), both_kept(cap["exp_fact"], n)
    pair_ret = min(1.0, cap["ret_pair"] / p)
    best_b = "pair_table" if pair_ret >= a_ret else "join"
    return {"N": n, "P": p, "E": e, "R": r, "codebook": m,
            "abm_chain_pred": chain, "abm_comp_pred": comp,
            "store_join_ret": a_ret, "store_join_exp": a_exp,
            "store_pair_ret": pair_ret, "store_B_choice": best_b,
            "store_B_pred": max(pair_ret, a_ret), "caps": cap}


class FastCleanup:
    """Cleanup vettoriale: primo codeword a distanza minima, come ItemMemory.cleanup."""

    def __init__(self, items):
        self.names = list(items._names)
        self.M = np.stack(items._states)

    def __call__(self, noisy):
        d = np.count_nonzero(self.M != noisy[None, :], axis=1)
        i = int(np.argmin(d))
        return self.names[i], int(d[i])


def build(facts, dim):
    mem = abm.Memory(dim)
    mem._facts = [mem.fact_hv(*f) for f in facts]      # identico a store() ripetuto
    mem._trace = abm.bundle(mem._facts)
    return mem


def measure_cell(facts, qs, dim, seed):
    rng = np.random.RandomState(seed + 1)
    mem = build(facts, dim)
    comp = mem.compile_pairs(qs)
    cl = FastCleanup(mem.items)

    def chain(x, r1, r2):
        y, _ = cl(abm.bind(mem._trace, mem.key(x, r1)))
        return cl(abm.bind(mem._trace, mem.key(y, r2)))[0]

    def compiled(x, r1, r2):
        k = abm.bind(mem.items.get(x), abm.bind(abm.permute(mem.items.get(r1), 1),
                                                abm.permute(mem.items.get(r2), 1)))
        return cl(abm.bind(comp._trace, k))[0]

    f1, f2 = qs[0]
    assert chain(f1[0], f1[1], f2[1]) == mem.chain(f1[0], [f1[1], f2[1]])[0]
    assert compiled(f1[0], f1[1], f2[1]) == comp.query_compiled(f1[0], f1[1], f2[1])[0]

    ok_chain = [chain(a[0], a[1], b[1]) == b[2] for a, b in qs]
    ok_comp = [compiled(a[0], a[1], b[1]) == b[2] for a, b in qs]

    b = store_bits(facts)
    out = {"abm_chain": float(np.mean(ok_chain)), "abm_comp": float(np.mean(ok_comp))}
    n = len(facts)
    for tag in ("ret", "exp"):
        cap = floor(dim / b[f"{tag}_fact"])
        keep = [facts[i] for i in rng.permutation(n)[:cap]]
        d = {(s, r): o for s, r, o in keep}
        out[f"store_join_{tag}"] = float(np.mean(
            [d.get((d.get((a[0], a[1])), b_[1])) == b_[2] for a, b_ in qs]))
    cap = floor(dim / b["ret_pair"])
    pt = {(a[0], a[1], b_[1]): b_[2] for a, b_ in [qs[i] for i in rng.permutation(len(qs))[:cap]]}
    out["store_pair_ret"] = float(np.mean([pt.get((a[0], a[1], b_[1])) == b_[2] for a, b_ in qs]))

    # C — tempo per domanda a due hop (vettoriale ABM contro dict + join, store pieno)
    full = {(s, r): o for s, r, o in facts}
    tq = qs[:TIME_Q]
    t0 = time.perf_counter()
    for a, b_ in tq:
        chain(a[0], a[1], b_[1])
    t_abm = (time.perf_counter() - t0) / len(tq)
    t0 = time.perf_counter()
    for _ in range(100):
        for a, b_ in tq:
            full.get((full.get((a[0], a[1])), b_[1]))
    t_store = (time.perf_counter() - t0) / (100 * len(tq))
    out.update({"t_abm_chain_s": t_abm, "t_store_join_s": t_store, "q": len(qs)})
    return out


def run(graphs, grid, seeds, do_measure, out_path=None):
    res = []
    for name, g in graphs.items():
        for dim, ns in grid.items():
            for n in ns:
                for s in range(seeds):
                    seed = SEED_BASE + 1009 * s + 7 * n + dim + (13 if name == "wn18rr" else 0)
                    facts, qs = sample_cell(g, n, seed)
                    if not qs:
                        print(name, dim, n, s, "nessuna domanda", flush=True)
                        continue
                    row = {"dataset": name, "D": dim, "seed": s, **predict_cell(facts, qs, dim)}
                    if do_measure:
                        row.update(measure_cell(facts, qs, dim, seed))
                    res.append(row)
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in row.items() if k != "caps"}), flush=True)
    summary = summarize(res, do_measure)
    if out_path:
        out_path.write_text(json.dumps({"cells": res, "summary": summary}, indent=1))
        print("->", out_path)
    return summary


def summarize(rows, measured):
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["D"], r["N"])].append(r)
    out = []
    for (ds, dim, n), rs in sorted(by.items()):
        c = {"dataset": ds, "D": dim, "N": n, "Q": int(sum(r["P"] for r in rs))}
        for k in ("abm_chain_pred", "abm_comp_pred", "store_join_ret", "store_join_exp",
                  "store_pair_ret", "store_B_pred"):
            c[k] = float(np.mean([r[k] for r in rs]))
        c["store_B_choice"] = rs[0]["store_B_choice"]
        if measured:
            w = np.array([r["q"] for r in rs], float)
            for k in ("abm_chain", "abm_comp", "store_join_ret", "store_join_exp",
                      "store_pair_ret"):
                c[k + "_meas"] = float(np.average([r[k] for r in rs], weights=w))
            c["store_B_meas"] = c["store_pair_ret_meas"] if c["store_B_choice"] == "pair_table" \
                else c["store_join_ret_meas"]
            q = c["Q"]
            se = lambda p: (max(p * (1 - p), 1.0 / q) / q) ** 0.5  # noqa: E731
            c["adv_A"] = c["abm_chain_meas"] - c["store_join_ret_meas"]
            c["se_A"] = (se(c["abm_chain_meas"]) ** 2 + se(c["store_join_ret_meas"]) ** 2) ** 0.5
            c["adv_B"] = c["abm_comp_meas"] - c["store_B_meas"]
            c["se_B"] = (se(c["abm_comp_meas"]) ** 2 + se(c["store_B_meas"]) ** 2) ** 0.5
            c["time_ratio"] = float(np.mean([r["t_abm_chain_s"] / r["t_store_join_s"] for r in rs]))
        out.append(c)
    print("\n dataset   D     N    Q | ABM catena prev  store join | ABM comp prev  store B")
    for c in out:
        line = (f"{c['dataset']:9s} {c['D']:5d} {c['N']:5d} {c['Q']:4d} | "
                f"{c['abm_chain_pred']:.3f}  {c['store_join_ret']:.3f} | "
                f"{c['abm_comp_pred']:.3f}  {c['store_B_pred']:.3f} ({c['store_B_choice']})")
        if measured:
            line += (f" || mis. catena {c['abm_chain_meas']:.3f}/{c['store_join_ret_meas']:.3f}"
                     f" comp {c['abm_comp_meas']:.3f}/{c['store_B_meas']:.3f}"
                     f" tempo x{c['time_ratio']:.0f}")
        print(line)
    if measured:
        print(verdict(out))
    return out


def verdict(cells):
    v = {}
    v["H1_cells_abm_advantage"] = sum(c["adv_A"] > 2 * c["se_A"] for c in cells)
    v["H2_cells_abm_advantage"] = sum(c["adv_B"] > 2 * c["se_B"] for c in cells)
    mae = defaultdict(list)
    for c in cells:
        mae[(c["dataset"], c["D"], "chain")].append(abs(c["abm_chain_meas"] - c["abm_chain_pred"]))
        mae[(c["dataset"], c["D"], "comp")].append(abs(c["abm_comp_meas"] - c["abm_comp_pred"]))
    v["H3_mae_points"] = {"/".join(map(str, k)): round(100 * float(np.mean(x)), 2)
                          for k, x in mae.items()}
    v["H4_min_time_ratio"] = min(c["time_ratio"] for c in cells)
    return json.dumps(v, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="grafo sintetico, griglia minima")
    ap.add_argument("--predict", action="store_true", help="solo previsioni sui dati reali")
    a = ap.parse_args()
    if a.smoke:
        run({"synthetic": Graph(synthetic_graph())}, {1024: [60, 120]}, 2, True)
        return
    graphs = {d: Graph(load(d)) for d in DATASETS}
    if a.predict:
        run(graphs, GRID, SEEDS, False)
    else:
        run(graphs, GRID, SEEDS, True, OUT)


if __name__ == "__main__":
    main()
