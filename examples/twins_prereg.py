"""twins_prereg.py — gemelli simmetrici e Law VII esatta.

Preregistrazione: docs/preregistration/twins.md. Committato insieme a quel file
e PRIMA di eseguire, a parte uno smoke test che stampa solo conteggi.

Ipotesi. L'encoding s ⊕ ρ(r) ⊕ o rende identici i vettori di (s, r, o) e (o, r, s):
una relazione simmetrica memorizzata nelle due direzioni è UN fatto di peso 2.
Il modello esatto che tratta ogni fatto come un vettore indipendente è quindi
pessimista dove i gemelli sono frequenti; con i pesi (Law VII esatta) non
dovrebbe esserlo.

  T1 — FB15k-237 a sottografi densi, seed 10-19, mai usati;
  T2 — WN18RR, 34% di triple con gemello: sottografi densi, e campioni uniformi
       come controllo (lì i gemelli sono rarissimi);
  W  — Law VII esatta su pesi sintetici: sostituisce conjecture7_results.json,
       il cui script non è mai stato committato.

Uso, dalla root:  python examples/twins_prereg.py [--smoke]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from math import erf, pi, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import abm  # noqa: E402
import exact_prereg as ep  # noqa: E402  build, measure, dense_sample, load_fb15k
from bsm.memory.exact_contract import (cleanup_accuracy, cleanup_accuracy_mixed,  # noqa: E402
                                       fact_key, fact_weights, p_agree,
                                       p_agree_weighted)

WN = ROOT / "data" / "external" / "wn18rr_train.csv"
WN_SHA = "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"
OUT = ROOT / "results" / "twins_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [50, 100, 150, 200, 300, 400],
        8192: [200, 400, 600, 800, 1200, 1600]}
T1_SEEDS = range(10, 20)
T2_SEEDS = range(10)
Q_MAX = 200
W_DIM, W_SEEDS = 2048, 20
W_CONFIGS = [  # (etichetta, singoli, [(quanti fatti, peso)])
    ("uniforme", 150, []),
    ("10 x w=3", 130, [(10, 3)]),
    ("10 x w=5", 110, [(10, 5)]),
    ("5 x w=8", 110, [(5, 8)]),
    ("20 x w=4", 100, [(20, 4)]),
    ("2 x w=14", 120, [(2, 14)]),
]
# -----------------------------------------------------------------------------


def load_wn18rr():
    if hashlib.sha256(WN.read_bytes()).hexdigest() != WN_SHA:
        raise SystemExit("sha256 inatteso per WN18RR")
    rows = list(csv.reader(WN.open()))[1:]
    triples = [(f"wn:{h}", f"wn:{r}", f"wn:{t}") for h, r, t in rows]
    incident = defaultdict(list)
    for i, (s, _r, o) in enumerate(triples):
        incident[s].append(i)
        incident[o].append(i)
    return triples, incident


def predict(sample, keys, objects, into, dim, m):
    """Tre predittori per ogni query: esatto con gemelli (primario), esatto senza
    gemelli (quello della preregistrazione 2), Law IV con k = 0.92."""
    weights = fact_weights(sample)
    all_w = list(weights.values())
    p_by_w = {}

    def p_of(vec):
        w = weights[vec]
        if w not in p_by_w:            # dipende solo da w e dagli altri pesi
            others = list(all_w)
            others.remove(w)
            p_by_w[w] = p_agree_weighted(w, others)
        return p_by_w[w]

    self_rels = {r for s, r, o in sample if s == o}
    n = len(sample)
    out = []
    for s, r in keys:
        good, bad = objects[(s, r)], into[(s, r)] - objects[(s, r)]
        # fact_key: dal fix dei self-loop (2026-10-01) un self-loop ha chiave
        # ("__self__", r), non frozenset({s}); stesse chiavi di fact_weights.
        cp = [p_of(fact_key(s, r, o)) for o in good]
        ap = [p_of(fact_key(x, r, s)) for x in bad]
        # un self-loop su r vale rho(c_r) e da' a OGNI query (s, r) il candidato s
        # stesso (come abm.exact.predict_queries)
        if r in self_rels and s not in good and s not in bad:
            ap.append(p_of(("__self__", r)))
        g, a = len(good), len(bad)     # no_twins e Law IV: come preregistrati
        margin = sqrt(2 * 0.92 * dim / (pi * n)) - abm.z_gumbel(max(m, 3))
        law = 0.5 * (1 + erf(margin / sqrt(2))) * g / (g + a)
        out.append({"twins": cleanup_accuracy_mixed(dim, m, cp, ap),
                    "no_twins": cleanup_accuracy(n, dim, m, correct=g, aliases=a),
                    "law_iv": law})
    return out


def kg_cell(kg, name, sampler, dim, n, seed):
    triples, incident = kg
    rng = np.random.RandomState(1299709 * seed + n + (7 if name == "wn18rr" else 0))
    if sampler == "dense":
        sample = ep.dense_sample(triples, incident, n, rng)
    else:
        sample = [triples[i] for i in rng.choice(len(triples), n, replace=False)]
    mem = ep.build(dim, sample)
    m = len(mem.items)
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= Q_MAX else rng.choice(len(keys), Q_MAX, replace=False)
    queries = [keys[i] for i in idx]
    ok = ep.measure(mem, queries, objects)
    preds = predict(sample, queries, objects, into, dim, m)
    w = fact_weights(sample)
    return {"part": "T1" if name == "fb15k237" else "T2", "kg": name, "sampler": sampler,
            "dim": dim, "n": n, "seed": seed, "codebook": m, "queries": len(ok),
            "twin_share": float(sum(c for c in w.values() if c > 1) / len(sample)),
            "measured": float(np.mean(ok)),
            "pred_twins": float(np.mean([p["twins"] for p in preds])),
            "pred_no_twins": float(np.mean([p["no_twins"] for p in preds])),
            "pred_law_iv": float(np.mean([p["law_iv"] for p in preds]))}


def w_cell(label, singles, heavy, seed):
    """Law VII: accuratezza misurata sui singoli e sui fatti pesanti, separatamente."""
    tag = f"w{seed}_{label}_"
    facts, weights = [], []
    for i in range(singles):
        facts.append((f"{tag}s{i}", f"{tag}r{i % 13}", f"{tag}o{i}")); weights.append(1)
    for k, (count, w) in enumerate(heavy):
        for i in range(count):
            facts.append((f"{tag}h{k}_{i}", f"{tag}r{i % 13}", f"{tag}ho{k}_{i}")); weights.append(w)
    mem = abm.Memory(W_DIM)
    for f, w in zip(facts, weights):
        for _ in range(w):
            mem._facts.append(mem.fact_hv(*f))
    mem._trace = abm.bundle(mem._facts)
    m = len(mem.items)
    objects = {(s, r): {o} for s, r, o in facts}
    ok = ep.measure(mem, [f[:2] for f in facts], objects)
    res = {}
    for kind, sel in (("singles", [w == 1 for w in weights]), ("heavy", [w > 1 for w in weights])):
        if not any(sel):
            continue
        idx = [i for i, s in enumerate(sel) if s]
        ws = sorted({weights[i] for i in idx})
        preds = []
        for i in idx:
            others = list(weights); others.pop(i)
            preds.append(cleanup_accuracy_mixed(W_DIM, m, [p_agree_weighted(weights[i], others)]))
        n_eff = sum(w * w for w in weights)
        res[kind] = {"measured": float(np.mean([ok[i] for i in idx])),
                     "pred_exact": float(np.mean(preds)),
                     "pred_neff_single": cleanup_accuracy(n_eff, W_DIM, m) if kind == "singles" else None,
                     "weights": ws}
    return {"part": "W", "label": label, "seed": seed, "codebook": m,
            "total_written": len(mem._facts), **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    fb, wn = ep.load_fb15k(), load_wn18rr()
    if args.smoke:
        a = kg_cell(fb, "fb15k237", "dense", 2048, 20, 10)
        b = kg_cell(wn, "wn18rr", "dense", 2048, 20, 0)
        c = w_cell("10 x w=3", 20, [(3, 3)], 0)
        print("smoke ok:", a["queries"], b["queries"], c["total_written"])
        return
    cells = []
    for dim, grid in GRID.items():
        for n in grid:
            for s in T1_SEEDS:
                cells.append(kg_cell(fb, "fb15k237", "dense", dim, n, s))
            for s in T2_SEEDS:
                cells.append(kg_cell(wn, "wn18rr", "dense", dim, n, s))
                cells.append(kg_cell(wn, "wn18rr", "uniform", dim, n, s))
            print("KG", dim, n, flush=True)
    for label, singles, heavy in W_CONFIGS:
        for s in range(W_SEEDS):
            cells.append(w_cell(label, singles, heavy, s))
        print("W", label, flush=True)
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/twins.md",
                               "cells": cells}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
