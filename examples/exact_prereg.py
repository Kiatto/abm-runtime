"""exact_prereg.py — il contratto esatto su configurazioni mai misurate.

Preregistrazione: docs/preregistration/exact_contract.md. Questo file e quello
sono committati INSIEME e PRIMA di qualsiasi esecuzione, a parte lo smoke test
dichiarato lì, che non stampa accuratezze.

Tre parti:
  A — sintetico, D = 16384: una dimensione mai usata in nessun esperimento;
  B — sintetico, query con g = 1, 2, 4 oggetti veri: mette alla prova l'unica
      ipotesi non esatta del modello (candidati a pari segnale indipendenti);
  C — FB15k-237 a sottografi densi (BFS): entità che si ripetono, risposte
      multiple e alias per la simmetria s/o, che il primo test preregistrato
      (fb15k237.md) non aveva messo alla prova.

Uso, dalla root:  python examples/exact_prereg.py [--smoke]
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
import abm  # noqa: E402
from bsm.memory.exact_contract import cleanup_accuracy  # noqa: E402

DATA = ROOT / "data" / "external" / "fb15k237_train.txt"
SHA256 = "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"
OUT = ROOT / "results" / "exact_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
SEEDS = range(10)
Q_MAX = 200
A_DIM, A_GRID = 16384, [400, 600, 900, 1200, 1600, 2200]
B_DIM, B_GRID, B_G = 4096, [100, 200, 300], [1, 2, 4]
C_GRID = {2048: [50, 100, 150, 200, 300, 400],
          8192: [200, 400, 600, 800, 1200, 1600]}
K_LAW_IV = 0.92   # solo per il confronto: il predittore del primo test
# -----------------------------------------------------------------------------


def law_iv_p(n, dim, m, k=K_LAW_IV):
    from math import erf, pi, sqrt
    margin = sqrt(2 * k * dim / (pi * n)) - abm.z_gumbel(max(m, 3))
    return 0.5 * (1 + erf(margin / sqrt(2)))


def build(dim, triples):
    mem = abm.Memory(dim)
    for s, r, o in triples:
        mem._facts.append(mem.fact_hv(s, r, o))
    mem._trace = abm.bundle(mem._facts)
    return mem


def measure(mem, queries, objects, check_first=3):
    """Per ogni (s, r): la risposta è uno degli oggetti veri?"""
    names = mem.items._names
    matrix = np.stack(mem.items._states)
    out = []
    for j, (s, r) in enumerate(queries):
        noisy = abm.bind(mem._trace, mem.key(s, r))
        ans = names[int(np.argmin(np.count_nonzero(matrix != noisy, axis=1)))]
        if j < check_first:
            assert ans == mem.query(s, r)[0], "cleanup vettoriale != reference"
        out.append(ans in objects[(s, r)])
    return out


def part_a(seed, n):
    rng = np.random.RandomState(7919 * seed + n)
    tag = f"a{seed}_{n}_"
    triples = [(f"{tag}s{i}", f"{tag}r{i % 13}", f"{tag}o{i}") for i in range(n)]
    mem = build(A_DIM, triples)
    m = len(mem.items)
    objects = {(s, r): {o} for s, r, o in triples}
    idx = range(n) if n <= Q_MAX else rng.choice(n, Q_MAX, replace=False)
    queries = [triples[i][:2] for i in idx]
    ok = measure(mem, queries, objects)
    return {"part": "A", "dim": A_DIM, "n": n, "seed": seed, "codebook": m,
            "queries": len(ok), "measured": float(np.mean(ok)),
            "pred_exact": cleanup_accuracy(n, A_DIM, m),
            "pred_law_iv": law_iv_p(n, A_DIM, m)}


def part_b(seed, n, g):
    rng = np.random.RandomState(104729 * seed + 31 * n + g)
    tag = f"b{seed}_{n}_{g}_"
    pairs = n // g
    triples = [(f"{tag}s{i}", f"{tag}r{i % 13}", f"{tag}o{i}_{j}")
               for i in range(pairs) for j in range(g)]
    mem = build(B_DIM, triples)
    m = len(mem.items)
    objects = defaultdict(set)
    for s, r, o in triples:
        objects[(s, r)].add(o)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= Q_MAX else rng.choice(len(keys), Q_MAX, replace=False)
    queries = [keys[i] for i in idx]
    ok = measure(mem, queries, objects)
    return {"part": "B", "dim": B_DIM, "n": len(triples), "g": g, "seed": seed,
            "codebook": m, "queries": len(ok), "measured": float(np.mean(ok)),
            "pred_exact": cleanup_accuracy(len(triples), B_DIM, m, correct=g),
            "pred_law_iv": law_iv_p(len(triples), B_DIM, m)}


def load_fb15k():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("sha256 inatteso per FB15k-237")
    triples = [tuple(line.rstrip("\n").split("\t")) for line in DATA.open()]
    incident = defaultdict(list)
    for i, (s, _r, o) in enumerate(triples):
        incident[s].append(i)
        incident[o].append(i)
    return triples, incident


def dense_sample(triples, incident, n, rng):
    """BFS non orientata da un'entità casuale: triple incidenti in ordine di
    scoperta, finché non se ne hanno n. Se la componente si esaurisce, si riparte
    da un'altra entità casuale."""
    entities = list(incident)
    chosen, seen_t, seen_e = [], set(), set()
    while len(chosen) < n:
        start = entities[rng.randint(len(entities))]
        if start in seen_e:
            continue
        queue = deque([start])
        seen_e.add(start)
        while queue and len(chosen) < n:
            e = queue.popleft()
            for ti in incident[e]:
                if ti in seen_t:
                    continue
                seen_t.add(ti)
                chosen.append(triples[ti])
                if len(chosen) >= n:
                    break
                s, _r, o = triples[ti]
                for x in (s, o):
                    if x not in seen_e:
                        seen_e.add(x)
                        queue.append(x)
    return chosen


def part_c(fb, dim, seed, n):
    triples, incident = fb
    rng = np.random.RandomState(1299709 * seed + n)
    sample = dense_sample(triples, incident, n, rng)
    mem = build(dim, sample)
    m = len(mem.items)
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)          # (x, r, s) memorizzato: x è un alias per (s, r)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= Q_MAX else rng.choice(len(keys), Q_MAX, replace=False)
    queries = [keys[i] for i in idx]
    ok = measure(mem, queries, objects)
    deg = defaultdict(int)
    for s, _r, o in sample:
        deg[s] += 1
        deg[o] += 1
    rows = []
    for (s, r), hit in zip(queries, ok):
        g = len(objects[(s, r)])
        a = len(into[(s, r)] - objects[(s, r)])
        rows.append({"ok": hit, "deg": deg[s],
                     "exact": cleanup_accuracy(len(sample), dim, m, correct=g, aliases=a),
                     "law_iv": law_iv_p(len(sample), dim, m) * g / (g + a),
                     "g": g, "a": a})
    cut = np.percentile([row["deg"] for row in rows], 75)

    def agg(rs, key):
        return float(np.mean([r_[key] for r_ in rs])) if rs else None
    hub = [r_ for r_ in rows if r_["deg"] > cut]
    rest = [r_ for r_ in rows if r_["deg"] <= cut]
    return {"part": "C", "dim": dim, "n": len(sample), "seed": seed, "codebook": m,
            "queries": len(rows), "measured": agg(rows, "ok"),
            "pred_exact": agg(rows, "exact"), "pred_law_iv": agg(rows, "law_iv"),
            "multi_object_share": float(np.mean([r_["g"] > 1 for r_ in rows])),
            "alias_share": float(np.mean([r_["a"] > 0 for r_ in rows])),
            "hub": {"n": len(hub), "measured": agg(hub, "ok"), "pred": agg(hub, "exact")},
            "rest": {"n": len(rest), "measured": agg(rest, "ok"), "pred": agg(rest, "exact")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    fb = load_fb15k()
    if args.smoke:
        for c in (part_a(0, 20), part_b(0, 20, 2), part_c(fb, 2048, 0, 20)):
            print("smoke ok:", c["part"], {k: c[k] for k in ("n", "codebook", "queries")})
        return
    cells = []
    for n in A_GRID:
        for s in SEEDS:
            cells.append(part_a(s, n)); print("A", n, s, flush=True)
    for g in B_G:
        for n in B_GRID:
            for s in SEEDS:
                cells.append(part_b(s, n, g)); print("B", g, n, s, flush=True)
    for dim, grid in C_GRID.items():
        for n in grid:
            for s in SEEDS:
                cells.append(part_c(fb, dim, s, n)); print("C", dim, n, s, flush=True)
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/exact_contract.md",
                               "cells": cells}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
