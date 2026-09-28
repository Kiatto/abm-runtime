"""fb15k237_prereg.py — Law IV su conoscenza reale, con previsioni preregistrate.

Preregistrazione: docs/preregistration/fb15k237.md. Questo file e quello sono
stati committati INSIEME e PRIMA di qualsiasi esecuzione oltre allo smoke test
dichiarato lì. Non modificare previsioni, griglia o criteri dopo aver visto i
risultati: se serve un cambiamento, va in un nuovo file con un nuovo commit, e
il risultato di questo resta com'è.

Dati: FB15k-237 (Toutanova & Chen, 2015), train split, dal mirror Hugging Face
KGraph/FB15k-237, CC-BY-4.0. Il file non è nel repo: si scarica con

    mkdir -p data/external
    curl -sL -o data/external/fb15k237_train.txt \\
      https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data/train.txt

e deve avere sha256 6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae.

Uso, dalla root del repo:
    python examples/fb15k237_prereg.py            # esecuzione completa
    python examples/fb15k237_prereg.py --smoke    # N = 5, verifica solo che giri
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from math import erf, log, pi, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402  la reference congelata, non il runtime bitpacked

DATA = ROOT / "data" / "external" / "fb15k237_train.txt"
SHA256 = "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"
OUT = ROOT / "results" / "fb15k237_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [50, 100, 150, 200, 300, 400],
        8192: [200, 400, 600, 800, 1200, 1600]}
SEEDS = range(10)
Q_MAX = 200          # query per cella e seed: tutte le triple se N <= 200
K_MEASURED = 0.92    # misurato su dati sintetici, mai su questo dataset
# -----------------------------------------------------------------------------


def predicted_p(n, dim, m, k):
    """Law IV in avanti, con la costante k: accuratezza della singola query."""
    margin = sqrt(2 * k * dim / (pi * n)) - abm.z_gumbel(max(m, 3))
    return 0.5 * (1 + erf(margin / sqrt(2)))


def load_triples():
    h = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if h != SHA256:
        raise SystemExit(f"sha256 inatteso: {h}")
    return [tuple(line.rstrip("\n").split("\t")) for line in DATA.open()]


def run_cell(triples, dim, n, seed):
    rng = np.random.RandomState(1000 * seed + n)
    sample = [triples[i] for i in rng.choice(len(triples), size=n, replace=False)]

    mem = abm.Memory(dim)
    for s, r, o in sample:                       # come store(), ma un solo bundle
        mem._facts.append(mem.fact_hv(s, r, o))
    mem._trace = abm.bundle(mem._facts)
    m = len(mem.items)                           # entità + relazioni: il codebook
                                                 # su cui compete ogni query

    objects = defaultdict(set)                   # (s, r) -> oggetti veri
    aliases = defaultdict(set)                   # (s, r) -> x con (x, r, s) memorizzato
    for s, r, o in sample:
        objects[(s, r)].add(o)
        aliases[(o, r)].add(s)

    names = mem.items._names
    matrix = np.stack(mem.items._states)
    qidx = (range(n) if n <= Q_MAX
            else rng.choice(n, size=Q_MAX, replace=False))
    queries = [sample[i] for i in qidx]

    p92 = predicted_p(n, dim, m, K_MEASURED)
    p1 = predicted_p(n, dim, m, 1.0)
    rows = []
    for j, (s, r, _o) in enumerate(queries):
        noisy = abm.bind(mem._trace, mem.key(s, r))
        dists = np.count_nonzero(matrix != noisy, axis=1)
        ans = names[int(np.argmin(dists))]      # argmin: a parità vince il primo,
        if j < 3:                                # come ItemMemory.cleanup
            assert ans == mem.query(s, r)[0], "cleanup vettoriale != reference"
        good = objects[(s, r)]
        bad = aliases[(s, r)] - good
        alias = len(good) / (len(good) + len(bad))
        rows.append({"ok": ans in good, "alias": alias, "subject": s})

    deg = defaultdict(int)
    for s, _r, o in sample:
        deg[s] += 1
        deg[o] += 1
    cut = np.percentile([deg[row["subject"]] for row in rows], 75)
    hub = [row for row in rows if deg[row["subject"]] > cut]
    rest = [row for row in rows if deg[row["subject"]] <= cut]

    def acc(rs):
        return float(np.mean([r_["ok"] for r_ in rs])) if rs else None

    def pred(rs, p):
        return float(np.mean([p * r_["alias"] for r_ in rs])) if rs else None

    return {
        "dim": dim, "n": n, "seed": seed, "codebook": m, "queries": len(rows),
        "measured": acc(rows),
        "pred_k092_alias": pred(rows, p92),        # PRIMARIA
        "pred_k1_alias": pred(rows, p1),
        "pred_k092_noalias": p92,
        "alias_factor_mean": float(np.mean([r_["alias"] for r_ in rows])),
        "hub": {"n": len(hub), "measured": acc(hub), "pred": pred(hub, p92)},
        "rest": {"n": len(rest), "measured": acc(rest), "pred": pred(rest, p92)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    triples = load_triples()
    if args.smoke:
        cell = run_cell(triples, 2048, 5, 0)
        print("smoke ok:", {k: cell[k] for k in ("n", "codebook", "queries")})
        return
    cells = []
    for dim, ns in GRID.items():
        for n in ns:
            for seed in SEEDS:
                c = run_cell(triples, dim, n, seed)
                cells.append(c)
                print(f"D={dim:5} N={n:5} seed={seed} M={c['codebook']:5} "
                      f"misurato={c['measured']:.3f} previsto={c['pred_k092_alias']:.3f}",
                      flush=True)
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/fb15k237.md",
                               "cells": cells}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
