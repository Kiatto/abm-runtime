"""sizing_prereg.py — il contratto usato come lo userebbe qualcuno: scegliere D prima.

Preregistrazione: docs/preregistration/sizing.md. Committato insieme a quel file
e PRIMA di misurare.

La tesi del paper è che l'accuratezza si possa dichiarare prima del deployment.
Qui la si usa: per ogni sottografo nuovo di un grafo reale, il modello sceglie la
dimensione MINIMA (multiplo di 64) la cui accuratezza prevista raggiunge un
obiettivo T; poi si misura a quella dimensione. Lo stesso con la Law IV
(k = 0.92, senza alias né gemelli), come fa oggi la reference.

E il tetto: con l'encoding simmetrico, un alias a pari segnale limita
l'accuratezza di una query a g/(g+a) qualunque sia D. Il tetto di un grafo si
calcola dai fatti prima di memorizzarli; per T sopra il tetto il modello dichiara
"irraggiungibile", e qui si verifica che lo resti anche a D grande.

Uso, dalla root:  python examples/sizing_prereg.py [--smoke]
"""
import argparse
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
import exact_prereg as ep  # noqa: E402
import twins_prereg as tp  # noqa: E402

OUT = ROOT / "results" / "sizing_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
SAMPLES = 20                     # sottografi densi nuovi per (grafo, N)
NS = [150, 300]
TARGETS = [0.70, 0.80]
SEED0 = 5000                     # seed mai usati
D_MAX = 64 * 512                 # 32 768
CEILING_D = 16384                # dimensione per il test del tetto
Q_MAX = 200
# -----------------------------------------------------------------------------


def structure(kg, name, n, seed):
    triples, incident = kg
    rng = np.random.RandomState((1299709 * seed + n + (7 if name == "wn18rr" else 0)) % 2**32)
    sample = ep.dense_sample(triples, incident, n, rng)
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= Q_MAX else rng.choice(len(keys), Q_MAX, replace=False)
    queries = [keys[i] for i in idx]
    m = len({x for t in sample for x in t})
    ceiling = float(np.mean([len(objects[q]) / (len(objects[q]) + len(into[q] - objects[q]))
                             for q in queries]))
    return sample, queries, objects, into, m, ceiling


def pred_exact(st, dim):
    sample, queries, objects, into, m, _ = st
    return float(np.mean([p["twins"] for p in tp.predict(sample, queries, objects, into, dim, m)]))


def pred_law_iv(st, dim):
    sample, _q, _o, _i, m, _c = st
    margin = sqrt(2 * 0.92 * dim / (pi * len(sample))) - abm.z_gumbel(max(m, 3))
    return 0.5 * (1 + erf(margin / sqrt(2)))


def min_dim(st, predictor, target):
    """D minima (multiplo di 64, <= D_MAX) con previsione >= target; None se irraggiungibile."""
    if predictor(st, D_MAX) < target:
        return None
    lo, hi = 0, D_MAX // 64
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if predictor(st, 64 * mid) >= target:
            hi = mid
        else:
            lo = mid
    return 64 * hi


def measure(st, dim):
    sample, queries, objects, _into, _m, _c = st
    return float(np.mean(ep.measure(ep.build(dim, sample), queries, objects)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    kgs = {"fb15k237": ep.load_fb15k(), "wn18rr": tp.load_wn18rr()}
    if args.smoke:
        st = structure(kgs["wn18rr"], "wn18rr", 30, SEED0)
        print("smoke ok:", len(st[1]), "query, D scelta", min_dim(st, pred_exact, 0.70))
        return
    rows = []
    for name, kg in kgs.items():
        for n in NS:
            for k in range(SAMPLES):
                st = structure(kg, name, n, SEED0 + k)
                row = {"kg": name, "n": n, "sample": k, "codebook": st[4], "ceiling": st[5]}
                for t in TARGETS:
                    for label, pred in (("exact", pred_exact), ("law_iv", pred_law_iv)):
                        d = min_dim(st, pred, t)
                        row[f"{label}_{t}"] = {
                            "dim": d, "predicted": pred(st, d) if d else None,
                            "measured": measure(st, d) if d else None}
                # il tetto: previsione e misura a D grande
                row["at_ceiling_d"] = {"dim": CEILING_D, "predicted": pred_exact(st, CEILING_D),
                                       "measured": measure(st, CEILING_D)}
                rows.append(row)
            print(name, n, flush=True)
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/sizing.md",
                               "rows": rows}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
