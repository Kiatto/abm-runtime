"""capacity_seed10.py — rimisura il carico N* al 50% di accuratezza per D = 512–4096.

`results/seed10_results.json` (law4.per_d) non ha lo script che lo ha prodotto
(audit 2026-09-30, punto 7). Questo script lo ricostruisce con il protocollo che
il file e `k_from_results.py` implicano: N fatti (s_i, r_{i mod 11}, o_i) con
soggetti e oggetti distinti, quindi M = 2N + 11 codeword; per ogni seed si misura
l'accuratezza su una griglia di N e si interpola il punto al 50%. Il modello
esatto (`abm.exact`) usa la stessa griglia e la stessa interpolazione.

Scrive `results/capacity_seed10_rerun.json`; non sovrascrive l'originale.
Uso, dalla root:  python examples/capacity_seed10.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

DIMS = (512, 1024, 2048, 4096)
SEEDS = 10
RELS = 11


def triples(n, tag):
    return [(f"{tag}s{i}", f"{tag}r{i % RELS}", f"{tag}o{i}") for i in range(n)]


def accuracy(dim, n, tag):
    trip = triples(n, tag)
    mem = abm.Memory(dim)
    for f in trip:
        mem._facts.append(mem.fact_hv(*f))
    mem._trace = abm.bundle(mem._facts)
    names = mem.items._names
    mat = np.stack(mem.items._states)
    hits = 0
    for s, r, o in trip:
        noisy = abm.bind(mem._trace, mem.key(s, r))
        hits += names[int(np.argmin(np.count_nonzero(mat != noisy, axis=1)))] == o
    return hits / n


def crossing(ns, accs):
    for (n0, a0), (n1, a1) in zip(zip(ns, accs), zip(ns[1:], accs[1:])):
        if a0 >= 0.5 > a1:
            return n0 + (a0 - 0.5) * (n1 - n0) / (a0 - a1)
    return float("nan")


def main():
    out = {}
    for dim in DIMS:
        centre = 0.068 * dim + 15
        ns = [int(round(centre * f)) for f in np.linspace(0.6, 1.5, 19)]
        per_seed = []
        for seed in range(SEEDS):
            accs = [accuracy(dim, n, f"d{dim}k{seed}n{n}_") for n in ns]
            per_seed.append(crossing(ns, accs))
        pred = [float(np.mean(exact.predict_queries(t := triples(n, "x"), dim,
                                                    [(s, r) for s, r, _ in t]))) for n in ns]
        mean = float(np.mean(per_seed))
        ci = float(1.96 * np.std(per_seed, ddof=1) / np.sqrt(SEEDS))
        out[str(dim)] = {"measured": [mean, ci], "per_seed": per_seed,
                         "exact": crossing(ns, pred)}
        print(f"D={dim}: misurato {mean:.1f} ± {ci:.1f}   esatto {out[str(dim)]['exact']:.1f}", flush=True)
    (ROOT / "results" / "capacity_seed10_rerun.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
