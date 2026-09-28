"""asymmetric_prereg.py — encoding simmetrico contro asimmetrico.

Preregistrazione: docs/preregistration/asymmetric.md. Committato insieme a quel
file e PRIMA di misurare. `--predict` stampa solo le previsioni della teoria (le
stesse del file di preregistrazione), senza misurare nulla.

L'encoding della reference, s ⊕ ρ(r) ⊕ o, è simmetrico in s e o: dà gli alias (un
fatto (x, r, s) risponde a (s, r) con x) e i gemelli ((s, r, o) e (o, r, s) sono lo
stesso vettore, un fatto di peso 2). La variante asimmetrica

    s ⊕ ρ(r) ⊕ ρ²(o),   decodifica: cleanup(ρ⁻²(T ⊕ s ⊕ ρ(r)))

non ha né alias né gemelli: ogni fatto memorizzato è un vettore indipendente.
Il modello esatto prevede l'accuratezza di entrambe senza parametri; qui si mette
alla prova la previsione, e in particolare la sua conseguenza meno ovvia: dove i
gemelli sono frequenti, la simmetria AIUTA.

La reference congelata non è toccata: la variante è una sottoclasse qui.
Uso:  python examples/asymmetric_prereg.py [--predict | --smoke]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import abm  # noqa: E402
import exact_prereg as ep  # noqa: E402
import twins_prereg as tp  # noqa: E402
from bsm.memory.exact_contract import cleanup_accuracy  # noqa: E402

OUT = ROOT / "results" / "asymmetric_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [100, 200, 300, 400], 8192: [400, 800, 1200, 1600]}
SEEDS = range(20, 30)            # mai usati per questi grafi
Q_MAX = 200
# -----------------------------------------------------------------------------


class AsymMemory(abm.Memory):
    """s ⊕ ρ(r) ⊕ ρ²(o): nessuna simmetria fra soggetto e oggetto."""

    def fact_hv(self, s, r, o):
        return abm.bind(self.key(s, r), abm.permute(self.items.add(o), 2))

    def decode(self, s, r):
        return abm.permute(abm.bind(self._trace, self.key(s, r)), -2)


def sample_of(kg, name, dim, n, seed):
    triples, incident = kg
    rng = np.random.RandomState(1299709 * seed + n + (7 if name == "wn18rr" else 0))
    sample = ep.dense_sample(triples, incident, n, rng)
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= Q_MAX else rng.choice(len(keys), Q_MAX, replace=False)
    return sample, [keys[i] for i in idx], objects, into


def codebook_size(sample):
    return len({x for s, r, o in sample for x in (s, r, o)})


def predictions(sample, queries, objects, into, dim):
    m = codebook_size(sample)
    sym = tp.predict(sample, queries, objects, into, dim, m)      # con gemelli e alias
    asym = [cleanup_accuracy(len(sample), dim, m, correct=len(objects[q]))
            for q in queries]                                     # indipendenti, nessun alias
    return float(np.mean([p["twins"] for p in sym])), float(np.mean(asym))


def measure_asym(sample, queries, objects, dim):
    mem = AsymMemory(dim)
    for s, r, o in sample:
        mem._facts.append(mem.fact_hv(s, r, o))
    mem._trace = abm.bundle(mem._facts)
    names = mem.items._names
    matrix = np.stack(mem.items._states)
    ok = []
    for s, r in queries:
        noisy = mem.decode(s, r)
        ok.append(names[int(np.argmin(np.count_nonzero(matrix != noisy, axis=1)))]
                  in objects[(s, r)])
    return float(np.mean(ok))


def measure_sym(sample, queries, objects, dim):
    return float(np.mean(ep.measure(ep.build(dim, sample), queries, objects)))


def run(mode):
    kgs = {"fb15k237": ep.load_fb15k(), "wn18rr": tp.load_wn18rr()}
    cells = []
    for name, kg in kgs.items():
        for dim, grid in GRID.items():
            for n in grid:
                ps, pa, ms, ma = [], [], [], []
                for seed in (SEEDS if mode != "smoke" else [SEEDS[0]]):
                    nn = n if mode != "smoke" else 20
                    sample, queries, objects, into = sample_of(kg, name, dim, nn, seed)
                    a, b = predictions(sample, queries, objects, into, dim)
                    ps.append(a); pa.append(b)
                    if mode == "run":
                        ms.append(measure_sym(sample, queries, objects, dim))
                        ma.append(measure_asym(sample, queries, objects, dim))
                cell = {"kg": name, "dim": dim, "n": n,
                        "pred_sym": float(np.mean(ps)), "pred_asym": float(np.mean(pa))}
                if mode == "run":
                    cell.update({"meas_sym": float(np.mean(ms)), "meas_asym": float(np.mean(ma)),
                                 "seeds_meas_sym": ms, "seeds_meas_asym": ma})
                cells.append(cell)
                if mode == "predict":
                    print(f"{name:9} D={dim:5} N={n:5}  simmetrico {100*cell['pred_sym']:5.1f}  "
                          f"asimmetrico {100*cell['pred_asym']:5.1f}  "
                          f"differenza {100*(cell['pred_sym']-cell['pred_asym']):+5.1f}", flush=True)
                if mode == "smoke":
                    return print("smoke ok:", len(queries), codebook_size(sample))
    if mode == "run":
        OUT.write_text(json.dumps({"preregistration": "docs/preregistration/asymmetric.md",
                                   "cells": cells}, indent=1))
        print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--predict", action="store_true")
    g.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run("predict" if args.predict else "smoke" if args.smoke else "run")


if __name__ == "__main__":
    main()
