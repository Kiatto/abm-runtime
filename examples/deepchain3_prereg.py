"""deepchain3_prereg.py — catene profonde, con la regola esatta dei pareggi.

Preregistrazione: docs/preregistration/deepchain3.md. Committato insieme a quel
file e PRIMA di misurare. `--predict` stampa solo le previsioni.

La preregistrazione 8 è fallita già al singolo hop, per la regola dei pareggi:
la reference sceglie il primo codeword inserito, e la risposta giusta precede i
distrattori. Rivalutati in modo esplorativo con la regola esatta (win_ordered),
quei dati coincidono con il modello con la dipendenza entro 1.4 SE a ogni h.
Qui la previsione si mette alla prova su dati nuovi, in due configurazioni.

Uso, dalla root:  python examples/deepchain3_prereg.py [--predict | --smoke]
"""
import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
import abm  # noqa: E402
from bsm.memory.exact_contract import (chain_accuracy_mc, cleanup_accuracy_ordered,  # noqa: E402
                                       win_ordered)

OUT = ROOT / "results" / "deepchain3_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
N, TRACES, DISTRACTORS = 12, 8000, 1000
DIMS = [256, 320]
HOPS = [1, 2, 3, 4, 6]
MC_TRIALS = 150_000            # per catena; 12/h catene per traccia
# -----------------------------------------------------------------------------

DISTR = [f"p9distr_{j}" for j in range(DISTRACTORS)]


def insertion_order(h):
    """Ordine in cui la reference inserisce gli item: key(s, r) aggiunge s e r, poi o."""
    order = []
    for c in range(N // h):
        for i in range(h):
            for name in (f"c{c}e{i}", f"r{i + 1}", f"c{c}e{i + 1}"):
                if name not in order:
                    order.append(name)
    return order


def predict(dim, h):
    order = insertion_order(h)
    m = len(order) + DISTRACTORS
    chains = []
    for c in range(N // h):
        idx = [order.index(f"c{c}e{i + 1}") for i in range(h)]
        wins = [win_ordered(dim, k, m - 1 - k) for k in idx]
        model = (cleanup_accuracy_ordered(N, dim, idx[0], m - 1 - idx[0]) if h == 1 else
                 chain_accuracy_mc(N, dim, m, h, trials=MC_TRIALS, seed=900 + 10 * h + c, wins=wins))
        law_v = float(np.prod([cleanup_accuracy_ordered(N, dim, k, m - 1 - k) for k in idx]))
        chains.append((model, law_v))
    model = float(np.mean([c[0] for c in chains]))
    law_v = float(np.mean([c[1] for c in chains]))
    mc_se = 0.0 if h == 1 else sqrt(model * (1 - model) / (MC_TRIALS * (N // h)))
    return {"dim": dim, "h": h, "codebook": m, "pred_model": model,
            "pred_mc_se": mc_se, "pred_law_v": law_v}


def measure(dim, h, traces, distr_mat):
    rels = [f"r{i}" for i in range(1, h + 1)]
    hits = total = 0
    for t in range(traces):
        mem = abm.Memory(dim)
        chains = []
        for c in range(N // h):
            ents = [f"p9d{dim}t{t}h{h}c{c}e{i}" for i in range(h + 1)]
            chains.append(ents)
            for i in range(h):
                mem._facts.append(mem.fact_hv(ents[i], rels[i], ents[i + 1]))
        mem._trace = abm.bundle(mem._facts)
        names = mem.items._names + DISTR
        matrix = np.concatenate([np.stack(mem.items._states), distr_mat])
        for k, ents in enumerate(chains):
            node = ents[0]
            for r in rels:
                noisy = abm.bind(mem._trace, mem.key(node, r))
                node = names[int(np.argmin(np.count_nonzero(matrix != noisy, axis=1)))]
            if t == 0 and k == 0:
                for name in DISTR:
                    mem.items.add(name)
                assert node == mem.chain(ents[0], rels)[0], "catena vettoriale != reference"
            hits += node == ents[-1]
            total += 1
    return hits / total, total


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--predict", action="store_true")
    g.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        mat = np.stack([abm.random_hv(n, DIMS[0]) for n in DISTR])
        _a, n = measure(DIMS[0], 3, 2, mat)
        print("smoke ok:", n, "catene")
        return
    rows = []
    for dim in DIMS:
        mat = np.stack([abm.random_hv(n, dim) for n in DISTR])
        for h in HOPS:
            row = predict(dim, h)
            if not args.predict:
                acc, n = measure(dim, h, TRACES, mat)
                row.update({"measured": acc, "chains": n, "se": sqrt(acc * (1 - acc) / n)})
            rows.append(row)
            print(f"D={dim} h={h}: modello {row['pred_model']:.4f}  Law V {row['pred_law_v']:.4f}  "
                  f"differenza {100*(row['pred_model']-row['pred_law_v']):+.2f}", flush=True)
    if not args.predict:
        OUT.write_text(json.dumps({"preregistration": "docs/preregistration/deepchain3.md",
                                   "rows": rows}, indent=1))
        print("->", OUT)


if __name__ == "__main__":
    main()
