"""deepchain_prereg.py — la dipendenza fra hop nelle catene profonde.

Preregistrazione: docs/preregistration/deepchain.md. Committato insieme a quel
file e PRIMA di misurare. `--predict` stampa solo le previsioni.

La preregistrazione 3 ha mostrato che due hop sulla stessa traccia sono
correlati negativamente, e che la Law V (Acc = p^h) è falsa come legge esatta.
Qui si chiede se la violazione CRESCE con la profondità, come prevede il modello
per bit (exact_contract.chain_accuracy_mc), in catene di h = 1, 2, 3, 4, 6 hop.

Uso, dalla root:  python examples/deepchain_prereg.py [--predict | --smoke]
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
from bsm.memory.exact_contract import chain_accuracy_mc, cleanup_accuracy  # noqa: E402

OUT = ROOT / "results" / "deepchain_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
N, D, TRACES = 12, 128, 6000
HOPS = [1, 2, 3, 4, 6]
MC_TRIALS = 400_000
# -----------------------------------------------------------------------------


def codebook(h):
    return (N // h) * (h + 1) + h        # entità delle catene + h relazioni


def predict(h):
    m = codebook(h)
    p = cleanup_accuracy(N, D, m)
    model = p if h == 1 else chain_accuracy_mc(N, D, m, h, trials=MC_TRIALS, seed=100 + h)
    mc_se = 0.0 if h == 1 else sqrt(model * (1 - model) / MC_TRIALS)
    return {"h": h, "codebook": m, "p": p, "pred_model": model, "pred_mc_se": mc_se,
            "pred_law_v": p ** h}


def measure(h, traces):
    rels = [f"r{i}" for i in range(1, h + 1)]
    hits = total = 0
    for t in range(traces):
        mem = abm.Memory(D)
        chains = []
        for c in range(N // h):
            ents = [f"t{t}h{h}c{c}e{i}" for i in range(h + 1)]
            chains.append(ents)
            for i in range(h):
                mem._facts.append(mem.fact_hv(ents[i], rels[i], ents[i + 1]))
        mem._trace = abm.bundle(mem._facts)
        assert len(mem.items) == codebook(h)
        for ents in chains:
            hits += mem.chain(ents[0], rels)[0] == ents[-1]
            total += 1
    return hits / total, total


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--predict", action="store_true")
    g.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _acc, n = measure(3, 2)
        print("smoke ok:", n, "catene")
        return
    rows = []
    for h in HOPS:
        row = predict(h)
        if not args.predict:
            acc, n = measure(h, TRACES)
            row.update({"measured": acc, "chains": n,
                        "se": sqrt(acc * (1 - acc) / n)})
        rows.append(row)
        print(f"h={h}: modello {row['pred_model']:.4f}  Law V {row['pred_law_v']:.4f}", flush=True)
    if not args.predict:
        OUT.write_text(json.dumps({"preregistration": "docs/preregistration/deepchain.md",
                                   "rows": rows}, indent=1))
        print("->", OUT)


if __name__ == "__main__":
    main()
