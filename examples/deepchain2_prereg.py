"""deepchain2_prereg.py — dipendenza fra hop nelle catene profonde, codebook grande.

Preregistrazione: docs/preregistration/deepchain2.md. Committato insieme a quel
file e PRIMA di misurare. `--predict` stampa solo le previsioni.

La preregistrazione 7 (deepchain.md) è fallita: con un codebook di ~20 voci il
recupero fuori percorso (un hop sbagliato che ricade per caso sull'entità giusta)
domina l'effetto della dipendenza, e la previsione lo aveva ignorato. Qui lo
stesso disegno ha 1000 distrattori nel codebook: il recupero diventa trascurabile
(stimato per eccesso, ≤ 0.18 punti), e resta solo la dipendenza fra hop.

Uso, dalla root:  python examples/deepchain2_prereg.py [--predict | --smoke]
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

OUT = ROOT / "results" / "deepchain2_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
N, D, TRACES, DISTRACTORS = 12, 320, 8000, 1000
HOPS = [1, 2, 3, 4, 6]
MC_TRIALS = 400_000
# -----------------------------------------------------------------------------

DISTR = [f"p8distr_{j}" for j in range(DISTRACTORS)]


def codebook(h):
    return (N // h) * (h + 1) + h + DISTRACTORS


def predict(h):
    m = codebook(h)
    p = cleanup_accuracy(N, D, m)
    model = p if h == 1 else chain_accuracy_mc(N, D, m, h, trials=MC_TRIALS, seed=200 + h)
    mc_se = 0.0 if h == 1 else sqrt(model * (1 - model) / MC_TRIALS)
    on = 1.0
    for _ in range(h):                       # recupero fuori percorso, per eccesso
        on = p * on + (1 / m) * (1 - on)
    return {"h": h, "codebook": m, "p": p, "pred_model": model, "pred_mc_se": mc_se,
            "pred_law_v": p ** h, "recovery_upper": on - p ** h}


def measure(h, traces, distr_mat):
    rels = [f"r{i}" for i in range(1, h + 1)]
    hits = total = 0
    for t in range(traces):
        mem = abm.Memory(D)
        chains = []
        for c in range(N // h):
            ents = [f"p8t{t}h{h}c{c}e{i}" for i in range(h + 1)]
            chains.append(ents)
            for i in range(h):
                mem._facts.append(mem.fact_hv(ents[i], rels[i], ents[i + 1]))
        mem._trace = abm.bundle(mem._facts)
        # codebook: item della memoria, poi i distrattori (conta solo per i pareggi)
        names = mem.items._names + DISTR
        matrix = np.concatenate([np.stack(mem.items._states), distr_mat])
        assert len(names) == codebook(h)
        for k, ents in enumerate(chains):
            node = ents[0]
            for r in rels:
                noisy = abm.bind(mem._trace, mem.key(node, r))
                node = names[int(np.argmin(np.count_nonzero(matrix != noisy, axis=1)))]
            if t == 0 and k == 0:            # equivalenza con la reference
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
    distr_mat = np.stack([abm.random_hv(n, D) for n in DISTR])
    if args.smoke:
        _a, n = measure(3, 2, distr_mat)
        print("smoke ok:", n, "catene")
        return
    rows = []
    for h in HOPS:
        row = predict(h)
        if not args.predict:
            acc, n = measure(h, TRACES, distr_mat)
            row.update({"measured": acc, "chains": n, "se": sqrt(acc * (1 - acc) / n)})
        rows.append(row)
        print(f"h={h}: modello {row['pred_model']:.4f}  Law V {row['pred_law_v']:.4f}  "
              f"recupero <= {100*row['recovery_upper']:.2f} punti", flush=True)
    if not args.predict:
        OUT.write_text(json.dumps({"preregistration": "docs/preregistration/deepchain2.md",
                                   "rows": rows}, indent=1))
        print("->", OUT)


if __name__ == "__main__":
    main()
