"""composition_prereg.py — composizione grounding × reasoning, senza calibrazione.

Preregistrazione: docs/preregistration/composition.md. Committato insieme a quel
file e PRIMA di eseguire, a parte uno smoke test che stampa solo conteggi.

Il paper (§7, "Resource Composition Law") afferma Acc = E_q[Pg(q)] · Pr(N_eff(ε))
"senza parametri". Nello script che lo misurava (extraction_robustness.py) però
Pr è preso dall'accuratezza MISURATA a ε = 0, la previsione dei fatti spuri usa
una scala calcolata dalle misure, e la forma "raffinata" per i fatti mancanti
descritta nel report non è implementata. Lo stress test con errori non i.i.d.
(composition_stress_results.json) non ha affatto uno script.

Qui la previsione è calcolata dai soli fatti memorizzati, senza misure:
una catena a due hop è riuscibile solo se entrambi i suoi fatti sono intatti, e
allora riesce con la probabilità esatta di two_hop_joint al carico e al codebook
effettivi (dipendenza fra hop inclusa); altrimenti la previsione è 0.

  R1 — i quattro tipi d'errore di extraction_robustness.py, stesso disegno;
  R2 — lo stress test ricostruito: errori i.i.d., a grappolo sulla catena, e
       concentrati sul secondo hop, a parità di tasso medio.

Uso, dalla root:  python examples/composition_prereg.py [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
import abm  # noqa: E402
from bsm.memory.exact_contract import two_hop_joint_fast as two_hop_joint  # noqa: E402

OUT = ROOT / "results" / "composition_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
D, N_CHAINS, SEEDS = 2048, 60, 10
R1_EPS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
R1_KINDS = ["missing", "wrong_relation", "wrong_entity", "spurious"]
R2_EPS = [0.10, 0.20, 0.30, 0.40]
R2_STRUCT = ["iid", "chain", "hop2"]
# -----------------------------------------------------------------------------


def chains(seed, tag):
    out = []
    for c in range(N_CHAINS):
        x, y, z = f"{tag}x{seed}_{c}", f"{tag}y{seed}_{c}", f"{tag}z{seed}_{c}"
        out.append(((x, "r1", y), (y, "r2", z)))
    return out


def corrupt_fact(fact, kind, idx, seed):
    s, r, o = fact
    if kind == "missing":
        return None
    if kind == "wrong_relation":
        return (s, "r_bogus", o)
    return (s, r, f"noise{seed}_{idx}")            # wrong_entity


def realise_r1(seed, eps, kind):
    rng = np.random.RandomState(4099 * seed + int(eps * 1000) + 17 * R1_KINDS.index(kind))
    ch = chains(seed, f"r1{kind}")
    stored, intact = [], []
    for c, (f1, f2) in enumerate(ch):
        ok = True
        for h, f in enumerate((f1, f2)):
            if kind != "spurious" and rng.rand() < eps:
                g = corrupt_fact(f, kind, 2 * c + h, seed)
                ok = False
                if g is not None:
                    stored.append(g)
            else:
                stored.append(f)
        intact.append(ok)
    if kind == "spurious":
        for j in range(int(eps * 2 * N_CHAINS)):
            stored.append((f"sp{seed}_{j}", f"spr{j % 7}", f"spo{seed}_{j}"))
    return ch, stored, intact


def realise_r2(seed, eps, struct):
    """Stesso tasso medio d'errore per fatto (eps), strutture diverse."""
    rng = np.random.RandomState(8191 * seed + int(eps * 1000) + 31 * R2_STRUCT.index(struct))
    ch = chains(seed, f"r2{struct}")
    stored, intact = [], []
    for c, (f1, f2) in enumerate(ch):
        if struct == "iid":
            bad = [rng.rand() < eps, rng.rand() < eps]
        elif struct == "chain":                    # tutta la catena, con prob. eps
            b = rng.rand() < eps
            bad = [b, b]
        else:                                      # hop2: solo il secondo, con prob. 2·eps
            bad = [False, rng.rand() < 2 * eps]
        ok = True
        for h, (f, b) in enumerate(zip((f1, f2), bad)):
            if b:
                stored.append(corrupt_fact(f, "wrong_entity", 2 * c + h, seed))
                ok = False
            else:
                stored.append(f)
        intact.append(ok)
    return ch, stored, intact


def evaluate(ch, stored, intact):
    mem = abm.Memory(D)
    for f in stored:
        mem._facts.append(mem.fact_hv(*f))
    mem._trace = abm.bundle(mem._facts)
    m = len(mem.items)
    ok = [mem.chain(f1[0], ["r1", "r2"])[0] == f2[2] for f1, f2 in ch]
    p_joint = two_hop_joint(len(stored), D, m)[1]
    p_single = two_hop_joint(len(stored), D, m)[0]
    share = float(np.mean(intact))
    return {"n": len(stored), "codebook": m, "measured": float(np.mean(ok)),
            "intact_share": share,
            "pred_exact": share * p_joint,               # PRIMARIA, per query
            "pred_exact_indep": share * p_single ** 2}   # stessa, senza dipendenza fra hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        a = evaluate(*realise_r1(0, 0.2, "missing"))
        b = evaluate(*realise_r2(0, 0.2, "chain"))
        print("smoke ok:", a["n"], a["codebook"], b["n"], b["codebook"])
        return
    cells = []
    for kind in R1_KINDS:
        for eps in R1_EPS:
            for s in range(SEEDS):
                cells.append({"part": "R1", "kind": kind, "eps": eps, "seed": s,
                              **evaluate(*realise_r1(s, eps, kind))})
        print("R1", kind, flush=True)
    for struct in R2_STRUCT:
        for eps in R2_EPS:
            for s in range(SEEDS):
                cells.append({"part": "R2", "struct": struct, "eps": eps, "seed": s,
                              **evaluate(*realise_r2(s, eps, struct))})
        print("R2", struct, flush=True)
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/composition.md",
                               "cells": cells}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
