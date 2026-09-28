"""dependence_prereg.py — dipendenza esatta fra hop e proiezione tipizzata.

Preregistrazione: docs/preregistration/dependence.md. Committato insieme a quel
file e PRIMA di eseguire, a parte uno smoke test che non stampa accuratezze.

Tre parti:
  E1 — correlazione per bit fra l'accordo della query con due fatti della
       stessa traccia. Previsione: -rho²/(1 - rho²), non 0.
  E2 — correlazione fra i SUCCESSI di due hop concatenati sulla stessa traccia.
       La Law V (Acc = p^h) assume che sia 0. Previsione: il valore esatto di
       exact_contract.two_hop_joint, negativo.
  G  — il guadagno di capacità del cleanup tipato (P3 del paper), ricostruito:
       lo script originale non era mai stato committato.

Uso, dalla root:  python examples/dependence_prereg.py [--smoke]
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
from bsm.memory.exact_contract import (bit_correlation, capacity,  # noqa: E402
                                       cleanup_accuracy, two_hop_joint)

OUT = ROOT / "results" / "dependence_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
E1 = {"dim": 1024, "ns": [10, 30, 90], "traces": 400}
E2 = [{"n": 10, "dim": 48, "traces": 8000},      # 5 catene per traccia: 40 000 coppie
      {"n": 30, "dim": 240, "traces": 8000}]     # 15 catene: 120 000 coppie
G = {"dim": 1024, "extra": [2000, 8000, 32000], "seeds": 30, "q_max": 100,
     "lo": 0.6, "hi": 1.6, "step": 1.1}
# -----------------------------------------------------------------------------


def chains(n, tag):
    """n/2 catene a due hop: (a_i, r1, b_i), (b_i, r2, c_i). M = 3n/2 + 2."""
    out = []
    for i in range(n // 2):
        out.append(((f"{tag}a{i}", f"{tag}r1", f"{tag}b{i}"),
                    (f"{tag}b{i}", f"{tag}r2", f"{tag}c{i}")))
    return out


def build(dim, facts):
    mem = abm.Memory(dim)
    for s, r, o in facts:
        mem._facts.append(mem.fact_hv(s, r, o))
    mem._trace = abm.bundle(mem._facts)
    return mem


def part_e1(n, dim, traces):
    """Correlazione per bit, sommata su tutte le coppie concatenate."""
    s1 = s2 = s12 = s11 = s22 = 0.0
    count = 0
    for t in range(traces):
        pairs = chains(n, f"e1_{n}_{t}_")
        mem = build(dim, [f for pair in pairs for f in pair])
        tr = mem._trace.astype(np.int64)
        for f1, f2 in pairs:
            a1 = tr * mem.fact_hv(*f1).astype(np.int64)
            a2 = tr * mem.fact_hv(*f2).astype(np.int64)
            s1 += a1.sum(); s2 += a2.sum(); s12 += (a1 * a2).sum()
            s11 += (a1 * a1).sum(); s22 += (a2 * a2).sum()
            count += dim
    m1, m2 = s1 / count, s2 / count
    cov = s12 / count - m1 * m2
    corr = cov / sqrt((s11 / count - m1 ** 2) * (s22 / count - m2 ** 2))
    return {"part": "E1", "n": n, "dim": dim, "bit_samples": count,
            "measured": corr, "se": 1 / sqrt(count), "predicted": bit_correlation(n),
            "law_v_assumption": 0.0}


def part_e2(n, dim, traces):
    """Successi dei due hop, ciascuno interrogato con la sua chiave vera."""
    hit1 = hit2 = both = total = 0
    ms = []
    for t in range(traces):
        pairs = chains(n, f"e2_{n}_{t}_")
        mem = build(dim, [f for pair in pairs for f in pair])
        ms.append(len(mem.items))
        for (s1, r1, o1), (s2, r2, o2) in pairs:
            h1 = mem.query(s1, r1)[0] == o1
            h2 = mem.query(s2, r2)[0] == o2
            hit1 += h1; hit2 += h2; both += h1 and h2; total += 1
    p1, p2, p12 = hit1 / total, hit2 / total, both / total
    phi = (p12 - p1 * p2) / sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    m = int(round(np.mean(ms)))
    p_pred, both_pred, p2_pred = two_hop_joint(n, dim, m)
    phi_pred = (both_pred - p2_pred) / (p_pred * (1 - p_pred))
    return {"part": "E2", "n": n, "dim": dim, "codebook": m, "pairs": total,
            "p1": p1, "p2": p2, "both": p12, "phi": phi, "se_phi": 1 / sqrt(total),
            "pred_p": p_pred, "pred_both": both_pred, "pred_phi": phi_pred,
            "law_v_both": p1 * p2}


def distractor_matrix(dim, count):
    return np.stack([abm.random_hv(f"distr_{j}", dim) for j in range(count)])


def acc_at(n, dim, extra_mat, typed, seeds, q_max, check=False):
    hits = total = 0
    for seed in range(seeds):
        tag = f"g{seed}_{n}_"
        facts = [(f"{tag}s{i}", f"{tag}r{i % 13}", f"{tag}o{i}") for i in range(n)]
        mem = build(dim, facts)
        objs = [o for _s, _r, o in facts]
        if typed:
            names = objs
            matrix = np.stack([mem.items.get(o) for o in objs])
        else:
            # il codebook pieno: gli item della memoria, poi i distrattori, come se
            # fossero stati aggiunti dopo i fatti (conta solo per i pareggi)
            names = mem.items._names + [f"distr_{j}" for j in range(len(extra_mat))]
            matrix = np.concatenate([np.stack(mem.items._states), extra_mat])
        for j, (s, r, o) in enumerate(facts[:q_max]):
            noisy = abm.bind(mem._trace, mem.key(s, r))
            ans = names[int(np.argmin(np.count_nonzero(matrix != noisy, axis=1)))]
            if check and seed == 0 and j < 3:
                ref = mem.query(s, r, subset=objs if typed else None)[0]
                if not typed:
                    for k in range(len(extra_mat)):
                        mem.items.add(f"distr_{k}")
                    ref = mem.query(s, r)[0]
                assert ans == ref, "cleanup vettoriale != reference"
            hits += ans == o
            total += 1
    return hits / total


def n_star(dim, extra_mat, typed, predicted):
    """Crossing del 50% su una griglia geometrica intorno alla previsione."""
    grid, n = [], G["lo"] * predicted
    while n <= G["hi"] * predicted:
        grid.append(max(int(round(n)), 2)); n *= G["step"]
    grid = sorted(set(grid))
    accs = [acc_at(k, dim, extra_mat, typed, G["seeds"], G["q_max"], check=(i == 0))
            for i, k in enumerate(grid)]
    for (n0, a0), (n1, a1) in zip(zip(grid, accs), zip(grid[1:], accs[1:])):
        if a0 >= 0.5 > a1:
            return n0 + (a0 - 0.5) * (n1 - n0) / (a0 - a1), list(zip(grid, accs))
    return None, list(zip(grid, accs))


def part_g():
    dim = G["dim"]
    pred_typed = capacity(dim, lambda n: n)
    typed, curve_t = n_star(dim, np.zeros((0, dim), dtype=np.int8), True, pred_typed)
    rows = []
    for extra in G["extra"]:
        mat = distractor_matrix(dim, extra)
        pred_full = capacity(dim, lambda n, e=extra: 2 * n + 13 + e)
        full, curve_f = n_star(dim, mat, False, pred_full)
        rows.append({"part": "G", "extra": extra, "pred_full": pred_full,
                     "measured_full": full, "pred_typed": pred_typed,
                     "measured_typed": typed,
                     "pred_gain": pred_typed / pred_full,
                     "measured_gain": (typed / full) if typed and full else None,
                     "curve_full": curve_f, "curve_typed": curve_t})
        print("G", extra, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        e1 = part_e1(10, 64, 2)
        e2 = part_e2(10, 48, 2)
        a = acc_at(10, 256, distractor_matrix(256, 50), False, 1, 3, check=True)
        print("smoke ok:", e1["bit_samples"], e2["pairs"], "G acc calcolata:", a is not None)
        return
    cells = [part_e1(n, E1["dim"], E1["traces"]) for n in E1["ns"]]
    print("E1 fatto", flush=True)
    cells += [part_e2(c["n"], c["dim"], c["traces"]) for c in E2]
    print("E2 fatto", flush=True)
    cells += part_g()
    OUT.write_text(json.dumps({"preregistration": "docs/preregistration/dependence.md",
                               "cells": cells}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
