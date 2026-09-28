"""prereg_summary.py — intervalli di confidenza sugli errori riassuntivi delle preregistrazioni.

Le preregistrazioni riportano un errore medio (assoluto e con segno) su un
insieme di celle, ciascuna mediata su 10–20 seed, e lo confrontano con una soglia.
Qui ogni errore riassuntivo ha un intervallo bootstrap al 95%: si ricampionano i
seed all'interno di ogni cella (2 000 ricampionamenti), come richiederebbe un
reviewer per sapere se un criterio è stato superato con margine o per un soffio.

Non cambia nessun esito: gli esiti restano quelli valutati con i criteri
preregistrati. Aggiunge l'incertezza che quei criteri non dichiaravano.

Uso, dalla root:  python examples/prereg_summary.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "prereg_summary_results.json"
B = 2000


def summary(cells, key_cell, meas, pred, rng):
    """(|err| medio, CI), (err con segno, CI), in punti, bootstrap sui seed."""
    groups = defaultdict(list)
    for c in cells:
        groups[key_cell(c)].append((meas(c), pred(c)))
    keys = sorted(groups)
    arr = [np.array(groups[k]) for k in keys]

    def stats(samples):
        errs = [100 * (s[:, 1].mean() - s[:, 0].mean()) for s in samples]
        return np.mean(np.abs(errs)), np.mean(errs)

    point = stats(arr)
    # pavimento di rumore: |err| atteso per un modello PERFETTO, dato il solo errore
    # di campionamento delle celle: media di sqrt(2/pi)·SE_cella
    se = [a[:, 0].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0 for a in arr]
    floor = 100 * float(np.mean(se)) * np.sqrt(2 / np.pi)
    boot = np.array([stats([a[rng.randint(len(a), size=len(a))] for a in arr])
                     for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"cells": len(keys), "noise_floor": floor,
            "abs": [float(point[0]), float(lo[0]), float(hi[0])],
            "signed": [float(point[1]), float(lo[1]), float(hi[1])]}


def load(name):
    return json.loads((RES / name).read_text())["cells"]


def main():
    rng = np.random.RandomState(0)
    rows = {}
    fb = load("fb15k237_prereg_results.json")
    for d in (2048, 8192):
        rows[f"1 · FB15k-237 uniforme, D={d}"] = summary(
            [c for c in fb if c["dim"] == d], lambda c: c["n"],
            lambda c: c["measured"], lambda c: c["pred_k092_alias"], rng)
    ex = load("exact_prereg_results.json")
    rows["2 · A, D=16384"] = summary([c for c in ex if c["part"] == "A"], lambda c: c["n"],
                                     lambda c: c["measured"], lambda c: c["pred_exact"], rng)
    for g in (1, 2, 4):
        rows[f"2 · B, g={g}"] = summary([c for c in ex if c["part"] == "B" and c["g"] == g],
                                        lambda c: c["n"], lambda c: c["measured"],
                                        lambda c: c["pred_exact"], rng)
    for d in (2048, 8192):
        rows[f"2 · C denso, D={d}"] = summary([c for c in ex if c["part"] == "C" and c["dim"] == d],
                                              lambda c: c["n"], lambda c: c["measured"],
                                              lambda c: c["pred_exact"], rng)
    tw = load("twins_prereg_results.json")
    for part, kg, samp in (("T1", "fb15k237", "dense"), ("T2", "wn18rr", "dense"),
                           ("T2", "wn18rr", "uniform")):
        for d in (2048, 8192):
            sel = [c for c in tw if c["part"] == part and c.get("sampler") == samp and c["dim"] == d]
            rows[f"4 · {kg} {samp}, D={d}, con gemelli"] = summary(
                sel, lambda c: c["n"], lambda c: c["measured"], lambda c: c["pred_twins"], rng)
            if samp == "dense" and kg == "wn18rr":
                rows[f"4 · {kg} {samp}, D={d}, SENZA gemelli"] = summary(
                    sel, lambda c: c["n"], lambda c: c["measured"], lambda c: c["pred_no_twins"], rng)
    co = load("composition_prereg_results.json")
    for kind in ("missing", "wrong_relation", "wrong_entity", "spurious"):
        rows[f"5 · R1 {kind}"] = summary([c for c in co if c.get("kind") == kind],
                                         lambda c: c["eps"], lambda c: c["measured"],
                                         lambda c: c["pred_exact"], rng)
    for st in ("iid", "chain", "hop2"):
        rows[f"5 · R2 {st}"] = summary([c for c in co if c.get("struct") == st],
                                       lambda c: c["eps"], lambda c: c["measured"],
                                       lambda c: c["pred_exact"], rng)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"{'test':42} {'celle':>5} {'|err|':>6} {'rumore':>7} {'con segno [CI 95%]':>24}  bias")
    for k, v in rows.items():
        a, s = v["abs"], v["signed"]
        bias = "sì" if s[1] > 0 or s[2] < 0 else "no"
        print(f"{k:42} {v['cells']:>5} {a[0]:6.2f} {v['noise_floor']:7.2f}   "
              f"{s[0]:+6.2f} [{s[1]:+5.2f}, {s[2]:+5.2f}]  {bias}")
    print("->", OUT)


if __name__ == "__main__":
    main()
