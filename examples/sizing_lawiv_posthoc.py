"""sizing_lawiv_posthoc.py — post hoc arms for test 10 (audit 2026-09-30, item 10).

NOT preregistered. The preregistered Law IV arm of sizing_prereg.py has neither
aliases nor twins. Here, on the same 80 subgraphs and targets, D is chosen with
two strengthened asymptotic laws, and accuracy is measured at that D:
  - law_iv_alias: Law IV (k = 0.92) x g/(g+a) per query;
  - law_iv_alias_twins: the same, with a twin of weight w given signal
    w * sqrt(2kD/(pi W)), W = total weight (a naive asymptotic twin term).
Also stores n_queries per subgraph.

Usage, from the root:  python examples/sizing_lawiv_posthoc.py
"""
import json
import sys
from math import erf, pi, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("reference", "", "examples"):
    sys.path.insert(0, str(ROOT / p))
import abm  # noqa: E402
import exact_prereg as ep  # noqa: E402
import sizing_prereg as sp  # noqa: E402
import twins_prereg as tp  # noqa: E402
from exact import fact_key, fact_weights  # noqa: E402

OUT = ROOT / "results" / "sizing_lawiv_posthoc_results.json"


def make_pred(twins):
    def pred(st, dim):
        sample, queries, objects, into, m, _ = st
        w = fact_weights(sample)
        total = sum(w.values()) if twins else len(sample)
        zg = abm.z_gumbel(max(m, 3))
        base = sqrt(2 * 0.92 * dim / (pi * total))
        acc = []
        for s, r in queries:
            good, bad = objects[(s, r)], into[(s, r)] - objects[(s, r)]
            g, a = len(good), len(bad)
            phi = [0.5 * (1 + erf(((w[fact_key(s, r, o)] if twins else 1) * base - zg) / sqrt(2)))
                   for o in good]
            acc.append(np.mean(phi) * g / (g + a))
        return float(np.mean(acc))
    return pred


ARMS = {"law_iv_alias": make_pred(False), "law_iv_alias_twins": make_pred(True)}


def main():
    kgs = {"fb15k237": ep.load_fb15k(), "wn18rr": tp.load_wn18rr()}
    old = {(r["kg"], r["n"], r["sample"]): r
           for r in json.loads(sp.OUT.read_text())["rows"]}
    rows = []
    for name, kg in kgs.items():
        for n in sp.NS:
            for k in range(sp.SAMPLES):
                st = sp.structure(kg, name, n, sp.SEED0 + k)
                o = old[(name, n, k)]
                assert st[4] == o["codebook"] and abs(st[5] - o["ceiling"]) < 1e-12
                row = {"kg": name, "n": n, "sample": k, "n_queries": len(st[1])}
                for t in sp.TARGETS:
                    for label, pred in ARMS.items():
                        d = sp.min_dim(st, pred, t)
                        row[f"{label}_{t}"] = {
                            "dim": d, "predicted": pred(st, d) if d else None,
                            "measured": sp.measure(st, d) if d else None}
                rows.append(row)
            print(name, n, flush=True)
    OUT.write_text(json.dumps({"post_hoc": True, "same_subgraphs_as": "results/sizing_prereg_results.json",
                               "rows": rows}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
