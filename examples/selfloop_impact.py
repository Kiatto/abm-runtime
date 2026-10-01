"""selfloop_impact.py — quanto cambia la previsione con i self-loop modellati.

ESPLORATIVO, non preregistrato. Il 2026-09-30 (commit 2a41f74) abm.exact ha
iniziato a modellare i self-loop (s, r, s): valgono ρ(c_r) qualunque sia s, quindi
sono tutti lo stesso vettore e ogni query (x, r) vede x come alias. L'audit diceva
che FB15k-237 non ne ha; ne ha 1625 (lo 0.6% delle triple, soprattutto
educational_institution/campuses, educational_institution_campus/educational_institution
e hud_county_place/place). Le preregistrazioni 4 (twins), 6 (asymmetric) e 10
(sizing) sono state calcolate prima della correzione, e i loro harness, congelati,
non girano più con il modello attuale.

Qui si rigenerano, con lo stesso codice e gli stessi seed, i campioni e le query
delle celle KG della preregistrazione 4, e per ognuna si confrontano:
  published — pred_twins nel file pubblicato;
  current   — abm.exact.predict_queries di adesso, sulle stesse query;
  measured  — l'accuratezza misurata pubblicata (la reference non è cambiata).
Controllo: nelle celle senza self-loop current deve coincidere con published.

Per la preregistrazione 10 (sizing), sui sottografi che contengono self-loop, si
rifà la scelta della D con il modello attuale, si misura a quella D e si guarda
se la promessa (misurata >= obiettivo) è mantenuta.

Scrive results/selfloop_impact_results.json. Uso, dalla root:
    python examples/selfloop_impact.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import exact  # noqa: E402
import exact_prereg as ep  # noqa: E402
import sizing_prereg as sp  # noqa: E402
import twins_prereg as tp  # noqa: E402

OUT = ROOT / "results" / "selfloop_impact_results.json"


def cell(kg, name, sampler, dim, n, seed):
    """Lo stesso campione e le stesse query di twins_prereg.kg_cell."""
    triples, incident = kg
    rng = np.random.RandomState(1299709 * seed + n + (7 if name == "wn18rr" else 0))
    if sampler == "dense":
        sample = ep.dense_sample(triples, incident, n, rng)
    else:
        sample = [triples[i] for i in rng.choice(len(triples), n, replace=False)]
    objects = defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= tp.Q_MAX else rng.choice(len(keys), tp.Q_MAX,
                                                                     replace=False)
    queries = [keys[i] for i in idx]
    loops = sum(s == o for s, _r, o in sample)
    return loops, float(np.mean(exact.predict_queries(sample, dim, queries)))


def pred_current(st, dim):
    sample, queries, *_ = st
    return float(np.mean(exact.predict_queries(sample, dim, queries)))


def sizing_rows(kgs):
    published = json.loads((ROOT / "results" / "sizing_prereg_results.json").read_text())
    rows, check = [], 0.0
    for row in published["rows"]:
        st = sp.structure(kgs[row["kg"]], row["kg"], row["n"], sp.SEED0 + row["sample"])
        loops = sum(s == o for s, _r, o in st[0])
        for t in sp.TARGETS:
            old = row[f"exact_{t}"]
            if not loops:
                if old["dim"]:
                    check = max(check, abs(pred_current(st, old["dim"]) - old["predicted"]))
                continue
            d = sp.min_dim(st, pred_current, t)
            new = {"dim": d, "predicted": pred_current(st, d) if d else None,
                   "measured": (old["measured"] if d == old["dim"] else sp.measure(st, d))
                   if d else None}
            rows.append({"kg": row["kg"], "n": row["n"], "sample": row["sample"], "target": t,
                         "self_loops": loops, "published": old, "current": new,
                         "kept_published": old["measured"] is not None and old["measured"] >= t,
                         "kept_current": new["measured"] is not None and new["measured"] >= t})
    return rows, check


def main():
    published = json.loads((ROOT / "results" / "twins_prereg_results.json").read_text())
    pub = {(c["kg"], c["sampler"], c["dim"], c["n"], c["seed"]): c
           for c in published["cells"] if c["part"] in ("T1", "T2")}
    fb, wn = ep.load_fb15k(), tp.load_wn18rr()
    rows = []
    for (kg, sampler, dim, n, seed), c in pub.items():
        loops, current = cell(fb if kg == "fb15k237" else wn, kg, sampler, dim, n, seed)
        rows.append({"kg": kg, "sampler": sampler, "dim": dim, "n": n, "seed": seed,
                     "self_loops": loops, "measured": c["measured"],
                     "published": c["pred_twins"], "current": current})
    clean = [r for r in rows if r["self_loops"] == 0]
    check = max(abs(r["current"] - r["published"]) for r in clean)
    summary = {}
    for kg in ("fb15k237", "wn18rr"):
        sel = [r for r in rows if r["kg"] == kg]
        hit = [r for r in sel if r["self_loops"]]
        summary[kg] = {
            "cells": len(sel), "cells_with_self_loops": len(hit),
            "mean_err_published": float(np.mean([r["published"] - r["measured"] for r in sel])),
            "mean_err_current": float(np.mean([r["current"] - r["measured"] for r in sel])),
            "mean_abs_err_published": float(np.mean([abs(r["published"] - r["measured"]) for r in sel])),
            "mean_abs_err_current": float(np.mean([abs(r["current"] - r["measured"]) for r in sel])),
            "max_abs_shift": float(max((abs(r["current"] - r["published"]) for r in sel),
                                       default=0.0))}
    srows, scheck = sizing_rows({"fb15k237": fb, "wn18rr": wn})
    OUT.write_text(json.dumps({"exploratory": True, "check_max_diff_no_loops": check,
                               "summary": summary, "rows": rows,
                               "sizing": {"check_max_diff_no_loops": scheck, "rows": srows}},
                              indent=1))
    print("controllo, celle senza self-loop: max |current - published| =", check)
    print(json.dumps(summary, indent=1))
    print("sizing, controllo senza self-loop:", scheck)
    for r in srows:
        print(r["kg"], r["n"], r["sample"], r["target"], "loops", r["self_loops"],
              "D", r["published"]["dim"], "->", r["current"]["dim"],
              "mis", r["published"]["measured"], "->", r["current"]["measured"],
              "promessa", r["kept_published"], "->", r["kept_current"])
    print("->", OUT)


if __name__ == "__main__":
    main()
