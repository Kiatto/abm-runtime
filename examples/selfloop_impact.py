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

Per la preregistrazione 6 (asymmetric) si ricalcola la previsione simmetrica di
ogni cella (media sui seed 20–29) e si rivalutano, con i criteri del file di
preregistrazione, le due ipotesi che ne dipendono: H2 (errore del simmetrico,
sostenuta se ≤ 2.5, falsificata se > 5) e H4 (errore sulla differenza su WN18RR,
stesse soglie). H1 e H3 non usano la previsione simmetrica.

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
import asymmetric_prereg as ap  # noqa: E402
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


def twins_hypotheses(rows, published):
    """H1–H4 della preregistrazione 4, con la previsione pubblicata e con quella
    attuale, sulle celle mediate sui seed. H5 (pesi sintetici) non ha self-loop."""
    nt = {(c["kg"], c["sampler"], c["dim"], c["n"], c["seed"]): (c["pred_no_twins"], c["twin_share"])
          for c in published["cells"] if c["part"] in ("T1", "T2")}
    out = {}
    for which in ("published", "current"):
        cells = defaultdict(list)
        for r in rows:
            cells[(r["kg"], r["sampler"], r["dim"], r["n"])].append(r)

        def err(kg, sampler, dim, key):
            e = []
            for (k, s_, d, _n), rs in cells.items():
                if (k, s_, d) == (kg, sampler, dim):
                    pred = np.mean([nt[(r["kg"], r["sampler"], r["dim"], r["n"], r["seed"])][0]
                                    if key == "no_twins" else r[which] for r in rs])
                    e.append(100 * (pred - np.mean([r["measured"] for r in rs])))
            return float(np.mean(np.abs(e))), float(np.mean(e))
        res = {}
        for dim in (2048, 8192):
            a, b = err("fb15k237", "dense", dim, "twins")
            res[f"H1 {dim}"] = {"abs": a, "signed": b,
                                "verdict": "sostenuta" if a <= 2 and abs(b) <= 1.5 else
                                "falsificata" if a > 5 or abs(b) > 3 else "in parte"}
            a, b = err("wn18rr", "dense", dim, "twins")
            res[f"H2 {dim}"] = {"abs": a, "signed": b,
                                "verdict": "sostenuta" if a <= 3 and abs(b) <= 2 else
                                "falsificata" if a > 6 or abs(b) > 4 else "in parte"}
            na, nb = err("wn18rr", "dense", dim, "no_twins")
            res[f"H3 {dim}"] = {"no_twins_signed": nb, "no_twins_abs": na, "twins_abs": a}
            a, _b = err("wn18rr", "uniform", dim, "twins")
            share = float(np.mean([v[1] for k, v in nt.items() if k[:3] == ("wn18rr", "uniform", dim)]))
            res[f"H4 {dim}"] = {"abs": a, "twin_share": share,
                                "verdict": "sostenuta" if share < 0.01 and a <= 3 else
                                "falsificata" if a > 6 else "in parte"}
        h3 = [res[f"H3 {d}"] for d in (2048, 8192)]
        res["H3"] = ("sostenuta" if all(h["no_twins_signed"] < -2 and h["twins_abs"] < h["no_twins_abs"]
                                         for h in h3) else
                     "falsificata" if any(h["no_twins_signed"] >= 0 for h in h3) else "in parte")
        out[which] = res
    return out


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


def asymmetric_rows(kgs):
    published = json.loads((ROOT / "results" / "asymmetric_prereg_results.json").read_text())
    rows, check = [], 0.0
    for c in published["cells"]:
        preds, loops = [], 0
        for seed in ap.SEEDS:
            sample, queries, _o, _i = ap.sample_of(kgs[c["kg"]], c["kg"], c["dim"], c["n"], seed)
            loops += sum(s == o for s, _r, o in sample)
            preds.append(float(np.mean(exact.predict_queries(sample, c["dim"], queries))))
        cur = float(np.mean(preds))
        if not loops:
            check = max(check, abs(cur - c["pred_sym"]))
        rows.append({"kg": c["kg"], "dim": c["dim"], "n": c["n"], "self_loops": loops,
                     "pred_sym_published": c["pred_sym"], "pred_sym_current": cur,
                     "pred_asym": c["pred_asym"], "meas_sym": c["meas_sym"],
                     "meas_asym": c["meas_asym"]})

    def verdict(err):
        return "sostenuta" if err <= 2.5 else "falsificata" if err > 5 else "in parte"
    hyp = {}
    for kg in ("fb15k237", "wn18rr"):
        for dim in (2048, 8192):
            sel = [r for r in rows if r["kg"] == kg and r["dim"] == dim]
            for which in ("published", "current"):
                h2 = 100 * np.mean([abs(r[f"pred_sym_{which}"] - r["meas_sym"]) for r in sel])
                key = f"{kg} {dim} {which}"
                hyp[key] = {"H2": float(h2), "H2_verdict": verdict(h2)}
                if kg == "wn18rr":
                    h4 = 100 * np.mean([abs((r[f"pred_sym_{which}"] - r["pred_asym"])
                                            - (r["meas_sym"] - r["meas_asym"])) for r in sel])
                    hyp[key].update({"H4": float(h4), "H4_verdict": verdict(h4)})
    return rows, check, hyp


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
    thyp = twins_hypotheses(rows, published)
    srows, scheck = sizing_rows({"fb15k237": fb, "wn18rr": wn})
    arows, acheck, ahyp = asymmetric_rows({"fb15k237": fb, "wn18rr": wn})
    OUT.write_text(json.dumps({"exploratory": True, "check_max_diff_no_loops": check,
                               "summary": summary, "hypotheses": thyp, "rows": rows,
                               "sizing": {"check_max_diff_no_loops": scheck, "rows": srows},
                               "asymmetric": {"check_max_diff_no_loops": acheck,
                                              "hypotheses": ahyp, "rows": arows}},
                              indent=1))
    print("controllo, celle senza self-loop: max |current - published| =", check)
    print(json.dumps(summary, indent=1))
    for which, res in thyp.items():
        print("twins", which)
        for k, v in res.items():
            print("  ", k, v if isinstance(v, str) else
                  {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    print("sizing, controllo senza self-loop:", scheck)
    for r in srows:
        print(r["kg"], r["n"], r["sample"], r["target"], "loops", r["self_loops"],
              "D", r["published"]["dim"], "->", r["current"]["dim"],
              "mis", r["published"]["measured"], "->", r["current"]["measured"],
              "promessa", r["kept_published"], "->", r["kept_current"])
    print("asymmetric, controllo senza self-loop:", acheck)
    for r in arows:
        if r["self_loops"]:
            print(" ", r["kg"], r["dim"], r["n"], "loops", r["self_loops"], "pred_sym",
                  round(100 * r["pred_sym_published"], 2), "->", round(100 * r["pred_sym_current"], 2),
                  "mis", round(100 * r["meas_sym"], 2))
    for k, v in ahyp.items():
        print(" ", k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    print("->", OUT)


if __name__ == "__main__":
    main()
