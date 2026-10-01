"""cycles_residuals.py — le dipendenze GF(2) dei campioni reali spiegano i residui?

ESPLORATIVO, NON preregistrato (audit 2026-09-30, punto 3). Usa solo dati già
pubblicati: le celle KG dense (e, per controllo, uniformi) della preregistrazione
4 (results/twins_prereg_results.json) e le celle C della preregistrazione 2
(results/exact_prereg_results.json), con i campioni e le query rigenerati con lo
stesso codice e gli stessi seed (twins_prereg.kg_cell, exact_prereg.part_c).
Le preregistrazioni non si toccano; nessuna nuova misura della reference.

Quali insiemi di fatti sono dipendenti. Il fatto (s, r, o) è il vettore
c_s ⊕ ρ(c_r) ⊕ c_o (bipolare: prodotto). Come insieme di simboli è {s, ρr, o};
un self-loop (s, r, s) è {ρr}. Un insieme di fatti DISTINTI ha XOR nullo su ogni
bit, per ogni scelta dei codeword, se e solo se ogni simbolo vi compare un numero
pari di volte (ρ(c_r) compreso). I gemelli (s,r,o)/(o,r,s) e i self-loop della
stessa relazione sono lo stesso vettore e il modello li conta già (peso 2, alias):
qui si lavora sui vettori distinti. Con tre fatti la dipendenza è impossibile
(3 occorrenze di relazione, dispari, salvo self-loop, che però danno insiemi di
simboli diversi); la più corta è di 4 fatti. Le 4-dipendenze sono esattamente:
  - i 4-cicli del grafo delle entità (orientamento ignorato, XOR è simmetrico)
    le cui 4 relazioni si appaiano (r1=r2 e r3=r4 in qualche accoppiamento),
    p.es. il rettangolo (a,r,b),(a,r,c),(d,r,b),(d,r,c) o (a,r,b),(a,q,c),(d,r,b),(d,q,c);
  - le coppie di archi paralleli con le stesse due relazioni:
    (a,r,b),(a,q,b),(c,r,d),(c,q,d);
  - combinazioni con self-loop: ρr, ρq, (a,r,b), (a,q,b).
Si contano tutte enumerando le coppie: A⊕B⊕C⊕D = 0 sse A⊕B = C⊕D, ogni 4-insieme
compare sotto 3 accoppiamenti. Misure per cella:
  dep4_per_fact — 4 · (#4-dipendenze) / (#vettori distinti): partecipazioni medie;
  dep4_share    — quota di vettori distinti in almeno una 4-dipendenza;
  deficit_share — (#vettori distinti − rango su GF(2)) / #vettori distinti, che
                  conta anche i cicli pari più lunghi.
Per ogni query anche q_dep4: partecipazioni medie dei suoi fatti corretti.

Errore per cella: previsione attuale di abm.exact.predict_queries (gemelli e
self-loop) meno l'accuratezza misurata pubblicata, in punti. Correlazione di
Pearson e Spearman, grezza e entro gruppo (kg, sampler, D, n) — gli scarti dalla
media di gruppo, perché carico e cicli crescono insieme — con intervallo bootstrap
95% sulle celle (2000 ricampionamenti, seed 0).

Atteso a priori (cycles_probe.py): il modello indipendente è OTTIMISTA sui cicli,
quindi più cicli → errore più positivo. Il bias denso di FB15k-237 a D = 2048 è
negativo (−0.81): se i cicli lo spiegassero, la correlazione dovrebbe avere il
segno opposto a quello atteso. Il residuo sugli hub (+3.3, test 2) ha il segno
atteso; qui si guarda se gli hub hanno più 4-dipendenze.

Per gli hub si ricalcola anche la statistica H5 del test 2 (pesata per query,
misurato − previsto, hub meno resto): con la previsione pubblicata deve tornare
+3.28; con quella attuale (gemelli e self-loop) cambia.

Scrive results/cycles_residuals_results.json. Uso, dalla root:
    python examples/cycles_residuals.py
(--root PERCORSO per usare un'altra radice del repo; --out per un altro file.)
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap_ = argparse.ArgumentParser()
ap_.add_argument("--root", default=None)
ap_.add_argument("--out", default=None)
ARGS = ap_.parse_args()
if ARGS.root:
    ROOT = Path(ARGS.root).resolve()
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import exact  # noqa: E402
import importlib.util as _ilu  # noqa: E402
import subprocess as _sp  # noqa: E402
import tempfile as _tf  # noqa: E402

# il modello con i gemelli ma senza i self-loop: abm.exact prima del commit 2a41f74
_old_src = _sp.run(["git", "show", "c2d7816:reference/exact.py"], cwd=ROOT, check=True,
                   capture_output=True, text=True).stdout
with _tf.NamedTemporaryFile("w", suffix="_exact_c2d7816.py", delete=False) as _f:
    _f.write(_old_src)
_spec = _ilu.spec_from_file_location("exact_c2d7816", _f.name)
exact_twins_only = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exact_twins_only)
import exact_prereg as ep  # noqa: E402
import twins_prereg as tp  # noqa: E402

OUT = Path(ARGS.out) if ARGS.out else ROOT / "results" / "cycles_residuals_results.json"
BOOT = 2000


def symbols(s, r, o):
    return frozenset([("r", r)]) if s == o else frozenset([("e", s), ("r", r), ("e", o)])


def dependencies(sample):
    """Vettori distinti, partecipazioni alle 4-dipendenze per vettore, deficit di rango."""
    vecs = sorted({symbols(*t) for t in sample}, key=lambda v: sorted(v))
    n = len(vecs)
    by_xor = defaultdict(list)
    for i, j in combinations(range(n), 2):
        by_xor[vecs[i] ^ vecs[j]].append((i, j))
    part = np.zeros(n)
    total = 0
    for pairs in by_xor.values():
        c = len(pairs)
        if c < 2:
            continue
        total += c * (c - 1) // 2
        for i, j in pairs:
            part[i] += c - 1
            part[j] += c - 1
    part /= 3.0
    total //= 3
    # rango su GF(2) con interi come bitset
    sym_id = {}
    basis = {}
    rank = 0
    for v in vecs:
        x = 0
        for sym in v:
            x |= 1 << sym_id.setdefault(sym, len(sym_id))
        while x:
            h = x.bit_length() - 1
            if h in basis:
                x ^= basis[h]
            else:
                basis[h] = x
                rank += 1
                break
    index = {v: i for i, v in enumerate(vecs)}
    return {"distinct": n, "dep4": int(total), "dep4_per_fact": float(4 * total / n),
            "dep4_share": float(np.mean(part > 0)), "deficit_share": float((n - rank) / n),
            "part": part, "index": index}


def queries_of(sample, rng):
    objects = defaultdict(set)
    for s, r, o in sample:
        objects[(s, r)].add(o)
    keys = list(objects)
    idx = range(len(keys)) if len(keys) <= tp.Q_MAX else rng.choice(len(keys), tp.Q_MAX,
                                                                     replace=False)
    return [keys[i] for i in idx], objects


def q_dep(dep, objects, q):
    s, r = q
    return float(np.mean([dep["part"][dep["index"][symbols(s, r, o)]] for o in objects[q]]))


def twins_cells(fb, wn):
    published = json.loads((ROOT / "results" / "twins_prereg_results.json").read_text())
    rows = []
    for c in published["cells"]:
        if c["part"] not in ("T1", "T2"):
            continue
        kg = fb if c["kg"] == "fb15k237" else wn
        triples, incident = kg
        rng = np.random.RandomState(1299709 * c["seed"] + c["n"] + (7 if c["kg"] == "wn18rr" else 0))
        if c["sampler"] == "dense":
            sample = ep.dense_sample(triples, incident, c["n"], rng)
        else:
            sample = [triples[i] for i in rng.choice(len(triples), c["n"], replace=False)]
        queries, _objects = queries_of(sample, rng)
        pred = float(np.mean(exact.predict_queries(sample, c["dim"], queries)))
        dep = dependencies(sample)
        rows.append({"src": "twins", "kg": c["kg"], "sampler": c["sampler"], "dim": c["dim"],
                     "n": c["n"], "seed": c["seed"], "measured": c["measured"],
                     "pred_current": pred, "pred_published": c["pred_twins"],
                     "err": 100 * (pred - c["measured"]),
                     **{k: dep[k] for k in ("distinct", "dep4", "dep4_per_fact", "dep4_share",
                                            "deficit_share")}})
        print(c["kg"], c["sampler"], c["dim"], c["n"], c["seed"], dep["dep4"], flush=True)
    return rows


def part_c_cells(fb):
    """Test 2, parte C: stesso campione, query e taglio hub (75° percentile del grado)."""
    published = json.loads((ROOT / "results" / "exact_prereg_results.json").read_text())
    triples, incident = fb
    rows = []
    for c in published["cells"]:
        if c["part"] != "C":
            continue
        rng = np.random.RandomState(1299709 * c["seed"] + c["n"])
        sample = ep.dense_sample(triples, incident, c["n"], rng)
        queries, objects = queries_of(sample, rng)
        preds = exact.predict_queries(sample, c["dim"], queries)
        preds_tw = exact_twins_only.predict_queries(sample, c["dim"], queries)
        deg = defaultdict(int)
        for s, _r, o in sample:
            deg[s] += 1
            deg[o] += 1
        degs = [deg[s] for s, _r in queries]
        cut = np.percentile(degs, 75)
        dep = dependencies(sample)
        qd = [q_dep(dep, objects, q) for q in queries]
        hub = [i for i, d in enumerate(degs) if d > cut]
        rest = [i for i, d in enumerate(degs) if d <= cut]
        assert len(hub) == c["hub"]["n"] and len(rest) == c["rest"]["n"]
        row = {"src": "exact_C", "kg": "fb15k237", "sampler": "dense", "dim": c["dim"],
               "n": c["n"], "seed": c["seed"], "measured": c["measured"],
               "pred_current": float(np.mean(preds)),
               "err": 100 * (float(np.mean(preds)) - c["measured"]),
               **{k: dep[k] for k in ("distinct", "dep4", "dep4_per_fact", "dep4_share",
                                      "deficit_share")}}
        for name, sel in (("hub", hub), ("rest", rest)):
            if not sel:
                row[name] = None
                continue
            pr = float(np.mean([preds[i] for i in sel]))
            row[name] = {"n": len(sel), "measured": c[name]["measured"], "pred_current": pr,
                         "pred_twins_only": float(np.mean([preds_tw[i] for i in sel])),
                         "pred_published": c[name]["pred"],
                         "err": 100 * (pr - c[name]["measured"]),
                         "q_dep4": float(np.mean([qd[i] for i in sel])),
                         "q_in_dep4": float(np.mean([qd[i] > 0 for i in sel]))}
        rows.append(row)
        print("C", c["dim"], c["n"], c["seed"], dep["dep4"], flush=True)
    return rows


def rankdata(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = np.asarray(x)[o]
    i = 0
    while i < len(xs):                       # ranghi medi sui pareggi
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def demean(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[(r["kg"], r["sampler"], r["dim"], r["n"])].append(r[key])
    m = {k: np.mean(v) for k, v in g.items()}
    return [r[key] - m[(r["kg"], r["sampler"], r["dim"], r["n"])] for r in rows]


def corr_block(rows, xkey):
    rng = np.random.RandomState(0)
    x, y = [r[xkey] for r in rows], [r["err"] for r in rows]
    xw, yw = demean(rows, xkey), demean(rows, "err")
    out = {"cells": len(rows),
           "pearson": pearson(x, y), "spearman": pearson(rankdata(x), rankdata(y)),
           "pearson_within": pearson(xw, yw),
           "slope_within": float(np.polyfit(xw, yw, 1)[0]) if np.std(xw) > 0 else None}
    bs = {"pearson": [], "pearson_within": []}
    n = len(rows)
    for _ in range(BOOT):
        i = rng.randint(n, size=n)
        bs["pearson"].append(pearson([x[k] for k in i], [y[k] for k in i]))
        bs["pearson_within"].append(pearson([xw[k] for k in i], [yw[k] for k in i]))
    for k, v in bs.items():
        v = np.array(v)
        v = v[~np.isnan(v)]
        out[k + "_ci95"] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                            if len(v) else None)
    return out


def summarise(rows):
    groups = {}
    for kg in ("fb15k237", "wn18rr"):
        for sampler in ("dense", "uniform"):
            for dim in (2048, 8192):
                sel = [r for r in rows if r["kg"] == kg and r["sampler"] == sampler and r["dim"] == dim]
                if not sel:
                    continue
                key = f"{kg} {sampler} {dim}"
                groups[key] = {"cells": len(sel),
                               "mean_err": float(np.mean([r["err"] for r in sel])),
                               **{f"mean_{k}": float(np.mean([r[k] for r in sel]))
                                  for k in ("dep4_per_fact", "dep4_share", "deficit_share")},
                               "corr": {k: corr_block(sel, k)
                                        for k in ("dep4_per_fact", "deficit_share")}}
    return groups


def main():
    fb, wn = ep.load_fb15k(), tp.load_wn18rr()
    trows = twins_cells(fb, wn)
    crows = part_c_cells(fb)
    dense = [r for r in trows if r["sampler"] == "dense"]
    hub_pairs = [r for r in crows if r["hub"] and r["rest"]]
    rng = np.random.RandomState(0)
    d_err = np.array([r["hub"]["err"] - r["rest"]["err"] for r in hub_pairs])
    d_dep = np.array([r["hub"]["q_dep4"] - r["rest"]["q_dep4"] for r in hub_pairs])
    bs = []
    for _ in range(BOOT):
        i = rng.randint(len(d_err), size=len(d_err))
        bs.append(pearson(d_dep[i], d_err[i]))
    bs = np.array(bs)
    bs = bs[~np.isnan(bs)]
    hubs = {
        "cells": len(hub_pairs),
        "hub_err_mean": float(np.mean([r["hub"]["err"] for r in hub_pairs])),
        "rest_err_mean": float(np.mean([r["rest"]["err"] for r in hub_pairs])),
        "hub_err_published_mean": float(np.mean([100 * (r["hub"]["pred_published"] - r["hub"]["measured"])
                                                 for r in hub_pairs])),
        "hub_q_dep4_mean": float(np.mean([r["hub"]["q_dep4"] for r in hub_pairs])),
        "rest_q_dep4_mean": float(np.mean([r["rest"]["q_dep4"] for r in hub_pairs])),
        "hub_q_in_dep4": float(np.mean([r["hub"]["q_in_dep4"] for r in hub_pairs])),
        "rest_q_in_dep4": float(np.mean([r["rest"]["q_in_dep4"] for r in hub_pairs])),
        "corr_diff_dep_vs_diff_err": pearson(d_dep, d_err),
        "corr_diff_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "hub_err_by_dim": {d: float(np.mean([r["hub"]["err"] for r in hub_pairs if r["dim"] == d]))
                           for d in (2048, 8192)},
    }
    def h5(rs, key):
        """H5 del test 2: (misurato − previsto)_hub − (misurato − previsto)_resto, pesato per query."""
        out = []
        for part in ("hub", "rest"):
            n = sum(r[part]["n"] for r in rs)
            out.append(100 * sum(r[part]["n"] * (r[part]["measured"] - r[part][key]) for r in rs) / n)
        return out[0] - out[1], out[0], out[1]
    boot = {"pred_current": [], "pred_twins_only": []}
    for _ in range(BOOT):
        i = rng.randint(len(hub_pairs), size=len(hub_pairs))
        for key in boot:
            boot[key].append(h5([hub_pairs[k] for k in i], key)[0])
    for key in ("pred_published", "pred_twins_only", "pred_current"):
        d, a, b = h5(hub_pairs, key)
        hubs[f"H5_{key}"] = {"diff": d, "hub_meas_minus_pred": a, "rest_meas_minus_pred": b}
    for key, v in boot.items():
        hubs[f"H5_{key}"]["diff_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {"exploratory": "non preregistrato; dati e campioni già pubblicati (prereg 2 e 4)",
           "dependency_definition": "insiemi di vettori distinti s⊕ρ(r)⊕o con XOR nullo "
                                    "(ogni simbolo, ρ(c_r) compreso, un numero pari di volte)",
           "twins_cells": summarise(trows),
           "dense_pooled": {k: corr_block(dense, k) for k in ("dep4_per_fact", "deficit_share")},
           "exact_C_cells": summarise(crows),
           "hubs_test2": hubs,
           "rows_twins": trows,
           "rows_exact_C": crows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("->", OUT)


if __name__ == "__main__":
    main()
