"""fks_gaussian_vs_exact.py — la previsione gaussiana di Frady, Kleyko & Sommer
(2018) contro il modello binomiale esatto (abm.exact), sulle misure già pubblicate.

Audit 2026-09-30, punto 2 e sezione 4: FKS danno già, a M finito e senza parametri
liberi, l'accuratezza del cleanup come

    Acc = ∫ φ(x) · Φ((D/2 − μ_c − σ_c x) / σ_n)^(M−1) dx,

con le distanze di Hamming approssimate da gaussiane. Qui la loro formula è
istanziata per il bundle a maggioranza MAP-B con la STESSA p_agree del nostro
modello (quindi l'unica differenza è gaussiana contro binomiale + regola dei
pareggi):

- distanza dal codeword giusto ~ N(D q, D q (1−q)), q = 1 − p_agree(N);
- distanze dei distrattori ~ N(D/2, D/4), indipendenti;
- con g oggetti veri (test 2, parte B) vince il minimo di g segnali gaussiani
  contro M − g nulle; nessun alias. La parte C (alias, hub, campioni FB15k-237)
  non è trattata: lì la formula gaussiana andrebbe estesa a candidati misti, e
  non è la formula di FKS.

Variante di controllo: la stessa gaussiana con correzione di continuità (soglia
spostata di 1/2 bit), per separare l'effetto "discreto/pareggi" dall'effetto
"forma della coda".

Analisi ESPLORATIVA su dati già pubblicati: nessuna preregistrazione è toccata,
nessuna soglia è stata fissata prima di guardare.

Celle: capacity_seed10 (N* al 50%, D = 512–4096, M = 2N + 11, stessa griglia e
interpolazione dello script originale), test 2 parti A (D = 16384) e B (D = 4096,
g = 1, 2, 4) da results/exact_prereg_results.json, e la cella D = 256, N = 20
del test di bsm/tests/test_exact_contract.py, rimisurata qui con lo stesso
protocollo (è deterministica).

Scrive results/fks_gaussian_vs_exact_results.json (o il percorso di --out).
Uso, dalla root:  python examples/fks_gaussian_vs_exact.py [--root R] [--out F]
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("BITKORE_ROOT", Path(__file__).resolve().parent.parent))

_ERFC = np.frompyfunc(math.erfc, 1, 1)
# griglia fine per l'integrale (la potenza M−1 rende l'integrando ripido)
_X = np.linspace(-12, 12, 24001)
_DX = _X[1] - _X[0]


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _log_ndtr_upper(z):
    """log P(Z > z) stabile, via erfc."""
    sf = 0.5 * _ERFC(np.asarray(z, dtype=np.float64) / sqrt(2)).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.log(sf)


def fks_accuracy(n_facts, dim, codebook, correct=1, continuity=False):
    """Previsione gaussiana FKS: P(il minimo dei `correct` segnali batte le nulle)."""
    q = 1.0 - exact.p_agree(n_facts)
    mu_c, sd_c = dim * q, sqrt(dim * q * (1 - q))
    mu_n, sd_n = dim / 2, sqrt(dim / 4)
    g, n_null = correct, codebook - correct
    d = mu_c + sd_c * _X                       # distanza (minima) del segnale
    shift = 0.5 if continuity else 0.0
    # densità del minimo di g gaussiane standardizzate: g φ(x) (1−Φ(x))^(g−1)
    log_dens = np.log(g) + np.log(_phi(_X) + 1e-300) + (g - 1) * _log_ndtr_upper(_X)
    log_win = n_null * _log_ndtr_upper((d + shift - mu_n) / sd_n)
    return float(np.sum(np.exp(log_dens + log_win)) * _DX)


def crossing(ns, accs):
    for (n0, a0), (n1, a1) in zip(zip(ns, accs), zip(ns[1:], accs[1:])):
        if a0 >= 0.5 > a1:
            return n0 + (a0 - 0.5) * (n1 - n0) / (a0 - a1)
    return float("nan")


def capacity_cells(seed10):
    out = []
    for dim in (512, 1024, 2048, 4096):
        centre = 0.068 * dim + 15
        ns = [int(round(centre * f)) for f in np.linspace(0.6, 1.5, 19)]
        rec = seed10[str(dim)]
        ex = crossing(ns, [exact.cleanup_accuracy(n, dim, 2 * n + 11) for n in ns])
        ga = crossing(ns, [fks_accuracy(n, dim, 2 * n + 11) for n in ns])
        gc = crossing(ns, [fks_accuracy(n, dim, 2 * n + 11, continuity=True) for n in ns])
        meas, ci = rec["measured"]
        out.append({"dim": dim, "measured": meas, "ci95": ci, "exact_published": rec["exact"],
                    "exact": ex, "gauss": ga, "gauss_cc": gc,
                    "law_iv_k1": abm.capacity(dim, int(round(2 * meas + 11)), k=1.0),
                    "err_exact": ex - meas, "err_gauss": ga - meas, "err_gauss_cc": gc - meas,
                    "rel_err_exact_pct": 100 * (ex - meas) / meas,
                    "rel_err_gauss_pct": 100 * (ga - meas) / meas})
    return out


def prereg_cells(prereg):
    groups = defaultdict(list)
    for c in prereg["cells"]:
        if c["part"] in ("A", "B"):
            groups[(c["part"], c["dim"], c["n"], c.get("g", 1), c["codebook"])].append(c)
    out = []
    for (part, dim, n, g, m), cs in sorted(groups.items()):
        hits = sum(c["measured"] * c["queries"] for c in cs)
        q = sum(c["queries"] for c in cs)
        meas = hits / q
        ex = exact.cleanup_accuracy(n, dim, m, correct=g)
        assert abs(ex - cs[0]["pred_exact"]) < 1e-9
        ga = fks_accuracy(n, dim, m, correct=g)
        gc = fks_accuracy(n, dim, m, correct=g, continuity=True)
        se = sqrt(max(ex * (1 - ex), 1e-12) / q)
        out.append({"part": part, "dim": dim, "n": n, "g": g, "codebook": m, "queries": q,
                    "measured": meas, "se": se, "exact": ex, "gauss": ga, "gauss_cc": gc,
                    "gauss_minus_exact_pts": 100 * (ga - ex),
                    "err_exact_pts": 100 * (ex - meas), "err_gauss_pts": 100 * (ga - meas),
                    "err_gauss_cc_pts": 100 * (gc - meas)})
    return out


def d256_cell():
    """Stesso protocollo di test_exact_reference_d256 (bsm/tests/test_exact_contract.py)."""
    dim, n, trials = 256, 20, 40
    hits = total = 0
    m_seen = []
    for t in range(trials):
        mem = abm.Memory(dim)
        facts = [(f"s{t}_{i}", f"r{t}_{i % 3}", f"o{t}_{i}") for i in range(n)]
        for s, r, o in facts:
            mem._facts.append(mem.fact_hv(s, r, o))
        mem._trace = abm.bundle(mem._facts)
        m_seen.append(len(mem.items))
        for s, r, o in facts:
            hits += mem.query(s, r)[0] == o
            total += 1
    meas, m = hits / total, int(np.mean(m_seen))
    ex, ga = exact.cleanup_accuracy(n, dim, m), fks_accuracy(n, dim, m)
    gc = fks_accuracy(n, dim, m, continuity=True)
    se = sqrt(ex * (1 - ex) / total)
    return {"dim": dim, "n": n, "codebook": m, "queries": total, "measured": meas, "se": se,
            "exact": ex, "gauss": ga, "gauss_cc": gc,
            "law_iv": abm.predicted_accuracy(n, dim, m),
            "err_exact_pts": 100 * (ex - meas), "err_gauss_pts": 100 * (ga - meas),
            "err_gauss_cc_pts": 100 * (gc - meas)}


def sweep():
    """Dove la gaussiana si stacca: |gauss − exact| in punti su una griglia D × carico."""
    rows = []
    for dim in (64, 128, 256, 512, 1024, 4096, 16384):
        for m in (16, 256, 4096, 65536):
            # carico tale che l'esatto sia ~0.5 e ~0.9 (bisezione sul carico)
            for target in (0.9, 0.5, 0.1):
                lo, hi = 1, 200000
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if exact.cleanup_accuracy(mid, dim, m) > target:
                        lo = mid
                    else:
                        hi = mid
                ex, ga = exact.cleanup_accuracy(lo, dim, m), fks_accuracy(lo, dim, m)
                rows.append({"dim": dim, "codebook": m, "target": target, "n": lo,
                             "exact": ex, "gauss": ga, "gauss_cc": fks_accuracy(lo, dim, m, continuity=True),
                             "gauss_minus_exact_pts": 100 * (ga - ex)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "fks_gaussian_vs_exact_results.json"))
    args = ap.parse_args()
    seed10 = json.loads((ROOT / "results" / "capacity_seed10_rerun.json").read_text())
    prereg = json.loads((ROOT / "results" / "exact_prereg_results.json").read_text())
    cap, cells, d256, sw = capacity_cells(seed10), prereg_cells(prereg), d256_cell(), sweep()
    mae = lambda xs: float(np.mean(np.abs(xs)))  # noqa: E731
    summary = {
        "capacity_mean_abs_err_exact": mae([c["err_exact"] for c in cap]),
        "capacity_mean_abs_err_gauss": mae([c["err_gauss"] for c in cap]),
        "capacity_mean_abs_relerr_exact_pct": mae([c["rel_err_exact_pct"] for c in cap]),
        "capacity_mean_abs_relerr_gauss_pct": mae([c["rel_err_gauss_pct"] for c in cap]),
    }
    for part in ("A", "B"):
        sub = [c for c in cells if c["part"] == part]
        summary[f"test2_{part}_mae_exact_pts"] = mae([c["err_exact_pts"] for c in sub])
        summary[f"test2_{part}_mae_gauss_pts"] = mae([c["err_gauss_pts"] for c in sub])
        summary[f"test2_{part}_mae_gauss_cc_pts"] = mae([c["err_gauss_cc_pts"] for c in sub])
        summary[f"test2_{part}_max_abs_gauss_minus_exact_pts"] = max(
            abs(c["gauss_minus_exact_pts"]) for c in sub)
    summary["sweep_max_abs_gauss_minus_exact_pts"] = max(abs(r["gauss_minus_exact_pts"]) for r in sw)
    res = {"analysis": "esplorativa, su dati già pubblicati; nessuna preregistrazione",
           "summary": summary, "capacity_seed10": cap, "test2": cells, "d256": d256, "sweep": sw}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    for c in cap:
        print(f"N* D={c['dim']}: mis {c['measured']:.1f}±{c['ci95']:.1f} esatto {c['exact']:.1f} "
              f"gauss {c['gauss']:.1f} gauss_cc {c['gauss_cc']:.1f}")
    for c in cells:
        print(f"{c['part']} D={c['dim']} n={c['n']} g={c['g']} M={c['codebook']}: mis {c['measured']:.3f} "
              f"esatto {c['exact']:.3f} gauss {c['gauss']:.3f} cc {c['gauss_cc']:.3f} se {c['se']:.3f}")
    print("D256", {k: round(v, 4) if isinstance(v, float) else v for k, v in d256.items()})
    for r in sw:
        print(f"sweep D={r['dim']} M={r['codebook']} t={r['target']} n={r['n']}: "
              f"esatto {r['exact']:.3f} gauss {r['gauss']:.3f} cc {r['gauss_cc']:.3f} "
              f"diff {r['gauss_minus_exact_pts']:+.2f}")


sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

if __name__ == "__main__":
    main()
