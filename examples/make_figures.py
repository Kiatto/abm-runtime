"""make_figures.py — Figure del paper dai JSON in results/.

Etichette in inglese perché il paper è in inglese.
"""
import json, sys
from math import log, pi, sqrt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"   # i JSON sono stati spostati qui dalla root
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def zg(m):
    z = sqrt(2 * log(m))
    return z - (log(log(m)) + log(4 * pi)) / (2 * z)


def fig_capacity():
    d = json.loads((RES / "seed10_results.json").read_text())["law4"]["per_d"]
    dims = sorted(int(k) for k in d)
    ns = [d[str(k)][0] for k in dims]
    cis = [d[str(k)][1] for k in dims]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(dims, ns, yerr=cis, fmt="o", color="#1a6faf", capsize=3,
                label="measured (10 seeds, 95% CI)")
    # The codebook grows with the load (M = 2N + 11), so N* is the fixed point
    # of N = k·2D/(π·z_G(2N+11)²). An earlier version computed M from the
    # MEASURED N, which let the data into the prediction; the fixed point does
    # not, and differs from it by less than one fact at every D.
    def n_star(dim, k):
        n = 100.0
        for _ in range(200):
            n = k * 2 * dim / (pi * zg(2 * n + 11) ** 2)
        return n
    xs = np.linspace(400, 4400, 100)
    ax.plot(xs, [n_star(x, 0.92) for x in xs], "--", color="#c0392b",
            label="Law IV, k = 0.92 (measured once)")
    ax.plot(xs, [n_star(x, 1.0) for x in xs], ":", color="#999",
            label="Law IV, k = 1 (pure theory)")
    ax.set_xlabel("D (dimension)"); ax.set_ylabel("N* (50% accuracy load)")
    ax.set_title("Law IV: capacity")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig1_capacity.png")


def fig_depth():
    d = json.loads((RES / "depth_scaling_results.json").read_text())["dmin"]
    hs = sorted(int(k) for k in d)
    ds = [d[str(h)]["dmin"] for h in hs]
    sd = [d[str(h)]["sd"] for h in hs]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    # 3 seeds, D searched on a geometric grid with 25% steps: bars are SD.
    ax.errorbar(hs, ds, yerr=sd, fmt="o-", color="#1a6faf",
                capsize=3, label="D_min, chain acc. ≥ 95%\n(mean ± SD, 3 seeds)")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 10000)
    ax.set_xlabel("h (hops)"); ax.set_ylabel("minimum D")
    ax.set_title("P1: required dimension vs depth (N = 120)")
    ax.legend(loc="lower right", fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig2_depth.png")


def fig_proofwriter():
    d = json.loads((RES / "seed10_results.json").read_text())["proofwriter"]
    depths = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(depths, [d[str(k)]["acc"] * 100 for k in depths],
                yerr=[d[str(k)]["ci95"] * 100 for k in depths],
                fmt="o-", color="#1a6faf", capsize=3, label="ABM (10 seeds, 95% CI)")
    ax.axhline(42, ls="--", color="#c0392b", label="majority baseline")
    ax.set_ylim(0, 105); ax.set_xlabel("inference depth")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("ProofWriter, parsable subset")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig3_proofwriter.png")


def fig_compiler():
    d = json.loads((RES / "compiler_bench_results.json").read_text())
    ns = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ns, [d[str(n)]["naive"] * 100 for n in ns], "o-",
            color="#c0392b", label="interpreted (2 cleanups)")
    ax.plot(ns, [d[str(n)]["p2"] * 100 for n in ns], ":",
            color="#c0392b", alpha=0.6, label="p² (Law V)")
    ax.plot(ns, [d[str(n)]["compiled"] * 100 for n in ns], "s-",
            color="#1a6faf", label="compiled (1 cleanup on T₂)")
    ax.set_xlabel("2-hop chains stored"); ax.set_ylabel("accuracy (%)")
    ax.set_title("P2: offline compilation")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig4_compiler.png")


def fig_contract():
    d = json.loads((RES / "capacity_contract_results.json").read_text())
    ns = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ns, [d[str(n)]["pred"] * 100 for n in ns], "--",
            color="#c0392b", label="predicted (pure theory, k = 1)")
    ax.plot(ns, [d[str(n)]["meas"] * 100 for n in ns], "o-",
            color="#1a6faf", label="measured")
    ax.set_xlabel("facts stored in 1 KB (D = 8192)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Capacity contract: optimistic at every N ≥ 300")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig5_contract.png")


def fig_robustness():
    d = json.loads((RES / "extraction_robustness_results.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    colors = {"missing": "#1a6faf", "wrong_relation": "#c0392b",
              "wrong_entity": "#8e44ad", "spurious": "#27ae60"}
    for kind, col in colors.items():
        eps = sorted(float(e) for e in d["measured"][kind])
        ms = [d["measured"][kind][str(e)]["acc"] * 100 for e in eps]
        ps = [d["predicted"][kind][str(e)] * 100 for e in eps]
        ax.plot(eps, ms, "o-", color=col, label=kind, ms=3)
        ax.plot(eps, ps, ":", color=col, alpha=0.6)
    ax.set_xlabel("extraction error rate ε"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Resource composition (dotted: prediction)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig6_robustness.png")


if __name__ == "__main__":
    for f in (fig_capacity, fig_depth, fig_proofwriter, fig_compiler,
              fig_contract, fig_robustness):
        f()
        print("✓", f.__name__)
    print("→", OUT)
