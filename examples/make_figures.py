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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bsm.memory.exact_contract import capacity as exact_capacity, cleanup_accuracy  # noqa: E402

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
            label="Law IV, k = 0.92 (fitted)")
    ax.plot(xs, [n_star(x, 1.0) for x in xs], ":", color="#999",
            label="Law IV, k = 1 (asymptotic)")
    xe = np.linspace(400, 4400, 12)
    ax.plot(xe, [exact_capacity(int(x)) for x in xe], "-", color="#27ae60", lw=1.2,
            label="exact, no parameter")
    ax.set_xlabel("D (dimension)"); ax.set_ylabel("N* (50% accuracy load)")
    ax.set_title("Capacity: exact vs asymptotic")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig1_capacity.png")


def fig_depth():
    d = json.loads((RES / "depth_scaling_s10_results.json").read_text())["dmin"]
    hs = sorted(int(k) for k in d)
    ds = [d[str(h)]["dmin"] for h in hs]
    sd = [d[str(h)]["sd"] for h in hs]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    # 10 seeds, D on a geometric grid with 10% steps (v1.4 rerun): bars are SD.
    ax.errorbar(hs, ds, yerr=sd, fmt="o-", color="#1a6faf",
                capsize=3, label="D_min, chain acc. ≥ 95%\n(mean ± SD, 10 seeds)")
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
    # 10 seed, 200 query per seed: la riesecuzione della v1.4. k = 0.92 viene da
    # ALTRI esperimenti ed è stato fissato prima della riesecuzione.
    d = json.loads((RES / "capacity_contract_s10_results.json").read_text())
    ns = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ns, [d[str(n)]["pred"] * 100 for n in ns], ":",
            color="#999", label="pure theory, k = 1")
    ax.plot(ns, [d[str(n)]["pred_k092"] * 100 for n in ns], "--",
            color="#c0392b", label="Law IV, k = 0.92")
    ax.plot(ns, [cleanup_accuracy(n, 8192, 2 * n + 13) * 100 for n in ns], "-",
            color="#27ae60", lw=1.2, label="exact, no parameter")
    ax.plot(ns, [d[str(n)]["meas"] * 100 for n in ns], "o",
            color="#1a6faf", label="measured (10 seeds)")
    ax.set_xlabel("facts stored in 1 KB (D = 8192)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Capacity contract, out of sample")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig5_contract.png")


def fig_robustness():
    """Preregistrazione 5: composizione senza calibrazione, misurato contro previsto."""
    cells = json.loads((RES / "composition_prereg_results.json").read_text())["cells"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    colors = {"missing": "#1a6faf", "wrong_relation": "#c0392b",
              "wrong_entity": "#8e44ad", "spurious": "#27ae60"}
    for kind, col in colors.items():
        eps = sorted({c["eps"] for c in cells if c.get("kind") == kind})
        ms = [100 * np.mean([c["measured"] for c in cells if c.get("kind") == kind and c["eps"] == e]) for e in eps]
        ps = [100 * np.mean([c["pred_exact"] for c in cells if c.get("kind") == kind and c["eps"] == e]) for e in eps]
        ax.plot(eps, ms, "o", color=col, label=kind, ms=4)
        ax.plot(eps, ps, "-", color=col, alpha=0.6)
    ax.set_xlabel("extraction error rate ε"); ax.set_ylabel("2-hop accuracy (%)")
    ax.set_title("Composition, preregistered (lines: prediction)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig6_robustness.png")


def _fig_robustness_calibrated_v14():
    d = json.loads((RES / "extraction_robustness_s10_results.json").read_text())
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
    ax.set_title("Resource composition, 10 seeds (dotted: prediction)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig6_robustness.png")


def fig_fb15k():
    """Law IV su FB15k-237: previsione preregistrata contro misura."""
    cells = json.loads((RES / "fb15k237_prereg_results.json").read_text())["cells"]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for dim, col in ((2048, "#1a6faf"), (8192, "#c0392b")):
        ns = sorted({c["n"] for c in cells if c["dim"] == dim})
        meas = [100 * np.mean([c["measured"] for c in cells
                               if c["dim"] == dim and c["n"] == n]) for n in ns]
        pred = [100 * np.mean([c["pred_k092_alias"] for c in cells
                               if c["dim"] == dim and c["n"] == n]) for n in ns]
        ax.plot(ns, pred, "--", color=col, alpha=0.8)
        ax.plot(ns, meas, "o", color=col, label=f"D = {dim}")
    ax.set_xscale("log")
    ax.set_xlabel("triples stored (FB15k-237)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Preregistered: FB15k-237 (dashed: prediction)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig7_fb15k237.png")


def fig_twins():
    """Preregistrazione 4: WN18RR denso, con e senza gemelli simmetrici."""
    cells = json.loads((RES / "twins_prereg_results.json").read_text())["cells"]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for dim, col in ((2048, "#1a6faf"), (8192, "#c0392b")):
        sel = [c for c in cells if c["part"] == "T2" and c["sampler"] == "dense" and c["dim"] == dim]
        ns = sorted({c["n"] for c in sel})
        f = lambda key: [100 * np.mean([c[key] for c in sel if c["n"] == n]) for n in ns]
        ax.plot(ns, f("measured"), "o", color=col, label=f"measured, D = {dim}")
        ax.plot(ns, f("pred_twins"), "-", color=col, alpha=0.8)
        ax.plot(ns, f("pred_no_twins"), ":", color=col, alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("triples stored (WN18RR, dense)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Symmetric twins (solid: counted, dotted: not)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig8_twins.png")


def fig_dependence():
    """Preregistrazione 3: la correlazione per bit fra due hop, prevista e misurata."""
    from bsm.memory.exact_contract import bit_correlation
    cells = json.loads((RES / "dependence_prereg_results.json").read_text())["cells"]
    e1 = [c for c in cells if c["part"] == "E1"]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ns = np.unique(np.geomspace(4, 400, 40).astype(int))
    ax.plot(ns, [bit_correlation(int(n)) for n in ns], "-", color="#27ae60",
            label="exact: −ρ²/(1−ρ²)")
    ax.errorbar([c["n"] for c in e1], [c["measured"] for c in e1],
                yerr=[3 * c["se"] for c in e1], fmt="o", color="#1a6faf",
                capsize=3, label="measured (±3 SE)")
    ax.axhline(0, ls="--", color="#999", label="Law V (independence)")
    ax.set_xscale("log"); ax.set_xlabel("facts in the trace (N)")
    ax.set_ylabel("per-bit correlation of two hops")
    ax.set_title("Hops on one trace are not independent")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig9_dependence.png")


def fig_asymmetric():
    """Preregistrazione 6: simmetrico − asimmetrico, previsto e misurato."""
    cells = json.loads((RES / "asymmetric_prereg_results.json").read_text())["cells"]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for (kg, dim), col, mk in ((("wn18rr", 2048), "#1a6faf", "o"), (("wn18rr", 8192), "#c0392b", "o"),
                               (("fb15k237", 2048), "#1a6faf", "^"), (("fb15k237", 8192), "#c0392b", "^")):
        sel = sorted((c for c in cells if c["kg"] == kg and c["dim"] == dim), key=lambda c: c["n"])
        ns = [c["n"] for c in sel]
        ax.plot(ns, [100 * (c["pred_sym"] - c["pred_asym"]) for c in sel], "-", color=col,
                alpha=0.5 if kg == "fb15k237" else 0.9)
        ax.plot(ns, [100 * (c["meas_sym"] - c["meas_asym"]) for c in sel], mk, color=col,
                label=f"{'WN18RR' if kg == 'wn18rr' else 'FB15k-237'}, D = {dim}", ms=5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("triples stored (dense)")
    ax.set_ylabel("symmetric − asymmetric (points)")
    ax.set_title("Encoding symmetry: cost, then benefit")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig10_asymmetric.png")


if __name__ == "__main__":
    for f in (fig_capacity, fig_depth, fig_proofwriter, fig_compiler,
              fig_contract, fig_robustness, fig_fb15k, fig_twins, fig_dependence, fig_asymmetric):
        f()
        print("✓", f.__name__)
    print("→", OUT)
