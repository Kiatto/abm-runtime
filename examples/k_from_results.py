"""k_from_results.py — ricava la costante k della Law IV dai risultati salvati.

Il paper cita k = 0.92 ± 0.03, ma nessuno script lo produceva: l'unico "k" nei
JSON (falsification_results.json, f1_lnM) è la costante della forma al primo
ordine, N*·ln M / D. Questo script ricalcola la costante della forma di Gumbel,

    k = N* · π · z_G(M)² / (2D)

dalle due serie già misurate, senza nuovi esperimenti:

- al variare del codebook M, a D = 2048 (falsification_results.json, f1_lnM);
- al variare della dimensione D, con M = 2N + 11 (seed10_results.json, law4).

Uso, dalla root del repo:  python examples/k_from_results.py
"""
import json
import statistics as st
from math import log, pi, sqrt
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def z_gumbel(m):
    z = sqrt(2 * log(m))
    return z - (log(log(m)) + log(4 * pi)) / (2 * z)


def k_of(n_star, m, dim):
    return n_star * pi * z_gumbel(m) ** 2 / (2 * dim)


def main():
    f1 = json.loads((RES / "falsification_results.json").read_text())["f1_lnM"]
    k_m = [k_of(v["nstar"], v["M"], 2048) for v in f1.values()]
    ms = [v["M"] for v in f1.values()]

    law4 = json.loads((RES / "seed10_results.json").read_text())["law4"]["per_d"]
    k_d = {int(d): k_of(n, 2 * n + 11, int(d)) for d, (n, _ci) in law4.items()}

    print(f"al variare di M (D = 2048, M {min(ms):.0f}–{max(ms):.0f}, "
          f"{max(ms) / min(ms):.0f}×):")
    print(f"  k = {st.mean(k_m):.3f} ± {st.stdev(k_m):.3f} (SD), "
          f"spread {(max(k_m) - min(k_m)) / st.mean(k_m):.1%}")
    print("al variare di D (M = 2N + 11):")
    for d in sorted(k_d):
        print(f"  D = {d:>5}: k = {k_d[d]:.3f}")
    kd = list(k_d.values())
    print(f"  k = {st.mean(kd):.3f} ± {st.stdev(kd):.3f} (SD)")
    allk = k_m + kd
    lo, hi = min(allk), max(allk)
    print(f"complessivo: {lo:.3f}–{hi:.3f}, cioè {(lo + hi) / 2:.2f} ± {(hi - lo) / 2:.2f}"
          f" (semi-ampiezza del range, non un intervallo di confidenza)")


if __name__ == "__main__":
    main()
