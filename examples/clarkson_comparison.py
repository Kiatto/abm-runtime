"""clarkson_comparison.py — Il limite di Clarkson et al. (2023) contro le costanti esatte.

Clarkson, Ubaru e Yang, "Capacity Analysis of Vector Symbolic Architectures"
(arXiv:2301.10352), Teorema 16: in MAP-B, dato il bundle x = sign(S v) di n
vettori atomici su un universo di d elementi, il test

    j è membro  <=>  x·S_j >= tau_C = sqrt(2 m ln(2d/delta))

è corretto per TUTTI i d elementi con probabilità >= 1 - delta, per un
m = O(n log(d/delta)). Dalla dimostrazione (§6.1.1) la costante è esplicita:
m/sqrt(7n) > 2·sqrt(2 m ln(2d/delta)), cioè

    m_C = 56 · n · ln(2d/delta).

Il confronto è equo perché usa lo STESSO compito e la STESSA regola di
decisione. Cambiano solo le costanti:

- m_C: la dimensione sufficiente secondo il teorema;
- m_exact: la stessa regola con le costanti esatte. Il segnale medio di un
  membro è m·sqrt(2/(pi n)) (assioma A2 del paper), il rumore ha deviazione
  sqrt(m); con soglia a metà e coda gaussiana z = Phi^-1(1 - delta/(2d)),
  serve m >= 2·pi·n·z²;
- m_measured: la dimensione minima misurata in simulazione per cui la
  frequenza di fallimento (almeno un errore sui d test) è <= delta, con la
  soglia a metà del segnale atteso;
- m_measured_tauC: lo stesso, ma con ESATTAMENTE la soglia del teorema,
  tau_C. Separa quanto dello scarto viene dalla costante e quanto dalla
  scelta della soglia.

Uso, dalla root del repo:  python examples/clarkson_comparison.py
"""
import json
from math import log, pi, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "clarkson_comparison_results.json"
D_UNIVERSE = 500
DELTA = 0.1
TRIALS = 200
NS = (10, 25, 50)
GRID = 1.1


def m_clarkson(n, d, delta):
    return 56 * n * log(2 * d / delta)


def m_exact(n, d, delta):
    z = NormalDist().inv_cdf(1 - delta / (2 * d))
    return 2 * pi * n * z * z


def failure_rate(m, n, d, trials, rng, tau):
    """Frazione di prove con almeno un errore di appartenenza sui d elementi."""
    fails = 0
    for _ in range(trials):
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(m, d))
        members = rng.choice(d, size=n, replace=False)
        acc = s[:, members].astype(np.int32).sum(axis=1)
        ties = acc == 0                                    # sign(0): ±1 a caso,
        acc[ties] = rng.choice([-1, 1], size=int(ties.sum()))  # come nel teorema
        x = np.sign(acc).astype(np.int32)
        scores = x @ s.astype(np.int32)                    # x·S_j per ogni j
        is_member = np.zeros(d, dtype=bool)
        is_member[members] = True
        fails += bool(np.any((scores >= tau(m, n)) != is_member))
    return fails / trials


def min_dim(n, d, delta, rng, tau, start):
    m = start
    while True:
        if failure_rate(int(m), n, d, TRIALS, rng, tau) <= delta:
            return int(m)
        m *= GRID


def main():
    rng = np.random.RandomState(0)
    half_signal = lambda m, n: 0.5 * m * sqrt(2 / (pi * n))
    tau_c = lambda m, n: sqrt(2 * m * log(2 * D_UNIVERSE / DELTA))
    rows = {}
    print(f"d = {D_UNIVERSE}, delta = {DELTA}, {TRIALS} prove per punto, griglia x{GRID}")
    print(f"{'n':>4} {'m_C (teorema)':>14} {'m esatto':>9} {'m misurato':>11} "
          f"{'m_C / misurato':>15} {'m misurato, tau_C':>18}")
    for n in NS:
        mc = m_clarkson(n, D_UNIVERSE, DELTA)
        me = m_exact(n, D_UNIVERSE, DELTA)
        mm = min_dim(n, D_UNIVERSE, DELTA, rng, half_signal, start=0.3 * me)
        mt = min_dim(n, D_UNIVERSE, DELTA, rng, tau_c, start=0.3 * me)
        rows[n] = {"m_clarkson": mc, "m_exact": me, "m_measured": mm,
                   "m_measured_tauC": mt,
                   "clarkson_over_measured": mc / mm, "exact_over_measured": me / mm,
                   "clarkson_over_measured_tauC": mc / mt}
        print(f"{n:>4} {mc:>14.0f} {me:>9.0f} {mm:>11} {mc / mm:>14.1f}x {mt:>18}")
    OUT.write_text(json.dumps({"d": D_UNIVERSE, "delta": DELTA, "trials": TRIALS,
                               "grid": GRID, "rows": rows}, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
