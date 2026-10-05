"""equal_bits_prereg.py — ABM contro uno store esatto e un filtro di Bloom, a pari numero di bit.

Preregistrazione: docs/preregistration/equal_bits.md. Committato insieme a quel
file e PRIMA di misurare; `--predict` stampa solo le previsioni del modello esatto.

Il §7 del paper dice che dimensione fissa e appartenenza con una sola distanza non
sono state confrontate con un filtro di Bloom o una tabella della stessa dimensione.

A — recupero (s, r) → o, a pari bit. Fatti (s_i, r_{i mod 13}, o_i), vocabolario di
2N entità e 13 relazioni (il protocollo del contratto di capacità). ABM: una traccia
di D bit. Store esatto ideale: ogni fatto costa b = 2·⌈log₂ 2N⌉ + ⌈log₂ 13⌉ bit, senza
alcun sovraccarico; con D bit ne tiene ⌊D/b⌋ e perde gli altri, quindi l'accuratezza
su una query a caso fra i fatti è min(1, ⌊D/b⌋/N). Il crossover è il carico N oltre
il quale ABM risponde meglio dello store esatto ideale.

B — appartenenza, a pari bit. ABM: Memory.member con z ≥ 3. Bloom: D bit, k funzioni
di hash, k = max(1, round(D/N · ln 2)). Errori: falsi negativi sui fatti memorizzati,
falsi positivi su fatti mai memorizzati dello stesso vocabolario.

Descrittivo, non un'ipotesi: il limite di Fano. Per rispondere a N query con
accuratezza a scegliendo fra 2N oggetti servono almeno
N · (log₂ 2N − h(1−a) − (1−a)·log₂(2N − 1)) bit; si riporta D diviso questo minimo.

Uso, dalla root:  python examples/equal_bits_prereg.py [--predict]
"""
import argparse
import hashlib
import json
import sys
from math import ceil, floor, log, log2
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

OUT = ROOT / "results" / "equal_bits_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [50, 100, 150, 200, 300, 400, 600],
        8192: [200, 400, 600, 800, 1200, 1600, 2400]}
RELS = 13
SEEDS = 3
NEG = 1000                      # fatti mai memorizzati per i falsi positivi
Z_MIN = 3.0
# -----------------------------------------------------------------------------


def facts(n, tag):
    return [(f"{tag}s{i}", f"{tag}r{i % RELS}", f"{tag}o{i}") for i in range(n)]


def exact_store_acc(n, dim):
    b = 2 * ceil(log2(2 * n)) + ceil(log2(RELS))
    return min(1.0, floor(dim / b) / n), b


def crossover(ns, abm_acc, store_acc):
    """Primo N (interpolato) in cui ABM − store passa da < 0 a ≥ 0; None se non accade."""
    d = [a - s for a, s in zip(abm_acc, store_acc)]
    for i in range(1, len(ns)):
        if d[i - 1] < 0 <= d[i]:
            return ns[i - 1] + (ns[i] - ns[i - 1]) * (-d[i - 1]) / (d[i] - d[i - 1])
    return None


def fano_bits(n, acc):
    m = 2 * n
    e = min(max(1 - acc, 1e-12), 1 - 1e-12)
    h = -e * log2(e) - (1 - e) * log2(1 - e)
    return max(n * (log2(m) - h - e * log2(m - 1)), 1e-9)


def predict():
    out = {}
    for dim, ns in GRID.items():
        pa = [float(np.mean(exact.predict_queries(facts(n, "p"), dim))) for n in ns]
        sa = [exact_store_acc(n, dim)[0] for n in ns]
        out[dim] = {"abm_pred": pa, "store": sa, "crossover_pred": crossover(ns, pa, sa)}
    return out


class Bloom:
    def __init__(self, bits, k):
        self.bits, self.k = bits, k
        self.a = np.zeros(bits, dtype=bool)

    def _idx(self, key):
        h = hashlib.sha256(key.encode()).digest()
        return [int.from_bytes(h[4 * j:4 * j + 4], "little") % self.bits for j in range(self.k)] \
            if self.k <= 8 else [int.from_bytes(hashlib.sha256(f"{j}|{key}".encode()).digest()[:8],
                                                "little") % self.bits for j in range(self.k)]

    def add(self, key):
        self.a[self._idx(key)] = True

    def __contains__(self, key):
        return bool(self.a[self._idx(key)].all())


def measure():
    pred = predict()
    res = {"grid": {}, "pred": pred}
    for dim, ns in GRID.items():
        rows = []
        for n in ns:
            acc, fn_a, fp_a, fn_b, fp_b = [], [], [], [], []
            for seed in range(SEEDS):
                tag = f"d{dim}n{n}s{seed}_"
                fs = facts(n, tag)
                mem = abm.Memory(dim)
                for f in fs:
                    mem.store(*f)
                acc.append(np.mean([mem.query(s, r)[0] == o for s, r, o in fs]))
                rng = np.random.RandomState(1000003 * seed + 7 * n + dim)
                neg = []
                stored = set(fs)
                while len(neg) < NEG:
                    i, j = rng.randint(n, size=2)
                    t = (f"{tag}s{i}", f"{tag}r{i % RELS}", f"{tag}o{j}")
                    if t not in stored:
                        neg.append(t)
                fn_a.append(np.mean([not mem.member(*f, z_min=Z_MIN) for f in fs]))
                fp_a.append(np.mean([mem.member(*f, z_min=Z_MIN) for f in neg]))
                k = max(1, round(dim / n * log(2)))
                bl = Bloom(dim, k)
                for f in fs:
                    bl.add("|".join(f))
                fn_b.append(np.mean(["|".join(f) not in bl for f in fs]))
                fp_b.append(np.mean(["|".join(f) in bl for f in neg]))
            a = float(np.mean(acc))
            store, b = exact_store_acc(n, dim)
            rows.append({"n": n, "abm_acc": a, "store_acc": store, "store_bits_per_fact": b,
                         "abm_member_fnr": float(np.mean(fn_a)), "abm_member_fpr": float(np.mean(fp_a)),
                         "bloom_k": max(1, round(dim / n * log(2))),
                         "bloom_fnr": float(np.mean(fn_b)), "bloom_fpr": float(np.mean(fp_b)),
                         "fano_ratio": dim / fano_bits(n, a)})
            print(dim, n, round(a, 3), round(store, 3), flush=True)
        res["grid"][dim] = rows
        res.setdefault("crossover_measured", {})[dim] = crossover(
            ns, [r["abm_acc"] for r in rows], [r["store_acc"] for r in rows])
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({"crossover_pred": {d: pred[d]["crossover_pred"] for d in GRID},
                      "crossover_measured": res["crossover_measured"]}, indent=1))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    if ap.parse_args().predict:
        p = predict()
        for dim in GRID:
            print(dim, "crossover previsto:", p[dim]["crossover_pred"])
            for n, a, s in zip(GRID[dim], p[dim]["abm_pred"], p[dim]["store"]):
                print(f"   N={n:5d}  ABM previsto {a:.3f}  store esatto {s:.3f}")
    else:
        measure()


if __name__ == "__main__":
    main()
