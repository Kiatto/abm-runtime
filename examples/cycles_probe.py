"""cycles_probe.py — i fatti su un ciclo pari non sono indipendenti su GF(2).

Esplorativo, NON preregistrato (audit 2026-09-30, punto 3). Confronta, su un
rettangolo e su un biclique K_{4,5} con e senza fatti distrattori:
  - abm.exact, che tratta i fatti come variabili di Rademacher indipendenti;
  - un modello "lineare" che campiona i bit delle codeword e ricava i fatti come
    XOR, quindi tiene la dipendenza nel voto (ma non nella cleanup);
  - la reference misurata.
Uso, dalla root:  python examples/cycles_probe.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import numpy as np, abm, exact
from collections import defaultdict

def agreement_linear(triples, samples=200000, seed=0):
    """p di accordo di ogni fatto, con i fatti come XOR di variabili indipendenti per bit."""
    var = {}
    def v(name): return var.setdefault(name, len(var))
    rows = []
    for s, r, o in triples:
        vs = [v(("e", s)), v(("r", r)), v(("e", o))]
        if s == o: vs = [v(("r", r))]
        rows.append(vs)
    rng = np.random.RandomState(seed)
    z = rng.randint(0, 2, size=(samples, len(var))).astype(np.int8)
    F = np.stack([np.bitwise_xor.reduce(z[:, vs], axis=1) if len(vs) > 1 else z[:, vs[0]] for vs in rows], axis=1)
    pm = 1 - 2 * F.astype(np.int16)                       # ±1
    tot = pm.sum(axis=1)
    tie = rng.choice([-1, 1], size=samples)
    T = np.where(tot > 0, 1, np.where(tot < 0, -1, tie))
    return (pm == T[:, None]).mean(axis=0)               # p per fatto

def predict_linear(triples, dim, queries, samples=200000):
    p = agreement_linear(triples, samples)
    idx = defaultdict(list)
    for j, (s, r, o) in enumerate(triples): idx[exact.fact_key(s, r, o)].append(j)
    objects, into, m = exact._structure(triples)
    pk = {k: float(np.mean(p[js])) for k, js in idx.items()}
    out = []
    for s, r in queries:
        good, bad = objects[(s, r)], into[(s, r)] - objects[(s, r)]
        cp = [pk[exact.fact_key(s, r, o)] for o in good]; ap = [pk[exact.fact_key(x, r, s)] for x in bad]
        out.append(exact.cleanup_accuracy_mixed(dim, m, cp, ap) if cp else 0.0)
    return out

def measure(triples, dim, queries, objects, reps=200, tag=""):
    hits = tot = 0
    for t in range(reps):
        ren = {x: f"{tag}{t}_{x}" for tr in triples for x in (tr[0], tr[2])}
        trip = [(ren[s], f"{tag}{t}_{r}", ren[o]) for s, r, o in triples]
        mem = abm.Memory(dim)
        for f in trip: mem._facts.append(mem.fact_hv(*f))
        mem._trace = abm.bundle(mem._facts)
        for (s, r) in queries:
            ans = mem.query(ren[s], f"{tag}{t}_{r}")[0]
            hits += ans in {ren[o] for o in objects[(s, r)]}; tot += 1
    return hits / tot, tot


def main():
    rect = [("a", "r", "b"), ("a", "r", "c"), ("d", "r", "b"), ("d", "r", "c")]
    print("rettangolo: p_agree lineare", np.round(agreement_linear(rect, 400000), 3),
          "indipendente", round(exact.p_agree(4), 4))
    bic = [(f"a{i}", "r", f"b{j}") for i in range(4) for j in range(5)]
    for extra, dim in [(0, 64), (40, 256), (90, 256)]:
        trip = bic + [(f"x{i}", f"q{i % 3}", f"y{i}") for i in range(extra)]
        objects, _into, m = exact._structure(trip)
        qs = [(f"a{i}", "r") for i in range(4)]
        lin = np.mean(predict_linear(trip, dim, qs))
        ind = np.mean(exact.predict_queries(trip, dim, qs))
        acc, n = measure(trip, dim, qs, objects, reps=400, tag=f"e{extra}")
        print(f"N={len(trip)} M={m} D={dim}: lineare {lin:.3f} indipendente {ind:.3f} "
              f"misurato {acc:.3f} ± {np.sqrt(acc * (1 - acc) / n):.3f}")


if __name__ == "__main__":
    main()
