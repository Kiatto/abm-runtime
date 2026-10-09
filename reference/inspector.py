"""
inspector.py — ABM Inspector: la teoria esposta come API.

SUPERATO dal modello esatto (abm.exact.contract_for): qui la Legge IV
asintotica, ottimista di qualche punto a D piccolo e cieca a gemelli
simmetrici e alias (preregistrazione 10). Resta per compatibilità.

Non tocca la reference congelata (abm.py): la osserva. Ogni campo di
stats() è una formula del formalismo, non una statistica descrittiva —
è la differenza tra mostrare numeri e mostrare garanzie.

    from abm import Memory
    from abm.inspector import stats, contract, report

    stats(mem)                      → dict con i campi del Memory Contract
    contract(mem, grounding=0.93)   → la specifica firmabile
    report(mem)                     → testo leggibile
"""

from math import sqrt

import numpy as np
try:
    from .abm import (Memory, capacity, predicted_accuracy, z_gumbel, hamming,
                      confidence)
except ImportError:
    # file usato fuori dal pacchetto: gli script di examples/ mettono reference/
    # nel sys.path da sé. Il modulo non tocca più il sys.path di chi lo importa.
    from abm import (Memory, capacity, predicted_accuracy,  # type: ignore
                     z_gumbel, hamming, confidence)


def _check_probability(name: str, p) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)) \
            or not 0.0 <= p <= 1.0:              # nan fallisce il confronto
        raise ValueError(f"{name} must be a number in [0, 1], got {p!r}")
    return float(p)


def stats(mem: Memory, extractor_precision: float = 1.0,
          horizon: int = 100, hops: int = 2,
          triples=None, queries=None) -> dict:
    """Il Memory Contract calcolato, non osservato. Se l'applicazione
    fornisce triples+queries (livello A), aggiunge la diagnosi di
    aliasing strutturale.

    Solleva ValueError su una memoria vuota (come exact.contract_for([])) e su
    una extractor_precision fuori da [0, 1]."""
    _check_probability("extractor_precision (grounding)", extractor_precision)
    n = len(mem._facts)
    if n == 0:
        raise ValueError("no facts: a contract on an empty memory promises nothing; "
                         "store at least one fact first")
    m = max(len(mem.items), 2)
    cap = capacity(mem.dim, m)
    acc_now = predicted_accuracy(n, mem.dim, m) if n else 1.0
    acc_h = predicted_accuracy(n + horizon, mem.dim,
                               m + 2 * horizon) if n else 1.0
    # margine di confidenza: z del segnale meno la soglia del codebook
    z_signal = sqrt(2 * mem.dim / (np.pi * max(n, 1)))
    margin = z_signal - z_gumbel(m)
    # dimensione raccomandata: la minima potenza di 2 con pressione ≤ 0.5
    rec = mem.dim
    while n > 0.5 * capacity(rec, m) and rec < 2 ** 20:
        rec *= 2
    pg = extractor_precision ** hops
    out_alias = {}
    if triples is not None and queries is not None:
        al = aliasing(triples, queries)
        out_alias = dict(al)
        out_alias["projected_with_aliasing"] = round(
            al["aliasing_factor"] * pg * acc_now, 3)
    return {
        "facts": n,
        "dimension": mem.dim,
        "codebook": m,
        "estimated_capacity": int(cap),
        "utilization": round(n / cap, 2) if cap else 0.0,
        "pressure": round(n / cap, 2) if cap else 0.0,
        "expected_accuracy": round(acc_now, 3),
        "grounding_contract": round(pg, 3),
        "projected_accuracy": round(pg * acc_now, 3),   # Composition Law
        "capacity_remaining": max(int(cap) - n, 0),
        "expected_after_horizon": round(acc_h, 3),
        "horizon": horizon,
        "confidence_margin_sigma": round(margin, 1),
        "recommended_dimension": rec,
        "dominant_error": ("grounding" if pg < acc_now else "capacity"),
        "max_reasoning_depth_p50": (
            int(np.floor(np.log(0.5) / np.log(acc_now)))
            if 0 < acc_now < 1 else 99),
        **out_alias,
    }


def aliasing(triples, queries) -> dict:
    """Diagnosi di aliasing strutturale (Aliasing Factor Hypothesis,
    corroborata a g ∈ {2,3,4,8}): per la simmetria dell'encoding ogni
    fatto è un arco non orientato, quindi a query(node, r) i fatti che
    hanno `node` come OGGETTO con la stessa relazione r sono candidati
    a pari segnale. Fattore atteso per query: Π 1/g_i.

    triples: lista simbolica (s, r, o) — vive al livello A, l'Inspector
    la riceve dall'applicazione. queries: lista (start, [r1..rh]).
    Calcolabile PRIMA di qualsiasi query. Rimedio: Projection sul
    sottoinsieme dei non visitati (operatore di disambiguazione).
    """
    from collections import defaultdict
    subj = defaultdict(dict)                    # (s, r) -> o
    obj_count = defaultdict(int)                # (o, r) -> n archi entranti
    for s, r, o in triples:
        subj[(s, r)] = o
        obj_count[(o, r)] += 1
    factors, bad_rels = [], set()
    for start, relations in queries:
        node, f = start, 1.0
        for r in relations:
            # candidati a pari segnale = fatti in avanti + fatti inversi
            # (simmetria degli archi); g=1 se l'unico candidato è quello
            # giusto, anche quando è memorizzato invertito
            g = max(1, (1 if (node, r) in subj else 0)
                    + obj_count[(node, r)])
            if g > 1:
                bad_rels.add(r)
            f /= g
            node = subj.get((node, r), node)
        factors.append(f)
    factor = float(np.mean(factors)) if factors else 1.0
    return {"aliasing_factor": round(factor, 3),
            "aliasing_relations": sorted(bad_rels),
            "remedy": ("guided projection over unvisited subset"
                       if bad_rels else None)}


def contract(mem: Memory, grounding: float = 1.0) -> str:
    """Il contratto della Legge IV asintotica (superato: per dimensionare una
    memoria usare abm.exact.contract_for)."""
    s = stats(mem, extractor_precision=grounding)
    return (
        f"MEMORY CONTRACT (asymptotic Law IV; superseded by abm.exact.contract_for)\n"
        f"  Capacity        <= {s['estimated_capacity']} facts "
        f"(D={s['dimension']}, codebook={s['codebook']})\n"
        f"  Expected accuracy >= {s['expected_accuracy']:.0%} "
        f"at current load ({s['facts']} facts)\n"
        f"  Grounding       >= {grounding:.0%} "
        f"(projected end-to-end {s['projected_accuracy']:.0%})\n"
        f"  Confidence      logistic of the margin, not calibrated (0.5 = chance), "
        f"margin {s['confidence_margin_sigma']}σ\n"
        f"  Max depth (p50) =  {s['max_reasoning_depth_p50']} hops\n"
        f"  Pressure        =  {s['pressure']:.2f}"
        + ("  ← consider D=" + str(s['recommended_dimension'])
           if s['recommended_dimension'] > s['dimension'] else "")
    )


def report(mem: Memory, extractor_precision: float = 1.0) -> str:
    s = stats(mem, extractor_precision)
    lines = [f"{k:.<28}{v}" for k, v in s.items()]
    return "ABM INSPECTOR\n" + "\n".join("  " + line for line in lines)


if __name__ == "__main__":                  # python -m abm.inspector
    mem = Memory(2048)
    for i in range(287):
        mem.store(f"s{i}", f"r{i % 17}", f"o{i}")
    print(report(mem, extractor_precision=0.93))
    print()
    print(contract(mem, grounding=0.93))
