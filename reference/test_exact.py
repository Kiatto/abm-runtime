"""Test di abm.exact, il modello esatto pubblicato nel pacchetto."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "examples"))

import exact  # noqa: E402  reference/exact.py, cioè abm.exact una volta installato


def _triples():
    """Gemelli simmetrici, alias, risposte multiple, fatti semplici."""
    t = [("a", "sib", "b"), ("b", "sib", "a"),          # gemelli: un fatto di peso 2
         ("x", "par", "s"), ("s", "par", "y"),          # (s, par) ha l'alias x
         ("s", "has", "o1"), ("s", "has", "o2")]        # due oggetti veri
    t += [(f"e{i}", f"r{i % 5}", f"f{i}") for i in range(60)]
    return t


def test_matches_the_preregistered_predictor():
    """La versione pubblicata deve essere quella che ha superato i test preregistrati."""
    import twins_prereg as tp
    triples = _triples()
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in triples:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)
    keys = list(objects)
    m = len({x for t in triples for x in t})
    for dim in (512, 2048):
        ours = exact.predict_queries(triples, dim, keys)
        theirs = [p["twins"] for p in tp.predict(triples, keys, objects, into, dim, m)]
        assert ours == pytest.approx(theirs, abs=1e-12)


def test_ceiling_counts_aliases():
    triples = _triples()
    c = exact.contract_for(triples, 1024)
    # solo (s, par) ha un alias: tetto 1/2 su quella query, 1 sulle altre
    n_queries = len({(s, r) for s, r, _o in triples})
    assert c["ceiling"] == pytest.approx(1 - 0.5 / n_queries)
    assert c["twin_share"] == pytest.approx(2 / len(triples))
    assert 0 < c["expected_accuracy"] <= c["ceiling"]


def test_accuracy_approaches_the_ceiling_not_one():
    triples = _triples()
    high = exact.contract_for(triples, 1 << 15)
    assert high["expected_accuracy"] == pytest.approx(high["ceiling"], abs=1e-3)
    assert high["expected_accuracy"] < 1


def test_min_dimension_keeps_the_promise_and_is_minimal():
    triples = _triples()
    d = exact.min_dimension(triples, 0.9)
    assert d is not None and d % 64 == 0
    assert exact.contract_for(triples, d)["expected_accuracy"] >= 0.9
    assert exact.contract_for(triples, d - 64)["expected_accuracy"] < 0.9


def test_min_dimension_refuses_targets_above_the_ceiling():
    triples = _triples()
    assert exact.min_dimension(triples, 0.9999) is None


def test_module_is_complete():
    for name in ("cleanup_accuracy", "contract_for", "min_dimension", "win_ordered",
                 "two_hop_joint_fast", "chain_accuracy_mc", "p_agree_weighted"):
        assert hasattr(exact, name), name


def test_degenerate_probabilities_give_no_nan():
    """Audit 2026-09-30: binom_pmf(d, 0) dava nan (0·log 0); N = 1 dava nan."""
    assert list(exact.binom_pmf(4, 0.0)) == [1, 0, 0, 0, 0]
    assert list(exact.binom_pmf(4, 1.0)) == [0, 0, 0, 0, 1]
    assert exact.cleanup_accuracy(1, 64, 5) == 1.0
    assert exact.min_dimension([("a", "r", "b")], 0.9) == 64


def test_unanswerable_query_scores_zero():
    assert exact.predict_queries([("a", "r", "b")], 256, [("zzz", "r")]) == [0.0]


def test_self_loops_match_the_reference():
    """Audit 2026-09-30: (s, r, s) vale ρ(c_r) e rende ogni x un alias di (x, r).

    Prima il modello prevedeva ~0.81 dove la reference misura ~0.48. Tolleranza
    larga: nei pareggi la reference favorisce il soggetto, inserito prima, mentre il
    modello li divide a metà (scarto residuo ~2 punti).
    """
    import abm
    dim, hits, total = 256, 0, 0
    for t in range(150):
        trip = [(f"s{t}_{i}", "rel", f"o{t}_{i}") for i in range(10)] + [(f"z{t}", "rel", f"z{t}")]
        mem = abm.Memory(dim)
        for f in trip:
            mem._facts.append(mem.fact_hv(*f))
        mem._trace = abm.bundle(mem._facts)
        for s, r, o in trip[:10]:
            hits += mem.query(s, r)[0] == o
            total += 1
    pred = float(np.mean(exact.predict_queries(trip, dim, [(s, r) for s, r, _o in trip[:10]])))
    assert abs(hits / total - pred) < 0.05, (hits / total, pred)
    assert pred < 0.6                      # la versione vecchia prevedeva ~0.81
