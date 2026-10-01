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


def test_unknown_query_raises_unless_asked_to_score_zero():
    """Audit 2026-10-01, punto 11: un refuso non deve valere 0.0 in silenzio."""
    with pytest.raises(KeyError):
        exact.predict_queries([("a", "r", "b")], 256, [("zzz", "r")])
    with pytest.raises(KeyError):
        exact.contract_for([("a", "r", "b")], 256, queries=[("a", "typo")])
    assert exact.predict_queries([("a", "r", "b")], 256, [("zzz", "r")],
                                 unknown="zero") == [0.0]


def test_alias_share_counts_self_loops_like_the_ceiling():
    c = exact.contract_for([("a", "r", "a"), ("b", "r", "c")], 4096)
    assert c["ceiling"] == pytest.approx(0.75)
    assert c["alias_share"] == pytest.approx(0.5)


def test_twin_share_ignores_plain_duplicates():
    assert exact.contract_for([("a", "r", "b"), ("a", "r", "b")], 256)["twin_share"] == 0
    assert exact.contract_for([("a", "r", "b"), ("b", "r", "a")], 256)["twin_share"] == 1


def test_codebook_dim_step_are_validated():
    t = [("a", "r", "b")]
    for bad in (1, 2, 3.0, 2.5, True):
        with pytest.raises(ValueError):
            exact.contract_for(t, 256, codebook=bad)
    assert exact.contract_for(t, 256, codebook=3)["codebook"] == 3
    for d in (0, -1, 64.0):
        with pytest.raises(ValueError):
            exact.contract_for(t, d)
    with pytest.raises(ValueError):
        exact.min_dimension(t, 0.9, step=0)
    with pytest.raises(ValueError):
        exact.min_dimension(t, 0.9, step=64, d_max=32)
    d = exact.min_dimension(_triples(), 0.9, step=64, d_max=1000)
    assert d is None or (d <= 1000 and exact.contract_for(_triples(), d)["expected_accuracy"] >= 0.9)


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


def test_alias_only_and_unanswerable_give_zero_not_errors():
    """Audit 2026-09-30, punto 21: niente ValueError in italiano, niente nan."""
    assert exact.cleanup_accuracy_mixed(256, 10, [], [0.6]) == 0.0
    assert exact.cleanup_accuracy(5, 256, 10, correct=0, aliases=1) == 0.0
    # (b, r) ha solo l'alias a: non rispondibile
    assert exact.predict_queries([("a", "r", "b")], 256, [("b", "r")]) == [0.0]
    for bad in (lambda: exact.contract_for([], 256),
                lambda: exact.min_dimension([], 0.9),
                lambda: exact.contract_for([("a", "r", "b")], 256, queries=[]),
                lambda: exact.cleanup_accuracy_mixed(256, 1, [0.6], [0.6]),
                lambda: exact.p_agree(0)):
        with pytest.raises(ValueError) as err:
            bad()
        assert err.value.args[0].isascii()


def test_codebook_and_queries_overrides():
    """Audit 2026-09-30, punto 22."""
    triples = _triples()
    m = len({x for t in triples for x in t})
    base = exact.predict_queries(triples, 512)
    assert exact.predict_queries(triples, 512, codebook=m) == base
    bigger = exact.predict_queries(triples, 512, codebook=10 * m)
    assert all(b <= a + 1e-15 for a, b in zip(base, bigger)) and sum(bigger) < sum(base)
    c = exact.contract_for(triples, 512, codebook=10 * m)
    assert c["codebook"] == 10 * m
    assert c["expected_accuracy"] == pytest.approx(float(np.mean(bigger)))
    assert exact.min_dimension(triples, 0.9, codebook=10 * m) >= exact.min_dimension(triples, 0.9)
    q = [("s", "has")]
    d = exact.min_dimension(triples, 0.9, queries=q)
    assert exact.contract_for(triples, d, queries=q)["expected_accuracy"] >= 0.9
    assert exact.contract_for(triples, d - 64, queries=q)["expected_accuracy"] < 0.9


def test_null_vector_is_cached_and_unchanged():
    """La cache non deve cambiare i numeri, e il vettore in cache non è scrivibile."""
    w = exact._null_win(256, 30)
    assert exact._null_win(256, 30) is w and not w.flags.writeable
    null = exact.binom_pmf(256, 0.5)
    sf = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)
    ref = np.clip(sf ** 30 + 0.5 * 30 * null * sf ** 29, 0.0, 1.0)
    assert np.array_equal(w, ref)
