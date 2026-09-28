"""Test del contratto esatto (bsm/memory/exact_contract.py)."""
import sys
from math import comb, sqrt, pi
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
import abm  # noqa: E402

from bsm.memory.exact_contract import (binom_pmf, capacity, cleanup_accuracy,  # noqa: E402
                                       p_agree)


def test_binom_pmf_matches_direct_formula():
    for dim, p in ((10, 0.5), (17, 0.3), (40, 0.9)):
        exact = [comb(dim, d) * p**d * (1 - p) ** (dim - d) for d in range(dim + 1)]
        assert np.allclose(binom_pmf(dim, p), exact, rtol=1e-10, atol=1e-15)
        assert abs(binom_pmf(dim, p).sum() - 1) < 1e-12


def test_p_agree_small_cases_by_hand():
    assert p_agree(1) == 1.0
    # n = 2: l'altro fatto concorda (somma 2) o no (pareggio, 1/2): (1 + 1/2) / 2
    assert p_agree(2) == pytest.approx(0.75)
    # n = 3: +,+ / +,- / -,+ concordano, -,- no
    assert p_agree(3) == pytest.approx(0.75)


def test_p_agree_matches_majority_asymptotics():
    # 1/2 + 1/sqrt(2 pi n) è la correlazione di maggioranza (Clarkson et al., §6.1.1)
    for n in (101, 1001):
        assert p_agree(n) == pytest.approx(0.5 + 1 / sqrt(2 * pi * n), rel=2e-3)


def test_accuracy_is_monotone_and_bounded():
    accs = [cleanup_accuracy(n, 2048, 2 * n + 11) for n in (10, 50, 100, 200, 400)]
    assert all(0 <= a <= 1 for a in accs)
    assert accs == sorted(accs, reverse=True)
    assert cleanup_accuracy(100, 4096, 211) > cleanup_accuracy(100, 2048, 211)


def test_aliases_scale_by_symmetry():
    one = cleanup_accuracy(50, 1024, 111, correct=1, aliases=0)
    two = cleanup_accuracy(50, 1024, 111, correct=1, aliases=1)
    assert two < one
    both = cleanup_accuracy(50, 1024, 111, correct=2, aliases=0)
    assert both > one        # due bersagli giusti: più facile colpirne uno


def test_capacity_grows_with_dimension():
    caps = [capacity(d) for d in (512, 1024, 2048)]
    assert caps == sorted(caps)


def test_exact_model_matches_reference_simulation():
    """Il calcolo esatto deve descrivere la reference, non solo sé stesso.

    Monte Carlo con reference/abm.py a D piccolo, dove il test è veloce, a un
    carico con accuratezza ~0.72, dove le approssimazioni della Law IV sbagliano
    di ~7 punti (misurato: 0.714 contro 0.722 esatto, 0.792 Law IV con k = 1, su
    200 prove). Tolleranza: 3 errori standard, abbastanza stretta da escludere
    la Law IV: il test deve distinguere i due modelli, non solo passare.
    """
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
    measured = hits / total
    m = int(np.mean(m_seen))
    predicted = cleanup_accuracy(n, dim, m)
    law_iv = abm.predicted_accuracy(n, dim, m)
    se = sqrt(predicted * (1 - predicted) / total)
    assert abs(measured - predicted) < 3 * se, (measured, predicted, se)
    assert abs(measured - law_iv) > 3 * se, "il test non distingue più dalla Law IV"
