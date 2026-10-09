"""Input degeneri: errori chiari invece di loop, traceback numpy o contratti al 100%.

Audit ostile 2026-10-09, punti 3, 4, 5 e 6 (parte abm.py).
"""
import math

import numpy as np
import pytest

import abm
from abm import Memory, capacity, confidence, predicted_accuracy, z_gumbel


@pytest.mark.parametrize("dim", [0, -8, 2.5, "4096", True, None, float("nan")])
def test_memory_rejects_non_positive_integer_dim(dim):
    with pytest.raises(ValueError, match="dim"):
        Memory(dim=dim)


@pytest.mark.parametrize("dim", [1, 7, np.int64(64), 4096])
def test_memory_accepts_any_positive_integer_dim(dim):
    m = Memory(dim=dim)
    m.store("a", "r", "b")
    assert m.query("a", "r")[0] in ("a", "r", "b")
    assert isinstance(m.dim, int)


def test_memory_rejects_item_memory_of_other_dimension():
    with pytest.raises(ValueError, match="dim"):
        Memory(dim=256, items=abm.ItemMemory(512))


@pytest.mark.parametrize("call", [
    lambda: predicted_accuracy(0, 64, 10),
    lambda: predicted_accuracy(5, 0, 10),
    lambda: predicted_accuracy(5, 64, 1),
    lambda: capacity(64, 1),
    lambda: capacity(0, 10),
    lambda: capacity(64, 10, k=0),
    lambda: z_gumbel(1),
    lambda: z_gumbel(0),
    lambda: confidence(1, 0),
    lambda: confidence(1, 64, temperature=0),
])
def test_law_iv_functions_raise_clear_value_errors(call):
    """Prima: ZeroDivisionError e 'math domain error' grezzi."""
    with pytest.raises(ValueError) as err:
        call()
    assert "math domain" not in str(err.value)


def test_law_iv_functions_unchanged_on_valid_input():
    assert predicted_accuracy(100, 2048, 50) == pytest.approx(
        0.5 * (1 + math.erf((math.sqrt(2 * 2048 / (math.pi * 100)) - z_gumbel(50))
                            / math.sqrt(2))), abs=0)
    assert capacity(2048, 2) > 0 and z_gumbel(2) == z_gumbel(2.0)


# --- punto 5: le letture non cambiano lo stato -------------------------------

def _small():
    m = Memory(512)
    for s, r, o in [("a", "r", "b"), ("b", "q", "c")]:
        m.store(s, r, o)
    return m


@pytest.mark.parametrize("read", [
    lambda m: m.query("zzz", "r"),
    lambda m: m.query("a", "never_stored"),
    lambda m: m.query_z("zzz", "r"),
    lambda m: m.member("a", "r", "zzz"),
    lambda m: m.chain("zzz", ["r", "q"]),
    lambda m: m.query_compiled("zzz", "r", "q"),
])
def test_reads_never_grow_the_codebook(read):
    """Prima query() faceva items.add del soggetto: il codebook M cresceva a ogni
    lettura, e il soggetto sconosciuto diventava un candidato (anche la risposta)."""
    m = _small()
    names, trace = list(m.items._names), m._trace.copy()
    read(m)
    assert m.items._names == names and np.array_equal(m._trace, trace)


def test_unknown_subject_is_answered_from_the_codebook_as_noise():
    m = _small()
    name, z = m.query_z("zzz", "r")
    assert name in ("a", "r", "b", "q", "c")
    assert abs(z) < 4                       # rumore: margine compatibile con il caso
    assert m.member("a", "r", "zzz") is False


def test_chain_needs_at_least_one_relation():
    with pytest.raises(ValueError, match="relation"):
        _small().chain("a", [])


def test_member_on_empty_memory_raises():
    with pytest.raises(ValueError):
        Memory(256).member("a", "r", "b")
