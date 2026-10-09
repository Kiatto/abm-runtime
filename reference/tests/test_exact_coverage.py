"""abm.exact: validazione degli input e copertura delle funzioni non testate
(audit ostile 2026-10-09, punti 6 e 8): capacity, two_hop_joint(_fast),
chain_accuracy_mc, win_ordered / cleanup_accuracy_ordered. Dove si può, il
confronto è con l'enumerazione esaustiva di tutti i bit; altrimenti con il
Monte Carlo del modello per bit, entro 4 errori standard."""
import itertools
import math

import numpy as np
import pytest

from abm import exact

T = [("a", "r", "b"), ("b", "q", "c")]


# --- validazione ---------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: exact.min_dimension(T, -1),
    lambda: exact.min_dimension(T, 0),
    lambda: exact.min_dimension(T, float("nan")),
    lambda: exact.min_dimension(T, 1.5),
    lambda: exact.capacity(0),
    lambda: exact.capacity(-3),
    lambda: exact.cleanup_accuracy(10, -5, 10),
    lambda: exact.cleanup_accuracy(10, 0, 10),
    lambda: exact.cleanup_accuracy(0, 64, 10),
    lambda: exact.two_hop_joint(2, 0, 10),
    lambda: exact.two_hop_joint(2, 64, 0),
    lambda: exact.two_hop_joint_fast(1, 64, 10),
    lambda: exact.two_hop_joint_fast(2, 0, 10),
    lambda: exact.chain_accuracy_mc(5, 64, 10, hops=0),
    lambda: exact.chain_accuracy_mc(5, 64, 10, hops=2, trials=0),
    lambda: exact.chain_accuracy_mc(5, 0, 10, hops=2),
    lambda: exact.chain_accuracy_mc(5, 64, 10, hops=2, wins=[np.ones(65)]),
    lambda: exact.win_ordered(0, 1, 1),
    lambda: exact.win_ordered(64, -1, 1),
    lambda: exact.cleanup_accuracy_ordered(0, 64, 1, 1),
    lambda: exact.p_agree_weighted(0, [1, 1]),
])
def test_degenerate_inputs_raise_value_error(call):
    """Prima: risultati silenziosi (64, 65536, 2.24, 0.0, 1.0) o traceback numpy."""
    with pytest.raises(ValueError) as err:
        call()
    assert err.value.args[0].isascii()


# --- confronto con l'enumerazione esaustiva ---------------------------------------

def _all_vectors(n_vec, dim):
    """Tutte le assegnazioni ±1 di n_vec vettori da dim bit: (2^(n*d), n_vec, dim)."""
    bits = np.array(list(itertools.product([-1, 1], repeat=n_vec * dim)), dtype=np.int64)
    return bits.reshape(-1, n_vec, dim)


@pytest.mark.parametrize("n_facts,dim", [(1, 4), (2, 4), (3, 4), (4, 3)])
def test_cleanup_accuracy_matches_enumeration_with_one_null(n_facts, dim):
    """codebook = 2 (il bersaglio e un nullo): pareggio diviso a metà, esatto."""
    v = _all_vectors(n_facts + 1, dim)
    facts, null = v[:, :n_facts], v[:, n_facts]
    s = facts.sum(axis=1)
    acc = 0.0
    # ogni bit a pareggio vale ±1 con probabilità 1/2: mediando su tutte le 2^dim
    # assegnazioni dei pareggi, ogni bit a pareggio è uniforme e indipendente
    for tie_bits in itertools.product([-1, 1], repeat=dim):
        t = np.where(s > 0, 1, np.where(s < 0, -1, np.array(tie_bits)))
        d_sig = (t != facts[:, 0]).sum(axis=1)
        d_null = (t != null).sum(axis=1)
        win = np.where(d_sig < d_null, 1.0, np.where(d_sig == d_null, 0.5, 0.0))
        acc += float(np.mean(win)) / 2 ** dim
    assert exact.cleanup_accuracy(n_facts, dim, 2) == pytest.approx(acc, abs=1e-12)


@pytest.mark.parametrize("n_before,n_after", [(0, 2), (1, 1), (2, 0)])
def test_cleanup_accuracy_ordered_matches_enumeration(n_before, n_after):
    """La regola della reference: vince il PRIMO codeword a distanza minima."""
    n_facts, dim = 3, 3                          # n dispari: nessun pareggio nella traccia
    v = _all_vectors(n_facts + n_before + n_after, dim)
    facts = v[:, :n_facts]
    t = np.where(facts.sum(axis=1) > 0, 1, -1)
    d_sig = (t != facts[:, 0]).sum(axis=1)
    d_before = [(t != v[:, n_facts + i]).sum(axis=1) for i in range(n_before)]
    d_after = [(t != v[:, n_facts + n_before + i]).sum(axis=1) for i in range(n_after)]
    win = np.ones(len(v), dtype=bool)
    for d in d_before:
        win &= d_sig < d
    for d in d_after:
        win &= d_sig <= d
    assert exact.cleanup_accuracy_ordered(n_facts, dim, n_before, n_after) == \
        pytest.approx(float(win.mean()), abs=1e-12)


def test_win_ordered_averages_to_split_ties_with_one_null():
    """Con un solo nullo, la media sulle due posizioni è la divisione a metà."""
    dim = 64
    avg = 0.5 * (exact.win_ordered(dim, 1, 0) + exact.win_ordered(dim, 0, 1))
    assert np.allclose(avg, exact._null_win(dim, 1), atol=1e-15)


# --- capacity -------------------------------------------------------------------

@pytest.mark.parametrize("dim", [256, 1024])
def test_capacity_is_the_fifty_percent_crossing(dim):
    n_star = exact.capacity(dim)
    lo, hi = math.floor(n_star) - 1, math.ceil(n_star) + 1
    assert exact.cleanup_accuracy(lo, dim, 2 * lo + 11) > 0.5
    assert exact.cleanup_accuracy(hi, dim, 2 * hi + 11) < 0.5


def test_capacity_grows_with_dim_and_shrinks_with_codebook():
    assert exact.capacity(2048) > exact.capacity(1024) > exact.capacity(512)
    assert exact.capacity(1024, lambda n: 10 * n) < exact.capacity(1024)


# --- due hop e catene -------------------------------------------------------------

@pytest.mark.parametrize("n_facts,dim,codebook", [(2, 32, 5), (5, 48, 12), (16, 64, 40)])
def test_two_hop_fast_matches_the_cubic_convolution(n_facts, dim, codebook):
    slow = exact.two_hop_joint(n_facts, dim, codebook)
    fast = exact.two_hop_joint_fast(n_facts, dim, codebook)
    assert fast == pytest.approx(slow, abs=1e-10)


def test_two_hop_single_hop_is_cleanup_accuracy():
    p, both, p2 = exact.two_hop_joint(9, 64, 20)
    assert p == pytest.approx(exact.cleanup_accuracy(9, 64, 20), abs=1e-12)
    assert p2 == pytest.approx(p * p) and 0 <= both <= p


@pytest.mark.parametrize("hops", [1, 2])
def test_chain_mc_matches_the_exact_values(hops):
    n_facts, dim, codebook, trials = 9, 64, 20, 20000
    mc = exact.chain_accuracy_mc(n_facts, dim, codebook, hops, trials=trials, seed=1)
    ref = (exact.cleanup_accuracy(n_facts, dim, codebook) if hops == 1
           else exact.two_hop_joint(n_facts, dim, codebook)[1])
    se = math.sqrt(ref * (1 - ref) / trials)
    assert abs(mc - ref) < 4 * se + 1e-3


def test_chain_mc_is_deterministic_and_accepts_ordered_wins():
    a = exact.chain_accuracy_mc(9, 64, 20, 3, trials=2000, seed=3)
    assert a == exact.chain_accuracy_mc(9, 64, 20, 3, trials=2000, seed=3)
    wins = [exact.win_ordered(64, 0, 19)] * 3      # bersaglio sempre inserito per primo
    assert exact.chain_accuracy_mc(9, 64, 20, 3, trials=2000, seed=3, wins=wins) >= a


def test_duplicates_are_weight_not_deduplicated():
    """Semantica documentata in predict_queries: una tripla ripetuta pesa doppio,
    come due Memory.store; i self-loop di una relazione sono un unico vettore."""
    base = [(f"s{i}", "r", f"o{i}") for i in range(12)]
    once = exact.predict_queries(base, 128, [("s0", "r")])[0]
    twice = exact.predict_queries(base + [base[0]], 128, [("s0", "r")])[0]
    assert twice > once
    assert exact.fact_key("a", "r", "a") == exact.fact_key("b", "r", "b")
    assert exact.fact_weights([("a", "r", "b"), ("b", "r", "a")]) == \
        {(frozenset(("a", "b")), "r"): 2}
