"""abm.families contro gli harness congelati (prereg. 21 e 22), che si importano senza
modificarli. Per ogni famiglia: (a) stesse risposte dell'harness sullo stesso input e seed
(sulle query senza pareggi; con pareggi i due tie-break usano generatori diversi, quindi
si confronta solo che la risposta stia fra i pari merito); (b) validazione; (c)
query_many == query ripetuta; (d) bits()."""
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import abm
from abm import families

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def v21():
    sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("vsa_families_prereg")


@pytest.fixture(scope="module")
def v22(v21):
    return importlib.import_module("generalization_prereg")


@pytest.fixture(scope="module")
def graph(v21):
    triples = v21.synthetic(n_ent=40, n_rel=4, n_facts=120, seed=7)
    objects = defaultdict(set)
    for s, r, o in triples:
        objects[(s, r)].add(o)
    queries = sorted(objects)
    cell = SimpleNamespace(triples=triples, objects=objects, queries=queries,
                           symbols=sorted({x for t in triples for x in t}))
    cell.m = len(cell.symbols)
    return cell


DIMS = {"MAP-B": 256, "MAP-I": 256, "FHRR": 128, "HRR": 256, "BSDC-OR": 4096, "BSDC-S": 1024}


def _mem(family, cell, seed):
    mem = families.make(family, DIMS[family], seed=seed)
    mem.store(cell.triples)
    return mem


def _tied_ok(mem, cell, got, harness_ok):
    """Accordo per query: identico dove il massimo è unico; dove c'è pareggio, la nostra
    risposta è fra i pari merito (l'harness pesca col suo flusso)."""
    s, r = mem._ids(cell.queries)
    sc = mem._scores(s, r)
    agree, ties = 0, 0
    for k, (q, (name, _)) in enumerate(zip(cell.queries, got)):
        top = np.flatnonzero(sc[k] == sc[k].max())
        assert mem.idx[name] in top
        if len(top) == 1:
            assert (name in cell.objects[q]) == harness_ok[k], (q, name)
            agree += 1
        else:
            ties += 1
    return agree, ties


@pytest.mark.parametrize("family", ["MAP-B", "MAP-I", "FHRR", "BSDC-OR", "BSDC-S"])
@pytest.mark.parametrize("seed", [0, 3])
def test_matches_vsa_families_harness(v21, graph, family, seed):
    if family == "MAP-B" and seed != 0:
        pytest.skip("l'harness 21 usa atomi non salati: confronto solo a seed 0")
    mem = _mem(family, graph, seed)
    got = mem.query_many(graph.queries)
    dim = mem.dim
    if family.startswith("BSDC"):
        assert mem.length == v21.bsdc_length(len(graph.triples))
    ok = v21.measure(graph, family, dim, np.random.default_rng(seed))
    agree, ties = _tied_ok(mem, graph, got, ok)
    assert agree + ties == len(graph.queries)
    assert agree >= len(graph.queries) // 4
    acc = np.mean([n in graph.objects[q] for q, (n, _) in zip(graph.queries, got)])
    assert abs(acc - np.mean(ok)) <= ties / len(graph.queries) + 1e-12


@pytest.mark.parametrize("seed", [0, 5])
def test_hrr_matches_generalization_engine(v22, graph, seed):
    mem = _mem("HRR", graph, seed)
    got = mem.query_many(graph.queries)
    atoms = mem.atoms

    class E(v22.Engine):                       # stesso codice dell'harness, nostri atomi
        def _gen(self, ix):
            return atoms[np.asarray(ix)]
    eng = E(graph, "HRR", mem.dim, np.random.default_rng(seed))
    assert np.allclose(eng.trace, mem.trace)
    ans = eng.answer([eng.idx[s] for s, _ in graph.queries], [eng.idx[r] for _, r in graph.queries])
    ok = [eng.sym[g] in graph.objects[q] for g, q in zip(ans, graph.queries)]
    agree, _ = _tied_ok(mem, graph, got, ok)
    assert agree == len(graph.queries)          # HRR continuo: nessun pareggio


def test_mapb_salted_matches_generalization_engine(v22, graph):
    mem = _mem("MAP-B", graph, 9)
    mem.bits()                                  # costruisce la traccia (pigra)
    cell = SimpleNamespace(**vars(graph), salt="9")
    eng = v22.Engine(cell, "MAP-B", mem.dim, np.random.default_rng(0))
    assert np.array_equal(eng.trace.astype(np.int8), mem.trace)


def test_mapb_seed0_is_abm_memory(graph):
    triples = list(graph.triples) + [graph.triples[0], ("e1", "r1", "e1")]   # peso e self-loop
    ref = abm.Memory(256)
    for t in triples:
        ref.store(*t)
    mem = families.make("MAP-B", 256)
    mem.store(triples)
    qs = graph.queries + [("e1", "r1")]
    assert mem.query_many(qs) == [ref.query(s, r) for s, r in qs]
    assert np.array_equal(mem.trace, ref._trace)
    assert mem.codebook_size() == len(ref.items)


def test_mapb_salt_changes_atoms(graph):
    a, b = _mem("MAP-B", graph, 0), _mem("MAP-B", graph, 1)
    a.bits(), b.bits()
    assert not np.array_equal(a.trace, b.trace)


@pytest.mark.parametrize("family", families.FAMILIES)
def test_query_many_equals_query(graph, family):
    mem = _mem(family, graph, 2)
    many = mem.query_many(graph.queries)
    one = [mem.query(s, r) for s, r in graph.queries]
    # stessi nomi; punteggi uguali a meno dell'arrotondamento (BLAS somma in ordine
    # diverso a seconda del numero di righe; esatti per MAP-B, MAP-I, BSDC)
    assert [n for n, _ in many] == [n for n, _ in one]
    exact = family in ("MAP-B", "MAP-I", "BSDC-OR", "BSDC-S")
    assert np.allclose([x for _, x in many], [x for _, x in one], rtol=0 if exact else 1e-5, atol=0)
    assert mem.query_many(graph.queries) == many            # pura: nessuna mutazione
    assert mem.codebook_size() == graph.m
    assert mem.query_many([]) == []


@pytest.mark.parametrize("family", families.FAMILIES)
def test_validation(family):
    for bad in (0, -3, 2.5, True, "64"):
        with pytest.raises(ValueError):
            families.make(family, bad)
    with pytest.raises(ValueError):
        families.make(family, 64, seed=-1)
    mem = families.make(family, 64)
    with pytest.raises(ValueError):
        mem.query("a", "r")                                  # memoria vuota
    for bad in [("a", "r"), ("a", "r", 3), "abc"]:
        with pytest.raises(ValueError):
            mem.store([bad])
    mem.store([("a", "r", "b")])
    with pytest.raises(KeyError):
        mem.query("zzz", "r")
    with pytest.raises(ValueError):
        mem.query_many([("a",)])
    assert mem.codebook_size() == 3


def test_unknown_family():
    with pytest.raises(ValueError):
        families.make("BSC", 64)


@pytest.mark.parametrize("family", families.FAMILIES)
def test_bits(graph, family):
    mem = _mem(family, graph, 0)
    n = len(graph.triples)
    per = {"MAP-B": 1, "BSDC-OR": 1, "FHRR": 32, "HRR": 32,
           "MAP-I": math.ceil(math.log2(n + 1)), "BSDC-S": math.ceil(math.log2(n + 1))}
    assert mem.bits() == mem.dim * per[family]


def test_bsdc_rounds_dim_up():
    mem = families.make("BSDC-OR", 1000)
    mem.store([(f"e{i}", "r", f"e{i + 1}") for i in range(100)])
    mem.bits()
    assert mem.length == families.bsdc_length(100) == 128
    assert mem.dim == 1024 and mem.dim_requested == 1000 and mem.blocks == 8


def test_store_accumulates(graph):
    a = families.make("MAP-I", 256, seed=1)
    a.store(graph.triples[:60])
    a.store(graph.triples[60:])
    b = _mem("MAP-I", graph, 1)
    assert a.query_many(graph.queries) == b.query_many(graph.queries)
