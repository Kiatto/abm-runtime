"""abm.families contro gli harness congelati (prereg. 21 e 22), che si importano senza
modificarli. Per ogni famiglia: (a) stesse risposte dell'harness sullo stesso input e seed
(sulle query senza pareggi; con pareggi i due tie-break usano generatori diversi, quindi
si confronta solo che la risposta stia fra i pari merito); (b) validazione; (c)
query_many == query ripetuta; (d) bits()."""
import importlib
import tracemalloc
import warnings
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
    s, r, _ = mem._ids(cell.queries)
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
    with pytest.warns(UserWarning, match="dim effettivo 1024"):
        mem.bits()
    assert mem.length == families.bsdc_length(100) == 128
    assert mem.dim == 1024 and mem.dim_requested == 1000 and mem.blocks == 8


def test_store_accumulates(graph):
    a = families.make("MAP-I", 256, seed=1)
    a.store(graph.triples[:60])
    a.store(graph.triples[60:])
    b = _mem("MAP-I", graph, 1)
    assert a.query_many(graph.queries) == b.query_many(graph.queries)


# -- review ostile 2026-10-13: memoria, simboli ignoti, pareggi, BSDC, atomi, bit ----------

def _rand_graph(n, m, seed=1, n_rel=8):
    rng = np.random.default_rng(seed)
    ents = [f"e{i}" for i in range(m - n_rel)]
    rels = [f"r{i}" for i in range(n_rel)]
    tr = [(ents[a], rels[b], ents[c]) for a, b, c in zip(
        rng.integers(len(ents), size=n), rng.integers(n_rel, size=n), rng.integers(len(ents), size=n))]
    return tr, sorted({(s, r) for s, r, _ in tr})


@pytest.mark.parametrize("family,dim,n,m", [
    ("HRR", 6000, 700, 1300), ("FHRR", 5000, 900, 1100), ("MAP-I", 7000, 500, 900),
    ("MAP-B", 12000, 600, 1500), ("BSDC-S", 30000, 1500, 700)])
def test_estimate_bytes_within_1_5(family, dim, n, m):
    """estimate_bytes contro il picco misurato (tracemalloc: atomi, traccia, temporanei)."""
    tr, qs = _rand_graph(n, m, seed=4)
    mem = families.make(family, dim)
    mem.store(tr)
    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem.query_many(qs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    est = families.estimate_bytes(family, dim, n, mem.codebook_size(), n_queries=len(qs))
    assert peak / 1.5 <= est <= peak * 1.5, (est, peak)


def test_estimate_bytes_validation():
    with pytest.raises(ValueError):
        families.estimate_bytes("BSC", 64, 1, 1)
    with pytest.raises(ValueError):
        families.estimate_bytes("HRR", 0, 1, 1)
    # la causa del finding: a N=1600, D=35712, M=2374 HRR teneva ≈1.36 GB di spettri
    assert families.estimate_bytes("HRR", 35712, 1600, 2374) < 0.5e9


@pytest.mark.parametrize("family", ["MAP-I", "FHRR", "HRR", "BSDC-S"])
def test_blockwise_atoms_equal_one_shot(monkeypatch, family):
    """Generazione a blocchi == estrazione in un colpo dallo stesso flusso (harness)."""
    monkeypatch.setattr(families, "_CHUNK_BYTES", 4096)
    tr, _ = _rand_graph(60, 40)
    mem = families.make(family, 64, seed=3)
    mem.store(tr)
    mem.bits()
    m, rng = mem.codebook_size(), np.random.default_rng(3)
    if family == "MAP-I":
        ref = (2 * rng.integers(2, size=(m, 64)) - 1).astype(np.float64)
    elif family == "FHRR":
        ref = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(m, 64))).astype(np.complex64)
    elif family == "HRR":
        ref = (rng.standard_normal((m, 64)) / math.sqrt(64)).astype(np.float32)
    else:
        ref = rng.integers(mem.length, size=(m, mem.blocks))
    assert np.array_equal(mem.atoms, ref)


@pytest.mark.parametrize("family", families.FAMILIES)
def test_chunking_does_not_change_answers(monkeypatch, graph, family):
    ref = _mem(family, graph, 1).query_many(graph.queries)
    monkeypatch.setattr(families, "_CHUNK_BYTES", 1)          # una riga per blocco
    got = _mem(family, graph, 1).query_many(graph.queries)
    assert [n for n, _ in got] == [n for n, _ in ref]
    assert np.allclose([x for _, x in got], [x for _, x in ref], rtol=1e-5, atol=0)


@pytest.mark.parametrize("family", families.FAMILIES)
def test_unknown_symbols(graph, family):
    mem = _mem(family, graph, 0)
    q = graph.queries[:3]
    with pytest.raises(KeyError, match="mai memorizzato 'zzz'.*query #1"):
        mem.query_many([q[0], ("zzz", q[1][1]), q[2]])
    got = mem.query_many([q[0], ("zzz", q[1][1]), (q[2][0], "yyy"), q[2]], unknown="skip")
    assert got[1] is None and got[2] is None
    assert [got[0], got[3]] == mem.query_many([q[0], q[2]])
    assert mem.query_many([("zzz", "r")], unknown="skip") == [None]
    with pytest.raises(ValueError):
        mem.query_many(q, unknown="ignore")


def test_ties_options():
    for fam in families.FAMILIES:
        if fam != "MAP-B":
            with pytest.raises(ValueError):
                families.make(fam, 64, ties="first")
            families.make(fam, 64, ties="random")
    with pytest.raises(ValueError):
        families.make("MAP-B", 64, ties="last")
    # MAP-B con un solo fatto a D piccolo: molti pareggi
    tr, qs = _rand_graph(40, 30, seed=2)
    first = families.make("MAP-B", 16)
    first.store(tr)
    rnd = families.make("MAP-B", 16, ties="random")
    rnd.store(tr)
    s, r, _ = first._ids(qs)
    sc = first._scores(s, r)
    got_f, got_r = first.query_many(qs), rnd.query_many(qs)
    assert [first.idx[n] for n, _ in got_f] == list(np.argmax(sc, axis=1))   # primo = argmax
    for k, (n, c) in enumerate(got_r):
        assert first.idx[n] in np.flatnonzero(sc[k] == sc[k].max())
        assert c == got_f[k][1]                       # stessa distanza, stessa confidenza
    assert got_r != got_f                             # i pareggi si risolvono diversamente
    assert rnd.query_many(qs) == got_r                # ma in modo deterministico


def test_bsdc_strict_and_no_warning_on_exact():
    tr = [(f"e{i}", "r", f"e{i + 1}") for i in range(100)]
    mem = families.make("BSDC-S", 1000, strict=True)
    mem.store(tr)
    with pytest.raises(ValueError, match="strict"):
        mem.bits()
    ok = families.make("BSDC-OR", 1024, strict=True)
    ok.store(tr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok.bits()
    assert ok.dim == 1024


@pytest.mark.parametrize("family", ["MAP-I", "FHRR", "HRR", "BSDC-OR"])
def test_per_symbol_atoms_stable(family):
    tr, qs = _rand_graph(80, 50, seed=5)
    def atoms_of(triples, mode):
        mem = families.make(family, 4096 if family.startswith("BSDC") else 2048, seed=2, atoms=mode)
        mem.store(triples)
        mem.bits()
        return mem, {x: mem.atoms[i] for x, i in mem.idx.items()}
    extra = tr + [("aaa_new", "r0", "e1")]           # nuovo simbolo, primo in ordine lessicografico
    if family.startswith("BSDC"):
        extra = tr[:-1] + [("aaa_new", "r0", "e1")]  # stesso N: stesso L
    a, A = atoms_of(tr, "per_symbol")
    b, B = atoms_of(extra, "per_symbol")
    assert all(np.array_equal(A[x], B[x]) for x in A if x in B)
    _, S = atoms_of(tr, "sorted")
    _, S2 = atoms_of(extra, "sorted")
    assert not all(np.array_equal(S[x], S2[x]) for x in S if x in S2)
    acc = np.mean([n in {o for s, r, o in tr if (s, r) == q} for q, (n, _) in zip(qs, a.query_many(qs))])
    assert acc > 0.5                                  # stessa distribuzione: funziona
    with pytest.raises(ValueError):
        families.make(family, 64, atoms="lexi")


def test_mapb_per_symbol_is_noop(graph):
    a = _mem("MAP-B", graph, 0)
    b = families.make("MAP-B", DIMS["MAP-B"], atoms="per_symbol")
    b.store(graph.triples)
    assert a.query_many(graph.queries) == b.query_many(graph.queries)


def test_bits_counts_duplicates_and_codebook_bits(graph):
    mem = _mem("MAP-I", graph, 0)
    dup = families.make("MAP-I", DIMS["MAP-I"])
    dup.store(graph.triples + graph.triples[:200])
    n2 = len(graph.triples) + min(200, len(graph.triples))
    assert dup.n_facts() == n2
    assert dup.bits() == dup.dim * math.ceil(math.log2(n2 + 1))
    m = graph.m
    assert mem.codebook_bits() == m * math.ceil(math.log2(m))
    one = families.make("HRR", 8)
    one.store([("a", "a", "a")])
    assert one.codebook_bits() == 0
