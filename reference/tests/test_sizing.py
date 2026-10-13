"""abm.sizing: le previsioni a priori del dimensionatore (fase 2 di `abm size`).

La regressione ricostruisce i sottografi come gli harness congelati delle preregistrazioni
21 e 22 (stessi semi, stesso campionamento, importandoli senza modificarli) e confronta
con le previsioni congelate in results/*_predictions.json. Salta se mancano i dati."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from abm import sizing

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SEEDS_CHECKED = 3


def _harness(name):
    sys.path.insert(0, str(EXAMPLES))
    return pytest.importorskip(name)


def _cells(v21, graph, seed_base, di, si, scheme, n, spawn):
    out = []
    for k in range(SEEDS_CHECKED):
        seqs = np.random.SeedSequence([seed_base, di, si, n, k]).spawn(spawn)
        rng = np.random.default_rng(seqs[0])
        out.append(v21.Cell(v21.sample(graph, n, scheme, rng), rng, seqs[1]))
    return out


def _check_rows(rows, cells):
    checked = 0
    for r in rows:
        if not r.get("D"):
            continue
        for k, c in enumerate(cells):
            p = sizing.predict(c.triples, r["family"], r["D"], c.queries)
            assert abs(p - r["pred_seed"][k]) <= 1e-9, (r["family"], r["D"], k)
            checked += 1
    return checked


def test_regression_prereg21_wn18rr_dense():
    v21 = _harness("vsa_families_prereg")
    try:
        graph = v21.Graph(v21.load("wn18rr"))
    except SystemExit as e:
        pytest.skip(str(e))
    cells = _cells(v21, graph, v21.SEED_BASE, 1, 1, "dense", 100, 2)
    rows = [r for r in json.loads((ROOT / "results" / "vsa_families_prereg_predictions.json")
                                  .read_text())
            if r["dataset"] == "wn18rr" and r["scheme"] == "dense" and r["N"] == 100
            and r["block"] == "calib" and r["family"] != "BSDC-S"]  # BSDC-S: modello della 22
    assert {r["family"] for r in rows} == {"MAP-B", "MAP-I", "FHRR", "BSDC-OR"}
    assert _check_rows(rows, cells) == len(rows) * SEEDS_CHECKED


def test_regression_prereg22_codex_dense():
    v22 = _harness("generalization_prereg")
    try:
        graph = v22.v21.Graph(v22.load("codex-m"))
    except SystemExit as e:
        pytest.skip(str(e))
    cells = _cells(v22.v21, graph, v22.SEED_BASE, 0, 1, "dense", 100, 3)
    data = json.loads((ROOT / "results" / "generalization_prereg_predictions.json").read_text())
    rows = [r for r in data["rows"] if r["dataset"] == "codex-m" and r["scheme"] == "dense"
            and r["N"] == 100 and r["hops"] == 1]
    assert {r["family"] for r in rows} == set(sizing.FAMILIES)
    assert _check_rows(rows, cells) == len(rows) * SEEDS_CHECKED


# ---- grafo sintetico ------------------------------------------------------------

def _synthetic(n_ent=5000, n_rel=20, n_facts=200, seed=0, twins=0.0):
    rng = np.random.default_rng(seed)
    fs = set()
    while len(fs) < n_facts:
        s, o = rng.integers(n_ent, size=2)
        if s != o:
            r = int(rng.integers(n_rel))
            fs.add((f"e{s}", f"r{r}", f"e{o}"))
            if rng.random() < twins:
                fs.add((f"e{o}", f"r{r}", f"e{s}"))
    return sorted(fs)


@pytest.mark.parametrize("family", sizing.FAMILIES)
def test_predict_monotone_and_min_dim(family):
    t = _synthetic()
    step = sizing.bsdc_length(len(t)) if family.startswith("BSDC") else sizing.STEP
    accs = [sizing.predict(t, family, step * k) for k in (1, 2, 4, 8, 16)]
    assert all(b >= a - 1e-9 for a, b in zip(accs, accs[1:])), accs
    d_lo = sizing.min_dim(t, family, 0.6)
    d_hi = sizing.min_dim(t, family, 0.9)
    assert d_lo is not None and d_hi is not None and d_lo <= d_hi
    assert d_lo % step == 0
    assert sizing.predict(t, family, d_hi) >= 0.9
    if d_hi > step:
        assert sizing.predict(t, family, d_hi - step) < 0.9


def test_min_dim_unreachable_above_ceiling():
    t = _synthetic(twins=0.1) + [("x", "r1", "e1"), ("e1", "r1", "y")]
    ceil = sizing._Graph(t).ceiling
    assert ceil < 1.0
    assert sizing.min_dim(t, "MAP-I", min(1.0, ceil + 0.01), d_max=4096) is None
    with pytest.raises(ValueError, match="alias ceiling"):
        sizing.recommend(t, min(1.0, ceil + 0.01))


@pytest.mark.parametrize("bad", [
    dict(family="MAP-X"), dict(dim=0), dict(dim=1.5), dict(dim=True)])
def test_predict_validation(bad):
    kw = dict(triples=_synthetic(), family="MAP-I", dim=256)
    kw.update(bad)
    with pytest.raises(ValueError):
        sizing.predict(**kw)


def test_input_validation():
    t = _synthetic()
    with pytest.raises(ValueError):
        sizing.predict([], "MAP-I", 256)
    with pytest.raises(ValueError):
        sizing.predict([("a", "b")], "MAP-I", 256)
    with pytest.raises(KeyError):
        sizing.predict(t, "MAP-I", 256, queries=[("nope", "r0")])
    with pytest.raises(ValueError):
        sizing.predict(t, "BSDC-S", 1)               # meno di un blocco
    with pytest.raises(ValueError):
        sizing.predict(t + [("e1", "r0", "e1")], "FHRR", 256)   # self-loop solo MAP-B
    for target in (0, 1.2, float("nan"), True, "0.8"):
        with pytest.raises(ValueError):
            sizing.min_dim(t, "MAP-I", target)
        with pytest.raises(ValueError):
            sizing.recommend(t, target)
    with pytest.raises(ValueError):
        sizing.recommend(t, 0.8, families=("MAP-X",))
    with pytest.raises(ValueError):
        sizing.recommend(t, 0.8, budget_bits=0)


def test_recommend_synthetic():
    t = _synthetic()
    recs = sizing.recommend(t, 0.85)
    assert {r.family for r in recs} == set(sizing.FAMILIES)
    bits = [r.bits for r in recs]
    assert bits == sorted(bits)
    for r in recs:
        assert r.predicted >= 0.85
        assert r.bits == r.dim * sizing.bits_per_component(r.family, len(t))
        assert r.kbytes == pytest.approx(r.bits / 8192)
        assert r.error_band.regime == "uniform"
        assert r.reliability == "validated"
        assert (r.family == "BSDC-S") == bool(r.warnings)
    cap = sorted(bits)[2]
    assert [r.bits for r in sizing.recommend(t, 0.85, budget_bits=cap)] == sorted(bits)[:3]
    only = sizing.recommend(t, 0.85, families=("FHRR",))
    assert [r.family for r in only] == ["FHRR"]


def test_recommend_warnings_dense_and_out_of_range():
    t = _synthetic(twins=0.1)
    g = sizing._Graph(t)
    assert g.regime == "dense" and g.twin_share <= sizing.TWIN_MAX_TESTED
    target = 0.8 * g.ceiling
    recs = {r.family: r for r in sizing.recommend(t, target)}
    assert recs["MAP-B"].reliability == "caution"
    assert any("YAGO3-10" in w for w in recs["MAP-B"].warnings)
    assert recs["MAP-B"].error_band.worst == 7.22
    assert recs["MAP-I"].reliability == "validated"
    # quasi tutti gemelli: oltre il massimo visto nei test
    many = [(f"a{i}", "r", f"b{i}") for i in range(150)] + \
           [(f"b{i}", "r", f"a{i}") for i in range(150)]
    rec = sizing.recommend(many, 0.6, families=("MAP-I",))[0]
    assert rec.reliability == "unvalidated"
    assert any("twin share" in w for w in rec.warnings)
    # sotto il carico minimo testato: cautela
    small = sizing.recommend(_synthetic(n_facts=40), 0.8, families=("FHRR",))[0]
    assert small.reliability == "caution"


def test_recommend_runtime_n1600():
    t = _synthetic(n_ent=20000, n_rel=20, n_facts=1600, seed=3)
    t0 = time.time()
    recs = sizing.recommend(t, 0.85)
    assert len(recs) == len(sizing.FAMILIES)
    assert time.time() - t0 < 120
