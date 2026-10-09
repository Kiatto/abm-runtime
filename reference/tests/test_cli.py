"""Smoke test della CLI `abm` (audit ostile 2026-10-09, punti 1, 4, 8)."""
import json
import subprocess
import sys

import pytest

from abm import exact
from abm.cli import main

TRIPLES = [["payment_service", "requires", "auth_service"],
           ["auth_service", "writes_to", "session_store"],
           ["a", "sib", "b"], ["b", "sib", "a"]]


@pytest.fixture
def triples_file(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(TRIPLES))
    return p


def test_inspect_prints_the_exact_contract(triples_file, capsys):
    assert main(["inspect", str(triples_file), "--dim", "1024", "--grounding", "0.9"]) == 0
    out = capsys.readouterr().out
    c = exact.contract_for([tuple(t) for t in TRIPLES], 1024)
    assert "exact model" in out
    assert f"{c['expected_accuracy']:.1%}" in out
    assert f"{c['ceiling']:.1%}" in out
    assert f"{0.9 * c['expected_accuracy']:.1%}" in out
    assert "Law IV" not in out              # il contratto superato solo a richiesta


def test_inspect_law_iv_is_opt_in_and_labelled(triples_file, capsys):
    assert main(["inspect", str(triples_file), "--law-iv"]) == 0
    out = capsys.readouterr().out
    assert "exact model" in out and "Law IV" in out and "superseded" in out


@pytest.mark.parametrize("dim", ["0", "-8", "2.5", "abc"])
def test_inspect_rejects_bad_dim(triples_file, dim, capsys):
    """Prima --dim 0 andava in loop infinito, --dim -8 dava una traceback numpy."""
    with pytest.raises(SystemExit) as e:
        main(["inspect", str(triples_file), "--dim", dim])
    assert e.value.code == 2
    assert "dim" in capsys.readouterr().err


@pytest.mark.parametrize("g", ["7", "-0.5", "nan"])
def test_inspect_rejects_grounding_outside_unit_interval(triples_file, g, capsys):
    """Prima --grounding 7 stampava 'Grounding >= 700%'."""
    with pytest.raises(SystemExit) as e:
        main(["inspect", str(triples_file), "--grounding", g])
    assert e.value.code == 2
    assert "grounding" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["[]", '[["a", "r"]]', "not json", '{"a": 1}'])
def test_inspect_rejects_degenerate_files(tmp_path, content, capsys):
    """Prima `[]` produceva un contratto al 100% su zero fatti."""
    p = tmp_path / "bad.json"
    p.write_text(content)
    assert main(["inspect", str(p)]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_inspect_missing_file(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_demo_uses_the_exact_contract(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "session_store" in out and "exact model" in out


def test_console_entry_point_runs_as_module():
    out = subprocess.run([sys.executable, "-m", "abm.cli", "demo"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "exact model" in out.stdout
