"""`abm size` (fase 4 del dimensionatore): budget, formati, codici di uscita, schema, export."""
import argparse
import json

import pytest

from abm import size_cli, sizing
from abm.cli import main


def _graph(n=120, seed=3):
    """Grafo uniforme piccolo dentro gli intervalli validati (N ≥ 100): soggetti e oggetti
    disgiunti, quindi niente alias né gemelli."""
    import random
    rnd = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        s, o = f"e{rnd.randrange(75)}", f"e{75 + rnd.randrange(75)}"
        r = f"r{rnd.randrange(4)}"
        if (s, r) in seen or frozenset((s, o)) in {frozenset((a, c)) for a, _b, c in out}:
            continue
        seen.add((s, r))
        out.append([s, r, o])
    return out


TRIPLES = _graph()


def _write(tmp_path, name, rows, sep=None):
    p = tmp_path / name
    if sep is None:
        p.write_text(json.dumps(rows))
    else:
        p.write_text("\n".join(sep.join(r) for r in rows) + "\n")
    return str(p)


@pytest.fixture
def jfile(tmp_path):
    return _write(tmp_path, "t.json", TRIPLES)


# ---- budget ------------------------------------------------------------------------

@pytest.mark.parametrize("text,bits", [("512000bits", 512000), ("1bit", 1), ("64KB", 64 * 8192),
                                       ("64kib", 64 * 8192), ("2B", 16), ("1.5MB", 12582912),
                                       ("1GB", 8 << 30), ("10 bytes", 80)])
def test_parse_size(text, bits):
    assert size_cli.parse_size(text) == bits


@pytest.mark.parametrize("text", ["64", "KB", "-1KB", "0bits", "3parsecs", ""])
def test_parse_size_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        size_cli.parse_size(text)


# ---- formati -------------------------------------------------------------------------

def test_formats_agree_and_normalize(tmp_path):
    rows = [["  a ", "r", "b"], ["b", "r", "c"]]
    j, fj = size_cli.load_rows(_write(tmp_path, "x.json", rows), 3)
    t, ft = size_cli.load_rows(_write(tmp_path, "x.tsv", [["# comment"], [""]] + rows,
                                      "\t"), 3)
    c, fc = size_cli.load_rows(_write(tmp_path, "x.csv", rows, ","), 3)
    assert (fj, ft, fc) == ("json", "tsv", "csv")
    assert j == t == c == [("a", "r", "b"), ("b", "r", "c")]


def test_csv_quoting(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text('"Smith, J.",born_in,"Rome"\n')
    assert size_cli.load_rows(str(p), 3)[0] == [("Smith, J.", "born_in", "Rome")]


@pytest.mark.parametrize("name,content", [
    ("x.txt", "[]"), ("x.json", "{"), ("x.json", "{}"), ("x.json", "[]"),
    ("x.json", '[["a", "b"]]'), ("x.tsv", "a\tb\n"), ("x.json", '[["a", " ", "c"]]')])
def test_bad_input_exit_1(tmp_path, name, content, capsys):
    p = tmp_path / name
    p.write_text(content)
    assert main(["size", str(p), "--target", "0.9"]) == 1
    assert "error" in capsys.readouterr().err


def test_missing_file_exit_1(tmp_path):
    assert main(["size", str(tmp_path / "nope.json"), "--target", "0.9"]) == 1


def test_case_collision_warned(tmp_path, capsys):
    rows = TRIPLES + [["Paris", "r9", "x1"], ["paris", "r9", "x2"]]
    assert main(["size", _write(tmp_path, "c.json", rows), "--target", "0.8",
                 "--families", "FHRR", "--json"]) in (0, 3)
    d = json.loads(capsys.readouterr().out)
    assert any("Paris" in w and "MAP-B" in w for w in d["input_warnings"])


# ---- codici di uscita ----------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--target", "1.5"], ["--target", "0"], [], ["--target", "0.9", "--families", "XYZ"],
    ["--target", "0.9", "--budget", "64"], ["--target", "0.9", "--seeds", "0", "--verify"],
    ["--target", "0.9", "--tol", "1"], ["--target", "0.9", "--export", "nengo"]])
def test_usage_errors_exit_2(jfile, argv):
    with pytest.raises(SystemExit) as e:
        main(["size", jfile] + argv)
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [["--seeds", "3"], ["--tol", "0.02"], ["--max-mem", "1GB"],
                                  ["--out", "x.json"]])
def test_options_needing_others_exit_2(jfile, argv):
    assert main(["size", jfile, "--target", "0.9"] + argv) == 2


def test_validated_exit_0(jfile, capsys):
    assert main(["size", jfile, "--target", "0.9"]) == 0
    out = capsys.readouterr().out
    assert out.index("GRAPH") < out.index("RECOMMENDATIONS") < out.index("NOTE")
    assert "not guarantees" in out and "[100, 1600]" in out


def test_small_graph_only_caution_exit_3(tmp_path, capsys):
    p = _write(tmp_path, "s.json", TRIPLES[:30])          # N < 100 → caution
    assert main(["size", p, "--target", "0.9"]) == 3
    assert "no reliable recommendation" in capsys.readouterr().out


def test_only_mapb_with_multirel_exit_3(tmp_path):
    rows = TRIPLES + [["e0", "rx", "e1"], ["e1", "ry", "e0"]]
    p = _write(tmp_path, "m.json", rows)
    assert main(["size", p, "--target", "0.9", "--families", "MAP-B"]) == 3


def test_above_ceiling_exit_1(tmp_path, capsys):
    rows = TRIPLES + [["a", "sib", "b"], ["b", "sib", "c"]]   # (b, sib) alias
    p = _write(tmp_path, "a.json", rows)
    assert main(["size", p, "--target", "1.0"]) == 1
    assert "ceiling" in capsys.readouterr().err


def test_budget_excludes_and_tsv_works(tmp_path, capsys):
    p = _write(tmp_path, "t.tsv", TRIPLES, "\t")
    code = main(["size", p, "--target", "0.9", "--budget", "1KB", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["input"]["format"] == "tsv"
    assert all(r["bits"] <= 8192 for r in d["recommendations"])
    assert {e["family"] for e in d["excluded"]} | {r["family"] for r in d["recommendations"]} \
        == set(sizing.FAMILIES)
    assert code == d["exit_code"]


def test_queries_file(tmp_path, capsys):
    q = _write(tmp_path, "q.csv", [r[:2] for r in TRIPLES[:10]], ",")
    p = _write(tmp_path, "t.json", TRIPLES)
    main(["size", p, "--target", "0.9", "--queries", q, "--json", "--families", "FHRR"])
    assert json.loads(capsys.readouterr().out)["graph"]["n_queries"] == 10


# ---- JSON e verifica -----------------------------------------------------------------

def test_json_schema_with_verify(jfile, capsys):
    code = main(["size", jfile, "--target", "0.85", "--families", "FHRR,bsdc-or",
                 "--verify", "--seeds", "2", "--json"])
    cap = capsys.readouterr()
    assert "VERIFY COST" in cap.err                    # stima prima della verifica, su stderr
    d = json.loads(cap.out)
    assert d["schema"] == size_cli.SCHEMA == "abm.size/1"
    assert set(d) >= {"abm_version", "input", "params", "graph", "input_warnings",
                      "recommendations", "excluded", "verify_cost_estimate", "reliable",
                      "exit_code", "note"}
    assert set(d["graph"]["stats"]) == set(size_cli.STAT_ORDER)
    assert d["params"]["families"] == ["FHRR", "BSDC-OR"] and d["params"]["seeds"] == 2
    bits = [r["bits"] for r in d["recommendations"]]
    assert bits == sorted(bits)
    for r in d["recommendations"]:
        v = r["verify"]
        assert set(v) >= {"measured", "measured_ci", "per_memory_q10", "verified", "tol"}
        assert 0 <= v["measured_ci"][0] <= v["measured"] <= v["measured_ci"][1] <= 1
    assert code == d["exit_code"] == (0 if d["reliable"] else 3)
    assert d["reliable"] == any(r["reliability"] == "validated" and r["verify"]["verified"]
                                for r in d["recommendations"])


def test_verify_text_table(jfile, capsys):
    main(["size", jfile, "--target", "0.85", "--families", "FHRR", "--verify", "--seeds", "2"])
    out = capsys.readouterr().out
    assert out.index("VERIFY COST") < out.index("RECOMMENDATIONS")
    assert "measured [95% CI]" in out and "q10" in out


# ---- export torchhd ------------------------------------------------------------------

def test_mapping_covers_all_families():
    assert set(size_cli.TORCHHD) == set(sizing.FAMILIES)
    valid = {"BSC", "MAP", "HRR", "FHRR", "BSBC", "VTB", "MCR", "CGR", None}
    assert all(v[0] in valid for v in size_cli.TORCHHD.values())
    assert size_cli.TORCHHD["BSDC-OR"][0] is None and size_cli.TORCHHD["BSDC-S"][0] is None


def test_export_out_file(jfile, tmp_path, capsys):
    out = tmp_path / "th.json"
    main(["size", jfile, "--target", "0.9", "--export", "torchhd", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["target_library"] == "torchhd" and "not installed" in d["verification"]
    fams = {m["family"]: m for m in d["models"]}
    assert fams["FHRR"]["torchhd_vsa"] == "FHRR" and "python" in fams["FHRR"]
    assert "vsa=\"FHRR\"" in fams["FHRR"]["python"]
    assert "python" not in fams["BSDC-OR"]
    compile(fams["MAP-B"]["python"], "<snippet>", "exec")      # sintassi valida


def test_export_in_json(jfile, capsys):
    main(["size", jfile, "--target", "0.9", "--families", "HRR", "--export", "torchhd",
          "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["export"]["models"][0]["torchhd_vsa"] == "HRR"
