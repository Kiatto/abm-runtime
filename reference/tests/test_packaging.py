"""Il pacchetto non deve sporcare il sys.path di chi lo importa (audit ostile 2026-10-09)."""
import subprocess
import sys
from pathlib import Path

import abm

PKG_PARENT = str(Path(abm.__file__).resolve().parents[1])


def test_import_abm_does_not_expose_top_level_modules(tmp_path):
    """Prima inspector.py faceva sys.path.insert(reference/): dopo `import abm`
    erano importabili `inspector`, `exact`, `cli` come moduli top-level."""
    code = ("import importlib.util, abm, abm.inspector, abm.cli, abm.exact\n"
            "print([m for m in ('inspector', 'exact', 'cli') "
            "if importlib.util.find_spec(m)])")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_public_names_are_exported():
    for name in ("margin_z", "Memory", "contract", "exact"):
        assert hasattr(abm, name)
    from abm.inspector import stats, contract, report, aliasing  # noqa: F401
