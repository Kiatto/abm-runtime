"""Rende importabile il pacchetto `abm` (= reference/) anche senza installarlo.

Con il pacchetto installato (pip install -e .) non fa nulla. Senza, registra
reference/ come pacchetto `abm`, così i test importano esattamente ciò che il
wheel spedisce, con gli import relativi, e mai i moduli come file top-level.
"""
import importlib.util
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1]

try:
    import abm  # noqa: F401
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "abm", REFERENCE / "__init__.py", submodule_search_locations=[str(REFERENCE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["abm"] = module
    spec.loader.exec_module(module)
