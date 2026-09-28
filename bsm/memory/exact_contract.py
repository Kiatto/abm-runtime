"""Rinvio: il modello esatto ora sta nel pacchetto, in reference/exact.py (abm.exact).

Tenuto perché gli harness delle preregistrazioni lo importano con questo nome, e
il loro codice non va cambiato dopo il commit che li ha registrati.
"""
import importlib.util as _util
from pathlib import Path as _Path

_spec = _util.spec_from_file_location(
    "_abm_exact", _Path(__file__).resolve().parents[2] / "reference" / "exact.py")
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
