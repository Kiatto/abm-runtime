"""ABM — Algebraic Binary Memory.

A deterministic algebraic memory runtime for compiled symbolic
knowledge. Five concepts:

    from abm import Memory

    mem = Memory()
    mem.store("payment_service", "requires", "auth_service")
    mem.query("payment_service", "requires")   # ("auth_service", 0.99)
    mem.chain("payment_service", ["requires", "writes_to"])
    print(contract(mem))                       # the Memory Contract

The package maps onto the frozen reference implementation
(FORMALISM v2.1): `abm.abm` is the executable specification,
`abm.inspector` is the Memory Contract API of the paper's first versions,
based on the asymptotic Law IV. `abm.exact` is the finite-dimension model
without fitted parameters (paper v1.5+), and the contract to use when sizing a
memory in advance:

    from abm import exact
    exact.contract_for(triples, dim=4096)     # expected accuracy, alias ceiling
    exact.min_dimension(triples, target=0.8)  # smallest D that keeps the promise
"""

from .abm import (Memory, ItemMemory, bind, bundle, permute, random_hv,
                  hamming, phi, margin_z, confidence, capacity, predicted_accuracy,
                  z_gumbel, __version__)
from .inspector import stats, contract, report, aliasing
from . import exact

__all__ = ["Memory", "ItemMemory", "bind", "bundle", "permute",
           "random_hv", "hamming", "phi", "margin_z", "confidence", "capacity",
           "predicted_accuracy", "z_gumbel", "stats", "contract",
           "report", "aliasing", "exact", "__version__"]
