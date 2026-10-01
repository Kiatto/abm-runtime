"""proofwriter_seeds.py — rigenera la parte "proofwriter" di results/seed10_results.json.

Il file pubblicato (ProofWriter a 10 seed, Fig. 3, 99.8/99.1/92.4) non aveva uno
script che lo scrivesse: era una chiamata manuale a
seed_robustness.proofwriter_seeds(seeds=10, n=100). Questo script la rende
ripetibile. Scrive solo il campo "proofwriter"; law4 e law5 dello stesso file
non vengono da qui (law4.per_d si ricontrolla con capacity_seed10.py).

Deterministico: i codeword dipendono solo dal seed e dai nomi.

Uso, dalla root del repo:
    python examples/proofwriter_seeds.py [--seeds 10] [--n 100] [--out PATH]
"""
import argparse
import json
from pathlib import Path

from seed_robustness import proofwriter_seeds

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "results" / "seed10_proofwriter_check.json"))
    a = ap.parse_args()
    res = {"proofwriter": proofwriter_seeds(seeds=a.seeds, n=a.n)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n  -> {a.out}")
