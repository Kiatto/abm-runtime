"""proofwriter_baseline.py — ProofWriter: traccia olografica contro insieme esatto.

La review del paper (v1.3) chiedeva la baseline più ovvia: un insieme esatto al
posto della traccia. Questo script fa girare lo STESSO forward-chaining sulle
STESSE domande di proofwriter_eval.py, cambiando solo l'oracolo di
appartenenza:

- olografico: member() è un test di Hamming contro una traccia di D bit;
- esatto:     member() è un `in` su un set Python.

E misura quanta memoria usa ciascuno, in due modi:

- traccia: D bit, fissi, qualunque sia il numero di fatti;
- insieme esatto, codifica minima: ogni fatto (entità, attributo) come due
  indici, ceil(log2 V) bit ciascuno, con V = vocabolario del problema. È un
  limite inferiore generoso verso l'insieme, che in pratica costa di più.

Uso, dalla root del repo:  python examples/proofwriter_baseline.py [n_per_depth]
"""
import json
import sys
from math import ceil, log2
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proofwriter_eval as pe  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "proofwriter_baseline_results.json"
D = 4096  # la dimensione usata da proofwriter_eval.AlgebraicProver


class ExactProver(pe.AlgebraicProver):
    """Identico al prover algebrico, ma l'oracolo è un set esatto."""

    def __init__(self):
        super().__init__(state_dim=D)
        self.facts = set()

    def add_fact(self, ent, attr):
        super().add_fact(ent, attr)   # tiene aggiornate anche le entità
        self.facts.add((ent, attr))

    def member(self, ent, attr):
        return (ent, attr) in self.facts


def answer(prover, facts, rules, ent, neg, attr):
    for e, a in facts:
        prover.add_fact(e, a)
    prover.forward_chain(rules)
    provable = prover.member(ent, attr)
    return ("False" if provable else "Unknown") if neg else \
           ("True" if provable else "Unknown")


def main():
    n_per_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rows = [r for r in pq.read_table(pe.PARQUET).to_pylist()
            if r["id"].startswith("AttNoneg")]
    results = {}
    print(f"{'depth':>5} {'olografico':>10} {'esatto':>7} {'disaccordi':>10} "
          f"{'fatti finali':>12} {'traccia':>8} {'insieme min.':>12}")
    for depth in (0, 1, 2, 3, 5):
        sample = [r for r in rows if r["config"] == f"depth-{depth}"][:n_per_depth * 3]
        holo_ok = exact_ok = disagree = 0
        n_facts, set_bits = [], []
        done = 0
        for r in sample:
            if done >= n_per_depth:
                break
            facts, rules, coverage = pe.parse_theory(r["theory"])
            mq = pe.RE_Q.match(r["question"].strip())
            if not mq or coverage < 0.99:
                continue            # stesso filtro di proofwriter_eval.eval_row
            ent, neg, attr = mq.group(1).lower(), bool(mq.group(2)), mq.group(3).lower()
            gold = str(r["answer"])
            h = answer(pe.AlgebraicProver(state_dim=D), facts, rules, ent, neg, attr)
            ex_prover = ExactProver()
            e = answer(ex_prover, facts, rules, ent, neg, attr)
            holo_ok += h == gold
            exact_ok += e == gold
            disagree += h != e
            nf = len(ex_prover.facts)
            vocab = len({x for f in ex_prover.facts for x in f}) + 1
            n_facts.append(nf)
            set_bits.append(nf * 2 * ceil(log2(max(vocab, 2))))
            done += 1
        results[depth] = {
            "n": done,
            "holographic_acc": holo_ok / done,
            "exact_acc": exact_ok / done,
            "disagreements": disagree,
            "final_facts_mean": float(np.mean(n_facts)),
            "trace_bytes": D // 8,
            "exact_min_bytes_mean": float(np.mean(set_bits)) / 8,
        }
        v = results[depth]
        print(f"{depth:>5} {v['holographic_acc']:>9.1%} {v['exact_acc']:>7.1%} "
              f"{disagree:>10} {v['final_facts_mean']:>12.1f} "
              f"{v['trace_bytes']:>6} B {v['exact_min_bytes_mean']:>10.1f} B")
    OUT.write_text(json.dumps(results, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
