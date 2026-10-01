"""Test 11, post-hoc statistics (not preregistered).

Reads results/human_questions_prereg_results.json and recomputes each question's
memory id with the deterministic batches() of human_questions_prereg.py. Reports:
the strict end-to-end score (answer right and relation right), cluster-bootstrap
intervals over the memories, the memory-level split by front-end correctness,
and the memory-level error by memory size. No language model needed.

    .venv/bin/python examples/human_questions_stats.py
"""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import human_questions_prereg as H  # noqa: E402

d = json.loads((ROOT / "results" / "human_questions_prereg_results.json").read_text())
H.check_data()
by_s, names, qs = H.load()
batch_of, trip = H.batches(qs, by_s)
P = d["per_question"]
n = len(P)
mid = np.array([batch_of[p["s"]] for p in P])
e2e = np.array([p["end_to_end_ok"] for p in P], float)
strict = np.array([p["end_to_end_ok"] and p["relation_ok"] for p in P], float)
fe = np.array([p["relation_ok"] for p in P], float)
mem = np.array([p["memory_ok_given_gold"] for p in P], float)
pred = np.array([H.exact.predict_queries(trip[batch_of[p["s"]]], H.DIM, [(p["s"], p["r"])])[0] for p in P])
C = d["contract"]["predicted"]
print("n", n, "memories", len(set(mid)), "of", len(trip))
print("end_to_end %.4f  strict %.4f  (vs pred %.4f: %+.1f, %+.1f pts)" % (
    e2e.mean(), strict.mean(), C, 100 * (e2e.mean() - C), 100 * (strict.mean() - C)))
print("memory given gold %.4f pred %.4f diff %+.2f pts; binomial SE %.4f" % (
    mem.mean(), pred.mean(), 100 * (mem.mean() - pred.mean()), sqrt(pred.mean() * (1 - pred.mean()) / n)))
rng = np.random.RandomState(20261001)
ids = np.unique(mid)
groups = {k: np.where(mid == k)[0] for k in ids}
bs = {"e2e": [], "strict": [], "memdiff": []}
for _ in range(10000):
    idx = np.concatenate([groups[k] for k in rng.choice(ids, len(ids))])
    bs["e2e"].append(e2e[idx].mean())
    bs["strict"].append(strict[idx].mean())
    bs["memdiff"].append(mem[idx].mean() - pred[idx].mean())
for k, v in bs.items():
    v = np.array(v)
    print("cluster bootstrap %s: sd %.4f, 95%% [%.4f, %.4f]" % (k, v.std(), *np.percentile(v, [2.5, 97.5])))
a, b = mem[fe == 1], mem[fe == 0]
p = mem.mean()
z = (a.mean() - b.mean()) / sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
print("memory | front ok %.4f (n=%d)  | front wrong %.4f (n=%d)  z=%.2f (exploratory)" % (
    a.mean(), len(a), b.mean(), len(b), z))
size = np.array([len(trip[k]) for k in mid])
for lo, hi in [(0, 100), (100, 500), (500, 1000), (1000, 10**6)]:
    m = (size >= lo) & (size < hi)
    if m.sum():
        print("size [%d,%d): n=%d memories=%d meas %.3f pred %.3f diff %+.1f pts" % (
            lo, hi, m.sum(), len(set(mid[m])), mem[m].mean(), pred[m].mean(),
            100 * (mem[m].mean() - pred[m].mean())))

out = {"n": n, "memories": int(len(ids)), "contract": C,
       "end_to_end": float(e2e.mean()), "strict": float(strict.mean()),
       "memory_given_gold": float(mem.mean()), "memory_pred": float(pred.mean()),
       "cluster_bootstrap": {k: {"sd": float(np.std(v)),
                                 "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
                             for k, v in bs.items()},
       "memory_by_front_end": {"ok": float(a.mean()), "n_ok": int(len(a)),
                               "wrong": float(b.mean()), "n_wrong": int(len(b)), "z": float(z)},
       "by_size": [{"lo": lo, "hi": hi, "n": int(((size >= lo) & (size < hi)).sum()),
                    "measured": float(mem[(size >= lo) & (size < hi)].mean()),
                    "predicted": float(pred[(size >= lo) & (size < hi)].mean())}
                   for lo, hi in [(0, 100), (100, 500), (500, 1000), (1000, 10**6)]
                   if ((size >= lo) & (size < hi)).sum()]}
(ROOT / "results" / "human_questions_stats_results.json").write_text(json.dumps(out, indent=1))
print("->", ROOT / "results" / "human_questions_stats_results.json")
