"""escalation2_prereg.py — instradamento e indice di confidenza, a parità di memoria.

Preregistrazione: docs/preregistration/escalation2.md. Committato insieme a quel
file e PRIMA di calcolare qualsiasi misura di questo test.

Nessun modello viene interrogato: le scelte di relazione di Gemma 4 E2B (piccolo)
e di Qwen3-4B (grande) sulle 643 domande di test sono deterministiche e già
registrate dalla preregistrazione 12 (results/escalation_small_answers.json,
results/escalation_big_answers.json). Cambia l'impianto:

  - una sola memoria ABM a D = 16 384 (quella del test 11) per entrambi i percorsi:
    piccolo = Gemma sceglie la relazione, grande = Qwen la sceglie; la memoria
    risponde. Il costo che l'escalation spende è la chiamata al modello grande.

Parte A — instradamento. Si passano al grande le domande con il valore più alto:
  F  — 1 − confidenza del piccolo;
  E  — p̂_mem × (1 − confidenza del piccolo): passare conviene se la memoria
       risponderà bene e il piccolo è insicuro (p̂_mem: abm.exact per entità
       collegata e relazione scelta dal piccolo);
  R  — a caso (valore atteso).
Parte B — indice di confidenza per rispondere o astenersi (solo percorso piccolo):
si risponde alle domande con il punteggio più alto, a copertura 50/70/90%, e si
misura l'accuratezza delle risposte date:
  F  — confidenza del piccolo;
  CF — p̂_mem × F;
  ZF — media dei ranghi di Z e di F, con Z il margine osservato della risposta:
       (D/2 − d) / (√D / 2), d la distanza di Hamming fra la query rumorosa e il
       codeword restituito.
Uso, dalla root:  python examples/escalation2_prereg.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import human_questions_prereg as H  # noqa: E402

RES = ROOT / "results"
OUT = RES / "escalation2_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
RATES = (0.2, 0.3, 0.4, 0.5)
COVERAGES = (0.5, 0.7, 0.9)
BOOT = 10000
BOOT_SEED = 20261004
# -----------------------------------------------------------------------------
H.DIM = DIM


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def escalate(value, small_ok, big_ok, forced, idx):
    """Accuratezza alle quote: prima le forzate, poi il valore più alto."""
    n = len(idx)
    order = sorted(idx, key=lambda i: (not forced[i], -value[i], i))
    out = {}
    for rate in RATES:
        k = max(int(round(rate * n)), sum(forced[i] for i in idx))
        esc = set(order[:k])
        out[rate] = float(np.mean([big_ok[i] if i in esc else small_ok[i] for i in idx]))
    return out


def escalate_random(small_ok, big_ok, forced, idx):
    n = len(idx)
    f = [i for i in idx if forced[i]]
    free = [i for i in idx if not forced[i]]
    out = {}
    for rate in RATES:
        k = max(int(round(rate * n)), len(f)) - len(f)
        frac = k / len(free)
        out[rate] = float((sum(big_ok[i] for i in f)
                           + sum(frac * big_ok[i] + (1 - frac) * small_ok[i] for i in free)) / n)
    return out


def selective(score, small_ok, forced, idx):
    """Accuratezza delle risposte date, rispondendo ai punteggi più alti."""
    n = len(idx)
    order = sorted(idx, key=lambda i: (forced[i], -score[i], i))
    out = {}
    for c in COVERAGES:
        k = int(round(c * n))
        out[c] = float(np.mean([small_ok[i] for i in order[:k]]))
    return out


def mean(d):
    return float(np.mean(list(d.values())))


def main():
    H.check_data()
    by_s, names, qs = H.load()
    batch_of, trip = H.batches(qs, by_s)
    order = list(range(len(qs)))
    np.random.RandomState(H.SHUFFLE_SEED + 1).shuffle(order)
    test_idx = order[H.AUDIT:]
    small = {r["i"]: r for r in json.loads((RES / "escalation_small_answers.json").read_text())["rows"]}
    big = {r["i"]: r for r in json.loads((RES / "escalation_big_answers.json").read_text())["rows"]}
    memories = {}
    small_ok, big_ok, forced, conf, pmem, z, mid = [], [], [], [], [], [], []
    for i in test_idx:
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        gold = bm.objects[(q["s"], q["r"])]
        s_, b_ = small[i], big[i]
        f = s_["chosen"] is None
        if f:
            small_ok.append(False); conf.append(0.0); pmem.append(0.0); z.append(-1e9)
        else:
            noisy = H.abm.bind(bm.mem._trace, bm.mem.key(s_["linked"], s_["chosen"]))
            dist = np.count_nonzero(bm.matrix != noisy, axis=1)
            j = int(np.argmin(dist))
            small_ok.append(bm.names[j] in gold)
            conf.append(s_["conf"] if s_["conf"] is not None else 0.0)
            pmem.append(H.exact.predict_queries(trip[k], DIM, [(s_["linked"], s_["chosen"])])[0])
            z.append(float((DIM / 2 - dist[j]) / (np.sqrt(DIM) / 2)))
        big_ok.append(bool(b_["chosen"]) and bm.query(b_["linked"], b_["chosen"]) in gold)
        forced.append(f)
        mid.append(k)
    n = len(test_idx)
    small_ok, big_ok = np.array(small_ok, float), np.array(big_ok, float)
    conf, pmem, z = np.array(conf), np.array(pmem), np.array(z)
    values = {"F": 1 - conf, "E": pmem * (1 - conf)}
    zf = (ranks(z) + ranks(conf)) / 2
    scores = {"F": conf, "CF": pmem * conf, "ZF": zf}

    def all_metrics(idx):
        a = {g: escalate(v, small_ok, big_ok, forced, idx) for g, v in values.items()}
        a["R"] = escalate_random(small_ok, big_ok, forced, idx)
        b = {g: selective(s, small_ok, forced, idx) for g, s in scores.items()}
        return a, b
    full_a, full_b = all_metrics(list(range(n)))
    pairs = {"A:E-F": ("A", "E", "F"), "A:E-R": ("A", "E", "R"),
             "B:ZF-CF": ("B", "ZF", "CF"), "B:CF-F": ("B", "CF", "F")}
    boot = {k: [] for k in pairs}
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mid))
    groups = {m: [i for i in range(n) if mid[i] == m] for m in ids}
    for _ in range(BOOT):
        idx = [i for m in rng.choice(ids, len(ids)) for i in groups[m]]
        a, b = all_metrics(idx)
        for key, (part, x, y) in pairs.items():
            d = a if part == "A" else b
            boot[key].append(mean(d[x]) - mean(d[y]))
    point = {}
    for key, (part, x, y) in pairs.items():
        d = full_a if part == "A" else full_b
        point[key] = {"point": mean(d[x]) - mean(d[y]),
                      "ci95": [float(v) for v in np.percentile(boot[key], [2.5, 97.5])]}
    res = {"preregistration": "docs/preregistration/escalation2.md", "dim": DIM, "n": n,
           "forced": int(sum(forced)), "small_accuracy": float(small_ok.mean()),
           "big_accuracy": float(big_ok.mean()),
           "small_right_big_wrong": int(sum((small_ok == 1) & (big_ok == 0))),
           "big_right_small_wrong": int(sum((big_ok == 1) & (small_ok == 0))),
           "routing": {g: {str(r): v for r, v in c.items()} for g, c in full_a.items()},
           "selective": {g: {str(c): v for c, v in d.items()} for g, d in full_b.items()},
           "diff": point,
           "per_question": [{"i": i, "memory": m, "small_ok": bool(a), "big_ok": bool(b),
                             "forced": f, "conf": float(c), "pmem": float(p), "z": float(zz)}
                            for i, m, a, b, f, c, p, zz in zip(test_idx, mid, small_ok, big_ok,
                                                                 forced, conf, pmem, z)]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "forced", "small_accuracy", "big_accuracy",
                                         "small_right_big_wrong", "big_right_small_wrong",
                                         "routing", "selective", "diff")}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
