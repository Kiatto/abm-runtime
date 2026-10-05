"""escalation3_prereg.py — una confidenza esatta del front-end, più domande, coperture corrette.

Preregistrazione: docs/preregistration/escalation3.md. Committato insieme a quel
file e PRIMA di interrogare i modelli per questo test.

Rispetto alla preregistrazione 13 (escalation2.md) cambiano tre cose:
  1. la confidenza del front-end piccolo non è più la probabilità del primo token,
     ma la distribuzione esatta sulle opzioni: P(opzione k) = prodotto delle
     probabilità dei token delle sue cifre, poi del token di fine turno, calcolate
     con llama-server (/apply-template, /completion con n_probs = 20); si
     esplorano le cifre con probabilità ≥ 0.01. FM = scarto fra le prime due
     opzioni, rinormalizzate sulle opzioni;
  2. tutte le 743 domande (anche le 100 di audit del test 11, che qui non servono);
  3. le coperture sono quote delle domande NON forzate: le forzate sono sempre
     astensioni.
Memoria: D = 16 384, la stessa per il percorso piccolo (Gemma) e il grande (Qwen).

Fasi:
  --stage small   Gemma sulla porta 8765: scelta (come nei test 11–13) e distribuzione
  --stage big     Qwen sulla porta 8766: scelta
  --stage analyze nessun modello
"""
import argparse
import json
import sys
import urllib.request
from math import exp, log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import escalation2_prereg as E2  # noqa: E402
import escalation_prereg as E1  # noqa: E402
import human_questions_prereg as H  # noqa: E402

RES = ROOT / "results"
OUT_SMALL = RES / "escalation3_small_answers.json"
OUT_BIG = RES / "escalation3_big_answers.json"
OUT = RES / "escalation3_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
RATES = (0.2, 0.3, 0.4, 0.5)            # quote di tutte le domande passate al grande
COVERAGES = (0.5, 0.7, 0.9)             # quote delle domande non forzate a cui rispondere
P_MIN = 0.01                            # cifre esplorate
BOOT = 10000
BOOT_SEED = 20261005
# -----------------------------------------------------------------------------
H.DIM = DIM
BASE = {"small": "http://127.0.0.1:8765", "big": "http://127.0.0.1:8766"}


def post(which, path, body):
    req = urllib.request.Request(BASE[which] + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def next_tokens(which, prompt):
    """Distribuzione (top 20) del prossimo token: {testo: probabilità}, più la massa di fine."""
    r = post(which, "/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 20,
                                    "temperature": 0})
    top = r["completion_probabilities"][0]["top_logprobs"]
    digits, end = {}, 0.0
    for t in top:
        p = exp(t["logprob"])
        if t["token"].strip() == "":          # fine turno, fine sequenza, a capo
            end += p
        elif t["token"].isdigit():
            digits[t["token"]] = digits.get(t["token"], 0.0) + p
    return digits, end


def option_distribution(which, question, name, relations):
    """P(opzione) per ogni indice 1..K, esplorando le cifre con probabilità ≥ P_MIN."""
    options = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(relations))
    msg = H.PROMPT.format(q=question, name=name, options=options)
    prompt = post(which, "/apply-template", {"messages": [{"role": "user", "content": msg}],
                                             "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    k_max = len(relations)
    probs = {}

    def walk(prefix, p):
        digits, end = next_tokens(which, prompt + prefix)
        if prefix and 1 <= int(prefix) <= k_max:
            probs[int(prefix)] = probs.get(int(prefix), 0.0) + p * end
        for d, q in digits.items():
            nxt = prefix + d
            if p * q >= P_MIN and int(nxt) <= k_max and not nxt.startswith("0"):
                walk(nxt, p * q)
    walk("", 1.0)
    return probs


def front(which, out_path):
    names, qs, batch_of, trip = setup()
    rows, memories = [], {}
    for n, i in enumerate(range(len(qs))):
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        row = {"i": i, "linked": linked, "k": len(rels), "chosen": None, "conf": None,
               "dist": None}
        if len(rels) == 1:
            row.update(chosen=rels[0], conf=1.0, dist={"1": 1.0})
        elif rels:
            E1.SERVERS[which] = BASE[which] + "/v1/chat/completions"
            row["chosen"], row["conf"], row["raw"] = E1.ask(which, q["q"], names.get(linked, linked)
                                                             .replace("_", " "), rels)
            if which == "small":
                row["dist"] = {str(a): b for a, b in option_distribution(
                    which, q["q"], names.get(linked, linked).replace("_", " "), rels).items()}
        rows.append(row)
        if n % 50 == 0:
            print(which, n, flush=True)
    out_path.write_text(json.dumps({"model": E1.MODELS[which], "rows": rows}, indent=1))
    print("->", out_path)


def setup():
    H.check_data()
    by_s, names, qs = H.load()
    batch_of, trip = H.batches(qs, by_s)
    return names, qs, batch_of, trip


def margin(dist):
    if not dist:
        return 0.0
    v = sorted(dist.values(), reverse=True)
    tot = sum(v)
    if tot <= 0:
        return 0.0
    v = [x / tot for x in v] + [0.0]
    return v[0] - v[1]


def selective(score, ok, forced, idx):
    free = [i for i in idx if not forced[i]]
    order = sorted(free, key=lambda i: (-score[i], i))
    return {c: float(np.mean([ok[i] for i in order[:int(round(c * len(free)))]]))
            for c in COVERAGES}


def analyze():
    names, qs, batch_of, trip = setup()
    small = {r["i"]: r for r in json.loads(OUT_SMALL.read_text())["rows"]}
    big = {r["i"]: r for r in json.loads(OUT_BIG.read_text())["rows"]}
    memories = {}
    ok_s, ok_b, forced, f1, fm, pmem, z, mid = [], [], [], [], [], [], [], []
    for i in range(len(qs)):
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        gold = bm.objects[(q["s"], q["r"])]
        s_, b_ = small[i], big[i]
        f = s_["chosen"] is None
        if f:
            ok_s.append(False); f1.append(0.0); fm.append(0.0); pmem.append(0.0); z.append(-1e9)
        else:
            noisy = H.abm.bind(bm.mem._trace, bm.mem.key(s_["linked"], s_["chosen"]))
            dist = np.count_nonzero(bm.matrix != noisy, axis=1)
            j = int(np.argmin(dist))
            ok_s.append(bm.names[j] in gold)
            f1.append(s_["conf"] if s_["conf"] is not None else 0.0)
            fm.append(margin(s_["dist"]))
            pmem.append(H.exact.predict_queries(trip[k], DIM, [(s_["linked"], s_["chosen"])])[0])
            z.append(float((DIM / 2 - dist[j]) / (np.sqrt(DIM) / 2)))
        ok_b.append(bool(b_["chosen"]) and bm.query(b_["linked"], b_["chosen"]) in gold)
        forced.append(f)
        mid.append(k)
    n = len(qs)
    ok_s, ok_b = np.array(ok_s, float), np.array(ok_b, float)
    f1, fm, pmem, z = map(np.array, (f1, fm, pmem, z))
    sel = {"F1": f1, "FM": fm, "CFM": pmem * fm, "ZFM": (E2.ranks(z) + E2.ranks(fm)) / 2}
    val = {"FM": 1 - fm, "EM": pmem * (1 - fm)}

    def metrics(idx):
        b = {g: selective(s, ok_s, forced, idx) for g, s in sel.items()}
        a = {g: E2.escalate(v, ok_s, ok_b, forced, idx) for g, v in val.items()}
        a["R"] = E2.escalate_random(ok_s, ok_b, forced, idx)
        return a, b
    E2.RATES = RATES
    full_a, full_b = metrics(list(range(n)))
    pairs = {"B:FM-F1": ("B", "FM", "F1"), "B:CFM-FM": ("B", "CFM", "FM"),
             "B:ZFM-FM": ("B", "ZFM", "FM"), "A:EM-FM": ("A", "EM", "FM"),
             "A:EM-R": ("A", "EM", "R")}
    boot = {k: [] for k in pairs}
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mid))
    groups = {m: [i for i in range(n) if mid[i] == m] for m in ids}
    for _ in range(BOOT):
        idx = [i for m in rng.choice(ids, len(ids)) for i in groups[m]]
        a, b = metrics(idx)
        for key, (part, x, y) in pairs.items():
            d = a if part == "A" else b
            boot[key].append(E2.mean(d[x]) - E2.mean(d[y]))
    diff = {}
    for key, (part, x, y) in pairs.items():
        d = full_a if part == "A" else full_b
        diff[key] = {"point": E2.mean(d[x]) - E2.mean(d[y]),
                     "ci95": [float(v) for v in np.percentile(boot[key], [2.5, 97.5])]}
    res = {"preregistration": "docs/preregistration/escalation3.md", "dim": DIM, "n": n,
           "forced": int(sum(forced)), "small_accuracy": float(ok_s.mean()),
           "big_accuracy": float(ok_b.mean()),
           "routing": {g: {str(r): v for r, v in c.items()} for g, c in full_a.items()},
           "selective": {g: {str(c): v for c, v in d.items()} for g, d in full_b.items()},
           "diff": diff,
           "per_question": [{"i": i, "memory": m, "small_ok": bool(a), "big_ok": bool(b),
                             "forced": f, "f1": float(x1), "fm": float(xm), "pmem": float(p),
                             "z": float(zz)}
                            for i, m, a, b, f, x1, xm, p, zz in zip(range(n), mid, ok_s, ok_b,
                                                                     forced, f1, fm, pmem, z)]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "forced", "small_accuracy", "big_accuracy",
                                         "routing", "selective", "diff")}, indent=1))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("small", "big", "analyze"), required=True)
    args = ap.parse_args()
    {"small": lambda: front("small", OUT_SMALL), "big": lambda: front("big", OUT_BIG),
     "analyze": analyze}[args.stage]()


if __name__ == "__main__":
    main()
