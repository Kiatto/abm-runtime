"""escalation_prereg.py — il contratto come criterio di escalation verso un LLM più grande.

Preregistrazione: docs/preregistration/escalation.md (ipotesi H6 di
PRODUCT_HYPOTHESES.md). Committato insieme a quel file e PRIMA di interrogare
qualsiasi modello su una domanda del dataset; lo smoke test usa una domanda inventata.

Domande, memorie, collegamento e prompt sono quelli del test 11
(examples/human_questions_prereg.py), con una sola differenza preregistrata: le
memorie sono a D = 8192 invece di 16 384, scelta perché la memoria sia un collo di
bottiglia (accuratezza prevista media 0.556 sulle coppie vere, calcolata solo da
abm.exact, prima di qualsiasi misura).

Due percorsi per ogni domanda:
  piccolo — Gemma 4 E2B sceglie la relazione, la memoria ABM risponde;
  grande  — Qwen3-4B sceglie la relazione, uno store esatto risponde.
Quattro criteri decidono quali domande passare al percorso grande, senza mai
guardare la risposta giusta:
  C  — contratto: abm.exact per (entità collegata, relazione scelta dal piccolo);
  F  — confidenza del piccolo: probabilità del primo token della sua risposta;
  CF — il prodotto C × F;
  R  — a caso (valore atteso, in forma chiusa).
Si escalano per primi i punteggi più bassi. Le domande che il piccolo non può
affrontare (nessuna entità collegata) vanno al grande con qualunque criterio.

Fasi, perché i due modelli non stanno insieme in 4 GB di VRAM:
  --stage small   server con Gemma sulla porta 8765: risposte e confidenze del piccolo
  --stage big     server con Qwen sulla porta 8766: risposte del grande
  --stage analyze nessun modello: memoria, store esatto, criteri, ipotesi
  --smoke         una domanda inventata, per provare i server
Uso, dalla root:
  llama-server -m <gemma.gguf> --port 8765 -c 4096 --parallel 1
  python examples/escalation_prereg.py --stage small
  llama-server -m <qwen.gguf> --port 8766 -c 4096 --parallel 1
  python examples/escalation_prereg.py --stage big
  python examples/escalation_prereg.py --stage analyze
"""
import argparse
import json
import sys
import urllib.request
from math import exp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import human_questions_prereg as H  # noqa: E402

RES = ROOT / "results"
OUT_SMALL = RES / "escalation_small_answers.json"
OUT_BIG = RES / "escalation_big_answers.json"
OUT = RES / "escalation_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 8192
RATES = (0.2, 0.3, 0.4, 0.5)          # quote passate al grande (9.2% sono forzate)
BOOT = 10000                          # bootstrap per cluster sulle 28 memorie
BOOT_SEED = 20261002
SERVERS = {"small": "http://127.0.0.1:8765/v1/chat/completions",
           "big": "http://127.0.0.1:8766/v1/chat/completions"}
MODELS = {"small": ("gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf",
                    "0a5bbc20f91f92da96ab4870fa71b356c45b8500a7b8b9c3e0eb48359b72da28"),
          "big": ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                  "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597")}
# -----------------------------------------------------------------------------
H.DIM = DIM          # BatchMemory costruisce la memoria con H.DIM: qui 8192, non 16 384


def ask(which, question, name, relations):
    """Indice scelto (o None) e probabilità del primo token generato."""
    options = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(relations))
    body = {"messages": [{"role": "user", "content": H.PROMPT.format(q=question, name=name,
                                                                     options=options)}],
            "temperature": 0, "max_tokens": 8, "logprobs": True, "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(SERVERS[which], data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    choice = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]
    text = choice["message"]["content"]
    digits = "".join(ch for ch in text if ch.isdigit())
    k = int(digits) - 1 if digits else -1
    content = (choice.get("logprobs") or {}).get("content") or []
    conf = exp(content[0]["logprob"]) if content else None
    return (relations[k] if 0 <= k < len(relations) else None), conf, text


def setup():
    H.check_data()
    by_s, names, qs = H.load()
    batch_of, trip = H.batches(qs, by_s)
    order = list(range(len(qs)))
    np.random.RandomState(H.SHUFFLE_SEED + 1).shuffle(order)
    return names, qs, batch_of, trip, order[H.AUDIT:]          # le 643 domande di test


def front(which, out_path):
    names, qs, batch_of, trip, test_idx = setup()
    rows, memories = [], {}
    for n, i in enumerate(test_idx):
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:                                   # serve solo a collegare
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        if not rels:
            chosen, conf, raw = None, None, None
        elif len(rels) == 1:
            chosen, conf, raw = rels[0], 1.0, None
        else:
            chosen, conf, raw = ask(which, q["q"], names.get(linked, linked).replace("_", " "),
                                    rels)
        rows.append({"i": i, "linked": linked, "chosen": chosen, "conf": conf, "raw": raw})
        if n % 50 == 0:
            print(which, n, flush=True)
    out_path.write_text(json.dumps({"model": MODELS[which], "rows": rows}, indent=1))
    print("->", out_path)


def curve(score, small_ok, big_ok, forced):
    """Accuratezza a ciascuna quota: si escalano prima le forzate, poi i punteggi più bassi."""
    n = len(small_ok)
    order = sorted(range(n), key=lambda i: (not forced[i], score[i], i))
    acc = {}
    for rate in RATES:
        k = max(int(round(rate * n)), int(sum(forced)))
        esc = set(order[:k])
        acc[rate] = float(np.mean([big_ok[i] if i in esc else small_ok[i] for i in range(n)]))
    return acc


def random_curve(small_ok, big_ok, forced):
    """Valore atteso della scelta a caso, a parità di forzate."""
    n = len(small_ok)
    f = [i for i in range(n) if forced[i]]
    free = [i for i in range(n) if not forced[i]]
    acc = {}
    for rate in RATES:
        k = max(int(round(rate * n)), len(f)) - len(f)
        frac = k / len(free)
        tot = sum(big_ok[i] for i in f) + sum(frac * big_ok[i] + (1 - frac) * small_ok[i]
                                              for i in free)
        acc[rate] = float(tot / n)
    return acc


def auc(c):
    return float(np.mean([c[r] for r in RATES]))


def analyze():
    names, qs, batch_of, trip, test_idx = setup()
    small = {r["i"]: r for r in json.loads(OUT_SMALL.read_text())["rows"]}
    big = {r["i"]: r for r in json.loads(OUT_BIG.read_text())["rows"]}
    memories = {}
    small_ok, big_ok, forced, sc, sf, mid, pred_kept = [], [], [], [], [], [], []
    for i in test_idx:
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        gold = bm.objects[(q["s"], q["r"])]
        s_, b_ = small[i], big[i]
        f = s_["chosen"] is None
        ans = bm.query(s_["linked"], s_["chosen"]) if not f else None
        small_ok.append(ans in gold)
        objs = sorted(bm.objects.get((b_["linked"], b_["chosen"]), ())) if b_["chosen"] else []
        big_ok.append(bool(objs) and objs[0] in gold)
        forced.append(f)
        c = 0.0 if f else H.exact.predict_queries(trip[k], DIM, [(s_["linked"], s_["chosen"])])[0]
        sc.append(c)
        sf.append(0.0 if f or s_["conf"] is None else s_["conf"])
        mid.append(k)
    small_ok, big_ok = np.array(small_ok, float), np.array(big_ok, float)
    scores = {"C": sc, "F": sf, "CF": [a * b for a, b in zip(sc, sf)]}

    def all_curves(idx):
        so, bo, fo = small_ok[idx], big_ok[idx], [forced[i] for i in idx]
        out = {g: curve([s[i] for i in idx], so, bo, fo) for g, s in scores.items()}
        out["R"] = random_curve(so, bo, fo)
        return out
    full = all_curves(np.arange(len(small_ok)))
    diffs = {"C-R": [], "C-F": [], "CF-F": [], "CF-C": []}
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mid))
    groups = {m: [i for i, x in enumerate(mid) if x == m] for m in ids}
    for _ in range(BOOT):
        idx = np.concatenate([groups[m] for m in rng.choice(ids, len(ids))]).astype(int)
        c = all_curves(idx)
        for key in diffs:
            a, b = key.split("-")
            diffs[key].append(auc(c[a]) - auc(c[b]))
    res = {"preregistration": "docs/preregistration/escalation.md", "dim": DIM,
           "n": len(small_ok), "forced": int(sum(forced)),
           "small_accuracy": float(small_ok.mean()), "big_accuracy": float(big_ok.mean()),
           "curves": {g: {str(r): v for r, v in c.items()} for g, c in full.items()},
           "auc": {g: auc(c) for g, c in full.items()},
           "diff": {k: {"point": auc(full[k.split("-")[0]]) - auc(full[k.split("-")[1]]),
                        "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
                    for k, v in diffs.items()},
           "per_question": [{"i": i, "memory": m, "small_ok": bool(a), "big_ok": bool(b),
                             "forced": f, "C": c, "F": x}
                            for i, m, a, b, f, c, x in zip(test_idx, mid, small_ok, big_ok,
                                                           forced, sc, sf)]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "forced", "small_accuracy", "big_accuracy",
                                         "auc", "diff")}, indent=1))
    print("->", OUT)


def smoke():
    rels = ["/people/person/nationality", "/people/person/place_of_birth"]
    for which in ("small", "big"):
        try:
            print(which, ask(which, "where was the poet born?", "the poet", rels))
        except OSError as e:
            print(which, "server non raggiungibile:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("small", "big", "analyze"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    if args.stage == "small":
        front("small", OUT_SMALL)
    elif args.stage == "big":
        front("big", OUT_BIG)
    elif args.stage == "analyze":
        analyze()
    else:
        ap.error("--stage o --smoke")


if __name__ == "__main__":
    main()
