"""human_questions_prereg.py — il contratto su domande scritte da persone.

Preregistrazione: docs/preregistration/human_questions.md. Committato insieme a
quel file e PRIMA di interrogare il modello linguistico su qualsiasi domanda del
dataset. Lo smoke test usa una domanda inventata, non del dataset.

Domande: SimpleQuestions v2 (Bordes et al., 2015), scritte da persone, ciascuna
con la tripla Freebase che la risponde; si usano le 743 la cui tripla è esattamente
in FB15k-237 (stessi identificativi Freebase).

Pipeline, come la userebbe qualcuno:
  1. collegamento: il nome più lungo, fra le entità della memoria, contenuto
     nella domanda (confronto in minuscolo, "_" come spazio);
  2. relazione: un modello linguistico LOCALE (Gemma 4 E2B, llama.cpp, temperatura
     0, senza ragionamento) sceglie per indice fra le relazioni memorizzate per
     l'entità collegata;
  3. memoria: la reference congelata risponde a (entità, relazione).
Contratto, emesso DOPO un audit di 100 domande e PRIMA delle 643 di test:
    accuratezza prevista = pi_audit × media sulle domande di test di
                           abm.exact (gemelli e alias) per (s, r) veri,
con l'intervallo al 95% di pi_audit (Wilson).

Serve il server locale:  llama-server -m <gguf> --port 8765 (vedi il file di
preregistrazione). Uso, dalla root:  python examples/human_questions_prereg.py [--smoke]

--memory-only rifà, senza modello linguistico, la parte deterministica: la
previsione di abm.exact e la risposta della memoria alla (s, r) vera, per le
stesse 643 domande di test. Scrive results/human_questions_memory_check.json
(non tocca i risultati pubblicati); replicate.py lo confronta con quelli.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

DATA = ROOT / "data" / "external"
OUT = ROOT / "results" / "human_questions_prereg_results.json"
OUT_MEMORY = ROOT / "results" / "human_questions_memory_check.json"
SHA = {"fb15k237_train.txt": "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae",
       "fb15k_mid2name.txt": "4da94b8059a83bc7e08c832f573d34221d38e85030576e332a0d0e9726d13d73",
       "SimpleQuestions_v2.tgz": "58f65630895de4f9712eeb33458ca20538972436fd48bf5913df4765e6788bf5"}
LLM = "http://127.0.0.1:8765/v1/chat/completions"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
BATCH_SUBJECTS = 25
AUDIT = 100
SHUFFLE_SEED = 20260928
PROMPT = ("Question: {q}\n"
          "Which relation of '{name}' answers the question? Reply with the number only.\n"
          "{options}")
# -----------------------------------------------------------------------------


def check_data():
    for name, sha in SHA.items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"sha256 inatteso: {name}")


def load():
    fb = [tuple(l.rstrip("\n").split("\t")) for l in (DATA / "fb15k237_train.txt").open()]
    fbs = set(fb)
    by_s = defaultdict(list)
    for t in fb:
        by_s[t[0]].append(t)
    names = {}
    for l in (DATA / "fb15k_mid2name.txt").open():
        m, n = l.rstrip("\n").split("\t", 1)
        names[m] = n
    qs = []
    for split in ("train", "valid", "test"):
        for l in (DATA / "SimpleQuestions_v2" / f"annotated_fb_data_{split}.txt").open():
            s, r, o, q = l.rstrip("\n").split("\t")
            s, r, o = (x.replace("www.freebase.com", "") for x in (s, r, o))
            if (s, r, o) in fbs:
                qs.append({"s": s, "r": r, "o": o, "q": q})
    return by_s, names, qs


def batches(qs, by_s):
    subj = sorted({q["s"] for q in qs})
    np.random.RandomState(SHUFFLE_SEED).shuffle(subj)
    groups = [subj[i:i + BATCH_SUBJECTS] for i in range(0, len(subj), BATCH_SUBJECTS)]
    batch_of = {s: k for k, g in enumerate(groups) for s in g}
    triples = [[t for s in g for t in by_s[s]] for g in groups]
    return batch_of, triples


def ask_llm(question, name, relations):
    options = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(relations))
    body = {"messages": [{"role": "user", "content": PROMPT.format(q=question, name=name,
                                                                  options=options)}],
            "temperature": 0, "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(LLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    digits = "".join(ch for ch in text if ch.isdigit())
    k = int(digits) - 1 if digits else -1
    return relations[k] if 0 <= k < len(relations) else None


class BatchMemory:
    def __init__(self, triples, names):
        self.mem = abm.Memory(DIM)
        for t in triples:
            self.mem._facts.append(self.mem.fact_hv(*t))
        self.mem._trace = abm.bundle(self.mem._facts)
        self.names = self.mem.items._names
        self.matrix = np.stack(self.mem.items._states)
        self.out = defaultdict(set)
        self.objects = defaultdict(set)
        for s, r, o in triples:
            self.out[s].add(r)
            self.objects[(s, r)].add(o)
        ents = {x for s, _r, o in triples for x in (s, o)}
        self.labels = sorted(((names[e].replace("_", " ").lower(), e) for e in ents if e in names),
                             key=lambda x: -len(x[0]))

    def link(self, question):
        q = " " + question.lower() + " "
        for label, e in self.labels:
            if label and f" {label} " in q:
                return e
        for label, e in self.labels:                     # senza confini di parola
            if label and label in q:
                return e
        return None

    def query(self, s, r):
        noisy = abm.bind(self.mem._trace, self.mem.key(s, r))
        return self.names[int(np.argmin(np.count_nonzero(self.matrix != noisy, axis=1)))]


def run_question(bm, names, q):
    linked = bm.link(q["q"])
    rels = sorted(bm.out.get(linked, [])) if linked else []
    if not rels:
        chosen = None
    elif len(rels) == 1:
        chosen = rels[0]
    else:
        chosen = ask_llm(q["q"], names.get(linked, linked).replace("_", " "), rels)
    answer = bm.query(linked, chosen) if linked and chosen else None
    return {"linked_ok": linked == q["s"], "relation_ok": linked == q["s"] and chosen == q["r"],
            "end_to_end_ok": answer in bm.objects[(q["s"], q["r"])],
            "memory_ok_given_gold": bm.query(q["s"], q["r"]) in bm.objects[(q["s"], q["r"])]}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--memory-only", action="store_true")
    args = ap.parse_args()
    check_data()
    by_s, names, qs = load()
    batch_of, trip = batches(qs, by_s)
    order = list(range(len(qs)))
    np.random.RandomState(SHUFFLE_SEED + 1).shuffle(order)
    audit_idx, test_idx = order[:AUDIT], order[AUDIT:]
    if args.smoke:                                       # domanda inventata, non del dataset
        k = 0
        bm = BatchMemory(trip[k], names)
        s = next(x for x in bm.out if x in names)
        fake = {"s": s, "r": sorted(bm.out[s])[0], "o": "", "q": f"tell me about {names[s].replace('_', ' ')}"}
        res = run_question(bm, names, fake)
        print("smoke ok:", len(qs), "domande,", len(trip), "memorie, chiavi:", sorted(res))
        return
    memories = {}
    if args.memory_only:
        mem_pred, rows = [], []
        for i in test_idx:
            q = qs[i]
            k = batch_of[q["s"]]
            if k not in memories:
                memories[k] = BatchMemory(trip[k], names)
            bm = memories[k]
            mem_pred.append(exact.predict_queries(trip[k], DIM, [(q["s"], q["r"])])[0])
            rows.append(bm.query(q["s"], q["r"]) in bm.objects[(q["s"], q["r"])])
        OUT_MEMORY.write_text(json.dumps(
            {"memory_pred": float(np.mean(mem_pred)),
             "memory_given_gold": float(np.mean(rows)),
             "memory_ok_given_gold": [bool(x) for x in rows]}, indent=1))
        print("->", OUT_MEMORY)
        return

    def bm_for(q):
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = BatchMemory(trip[k], names)
        return memories[k]

    # 1. audit
    audit = [run_question(bm_for(qs[i]), names, qs[i]) for i in audit_idx]
    k = sum(a["relation_ok"] for a in audit)
    pi, lo, hi = wilson(k, len(audit))
    # 2. contratto, prima delle domande di test
    mem_pred = []
    for i in test_idx:
        q = qs[i]
        mem_pred.append(exact.predict_queries(trip[batch_of[q["s"]]], DIM, [(q["s"], q["r"])])[0])
    m = float(np.mean(mem_pred))
    contract = {"pi_audit": pi, "pi_ci": [lo, hi], "memory_pred": m,
                "predicted": pi * m, "predicted_ci": [lo * m, hi * m]}
    print("contratto:", {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in contract.items()},
          flush=True)
    # 3. domande di test
    test = [run_question(bm_for(qs[i]), names, qs[i]) for i in test_idx]
    out = {"preregistration": "docs/preregistration/human_questions.md",
           "audit": {"n": len(audit), "front_end_ok": k,
                     "linked_ok": sum(a["linked_ok"] for a in audit)},
           "contract": contract,
           "test": {"n": len(test),
                    "end_to_end": float(np.mean([t["end_to_end_ok"] for t in test])),
                    "front_end": float(np.mean([t["relation_ok"] for t in test])),
                    "linked": float(np.mean([t["linked_ok"] for t in test])),
                    "memory_given_gold": float(np.mean([t["memory_ok_given_gold"] for t in test])),
                    "memory_pred": m},
           "per_question": [{**qs[i], **t} for i, t in zip(test_idx, test)]}
    OUT.write_text(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
