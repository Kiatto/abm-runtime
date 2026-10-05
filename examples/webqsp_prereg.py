"""webqsp_prereg.py — il front-end a embedding su domande naturali (WebQuestionsSP).

Preregistrazione: docs/preregistration/webqsp.md. Committato insieme a quel file
e PRIMA di interrogare qualsiasi modello su queste domande.

La preregistrazione 16 ha trovato che la sola relazione più simile alla domanda
(bge-small) batte Gemma, ma su SimpleQuestions, dove le domande sono state scritte
guardando la tripla. Qui le domande vengono dalle ricerche Google (WebQuestionsSP,
Yih et al. 2016): le 515 di train e test con catena di una relazione e la tripla
(entità, relazione, risposta) in FB15k-237.

Pipeline come nel test 11: collegamento per nome, memorie ABM da 10 soggetti con
tutte le loro triple di FB15k-237, D = 16 384. Quattro front-end per la relazione:
  E — solo embedding: la relazione più simile alla domanda;
  S — shortlist di 3 (k fissato nella preregistrazione 16) + Gemma con anteprima;
  G — Gemma con tutte le opzioni e anteprima (prompt della preregistrazione 15);
  Q — Qwen3-4B con tutte le opzioni, prompt dei test 11–14.
Una risposta è giusta se è una delle risposte annotate (entità) della domanda.

Fasi:
  --stage embed   server bge-small sulla porta 8767: E e le shortlist
  --stage gemma   server Gemma sulla porta 8765 (-c 8192): S e G
  --stage qwen    server Qwen sulla porta 8766 (-c 8192): Q
  --stage analyze nessun modello
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import human_questions_prereg as H  # noqa: E402
import memory_preview_prereg as MP  # noqa: E402
import shortlist_prereg as SL  # noqa: E402

DATA = ROOT / "data" / "external"
RES = ROOT / "results"
OUT_ANS = RES / "webqsp_answers.json"
OUT = RES / "webqsp_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
K = 3                                   # dalla preregistrazione 16, non ritarato
WEBQSP_SHA = "95cb9cd2f6b4e1116ba4bec348f08a7f5ca2a5df4f223ea5af3f9b25a971b0ee"
BATCH_SUBJECTS = 10                     # 25 dava solo 11 memorie (11 cluster)
SHUFFLE_SEED = 20261008
BOOT = 10000
BOOT_SEED = 20261008
NONINF = 0.05                           # margine di non inferiorità di E rispetto a Q
# -----------------------------------------------------------------------------
H.DIM = DIM
QWEN = "http://127.0.0.1:8766/v1/chat/completions"


def mid(x):
    return "/" + x.replace(".", "/")


def load():
    if hashlib.sha256((DATA / "WebQSP.zip").read_bytes()).hexdigest() != WEBQSP_SHA:
        raise SystemExit("sha256 inatteso per WebQSP.zip")
    H.check_data()
    fb = [tuple(l.rstrip("\n").split("\t")) for l in (DATA / "fb15k237_train.txt").open()]
    fbs = set(fb)
    by_s = defaultdict(list)
    for t in fb:
        by_s[t[0]].append(t)
    ents = set(by_s) | {o for _s, _r, o in fb}
    names = {}
    for l in (DATA / "fb15k_mid2name.txt").open():
        m, n = l.rstrip("\n").split("\t", 1)
        names[m] = n
    qs = []
    for split in ("train", "test"):
        for q in json.loads((DATA / "WebQSP" / "data" / f"WebQSP.{split}.json").read_text())["Questions"]:
            p = q["Parses"][0]
            ch = p.get("InferentialChain") or []
            if len(ch) != 1 or not p.get("TopicEntityMid"):
                continue
            s, r = mid(p["TopicEntityMid"]), mid(ch[0])
            ans = [mid(a["AnswerArgument"]) for a in p["Answers"] if a["AnswerType"] == "Entity"]
            if s in ents and any((s, r, a) in fbs for a in ans):
                qs.append({"id": q["QuestionId"], "s": s, "r": r, "answers": ans,
                           "q": q["ProcessedQuestion"]})
    subj = sorted({q["s"] for q in qs})
    np.random.RandomState(SHUFFLE_SEED).shuffle(subj)
    groups = [subj[i:i + BATCH_SUBJECTS] for i in range(0, len(subj), BATCH_SUBJECTS)]
    batch_of = {s: k for k, g in enumerate(groups) for s in g}
    trip = [[t for s in g for t in by_s[s]] for g in groups]
    return names, qs, batch_of, trip


def memories(names, trip):
    cache = {}

    def get(k):
        if k not in cache:
            cache[k] = H.BatchMemory(trip[k], names)
        return cache[k]
    return get


def ask_qwen(question, name, relations):
    options = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(relations))
    body = {"messages": [{"role": "user", "content": H.PROMPT.format(q=question, name=name,
                                                                     options=options)}],
            "temperature": 0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(QWEN, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    digits = "".join(ch for ch in text if ch.isdigit())
    k = int(digits) - 1 if digits else -1
    return relations[k] if 0 <= k < len(relations) else None


def stage(which):
    names, qs, batch_of, trip = load()
    mem = memories(names, trip)
    store = json.loads(OUT_ANS.read_text()) if OUT_ANS.exists() else {}
    for n, q in enumerate(qs):
        bm = mem(batch_of[q["s"]])
        row = store.setdefault(q["id"], {})
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        row["linked"], row["k"] = linked, len(rels)
        one = rels[0] if len(rels) == 1 else None
        if which == "embed":
            row["E"] = one or (SL.shortlist(q["q"], rels, 1)[0] if rels else None)
            row["shortlist"] = SL.shortlist(q["q"], rels, K) if rels else []
        elif which == "gemma":
            nm = MP.readable(names, linked) if linked else None
            short = row["shortlist"]
            row["S"] = one or (MP.ask(q["q"], nm, short,
                                      [MP.readable(names, bm.query(linked, r)) for r in short])[0]
                               if len(short) > 1 else (short[0] if short else None))
            row["G"] = one or (MP.ask(q["q"], nm, rels,
                                      [MP.readable(names, bm.query(linked, r)) for r in rels])[0]
                               if rels else None)
        elif which == "qwen":
            row["Q"] = one or (ask_qwen(q["q"], MP.readable(names, linked), rels) if rels else None)
        if n % 50 == 0:
            print(which, n, flush=True)
    OUT_ANS.write_text(json.dumps(store, indent=1))
    print("->", OUT_ANS)


def analyze():
    names, qs, batch_of, trip = load()
    mem = memories(names, trip)
    store = json.loads(OUT_ANS.read_text())
    arms = ("E", "S", "G", "Q")
    ok = {a: [] for a in arms}
    rel = {a: [] for a in arms}
    mids = []
    for q in qs:
        bm = mem(batch_of[q["s"]])
        row = store[q["id"]]
        for a in arms:
            r = row.get(a)
            rel[a].append(row["linked"] == q["s"] and r == q["r"])
            ok[a].append(bool(r) and bm.query(row["linked"], r) in set(q["answers"]))
        mids.append(batch_of[q["s"]])
    ok = {a: np.array(v, float) for a, v in ok.items()}
    rel = {a: np.array(v, float) for a, v in rel.items()}
    pairs = {"H1:E-G": ("E", "G"), "H2:S-G": ("S", "G"), "H3:E-Q": ("E", "Q"), "H4:E-S": ("E", "S")}
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mids))
    groups = {m: [i for i in range(len(qs)) if mids[i] == m] for m in ids}
    boot = {k: [] for k in pairs}
    for _ in range(BOOT):
        idx = np.array([i for m in rng.choice(ids, len(ids)) for i in groups[m]])
        for k, (x, y) in pairs.items():
            boot[k].append(ok[x][idx].mean() - ok[y][idx].mean())
    res = {"preregistration": "docs/preregistration/webqsp.md", "dim": DIM, "n": len(qs),
           "memories": len(ids),
           "linked_ok": float(np.mean([store[q["id"]]["linked"] == q["s"] for q in qs])),
           "shortlist_recall": float(np.mean([q["r"] in store[q["id"]]["shortlist"] for q in qs
                                              if store[q["id"]]["linked"] == q["s"]])),
           "answer_accuracy": {a: float(v.mean()) for a, v in ok.items()},
           "relation_accuracy": {a: float(v.mean()) for a, v in rel.items()},
           "diff": {k: {"point": float(ok[x].mean() - ok[y].mean()),
                        "ci95": [float(v) for v in np.percentile(boot[k], [2.5, 97.5])]}
                    for k, (x, y) in pairs.items()}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("embed", "gemma", "qwen", "analyze"), required=True)
    s = ap.parse_args().stage
    analyze() if s == "analyze" else stage(s)


if __name__ == "__main__":
    main()
