"""shortlist_prereg.py — meno opzioni al front-end piccolo: una shortlist per somiglianza.

Preregistrazione: docs/preregistration/shortlist.md. Committato insieme a quel
file e PRIMA di misurare qualsiasi cosa con la shortlist.

Diagnosi (dai dati della preregistrazione 15): con 10 o più relazioni candidate
Gemma sceglie quella giusta nel 20% dei casi, con 2–9 nel 64%. Qui un modello di
embedding piccolo (bge-small-en-v1.5, via llama-server --embedding) ordina le
relazioni candidate per somiglianza con la domanda e ne tiene k; Gemma sceglie
fra quelle con il prompt della preregistrazione 15 (anteprima della risposta della
memoria). k si sceglie sulle 100 domande di audit, con una regola fissata qui;
il confronto si fa sulle 643 di test, contro le scelte della preregistrazione 15.

Uso, dalla root:
  llama-server -m data/models/bge-small-en-v1.5-f16.gguf --port 8767 --embedding --pooling cls -c 512
  python examples/shortlist_prereg.py --stage choose_k
  llama-server -m <gemma.gguf> --port 8765 -c 8192 --parallel 1
  python examples/shortlist_prereg.py --stage small
  python examples/shortlist_prereg.py --stage analyze
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import human_questions_prereg as H  # noqa: E402
import memory_preview_prereg as MP  # noqa: E402

RES = ROOT / "results"
BASELINE = RES / "memory_preview_small_answers.json"
OUT_K = RES / "shortlist_k.json"
OUT_SMALL = RES / "shortlist_small_answers.json"
OUT = RES / "shortlist_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
EMBED = "http://127.0.0.1:8767/v1/embeddings"
EMBED_MODEL = ("bge-small-en-v1.5-f16.gguf",
               "f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_GRID = (3, 5, 8, 12)
RECALL_TARGET = 0.90
BOOT = 10000
BOOT_SEED = 20261007
# -----------------------------------------------------------------------------
H.DIM = DIM
_cache = {}


def embed(texts):
    todo = [t for t in texts if t not in _cache]
    for j in range(0, len(todo), 64):
        body = {"input": todo[j:j + 64]}
        req = urllib.request.Request(EMBED, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for t, d in zip(todo[j:j + 64], json.load(urllib.request.urlopen(req, timeout=300))["data"]):
            v = np.array(d["embedding"])
            _cache[t] = v / np.linalg.norm(v)
    return np.stack([_cache[t] for t in texts])


def rel_text(r):
    """/people/person/place_of_birth -> 'people person place of birth'."""
    return " ".join(r.replace("/", " ").replace("_", " ").split())


def shortlist(question, rels, k):
    """Le k relazioni più simili alla domanda, restituite in ordine alfabetico."""
    if len(rels) <= k:
        return list(rels)
    q = embed([QUERY_PREFIX + question])[0]
    sims = embed([rel_text(r) for r in rels]) @ q
    top = sorted(range(len(rels)), key=lambda i: (-sims[i], rels[i]))[:k]
    return sorted(rels[i] for i in top)


def split():
    H.check_data()
    by_s, names, qs = H.load()
    batch_of, trip = H.batches(qs, by_s)
    order = list(range(len(qs)))
    np.random.RandomState(H.SHUFFLE_SEED + 1).shuffle(order)
    return names, qs, batch_of, trip, order[:H.AUDIT], order[H.AUDIT:]


def choose_k():
    """Sull'audit: il k più piccolo della griglia con richiamo ≥ RECALL_TARGET, fra le
    domande con entità collegata giusta e almeno 2 relazioni candidate."""
    names, qs, batch_of, trip, audit, _test = split()
    memories, cases = {}, []
    for i in audit:
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        if linked == q["s"] and len(rels) >= 2:
            cases.append((q, rels))
    recall = {kk: float(np.mean([q["r"] in shortlist(q["q"], rels, kk) for q, rels in cases]))
              for kk in K_GRID}
    chosen = next((kk for kk in K_GRID if recall[kk] >= RECALL_TARGET), K_GRID[-1])
    OUT_K.write_text(json.dumps({"cases": len(cases), "recall": recall, "k": chosen}, indent=1))
    print(json.dumps({"cases": len(cases), "recall": recall, "k": chosen}))


def small():
    k_sel = json.loads(OUT_K.read_text())["k"]
    names, qs, batch_of, trip, _audit, test = split()
    rows, memories = [], {}
    for n, i in enumerate(test):
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        short = shortlist(q["q"], rels, k_sel) if rels else []
        if not short:
            chosen, raw = None, None
        elif len(short) == 1:
            chosen, raw = short[0], None
        else:
            previews = [MP.readable(names, bm.query(linked, r)) for r in short]
            chosen, raw = MP.ask(q["q"], MP.readable(names, linked), short, previews)
        top1 = shortlist(q["q"], rels, 1)[0] if rels else None
        rows.append({"i": i, "linked": linked, "k": len(rels), "shortlist": short,
                     "chosen": chosen, "raw": raw, "embed_top1": top1})
        if n % 50 == 0:
            print("small", n, flush=True)
    OUT_SMALL.write_text(json.dumps({"k": k_sel, "rows": rows}, indent=1))
    print("->", OUT_SMALL)


def analyze():
    names, qs, batch_of, trip, _audit, test = split()
    base = {r["i"]: r for r in json.loads(BASELINE.read_text())["rows"]}
    new = {r["i"]: r for r in json.loads(OUT_SMALL.read_text())["rows"]}
    memories = {}
    acc = {a: {"rel": [], "e2e": []} for a in ("base", "short", "embed_top1")}
    mid, in_short = [], []
    for i in test:
        q = qs[i]
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        gold = bm.objects[(q["s"], q["r"])]
        for arm, linked, chosen in (("base", base[i]["linked"], base[i]["chosen"]),
                                    ("short", new[i]["linked"], new[i]["chosen"]),
                                    ("embed_top1", new[i]["linked"], new[i]["embed_top1"])):
            acc[arm]["rel"].append(linked == q["s"] and chosen == q["r"])
            acc[arm]["e2e"].append(bool(chosen) and bm.query(linked, chosen) in gold)
        if new[i]["linked"] == q["s"] and new[i]["k"] >= 2:
            in_short.append(q["r"] in new[i]["shortlist"])
        mid.append(k)
    acc = {a: {m: np.array(v, float) for m, v in d.items()} for a, d in acc.items()}
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mid))
    groups = {m: [j for j in range(len(test)) if mid[j] == m] for m in ids}
    boots = {"rel": [], "e2e": []}
    for _ in range(BOOT):
        idx = np.array([j for m in rng.choice(ids, len(ids)) for j in groups[m]])
        for m in boots:
            boots[m].append(acc["short"][m][idx].mean() - acc["base"][m][idx].mean())
    res = {"preregistration": "docs/preregistration/shortlist.md", "dim": DIM, "n": len(test),
           "k": json.loads(OUT_K.read_text())["k"],
           "shortlist_recall_test": float(np.mean(in_short)),
           "accuracy": {a: {m: float(v.mean()) for m, v in d.items()} for a, d in acc.items()},
           "diff": {m: {"point": float(acc["short"][m].mean() - acc["base"][m].mean()),
                        "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])]}
                    for m, b in boots.items()}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("choose_k", "small", "analyze"), required=True)
    s = ap.parse_args().stage
    {"choose_k": choose_k, "small": small, "analyze": analyze}[s]()


if __name__ == "__main__":
    main()
