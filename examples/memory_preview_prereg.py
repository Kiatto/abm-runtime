"""memory_preview_prereg.py — il front-end piccolo sceglie la relazione vedendo la risposta della memoria.

Preregistrazione: docs/preregistration/memory_preview.md. Committato insieme a quel
file e PRIMA di interrogare il modello con questo prompt.

Nei test 11–14 Gemma 4 E2B sceglie la relazione vedendo solo i nomi delle relazioni
memorizzate per l'entità collegata, e sceglie quella giusta nel 41% dei casi. Qui
ogni opzione mostra anche la risposta che la memoria ABM (D = 16 384, la stessa dei
test 13–14) dà per quella relazione, con il suo nome leggibile:

    3. /people/person/place_of_birth (stored answer: Seattle)

così il modello può scegliere guardando il tipo di risposta. Il confronto è con le
scelte senza anteprima della preregistrazione 14 (results/escalation3_small_answers.json),
stesso modello, stesse domande, stessa memoria.

Uso, dalla root:
  llama-server -m <gemma.gguf> --port 8765 -c 8192 --parallel 1
  python examples/memory_preview_prereg.py --stage small
  python examples/memory_preview_prereg.py --stage analyze
  python examples/memory_preview_prereg.py --smoke
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

RES = ROOT / "results"
BASELINE = RES / "escalation3_small_answers.json"
OUT_SMALL = RES / "memory_preview_small_answers.json"
OUT = RES / "memory_preview_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
DIM = 16384
SERVER = "http://127.0.0.1:8765/v1/chat/completions"
PROMPT2 = ("Question: {q}\n"
           "Which relation of '{name}' answers the question? Each option shows the answer "
           "stored for it. Reply with the number only.\n"
           "{options}")
BOOT = 10000
BOOT_SEED = 20261006
# -----------------------------------------------------------------------------
H.DIM = DIM


def readable(names, x):
    return names.get(x, x).replace("_", " ")


def ask(question, name, relations, previews):
    options = "\n".join(f"{i + 1}. {r} (stored answer: {p})"
                        for i, (r, p) in enumerate(zip(relations, previews)))
    body = {"messages": [{"role": "user", "content": PROMPT2.format(q=question, name=name,
                                                                    options=options)}],
            "temperature": 0, "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    digits = "".join(ch for ch in text if ch.isdigit())
    k = int(digits) - 1 if digits else -1
    return (relations[k] if 0 <= k < len(relations) else None), text


def setup():
    H.check_data()
    by_s, names, qs = H.load()
    batch_of, trip = H.batches(qs, by_s)
    return names, qs, batch_of, trip


def small():
    names, qs, batch_of, trip = setup()
    rows, memories = [], {}
    for i, q in enumerate(qs):
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        linked = bm.link(q["q"])
        rels = sorted(bm.out.get(linked, [])) if linked else []
        if not rels:
            chosen, raw = None, None
        elif len(rels) == 1:
            chosen, raw = rels[0], None
        else:
            previews = [readable(names, bm.query(linked, r)) for r in rels]
            chosen, raw = ask(q["q"], readable(names, linked), rels, previews)
        rows.append({"i": i, "linked": linked, "chosen": chosen, "raw": raw})
        if i % 50 == 0:
            print("small", i, flush=True)
    OUT_SMALL.write_text(json.dumps({"prompt": PROMPT2, "rows": rows}, indent=1))
    print("->", OUT_SMALL)


def analyze():
    names, qs, batch_of, trip = setup()
    base = {r["i"]: r for r in json.loads(BASELINE.read_text())["rows"]}
    new = {r["i"]: r for r in json.loads(OUT_SMALL.read_text())["rows"]}
    memories = {}
    rel_b, rel_n, e2e_b, e2e_n, mid = [], [], [], [], []
    for i, q in enumerate(qs):
        k = batch_of[q["s"]]
        if k not in memories:
            memories[k] = H.BatchMemory(trip[k], names)
        bm = memories[k]
        gold = bm.objects[(q["s"], q["r"])]
        for row, rel, e2e in ((base[i], rel_b, e2e_b), (new[i], rel_n, e2e_n)):
            ok_rel = row["linked"] == q["s"] and row["chosen"] == q["r"]
            rel.append(ok_rel)
            e2e.append(bool(row["chosen"]) and bm.query(row["linked"], row["chosen"]) in gold)
        mid.append(k)
    rel_b, rel_n, e2e_b, e2e_n = map(lambda x: np.array(x, float), (rel_b, rel_n, e2e_b, e2e_n))
    rng = np.random.RandomState(BOOT_SEED)
    ids = sorted(set(mid))
    groups = {m: [i for i in range(len(qs)) if mid[i] == m] for m in ids}
    boots = {"relation": [], "end_to_end": []}
    for _ in range(BOOT):
        idx = np.array([i for m in rng.choice(ids, len(ids)) for i in groups[m]])
        boots["relation"].append(rel_n[idx].mean() - rel_b[idx].mean())
        boots["end_to_end"].append(e2e_n[idx].mean() - e2e_b[idx].mean())
    res = {"preregistration": "docs/preregistration/memory_preview.md", "dim": DIM,
           "n": len(qs),
           "relation": {"baseline": float(rel_b.mean()), "preview": float(rel_n.mean())},
           "end_to_end": {"baseline": float(e2e_b.mean()), "preview": float(e2e_n.mean())},
           "diff": {k: {"point": float((rel_n.mean() - rel_b.mean()) if k == "relation"
                                       else (e2e_n.mean() - e2e_b.mean())),
                        "ci95": [float(v) for v in np.percentile(b, [2.5, 97.5])]}
                    for k, b in boots.items()},
           "changed_choice": int(sum(base[i]["chosen"] != new[i]["chosen"] for i in range(len(qs)))),
           "per_question": [{"i": i, "memory": m, "rel_base": bool(a), "rel_preview": bool(b),
                             "e2e_base": bool(c), "e2e_preview": bool(d)}
                            for i, m, a, b, c, d in zip(range(len(qs)), mid, rel_b, rel_n,
                                                        e2e_b, e2e_n)]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "relation", "end_to_end", "diff",
                                         "changed_choice")}, indent=1))
    print("->", OUT)


def smoke():
    rels = ["/people/person/nationality", "/people/person/place_of_birth"]
    print(ask("where was the poet born?", "the poet", rels, ["Italy", "Florence"]))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=("small", "analyze"))
    g.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    small() if args.stage == "small" else analyze()


if __name__ == "__main__":
    main()
