"""bit_errors_prereg.py — ABM contro store esatti (con e senza codice correttore) sotto bit flip.

Preregistrazione: docs/preregistration/bit_errors.md. Committato insieme a quel file e
PRIMA di misurare sui dati reali; prima del commit è stato eseguito solo con `--smoke`
(grafo sintetico generato qui) e `--predict` (previsioni analitiche: legge i dati,
costruisce le strutture dello store per sapere quante chiavi tiene, NON costruisce
nessuna memoria ABM sui dati reali e non applica rumore).

Domanda: con errori di bit durante la conservazione (bit flip iid con probabilità ε
sulla traccia memorizzata), esiste un ε in cui ABM batte, a pari bit totali D, il
migliore store esatto realizzabile? Disegno ostile ad ABM.

Per ogni (dataset, D, N, seed): N fatti distinti campionati uniformemente dal grafo;
domande = chiavi (s, r) distinte del campione (al più Q_MAX); giusta se la risposta è
uno degli oggetti veri di (s, r) nel campione.

ABM: traccia di D bit (reference), rumore: maschera Bernoulli(ε) XOR sulla traccia.
     Codebook (item memory) senza rumore: si rigenera dai nomi.
Store: funzione statica a chiavi implicite (XOR filter a 3 segmenti, peeling),
     celle di le = ⌈log₂ #oggetti⌉ bit, ⌈1.23·c⌉ celle (arrotondate a multiplo di 3)
     per c chiavi tenute; la lettura è lo XOR di 3 celle. Le celle, serializzate in
     bit, passano per un codice C ∈ CODES e il risultato (≤ D bit) subisce lo stesso
     rumore. C = "none" è l'avversario (a); gli altri sono (b).
Regola dell'avversario (fissata ora): per ogni (dataset, D, N, seed, ε) si sceglie il
     codice C (e quindi c, il massimo che entra in D bit) che massimizza l'accuratezza
     PREVISTA (analitica esatta per none/ripetizione; per Hamming, Monte Carlo sul solo
     store con STORE_MC rumori di un flusso separato, perché il limite analitico è lasco);
     si misura quello (e sempre anche "none").

Uso, dalla root:
    python examples/bit_errors_prereg.py --smoke     # sintetico piccolo + verifica Monte Carlo
    python examples/bit_errors_prereg.py --predict   # solo previsioni (nessuna misura)
    python examples/bit_errors_prereg.py             # esecuzione completa
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from math import ceil, comb, log2
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import abm  # noqa: E402
import exact  # noqa: E402

EXT = ROOT / "data" / "external"
FILES = {
    "fb15k237": ("fb15k237_train.txt",
                 "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"),
    "wn18rr": ("wn18rr_train.csv",
               "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"),
}
OUT = ROOT / "results" / "bit_errors_prereg_results.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [50, 100, 200, 400], 8192: [200, 400, 800, 1600]}
EPS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.2]
DATASETS = ("fb15k237", "wn18rr")
SEEDS = 5
Q_MAX = 300
RET_OVERHEAD = 1.23           # celle per chiave dello XOR filter (come nel test 19)
BUILD_TRIES = 20              # semi di hash provati prima di scendere a c - 1
CODES = (["none"] + [f"rep{r}" for r in (3, 5, 7, 9, 11, 15, 21)]
         + [f"ham{m}" for m in (3, 4, 5, 6, 7, 8)])   # ham_m = Hamming esteso (2^m, 2^m-m-1)
STORE_MC = 20                 # rumori simulati per prevedere gli store con Hamming
BURST_EPS = [1e-2, 3e-2]      # descrittivo: errori a raffica, stessa frequenza media
BURST_LEN = 16
SEED_BASE = 20_000_003
# -----------------------------------------------------------------------------


def load(name):
    fn, sha = FILES[name]
    path = EXT / fn
    if not path.exists():
        raise SystemExit(f"{path} mancante: scaricalo come in examples/replicate.py (DATASETS)")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"{fn}: sha256 {got}, atteso {sha}")
    if name == "wn18rr":
        rows = list(csv.reader(path.open()))[1:]
        return [(f"wn{h}", r, f"wn{t}") for h, r, t in rows]
    return [tuple(line.rstrip("\n").split("\t")) for line in path.open()]


def synthetic_graph(n_ent=300, n_rel=12, n_facts=4000, seed=0):
    rng = np.random.RandomState(seed)
    fs = set()
    while len(fs) < n_facts:
        s, o = rng.randint(n_ent, size=2)
        if s != o:
            fs.add((f"e{s}", f"r{rng.randint(n_rel)}", f"e{o}"))
    return sorted(fs)


def sample_cell(triples, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(triples), size=n, replace=False)
    facts = [triples[i] for i in sorted(idx)]
    objs = defaultdict(list)
    for s, r, o in facts:
        objs[(s, r)].append(o)
    keys = sorted(objs)
    perm = rng.permutation(len(keys))
    qs = [keys[i] for i in perm[:Q_MAX]]
    return facts, objs, keys, qs


# ---- modello esatto di ABM con rumore ---------------------------------------
# Un bit della traccia concorda con il fatto con probabilità p. Il flip è indipendente
# da tutto (prob. ε): p' = p(1-ε) + (1-p)ε = ε + p(1-2ε). I codeword nulli restano
# Binomial(D, 1/2) (un bit uniforme XOR un flip indipendente resta uniforme): cambia
# solo la p di ogni candidato a segnale, poi cleanup_accuracy_mixed come senza rumore.

def p_noisy(p, eps):
    return eps + p * (1.0 - 2.0 * eps)


def abm_predict(triples, dim, queries, eps):
    objects, into, m = exact._structure(triples)
    weights = exact.fact_weights(triples)
    all_w = list(weights.values())
    cache = {}

    def p_of(vec):
        w = weights[vec]
        if w not in cache:
            others = list(all_w)
            others.remove(w)
            cache[w] = p_noisy(exact.p_agree_weighted(w, others), eps)
        return cache[w]

    self_rels = {r for s, r, o in triples if s == o}
    out = []
    for s, r in queries:
        good, bad = objects[(s, r)], into[(s, r)] - objects[(s, r)]
        cp = [p_of(exact.fact_key(s, r, o)) for o in good]
        ap = [p_of(exact.fact_key(x, r, s)) for x in bad]
        if r in self_rels and s not in good and s not in bad:
            ap.append(p_of(("__self__", r)))
        out.append(exact.cleanup_accuracy_mixed(dim, m, cp, ap))
    return out


# ---- store: XOR filter + codice --------------------------------------------

def code_params(code):
    """(bit di dati per blocco k, bit codificati per blocco n)."""
    if code == "none":
        return 1, 1
    if code.startswith("rep"):
        return 1, int(code[3:])
    m = int(code[3:])
    return 2 ** m - m - 1, 2 ** m


def n_cells(c):
    return 3 * max(1, ceil(ceil(RET_OVERHEAD * c) / 3))


def coded_bits(c, le, code):
    k, n = code_params(code)
    return ceil(n_cells(c) * le / k) * n


def c_max(dim, le, code, n_keys):
    c = n_keys
    while c > 0 and coded_bits(c, le, code) > dim:
        c -= 1
    return c


def _h3(key, seed, seg):
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).digest()
    return [j * seg + int.from_bytes(h[8 * j:8 * j + 8], "little") % seg for j in range(3)]


def xor_build(keys, values, seed0):
    """XOR filter a 3 segmenti per peeling. None se nessuno dei BUILD_TRIES semi riesce."""
    c = len(keys)
    m = n_cells(c)
    seg = m // 3
    for t in range(BUILD_TRIES):
        seed = seed0 * 1000 + t
        hs = [_h3(k, seed, seg) for k in keys]
        count = np.zeros(m, int)
        xork = np.zeros(m, int)
        for i, h in enumerate(hs):
            for x in h:
                count[x] += 1
                xork[x] ^= i
        stack, queue = [], [x for x in range(m) if count[x] == 1]
        while queue:
            x = queue.pop()
            if count[x] != 1:
                continue
            i = xork[x]
            stack.append((i, x))
            for y in hs[i]:
                count[y] -= 1
                xork[y] ^= i
                if count[y] == 1:
                    queue.append(y)
        if len(stack) < c:
            continue
        cells = np.zeros(m, np.int64)
        for i, x in reversed(stack):
            a, b, d = hs[i]
            cells[x] = values[i] ^ cells[a] ^ cells[b] ^ cells[d] ^ cells[x]
        return cells, hs, seed
    return None


def build_store(keys, objs, obj_index, le, dim, code, rng_perm):
    """Il massimo c ≤ c_max per cui il peeling riesce; chiavi tenute = prefisso di rng_perm."""
    c = c_max(dim, le, code, len(keys))
    while c > 0:
        kept = [keys[i] for i in rng_perm[:c]]
        vals = [obj_index[objs[k][0]] for k in kept]
        b = xor_build(kept, vals, c)
        if b is not None:
            cells, hs, seed = b
            return {"code": code, "c": c, "kept": kept, "cells": cells,
                    "pos": dict(zip(kept, hs)), "seed": seed, "le": le}
        c -= 1
    return {"code": code, "c": 0, "kept": [], "cells": np.zeros(0, np.int64), "pos": {},
            "seed": None, "le": le}


def _odd3(e):
    """P(numero dispari di errori fra 3 bit indipendenti con errore e)."""
    return (1.0 - (1.0 - 2.0 * e) ** 3) / 2.0


def rep_residual(r, e):
    return sum(comb(r, j) * e ** j * (1 - e) ** (r - j) for j in range(r // 2 + 1, r + 1))


def store_key_prob(st, key, eps):
    """P(lettura esatta della chiave). Esatta per none/rep; per Hamming è il limite
    inferiore P(ogni blocco toccato ha ≤ 1 errore)."""
    code, le = st["code"], st["le"]
    if code == "none" or code.startswith("rep"):
        e = eps if code == "none" else rep_residual(int(code[3:]), eps)
        return (1.0 - _odd3(e)) ** le
    k, n = code_params(code)
    blocks = {(x * le + j) // k for x in st["pos"][key] for j in range(le)}
    pb = (1 - eps) ** n + n * eps * (1 - eps) ** (n - 1)
    return pb ** len(blocks)


def store_predict(st, queries, eps):
    return [store_key_prob(st, q, eps) if q in st["pos"] else 0.0 for q in queries]


# -- serializzazione, codici, rumore, decodifica (solo per la misura) --

def _to_bits(cells, le):
    return ((cells[:, None] >> np.arange(le)[None, :]) & 1).astype(np.uint8).ravel()


def _from_bits(bits, le):
    return (bits.reshape(-1, le).astype(np.int64) << np.arange(le)[None, :]).sum(1)


def _ham_layout(m):
    n = 2 ** m
    data_pos = np.array([i for i in range(1, n) if i & (i - 1)])       # non potenze di 2
    return n, data_pos


def encode(bits, code):
    if code == "none":
        return bits.copy()
    if code.startswith("rep"):
        return np.tile(bits, int(code[3:]))            # copie interlacciate a distanza L
    m = int(code[3:])
    n, dp = _ham_layout(m)
    k = len(dp)
    nb = ceil(len(bits) / k)
    data = np.zeros(nb * k, np.uint8)
    data[:len(bits)] = bits
    data = data.reshape(nb, k)
    cw = np.zeros((nb, n), np.uint8)
    cw[:, dp] = data
    for j in range(m):
        p = 1 << j
        mask = (np.arange(n) & p) != 0
        cw[:, p] = cw[:, mask].sum(1) % 2
    cw[:, 0] = cw[:, 1:].sum(1) % 2
    return cw.ravel()


def decode(coded, code, n_data):
    if code == "none":
        return coded[:n_data]
    if code.startswith("rep"):
        r = int(code[3:])
        return (coded.reshape(r, -1).sum(0) > r // 2).astype(np.uint8)[:n_data]
    m = int(code[3:])
    n, dp = _ham_layout(m)
    cw = coded.reshape(-1, n).copy()
    idx = np.arange(n)
    syn = np.zeros(len(cw), int)
    for j in range(m):
        syn |= (cw[:, (idx >> j) & 1 == 1].sum(1).astype(int) % 2) << j
    par = cw.sum(1) % 2
    fix = (par == 1) & (syn != 0)                       # errore singolo: correggi
    rows = np.nonzero(fix)[0]
    cw[rows, syn[rows]] ^= 1
    # par == 0 e syn != 0: doppio errore rilevato, dati lasciati come sono
    return cw[:, dp].ravel()[:n_data]


def noise_mask(n, eps, rng, burst=0):
    if eps <= 0:
        return np.zeros(n, np.uint8)
    if not burst:
        return (rng.rand(n) < eps).astype(np.uint8)
    starts = np.nonzero(rng.rand(n) < eps / burst)[0]
    m = np.zeros(n, np.uint8)
    for s in starts:
        m[s:s + burst] = 1
    return m


def store_measure(st, objs, obj_names, queries, dim, eps, rng, burst=0):
    if st["c"] == 0:
        return np.zeros(len(queries))
    le = st["le"]
    bits = _to_bits(st["cells"], le)
    coded = encode(bits, st["code"])
    assert len(coded) <= dim
    coded ^= noise_mask(len(coded), eps, rng, burst)
    cells = _from_bits(decode(coded, st["code"], len(bits)), le)
    out = []
    for q in queries:
        if q not in st["pos"]:
            out.append(0.0)     # chiave non tenuta: risposta arbitraria, contata sbagliata
            continue
        a, b, d = st["pos"][q]
        v = int(cells[a] ^ cells[b] ^ cells[d])
        out.append(float(v < len(obj_names) and obj_names[v] in objs[q]))
    return np.array(out)


# ---- ABM: misura ------------------------------------------------------------

def abm_build(facts, dim):
    mem = abm.Memory(dim)
    hv = [mem.fact_hv(*f) for f in facts]
    mem._facts = hv
    mem._trace = abm.bundle(hv)
    return mem


def abm_measure(mem, objs, queries, eps, rng, burst=0, check=False):
    names = list(mem.items._names)
    C = np.stack(mem.items._states).astype(np.float32)          # M × D, ±1
    keys = np.stack([mem._read_key(s, r) for s, r in queries]).astype(np.int8)
    flip = noise_mask(mem.dim, eps, rng, burst).astype(bool)
    trace = np.where(flip, -mem._trace, mem._trace).astype(np.int8)
    noisy = (keys * trace[None, :]).astype(np.float32)
    best = np.argmax(noisy @ C.T, axis=1)          # primo a distanza minima, come cleanup
    if check:
        saved = mem._trace
        mem._trace = trace
        ref = mem.query(*queries[0])[0]
        mem._trace = saved
        assert ref == names[best[0]], "cleanup vettoriale diversa dalla reference"
    return np.array([float(names[b] in objs[q]) for b, q in zip(best, queries)])


# ---- una cella --------------------------------------------------------------

def shannon_store(dim, le, eps, n_keys):
    """Descrittivo, NON realizzabile: store con un codice ideale a capacità 1-h(ε)."""
    h = 0.0 if eps <= 0 else -eps * log2(eps) - (1 - eps) * log2(1 - eps)
    c = int(dim * (1 - h) // (RET_OVERHEAD * le))
    return min(1.0, c / n_keys)


def run_cell(triples, dim, n, seed, do_measure, smoke_mc=False):
    facts, objs, keys, qs = sample_cell(triples, n, seed)
    obj_names = sorted({o for _, _, o in facts})
    obj_index = {o: i for i, o in enumerate(obj_names)}
    le = max(1, ceil(log2(len(obj_names))))
    perm = np.random.RandomState(seed + 1).permutation(len(keys))
    stores = {code: build_store(keys, objs, obj_index, le, dim, code, perm) for code in CODES}
    row = {"N": n, "seed": seed, "K": len(keys), "Q": len(qs), "le": le,
           "codebook": len({x for t in facts for x in t}),
           "c": {k: v["c"] for k, v in stores.items()}, "eps": {}}
    mem = abm_build(facts, dim) if do_measure else None
    for ei, eps in enumerate(EPS):
        pa = float(np.mean(abm_predict(facts, dim, qs, eps)))
        sp = {code: float(np.mean(store_predict(st, qs, eps))) for code, st in stores.items()}
        sp_bound = dict(sp)
        mc_rng = np.random.RandomState((seed * 7919 + ei + 10 ** 6) % 2 ** 32)   # flusso separato dalla misura
        for code in CODES:
            if code.startswith("ham") and eps > 0:
                sp[code] = float(np.mean([np.mean(store_measure(
                    stores[code], objs, obj_names, qs, dim, eps, mc_rng)) for _ in range(STORE_MC)]))
        best = max(CODES, key=lambda c: (sp[c], -CODES.index(c)))   # pari: il primo in CODES
        e = {"abm_pred": pa, "store_pred": sp, "best_code": best, "best_pred": sp[best],
             "none_pred": sp["none"], "ham_bound": {c: v for c, v in sp_bound.items()
                                                 if c.startswith("ham")}, "shannon_store": shannon_store(dim, le, eps, len(keys))}
        if do_measure:
            rng = np.random.RandomState((seed * 101 + ei) % 2 ** 32)
            e["abm_meas"] = abm_measure(mem, objs, qs, eps, rng, check=(ei == 0)).tolist()
            e["best_meas"] = store_measure(stores[best], objs, obj_names, qs, dim, eps, rng).tolist()
            e["none_meas"] = store_measure(stores["none"], objs, obj_names, qs, dim, eps, rng).tolist()
            if eps in BURST_EPS:
                e["abm_burst"] = abm_measure(mem, objs, qs, eps, rng, burst=BURST_LEN).tolist()
                e["best_burst"] = store_measure(stores[best], objs, obj_names, qs, dim, eps, rng,
                                                burst=BURST_LEN).tolist()
            if smoke_mc:      # Monte Carlo del limite di Hamming e dello store per ogni codice
                e["store_mc"] = {c: float(np.mean([np.mean(store_measure(
                    stores[c], objs, obj_names, qs, dim, eps, rng)) for _ in range(5)]))
                    for c in CODES}
        row["eps"][str(eps)] = e
    return row


# ---- riepilogo e verdetto ----------------------------------------------------

def se_cell(per_seed, pooled):
    """SE per cella: il massimo fra binomiale sul totale e SD fra semi / √S."""
    q = len(pooled)
    p = float(np.mean(pooled))
    se_b = (max(p * (1 - p), 1.0 / q) / q) ** 0.5
    se_s = float(np.std(per_seed, ddof=1) / len(per_seed) ** 0.5) if len(per_seed) > 1 else 0.0
    return max(se_b, se_s)


def summarize(rows, measured):
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["D"], r["N"])].append(r)
    cells = []
    for (ds, dim, n), rs in sorted(by.items()):
        q = sum(r["Q"] for r in rs)
        for eps in EPS:
            es = [r["eps"][str(eps)] for r in rs]
            w = np.array([r["Q"] for r in rs], float)
            c = {"dataset": ds, "D": dim, "N": n, "eps": eps, "Q": q,
                 "abm_pred": float(np.average([e["abm_pred"] for e in es], weights=w)),
                 "best_pred": float(np.average([e["best_pred"] for e in es], weights=w)),
                 "none_pred": float(np.average([e["none_pred"] for e in es], weights=w)),
                 "shannon_store": float(np.mean([e["shannon_store"] for e in es])),
                 "codes": sorted({e["best_code"] for e in es})}
            pa, pb = c["abm_pred"], c["best_pred"]
            se_pred = ((pa * (1 - pa) + pb * (1 - pb)) / q) ** 0.5
            c["mde_80"] = 2.84 * max(se_pred, (2.0 / q) ** 0.5 / 2)   # bias minimo rilevabile
            if measured:
                for k in ("abm", "best", "none"):
                    pooled = sum((e[k + "_meas"] for e in es), [])
                    c[k + "_meas"] = float(np.mean(pooled))
                    c[k + "_se"] = se_cell([np.mean(e[k + "_meas"]) for e in es], pooled)
                c["adv"] = c["abm_meas"] - c["best_meas"]
                c["se_adv"] = (c["abm_se"] ** 2 + c["best_se"] ** 2) ** 0.5
                c["abm_wins_all_seeds"] = all(np.mean(e["abm_meas"]) > np.mean(e["best_meas"])
                                              for e in es)
                if "abm_burst" in es[0]:
                    c["abm_burst"] = float(np.mean(sum((e["abm_burst"] for e in es), [])))
                    c["best_burst"] = float(np.mean(sum((e["best_burst"] for e in es), [])))
            cells.append(c)
    return cells


def eps_star(series, key_a, key_b):
    """Primo ε della griglia in cui a > b; None se mai."""
    for c in series:
        if c[key_a] > c[key_b]:
            return c["eps"]
    return None


def print_table(cells, measured):
    print("\ndataset    D     N   eps     | ABM prev  avv prev (codice)  none prev  Shannon  MDE")
    for c in cells:
        line = (f"{c['dataset']:9s} {c['D']:5d} {c['N']:5d} {c['eps']:<7g} | {c['abm_pred']:.3f}"
                f"     {c['best_pred']:.3f} ({','.join(c['codes'])})  {c['none_pred']:.3f}"
                f"      {c['shannon_store']:.3f}    {c['mde_80']:.3f}")
        if measured:
            line += (f" || mis. ABM {c['abm_meas']:.3f} avv {c['best_meas']:.3f}"
                     f" none {c['none_meas']:.3f} vant {c['adv']:+.3f}±{c['se_adv']:.3f}")
        print(line)
    series = defaultdict(list)
    for c in cells:
        series[(c["dataset"], c["D"], c["N"])].append(c)
    print("\nε* previsto (primo ε con ABM previsto > miglior avversario previsto):")
    for k, s in series.items():
        line = f"  {k}: {eps_star(s, 'abm_pred', 'best_pred')}"
        if measured:
            line += f"   osservato: {eps_star(s, 'abm_meas', 'best_meas')}"
        print(line)


def verdict(cells):
    v = {}
    wins = [c for c in cells if c["adv"] > 2 * c["se_adv"]]
    v["H1_cells_adv_gt_2se"] = len(wins)
    v["H1_cells_adv_gt_2se_and_all_seeds"] = sum(c["abm_wins_all_seeds"] for c in wins)
    v["H1_cells_adv_gt_3.5se_bonferroni"] = sum(c["adv"] > 3.5 * c["se_adv"] for c in cells)
    g = defaultdict(list)
    for c in cells:
        g[f"{c['dataset']}/{c['D']}"].append(abs(c["abm_meas"] - c["abm_pred"]))
    v["H2_mae_points"] = {k: round(100 * float(np.mean(x)), 2) for k, x in g.items()}
    v["H2_max_cell_points"] = round(100 * max(abs(c["abm_meas"] - c["abm_pred"]) for c in cells), 2)
    series = defaultdict(list)
    for c in cells:
        series[(c["dataset"], c["D"], c["N"])].append(c)
    ok = 0
    for s in series.values():
        a, b = eps_star(s, "abm_pred", "best_pred"), eps_star(s, "abm_meas", "best_meas")
        ia = EPS.index(a) if a is not None else len(EPS)
        ib = EPS.index(b) if b is not None else len(EPS)
        ok += abs(ia - ib) <= 1
    v["H3_series_eps_star_within_one_step"] = f"{ok}/{len(series)}"
    g = defaultdict(list)
    for c in cells:
        g[f"{c['dataset']}/{c['D']}"].append(abs(c["best_meas"] - c["best_pred"]))
    v["H4_store_mae_points"] = {k: round(100 * float(np.mean(x)), 2) for k, x in g.items()}
    return v


def run(graphs, grid, seeds, do_measure, out_path=None, smoke_mc=False):
    rows = []
    t0 = time.time()
    for name, triples in graphs.items():
        for dim, ns in grid.items():
            for n in ns:
                for s in range(seeds):
                    seed = SEED_BASE + 1009 * s + 7 * n + dim + (13 if name == "wn18rr" else 0)
                    row = {"dataset": name, "D": dim, **run_cell(triples, dim, n, seed,
                                                                 do_measure, smoke_mc)}
                    rows.append(row)
                    print(f"{name} D={dim} N={n} seed={s} K={row['K']} le={row['le']} "
                          f"c(none)={row['c']['none']} [{time.time() - t0:.0f}s]", flush=True)
    cells = summarize(rows, do_measure)
    print_table(cells, do_measure)
    out = {"cells": cells, "rows": rows if do_measure else None}
    if do_measure:
        out["verdict"] = verdict(cells)
        print(json.dumps(out["verdict"], indent=1))
    if out_path:
        out_path.write_text(json.dumps(out, indent=1))
        print("->", out_path)
    return rows, cells


def smoke():
    """Sintetico minuscolo: gira tutto, e verifica col Monte Carlo la formula p' e i limiti."""
    g = synthetic_graph()
    rows, cells = run({"synthetic": g}, {1024: [40, 120]}, 3, True, smoke_mc=True)
    print("\nMonte Carlo, ABM: |misura - previsione| per cella (3 semi):")
    for c in cells:
        print(f"  N={c['N']} eps={c['eps']:<7g} prev {c['abm_pred']:.3f} mis {c['abm_meas']:.3f}"
              f" ±{c['abm_se']:.3f}")
    print("\nMonte Carlo, store: previsione analitica vs simulazione (5 rumori × semi):")
    for r in rows:
        for eps in (1e-2, 0.1):
            e = r["eps"][str(eps)]
            print(f"  N={r['N']} eps={eps}: " + "  ".join(
                f"{c}:{e['store_pred'][c]:.3f}/{e['store_mc'][c]:.3f}" for c in
                ("none", "rep3", "rep7", "ham4", "ham6")))
    # verifica diretta della formula p' = ε + p(1-2ε) su bit
    rng = np.random.RandomState(0)
    for p, eps in ((0.6, 0.1), (0.55, 0.2)):
        a = rng.rand(10 ** 6) < p
        f = rng.rand(10 ** 6) < eps
        print(f"  p'={p_noisy(p, eps):.4f}  MC={np.mean(a ^ f):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
        return
    graphs = {d: [t for t in load(d) if t[0] != t[2]] for d in DATASETS}
    if a.predict:
        run(graphs, GRID, SEEDS, False)
    else:
        run(graphs, GRID, SEEDS, True, OUT)


if __name__ == "__main__":
    main()
