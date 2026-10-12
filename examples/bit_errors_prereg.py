"""bit_errors_prereg.py — ABM contro store esatti con codici correttori, sotto bit flip.

Preregistrazione: docs/preregistration/bit_errors.md (revisione del disegno prima dei
dati). Committato PRIMA di misurare sui dati reali; prima del commit è stato eseguito
solo con `--smoke` (grafo sintetico generato qui) e `--predict` (legge i dati reali,
costruisce SOLO gli store e ne simula il rumore per scegliere il codice; NON costruisce
nessuna memoria ABM sui dati reali).

Domanda: con errori di bit durante la conservazione (bit flip iid con probabilità ε su
tutto lo stato che dipende dai fatti), esiste un ε in cui ABM batte, a pari bit totali
D, il migliore store esatto realizzabile scelto a priori? Disegno ostile ad ABM.

Per ogni (dataset, D, N, seed): N fatti distinti campionati uniformemente; domande =
chiavi (s, r) distinte del campione (al più Q_MAX); giusta se la risposta è uno degli
oggetti veri di (s, r) nel campione.

ABM: traccia di D bit (reference); rumore = maschera Bernoulli(ε) sulla traccia.
Store = layout × codice, tutto nei D bit rumorosi:
  layout  XOR  : funzione statica (XOR filter a 3 segmenti), lettura = XOR di 3 celle;
          MPHF : hash-and-displace (CHD semplificato, spostamenti a larghezza fissa)
                 + array di valori; la MPHF sta nella memoria rumorosa e codificata;
                 lettura = 1 spostamento + 1 cella.
  codice  none | rep (r/r+2 misti, usa tutti i bit) | BCH (n ∈ 63…1023, accorciato) |
          polare SC (N_b ∈ 256…2048, costruzione di Bhattacharyya per BSC(ε),
          accorciamento al tasso esatto) | rep3 interna + BCH/polare esterno (ε ≥ 0.03).
  chiavi  c ∈ FRACS × c_max(none): "meno chiavi, codice più forte" è ammesso.
Regola (fissata ora): per (cella, seed, ε) si sceglie layout × codice × c con la massima
accuratezza PREVISTA (none/rep: esatta; BCH: P(Bin(n, ε) ≤ t) per blocco; polare:
Monte Carlo del solo store su un flusso RNG separato). Dopo la scelta, best_pred è
ristimata con un terzo flusso indipendente (REEST_R rumori, simulazione completa).

Uso, dalla root:
    python examples/bit_errors_prereg.py --smoke     # sintetico piccolo + controlli A4
    python examples/bit_errors_prereg.py --predict   # solo previsioni (nessuna misura ABM)
    python examples/bit_errors_prereg.py             # esecuzione completa
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from functools import lru_cache  # noqa: E402
from math import ceil, comb, erfc, log2, sqrt  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

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
PRED_OUT = ROOT / "results" / "bit_errors_prereg_predictions.json"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
GRID = {2048: [50, 100, 200, 400], 8192: [200, 400, 800, 1600]}
EPS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.2]
DATASETS = ("fb15k237", "wn18rr")
SEEDS = 5
Q_MAX = 300
R_NOISE = 20                  # rumori indipendenti per memoria e per ε (misura)
REEST_R = 50                  # A6: rumori del terzo flusso per ristimare best_pred
POLAR_BANK = 1024             # blocchi simulati per la previsione dei polari
POLAR_SIMS = 64               # store simulati (dai blocchi) per la previsione dei polari
XOR_OVERHEAD = 1.23           # celle per chiave dello XOR filter (come nel test 19)
MPHF_LAMBDA = 4               # chiavi per bucket della MPHF
BUILD_TRIES = 20
FRACS = (1.0, 0.8, 0.64, 0.5, 0.4, 0.3, 0.2, 0.12)
BCH_N = (63, 127, 255, 511, 1023)
POLAR_N = (256, 512, 1024, 2048)
CONCAT_MIN_EPS = 0.03
BURST_EPS = [1e-2, 3e-2]      # descrittivo: raffiche, stessa frequenza media di flip
BURST_LEN = 16
CB_NOISE_R = 5                # descrittivo B2: rumori sul codebook ABM
ALPHA = 0.05                  # Holm, one-sided, famiglia confermativa
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


def streams(seed):
    """B7: un sotto-flusso RNG indipendente per sistema e per scopo."""
    names = ("abm", "store_meas", "store_select", "store_reest", "none_meas",
             "burst", "abm_cb")
    kids = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(k) for n, k in zip(names, kids)}


# ---- modello esatto di ABM con rumore ---------------------------------------
# p' = p(1-ε) + (1-p)ε = ε + p(1-2ε) per ogni candidato a segnale; i nulli restano
# Binomial(D, 1/2). Approssimazione (A4): il flip è comune a tutti i candidati di una
# query; il modello tratta le distanze come indipendenti, come già senza rumore.

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


# ---- layout: XOR filter e MPHF ----------------------------------------------

def _hash_ints(key, seed, k):
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).digest()
    return [int.from_bytes(h[8 * j:8 * j + 8], "little") for j in range(k)]


def xor_cells(c):
    return 3 * max(1, ceil(ceil(XOR_OVERHEAD * c) / 3))


def build_xor(kept, vals, le, seed0):
    c = len(kept)
    m = xor_cells(c)
    seg = m // 3
    for t in range(BUILD_TRIES):
        seed = seed0 * 1000 + t
        hs = [[j * seg + x % seg for j, x in enumerate(_hash_ints(k, seed, 3))] for k in kept]
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
            cells[x] = vals[i] ^ cells[a] ^ cells[b] ^ cells[d]
        return {"layout": "xor", "c": c, "le": le, "seed": seed, "m": m,
                "pos": dict(zip(kept, hs)), "bits": _to_bits(cells, le)}
    return None


def _mphf_pos(a, b, d, m):
    return (a + d * b + d * d) % m


def build_mphf(kept, vals, le, seed0):
    c = len(kept)
    nb = max(1, ceil(c / MPHF_LAMBDA))
    for t in range(BUILD_TRIES):
        seed = seed0 * 1000 + 500 + t
        hv = [_hash_ints(k, seed, 3) for k in kept]
        a = np.array([x[0] % c for x in hv], np.int64)
        b = np.array([x[1] % c for x in hv], np.int64)
        g = np.array([x[2] % nb for x in hv], np.int64)
        buckets = defaultdict(list)
        for i, gi in enumerate(g):
            buckets[gi].append(i)
        used = np.zeros(c, bool)
        disp = np.zeros(nb, np.int64)
        ok = True
        for gi in sorted(buckets, key=lambda x: (-len(buckets[x]), x)):
            idx = np.array(buckets[gi])
            found = None
            for start in range(0, 1 << 16, 1024):
                d = np.arange(start, start + 1024, dtype=np.int64)
                pos = _mphf_pos(a[idx][:, None], b[idx][:, None], d[None, :], c)  # k × T
                free = ~used[pos].any(0)
                if len(idx) > 1:
                    s = np.sort(pos, axis=0)
                    free &= (np.diff(s, axis=0) != 0).all(0)
                hit = np.nonzero(free)[0]
                if len(hit):
                    found = int(d[hit[0]])
                    break
            if found is None:
                ok = False
                break
            disp[gi] = found
            used[_mphf_pos(a[idx], b[idx], found, c)] = True
        if not ok:
            continue
        w = max(1, ceil(log2(int(disp.max()) + 1)))
        table = np.zeros(c, np.int64)
        slot = {}
        for i, k in enumerate(kept):
            p = int(_mphf_pos(a[i], b[i], disp[g[i]], c))
            table[p] = vals[i]
            slot[k] = (int(g[i]), p, int(a[i]), int(b[i]))
        bits = np.concatenate([_to_bits(disp, w), _to_bits(table, le)])
        return {"layout": "mphf", "c": c, "le": le, "seed": seed, "nb": nb, "w": w,
                "slot": slot, "bits": bits}
    return None


def _to_bits(vals, width):
    vals = np.asarray(vals, np.int64)
    return ((vals[:, None] >> np.arange(width)[None, :]) & 1).astype(np.uint8).ravel()


def _from_bits(bits, width):
    return (bits.reshape(-1, width).astype(np.int64) << np.arange(width)[None, :]).sum(1)


def deps(st, key):
    """Bit di dati da cui dipende la lettura di una chiave tenuta."""
    le = st["le"]
    if st["layout"] == "xor":
        return [x * le + j for x in st["pos"][key] for j in range(le)]
    g, p, _, _ = st["slot"][key]
    w = st["w"]
    off = st["nb"] * w
    return list(range(g * w, g * w + w)) + list(range(off + p * le, off + p * le + le))


def read(st, bits, key, n_obj, fallback):
    """Lettura vera su bit (eventualmente decodificati con errori). (indice, valido)."""
    le = st["le"]
    if st["layout"] == "xor":
        h = _hash_ints(key, st["seed"], 3)
        seg = st["m"] // 3
        cells = [j * seg + x % seg for j, x in enumerate(h)]
        v = 0
        for x in cells:
            v ^= int(_from_bits(bits[x * le:(x + 1) * le], le)[0])
    else:
        c, nb, w = st["c"], st["nb"], st["w"]
        hv = _hash_ints(key, st["seed"], 3)
        a, b, g = hv[0] % c, hv[1] % c, hv[2] % nb
        d = int(_from_bits(bits[g * w:(g + 1) * w], w)[0])
        p = int(_mphf_pos(a, b, d, c))
        off = nb * w
        v = int(_from_bits(bits[off + p * le:off + (p + 1) * le], le)[0])
    return (v, True) if v < n_obj else (fallback, False)


# ---- codici -----------------------------------------------------------------

def rep_residual(r, e):
    return sum(comb(r, j) * e ** j * (1 - e) ** (r - j) for j in range(r // 2 + 1, r + 1))


@lru_cache(maxsize=None)
def bch_k(n, t):
    """Dimensione del BCH binario primitivo (n, k) che corregge t errori (cosets ciclotomici)."""
    roots = set()
    for i in range(1, 2 * t + 1):
        x = i % n
        while x not in roots:
            roots.add(x)
            x = (2 * x) % n
    return n - len(roots)


@lru_cache(maxsize=None)
def bch_ts(n):
    out, t = [], 1
    while True:
        k = bch_k(n, t)
        if k <= 1:
            return out
        if not out or k < out[-1][1]:
            out.append((t, k))
        t += 1


def code_fit(kind, L, B, eps):
    """Il codice più forte del tipo dato che porta L bit di dati in ≤ B bit. None se impossibile."""
    if L <= 0:
        return None
    if kind == "none":
        return {"kind": "none", "B": L} if L <= B else None
    if kind == "rep":
        r = B // L
        r = r if r % 2 else r - 1
        if r < 3:
            return None
        x = min(L, (B - r * L) // 2)
        return {"kind": "rep", "r": r, "x": x, "B": r * L + 2 * x}
    if kind.startswith("bch"):
        n = int(kind[3:])
        best = None
        for t, k in bch_ts(n):
            b = ceil(L / k)
            if L + b * (n - k) <= B:
                best = {"kind": kind, "n": n, "t": t, "k": k, "blocks": b, "kb": ceil(L / b),
                        "B": b * (ceil(L / b) + n - k)}
        return best
    if kind.startswith("polar"):
        nb_ = int(kind[5:])
        b = ceil(B / nb_)
        nt = B // b
        K = ceil(L / b)
        if nt > nb_ or K > nt - 1:
            return None
        return {"kind": kind, "N": nb_, "blocks": b, "nt": nt, "K": K, "B": b * nt}
    raise ValueError(kind)


def code_list(L, B, eps):
    out = []
    kinds = ["none"] if eps <= 0 else (["none", "rep"] + [f"bch{n}" for n in BCH_N]
                                        + [f"polar{n}" for n in POLAR_N])
    for k in kinds:
        f = code_fit(k, L, B, eps)
        if f:
            out.append(f)
    if eps >= CONCAT_MIN_EPS:
        for k in [f"bch{n}" for n in BCH_N] + [f"polar{n}" for n in POLAR_N]:
            f = code_fit(k, L, B // 3, eps)
            if f:
                out.append({"kind": "concat", "outer": f, "B": 3 * f["B"]})
    return out


def code_name(cd):
    if cd["kind"] == "concat":
        return "rep3+" + code_name(cd["outer"])
    if cd["kind"] == "rep":
        return f"rep{cd['r']}/{cd['r'] + 2}"
    if cd["kind"].startswith("bch"):
        return f"bch({cd['n']},{cd['k']},t={cd['t']})"
    if cd["kind"].startswith("polar"):
        return f"polar(N={cd['N']},K={cd['K']},n={cd['nt']})"
    return "none"


# -- polari (SC) --

@lru_cache(maxsize=None)
def polar_info(N, nt, K, eps):
    z = np.full(N, 2 * sqrt(eps * (1 - eps)))
    z[nt:] = 0.0                                  # accorciati: noti a zero

    def rec(z):
        if len(z) == 1:
            return z
        h = len(z) // 2
        zl, zr = z[:h], z[h:]
        return np.concatenate([rec(zl + zr - zl * zr), rec(zl * zr)])
    zu = rec(z)
    zu[nt:] = np.inf                              # u accorciati: congelati (x_i = 0, i ≥ nt)
    info = np.sort(np.argsort(zu, kind="stable")[:K])
    frozen = np.ones(N, bool)
    frozen[info] = False
    return info, frozen


def polar_encode(u):
    n = u.shape[1]
    if n == 1:
        return u
    h = n // 2
    v1, v2 = polar_encode(u[:, :h]), polar_encode(u[:, h:])
    return np.concatenate([v1 ^ v2, v2], axis=1)


def polar_sc(L, frozen):
    """Decodifica SC (min-sum), batch sulle righe. Restituisce (u, x) stimati."""
    n = L.shape[1]
    if frozen.all():
        z = np.zeros(L.shape, np.uint8)
        return z, z
    if not frozen.any():
        x = (L < 0).astype(np.uint8)
        return polar_encode(x), x                  # G è un'involuzione
    if n == 1:
        u = (L < 0).astype(np.uint8)
        return u, u
    h = n // 2
    La, Lb = L[:, :h], L[:, h:]
    L1 = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    u1, v1 = polar_sc(L1, frozen[:h])
    L2 = Lb + (1 - 2 * v1.astype(np.float64)) * La
    u2, v2 = polar_sc(L2, frozen[h:])
    return np.concatenate([u1, u2], 1), np.concatenate([v1 ^ v2, v2], 1)


def _llr(rx, eps, N, nt):
    ell = np.log((1 - eps) / eps) if 0 < eps < 0.5 else 30.0
    L = np.full((rx.shape[0], N), 1e9)
    L[:, :nt] = ell * (1 - 2 * rx.astype(np.float64))
    # i pareggi (L = 0, frequenti su un BSC) vanno rotti a caso, non verso 0: senza questo
    # la parola nulla del banco risulterebbe ottimista. Rumore minuscolo, fisso, simmetrico.
    L[:, :nt] += 1e-6 * ell * np.random.default_rng(nt).standard_normal(L[:, :nt].shape)
    return L


@lru_cache(maxsize=4096)
def polar_bank(N, nt, K, eps, seed):
    """Errori sui K bit d'informazione di POLAR_BANK blocchi (parola nulla: canale simmetrico)."""
    info, frozen = polar_info(N, nt, K, eps)
    rng = np.random.default_rng(seed)
    rx = (rng.random((POLAR_BANK, nt)) < eps).astype(np.uint8)
    u, _ = polar_sc(_llr(rx, eps, N, nt), frozen)
    return u[:, info]


# -- encode / decode reali (misura) --

def encode(bits, cd):
    k = cd["kind"]
    if k == "none":
        return bits.copy()
    if k == "rep":
        return np.concatenate([np.tile(bits, cd["r"]), np.tile(bits[:cd["x"]], 2)])
    if k == "concat":
        return np.tile(encode(bits, cd["outer"]), 3)
    if k.startswith("bch"):
        code = _bch(cd["n"], cd["t"])
        import galois
        b, kb = cd["blocks"], cd["kb"]
        msg = np.zeros(b * kb, np.uint8)
        msg[:len(bits)] = bits
        msg = msg.reshape(b, kb)
        full = np.zeros((b, code.k), np.uint8)
        full[:, code.k - kb:] = msg                    # accorciamento: zeri in testa
        cw = np.array(code.encode(galois.GF2(full)))[:, code.k - kb:]
        return cw.ravel()
    if k.startswith("polar"):
        N, b, nt, K = cd["N"], cd["blocks"], cd["nt"], cd["K"]
        info, _ = polar_info(N, nt, K, cd["_eps"])
        msg = np.zeros(b * K, np.uint8)
        msg[:len(bits)] = bits
        u = np.zeros((b, N), np.uint8)
        u[:, info] = msg.reshape(b, K)
        return polar_encode(u)[:, :nt].ravel()
    raise ValueError(k)


def decode(rx, cd, L, eps):
    k = cd["kind"]
    if k == "none":
        return rx[:L]
    if k == "rep":
        r, x = cd["r"], cd["x"]
        votes = rx[:r * L].reshape(r, L).sum(0).astype(int)
        cnt = np.full(L, r)
        if x:
            votes[:x] += rx[r * L:].reshape(2, x).sum(0)
            cnt[:x] += 2
        return (2 * votes > cnt).astype(np.uint8)
    if k == "concat":
        inner = (rx.reshape(3, -1).sum(0) >= 2).astype(np.uint8)
        e3 = rep_residual(3, eps)
        return decode(inner, cd["outer"], L, e3)
    if k.startswith("bch"):
        code = _bch(cd["n"], cd["t"])
        import galois
        b, kb = cd["blocks"], cd["kb"]
        nb_ = kb + cd["n"] - cd["k"]
        r = rx.reshape(b, nb_)
        sent = cd["_sent"].reshape(b, nb_)
        nerr = (r != sent).sum(1)
        out = r[:, :kb].copy()                          # sistematico: messaggio in testa
        bad = np.nonzero(nerr > cd["t"])[0]            # ≤ t errori: corretti per costruzione
        good = np.nonzero(nerr <= cd["t"])[0]
        out[good] = sent[good, :kb]
        if len(bad):
            pad = np.zeros((len(bad), code.n - nb_), np.uint8)
            dec = np.array(code.decode(galois.GF2(np.concatenate([pad, r[bad]], 1))))
            out[bad] = dec[:, code.k - kb:]
        return out.ravel()[:L]
    if k.startswith("polar"):
        N, b, nt, K = cd["N"], cd["blocks"], cd["nt"], cd["K"]
        info, frozen = polar_info(N, nt, K, cd["_eps"])
        u, _ = polar_sc(_llr(rx.reshape(b, nt), eps, N, nt), frozen)
        return u[:, info].ravel()[:L]
    raise ValueError(k)


@lru_cache(maxsize=None)
def _bch(n, t):
    import galois
    return galois.BCH(n, d=2 * t + 1)


def block_of_coded(cd, total):
    """Indice di blocco di ogni bit codificato, per l'interlacciatore a blocchi (B6)."""
    k = cd["kind"]
    if k.startswith("bch"):
        nb_ = cd["kb"] + cd["n"] - cd["k"]
        return np.repeat(np.arange(cd["blocks"]), nb_), np.tile(np.arange(nb_), cd["blocks"])
    if k.startswith("polar"):
        return np.repeat(np.arange(cd["blocks"]), cd["nt"]), np.tile(np.arange(cd["nt"]),
                                                                       cd["blocks"])
    if k == "concat":
        bo, off = block_of_coded(cd["outer"], total // 3)
        return np.tile(bo, 3), np.tile(off, 3)
    return np.zeros(total, int), np.arange(total)


def interleaver(cd, total):
    blk, off = block_of_coded(cd, total)
    return np.lexsort((blk, off))                     # ordina per (posizione nel blocco, blocco)


def noise(n, eps, rng, burst=0):
    if eps <= 0:
        return np.zeros(n, np.uint8)
    if not burst:
        return (rng.random(n) < eps).astype(np.uint8)
    m = np.zeros(n + burst, np.uint8)
    for s in np.nonzero(rng.random(n) < eps / burst)[0]:
        m[s:s + burst] = 1
    return m[:n]


# ---- previsione dello store -------------------------------------------------

def bit_residuals(cd, L, eps):
    k = cd["kind"]
    if k == "none":
        return np.full(L, eps)
    if k == "rep":
        e = np.full(L, rep_residual(cd["r"], eps))
        e[:cd["x"]] = rep_residual(cd["r"] + 2, eps)
        return e
    return None


def _binom_cdf(n, p, t):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(t + 1))


def store_predict(st, cd, queries, eps, rng):
    """Accuratezza prevista della configurazione (layout, codice, c) sulle domande."""
    L = len(st["bits"])
    kept = st["pos"] if st["layout"] == "xor" else st["slot"]
    qk = [q for q in queries if q in kept]
    if not qk:
        return 0.0
    dp = np.array([deps(st, q) for q in qk])            # Qk × bit
    ch_eps = rep_residual(3, eps) if cd["kind"] == "concat" else eps
    core = cd["outer"] if cd["kind"] == "concat" else cd
    e = bit_residuals(core, L, ch_eps)
    le = st["le"]
    if e is not None:
        if st["layout"] == "xor":
            f = (1 - 2 * e[dp]).reshape(len(qk), 3, le).prod(1)
            return float(np.prod((1 + f) / 2, axis=1).sum() / len(queries))
        return float(np.prod(1 - e[dp], axis=1).sum() / len(queries))
    if core["kind"].startswith("bch"):
        nb_ = core["kb"] + core["n"] - core["k"]
        pok = _binom_cdf(nb_, ch_eps, core["t"])
        nblk = np.array([len(set((row // core["kb"]).tolist())) for row in dp])
        return float((pok ** nblk).sum() / len(queries))
    # polare: Monte Carlo del solo store, blocchi pescati da un banco simulato
    bank = polar_bank(core["N"], core["nt"], core["K"], ch_eps, int(rng.integers(1 << 31)))
    pick = rng.integers(len(bank), size=(POLAR_SIMS, core["blocks"]))
    err = bank[pick].reshape(POLAR_SIMS, -1)[:, :L]       # SIMS × L
    x = err[:, dp]                                         # SIMS × Qk × bit
    if st["layout"] == "xor":
        ok = ~(x.reshape(POLAR_SIMS, len(qk), 3, le).sum(2) % 2).any(2)
    else:
        ok = ~x.any(2)
    return float(ok.sum() / POLAR_SIMS / len(queries))


def store_simulate(st, cd, queries, objs, obj_names, fallback, eps, rng, R, burst=0,
                   interleave=False):
    """Simulazione completa: codifica, rumore, decodifica, lettura vera. (acc, acc selettiva)."""
    bits = st["bits"]
    L = len(bits)
    sent = encode(bits, cd)
    core = cd["outer"] if cd["kind"] == "concat" else cd
    if core["kind"].startswith("bch"):
        core["_sent"] = encode(bits, core)
    perm = interleaver(cd, len(sent)) if interleave else None
    acc, sel = [], []
    for _ in range(R):
        nz = noise(len(sent), eps, rng, burst)
        if perm is not None:
            tmp = np.zeros_like(nz)
            tmp[perm] = nz                    # la raffica colpisce posizioni fisiche consecutive
            nz = tmp
        dec = decode(sent ^ nz, cd, L, eps)
        ok, ans = 0, 0
        for q in queries:
            v, valid = read(st, dec, q, len(obj_names), fallback)
            hit = obj_names[v] in objs[q]
            ok += hit
            if valid:
                ans += 1
                sel.append(hit)
        acc.append(ok / len(queries))
    return float(np.mean(acc)), (float(np.mean(sel)) if sel else float("nan"))


def ppv_store(dim, le, eps, n_keys):
    """B1, descrittivo: tetto con un codice alla PPV (approssimazione normale) e overhead 1.0."""
    if eps <= 0:
        return min(1.0, (dim // le) / n_keys)
    h = -eps * log2(eps) - (1 - eps) * log2(1 - eps)
    C = 1 - h
    V = eps * (1 - eps) * log2((1 - eps) / eps) ** 2
    best = 0.0
    for c in range(1, n_keys + 1):
        k = c * le
        z = (dim * C - k + 0.5 * log2(dim)) / sqrt(dim * V)
        pe = 0.5 * erfc(z / sqrt(2))
        best = max(best, c / n_keys * (1 - pe))
    return best


# ---- ABM: misura -------------------------------------------------------------

def abm_build(facts, dim):
    mem = abm.Memory(dim)
    hv = [mem.fact_hv(*f) for f in facts]
    mem._facts = hv
    mem._trace = abm.bundle(hv)
    return mem


def abm_measure(mem, objs, queries, eps, rng, R, burst=0, cb_noise=False, check=False):
    names = list(mem.items._names)
    C0 = np.stack(mem.items._states).astype(np.float32)
    keys = np.stack([mem._read_key(s, r) for s, r in queries]).astype(np.float32)
    acc = []
    for i in range(R):
        flip = noise(mem.dim, eps, rng, burst).astype(bool)
        trace = np.where(flip, -mem._trace, mem._trace).astype(np.float32)
        C = C0
        if cb_noise:
            C = np.where(rng.random(C0.shape) < eps, -C0, C0).astype(np.float32)
        best = np.argmax((keys * trace[None, :]) @ C.T, axis=1)
        if check and i == 0 and not cb_noise:
            saved = mem._trace
            mem._trace = trace.astype(np.int8)
            ref = mem.query(*queries[0])[0]
            mem._trace = saved
            assert ref == names[best[0]], "cleanup vettoriale diversa dalla reference"
        acc.append(np.mean([names[b] in objs[q] for b, q in zip(best, queries)]))
    return float(np.mean(acc))


# ---- una memoria (dataset, D, N, seed) --------------------------------------

def run_cell(args):
    name, dim, n, seed, do_measure = args
    triples = GRAPHS[name]
    facts, objs, keys, qs = sample_cell(triples, n, seed)
    obj_names = sorted({o for _, _, o in facts})
    obj_index = {o: i for i, o in enumerate(obj_names)}
    fallback = obj_index[Counter(o for _, _, o in facts).most_common(1)[0][0]]
    le = max(1, ceil(log2(len(obj_names))))
    perm = np.random.RandomState(seed + 1).permutation(len(keys))
    rs = streams(seed)

    def build(layout, c):
        kept = [keys[i] for i in perm[:c]]
        vals = [obj_index[objs[k][0]] for k in kept]
        while c > 0:
            st = (build_xor if layout == "xor" else build_mphf)(kept[:c], vals[:c], le, c)
            if st is not None:
                return st
            c -= 1
        return None

    stores = []
    for layout in ("xor", "mphf"):
        c = min(len(keys), dim // le)       # c_max(none): il più grande c i cui bit entrano in D
        st = None
        while c > 0:
            st = build(layout, c)
            if st is not None and len(st["bits"]) <= dim:
                break
            over = len(st["bits"]) - dim if st is not None else le
            c = min(c - 1, st["c"] - ceil(over / (le + 2)) if st is not None else c - 1)
            st = None
        if st is None:
            continue
        cmax, seen = st["c"], set()
        for f in FRACS:
            cf = max(1, int(round(f * cmax)))
            s2 = st if cf == cmax else build(layout, cf)
            if s2 is not None and len(s2["bits"]) <= dim and s2["c"] not in seen:
                seen.add(s2["c"])
                stores.append(s2)
    row = {"dataset": name, "D": dim, "N": n, "seed": seed, "K": len(keys), "Q": len(qs),
           "le": le, "codebook": len({x for t in facts for x in t}), "eps": {}}
    mem = abm_build(facts, dim) if do_measure else None
    xor_full = max((s for s in stores if s["layout"] == "xor"), key=lambda s: s["c"])
    for eps in EPS:
        pa = float(np.mean(abm_predict(facts, dim, qs, eps)))
        best, best_cfg = -1.0, None
        for st in stores:
            for cd in code_list(len(st["bits"]), dim, eps):
                _tag(cd, eps)
                p = store_predict(st, cd, qs, eps, rs["store_select"])
                if p > best + 1e-12:
                    best, best_cfg = p, (st, cd)
        st, cd = best_cfg
        e = {"abm_pred": pa, "best_pred": best, "choice": f"{st['layout']}/c={st['c']}/"
             f"{code_name(cd)}", "layout": st["layout"], "code": code_name(cd).split("(")[0],
             "c_kept": st["c"], "ppv_store": ppv_store(dim, le, eps, len(keys))}
        e["best_reest"], _ = store_simulate(st, cd, qs, objs, obj_names, fallback, eps,
                                            rs["store_reest"], REEST_R)
        if do_measure:
            e["abm_meas"] = abm_measure(mem, objs, qs, eps, rs["abm"], R_NOISE,
                                        check=(eps == 0))
            e["best_meas"], e["best_selective"] = store_simulate(
                st, cd, qs, objs, obj_names, fallback, eps, rs["store_meas"], R_NOISE)
            e["none_meas"], _ = store_simulate(xor_full, {"kind": "none", "B": 0}, qs, objs,
                                               obj_names, fallback, eps, rs["none_meas"], R_NOISE)
            if eps > 0:
                e["abm_cbnoise"] = abm_measure(mem, objs, qs, eps, rs["abm_cb"], CB_NOISE_R,
                                               cb_noise=True)
                e["name_factor"] = float(np.mean([(1 - eps) ** (8 * len(o)) for o in obj_names]))
            if eps in BURST_EPS:
                e["abm_burst"] = abm_measure(mem, objs, qs, eps, rs["burst"], R_NOISE,
                                             burst=BURST_LEN)
                e["best_burst"], _ = store_simulate(st, cd, qs, objs, obj_names, fallback, eps,
                                                    rs["burst"], R_NOISE, burst=BURST_LEN,
                                                    interleave=True)
        row["eps"][str(eps)] = e
    return row


def _tag(cd, eps):
    cd["_eps"] = eps if cd["kind"] != "concat" else None
    if cd["kind"] == "concat":
        cd["outer"]["_eps"] = rep_residual(3, eps)


# ---- statistica --------------------------------------------------------------

def t4_sf(t):
    """P(T > t), Student con 4 gradi di libertà (forma chiusa)."""
    u = t / sqrt(4 + t * t)
    return 0.5 - 0.75 * u * (1 - u * u / 3)


def t4_q(p):
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t4_sf(mid) > 1 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def family(cells):
    """A5: celle previste a favore di ABM con margine > MDE(α/m Holm, potenza 80%), iterato."""
    cand = [c for c in cells if c["d_pred"] > 0]
    while True:
        m = max(1, len(cand))
        q = t4_q(1 - ALPHA / m) + t4_q(0.8)
        new = [c for c in cand if c["d_pred"] > q * c["sd_pred"] / sqrt(SEEDS)]
        for c in cells:
            c["mde"] = q * c["sd_pred"] / sqrt(SEEDS)
        if len(new) == len(cand):
            return new
        cand = new


def summarize(rows, measured):
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["D"], r["N"])].append(r)
    cells = []
    for (ds, dim, n), rs in sorted(by.items()):
        for eps in EPS:
            es = [r["eps"][str(eps)] for r in rs]
            qs = [r["Q"] for r in rs]
            pa = np.array([e["abm_pred"] for e in es])
            pb = np.array([e["best_pred"] for e in es])
            d = pa - pb
            within = np.mean([(a * (1 - a) + b * (1 - b)) / q for a, b, q in zip(pa, pb, qs)])
            c = {"dataset": ds, "D": dim, "N": n, "eps": eps, "Q": int(sum(qs)),
                 "abm_pred": float(pa.mean()), "best_pred": float(pb.mean()),
                 "best_reest": float(np.mean([e["best_reest"] for e in es])),
                 "d_pred": float(d.mean()),
                 "sd_pred": float(sqrt(np.var(d, ddof=1) + within)),
                 "ppv_store": float(np.mean([e["ppv_store"] for e in es])),
                 "choices": sorted({e["choice"] for e in es}),
                 "codes": sorted({e["layout"] + "/" + e["code"] for e in es})}
            if measured:
                ma = np.array([e["abm_meas"] for e in es])
                mb = np.array([e["best_meas"] for e in es])
                dd = ma - mb
                c.update(abm_meas=float(ma.mean()), best_meas=float(mb.mean()),
                         none_meas=float(np.mean([e["none_meas"] for e in es])),
                         d_meas=float(dd.mean()), se_d=float(dd.std(ddof=1) / sqrt(len(dd))),
                         best_selective=float(np.nanmean([e["best_selective"] for e in es])))
                c["t"] = c["d_meas"] / c["se_d"] if c["se_d"] > 0 else float("inf") * np.sign(
                    c["d_meas"])
                c["p_one_sided"] = t4_sf(c["t"]) if np.isfinite(c["t"]) else float(c["t"] < 0)
                for k in ("abm_burst", "best_burst", "abm_cbnoise", "name_factor"):
                    if k in es[0]:
                        c[k] = float(np.mean([e[k] for e in es]))
            cells.append(c)
    fam = family(cells)
    for c in cells:
        c["confirmatory"] = c in fam
    return cells


def eps_star(series, key):
    for c in series:
        if c[key] > 0:
            return c["eps"]
    return None


def eligible(series):
    """B4: ε* si valuta solo dove la differenza prevista al cambio di segno supera l'MDE
    (o, per 'mai', dove è sotto −MDE a ogni ε)."""
    s = eps_star(series, "d_pred")
    if s is None:
        return all(c["d_pred"] < -c["mde"] for c in series)
    c = next(c for c in series if c["eps"] == s)
    return c["d_pred"] > c["mde"]


def print_table(cells, measured):
    print("\ndataset    D     N   eps    | ABM prev  avv prev  avv ristima  PPV    MDE   conf | scelta")
    for c in cells:
        line = (f"{c['dataset']:9s} {c['D']:5d} {c['N']:5d} {c['eps']:<6g} | {c['abm_pred']:.3f}"
                f"     {c['best_pred']:.3f}     {c['best_reest']:.3f}     {c['ppv_store']:.3f}"
                f"  {c['mde']:.3f}  {'*' if c['confirmatory'] else ' '}   | "
                f"{','.join(c['codes'])}")
        if measured:
            line += (f" || mis. ABM {c['abm_meas']:.3f} avv {c['best_meas']:.3f}"
                     f" d {c['d_meas']:+.3f}±{c['se_d']:.3f}")
        print(line)
    series = defaultdict(list)
    for c in cells:
        series[(c["dataset"], c["D"], c["N"])].append(c)
    print("\nε* previsto (primo ε con ABM previsto > avversario previsto), valutabile (B4):")
    for k, s in series.items():
        line = f"  {k}: {eps_star(s, 'd_pred')}  valutabile={eligible(s)}"
        if measured:
            line += f"   osservato: {eps_star(s, 'd_meas')}"
        print(line)
    fam = [c for c in cells if c["confirmatory"]]
    print(f"\nfamiglia confermativa: {len(fam)} celle")
    for c in fam:
        print(f"  {c['dataset']} D={c['D']} N={c['N']} ε={c['eps']}: d prev {c['d_pred']:+.3f}"
              f"  MDE {c['mde']:.3f}")


def verdict(cells):
    v = {}
    fam = sorted([c for c in cells if c["confirmatory"]], key=lambda c: c["p_one_sided"])
    m = len(fam)
    rej = 0
    for i, c in enumerate(fam):
        if c["p_one_sided"] <= ALPHA / (m - i):
            rej += 1
        else:
            break
    v["H1_family_size"] = m
    v["H1_holm_rejections"] = rej
    v["H1_rejected_cells"] = [f"{c['dataset']}/{c['D']}/{c['N']}/{c['eps']}" for c in fam[:rej]]
    # famiglia vuota (prevista: nessuna cella a favore di ABM): H1 si valuta su TUTTE le
    # celle con Bonferroni one-sided α/112; una vittoria qui falsifica la previsione
    allc = len(cells)
    v["H1_all_cells_bonferroni_wins"] = [f"{c['dataset']}/{c['D']}/{c['N']}/{c['eps']}"
                                         for c in cells if c["p_one_sided"] <= ALPHA / allc]
    v["H1_gap_to_ppv_in_rejected"] = [round(c["ppv_store"] - c["abm_meas"], 3) for c in fam[:rej]]
    g = defaultdict(list)
    for c in cells:
        g[f"{c['dataset']}/{c['D']}"].append(abs(c["abm_meas"] - c["abm_pred"]))
    v["H2_mae_points"] = {k: round(100 * float(np.mean(x)), 2) for k, x in g.items()}
    v["H2_max_cell_points"] = round(100 * max(abs(c["abm_meas"] - c["abm_pred"]) for c in cells), 2)
    v["H2_max_cell_eps02_points"] = round(100 * max(abs(c["abm_meas"] - c["abm_pred"])
                                                    for c in cells if c["eps"] == 0.2), 2)
    series = defaultdict(list)
    for c in cells:
        series[(c["dataset"], c["D"], c["N"])].append(c)
    ok = tot = 0
    for s in series.values():
        if not eligible(s):
            continue
        tot += 1
        a, b = eps_star(s, "d_pred"), eps_star(s, "d_meas")
        ia = EPS.index(a) if a is not None else len(EPS)
        ib = EPS.index(b) if b is not None else len(EPS)
        ok += abs(ia - ib) <= 1
    v["H3_eligible_series_within_one_step"] = f"{ok}/{tot}"
    g = defaultdict(list)
    for c in cells:
        g[f"{c['dataset']}/{c['D']}"].append(abs(c["best_meas"] - c["best_reest"]))
    v["H4_store_mae_points_vs_reest"] = {k: round(100 * float(np.mean(x)), 2) for k, x in g.items()}
    v["H0_eps0_series_abm_below"] = sum(c["d_meas"] < 0 for c in cells if c["eps"] == 0)
    return v


GRAPHS = {}


def run(graphs, grid, seeds, do_measure, out_path=None, workers=None):
    GRAPHS.update(graphs)
    jobs = []
    for name in graphs:
        for dim, ns in grid.items():
            for n in ns:
                for s in range(seeds):
                    seed = SEED_BASE + 1009 * s + 7 * n + dim + (13 if name == "wn18rr" else 0)
                    jobs.append((name, dim, n, seed, do_measure))
    t0 = time.time()
    rows = []
    with Pool(workers or os.cpu_count()) as pool:
        for row in pool.imap_unordered(run_cell, jobs):
            rows.append(row)
            print(f"{row['dataset']} D={row['D']} N={row['N']} seed={row['seed']} K={row['K']}"
                  f" [{time.time() - t0:.0f}s]", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["D"], r["N"], r["seed"]))
    cells = summarize(rows, do_measure)
    print_table(cells, do_measure)
    out = {"cells": cells, "rows": rows, "elapsed_s": time.time() - t0}
    if do_measure:
        out["verdict"] = verdict(cells)
        print(json.dumps(out["verdict"], indent=1))
    if out_path:
        out_path.write_text(json.dumps(out, indent=1, default=float))
        print("->", out_path)
    return rows, cells


# ---- smoke --------------------------------------------------------------------

def smoke_a4():
    """A4: su tracce ABM reali (sintetiche), concordanza per candidato e correlazione fra
    candidati, senza e con flip. Il flip è comune a tutti i candidati di una query."""
    g = synthetic_graph()
    dim, n = 1024, 60
    facts, objs, keys, qs = sample_cell(g, n, 7)
    mem = abm_build(facts, dim)
    names = list(mem.items._names)
    C = np.stack(mem.items._states).astype(np.int8)
    rng = np.random.default_rng(1)
    weights = exact.fact_weights(facts)
    all_w = list(weights.values())
    w0 = all_w[0]
    others = list(all_w)
    others.remove(w0)
    p_model = exact.p_agree_weighted(w0, others)
    print(f"\nA4 — D={dim}, N={n}, {len(qs)} domande; p del modello (peso 1) = {p_model:.4f}")
    for eps in (0.0, 0.1, 0.2):
        agree_true, d_true, d_n1, d_n2 = [], [], [], []
        for _ in range(200):
            flip = noise(dim, eps, rng).astype(bool)
            tr = np.where(flip, -mem._trace, mem._trace)
            for s, r in qs[:20]:
                noisy = mem._read_key(s, r) * tr
                o = objs[(s, r)][0]
                agree_true.append(np.mean(noisy == C[names.index(o)]))
                d = (noisy[None, :] != C).sum(1)
                d_true.append(d[names.index(o)])
                nul = [i for i in range(len(names)) if names[i] not in objs[(s, r)]][:2]
                d_n1.append(d[nul[0]])
                d_n2.append(d[nul[1]])
        a = np.mean(agree_true)
        # correlazioni ENTRO la domanda (al variare del solo rumore): si toglie la media per
        # domanda, altrimenti si misura la struttura fra domande, non l'effetto del flip
        def within(x):
            x = np.array(x, float).reshape(200, -1)
            return (x - x.mean(0)).ravel()
        if eps > 0:
            ct = np.corrcoef(within(d_true), within(d_n1))[0, 1]
            cn = np.corrcoef(within(d_n1), within(d_n2))[0, 1]
            extra = f" | entro domanda: corr(vero, nullo) {ct:+.3f}  corr(nullo, nullo) {cn:+.3f}"
        else:
            extra = " | senza flip le distanze sono fisse per domanda"
        print(f"  ε={eps}: concordanza del vero misurata {a:.4f}, prevista p'={p_noisy(p_model, eps):.4f}"
              f"{extra}  var(nullo fra domande)/(D/4) {np.var(d_n1) / (dim / 4):.3f}")


def smoke():
    smoke_a4()
    g = synthetic_graph()
    rows, cells = run({"synthetic": g}, {1024: [40, 120]}, 3, True, workers=6)
    print("\nsmoke: ABM previsto vs misurato (R=20 rumori, 3 seed):")
    for c in cells:
        print(f"  N={c['N']} ε={c['eps']:<7g} prev {c['abm_pred']:.3f} mis {c['abm_meas']:.3f}"
              f" | store prev {c['best_pred']:.3f} ristima {c['best_reest']:.3f}"
              f" mis {c['best_meas']:.3f} ({','.join(c['codes'])})")
    print(json.dumps(verdict(cells), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.smoke:
        smoke()
        return
    graphs = {d: [t for t in load(d) if t[0] != t[2]] for d in DATASETS}
    if a.predict:
        run(graphs, GRID, SEEDS, False, PRED_OUT, a.workers)
    else:
        run(graphs, GRID, SEEDS, True, OUT, a.workers)


if __name__ == "__main__":
    main()
