"""sizing_tool2_prereg.py — preregistrazione 24: il collaudo del dimensionatore
`abm.sizing.recommend` della 23, ridisegnato SOLO per il tempo.

docs/preregistration/sizing_tool2.md. Ipotesi, soglie, unità, analisi per cluster e minimo
di 30 cluster sono quelli della 23 (`examples/sizing_tool_prereg.py`, congelato: qui lo si
IMPORTA e non lo si modifica). Cambia solo come si spende il tempo:

1. tempi misurati su sottografi REALI (`--timing-real`: solo costruzioni e query, MAI
   accuratezze — asserito), da cui la stima del costo di ogni cella;
2. costo per raccomandazione ridotto con regole scritte (vero su 10 memorie, D minimo e
   baseline su perimetri ridotti, HRR con un limite di memoria più basso; la verifica di
   `recommend` resta col default dello strumento, 8 semi);
3. cella = (sottografo, target, famiglia), eseguita in ordine di priorità per fasi;
4. tetto rigido controllato PER CELLA (SIGALRM nel worker) e checkpoint per cella con
   flush e ripresa idempotente; le celle abbandonate sono «non misurate»;
5. griglia scelta perché la stima (× 1.5) stia sotto 20 h con 10 processi; tetto 30 h.

Uso, dalla root:
    python examples/sizing_tool2_prereg.py --smoke        # impianto intero, solo sintetico
    python examples/sizing_tool2_prereg.py --predict      # previsioni (nessuna memoria)
    python examples/sizing_tool2_prereg.py --timing-real  # SOLO tempi su sottografi reali dedicati
    python examples/sizing_tool2_prereg.py --estimate     # stima per passo (dai tempi salvati)
    python examples/sizing_tool2_prereg.py --power        # potenza per cluster (previsioni)
    python examples/sizing_tool2_prereg.py                # il run preregistrato (NON eseguito qui)
"""
import os

# Thread BLAS fissati PRIMA di importare numpy (revisione): 1 thread per processo, 10
# processi. Il primo --timing-real (file _timing.json) era senza questo vincolo (OpenBLAS
# multi-thread in 10 processi): la stima usa la ripetizione a 1 thread (_timing_t1.json).
BLAS_THREADS = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = BLAS_THREADS

import argparse   # noqa: E402
import json
import math
import multiprocessing as mp
import signal
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sizing_tool_prereg as P   # noqa: E402  (la 23, congelata: solo import)

from abm import families, sizing   # noqa: E402

ROOT = P.ROOT
OUT = ROOT / "results" / "sizing_tool2_prereg_results.json"
CKPT = ROOT / "results" / "sizing_tool2_prereg_results.jsonl"
META = ROOT / "results" / "sizing_tool2_prereg_meta.json"
PRED_OUT = ROOT / "results" / "sizing_tool2_prereg_predictions.json"
TIMING_OUT0 = ROOT / "results" / "sizing_tool2_prereg_timing.json"   # primo, BLAS libero
TIMING_OUT = ROOT / "results" / "sizing_tool2_prereg_timing_t1.json"  # 1 thread: la stima
TIMING_MAX_N = 400                  # la ripetizione a 1 thread misura solo N ≤ 400 (N = 1600 fuori)
WATCHDOG_SLACK_S = 600              # il padre abbandona una cella dopo tetto + 10 min
P23_CKPT_SHA = "ba2869f7746e8ef0adb62ed347e2dbaec0d7fdfda44a1262b1bde2a84d292882"

# ---- preregistrato: non cambiare dopo aver visto i risultati ----------------
SEED_BASE = 24_000_024              # nuovo: nessun sottografo della 23 (5 già misurati) rientra
K_OFFSET = 60_000                   # sottografi di valutazione: k = 60_000 + j
K_TIMING = 90_000                   # sottografi SOLO per i tempi: k = 90_000 (uno per cella)
TRUTH_SEEDS = 10                    # 23: 20. Vero = media su 10 memorie fresche
VERIFY_SEEDS = P.VERIFY_SEEDS       # 8: default dello strumento (il prodotto), non toccato
TIMING_REPEATS = 3                  # --timing-real: massimo di 3 ripetizioni per misura
MARGIN = 1.5                        # stima = massimo di 3 ripetizioni × 1.5
RUN_BUDGET_H = 20.0                 # la stima (con margine) deve stare sotto 20 h
HARD_CAP_H = 30.0                   # tetto rigido (orologio da muro dal primo avvio)
WORKERS = 10
CELL_CAP_MIN_S = 1800               # tetto per cella: max(30 min, 10 × stima della cella)
CELL_CAP_FACTOR = 10
# Costo della verifica di recommend per raccomandazione (memorie costruite, per famiglia) e
# quota di D corretti: conteggi del checkpoint della 23 (90 raccomandazioni FB15k-237; solo
# conteggi, nessuna accuratezza), usati SOLO dalla stima del tempo.
VERIFY_MEMS = {"MAP-B": 62.3, "MAP-I": 34.7, "FHRR": 51.7, "HRR": 32.7, "BSDC-OR": 20.3,
               "BSDC-S": 22.9}
P_ADJ = {"MAP-B": 0.79, "MAP-I": 0.40, "FHRR": 0.67, "HRR": 0.25, "BSDC-OR": 0.13,
         "BSDC-S": 0.20}
# Griglia: passi in ordine; si sceglie il PRIMO la cui stima × MARGIN sta sotto RUN_BUDGET_H.
# k_in/k_un: sottografi base per cella (N ≤ 400; popolazione di H3, H5, H6);
# k_val/k_val_un: sottografi con almeno una validated prevista per cella (H1, H2, H1u, H2u);
# k_1600: sottografi base a N = 1600 per cella (fase «resto», secondari);
# dmin: (target, N max, 1 sottografo ogni `every`, in ordine di k) per H4;
# h7: (target, N max, every) per la baseline di H7; hrr: max_mem di HRR (byte);
# hrr_1600: False = HRR non si costruisce a N = 1600 (escluse, elencate).
# Seconda revisione (prima dei dati): passi in ORDINE DI PREFERENZA, regola scritta prima di
# guardare le stime: si sceglie il PRIMO la cui stima × 1.5 sta sotto 20 h con 10 processi.
# full1600: N = 1600 in tutti i ruoli come nella 23 (base k_in/k_un, val k_val/k_val_un);
# truth: memorie del vero; hrr1600_cap: max_mem di HRR a N = 1600 (None = come hrr).
_D23 = dict(k_in=4, k_val=6, k_un=8, k_val_un=30, k_1600=0, full1600=True,
            dmin=((0.85,), 400, 1), h7=((0.85,), 100, 1), hrr=256 << 20, hrr_1600=True,
            hrr1600_cap=None, truth=20)
REDUCTION = [
    dict(_D23),                                         # (a) disegno della 23 pieno
    dict(_D23, truth=10),                               # (b) come (a), vero su 10
    dict(_D23, truth=10, hrr1600_cap=64 << 20),         # (c) come (b), HRR 64 MiB a N = 1600
    dict(k_in=2, k_val=5, k_un=4, k_val_un=25, k_1600=0, full1600=False,   # (d) = passo 4
         dmin=((0.85,), 400, 3), h7=((0.85,), 100, 2), hrr=64 << 20, hrr_1600=False,
         hrr1600_cap=None, truth=10),
]
GRID_STEP = 1                       # seconda revisione: (b), 11.64 h × 1.5 = 17.45 h ≤ 20 h; (a) 25.2 h oltre
CONTROLS_FIRST = True               # controlli: ~1 h di CPU nello smoke (~6 min con 10 processi)
# -----------------------------------------------------------------------------

CFG = {}
TIMING_MODE = False                 # True in --timing-real: nessuna accuratezza, mai


def configure(step):
    """Applica il passo (run, smoke, stima, potenza: stessa via). Scrive anche la
    configurazione della 23 (P.CFG) che le sue funzioni importate leggono (k_*, hrr)."""
    CFG.clear()
    CFG.update(REDUCTION[step], step=step)
    P.CFG.clear()
    P.CFG.update(k_in=CFG["k_in"], k_val=CFG["k_val"], k_un=CFG["k_un"],
                 k_val_un=CFG["k_val_un"], hrr=CFG["hrr"], dmin=((), 0), h7=((), 0))


def hrr_off(family, n):
    return family == "HRR" and n > 400 and not CFG["hrr_1600"]


def mem_cap(family, n):
    if family == "HRR" and n > 400 and CFG.get("hrr1600_cap"):
        return CFG["hrr1600_cap"]
    return P.mem_cap(family)


def excluded(family, d, n, m, q):
    """L'esclusione della 23 (stessa stima di memoria della verifica) più HRR a N = 1600 se
    il passo la spegne."""
    return hrr_off(family, n) or families.estimate_bytes(
        family, d, n_facts=n, n_symbols=m, n_queries=q) > mem_cap(family, n)


# ---- sottografi (campionamento della 23, semi nuovi) -----------------------------------

def subgraph(name, graph, si, n, k):
    di = P.DATASETS.index(name)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, di, si, n, k]).spawn(1)[0])
    return P.make_sub(name, di, si, P.SCHEMES[si], n, k, P.sample(graph, n, P.SCHEMES[si], rng),
                      rng)


def all_subgraphs(graphs, offset=K_OFFSET, kmax=None):
    out = []
    for name, g in graphs.items():
        for si in range(len(P.SCHEMES)):
            for n in P.LOADS:
                kk = kmax or P.K_PRED["in" if name not in P.UNSEEN else
                                      "unseen_1600" if n == 1600 else "unseen"]
                if n <= len(g.triples) // 2:
                    out += [subgraph(name, g, si, n, offset + j) for j in range(kk)]
    return out


def target_seeds(sg, ti):
    """Come la 23 (verifica, vero, D minimo, baseline; insiemi disgiunti) con SEED_BASE 24."""
    ver, tru, dmn, bas = np.random.SeedSequence(
        [SEED_BASE, sg["di"], sg["si"], sg["N"], sg["k"], 1000 + ti]).spawn(4)
    b = P.seeds_of(bas, 2 * VERIFY_SEEDS, 1 << 33)
    return {"rng_seed": int(ver.generate_state(1)[0]),
            "truth": P.seeds_of(tru, CFG.get("truth", TRUTH_SEEDS), 1 << 31),
            "dmin": P.seeds_of(dmn, P.DMIN_SEEDS, 1 << 32), "base_sel": b[:VERIFY_SEEDS],
            "base_rep": b[VERIFY_SEEDS:]}


# ---- selezione e celle (dalle sole previsioni) ------------------------------------------

def _rows_by_sub(pred):
    d = defaultdict(list)
    for r in pred:
        d[r["key"]].append(r)
    return d


def _valid_cell(r):
    return r["family"] and not r["discarded"]


def _validated(r):
    return _valid_cell(r) and r["reliability"] == "validated" and not excluded(
        r["family"], r["dim_predicted"], r["N"], r["M"], r["Q"])


def select(pred):
    """Sottografi per ruolo. base (N ≤ 400): i primi k_in/k_un per cella (H3, H5, H6, H8u);
    val: base o extra con almeno una validated prevista, fino a k_val/k_val_un per cella
    (H1, H2, H1u, H2u: come la 23, extra in ordine di k); b1600: i primi k_1600 a N = 1600
    (secondari). Ogni lista è in ordine a giri (posizione nella cella), poi grafo, schema, N."""
    by = _rows_by_sub(pred)
    cells = defaultdict(list)
    for key, rs in by.items():
        r0 = rs[0]
        cells[(r0["dataset"], r0["scheme"], r0["N"])].append((r0["k"], key, rs))
    out = {"base": [], "val": [], "b1600": []}
    for (ds, sc, n), subs in cells.items():
        subs.sort()
        unseen = ds in P.UNSEEN
        order = (P.DATASETS.index(ds), P.SCHEMES.index(sc), n)
        if n > 400 and not CFG["full1600"]:   # passo (d): nessuna conclusione a N = 1600
            for pos, (_k, key, _rs) in enumerate(subs[:CFG["k_1600"]]):
                out["b1600"].append(((pos,) + order, key))
            continue
        kb = CFG["k_un"] if unseen else CFG["k_in"]
        kv = CFG["k_val_un"] if unseen else CFG["k_val"]
        nval = 0
        for pos, (_k, key, rs) in enumerate(subs):
            v = any(_validated(r) for r in rs)
            if pos < kb:
                out["base"].append(((pos,) + order, key))
            if v and nval < kv:
                out["val"].append(((nval,) + order, key))
                nval += 1
    res = {k: [key for _o, key in sorted(v)] for k, v in out.items()}
    if not CFG["full1600"]:
        assert all(by[k][0]["N"] <= 400 for k in res["base"] + res["val"])
    return res


def cells_of(pred, sel):
    """Fasi in ordine di priorità; ogni fase è una lista di celle (sottografo, target,
    famiglia, tipo). Le celle già in una fase precedente non si ripetono.
    F1: celle validated dei sottografi 'val', grafi di calibrazione e mai visti INTERCALATI
        (H1, H2, H1u, H2u);
    F2: tutte le celle dei sottografi base N ≤ 400 (H3, H5, H6, H8u);
    F3: D minimo (H4) e baseline (H7) nei perimetri, sulle celle verificate di F1-F2;
    F4: il resto: famiglie non validated dei sottografi 'val' extra, base N = 1600.
    I controlli sintetici vanno prima di F1 se CONTROLS_FIRST, altrimenti dopo F4."""
    by = _rows_by_sub(pred)

    def recs(key, only_validated=False):
        out = []
        for r in sorted(by[key], key=lambda r: (r["target"], P.FAMILIES.index(r["family"])
                                                 if r["family"] else 9)):
            if not _valid_cell(r) or (only_validated and not _validated(r)):
                continue
            out.append({"key": key, "target": r["target"], "family": r["family"],
                        "kind": "rec"})
        return out

    seen = set()

    def add(lst, cells):
        for c in cells:
            cid = (c["key"], c["target"], c["family"], c["kind"])
            if cid not in seen:
                seen.add(cid)
                lst.append(c)

    ind = [k for k in sel["val"] if by[k][0]["dataset"] in P.IN_DIST]
    uns = [k for k in sel["val"] if by[k][0]["dataset"] in P.UNSEEN]
    f1 = []
    for i in range(max(len(ind), len(uns))):
        for lst in (ind, uns):
            if i < len(lst):
                add(f1, recs(lst[i], only_validated=True))
    f2 = []
    for k in sel["base"]:
        add(f2, recs(k))
    f3 = []
    dts, dn, dev = CFG["dmin"]
    hts, hn, hev = CFG["h7"]
    ordered = list(dict.fromkeys(sel["val"] + sel["base"]))
    for kind, ts, nmax, every in (("dmin", dts, dn, dev), ("base", hts, hn, hev)):
        per_cell = Counter()
        for k in ordered:
            r0 = by[k][0]
            if r0["N"] > nmax:
                continue
            cell = (r0["dataset"], r0["scheme"], r0["N"])
            take = per_cell[cell] % every == 0
            per_cell[cell] += 1
            if not take:
                continue
            for c in recs(k):
                if c["target"] in ts:
                    add(f3, [dict(c, kind=kind)])
    f4 = []
    for k in sel["val"]:
        add(f4, recs(k))
    for k in sel["b1600"]:
        add(f4, recs(k))
    return [("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4)]


def cid(c):
    return f"{c['key']}|{c['target']}|{c['family']}|{c['kind']}"


# ---- misure di una cella ------------------------------------------------------------

class CellTimeout(BaseException):
    """BaseException: nessun `except Exception` del codice chiamato può inghiottirla."""


def _alarm(_sig, _frm):
    raise CellTimeout()


class Truth(P.Truth):
    """Il vero della 23 (memorie fresche, tutte le chiavi e Q) con l'esclusione del passo;
    vietato in modalità tempi."""

    def fits(self, family, d):
        return not excluded(family, d, self.n, self.m, len(self.sg["queries"]))

    def __call__(self, *a, **kw):
        assert not TIMING_MODE, "accuratezza in modalità tempi"
        return super().__call__(*a, **kw)


_STRUCT = {}


def _struct(sg):
    if sg["key"] not in _STRUCT:
        _STRUCT.clear()
        _STRUCT[sg["key"]] = P.struct(sg)
    return _STRUCT[sg["key"]]


def rec_row(sg, target, family):
    """La raccomandazione di una famiglia e il suo vero (stessi campi delle righe della 23,
    così si usa P.summarize). recommend(families=[f]) dà lo stesso risultato della chiamata
    congiunta: la divisione semi/query dipende solo da (grafo, rng_seed, seeds)."""
    assert not TIMING_MODE
    st = _struct(sg)
    ti = list(P.TARGETS).index(target) if target in P.TARGETS else \
        P.control_targets(sg).index(target)
    sd = target_seeds(sg, ti)
    truth = Truth(sg)
    t0 = time.time()
    if hrr_off(family, len(sg["triples"])):
        recs = sizing.recommend(sg["triples"], target, families=[family],
                                queries=sg["queries"])
        cost = (0, 0.0)
    else:
        with P._CountMake() as cm:
            recs = sizing.recommend(sg["triples"], target, families=[family],
                                    queries=sg["queries"], verify=True, seeds=VERIFY_SEEDS,
                                    tol=P.TOL, rng_seed=sd["rng_seed"],
                                    max_mem_bytes=mem_cap(family, len(sg["triples"])))
        cost = (cm.c[family], time.time() - t0)
    if not recs:
        return None, truth, sd
    r = recs[0]
    row = {k: sg[k] for k in ("dataset", "scheme", "N", "k", "key")}
    row.update(target=target, family=r.family, reliability=r.reliability,
               warnings=list(r.warnings), dim_predicted=r.dim_predicted, dim=r.dim,
               adjusted=r.adjusted, verified=r.verified, measured=r.measured,
               selection_measured=r.selection_measured, bits=r.bits,
               tool_memories=cost[0], tool_seconds_call=cost[1],
               pred_all=float(sizing.predict(sg["triples"], r.family, r.dim_predicted)),
               pred_q=float(sizing.predict(sg["triples"], r.family, r.dim_predicted,
                                           queries=sg["queries"])),
               excluded_model=not truth.fits(r.family, r.dim_predicted),
               excluded_final=not truth.fits(r.family, r.dim), d_min=None, baseline=None,
               **st)
    for name, d in (("model", r.dim_predicted), ("final", r.dim)):
        row[f"truth_{name}"] = P.mean_or_none(truth(r.family, d, sd["truth"]))
        row[f"truth_{name}_q"] = P.mean_or_none(truth(r.family, d, sd["truth"], "q"))
    return row, truth, sd


def run_cell(task):
    """Una cella nel worker, con il tetto (SIGALRM): min(tetto della cella, tempo rimasto
    al tetto rigido). Restituisce sempre un record (ok / timeout / error / skipped)."""
    c, sg, prev, cap_s, deadline = task
    left = deadline - time.time()
    out = {"cell": cid(c), **c}
    if left <= 1:
        return dict(out, status="not_started", seconds=0.0)
    lim = max(1, int(min(cap_s, left)))
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(lim)
    t0 = time.time()
    try:
        if c["key"].startswith("smoketest:"):         # solo smoke: guasti provocati
            if c["kind"] == "raise":
                raise ValueError("guasto provocato (smoke)")
            if c["kind"] == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
        if c["kind"] == "rec":
            row, _t, _s = rec_row(sg, c["target"], c["family"])
            out.update(status="ok" if row else "absent", row=row)
        else:
            if not prev or prev.get("status") != "ok" or not prev["row"]["verified"] \
                    or prev["row"]["truth_final"] is None:
                out.update(status="skipped")
            else:
                pr = prev["row"]
                ti = list(P.TARGETS).index(c["target"])
                sd = target_seeds(sg, ti)
                truth = Truth(sg)
                n = len(sg["triples"])
                if c["kind"] == "dmin":
                    out.update(status="ok", d_min=P.d_min_measured(
                        truth, c["family"], c["target"], pr["dim"], n, sd["dmin"]))
                else:
                    bd, bver, bmem, bsec = P.baseline(truth, c["family"], c["target"], n,
                                                      sd["base_sel"], sd["base_rep"])
                    out.update(status="ok", baseline={
                        "dim": bd, "verified": bver, "memories": bmem, "seconds": bsec,
                        "bits": None if bd is None else
                        bd * sizing.bits_per_component(c["family"], n),
                        "truth": None if bd is None else
                        P.mean_or_none(truth(c["family"], bd, sd["truth"]))})
    except CellTimeout:
        out.update(status="timeout")
    except Exception as e:                       # noqa: BLE001  (registrato, il run prosegue)
        out.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        try:
            signal.alarm(0)
        except CellTimeout:                      # allarme scattato tardi: già fuori dalla cella
            signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    out["seconds"] = time.time() - t0
    out["cap_s"] = lim
    return out


# ---- esecuzione per fasi, checkpoint per cella --------------------------------------

def _read_ckpt(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["cell"]] = d
    return done


def _write(fh, done, rec, name):
    rec["phase"] = name
    fh.write(json.dumps(rec, default=P._jdef) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
    done[rec["cell"]] = rec


def _run_phase(name, todo, ctx, workers, deadline, fh, done):
    """Finestra di al più `workers` celle in volo (apply_async). Watchdog nel padre: una
    cella in volo da più di min(tetto della cella, scadenza rigida) + WATCHDOG_SLACK_S (un
    worker ucciso da OOM/SIGKILL non risponde mai) è scritta come `timeout` (watchdog);
    il pool si termina e si ricrea; le altre celle in volo, sane, tornano in testa alla
    coda (non scritte: si rifanno). Il run non si blocca."""
    queue = list(todo)
    pool = ctx.Pool(workers, maxtasksperchild=50)
    inflight = {}
    try:
        while queue or inflight:
            while queue and len(inflight) < workers:
                task = queue.pop(0)
                inflight[cid(task[0])] = (pool.apply_async(run_cell, (task,)), time.time(), task)
            time.sleep(0.05)
            for k, (ar, _t0, task) in list(inflight.items()):
                if ar.ready():
                    try:
                        rec = ar.get()
                    except Exception as e:           # noqa: BLE001
                        rec = {"cell": k, **task[0], "status": "error",
                               "error": f"{type(e).__name__}: {e}", "seconds": 0.0}
                    _write(fh, done, rec, name)
                    del inflight[k]
            now = time.time()
            stuck = [k for k, (_ar, t0, task) in inflight.items()
                     if now > min(t0 + task[3], deadline) + WATCHDOG_SLACK_S]
            if stuck:
                pool.terminate()
                pool.join()
                for k in stuck:
                    _ar, t0, task = inflight.pop(k)
                    _write(fh, done, {"cell": k, **task[0], "status": "timeout",
                                      "watchdog": True, "seconds": now - t0}, name)
                queue = [task for _ar, _t0, task in inflight.values()] + queue
                inflight = {}
                pool = ctx.Pool(workers, maxtasksperchild=50)
    finally:
        pool.terminate()
        pool.join()


def execute(phases, subs, cost, ckpt, meta, workers, step, cap_h=HARD_CAP_H):
    """Esegue le fasi in ordine con un pool; una riga JSONL per cella (flush); ripresa
    idempotente (le celle già scritte, anche timeout, non si rifanno); tetto rigido dal
    primo avvio registrato in `meta`."""
    if meta.exists():
        m = json.loads(meta.read_text())
        if m["grid_step"] != step:
            raise SystemExit("checkpoint con un altro passo: deviazione da dichiarare")
    else:
        m = {"grid_step": step, "t_start": time.time(), "hard_cap_h": cap_h}
        meta.write_text(json.dumps(m))
    deadline = m["t_start"] + cap_h * 3600
    done = _read_ckpt(ckpt)
    ctx = mp.get_context("fork")
    with ckpt.open("a") as fh:
        for name, cells in phases:
            todo = []
            for c in cells:
                if cid(c) in done:
                    continue
                prev = done.get(cid(dict(c, kind="rec"))) if c["kind"] != "rec" else None
                cap = max(CELL_CAP_MIN_S, CELL_CAP_FACTOR * cost.get(cid(c), 0.0))
                todo.append((c, subs[c["key"]], prev, cap, deadline))
            if not todo:
                continue
            print(f"fase {name}: {len(todo)} celle", flush=True)
            _run_phase(name, todo, ctx, workers, deadline, fh, done)
    return done


def merge_rows(done):
    """Righe nel formato della 23 (con d_min e baseline dalle celle F3) e celle non misurate."""
    rows, unmeasured = {}, []
    for d in done.values():
        if d["kind"] == "rec":
            if d["status"] == "ok":
                rows[(d["key"], d["target"], d["family"])] = dict(d["row"])
            elif d["status"] in ("timeout", "not_started", "error", "absent"):
                unmeasured.append(d)
    for d in done.values():
        r = rows.get((d["key"], d["target"], d["family"]))
        if d["kind"] == "dmin" and d["status"] == "ok" and r:
            r["d_min"] = d["d_min"]
        if d["kind"] == "base" and d["status"] == "ok" and r:
            r["baseline"] = d["baseline"]
    return list(rows.values()), unmeasured


def summarize2(rows, unmeasured, pred, sel, n_min=None, planned=None):
    """P.summarize (le ipotesi della 23) sulle celle misurate, con la popolazione base
    N ≤ 400; più il WORST CASE: ogni cella validated prevista (fra le pianificate) non
    misurata conta come fallimento in H1, H2, H1u, H2u. Decisione: H1/H2/H1u/H2u
    «sostenute» solo se lo sono sia sulle misurate sia nel worst case."""
    base_keys = set(sel["base"])
    s = P.summarize(rows, base_keys, n_min=n_min)
    s["secondary_with_1600"] = {k: v for k, v in P.summarize(
        rows, base_keys | set(sel["b1600"]), n_min=n_min).items() if k in ("H3", "H5", "H6")}
    by = {(r["key"], r["target"], r["family"]): r for r in pred if _valid_cell(r)}
    um = {(d["key"], d["target"], d["family"]) for d in unmeasured}
    planned = planned if planned is not None else um
    val_planned = [k for k in planned if k in by and _validated(by[k])]
    nm = P.MIN_CLUSTERS if n_min is None else n_min
    for h, dss, claim in (("H1", P.IN_DIST, P.H1), ("H2", P.IN_DIST, P.H2),
                          ("H1u", P.UNSEEN, P.H1), ("H2u", P.UNSEEN, P.H2)):
        u = defaultdict(list)
        for r in rows:
            if r["dataset"] not in dss or r["reliability"] != "validated":
                continue
            if h.startswith("H1"):
                if not r["verified"] and not r["excluded_final"]:
                    continue
                u[r["key"]].append(r["truth_final"] is not None
                                   and r["truth_final"] >= r["target"] - P.TOL)
            else:
                u[r["key"]].append(r["truth_model"] is not None
                                   and r["truth_model"] >= r["target"] - P.BAND)
        for k in val_planned:
            if k in um and by[k]["dataset"] in dss:
                u[k[0]].append(False)
        wc = P.verdict_boot(u, *claim, n_min=nm)
        s[f"{h}_worst_case_unmeasured"] = wc
        prim = s[h]["verdict"]
        s[f"{h}_decision"] = "sostenuta" if prim == "sostenuta" and wc["verdict"] == \
            "sostenuta" else "falsificata" if "falsificata" in (prim, wc["verdict"]) and \
            prim != "sostenuta" else prim if prim != "sostenuta" else "non conclusiva"
    s["unmeasured"] = {"n": len(unmeasured), "by_status": Counter(d["status"] for d in unmeasured),
                       "by_phase": Counter(d.get("phase") for d in unmeasured),
                       "cells": sorted(cid(d) for d in unmeasured)}
    return s


# ---- tempi su sottografi reali (SOLO tempi) ---------------------------------------------

def _no_accuracy(*_a, **_k):
    raise AssertionError("accuratezza vietata in --timing-real")


def _time_mem(f, d, triples, q, allk):
    """Costruzione (traccia pigra inclusa) e query per chiave; le risposte NON si confrontano
    con nessun oggetto."""
    assert TIMING_MODE
    t0 = time.time()
    mem = families.make(f, d, 1)
    mem.store(triples)
    mem.query_many(q[:1])
    t1 = time.time()
    mem.query_many(q)
    t2 = time.time()
    mem.query_many(allk)
    t3 = time.time()
    return t1 - t0, (t2 - t1) / max(len(q), 1), (t3 - t2) / max(len(allk), 1)


def _timing_job(job):
    global TIMING_MODE
    TIMING_MODE = True
    sizing._measure = _no_accuracy          # la verifica non può girare
    P.Truth.__call__ = _no_accuracy         # il vero non può girare (solo in questo worker)
    sg, f, d, est = job
    allk = P.keys_of(sg["triples"])
    meas = [_time_mem(f, d, sg["triples"], sg["queries"], allk) for _ in range(TIMING_REPEATS)]
    return {"key": sg["key"], "dataset": sg["dataset"], "scheme": sg["scheme"], "N": sg["N"],
            "family": f, "D": d, "M": len({x for t in sg["triples"] for x in t}),
            "n_keys": len(allk), "Q": len(sg["queries"]), "est_bytes": est,
            "build_s": max(m[0] for m in meas), "q_s": max(m[1] for m in meas),
            "allk_s": max(m[2] for m in meas), "reps": meas}


def verify_counts_23():
    """Rende verificabili VERIFY_MEMS e P_ADJ: legge dal checkpoint della 23 (sha256
    P23_CKPT_SHA) SOLO i campi family, tool_memories, adjusted (nessuna accuratezza) e
    controlla che le costanti siano le medie arrotondate."""
    path = ROOT / "results" / "sizing_tool_prereg_results.jsonl"
    assert P.sha256(path) == P23_CKPT_SHA, "checkpoint della 23 cambiato"
    mem, adj = defaultdict(list), defaultdict(list)
    for line in path.read_text().splitlines():
        for r in json.loads(line)["rows"]:
            r = {k: r[k] for k in ("family", "tool_memories", "adjusted")}
            if r["tool_memories"]:
                mem[r["family"]].append(r["tool_memories"])
                adj[r["family"]].append(r["adjusted"])
    for f in P.FAMILIES:
        assert round(float(np.mean(mem[f])), 1) == VERIFY_MEMS[f], f
        assert round(float(np.mean(adj[f])), 2) == P_ADJ[f], f
    return {f: (len(mem[f]), VERIFY_MEMS[f], P_ADJ[f]) for f in P.FAMILIES}


def timing_real(graphs, workers, max_n=None, min_n=None):
    """Un sottografo reale per grafo × schema × N (k = K_TIMING, mai di valutazione): per ogni
    famiglia, ai D previsti (verify=False) dei tre target e a 2× il massimo (correzioni,
    D minimo), costruzione e query (su Q e su tutte le chiavi), massimo di 3 ripetizioni,
    con `workers` processi in parallelo (carico come nel run). Nessuna accuratezza: asserito
    (sizing._measure e Truth sostituiti da funzioni che falliscono)."""
    global TIMING_MODE
    TIMING_MODE = True
    jobs = []
    for sg in all_subgraphs(graphs, offset=K_TIMING, kmax=1):
        if (max_n and sg["N"] > max_n) or (min_n and sg["N"] < min_n):
            continue
        st = P.struct(sg)
        ds = defaultdict(set)
        for t in P.targets_of(sg, st["ceiling"]):
            for r in sizing.recommend(sg["triples"], t, queries=sg["queries"]):
                ds[r.family].add(r.dim)
        for f, dd in ds.items():
            pts = sorted(dd) + [2 * max(dd)]
            for d in pts:
                est = families.estimate_bytes(f, d, n_facts=len(sg["triples"]),
                                              n_symbols=st["M"], n_queries=st["n_keys"])
                if est > P.MAX_MEM:
                    continue
                jobs.append((sg, f, d, est))
    jobs.sort(key=lambda j: -j[0]["N"] * j[2])       # i più lunghi prima
    print(f"{len(jobs)} misure di tempo", flush=True)
    out = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(_timing_job, jobs, chunksize=1):
            out.append(r)
            print(f"  {r['dataset']:9s} {r['scheme']:7s} N={r['N']:4d} {r['family']:7s} "
                  f"D={r['D']:6d} M={r['M']} costr={r['build_s']:.2f}s "
                  f"Q={1e3 * r['q_s']:.2f}ms/ch tutte={1e3 * r['allk_s']:.2f}ms/ch", flush=True)
    return out


# ---- stima del costo di ogni cella ------------------------------------------------------

def _cost_fn(tim):
    tab = defaultdict(list)
    for t in tim:
        tab[(t["dataset"], t["scheme"], t["N"], t["family"])].append(t)
    for v in tab.values():
        v.sort(key=lambda t: t["D"])

    def c(ds, sc, n, f, d, nq, m, kind="allk"):
        """Secondi per UNA memoria a D con nq query: interpolazione lineare in D (estrapolazione
        lineare oltre il massimo misurato), query scalate per M / M del sottografo dei tempi."""
        ts = tab.get((ds, sc, n, f))
        if not ts:
            return float("inf")
        xs = np.array([t["D"] for t in ts], float)
        b = np.array([t["build_s"] for t in ts])
        q = np.array([t["q_s" if kind == "q" else "allk_s"] for t in ts])

        def ip(y):
            if d <= xs[-1] or len(xs) < 2:
                return float(np.interp(d, xs, y))
            return float(y[-1] + (y[-1] - y[-2]) / (xs[-1] - xs[-2]) * (d - xs[-1]))
        return ip(b) + ip(q) * nq * m / max(ts[0]["M"], 1)
    return c


def cell_cost(c, r, cf):
    """Stima (secondi, senza margine) di una cella dalle previsioni r e dai tempi reali.
    rec: verifica VERIFY_MEMS[f] memorie su Q a 1.25·D (le correzioni salgono di D) + vero
    TRUTH_SEEDS a D (+ TRUTH_SEEDS a 1.5·D con probabilità P_ADJ[f]) su tutte le chiavi.
    dmin: (2 + ⌈log2(D / 2·passo)⌉) × 20 memorie a 0.75·D su tutte le chiavi.
    base: (2 + ⌈log2(D/256)⌉ + ⌈log2(D / 2·passo)⌉) × 8 memorie a 0.75·D su Q + TRUTH_SEEDS
    su tutte le chiavi. Una cella HRR esclusa costa 0."""
    f, n, d = r["family"], r["N"], r["dim_predicted"]
    if excluded(f, d, n, r["M"], r["Q"]):
        return 0.0
    a = (r["dataset"], r["scheme"], n, f)
    stp = P.step_of(f, n)
    if c["kind"] == "rec":
        return VERIFY_MEMS[f] * cf(*a, 1.25 * d, r["Q"], r["M"], "q") + \
            CFG["truth"] * (cf(*a, d, r["n_keys"], r["M"]) +
                           P_ADJ[f] * cf(*a, 1.5 * d, r["n_keys"], r["M"]))
    if c["kind"] == "dmin":
        ev = 2 + max(0, math.ceil(math.log2(max(d / (2 * stp), 1))))
        return ev * P.DMIN_SEEDS * cf(*a, 0.75 * d, r["n_keys"], r["M"])
    ev = 2 + max(0, math.ceil(math.log2(max(d / P.BASE_D0, 1)))) + \
        max(0, math.ceil(math.log2(max(d / (2 * stp), 1))))
    return ev * VERIFY_SEEDS * cf(*a, 0.75 * d, r["Q"], r["M"], "q") + \
        CFG["truth"] * cf(*a, d, r["n_keys"], r["M"])


def schedule(phases, costs, workers):
    """Makespan simulato: lista in ordine, ogni cella al primo processo libero, barriera fra
    le fasi (come execute)."""
    t = 0.0
    out = []
    for name, cells in phases:
        free = [t] * workers
        for c in cells:
            i = int(np.argmin(free))
            free[i] += costs[cid(c)]
        t = max(free)
        out.append((name, len(cells), t))
    return t, out


def estimate(pred, tim, workers=WORKERS, verbose=True):
    cf = _cost_fn(tim)
    by = {(r["key"], r["target"], r["family"]): r for r in pred if _valid_cell(r)}
    res = []
    for i in range(len(REDUCTION)):
        configure(i)
        sel = select(pred)
        phases = cells_of(pred, sel)
        costs = {cid(c): cell_cost(c, by[(c["key"], c["target"], c["family"])], cf)
                 for _n, cs in phases for c in cs}
        span, per = schedule(phases, costs, workers)
        cpu = sum(costs.values())
        big = max(costs.values())
        res.append({"step": i, "subgraphs": len(set(sel["base"] + sel["val"] + sel["b1600"])),
                    "cells": len(costs), "cpu_h": cpu / 3600, "makespan_h": span / 3600,
                    "with_margin_h": MARGIN * span / 3600, "largest_cell_h": big / 3600,
                    "phases": [(n, k, round(s / 3600, 2)) for n, k, s in per]})
        if verbose:
            print(f"passo {i}: {REDUCTION[i]}\n   {res[-1]}", flush=True)
    return res


def peak_mem(pred):
    """Memoria di picco stimata (families.estimate_bytes con tutte le chiavi) della cella più
    grande del passo attivo, per scegliere i processi."""
    sel = select(pred)
    keys = set(sel["base"] + sel["val"] + sel["b1600"])
    best = 0
    for r in pred:
        if r["key"] in keys and _valid_cell(r) and not excluded(
                r["family"], r["dim_predicted"], r["N"], r["M"], r["Q"]):
            for mult in (1, 2):   # le correzioni e il D minimo arrivano a ~2× (oltre: limite)
                b = families.estimate_bytes(r["family"], mult * r["dim_predicted"],
                                            n_facts=r["N"], n_symbols=r["M"],
                                            n_queries=r["n_keys"])
                b = min(b, P.mem_cap(r["family"]))
                best = max(best, b)
    return best


# ---- potenza -----------------------------------------------------------------------

def power(pred):
    sel = select(pred)
    by = _rows_by_sub(pred)
    tab = {"selection": {k: len(v) for k, v in sel.items()}}
    for grp, dss in (("in", P.IN_DIST), ("unseen", P.UNSEEN)):
        sizes = sorted(Counter(r["key"] for k in sel["val"] for r in by[k]
                               if r["dataset"] in dss and _validated(r)).values())
        tab[f"validated_clusters_{grp}"] = len(sizes)
        tab[f"validated_recs_{grp}"] = int(sum(sizes))
        if len(sizes) >= P.MIN_CLUSTERS:
            for h, (cl, fl) in (("H1", P.H1), ("H2", P.H2)):
                tab[f"{h}_{grp}"] = {f"p={p},rho={rho}": P.power_cluster(sizes, p, cl, fl, rho)
                                     for p in (0.99, 0.98, 0.97, 0.95, 0.92, 0.90, 0.85)
                                     for rho in (0.0, 0.2)}
    pairs = defaultdict(bool)
    for k in sel["base"]:
        for r in by[k]:
            if not r["discarded"]:
                pairs[(k, r["target"])] |= bool(_validated(r))
    r16 = [r for r in pred if r["N"] > 400 and _valid_cell(r)]
    tab["validated_1600_untested"] = sum(r["reliability"] == "validated" for r in r16)
    tab["hrr_validated_1600_excluded"] = sorted(
        f"{r['key']}|{r['target']}|HRR" for r in r16
        if r["family"] == "HRR" and r["reliability"] == "validated")
    tab["base_clusters"] = len(sel["base"])
    tab["H5_pairs"] = len(pairs)
    tab["H5_predicted_max_coverage"] = P._prop(list(pairs.values()))
    dts, dn, dev = CFG["dmin"]
    hts, hn, hev = CFG["h7"]
    phases = dict(cells_of(pred, sel))
    for kind, h in (("dmin", "H4"), ("base", "H7")):
        cs = [c for c in phases["F3"] if c["kind"] == kind]
        vk = {c["key"] for c in cs if _validated(next(
            r for r in by[c["key"]] if r["target"] == c["target"] and r["family"] == c["family"]))}
        tab[f"{h}_subgraphs"] = len({c["key"] for c in cs})
        tab[f"{h}_cells"] = len(cs)
        tab[f"{h}_subgraphs_with_validated"] = len(vk)
    return tab


# ---- smoke (solo sintetico) ------------------------------------------------------------

def smoke(workers=4):
    """Impianto intero su grafi SINTETICI con il passo scelto (o l'ultimo): previsioni,
    selezione, fasi, celle con tetto, checkpoint, ripresa idempotente, riepilogo con worst
    case, controlli; un tetto per cella minuscolo deve produrre timeout registrati; il
    modello rotto (+5 punti) deve far fallire H2 e H3."""
    configure(GRID_STEP if GRID_STEP is not None else len(REDUCTION) - 1)
    g = P.Graph(P._syn(3000, 400, seed=7))
    sgs = [P.make_sub("fb15k237", 0, 0, "uniform", 100, K_OFFSET + j,
                      P.sample(g, 100, "uniform", np.random.default_rng(j))) for j in range(3)]
    ctrls = P.controls()
    pred = [r for sg in sgs for r in P.predict_sub(sg)]
    sel = select(pred)
    assert sel["base"] and sel["val"], sel
    phases = cells_of(pred, sel)
    for _n, cs in phases:
        assert len({cid(c) for c in cs}) == len(cs)
    subs = {sg["key"]: sg for sg in sgs}
    # controlli come celle rec (tutti i target dei controlli, tutte le famiglie)
    cpred = [r for c in ctrls for r in _control_pred(c)]
    ccells = [{"key": r["key"], "target": r["target"], "family": r["family"], "kind": "rec"}
              for r in cpred]
    subs.update({c["key"]: c for c in ctrls})
    phases = [("C", ccells)] + phases
    tmp = Path(tempfile.mkdtemp(prefix="sizing2_smoke_"))
    ck, mt = tmp / "ck.jsonl", tmp / "meta.json"
    t0 = time.time()
    done = execute(phases, subs, {}, ck, mt, workers, CFG["step"])
    t_full = time.time() - t0
    n1 = len(ck.read_text().splitlines())
    done2 = execute(phases, subs, {}, ck, mt, workers, CFG["step"])     # ripresa: niente da fare
    assert len(ck.read_text().splitlines()) == n1 and done2.keys() == done.keys()
    assert all(d["status"] in ("ok", "skipped", "absent") for d in done.values()), \
        Counter(d["status"] for d in done.values())
    # tetto per cella: con 1 s le celle grandi vanno in timeout e restano registrate
    ck3, mt3 = tmp / "ck3.jsonl", tmp / "meta3.json"
    old = (globals()["CELL_CAP_MIN_S"], globals()["CELL_CAP_FACTOR"])
    globals()["CELL_CAP_MIN_S"], globals()["CELL_CAP_FACTOR"] = 1, 0
    try:
        d3 = execute([phases[1]], subs, {}, ck3, mt3, workers, CFG["step"])
    finally:
        globals()["CELL_CAP_MIN_S"], globals()["CELL_CAP_FACTOR"] = old
    st3 = Counter(d["status"] for d in d3.values())
    print("tetto 1 s per cella:", dict(st3))
    assert st3["timeout"] > 0
    # tetto rigido già scaduto: nessuna cella parte, tutte registrate come not_started
    ck4, mt4 = tmp / "ck4.jsonl", tmp / "meta4.json"
    mt4.write_text(json.dumps({"grid_step": CFG["step"], "t_start": time.time() - 31 * 3600,
                               "hard_cap_h": HARD_CAP_H}))
    d4 = execute([phases[1]], subs, {}, ck4, mt4, workers, CFG["step"])
    assert {d["status"] for d in d4.values()} == {"not_started"}
    # guasti provocati: ValueError nel worker e worker ucciso (SIGKILL), fra celle vere
    ck6, mt6 = tmp / "ck6.jsonl", tmp / "meta6.json"
    subs["smoketest:x"] = {"key": "smoketest:x"}
    faults = [{"key": "smoketest:x", "target": 0.85, "family": "MAP-B", "kind": "raise"},
              {"key": "smoketest:x", "target": 0.85, "family": "MAP-I", "kind": "kill"}]
    old = (globals()["CELL_CAP_MIN_S"], globals()["WATCHDOG_SLACK_S"])
    globals()["CELL_CAP_MIN_S"], globals()["WATCHDOG_SLACK_S"] = 20, 5
    try:
        d6 = execute([("T", faults + phases[1][1][:6])], subs, {}, ck6, mt6, workers,
                     CFG["step"])
    finally:
        globals()["CELL_CAP_MIN_S"], globals()["WATCHDOG_SLACK_S"] = old
    st6 = {k: d["status"] for k, d in d6.items()}
    print("guasti provocati:", st6)
    assert st6["smoketest:x|0.85|MAP-B|raise"] == "error" and \
        "ValueError" in d6["smoketest:x|0.85|MAP-B|raise"]["error"]
    assert st6["smoketest:x|0.85|MAP-I|kill"] == "timeout" and \
        d6["smoketest:x|0.85|MAP-I|kill"].get("watchdog")
    assert len(d6) == 8 and sum(v == "ok" for v in st6.values()) == 6
    rows, um = merge_rows(done)
    for r in rows:
        if r["dim"] != r["dim_predicted"]:
            assert r["adjusted"]
        if r["d_min"] is not None:
            assert r["d_min"] % P.step_of(r["family"], r["N"]) == 0
    summ = summarize2(rows, um, pred + cpred, sel, n_min=1)
    rows3, um3 = merge_rows(d3)
    s3 = summarize2(rows3, um3, pred, sel, n_min=1,
                    planned={(d["key"], d["target"], d["family"]) for d in d3.values()})
    print("worst case con i timeout:", json.dumps({k: s3[k] for k in s3 if "worst" in k
                                                   and "unmeasured" in k}, default=P._jdef))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("descriptive", "excluded")},
                     indent=1, default=P._jdef))
    bad = [k for k, v in summ["controls"].items() if not v["passed"]]
    print("controlli non superati:", bad)
    ctrl_s = sum(d["seconds"] for d in done.values() if d["key"].startswith("control:"))
    print(f"smoke: {t_full:.0f} s a muro; controlli {ctrl_s:.0f} s di CPU")
    # modello rotto
    orig = sizing._predict_graph
    sizing._predict_graph = lambda g_, f, d: min(1.0, orig(g_, f, d) + 0.05)
    try:
        cneg = [c for c in ctrls if c["dataset"] == "control:negative"]
        bpred = [r for sg in sgs for r in P.predict_sub(sg)] + \
            [r for c in cneg for r in _control_pred(c)]
        bcells = [{"key": r["key"], "target": r["target"], "family": r["family"],
                   "kind": "rec"} for r in bpred if _valid_cell(r)]
        ck5, mt5 = tmp / "ck5.jsonl", tmp / "meta5.json"
        db = execute([("B", bcells)], subs, {}, ck5, mt5, workers, CFG["step"])
    finally:
        sizing._predict_graph = orig
    broken, _ = merge_rows(db)
    val = [r for r in broken if r["reliability"] == "validated" and r["truth_model"] is not None]
    pos = [r for r in broken if r["truth_model"] is not None and P._err(r) > P.ERR_BIG]
    p2 = P._prop([r["truth_model"] >= r["target"] - P.BAND for r in val])
    s3b = P._prop([r["reliability"] != "validated" for r in pos])
    print(f"modello rotto: H2 quota {p2} su {len(val)}, H3 sensibilità {s3b} su {len(pos)}")
    assert p2 is not None and p2 < P.H2[0], "modello rotto: H2 deve fallire"
    assert s3b is not None and s3b < P.H3[0], "modello rotto: H3 deve fallire"
    return {"seconds": t_full, "controls_cpu_s": ctrl_s, "controls_failed": bad}


def _control_pred(sg):
    """Righe 'previsione' di un controllo (tutti i suoi target), senza memorie."""
    st = P.struct(sg)
    out = []
    for t in P.targets_of(sg, st["ceiling"]):
        for r in sizing.recommend(sg["triples"], t, queries=sg["queries"]):
            out.append({"dataset": sg["dataset"], "scheme": sg["scheme"], "N": sg["N"],
                        "k": sg["k"], "key": sg["key"], "target": t, "family": r.family,
                        "dim_predicted": r.dim, "reliability": r.reliability,
                        "discarded": None, **st})
    return out


# ---- main ---------------------------------------------------------------------------

def run(graphs, workers):
    if GRID_STEP is None:
        raise SystemExit("GRID_STEP non scelto: esegui --timing-real e --estimate")
    configure(GRID_STEP)
    pred = json.loads(PRED_OUT.read_text())["rows"]
    tim = json.loads(TIMING_OUT.read_text())["timing"]
    sel = select(pred)
    phases = cells_of(pred, sel)
    subs = {sg["key"]: sg for sg in all_subgraphs(graphs)}
    ctrls = P.controls()
    cpred = [r for c in ctrls for r in _control_pred(c)]
    ccells = [{"key": r["key"], "target": r["target"], "family": r["family"], "kind": "rec"}
              for r in cpred]
    subs.update({c["key"]: c for c in ctrls})
    phases = [("C", ccells)] + phases if CONTROLS_FIRST else phases + [("C", ccells)]
    graphs.clear()                                       # i worker ricevono solo i sottografi
    cf = _cost_fn(tim)
    by = {(r["key"], r["target"], r["family"]): r for r in pred if _valid_cell(r)}
    cost = {cid(c): cell_cost(c, by[(c["key"], c["target"], c["family"])], cf)
            for n, cs in phases if n != "C" for c in cs}
    done = execute(phases, subs, cost, CKPT, META, workers, GRID_STEP)
    rows, um = merge_rows(done)
    planned = {(c["key"], c["target"], c["family"]) for n, cs in phases for c in cs
               if c["kind"] == "rec"}
    um += [dict(c, status="not_started") for n, cs in phases for c in cs
           if c["kind"] == "rec" and cid(c) not in done]
    summ = summarize2(rows, um, pred + cpred, sel, planned=planned)
    OUT.write_text(json.dumps({"summary": summ, "grid_step": GRID_STEP,
                               "config": REDUCTION[GRID_STEP]}, indent=1, default=P._jdef))
    print(json.dumps(summ, indent=1, default=P._jdef))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--timing-real", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--timing-1600", action="store_true",
                    help="seconda revisione: solo tempi a N = 1600 (1 thread), aggiunti al file t1")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        print("conteggi della 23:", verify_counts_23())
        print(json.dumps(smoke(min(a.workers, 4)), default=P._jdef))
    elif a.estimate or a.power:
        d = json.loads(PRED_OUT.read_text())
        if a.estimate:
            tim = json.loads(TIMING_OUT.read_text())["timing"]
            est = estimate(d["rows"], tim, a.workers)
            for e in est:
                configure(e["step"])
                e["peak_bytes_per_process"] = peak_mem(d["rows"])
            d["estimate"] = est
            ok = [e for e in est if e["with_margin_h"] <= RUN_BUDGET_H]
            print(f"budget {RUN_BUDGET_H} h (× {MARGIN}): primo passo sotto "
                  f"{ok[0]['step'] if ok else None}")
        if a.power:
            configure(GRID_STEP)
            d["grid_step"] = GRID_STEP
            d["selection"] = select(d["rows"])
            d["power"] = power(d["rows"])
            print(json.dumps(d["power"], indent=1, default=P._jdef))
        PRED_OUT.write_text(json.dumps(d, indent=1, default=P._jdef))
    else:
        graphs = {n: P.Graph(P.load(n)) for n in P.DATASETS}
        if a.predict:
            sgs = all_subgraphs(graphs)
            pred = [r for rs in P._pool_map(P.predict_sub, sgs, a.workers) for r in rs]
            PRED_OUT.write_text(json.dumps({
                "sha256": {n: P.DATA[n][2] for n in P.DATASETS}, "K_PRED": P.K_PRED,
                "seed_base": SEED_BASE, "k_offset": K_OFFSET, "rows": pred}, indent=1))
            print(f"{len(sgs)} sottografi, {len(pred)} righe")
        elif a.timing_1600:
            tim = timing_real(graphs, a.workers, None, 1600)
            d = json.loads(TIMING_OUT.read_text())
            d["timing"] = [t for t in d["timing"] if t["N"] < 1600] + tim
            d["n1600"] = {"repeats": TIMING_REPEATS, "blas_threads": BLAS_THREADS,
                          "workers": a.workers}
            TIMING_OUT.write_text(json.dumps(d, indent=1, default=P._jdef))
        elif a.timing_real:
            tim = timing_real(graphs, a.workers, TIMING_MAX_N)
            TIMING_OUT.write_text(json.dumps({"k_timing": K_TIMING, "repeats": TIMING_REPEATS,
                                              "blas_threads": BLAS_THREADS,
                                              "max_n": TIMING_MAX_N,
                                              "workers": a.workers, "timing": tim},
                                             indent=1, default=P._jdef))
        else:
            run(graphs, a.workers)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
