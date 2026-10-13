"""`abm size`: quale famiglia VSA, a quale dimensione, per queste triple (CLI di abm.sizing).

    abm size FILE --target 0.9 [--families MAP-B,FHRR] [--budget 64KB|512000bits]
             [--verify] [--seeds N] [--tol 0.01] [--queries FILE] [--json]
             [--max-mem 1GB] [--export torchhd [--out FILE]]

Input (dal suffisso del file):
- `.json`: lista di [s, r, o] (come `abm inspect`); per --queries, lista di [s, r].
- `.tsv` (tab) e `.csv` (virgola, quoting CSV standard): 3 colonne per riga (2 per
  --queries), nessuna intestazione (una riga d'intestazione sarebbe letta come una tripla);
  righe vuote e righe che iniziano con `#` ignorate.
Normalizzazione dei nomi: solo `str()` e strip degli spazi ai bordi. Niente minuscole: il
codebook di abm.Memory e di families distingue "Paris" da "paris", ma gli atomi di MAP-B
(`abm.random_hv`) sono derivati dal nome in minuscolo, quindi due nomi che differiscono solo
per le maiuscole hanno lo STESSO atomo MAP-B (pareggio garantito nel cleanup) mentre il
modello li tratta come indipendenti. Questi casi sono segnalati (avviso d'input).

Budget e memoria: un numero con unità obbligatoria. `bits`/`bit`, `B`/`bytes`, `KB`/`KiB`,
`MB`/`MiB`, `GB`/`GiB`; le unità sono binarie (KB = 1024 byte, come `kbytes` nelle
raccomandazioni). --budget conta i bit della traccia (come `Recommendation.bits`).

Codici di uscita: 0 almeno una raccomandazione affidabile ("validated", entro il budget e,
con --verify, verificata); 3 nessuna (solo "caution"/"unvalidated"/non verificate, o nessuna
famiglia raggiunge il target); 2 errore d'uso (argomenti); 1 errore d'input (file illeggibile,
formato errato, target sopra il tetto degli alias).

JSON (--json): schema `abm.size/1`, vedi `SCHEMA` e README. Stabile: i campi si aggiungono
con la stessa versione, si rimuovono o cambiano di significato solo con una versione nuova.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

SCHEMA = "abm.size/1"

_UNITS = {"bit": 1, "bits": 1, "b": 8, "byte": 8, "bytes": 8,
          "kb": 8 << 10, "kib": 8 << 10, "mb": 8 << 20, "mib": 8 << 20,
          "gb": 8 << 30, "gib": 8 << 30}

STAT_ORDER = ("alias_share", "twin_share", "pair_multirel_share", "multiobj_share",
              "s_max_share", "s_p99_share", "o_max_share", "o_p99_share",
              "r_max_share", "r_p99_share")

HONEST_NOTE = (
    "Predicted = model (abm.sizing), computed from the structure of these triples, no "
    "memory built. Measured = accuracy of real memories of our encoding on fresh seeds "
    "(only with --verify; a finite sample, CI over seeds). 'validated' only means the "
    "graph's structure lies inside the ranges tested in preregistrations 21-22; the "
    "observed-error bands are errors seen on those benchmark subgraphs, not guarantees "
    "for this graph. All accuracies are means over the (subject, relation) queries.")


# ---- parsing --------------------------------------------------------------------

def parse_size(text, what="size"):
    """'64KB' → bit. Unità obbligatoria, binaria (KB = 1024 byte)."""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*", str(text))
    if not m or m.group(2).lower() not in _UNITS:
        raise argparse.ArgumentTypeError(
            f"{what} must be a number with a unit (bits, B, KB, MB, GB; binary), "
            f"got {text!r}")
    bits = int(float(m.group(1)) * _UNITS[m.group(2).lower()])
    if bits < 1:
        raise argparse.ArgumentTypeError(f"{what} must be positive, got {text!r}")
    return bits


def _budget(text):
    return parse_size(text, "budget")


def _max_mem(text):
    bits = parse_size(text, "max-mem")
    if bits < 8:
        raise argparse.ArgumentTypeError(f"max-mem must be at least 1 byte, got {text!r}")
    return bits // 8


def _target(text):
    try:
        v = float(text)
    except ValueError:
        v = math.nan
    if not 0.0 < v <= 1.0:
        raise argparse.ArgumentTypeError(f"target must be an accuracy in (0, 1], got {text!r}")
    return v


def _tol(text):
    try:
        v = float(text)
    except ValueError:
        v = math.nan
    if not 0.0 <= v < 1.0:
        raise argparse.ArgumentTypeError(f"tol must be in [0, 1), got {text!r}")
    return v


def _pos_int(text):
    try:
        v = int(text)
    except ValueError:
        v = 0
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return v


def _families_arg(text):
    from .sizing import FAMILIES
    by_upper = {f.upper(): f for f in FAMILIES}
    out = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if part.upper() not in by_upper:
            raise argparse.ArgumentTypeError(
                f"unknown family {part!r}; valid: {', '.join(FAMILIES)}")
        if by_upper[part.upper()] not in out:
            out.append(by_upper[part.upper()])
    if not out:
        raise argparse.ArgumentTypeError("no families given")
    return out


class InputError(Exception):
    pass


def load_rows(path, width):
    """Righe di `width` stringhe normalizzate (strip) da .json, .tsv o .csv.
    Restituisce (righe, formato). InputError su file o formato non validi."""
    ext = os.path.splitext(path)[1].lower()
    fmt = {".json": "json", ".tsv": "tsv", ".csv": "csv"}.get(ext)
    if fmt is None:
        raise InputError(f"{path}: unknown format {ext or '(no suffix)'!r}; "
                         "use .json, .tsv or .csv")
    kind = "[s, r, o] triples" if width == 3 else "[s, r] queries"
    try:
        with open(path, newline="", encoding="utf-8") as f:
            if fmt == "json":
                data = json.load(f)
                if not isinstance(data, list):
                    raise InputError(f"{path} must be a JSON array of {kind}")
                rows = list(data)
                where = [f"entry {i}" for i in range(len(rows))]
            else:
                reader = csv.reader(f, delimiter="\t" if fmt == "tsv" else ",")
                rows, where = [], []
                for i, row in enumerate(reader, 1):
                    if not row or all(not c.strip() for c in row) \
                            or row[0].lstrip().startswith("#"):
                        continue
                    rows.append(row)
                    where.append(f"line {i}")
    except InputError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as e:
        raise InputError(f"cannot read {path}: {e}")
    if not rows:
        raise InputError(f"{path} contains no {kind}")
    out = []
    for row, w in zip(rows, where):
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise InputError(f"{path}, {w}: expected {width} fields ({kind}), got {row!r}")
        vals = tuple(str(x).strip() for x in row)
        if any(not v for v in vals):
            raise InputError(f"{path}, {w}: empty name in {row!r}")
        out.append(vals)
    return out, fmt


def case_collisions(triples):
    """Gruppi di nomi distinti che coincidono in minuscolo (stesso atomo MAP-B)."""
    by_low = defaultdict(set)
    for t in triples:
        for x in t:
            by_low[x.lower()].add(x)
    return sorted(sorted(v) for v in by_low.values() if len(v) > 1)


# ---- torchhd ----------------------------------------------------------------------
# Verificato sui sorgenti di torchhd (github.com/hyperdimensional-computing/torchhd, branch
# main, commit 9d73e1b del 2025-06-19; PyPI torch-hd 5.8.4), NON installando né eseguendo
# torchhd: torchhd/types.py (VSAOptions = BSC, MAP, HRR, FHRR, BSBC, VTB, MCR, CGR),
# functional.get_vsa_tensor_class e tensors/{bsc,map,fhrr,hrr,bsbc}.py (bind, multibundle,
# permute, inverse). Lo snippet Python generato NON è stato eseguito.
TORCHHD_SOURCE = ("torchhd source, branch main @ 9d73e1b (2025-06-19), PyPI torch-hd 5.8.4; "
                  "names and operations read from the source, torchhd not installed or run")
TORCHHD = {
    "MAP-B": ("BSC", "close", [
        "bind: XOR on {0,1} = our product on ±1 (same algebra); permute(shifts=1) = our ρ "
        "(roll by 1); multiset: majority with a random tie bit per component for an even "
        "count (ours: a fixed pseudo-random tie vector, same distribution)",
        "atoms: torchhd draws them from a torch generator, ours from md5(name): same "
        "distribution, different vectors; cleanup ties: ours take the first symbol, "
        "argmax in torch also takes the first index, but the symbol order differs",
        "similarity: use hamming_similarity (argmax); 1 bit per component as in the table"]),
    "MAP-I": ("MAP", "close", [
        "bind: elementwise product, multiset: integer sum (no sign), permute: roll — as "
        "ours; torchhd.random(..., vsa='MAP') gives ±1 in float32 by default",
        "storage: torchhd keeps the sum in float32 (32 bits/component); our KB assume "
        "ceil(log2(N+1))-bit integers, so torchhd uses more memory than the table says",
        "similarity: dot_similarity (argmax)"]),
    "FHRR": ("FHRR", "differs", [
        "bind: elementwise complex product, multiset: complex sum, permute: roll, "
        "inverse: conjugate — as ours",
        "precision: our trace is quantized to float16 real + float16 imaginary (32 bits/"
        "component) and the model includes it; torchhd keeps complex64 (64 bits/"
        "component). The prediction is for the float16 trace, not for torchhd's",
        "similarity: dot_similarity (real part, argmax)"]),
    "HRR": ("HRR", "close", [
        "bind: circular convolution via FFT, multiset: sum, permute: roll — as ours",
        "unbinding: use torchhd.inverse (the involution, i.e. circular correlation, which "
        "is ours); exact_inverse (1/FFT) is NOT what our model covers",
        "atoms: torchhd normalizes each atom to unit norm, ours are N(0, 1/D) (norm ≈ 1): "
        "close but not identical",
        "similarity: dot_similarity (argmax); float32 as in the table"]),
    "BSDC-OR": (None, "none", [
        "no torchhd model stores this trace. Nearest: 'BSBC' (block codes, one index per "
        "block of size L): its bind (index addition mod L) is ours, but its multiset "
        "keeps one index per block (torch.mode), not the OR of all blocks",
        "our predictions do not apply to BSBC bundling"]),
    "BSDC-S": (None, "none", [
        "no torchhd model stores this trace. Nearest: 'BSBC' (bind = ours, index addition "
        "mod L), but its multiset keeps one index per block (torch.mode), not the counts",
        "our predictions do not apply to BSBC bundling"]),
}

_SNIPPET = '''\
# Encoding of abm.families in torchhd ({family} -> vsa="{vsa}", D = {dim}).
# NOT executed (torchhd was not installed when this was generated): check it before use.
# The accuracy predicted by `abm size` is for OUR implementation; see the notes.
import torch
import torchhd

def build(triples, D={dim}, vsa="{vsa}"):
    symbols = sorted({{x for t in triples for x in t}})   # entities and relations
    idx = {{x: i for i, x in enumerate(symbols)}}
    A = torchhd.random(len(symbols), D, vsa=vsa)

    def key(s, r):                       # s * rho(r)
        return torchhd.bind(A[idx[s]], torchhd.permute(A[idx[r]], shifts=1))

    facts = torch.stack([torchhd.bind(key(s, r), A[idx[o]]) for s, r, o in triples])
    trace = torchhd.multiset(facts)      # {bundle}

    def query(s, r):                     # cleanup over ALL symbols
        y = torchhd.bind(trace, torchhd.inverse(key(s, r)))
        return symbols[int(torchhd.{sim}(y, A).argmax())]
    return query
'''


def torchhd_export(recs):
    out = {"target_library": "torchhd", "verification": TORCHHD_SOURCE,
           "encoding": "f = s * permute(r, 1) * o; trace = multiset of the facts; query "
                       "(s, r): bind(trace, inverse(s * permute(r, 1))), argmax similarity "
                       "over all stored symbols (entities and relations)",
           "caveat": "the predictions and measurements of `abm size` are for the abm "
                     "encoding (abm.families); where torchhd differs (see status and "
                     "notes) they do not transfer",
           "models": []}
    for r in recs:
        vsa, status, notes = TORCHHD[r.family]
        item = {"family": r.family, "dim": r.dim, "torchhd_vsa": vsa, "status": status,
                "notes": notes}
        if vsa is not None:
            item["python"] = _SNIPPET.format(
                family=r.family, vsa=vsa, dim=r.dim,
                bundle="majority" if vsa == "BSC" else "sum",
                sim="hamming_similarity" if vsa == "BSC" else "dot_similarity")
        out["models"].append(item)
    return out


# ---- report -------------------------------------------------------------------------

def _rec_dict(r):
    oe = r.observed_error
    d = {"family": r.family, "dim": r.dim, "dim_predicted": r.dim_predicted,
         "bits": r.bits, "kbytes": round(r.kbytes, 4), "predicted": r.predicted,
         "reliability": r.reliability, "warnings": list(r.warnings),
         "observed_error": {"regime": oe.regime, "mae_points": oe.mae,
                            "worst_points": oe.worst, "cells": oe.cells,
                            "worst_cell": oe.worst_cell, "context": oe.context},
         "verify": None}
    if r.tol is not None:
        d["verify"] = {
            "measured": r.measured, "measured_ci": list(r.measured_ci) if r.measured_ci
            else None, "per_memory_q10": r.per_memory_q10,
            "selection_measured": r.selection_measured,
            "selection_ci": list(r.selection_ci) if r.selection_ci else None,
            "verified": r.verified, "tol": r.tol, "adjusted": r.adjusted,
            "over_budget": r.over_budget}
    return d


def is_reliable(r, verify):
    return (r.reliability == "validated" and not r.over_budget
            and (r.verified if verify else True))


def graph_dict(g):
    from . import sizing
    stats = {}
    for name in STAT_ORDER:
        v = sizing._range_value(g, name)
        lo, hi, _a, _b = sizing.STRUCT_RANGES[name]
        stats[name] = {"value": v, "validated_min": lo, "validated_max": hi,
                       "inside": lo - 1e-12 <= v <= hi + 1e-12}
    return {"n_facts": g.n, "n_distinct": len(set(g.triples)), "n_symbols": g.m,
            "n_queries": len(g.queries), "regime": g.regime, "alias_ceiling": g.ceiling,
            "self_loops": g.self_loops,
            "n_range": {"value": g.n, "validated_min": sizing.N_MIN_TESTED,
                        "validated_max": sizing.N_MAX_TESTED,
                        "inside": sizing.N_MIN_TESTED <= g.n <= sizing.N_MAX_TESTED},
            "stats": stats}


def graph_text(gd, path, fmt):
    lines = [f"GRAPH  {path} ({fmt})",
             f"  facts N = {gd['n_facts']} (weighted; {gd['n_distinct']} distinct), "
             f"symbols M = {gd['n_symbols']}, queries = {gd['n_queries']}, "
             f"regime = {gd['regime']}, alias ceiling = {gd['alias_ceiling']:.4f}",
             f"  {'statistic':<22}{'value':>9}   validated range (preregs 21-22)"]
    nr = gd["n_range"]
    lines.append(f"  {'N':<22}{nr['value']:>9}   [{nr['validated_min']}, "
                 f"{nr['validated_max']}]{'' if nr['inside'] else '   OUT'}")
    labels = {"alias_share": "alias (queries)", "twin_share": "twins (vectors)",
              "pair_multirel_share": "pair multi-relation", "multiobj_share": "multi-object",
              "s_max_share": "hub subject max", "s_p99_share": "hub subject p99",
              "o_max_share": "hub object max", "o_p99_share": "hub object p99",
              "r_max_share": "relation max", "r_p99_share": "relation p99"}
    for name in STAT_ORDER:
        s = gd["stats"][name]
        lines.append(f"  {labels[name]:<22}{s['value']:>9.4f}   [{s['validated_min']:.4g}, "
                     f"{s['validated_max']:.4g}]{'' if s['inside'] else '   OUT'}")
    return "\n".join(lines)


def table_text(recs, verify):
    head = f"  {'family':<8}{'D':>8}{'KB':>10}{'predicted':>11}"
    if verify:
        head += f"{'measured [95% CI]':>27}{'q10':>8}{'verified':>10}"
    head += f"  {'reliability':<12}"
    lines = [head]
    for r in recs:
        row = f"  {r.family:<8}{r.dim:>8}{r.kbytes:>10.2f}{r.predicted:>11.4f}"
        if verify:
            if r.measured is None:
                row += f"{'not measured':>27}{'-':>8}{'no':>10}"
            else:
                m = f"{r.measured:.4f} [{r.measured_ci[0]:.4f}, {r.measured_ci[1]:.4f}]"
                row += f"{m:>27}{r.per_memory_q10:>8.4f}{'yes' if r.verified else 'no':>10}"
        row += f"  {r.reliability:<12}"
        lines.append(row)
    return "\n".join(lines)


def warnings_text(recs):
    lines = []
    for r in recs:
        oe = r.observed_error
        lines.append(f"  {r.family}: observed benchmark error ({oe.regime}) mean "
                     f"{oe.mae} pt, worst {oe.worst} pt over {oe.cells} cells")
        for w in r.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)


# ---- stima dei costi prima della verifica ---------------------------------------

def cost_estimates(g, recs, seeds, max_mem):
    """Per famiglia: memoria di picco stimata al D del modello e a 8·D (tetto della
    correzione), e tempo GREZZO: una memoria costruita e interrogata al D del modello
    (seme 0, non riusata), × 2·seeds memorie; ogni passo di correzione ne aggiunge
    `seeds` a un D più grande, quindi è un limite inferiore se il modello è ottimista."""
    from . import families, sizing
    out = []
    nq = min(len(g.queries), 2000)
    sample = list(g.queries)[:nq]
    for r in recs:
        peak = sizing._peak_bytes(g, r.family, r.dim, nq)
        top = min(8 * r.dim, sizing.D_MAX)
        peak_top = sizing._peak_bytes(g, r.family, top, nq)
        if peak > max_mem:
            out.append({"family": r.family, "dim": r.dim, "peak_bytes": peak,
                        "peak_bytes_at_8x": peak_top, "seconds_per_memory": None,
                        "seconds_total_min": None})
            continue
        t0 = time.perf_counter()
        mem = families.make(r.family, r.dim, 0)
        mem.store(g.triples)
        mem.query_many(sample)
        dt = time.perf_counter() - t0
        out.append({"family": r.family, "dim": r.dim, "peak_bytes": peak,
                    "peak_bytes_at_8x": peak_top, "seconds_per_memory": round(dt, 4),
                    "seconds_total_min": round(dt * 2 * seeds, 2)})
    return out


def costs_text(costs, seeds):
    lines = [f"VERIFY COST (before running): {2 * seeds} memories per family "
             f"({seeds} selection + {seeds} report seeds); time is a crude estimate (one "
             "pilot memory × count; more if D is corrected), memory from "
             "families.estimate_bytes"]
    for c in costs:
        t = "skipped: over --max-mem" if c["seconds_per_memory"] is None else \
            f"~{c['seconds_total_min']:.1f} s or more"
        lines.append(f"  {c['family']:<8} D={c['dim']:<7} peak ≈ {c['peak_bytes'] / 2**20:.1f} MiB "
                     f"(≈ {c['peak_bytes_at_8x'] / 2**20:.1f} MiB at 8·D)  time {t}")
    return "\n".join(lines)


# ---- comando -------------------------------------------------------------------------

def add_parser(sub):
    p = sub.add_parser(
        "size", help="recommend a VSA family and dimension for a set of triples",
        description="Recommend, per VSA family, the smallest dimension predicted to reach "
                    "a mean query accuracy. Input: .json (list of [s, r, o]), .tsv or .csv "
                    "(3 columns, no header). Exit: 0 reliable recommendation, 3 none "
                    "reliable, 2 usage error, 1 input error.")
    p.add_argument("file", help=".json, .tsv or .csv triples")
    p.add_argument("--target", type=_target, required=True,
                   help="mean accuracy over the (s, r) queries, in (0, 1]")
    p.add_argument("--families", type=_families_arg, default=None,
                   help="comma-separated subset (default: all six)")
    p.add_argument("--budget", type=_budget, default=None,
                   help="max trace size with unit: 512000bits, 64KB (binary units)")
    p.add_argument("--verify", action="store_true",
                   help="measure real memories on fresh seeds (slow)")
    p.add_argument("--seeds", type=_pos_int, default=None,
                   help="seeds per verification stage (default 8)")
    p.add_argument("--tol", type=_tol, default=None,
                   help="verification tolerance, absolute accuracy (default 0.01)")
    p.add_argument("--queries", default=None,
                   help="file of [s, r] queries (same formats; default: all stored pairs)")
    p.add_argument("--json", action="store_true", help=f"machine output, schema {SCHEMA}")
    p.add_argument("--max-mem", type=_max_mem, default=None,
                   help="skip verification dims whose estimated peak exceeds this (1GB)")
    p.add_argument("--export", choices=["torchhd"], default=None,
                   help="emit a torchhd configuration for the recommendations")
    p.add_argument("--out", default=None, help="write the --export output to this file")
    p.set_defaults(fn=cmd_size)
    return p


def _err(msg, code):
    print(f"error: {msg}", file=sys.stderr)
    return code


def cmd_size(args):
    from . import __version__, sizing
    if args.out and not args.export:
        return _err("--out needs --export", 2)
    if not args.verify and (args.seeds or args.tol is not None or args.max_mem):
        return _err("--seeds, --tol and --max-mem need --verify", 2)
    seeds = args.seeds or sizing.VERIFY_SEEDS
    tol = sizing.VERIFY_TOL if args.tol is None else args.tol
    max_mem = args.max_mem or sizing.MAX_MEM_BYTES
    try:
        triples, fmt = load_rows(args.file, 3)
        queries = load_rows(args.queries, 2)[0] if args.queries else None
        g = sizing._Graph(triples, queries)
    except InputError as e:
        return _err(str(e), 1)
    except (ValueError, KeyError) as e:
        return _err(f"{args.file}: {e}", 1)

    input_warnings = []
    for grp in case_collisions(triples):
        input_warnings.append(
            f"names equal up to case {grp}: distinct symbols, but the SAME MAP-B atom "
            "(abm.random_hv lowercases), so MAP-B always ties them; the model does not "
            "know. Normalize them upstream if they are the same entity")
    fams = list(args.families or sizing.FAMILIES)
    excluded = []
    if g.self_loops:
        for f in [f for f in fams if f != "MAP-B"]:
            fams.remove(f)
            excluded.append({"family": f, "reason": "self-loops (s, r, s) are modelled "
                             "only for MAP-B"})
    out_json = {"schema": SCHEMA, "abm_version": __version__,
                "input": {"file": args.file, "format": fmt, "n_rows": len(triples),
                          "queries_file": args.queries,
                          "normalization": "str() and strip; case preserved"},
                "params": {"target": args.target, "families": list(args.families or
                                                                   sizing.FAMILIES),
                           "budget_bits": args.budget, "verify": args.verify,
                           "seeds": seeds if args.verify else None,
                           "tol": tol if args.verify else None,
                           "max_mem_bytes": max_mem if args.verify else None},
                "graph": graph_dict(g), "input_warnings": input_warnings}
    if args.target > g.ceiling:
        return _err(f"target {args.target} is above the alias ceiling {g.ceiling:.4f} of "
                    "this graph: no family and no dimension reaches it on average; lower "
                    "the target or remove the aliased facts", 1)
    recs = sizing.recommend(g, args.target, families=fams, budget_bits=args.budget) \
        if fams else []
    got = {r.family for r in recs}
    for f in fams:
        if f not in got:
            reason = ("not reachable within D_max = %d" % sizing.D_MAX
                      if sizing.min_dim(g, f, args.target) is None
                      else "over the budget at the predicted D")
            excluded.append({"family": f, "reason": reason})

    text = []
    if not args.json:
        text.append(graph_text(out_json["graph"], args.file, fmt))
        for w in input_warnings:
            text.append(f"  input warning: {w}")
        print("\n".join(text), flush=True)
    costs = None
    if args.verify and recs:
        costs = cost_estimates(g, recs, seeds, max_mem)
        msg = costs_text(costs, seeds)
        print("\n" + msg if not args.json else msg, flush=True,
              file=sys.stderr if args.json else sys.stdout)
        recs = sizing.recommend(g, args.target, families=fams, budget_bits=args.budget,
                                verify=True, seeds=seeds, tol=tol, max_mem_bytes=max_mem)
    reliable = any(is_reliable(r, args.verify) for r in recs)
    code = 0 if reliable else 3
    out_json.update({"recommendations": [_rec_dict(r) for r in recs], "excluded": excluded,
                     "verify_cost_estimate": costs, "reliable": reliable,
                     "exit_code": code, "note": HONEST_NOTE})
    export = torchhd_export(recs) if args.export == "torchhd" else None
    if export is not None and args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except OSError as e:
            return _err(f"cannot write {args.out}: {e}", 1)
    if args.json:
        if export is not None and not args.out:
            out_json["export"] = export
        print(json.dumps(out_json, indent=2, ensure_ascii=False))
        return code

    lines = ["", f"RECOMMENDATIONS  target = {args.target} (mean accuracy over queries), "
             f"sorted by bits" + (f", budget {args.budget} bits" if args.budget else "")]
    lines.append(table_text(recs, args.verify) if recs else "  none")
    for e in excluded:
        lines.append(f"  {e['family']}: excluded, {e['reason']}")
    if recs:
        lines += ["", "WARNINGS AND BENCHMARK ERROR", warnings_text(recs)]
    lines += ["", "VERDICT  " + (
        "at least one recommendation is 'validated'" + (" and verified" if args.verify
                                                        else " (not measured: add --verify)")
        if reliable else "no reliable recommendation (only caution/unvalidated/unverified)")
        + f"; exit {code}", "", "NOTE  " + HONEST_NOTE]
    if export is not None:
        if args.out:
            lines.append(f"\ntorchhd export written to {args.out}")
        else:
            lines += ["", "TORCHHD EXPORT", json.dumps(export, indent=2, ensure_ascii=False)]
    print("\n".join(lines))
    return code
