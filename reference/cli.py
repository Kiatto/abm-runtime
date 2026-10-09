"""abm — command-line interface (SDK tooling, not part of the frozen
specification).

    abm inspect triples.json [--dim 8192] [--grounding 0.93] [--law-iv]
    abm demo

`triples.json` is a JSON array of [subject, relation, object] triples —
the natural output format of an LLM extractor. The contract printed is the
exact finite-dimension model (abm.exact.contract_for); `--law-iv` adds the
older asymptotic Law IV report, superseded and labelled as such.
"""

import argparse
import json
import math
import sys


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"dim must be a positive integer, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"dim must be a positive integer, got {text!r}")
    return value


def _probability(text):
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:                    # nan included
        raise argparse.ArgumentTypeError(
            f"grounding must be a number in [0, 1], got {text!r}")
    return value


def exact_contract_text(triples, dim, grounding=1.0):
    """The exact contract (abm.exact.contract_for) as text."""
    from . import exact
    c = exact.contract_for(triples, dim)
    return (
        f"MEMORY CONTRACT (exact model, abm.exact.contract_for)\n"
        f"  Facts           =  {c['facts']} (D={c['dim']}, codebook={c['codebook']})\n"
        f"  Expected accuracy = {c['expected_accuracy']:.1%} "
        f"(mean over the stored (subject, relation) queries)\n"
        f"  Alias ceiling   =  {c['ceiling']:.1%} (whatever D)\n"
        f"  Symmetric twins =  {c['twin_share']:.1%} of triples; "
        f"queries with aliases {c['alias_share']:.1%}\n"
        f"  Grounding       =  {grounding:.1%} "
        f"(projected single query {grounding * c['expected_accuracy']:.1%})\n"
        f"  A mean over queries, not a per-query guarantee."
    )


def cmd_inspect(args):
    from . import Memory, report
    try:
        with open(args.file) as f:
            triples = json.load(f)
    except Exception as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 1
    if not isinstance(triples, list) or not triples:
        print(f"error: {args.file} must be a non-empty JSON array of "
              f"[s, r, o] triples", file=sys.stderr)
        return 1
    bad = [t for t in triples
           if not (isinstance(t, (list, tuple)) and len(t) == 3)]
    if bad:
        print(f"error: {len(bad)} entries are not [s, r, o] triples "
              f"(first: {bad[0]!r})", file=sys.stderr)
        return 1
    triples = [(str(s), str(r), str(o)) for s, r, o in triples]
    print(exact_contract_text(triples, args.dim, args.grounding))
    if args.law_iv:
        mem = Memory(dim=args.dim)
        for t in triples:
            mem.store(*t)
        print("\nLegacy report — asymptotic Law IV, superseded: optimistic at "
              "small D, blind to twins and aliases")
        print(report(mem, extractor_precision=args.grounding))
    return 0


def cmd_demo(args):
    from . import Memory
    mem = Memory(dim=4096)
    facts = [("payment_service", "requires", "auth_service"),
             ("auth_service", "writes_to", "session_store"),
             ("session_store", "deployed_in", "eu_west")]
    for f in facts:
        mem.store(*f)
    print("stored:", *[f"  {s} --{r}--> {o}" for s, r, o in facts],
          sep="\n")
    ans, conf = mem.chain("payment_service", ["requires", "writes_to"])
    print(f"\nchain(payment_service, [requires, writes_to]) "
          f"= {ans} (confidence {conf:.2f}, not calibrated)\n")
    print(exact_contract_text(facts, mem.dim, grounding=0.93))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="abm",
        description="ABM — algebraic memory runtime with predictive "
                    "Memory Contracts")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect",
                        help="load triples and print the Memory Contract")
    pi.add_argument("file", help="JSON array of [s, r, o] triples")
    pi.add_argument("--dim", type=_positive_int, default=8192,
                    help="trace size in bits (default 8192)")
    pi.add_argument("--grounding", type=_probability, default=1.0,
                    help="audited precision of your extractor (0..1)")
    pi.add_argument("--law-iv", action="store_true",
                    help="also print the superseded asymptotic Law IV report")
    pi.set_defaults(fn=cmd_inspect)

    pd = sub.add_parser("demo", help="30-second tour")
    pd.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
