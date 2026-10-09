# ABM SDK — Developer Guide

Read the README's [Limits](../README.md#limits) first: at equal bits an exact
store, a Bloom filter or a `dict` + join beat ABM (preregistered tests 18 and
19). What ABM offers is an accuracy computable before the memory is built, as a
mean over queries.

## Install

ABM is not on PyPI:

```bash
pip install git+https://github.com/Kiatto/abm-runtime
```

## Size the memory first

```python
from abm import exact

triples = [("payment_service", "requires", "auth_service"),
           ("auth_service", "writes_to", "session_store")]

c = exact.contract_for(triples, dim=4096)
c["expected_accuracy"]   # mean over the stored (subject, relation) queries
c["ceiling"]             # cap set by aliases of symmetric relations, whatever D
exact.min_dimension(triples, 0.9)   # smallest D predicted to reach 90%,
                                    # None if 0.9 is above the ceiling
```

It is a mean, not a per-query guarantee, and the model assumes independent
codewords and facts: even cycles break it by up to 6 points (README, Limits).

## Store and query

```python
from abm import Memory

mem = Memory(dim=4096)
for t in triples:
    mem.store(*t)

mem.query("payment_service", "requires")               # (answer, confidence)
mem.chain("payment_service", ["requires", "writes_to"]) # (answer, confidence)
mem.member("payment_service", "requires", "auth_service")  # True / False
```

`confidence` ranks answers (0.5 = noise) but is **not calibrated**. Duplicates
are not removed; `(s, r, o)` and `(o, r, s)` are the same vector, so inverse
queries produce aliases. `abm.inspector` holds the older asymptotic Law IV
contract, superseded by `abm.exact` and optimistic with symmetric relations.

## CLI

```bash
abm demo
abm inspect examples/triples.json --dim 8192 --grounding 0.93
```

The input is a JSON array of `[subject, relation, object]` triples
([example](../examples/triples.json)); `--grounding` is your extractor's audited
precision, multiplied into the projected accuracy. `--law-iv` adds the superseded
report.

## Examples

- [examples/sdk/01_faq.py](../examples/sdk/01_faq.py)
- [examples/sdk/02_knowledge_base.py](../examples/sdk/02_knowledge_base.py)
- [examples/sdk/03_llm_memory.py](../examples/sdk/03_llm_memory.py)
