# Bozza: email a Denis Kleyko ed E. Paxon Frady

Note per kiatto (non fanno parte dell'email):

- Destinatari consigliati dall'audit del 2026-10-01, §5. Mandarla solo da te, dal tuo
  indirizzo; la firma è da completare.
- Tono: chiedere critica e riferimenti, non approvazione. Niente "exact distributions"
  come punto centrale: il confronto numerico con il loro integrale mostra che non
  aggiungono nulla di misurabile, e il paper lo dice.
- Prima di mandarla: controllare che il link al tag `paper-v1.10` (o a quello più
  recente) funzioni da una finestra anonima.
- Lunghezza: tenerla così. Chi riceve email di questo tipo legge le prime righe.

---

**Subject:** A preregistered test of finite-size retrieval theory on MAP-B memories — questions on two points

Dear Dr. Kleyko, dear Dr. Frady,

I am an independent researcher working on the readout accuracy of binary
majority-bundled memories (MAP-B) used as knowledge-graph stores. Your finite-size
retrieval theory is the basis of the work, and I would value your view on two
specific questions before I submit it anywhere.

What the work does, briefly:

- It applies the finite-M readout of Frady, Kleyko and Sommer (2018) to XOR-bound
  triples in one majority trace, and adds the accounting a knowledge graph needs:
  the exact tie rule of the reference implementation, several true answers per
  query, aliases from symmetric encoding, facts of weight 2 (symmetric twins), and
  the correlation between hops on one trace, −μ²/(1−μ²) per bit.
- We compared it numerically with your integral, given the same agreement
  probability: the two agree within 0.1 points on every configuration we test. The
  paper says plainly that the gain over asymptotic laws is yours, and that our
  contribution is the accounting.
- Eighteen predictions were preregistered (criteria and harnesses committed before
  any data), on synthetic memories and on FB15k-237 and WN18RR subgraphs. Three
  failed and are reported with their causes. Every number reruns from a clean clone
  with one script.

My two questions:

1. Is this accounting (ordered ties, aliases, weighted twins, hop dependence) a
   worthwhile refinement over your finite-M treatment, or is it already known in a
   form I have missed?
2. Facts that close even cycles in a knowledge graph are dependent over GF(2) (four
   facts on a rectangle XOR to the identity), and on a loaded biclique the
   independent model is several points optimistic. Is there a known treatment of
   such dependent bundles?

Paper, data and code: https://github.com/Kiatto/abm-runtime (paper in
`docs/paper.md`; the comparison with your integral is in §3.2; all results and how
to rerun them are listed in `BENCHMARKS.md`).

Thank you for your time. A short answer, or a pointer to prior work, would already
help a great deal.

Kind regards,
[nome e cognome]
