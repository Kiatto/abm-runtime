# FREEZE — 2026-07-16

Da questa data sono **definitivamente congelati**:

- l'algebra (operatori: bind, bundle, permute, cleanup, projection)
- il formalismo ([docs/FORMALISM.md](docs/FORMALISM.md) v2.1)
- il paper ([docs/paper.md](docs/paper.md) v1.5 — era v1.1; vedi le note sotto)
- la reference implementation ([reference/abm.py](reference/abm.py) v1.0.0)

**Nota del 2026-09-28 — perché il paper è passato a v1.2.** Il congelamento è
stato riaperto una volta, in preparazione alla pubblicazione, e solo per
documentare risultati già esistenti nel repo: il pilota su documento reale
(presentato come evidenza della Law VIII, *non* della Law IV, dato che girava
sotto il 5% della capacità), le falsificazioni F4 e F5 del livello
implementativo, la bibliografia (prima assente) e un errore di versione nel piè
di pagina. Nessuna nuova Law, teorema, assioma od operatore.

**Nota del 2026-09-28 — v1.3, dopo una review ostile.** Solo correzioni: tolte le
affermazioni che le sezioni non sostenevano ("zero parametri" per N\*, "teorema" per
una composizione che assume l'indipendenza, il confronto con una crescita lineare
che nessuno propone, l'1.7% nell'abstract), aggiunti i precedenti diretti
(Clarkson et al. 2023 su MAP-B) che ridimensionano la novità della Law IV, inserite
le figure che esistevano già, e reso riproducibile k da uno script. Abstract entro
il limite di arXiv. Nessuna nuova Law, teorema, assioma od operatore.

**Nota del 2026-09-28 — v1.4, riaperta da kiatto** per i punti A e B della review:
riesecuzioni a 10 seed degli esperimenti sintetici con 3 seed, un test
preregistrato su un knowledge graph reale (FB15k-237), la baseline con insieme
esatto su ProofWriter e il confronto quantitativo con Clarkson et al. Nessuna
nuova Law, teorema, assioma od operatore; i risultati vecchi restano in results/
accanto ai nuovi (suffisso _s10). Dalla v1.4 in poi vale di nuovo la regola sotto.

**Nota del 2026-09-28 — v1.5, mandato di kiatto "ad oltranza fino a 9/10".** Il
centro del paper cambia: una teoria esatta a D finito senza parametri
(`bsm/memory/exact_contract.py`), derivata dagli assiomi esistenti, e cinque
preregistrazioni (`docs/preregistration/`). La Law V è falsificata come legge
esatta, con una violazione prevista in anticipo; la Law IV resta come
approssimazione asintotica. Nessun assioma nuovo. Finché vale il mandato, ogni
modifica sostanziale al paper richiede una preregistrazione.

**Modifiche consentite:** correzioni di bug, performance, documentazione,
packaging. Nient'altro: niente nuove Law, teoremi, assiomi, operatori,
estensioni del calculus, benchmark sintetici interni.

**Criterio di riapertura:** contraddizioni prodotte da benchmark
industriali reali o da utenti esterni. Nessun'altra ragione riapre il
formalismo.

**Regola di priorità per ogni attività:** deve aumentare almeno uno tra
adozione, evidenza esterna, maturità del prodotto, credibilità.
Altrimenti si rimanda.

Razionale: il collo di bottiglia del progetto non è più la teoria
(coerenza interna ~9/10) ma il trasferimento (evidenza esterna ~4.5/10,
prodotto ~6.5/10). Vedi docs/phase1b_report.md e la due diligence del
16-07-2026.
