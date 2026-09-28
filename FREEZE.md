# FREEZE — 2026-07-16

Da questa data sono **definitivamente congelati**:

- l'algebra (operatori: bind, bundle, permute, cleanup, projection)
- il formalismo ([docs/FORMALISM.md](docs/FORMALISM.md) v2.1)
- il paper ([docs/paper.md](docs/paper.md) v1.2 — era v1.1; vedi la nota sotto)
- la reference implementation ([reference/abm.py](reference/abm.py) v1.0.0)

**Nota del 2026-09-28 — perché il paper è passato a v1.2.** Il congelamento è
stato riaperto una volta, in preparazione alla pubblicazione, e solo per
documentare risultati già esistenti nel repo: il pilota su documento reale
(presentato come evidenza della Law VIII, *non* della Law IV, dato che girava
sotto il 5% della capacità), le falsificazioni F4 e F5 del livello
implementativo, la bibliografia (prima assente) e un errore di versione nel piè
di pagina. Nessuna nuova Law, teorema, assioma od operatore. Dalla v1.2 in poi
vale di nuovo la regola sotto.

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
