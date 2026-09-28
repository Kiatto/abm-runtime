# Preregistrazione — il contratto esatto su configurazioni mai misurate

**Data: 2026-09-28.** Committato insieme a
[`examples/exact_prereg.py`](../../examples/exact_prereg.py) **prima** di
qualsiasi esecuzione, a parte uno smoke test con N = 20 che stampa solo numero
di fatti, codebook e query. La regola è quella di
[`fb15k237.md`](fb15k237.md): previsioni, griglia e criteri non si toccano dopo
aver visto i risultati, e l'esito si registra anche se è negativo.

## Cosa si mette alla prova

[`bsm/memory/exact_contract.py`](../../bsm/memory/exact_contract.py) calcola
l'accuratezza di cleanup dagli assiomi, con distribuzioni esatte e **nessun
parametro**. Sui dati già pubblicati sta dentro gli intervalli di confidenza a
ogni D, ma quei dati li avevo già visti: erano controlli di coerenza, non un
test. Qui le configurazioni sono nuove.

**Cosa ho visto prima di scrivere questo file:** le previsioni del modello
esatto a D = 16384 (N\* ≈ 900 con M = 2N + 13), usate per scegliere la griglia;
una simulazione Monte Carlo a D ≤ 512, già nei test del modulo; lo smoke test.
**Non ho visto** nessuna accuratezza misurata a D = 16384, con più oggetti veri
per query, o su sottografi densi di FB15k-237.

## Le tre parti

**A — una dimensione mai usata.** D = 16384, fatti sintetici uniformi (13
relazioni a rotazione, come `capacity_contract.py`),
N ∈ {400, 600, 900, 1200, 1600, 2200}. È fuori da tutto il range misurato
finora (D ≤ 8192).

**B — l'unica ipotesi non esatta del modello.** Il modello tratta come
indipendenti più candidati a pari segnale. Qui ogni (s, r) ha g ∈ {1, 2, 4}
oggetti veri, con D = 4096 e N ∈ {100, 200, 300}; una risposta è corretta se è uno
qualsiasi degli oggetti veri.

**C — FB15k-237 a sottografi densi.** Il primo test preregistrato campionava
triple uniformi, quindi il grafo risultava rado, gli alias erano praticamente
assenti (fattore 0.994–1.000) e gli hub erano pochi. Qui il campione è una
visita in ampiezza non orientata da un'entità casuale, che raccoglie le triple
incidenti finché non ne ha N: le entità si ripetono, e con loro risposte
multiple e alias. Stessa griglia del primo test: D = 2048 con
N ∈ {50 … 400}, D = 8192 con N ∈ {200 … 1600}.

In tutte le parti: 10 seed per cella, al più 200 query per cella e seed, la
reference congelata, e un cleanup vettoriale verificato contro `Memory.query`.

## Previsione

Per ogni query, `cleanup_accuracy(N, D, M, correct = g, aliases = a)`, con M il
codebook (entità più relazioni), g gli oggetti veri di (s, r) e a gli x non tra
quelli con (x, r, s) memorizzato. Media per cella, poi sui seed.

Come confronto, riportato ma non come criterio: il predittore del primo test,
Law IV con k = 0.92 per g / (g + a).

## Criteri, fissati ora

Errori in punti percentuali, calcolati come previsto − misurato sulle celle
mediate sui seed. Sono più stretti del primo test perché l'affermazione è più
forte: zero parametri.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — A, D = 16384 | errore medio assoluto ≤ 2 **e** con segno entro ±1.5 | medio assoluto > 5, **o** con segno oltre ±3 |
| **H2** — B, per ciascun g | errore medio assoluto ≤ 3 per ogni g | > 6 per un qualsiasi g |
| **H3** — C, per ciascuna D | errore medio assoluto ≤ 3 **e** con segno entro ±2 | medio assoluto > 6, **o** con segno oltre ±4 |
| **H4** — C, confronto | l'errore medio assoluto del modello esatto è minore di quello della Law IV con k = 0.92, sulle due D insieme | il modello esatto sbaglia di più |
| **H5** — C, hub | \|(misurato − previsto)_hub − (misurato − previsto)_resto\| ≤ 3, pesato per query | > 8 |

Tutto ciò che non ricade in "sostenuta" o in "falsificata" è **sostenuto in
parte**, e va scritto così.

## Limiti dichiarati prima

- La visita in ampiezza favorisce le entità ad alto grado vicine all'origine:
  è un campione denso, non un campione rappresentativo del grafo.
- In B e C il modello assume indipendenza fra candidati a pari segnale: B serve
  proprio a misurare quanto costa questa ipotesi.
- Collisioni del seed a 32 bit: con M ≤ ~4500, probabilità ≈ 0,2%.

---

## Esito — 2026-09-28, eseguito dopo il commit `0856406`

Previsioni e criteri qui sopra **non sono stati modificati**. Risultati:
[`results/exact_prereg_results.json`](../../results/exact_prereg_results.json).
Errori come previsto − misurato, in punti.

| ipotesi | modello esatto | Law IV, k = 0.92 (confronto) | esito |
|---|---|---|---|
| H1 — A, D = 16384 | 0.27, con segno −0.09 | 0.69, −0.58 | **sostenuta** |
| H2 — B, g = 1 | 0.58 | 0.60 | **sostenuta** |
| H2 — B, g = 2 | 0.86 | 13.1 | **sostenuta** |
| H2 — B, g = 4 | 0.32 | 21.1 | **sostenuta** |
| H3 — C, D = 8192 | 1.06, con segno −1.06 | 2.79, −2.79 | **sostenuta** |
| H3 — C, D = 2048 | 2.88, con segno **−2.61** | 4.56, −4.32 | **sostenuta in parte**: il bias con segno supera ±2, non ±4 |
| H4 — C, esatto contro Law IV | 1.97 | 3.68 | **sostenuta** |
| H5 — C, hub | differenza +3.28 | — | **né sostenuta né falsificata** (soglie 3 e 8) |

### Cosa dice

- **Zero parametri reggono in una dimensione mai misurata**: a D = 16384 l'errore
  medio è 0.27 punti.
- **L'ipotesi di indipendenza fra candidati a pari segnale costa meno di un
  punto** (parte B), mentre ignorare i candidati multipli, come fa la Law IV,
  costa 13–21 punti.
- Sui sottografi densi reali il modello esatto sbaglia meno della metà della
  Law IV.

### Cosa non torna

- **Sui sottografi densi a D = 2048 il modello è pessimista**, soprattutto ad
  alto carico: a N = 400 misurato 27.6, previsto 19.3; a N = 200 53.5 contro
  49.6. La direzione è la stessa degli hub (+3.28). Qualcosa nella struttura di
  un grafo reale denso rende le query più facili di quanto prevedano le
  distribuzioni esatte per fatti indipendenti. **Non è spiegato**; non va
  corretto a posteriori, va capito e messo alla prova con una nuova
  preregistrazione.
