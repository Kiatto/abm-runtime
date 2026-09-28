# Preregistrazione — il contratto usato per scegliere D prima

**Data: 2026-09-28.** Committato insieme a
[`examples/sizing_prereg.py`](../../examples/sizing_prereg.py) **prima** di
misurare, a parte uno smoke test con N = 30 che stampa solo il numero di query e la
dimensione scelta. Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

Le preregistrazioni precedenti confrontano previsioni e misure. La tesi del paper
però è pratica: *l'accuratezza si può dichiarare prima del deployment*. Questo
test usa il modello come lo userebbe qualcuno: per un sottografo nuovo, scegliere
**la dimensione minima** che promette un'accuratezza obiettivo, e poi verificare
che la promessa regga.

**Un risultato emerso preparandolo, dichiarato qui:** con l'encoding simmetrico
un alias a pari segnale limita l'accuratezza di una query a g/(g + a), qualunque
sia D. Su 5 sottografi densi di WN18RR con N = 300 il tetto medio è 0.934 (l'11.4%
delle query ha un alias), su FB15k-237 0.989. Un contratto deve dichiararlo; per
un obiettivo sopra il tetto la dimensione giusta non esiste. **Ho visto:** solo
questi tetti, calcolati dalla struttura, e le previsioni di un sottografo usato
per stimare i tempi. **Non ho visto** nessuna misura.

## Disegno

- 20 sottografi densi **nuovi** (seed da 5000) per ciascun grafo (FB15k-237,
  WN18RR) e ciascun N ∈ {150, 300}: 80 sottografi.
- Obiettivi T ∈ {0.70, 0.80}, sotto i tetti.
- Per ogni sottografo e obiettivo, due dimensionamenti:
  - **esatto**: la D minima (multiplo di 64, al più 32 768) con accuratezza
    prevista ≥ T dal modello con gemelli e alias;
  - **Law IV**: la stessa scelta con la Law IV, k = 0.92, senza alias né gemelli.
  Poi si misura l'accuratezza alla D scelta, con la reference.
- **Tetto**: per ogni sottografo, previsione e misura a D = 16 384.

## Criteri, fissati ora

Differenze in punti. "Gruppo" = (grafo, N, T), 20 sottografi.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il dimensionamento esatto mantiene la promessa | in ogni gruppo, media di (misurato − T) ≥ −1.5, **e** al più 2 sottografi su 20 sotto T − 6 | media di (misurato − T) < −4 in un qualsiasi gruppo |
| **H2** — conta usare il modello esatto | su WN18RR, la media di \|misurato − T\| con il dimensionamento esatto è minore che con la Law IV, per ogni N e T | è maggiore per un qualsiasi gruppo di WN18RR |
| **H3** — il tetto | su WN18RR, a D = 16 384, misurato ≤ tetto + 2 in almeno 36 sottografi su 40, **e** \|misurato − previsto\| medio ≤ 2 | misurato > tetto + 4 in più di 4 sottografi |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- 200 query per sottografo: il rumore di campionamento su un singolo sottografo è
  circa 3 punti a T = 0.8, da cui la soglia di 6 punti per "mancato di molto".
- La D scelta è un multiplo di 64: la previsione alla D scelta supera T di poco,
  non esattamente.
- Il tetto calcolato assume che fra candidati a pari segnale vinca uno a caso; la
  regola vera della reference dipende dall'ordine di inserimento.
