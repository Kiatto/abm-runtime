# Preregistrazione — la dipendenza fra hop nelle catene profonde

**Data: 2026-09-28.** Committato insieme a
[`examples/deepchain_prereg.py`](../../examples/deepchain_prereg.py) **prima** di
misurare, a parte uno smoke test su 8 catene che non stampa accuratezze. Stesse
regole di [`fb15k237.md`](fb15k237.md).

## Perché

La preregistrazione 3 ha mostrato che due hop sulla stessa traccia sono correlati
negativamente, e ha respinto la Law V (Acc = p^h) come legge esatta a due hop. Un
reviewer chiederebbe se conta nelle catene profonde. Il modello per bit
(`chain_accuracy_mc`: i bit dei fatti della catena ±1 uniformi e indipendenti,
gli altri fatti una binomiale, la traccia il segno del totale; valutato per
Monte Carlo, senza memoria né codeword) prevede che la violazione **cresca con
la profondità**. È il problema aperto 3 del paper.

**Cosa ho visto prima:** solo previsioni del modello, per scegliere N e D. Il
modello Monte Carlo coincide con la formula esatta a due hop (0.2189 contro
0.2189 a N = 10, D = 48; 0.2576 contro 0.2573 a N = 30, D = 240).

## Disegno

N = 12 fatti per traccia, organizzati in 12/h catene di h hop con entità
distinte e le relazioni r₁ … r_h condivise fra le catene (nessun alias per
costruzione). D = 128. 6 000 tracce per ogni h; ogni catena interrogata con
`Memory.chain` della reference congelata, partendo dalla prima entità.

## Previsioni

| h | M | modello | Law V (p^h) | differenza | catene | SE misura | differenza / SE |
|---|---|---|---|---|---|---|---|
| 1 | 25 | 0.7095 (esatto) | 0.7095 | 0 | 72 000 | 0.0017 | — |
| 2 | 20 | 0.5414 | 0.5460 | −0.46 | 36 000 | 0.0026 | 1.7 |
| 3 | 19 | 0.4047 | 0.4144 | −0.97 | 24 000 | 0.0032 | 3.1 |
| 4 | 19 | 0.2943 | 0.3089 | −1.47 | 18 000 | 0.0034 | 4.3 |
| 6 | 20 | 0.1429 | 0.1627 | −1.99 | 12 000 | 0.0032 | 6.2 |

Il modello per h > 1 è stimato con 400 000 campioni (errore Monte Carlo
0.0006–0.0008), incluso nel confronto.

## Criteri, fissati ora

SE = √(SE_misura² + SE_MonteCarlo²).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il modello, per ogni h | \|misurato − modello\| ≤ 3 SE | > 5 SE per un qualsiasi h |
| **H2** — la Law V nelle catene profonde | a h = 4 **e** h = 6, misurato < p^h di oltre 3 SE | a h = 6, misurato ≥ p^h − 1 SE |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- Il modello tratta come indipendenti le distanze nulle dei diversi hop.
- A h ≤ 2 la differenza con la Law V è sotto i 2 SE: lì il test riguarda solo H1.
- Le catene hanno entità distinte; in un grafo con cicli i bit dei fatti non sono
  più indipendenti, e il modello non si applica così com'è.
