# Preregistrazione — encoding simmetrico contro asimmetrico

**Data: 2026-09-28.** Committato insieme a
[`examples/asymmetric_prereg.py`](../../examples/asymmetric_prereg.py) **prima**
di misurare. Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

L'encoding della reference, s ⊕ ρ(r) ⊕ o, è simmetrico in s e o. Ne seguono due
effetti, entrambi già messi alla prova: gli **alias** (un fatto (x, r, s) risponde
a (s, r) con x, allo stesso segnale della risposta giusta) e i **gemelli** ((s, r, o)
e (o, r, s) sono lo stesso vettore, un fatto di peso 2). La domanda ovvia di un
reviewer: perché non un encoding asimmetrico,

    s ⊕ ρ(r) ⊕ ρ²(o),   decodifica: cleanup(ρ⁻²(T ⊕ s ⊕ ρ(r))),

che non ha né l'uno né l'altro, al costo di perdere le query inverse gratuite?

Il modello esatto prevede l'accuratezza di entrambi senza parametri, e la sua
previsione non è ovvia: la simmetria **danneggia** a basso carico, dove l'alias è
l'errore dominante, e **aiuta** ad alto carico, dove fondere i gemelli in un solo
fatto riduce il rumore. Il segno della differenza si inverte con il carico.

## Disegno

- Sottografi densi (la stessa visita in ampiezza delle preregistrazioni 2 e 4) di
  FB15k-237 e WN18RR, seed **20–29**, mai usati.
- D = 2048 con N ∈ {100, 200, 300, 400}; D = 8192 con N ∈ {400, 800, 1200, 1600}.
- Stesse triple e stesse query per i due encoding; al più 200 query per cella e
  seed; una risposta è corretta se è uno qualsiasi degli oggetti veri.
- Simmetrico: la reference congelata. Asimmetrico: una sottoclasse nell'harness
  che cambia solo `fact_hv` e la decodifica.

**Previsioni** — il modello con i gemelli e gli alias per il simmetrico, fatti
indipendenti e nessun alias per l'asimmetrico. Calcolate con `--predict`, prima
di qualsiasi misura, sulle stesse triple che verranno misurate:

| grafo | D | N | simmetrico | asimmetrico | differenza |
|---|---|---|---|---|---|
| FB15k-237 | 2048 | 100 | 84.3 | 86.2 | −1.9 |
| | | 200 | 49.3 | 49.5 | −0.2 |
| | | 300 | 29.2 | 26.9 | +2.3 |
| | | 400 | 16.5 | 16.0 | +0.5 |
| | 8192 | 400 | 74.2 | 74.7 | −0.5 |
| | | 800 | 39.2 | 38.6 | +0.5 |
| | | 1200 | 18.4 | 17.0 | +1.4 |
| | | 1600 | 9.0 | 8.5 | +0.4 |
| WN18RR | 2048 | 100 | 80.4 | 85.6 | **−5.2** |
| | | 200 | 51.0 | 49.8 | +1.2 |
| | | 300 | 33.9 | 27.7 | **+6.2** |
| | | 400 | 20.0 | 16.7 | **+3.3** |
| | 8192 | 400 | 69.5 | 75.7 | **−6.1** |
| | | 800 | 39.5 | 34.5 | **+5.0** |
| | | 1200 | 23.4 | 16.4 | **+6.9** |
| | | 1600 | 15.8 | 9.7 | **+6.0** |

## Criteri, fissati ora

Errori come previsto − misurato, in punti, sulle celle mediate sui seed.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — asimmetrico, per grafo e D | errore medio assoluto ≤ 2 **e** con segno entro ±1.5 | > 5, **o** con segno oltre ±3 |
| **H2** — simmetrico, per grafo e D | errore medio assoluto ≤ 2.5 | > 5 |
| **H3** — l'inversione su WN18RR | in ciascuna D, differenza misurata **negativa** al carico più basso e **positiva** ai due più alti | segno sbagliato in 2 o più di queste 6 celle |
| **H4** — l'entità della differenza, WN18RR | errore medio assoluto sulla differenza ≤ 2.5 per D | > 5 |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- Su FB15k-237 le differenze previste sono piccole (≤ 2.3 punti), vicine al rumore:
  lì il test è sulla precisione di H1 e H2, non sull'inversione.
- L'asimmetrico perde le query inverse gratuite; questo test non ne misura il
  costo, che dipende dall'uso.
