# Preregistrazione — catene profonde con codebook grande

**Data: 2026-09-28.** Committato insieme a
[`examples/deepchain2_prereg.py`](../../examples/deepchain2_prereg.py) **prima**
di misurare, a parte uno smoke test su 8 catene che non stampa accuratezze.
Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

La preregistrazione 7 ([`deepchain.md`](deepchain.md)) è **fallita**: con un
codebook di ~20 voci il recupero fuori percorso domina, e la previsione lo aveva
ignorato. La domanda originale è rimasta aperta: la dipendenza negativa fra hop
cresce con la profondità? Qui la si isola. Stesso disegno, con **1 000
distrattori** nel codebook: il recupero diventa trascurabile, e resta la
dipendenza.

**Cosa ho visto prima:** l'esito della preregistrazione 7, e le previsioni del
modello a D ∈ {256, 320, 384} per scegliere D. Nessuna misura di questo disegno.

## Disegno

N = 12 fatti per traccia in 12/h catene di h hop (entità distinte, relazioni
r₁ … r_h condivise, nessun alias); 1 000 distrattori nel codebook; D = 320;
8 000 tracce per h, entità mai usate. Il cleanup è vettoriale, con la stessa
semantica della reference, e l'harness verifica l'equivalenza con `Memory.chain`
sulla prima catena, dopo aver aggiunto davvero i distrattori.

## Previsioni

| h | M | modello (dipendenza) | Law V (p^h) | differenza | recupero, per eccesso | catene | SE misura |
|---|---|---|---|---|---|---|---|
| 1 | 1 025 | 0.7778 | 0.7778 | 0 | 0 | 96 000 | 0.0013 |
| 2 | 1 020 | 0.6011 | 0.6056 | −0.45 | +0.02 | 48 000 | 0.0022 |
| 3 | 1 019 | 0.4617 | 0.4714 | −0.97 | +0.06 | 32 000 | 0.0028 |
| 4 | 1 019 | 0.3513 | 0.3669 | −1.56 | +0.10 | 24 000 | 0.0031 |
| 6 | 1 020 | 0.1978 | 0.2221 | −2.43 | +0.18 | 16 000 | 0.0032 |

Il modello per h > 1 è stimato con 400 000 campioni. Il "recupero, per eccesso" è
la correzione di una catena di Markov con ritorno uniforme 1/M: la preregistrazione
7 ha mostrato che 1/M sovrastima il recupero vero, quindi qui è un limite superiore.

## Criteri, fissati ora

SE = √(SE_misura² + SE_MonteCarlo²).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il modello, per ogni h | \|misurato − modello\| ≤ 3 SE | > 5 SE per un qualsiasi h |
| **H2** — la Law V, catene profonde | a h = 4 **e** h = 6, misurato < p^h di oltre 3 SE | a h = 6, misurato ≥ p^h − 1 SE |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- Il modello tratta come indipendenti le distanze nulle dei diversi hop.
- Se questo test passa, dice che la dipendenza cresce con la profondità **quando
  il recupero è trascurabile**; non dà un modello per i codebook piccoli, dove la
  preregistrazione 7 è fallita.
