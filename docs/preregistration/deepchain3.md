# Preregistrazione — catene profonde, con la regola esatta dei pareggi

**Data: 2026-09-28.** Committato insieme a
[`examples/deepchain3_prereg.py`](../../examples/deepchain3_prereg.py) **prima**
di misurare, a parte uno smoke test su 8 catene che non stampa accuratezze.
Stesse regole di [`fb15k237.md`](fb15k237.md).

## Storia, dichiarata per intero

- La preregistrazione 7 ([`deepchain.md`](deepchain.md)) è **fallita**: con un
  codebook minuscolo il recupero fuori percorso domina.
- La preregistrazione 8 ([`deepchain2.md`](deepchain2.md)) è **fallita** già al
  singolo hop: il modello divideva i pareggi a metà, mentre la reference sceglie il
  primo codeword inserito, e la risposta giusta precede i distrattori.
- **Esplorativo, sui dati della preregistrazione 8, già visti:** con la regola
  esatta dei pareggi (`win_ordered`: P(vince | d) = P(nullo > d)^prima ·
  P(nullo ≥ d)^dopo, esatta date le distanze nulle indipendenti) il modello con la
  dipendenza coincide con la misura entro 1.4 SE a ogni h, mentre la Law V, anche
  con la p corretta, sbaglia di −3.1 / −5.2 / −5.4 SE a h = 3 / 4 / 6.

Quell'accordo è un adattamento a posteriori di una regola, non una prova. Questa
preregistrazione lo mette alla prova su dati nuovi.

## Disegno

Quello della preregistrazione 8, con entità nuove (prefisso `p9`) e **due**
configurazioni invece di una: N = 12 fatti in 12/h catene di h hop, 1 000
distrattori inseriti dopo gli item della memoria, D ∈ {256, 320}, 8 000 tracce per
h e per D, catene interrogate come in `Memory.chain` (cleanup vettoriale verificato
contro la reference sulla prima catena).

## Previsioni

Modello: dipendenza fra hop (`chain_accuracy_mc`) con la regola esatta dei
pareggi, calcolata per ogni catena dalla posizione dei suoi bersagli nel codebook
e mediata sulle catene. Law V: prodotto delle accuratezze esatte per hop.

| D | h | modello | Law V | differenza |
|---|---|---|---|---|
| 256 | 1 | 0.6647 | 0.6647 | 0 |
| 256 | 2 | 0.4362 | 0.4426 | −0.65 |
| 256 | 3 | 0.2827 | 0.2946 | −1.19 |
| 256 | 4 | 0.1784 | 0.1960 | −1.76 |
| 256 | 6 | 0.0677 | 0.0867 | −1.90 |
| 320 | 1 | 0.7952 | 0.7952 | 0 |
| 320 | 2 | 0.6295 | 0.6331 | −0.37 |
| 320 | 3 | 0.4944 | 0.5039 | −0.95 |
| 320 | 4 | 0.3860 | 0.4010 | −1.49 |
| 320 | 6 | 0.2299 | 0.2537 | −2.38 |

## Criteri, fissati ora

SE = √(SE_misura² + SE_MonteCarlo²).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il modello, per ogni D e h | \|misurato − modello\| ≤ 3 SE | > 5 SE per una qualsiasi cella |
| **H2** — la Law V, catene profonde | per ciascuna D, a h = 4 **e** h = 6, misurato < Law V di oltre 3 SE | per una qualsiasi D, a h = 6, misurato ≥ Law V − 1 SE |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- Il modello tratta come indipendenti le distanze nulle dei diversi hop.
- Il recupero fuori percorso è trascurato: con 1 000 distrattori, la stima per
  eccesso è sotto 0.2 punti (preregistrazione 8).
- Il modello non vale ancora per codebook piccoli (preregistrazione 7).

---

## Esito — 2026-09-28, eseguito dopo il commit `794def8`

Previsioni e criteri **non modificati**. Risultati:
[`results/deepchain3_prereg_results.json`](../../results/deepchain3_prereg_results.json).

| D | h | catene | misurato | modello | scarto | Law V | scarto |
|---|---|---|---|---|---|---|---|
| 256 | 1 | 96 000 | 0.6638 | 0.6647 | −0.6 SE | 0.6647 | −0.6 SE |
| 256 | 2 | 48 000 | 0.4356 | 0.4362 | −0.3 SE | 0.4426 | −3.0 SE |
| 256 | 3 | 32 000 | 0.2859 | 0.2827 | +1.2 SE | 0.2946 | −3.4 SE |
| 256 | 4 | 24 000 | 0.1797 | 0.1784 | +0.5 SE | 0.1960 | **−6.4 SE** |
| 256 | 6 | 16 000 | 0.0691 | 0.0677 | +0.7 SE | 0.0867 | **−8.5 SE** |
| 320 | 1 | 96 000 | 0.7938 | 0.7952 | −1.1 SE | 0.7952 | −1.1 SE |
| 320 | 2 | 48 000 | 0.6288 | 0.6295 | −0.3 SE | 0.6331 | −1.9 SE |
| 320 | 3 | 32 000 | 0.4948 | 0.4944 | +0.2 SE | 0.5039 | −3.2 SE |
| 320 | 4 | 24 000 | 0.3851 | 0.3860 | −0.3 SE | 0.4010 | **−4.9 SE** |
| 320 | 6 | 16 000 | 0.2390 | 0.2299 | +2.6 SE | 0.2537 | **−4.3 SE** |

**H1 — sostenuta**: tutte le 10 celle entro 3 SE (massimo +2.6).
**H2 — sostenuta**: in entrambe le D, a h = 4 e h = 6 il misurato è sotto la Law V
di oltre 3 SE.

### Cosa dice

La dipendenza negativa fra hop sulla stessa traccia **cresce con la profondità**,
della quantità che il modello per bit prevede, una volta che la regola dei pareggi
è quella vera della reference. È la stessa previsione della preregistrazione 7;
ci sono voluti due fallimenti, con le loro cause (recupero fuori percorso, regola
dei pareggi), per metterla alla prova in un disegno che poteva rispondere.

### Cosa non dice

- Con codebook piccoli il modello non vale: il recupero fuori percorso non è
  modellato (preregistrazione 7).
- La regola esatta dei pareggi è stata introdotta dopo aver visto la
  preregistrazione 8; questa è la sua prima prova su dati nuovi, e l'ha superata.
