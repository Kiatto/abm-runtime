# Preregistrazione 18 — ABM contro uno store esatto e un filtro di Bloom, a pari bit

Committata **prima** di misurare, insieme all'harness `examples/equal_bits_prereg.py`.
Chiude un'affermazione del §7 non misurata ("we have not compared them with a Bloom
or cuckoo filter or a hash table of the same size") e il crossover di bit per fatto
ritirato dopo l'audit del 2026-09-30 (nessuno script lo riproduceva).

## Impianto

**A — recupero (s, r) → o, a pari bit.** Fatti (s_i, r_{i mod 13}, o_i), 2N entità e
13 relazioni (il protocollo del contratto di capacità); 3 seed per cella.
- **ABM:** una traccia di D bit, reference congelata; accuratezza sulle N query.
- **Store esatto ideale:** ogni fatto costa b = 2·⌈log₂ 2N⌉ + ⌈log₂ 13⌉ bit, **senza**
  sovraccarico (nessuna tabella reale fa così bene: è un confronto a favore dello
  store); con D bit tiene ⌊D/b⌋ fatti e perde gli altri; accuratezza min(1, ⌊D/b⌋/N).
- Griglia: D = 2048 con N ∈ {50, 100, 150, 200, 300, 400, 600}; D = 8192 con
  N ∈ {200, 400, 600, 800, 1200, 1600, 2400}.

**B — appartenenza, a pari bit.** ABM: `Memory.member` con z ≥ 3. Bloom: D bit,
k = max(1, round(D/N · ln 2)) funzioni di hash (sha256). Falsi negativi sugli N fatti,
falsi positivi su 1 000 fatti mai memorizzati dello stesso vocabolario
(s_i, r_{i mod 13}, o_j), j ≠ i.

**Descrittivo, non un'ipotesi:** D diviso il minimo di Fano per rispondere a N query
con l'accuratezza misurata scegliendo fra 2N oggetti.

## Previsioni del modello esatto, calcolate ora (`--predict`)

| D | N | ABM previsto | store esatto ideale |
|---|---|---|---|
| 2048 | 50 | 0.990 | 1.000 |
| | 100 | 0.783 | 1.000 |
| | 150 | 0.522 | 0.620 |
| | 200 | 0.346 | 0.465 |
| | 300 | 0.170 | 0.283 |
| | 400 | 0.096 | 0.212 |
| | 600 | 0.042 | 0.130 |
| 8192 | 200 | 0.976 | 1.000 |
| | 400 | 0.658 | 0.853 |
| | 600 | 0.372 | 0.525 |
| | 800 | 0.218 | 0.394 |
| | 1200 | 0.091 | 0.243 |
| | 1600 | 0.047 | 0.182 |
| | 2400 | 0.018 | 0.114 |

Il modello prevede **nessun crossover**: a ogni carico lo store esatto ideale batte
ABM a pari bit, anche quando entrambi sono sovraccarichi.

## Ipotesi e criteri, fissati ora

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — recupero: in ogni cella, accuratezza misurata di ABM < store esatto ideale | 14 celle su 14 | ABM ≥ store in almeno una cella |
| **H2** — il modello esatto prevede la misura di ABM | errore medio assoluto ≤ 2 punti per D | > 5 punti per una D |
| **H3** — appartenenza: errore totale (falsi negativi + falsi positivi) del Bloom < quello di ABM | in ogni cella | Bloom ≥ ABM in più di una cella |

Il resto è **sostenuto in parte**.

**Previsioni (non criteri).** H1, H2 e H3 sostenute. ABM non conviene mai bit per bit
nel recupero di un salto né nell'appartenenza; quello che offre, se offre qualcosa,
è altrove (composizione algebrica, degradazione prevedibile), e questo test non lo
misura.

## Cosa ho visto prima

- Solo le previsioni del modello qui sopra. Nessuna misura di questo protocollo con
  queste dimensioni e questo confronto.

## Limiti dichiarati prima

- Lo store esatto è idealizzato (zero sovraccarico, chiavi implicite): a favore dello
  store. Una tabella reale ha un fattore di riempimento e impronte delle chiavi.
- Un solo protocollo sintetico (fatti distinti, una relazione ogni 13); i grafi reali
  hanno alias e gemelli.
- Il Bloom risponde solo all'appartenenza; ABM risponde anche a (s, r) → o. Il
  confronto B è sulla sola appartenenza.
