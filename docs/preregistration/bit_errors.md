# Preregistrazione 20 — ABM contro store esatti con codice correttore, sotto errori di bit

**Data: 2026-10-09; revisione del disegno 2026-10-12, prima dei dati.** Committata **prima** di misurare, insieme all'harness
[`examples/bit_errors_prereg.py`](../../examples/bit_errors_prereg.py). Prima del
commit l'harness è stato eseguito solo con `--smoke` (grafo sintetico generato nello
script, mai FB15k-237 o WN18RR) e con `--predict` (legge i dati reali, costruisce le
strutture degli store per sapere quante chiavi tengono e ne simula il rumore per la
previsione dei codici; **non** costruisce nessuna memoria ABM sui dati
reali). Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

I test 18 ([`equal_bits.md`](equal_bits.md)) e 19 ([`algebra.md`](algebra.md)) hanno
mostrato che a pari bit store esatto, Bloom e dict + join battono ABM ovunque, su
memoria **perfetta**. Resta non testato il perimetro hardware: memorie che sbagliano
bit durante la conservazione (memorie analogiche, in-memory computing, celle
multilivello). L'argomento classico per le rappresentazioni distribuite è che
degradano con grazia; uno store esatto, invece, perde la chiave intera a un solo bit
sbagliato, a meno di spendere bit in un codice correttore. Qui si misura chi vince, a
pari bit totali, quando il codice lo sceglie l'avversario al meglio.

## Revisione prima dei dati (2026-10-12)

La prima versione (commit `99ccc03`) usava come avversario solo ripetizione e Hamming
SECDED e prevedeva vittorie di ABM in 21 celle. Una revisione ostile, prima di
qualunque misura, ha chiesto un avversario moderno; con quello **il modello non
prevede più nessuna vittoria di ABM**. Le modifiche, punto per punto:

- **A1 avversario moderno.** Codici polari con decodifica SC (min-sum, costruzione
  di Bhattacharyya per BSC(ε)), N_blocco ∈ {256, 512, 1024, 2048}; BCH binari
  primitivi (libreria `galois`, dipendenza opzionale `[replicate]`), n ∈ {63, 127,
  255, 511, 1023}, ogni t possibile; concatenazione ripetizione-3 interna + polare
  o BCH esterno per ε ≥ 0.03. Ripetizione e Hamming SECDED tolti: la ripetizione resta
  come r/r+2 misti; l'Hamming è dominato dai BCH con t = 1.
- **A2 tutti i D bit.** Ripetizione mista r/r+2; polari accorciati al tasso esatto
  (bit finali noti a zero, u corrispondenti congelati), K = ⌈L/blocchi⌉ invece della
  griglia 1/64 (più fine); BCH accorciati (resta inutilizzato meno di un blocco di
  parità). "Meno chiavi, codice più forte": c ∈ {1, 0.8, 0.64, 0.5, 0.4, 0.3, 0.2,
  0.12} × c_max.
- **A3 layout a lettura singola.** MPHF hash-and-displace (CHD semplificato, 4 chiavi
  per bucket, spostamenti a larghezza fissa) + array di valori, con la MPHF nella
  memoria rumorosa e codificata, bit contati in D. La regola sceglie fra
  {XOR, MPHF} × codice × c. Nelle previsioni vince quasi sempre XOR: con tabella
  minima gli spostamenti costano più dell'overhead 0.23·le dello XOR.
- **A4 flip comune ai candidati.** Smoke su tracce ABM vere (sintetiche, D = 1024,
  N = 60): concordanza del vero misurata 0.5468 / 0.5377 / 0.5280 contro p′ previsto
  0.5513 / 0.5410 / 0.5308 a ε = 0 / 0.1 / 0.2; correlazioni **entro la domanda** al
  variare del solo rumore: vero–nullo +0.008 / +0.010, nullo–nullo −0.058 / −0.008 a
  ε = 0.1 / 0.2. Il flip è comune, ma i codeword nulli sono indipendenti dal vettore
  rumoroso, quindi l'approssimazione d'indipendenza del modello non peggiora con ε.
  Il controllo tautologico (p′ su bit simulati) è stato tolto.
- **A5 inferenza.** R = 20 rumori indipendenti per memoria e per ε; unità = seed
  (5): per seed la differenza media ABM − avversario su rumori e domande; test t
  one-sided con 4 gradi di libertà. Famiglia confermativa = celle con differenza
  prevista > MDE (MDE = (t₄(1−α/m) + t₄(0.8))·sd/√5, sd = deviazione fra seed delle
  differenze previste più la varianza binomiale entro seed, iterato su m); Holm
  α = 0.05.
- **A6 ristima.** Dopo la scelta, l'accuratezza dell'avversario è ristimata con un
  terzo flusso RNG (50 rumori, simulazione completa: codifica, rumore, decodifica,
  lettura vera); H4 usa la ristima. La previsione di scelta ha un piccolo winner's
  curse (fino a 4 punti a D = 2048, N = 50, ε = 0.2: 0.837 contro 0.801).
- **B1** tetto descrittivo PPV (approssimazione normale di Polyanskiy–Poor–Verdú, un
  codice lungo D, overhead 1.0: solo i bit di valore) al posto di "Shannon".
- **B2** "memoria" = solo lo stato che dipende dai fatti (traccia ABM; celle e MPHF
  dello store). Descrittivo: ABM con rumore anche sul codebook (5 rumori), e il
  fattore (1−ε)^(8·lunghezza del nome) per i nomi, identico per i due sistemi.
- **B3** indice non valido → l'oggetto più frequente del campione (gratis); riportata
  anche l'accuratezza selettiva dello store (solo letture con indice valido).
- **B4** ε* valutato solo nelle serie in cui la differenza prevista al cambio di segno
  supera l'MDE (o, per "mai", resta sotto −MDE a ogni ε).
- **B5** H0 è un controllo, non una conferma.
- **B6** raffiche: interlacciatore a blocchi (posizione nel blocco, poi blocco) sul
  codice scelto.
- **B7** `SeedSequence.spawn`: un sotto-flusso per ABM, misura dello store, scelta,
  ristima, `none`, raffiche, codebook.
- Errore trovato e corretto durante la revisione: nella decodifica SC i pareggi
  (LLR = 0, frequenti su un BSC) andavano tutti a 0, il che rende ottimista la
  simulazione con la parola nulla (tasso d'errore di blocco 0.07 contro 0.48 vero in
  un caso). Ora un rumore minuscolo, fisso e simmetrico rompe i pareggi; banco e
  simulazione completa concordano (0.43/0.44, 0.30/0.29, 0.90/0.92).

## Impianto (ostile ad ABM)

Grafi: FB15k-237 e WN18RR, train split, stessi file e sha256 di
`examples/replicate.py` (già in `data/external/`; self-loop esclusi). Per ogni
(dataset, D, N, seed): N fatti distinti campionati uniformemente; domande = chiavi
(s, r) distinte del campione, al più 300 per seed; giusta se la risposta è uno degli
oggetti veri di (s, r) nel campione. Griglia: D = 2048 con N ∈ {50, 100, 200, 400};
D = 8192 con N ∈ {200, 400, 800, 1600}; ε ∈ {0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.2};
5 seed. 16 serie × 7 ε = **112 celle**.

**Rumore:** ogni bit conservato si inverte indipendentemente con probabilità ε (rumore
di conservazione). Per ABM: la traccia; il codebook si rigenera dai nomi. Per lo
store: tutti i bit codificati (celle o spostamenti + valori, più la ridondanza).

**Store:** layout {XOR filter a 3 segmenti (⌈1.23·c⌉ celle di le = ⌈log₂ #oggetti⌉
bit, come nel test 19), MPHF + valori} × codice {none, rep r/r+2, BCH, polare SC,
rep3 + BCH/polare} × c. **Regola fissata:** per (cella, seed, ε) la combinazione
con l'accuratezza prevista più alta; previsione esatta per none/rep, P(Bin(n, ε) ≤ t)
per blocco BCH (limite inferiore: una decodifica fallita può lasciare giusti i bit
letti), Monte Carlo del solo store per i polari (banco di 1024 blocchi, 64 store
simulati, flusso separato). Il Bloom non è pertinente ((s, r) → o, non appartenenza).

## Modello esatto di ABM con rumore

Un flip indipendente con probabilità ε dà p′ = p(1 − ε) + (1 − p)ε = ε + p(1 − 2ε)
per ogni candidato a segnale; un codeword nullo resta Binomial(D, ½). Si sostituisce
p con p′ in `cleanup_accuracy_mixed` (`abm_predict` nell'harness). Approssimazione
dichiarata: il flip è comune ai candidati di una query; il modello tratta le distanze
come indipendenti, come senza rumore (A4: correlazioni entro domanda ≤ 0.06).

## Previsioni, calcolate ora (`--predict`, 57 min con 8 processi)

ABM previsto / miglior avversario previsto (regola di scelta), media sui 5 seed.

| dataset | D | N | ε=0 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| fb15k237 | 2048 | 50 | 0.986 / 1.000 | 0.986 / 1.000 | 0.986 / 1.000 | 0.983 / 1.000 | 0.975 / 1.000 | 0.910 / 1.000 | 0.661 / 0.837 |
| fb15k237 | 2048 | 100 | 0.773 / 1.000 | 0.772 / 1.000 | 0.771 / 1.000 | 0.752 / 1.000 | 0.708 / 1.000 | 0.531 / 0.783 | 0.275 / 0.383 |
| fb15k237 | 2048 | 200 | 0.337 / 1.000 | 0.337 / 1.000 | 0.335 / 0.976 | 0.320 / 0.799 | 0.286 / 0.619 | 0.183 / 0.341 | 0.083 / 0.142 |
| fb15k237 | 2048 | 400 | 0.098 / 0.465 | 0.098 / 0.463 | 0.097 / 0.452 | 0.092 / 0.369 | 0.081 / 0.281 | 0.051 / 0.157 | 0.024 / 0.067 |
| fb15k237 | 8192 | 200 | 0.974 / 1.000 | 0.974 / 1.000 | 0.973 / 1.000 | 0.968 / 1.000 | 0.952 / 1.000 | 0.844 / 1.000 | 0.524 / 0.629 |
| fb15k237 | 8192 | 400 | 0.662 / 1.000 | 0.662 / 1.000 | 0.660 / 1.000 | 0.637 / 1.000 | 0.585 / 1.000 | 0.397 / 0.633 | 0.172 / 0.289 |
| fb15k237 | 8192 | 800 | 0.233 / 0.863 | 0.233 / 0.860 | 0.232 / 0.837 | 0.218 / 0.685 | 0.191 / 0.510 | 0.112 / 0.266 | 0.044 / 0.105 |
| fb15k237 | 8192 | 1600 | 0.057 / 0.405 | 0.057 / 0.403 | 0.057 / 0.392 | 0.053 / 0.335 | 0.046 / 0.243 | 0.027 / 0.122 | 0.011 / 0.048 |
| wn18rr | 2048 | 50 | 0.990 / 1.000 | 0.990 / 1.000 | 0.990 / 1.000 | 0.987 / 1.000 | 0.980 / 1.000 | 0.922 / 1.000 | 0.688 / 0.870 |
| wn18rr | 2048 | 100 | 0.785 / 1.000 | 0.785 / 1.000 | 0.783 / 1.000 | 0.765 / 1.000 | 0.722 / 1.000 | 0.549 / 0.782 | 0.293 / 0.361 |
| wn18rr | 2048 | 200 | 0.350 / 1.000 | 0.350 / 1.000 | 0.348 / 0.981 | 0.332 / 0.799 | 0.298 / 0.625 | 0.193 / 0.340 | 0.089 / 0.145 |
| wn18rr | 2048 | 400 | 0.100 / 0.459 | 0.100 / 0.457 | 0.099 / 0.446 | 0.094 / 0.366 | 0.083 / 0.272 | 0.052 / 0.145 | 0.024 / 0.059 |
| wn18rr | 8192 | 200 | 0.975 / 1.000 | 0.975 / 1.000 | 0.974 / 1.000 | 0.969 / 1.000 | 0.954 / 1.000 | 0.851 / 1.000 | 0.537 / 0.621 |
| wn18rr | 8192 | 400 | 0.661 / 1.000 | 0.661 / 1.000 | 0.659 / 1.000 | 0.636 / 1.000 | 0.584 / 1.000 | 0.397 / 0.623 | 0.172 / 0.289 |
| wn18rr | 8192 | 800 | 0.224 / 0.845 | 0.224 / 0.843 | 0.223 / 0.820 | 0.210 / 0.673 | 0.183 / 0.496 | 0.107 / 0.261 | 0.042 / 0.113 |
| wn18rr | 8192 | 1600 | 0.051 / 0.385 | 0.051 / 0.384 | 0.051 / 0.373 | 0.047 / 0.314 | 0.041 / 0.227 | 0.024 / 0.121 | 0.010 / 0.050 |

Avversario ristimato (A6, terzo flusso) / tetto PPV (B1, non realizzato) / MDE:

| dataset | D | N | ε=0 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| fb15k237 | 2048 | 50 | 1.000 / 1.000 / 0.023 | 1.000 / 1.000 / 0.023 | 1.000 / 1.000 / 0.024 | 1.000 / 1.000 / 0.026 | 1.000 / 1.000 / 0.031 | 1.000 / 1.000 / 0.056 | 0.801 / 1.000 / 0.130 |
| fb15k237 | 2048 | 100 | 1.000 / 1.000 / 0.058 | 1.000 / 1.000 / 0.058 | 1.000 / 1.000 / 0.058 | 1.000 / 1.000 / 0.060 | 1.000 / 1.000 / 0.063 | 0.788 / 1.000 / 0.091 | 0.386 / 0.705 / 0.091 |
| fb15k237 | 2048 | 200 | 1.000 / 1.000 / 0.046 | 0.998 / 1.000 / 0.046 | 0.977 / 1.000 / 0.048 | 0.800 / 1.000 / 0.060 | 0.632 / 0.971 / 0.065 | 0.336 / 0.617 / 0.065 | 0.151 / 0.308 / 0.048 |
| fb15k237 | 2048 | 400 | 0.469 / 0.580 / 0.051 | 0.468 / 0.576 / 0.051 | 0.457 / 0.564 / 0.051 | 0.375 / 0.512 / 0.052 | 0.290 / 0.440 / 0.045 | 0.159 / 0.280 / 0.041 | 0.070 / 0.140 / 0.024 |
| fb15k237 | 8192 | 200 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.017 | 1.000 / 1.000 / 0.021 | 1.000 / 1.000 / 0.035 | 0.630 / 1.000 / 0.068 |
| fb15k237 | 8192 | 400 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 0.999 / 1.000 / 0.039 | 0.625 / 1.000 / 0.057 | 0.300 / 0.595 / 0.047 |
| fb15k237 | 8192 | 800 | 0.864 / 1.000 / 0.051 | 0.861 / 1.000 / 0.051 | 0.839 / 1.000 / 0.052 | 0.692 / 0.954 / 0.057 | 0.513 / 0.827 / 0.056 | 0.274 / 0.534 / 0.052 | 0.113 / 0.272 / 0.033 |
| fb15k237 | 8192 | 1600 | 0.407 / 0.501 / 0.061 | 0.406 / 0.498 / 0.061 | 0.394 / 0.490 / 0.060 | 0.339 / 0.450 / 0.059 | 0.246 / 0.390 / 0.058 | 0.129 / 0.252 / 0.032 | 0.051 / 0.128 / 0.033 |
| wn18rr | 2048 | 50 | 1.000 / 1.000 / 0.019 | 1.000 / 1.000 / 0.019 | 1.000 / 1.000 / 0.020 | 1.000 / 1.000 / 0.022 | 1.000 / 1.000 / 0.027 | 1.000 / 1.000 / 0.052 | 0.814 / 1.000 / 0.121 |
| wn18rr | 2048 | 100 | 1.000 / 1.000 / 0.057 | 1.000 / 1.000 / 0.057 | 1.000 / 1.000 / 0.057 | 1.000 / 1.000 / 0.059 | 1.000 / 1.000 / 0.062 | 0.790 / 1.000 / 0.089 | 0.370 / 0.705 / 0.094 |
| wn18rr | 2048 | 200 | 1.000 / 1.000 / 0.047 | 0.999 / 1.000 / 0.047 | 0.982 / 1.000 / 0.050 | 0.800 / 1.000 / 0.060 | 0.624 / 0.975 / 0.066 | 0.347 / 0.620 / 0.068 | 0.146 / 0.309 / 0.049 |
| wn18rr | 2048 | 400 | 0.462 / 0.572 / 0.049 | 0.461 / 0.567 / 0.049 | 0.450 / 0.555 / 0.048 | 0.371 / 0.504 / 0.046 | 0.284 / 0.433 / 0.043 | 0.148 / 0.275 / 0.034 | 0.065 / 0.137 / 0.023 |
| wn18rr | 8192 | 200 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.016 | 1.000 / 1.000 / 0.017 | 1.000 / 1.000 / 0.021 | 0.999 / 1.000 / 0.035 | 0.617 / 1.000 / 0.069 |
| wn18rr | 8192 | 400 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 1.000 / 1.000 / 0.038 | 0.996 / 1.000 / 0.039 | 0.625 / 1.000 / 0.057 | 0.297 / 0.586 / 0.050 |
| wn18rr | 8192 | 800 | 0.846 / 1.000 / 0.052 | 0.844 / 1.000 / 0.052 | 0.822 / 1.000 / 0.052 | 0.677 / 0.935 / 0.060 | 0.489 / 0.811 / 0.059 | 0.262 / 0.524 / 0.053 | 0.112 / 0.267 / 0.032 |
| wn18rr | 8192 | 1600 | 0.389 / 0.476 / 0.045 | 0.388 / 0.473 / 0.045 | 0.376 / 0.466 / 0.045 | 0.317 / 0.427 / 0.047 | 0.230 / 0.371 / 0.041 | 0.120 / 0.239 / 0.031 | 0.053 / 0.122 / 0.023 |

Layout e codice scelti dalla regola (più voci: scelte diverse fra seed):

| dataset | D | N | ε=0 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| fb15k237 | 2048 | 50 | xor/none | xor/bch | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar |
| fb15k237 | 2048 | 100 | xor/none | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar | xor/polar |
| fb15k237 | 2048 | 200 | xor/none | xor/polar | xor/none | xor/bch | xor/bch,xor/polar | xor/polar | xor/polar |
| fb15k237 | 2048 | 400 | xor/none | xor/none | xor/none | xor/bch | xor/polar | xor/polar | mphf/polar,xor/polar |
| fb15k237 | 8192 | 200 | xor/none | xor/bch | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar |
| fb15k237 | 8192 | 400 | xor/none | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar | xor/polar |
| fb15k237 | 8192 | 800 | xor/none | xor/none | xor/none | xor/bch | xor/polar | xor/polar | xor/polar |
| fb15k237 | 8192 | 1600 | xor/none | xor/none | xor/none | xor/bch | xor/bch,xor/polar | xor/polar | xor/polar |
| wn18rr | 2048 | 50 | xor/none | xor/bch | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar |
| wn18rr | 2048 | 100 | xor/none | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar | xor/polar |
| wn18rr | 2048 | 200 | xor/none | xor/polar | xor/bch,xor/none | xor/bch | xor/bch,xor/polar | xor/polar | mphf/polar,xor/polar |
| wn18rr | 2048 | 400 | xor/none | xor/none | xor/none | xor/bch | xor/bch,xor/polar | xor/polar | xor/polar |
| wn18rr | 8192 | 200 | xor/none | xor/bch | xor/bch | xor/bch | xor/bch | xor/polar | xor/polar |
| wn18rr | 8192 | 400 | xor/none | xor/bch | xor/bch | xor/bch | xor/bch,xor/polar | xor/polar | xor/polar |
| wn18rr | 8192 | 800 | xor/none | xor/none | xor/none | xor/bch | xor/polar | xor/polar | xor/polar |
| wn18rr | 8192 | 1600 | xor/none | xor/none | xor/none | xor/bch | xor/polar | xor/polar | xor/polar |

**Lettura.** Il modello prevede **ABM sotto il miglior avversario in tutte le 112
celle**. Il margine più piccolo a ε ≥ 0.1 è di circa 7 punti (WN18RR, D = 2048,
N = 100, ε = 0.2: 0.293 contro 0.361); a ε = 0.2 e carico basso i polari perdono
molto (0.84 a D = 2048, N = 50) ma restano sopra ABM (0.66). Con ripetizione e
Hamming la prima versione prevedeva 21 vittorie: erano vittorie contro codici deboli.

**Famiglia confermativa: 0 celle.** **ε\* previsto: mai, in tutte le 16 serie**
(valutabili per B4: 13 su 16; non valutabili FB15k-237 D = 2048 N = 50 e WN18RR
D = 2048 N ∈ {50, 100}, dove la differenza prevista non resta sotto −MDE a ogni ε).

## Ipotesi e criteri, fissati ora

Unità = seed; d_s = media su 20 rumori e sulle domande di (ABM − avversario scelto).
Test t one-sided, 4 gradi di libertà.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — esiste un ε in cui ABM batte il miglior avversario (direzionale) | famiglia confermativa vuota, quindi H1 si valuta su **tutte** le 112 celle: almeno una con p ≤ 0.05/112 (Bonferroni one-sided) | nessuna cella così (esito previsto) |
| **H2** — il modello esatto prevede ABM sotto rumore | errore medio assoluto ≤ 3 punti in ogni gruppo (dataset × D, 28 celle) **e** errore massimo di cella a ε = 0.2 ≤ 8 punti | > 5 punti in un gruppo, o > 8 punti in una cella a ε = 0.2 |
| **H3** — ε\* previsto vs osservato (B4) | entro un passo di griglia nelle serie valutabili (≥ 10 su 13) | in < 7 su 13 |
| **H4** — il modello dell'avversario | errore medio assoluto ≤ 3 punti per gruppo fra misura e ristima A6 | > 5 punti in un gruppo |

Il resto è **sostenuto in parte**. **H0, controllo e non conferma:** a ε = 0 ABM
sotto l'avversario in 16 serie su 16 (repliche dei test 18–19). Si riportano sempre:
il divario dal tetto PPV, lo store `none` (XOR a c massimo), l'accuratezza selettiva,
le raffiche, il rumore sul codebook.

**Previsioni (non criteri).** H1 **falsificata** (nessuna vittoria di ABM), H2, H3, H4
sostenute. Conseguenza attesa: il perimetro hardware non salva ABM; a pari bit,
anche con memoria rumorosa fino a ε = 0.2, uno store esatto con un codice moderno
(polare SC, BCH) recupera di più. Se l'esito sarà questo, l'argomento "degrada con
grazia" vale solo contro store senza codice o con codici semplici.

## Bias minimo rilevabile

MDE per cella nella seconda tabella: da 1.6 punti (D = 8192, N = 200, ε piccolo) a
13 punti (D = 2048, N = 50, ε = 0.2). È più alto a ε = 0.2 e carico basso, dove
pochi seed e la varianza fra campioni pesano di più. Un vantaggio di ABM più piccolo
dell'MDE non è escluso da un esito nullo.

## Tempo stimato del run completo

`--predict` ha richiesto 57 min con 8 processi (scelta del codice, ristima A6). Il
run completo aggiunge per ogni memoria: 20 rumori ABM × 9 condizioni (7 ε + 2
raffiche), 5 rumori sul codebook × 6 ε, 3 × 20 simulazioni complete dello store per
ε. Stima: 1.5–2.5 ore con 8 processi.

## Cosa ho visto prima

- Le previsioni qui sopra (`--predict`), che leggono i dati reali ma non costruiscono
  memorie ABM.
- La prima versione (`99ccc03`) e le sue previsioni con ripetizione e Hamming.
- Lo smoke sintetico (D = 1024, N ∈ {40, 120}, 3 seed, R = 20): ABM previsto contro
  misurato, errore medio 2.0 punti, massimo 3.8, a ε = 0.2 1.2; avversario misurato
  contro ristimato 0.4 punti; ABM sotto l'avversario ovunque tranne un pareggio
  (N = 40, ε = 0.2: 0.503 contro 0.503). Nessuna misura ABM sui dati reali.

## Limiti dichiarati prima

- **Decodifica SC, non SCL/CRC.** Un polare con list decoding, o un LDPC, farebbe
  meglio: a favore di ABM. Il tetto PPV è riportato; il divario fra avversario e tetto
  arriva a 20–40 punti a ε alto.
- **Codebook e nomi senza errori** nell'analisi principale; il rumore sul codebook ABM
  e il fattore dei nomi sono solo descrittivi.
- **Rumore iid, di conservazione, unico per memoria** (con 20 estrazioni); niente
  errori di lettura né deriva. Le raffiche sono solo descrittive.
- **Previsione BCH come limite inferiore** (decodifica fallita = blocco perso); la
  scelta fra BCH e polari può quindi sottostimare i BCH di poco.
- **Winner's curse nella scelta** (previsione massimizzata su ~200 configurazioni):
  per questo H4 usa la ristima indipendente.
- **MPHF semplificata** (spostamenti a larghezza fissa, non compressi come CHD):
  il layout a lettura singola potrebbe costare meno bit di così.
- **Campione uniforme di fatti**, non sottografi connessi: pochi gemelli e alias.
- **5 seed come unità**: test con 4 gradi di libertà, potenza modesta.

## Esito (2026-10-12)

Eseguito una volta, dopo il push e le marche OpenTimestamps (`78a3234`), con
`python examples/bit_errors_prereg.py --workers 6` (98 min); risultati in
`results/bit_errors_prereg_results.json`.

| ipotesi | risultato | esito |
|---|---|---|
| **H1** — ABM batte il miglior avversario in almeno una cella | famiglia confermativa vuota; su tutte le 112 celle nessun vantaggio di ABM. L'harness ne segnala 2 (WN18RR, D = 2048, N = 50, ε = 0 e 1e-4), ma sono **pareggi esatti** (ABM 1.000, avversario 1.000, d = 0, SE = 0): t = 0/0 = NaN, che il codice converte in p = 0. È un bug dell'harness, scoperto dopo il run; un pareggio non soddisfa "ABM batte", quindi conta il testo del criterio. L'output congelato non è stato modificato. | **falsificata** (come previsto) |
| **H2** — il modello prevede ABM sotto rumore | MAE 0.38–0.73 punti per gruppo; massimo di cella 1.48 (anche a ε = 0.2) | **sostenuta** |
| **H3** — ε\* previsto contro osservato | 13/13 serie valutabili entro un passo (ε\* = "mai" ovunque) | **sostenuta** |
| **H4** — il modello dell'avversario | MAE 0.13–0.39 punti contro la ristima indipendente | **sostenuta** |
| **H0** — controllo a ε = 0, ABM sotto | 15/16 serie; la sedicesima è il pareggio a 1.000 sopra | controllo coerente |

### Cosa dice

- Sotto bit flip iid fino a ε = 0.2, a pari bit, uno store esatto con codici pratici
  (polari SC, BCH, concatenati) batte ABM in ogni cella in cui c'è una differenza; il
  distacco più piccolo a ε alto è ~4–13 punti, e il tetto PPV dello store è ancora più su.
- La prima versione del disegno prevedeva 21 vittorie di ABM: erano vittorie contro
  ripetizione e Hamming. La review ostile del disegno le ha tolte prima dei dati.
- Il perimetro "hardware rumoroso" non salva ABM, almeno con rumore iid e codebook senza
  errori. Resta la sola cosa che regge: il modello esatto prevede ABM anche sotto rumore
  (entro 1.5 punti per cella).
- Bug da correggere nell'harness per il futuro: con d = 0 e SE = 0 il test deve dare
  p = 1 (o 0.5), non 0.
