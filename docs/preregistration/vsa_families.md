# Preregistrazione 21 — la stima a priori del cleanup su quattro famiglie VSA

**Data: 2026-10-09.** Committata **prima** di misurare, insieme all'harness
[`examples/vsa_families_prereg.py`](../../examples/vsa_families_prereg.py) e alle
previsioni congelate `results/vsa_families_prereg_predictions.json`. Prima del commit
l'harness è stato eseguito solo con `--smoke` (grafo sintetico generato nello script)
e con `--predict` (legge FB15k-237 e WN18RR, campiona i sottografi, sceglie D e
calcola le previsioni; **non costruisce nessuna memoria** sui dati reali). Stesse
regole di [`fb15k237.md`](fb15k237.md).

## Perché

Il modello esatto (`reference/exact.py`) prevede l'accuratezza del cleanup di ABM
(MAP-B) prima di costruire la memoria, e lo ha fatto entro 1–2 punti nelle
preregistrazioni 10, 18 e 19. Qui si chiede se la stessa procedura — *dimensionare una
memoria VSA dalle triple, prima di costruirla* — regge su altre famiglie VSA, così che
il contributo valga per la comunità VSA e non solo per ABM.

## Cosa è nuovo rispetto a Frady-Kleyko-Sommer, e cosa no

Frady, Kleyko e Sommer (2018, `frady2018sequence`) danno già la teoria gaussiana del
recupero per MAP, HRR e FHRR con bundling per somma e argmax sul codebook; Kleyko et
al. (2023) la estendono; Schlegel et al. (2022) confrontano le famiglie empiricamente;
la formula per BSDC con OR è un argomento di occupazione alla Bloom, non nuovo. **Non
è nuova nessuna teoria della capacità.** È nuovo solo:

1. la **contabilità della struttura di un grafo reale** dentro quelle formule:
   gemelli simmetrici (s, r, o) / (o, r, s) come un vettore di peso 2 (la varianza del
   rumore è S2 = Σ w², non N), alias (x, r, s) come candidati con lo stesso segnale
   dell'oggetto vero, query con più oggetti veri come più bersagli, codebook M =
   entità + relazioni del sottografo;
2. il **test** di queste previsioni su sottografi di FB15k-237 e WN18RR, compresi
   sottografi densi;
3. la **procedura**: D scelto dalla previsione, prima di costruire la memoria.

Per MAP-B il modello è quello esatto della reference (binomiale, pareggi), non
gaussiano: è il controllo, già testato.

## Famiglie e modelli di previsione

Lo stesso encoding per tutte: f = s ∘ ρ(r) ∘ o, ρ = shift ciclico di una posizione
(BSDC: dei blocchi); è simmetrico in s, o in tutte e quattro, quindi gemelli e alias
hanno la stessa contabilità. Unbinding con s ∘ ρ(r), cleanup sul codebook di M
simboli. Corretto = la risposta è uno degli oggetti veri di (s, r). Pareggi divisi a
caso (MAP-B: la regola della reference, prima nell'ordine; il modello li divide a metà).

| famiglia | atomi | bundling | cleanup | previsione per query |
|---|---|---|---|---|
| **MAP-B** (ABM, controllo) | bipolari, D bit | maggioranza (traccia binarizzata) | min Hamming | `exact.predict_queries` (esatta, binomiale) |
| **MAP-I** | bipolari | somma intera, **non** binarizzata | argmax ⟨y, c⟩ | gaussiana FKS, v = 1 |
| **FHRR** | fasori e^{iθ}, θ ~ U(0, 2π) | somma complessa | argmax Re⟨y, c̄⟩ | gaussiana FKS, v = 1/2 |
| **BSDC** (codici sparsi a blocchi) | B blocchi × L, un 1 per blocco | OR (traccia di B·L bit) | n. di blocchi colpiti | binomiale di occupazione |

**Gaussiana (MAP-I, FHRR).** Punteggi indipendenti: oggetto vero o alias di peso w ~
N(w·D, v·D·(S2 − w²)); nullo ~ N(0, v·D·S2). Accuratezza = P(il massimo è un oggetto
vero), per integrazione numerica. v = 1/2 per FHRR perché ogni fatto estraneo
contribuisce Re Σ e^{iφ}, varianza D/2: a pari D, FHRR ha metà del rumore di MAP-I.

**BSDC.** Un candidato a segnale (oggetto vero o alias) colpisce sempre tutti i B
blocchi; un nullo li colpisce tutti con probabilità π = (1 − (1 − 1/L)^n)^B, n = vettori
distinti. Accuratezza = E[g / (g + a + X)], X ~ Binomiale(M − g − a, π). Approssima:
occupazione media al posto di quella realizzata, blocchi e nulli indipendenti.

## Impianto

- **Dati:** FB15k-237 e WN18RR, train split, già in `data/external/` con gli sha256 di
  `examples/replicate.py`; self-loop esclusi, triple deduplicate.
- **Sottografi:** N ∈ {100, 400} triple; *uniform* (a caso) e *dense* (BFS non
  orientata da un'entità a caso, vicinati interi: gemelli, alias, query a più
  oggetti). **10 seed** per cella (≥ 3 richiesti; 10 perché con 3 il bias minimo
  rilevabile superava i 3 punti del criterio).
- **Query:** tutte le (s, r) distinte del sottografo, al più 200 per seed.
- **Regola per D (blocco calib), scritta prima:** per ogni (dataset, campione, N,
  famiglia) e target t ∈ {0.50, 0.85}, D = il minimo multiplo di 64 (BSDC: di L) con
  previsione media sui 10 seed ≥ t × tetto degli alias (media di g/(g+a)), D ≤ 65 536.
  Così le previsioni cadono fra 0.47 e 0.90 (BSDC: il passo di L le fa salire sopra
  il target fino a +15 punti).
- **BSDC, L:** potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà):
  L = 128 per N = 100, 512 per N = 400.
- **Bit per componente (contati solo per la traccia; il codebook non si conta, come
  nelle preregistrazioni 18–19):** MAP-B e BSDC 1; MAP-I ⌈log₂(2N + 1)⌉ (8 per N = 100,
  10 per N = 400), la larghezza intera minima senza perdita perché |T_d| ≤ N; FHRR 32,
  float16 per la parte reale e per l'immaginaria (la traccia misurata **è**
  quantizzata a float16: i bit dichiarati sono quelli usati). Scelta: il formato di
  memorizzazione più piccolo ovvio senza perdita (MAP-I) o con perdita trascurabile
  rispetto al rumore di crosstalk (FHRR: errore relativo 2⁻¹¹ contro un SNR di
  qualche unità); non è l'entropia (vedi limiti).
- **Blocco pari bit:** per ogni (dataset, campione, N) il budget è il numero di bit
  che MAP-B usa al target 0.85; le altre famiglie ricevono D = budget / bit per
  componente (BSDC: arrotondato a un multiplo di L). Qui le previsioni cadono fuori
  da [0.3, 0.95] (MAP-I e FHRR in basso, BSDC al tetto): il blocco serve solo all'ipotesi
  sull'ordine (H7), non alla calibrazione.
- **SE e bias minimo rilevabile per cella:** SE = √(Σ p_q(1 − p_q)) / Q con p_q le
  previsioni per query (Poisson-binomiale, query indipendenti); MDB = 2.8 · SE (α =
  0.05 bilaterale, potenza 0.8). z = (misurato − previsto) / SE.

## Previsioni, calcolate ora (`--predict`), una riga per cella

Media sui 10 seed; "bit" = D × bit per componente. Struttura dei sottografi (medie sui
seed): FB15k-237 uniform: alias 0.1–0.3 % delle query, gemelli 0 %; WN18RR uniform:
idem; FB15k-237 dense: alias 2.9 % (N = 100) e 0.8 % (400), gemelli 4.4 % e 1.5 %;
WN18RR dense: alias 9.9 % e 8.1 %, gemelli 15 % e 15 %, tetto 0.937 e 0.950.

| dataset | campione | N | blocco | famiglia | D | bit | tetto | previsto | MDB (punti) |
|---|---|---|---|---|---|---|---|---|---|
| FB15k-237 | uniform | 100 | 0.50·tetto | MAP-B | 1280 | 1280 | 1.000 | 0.517 | 4.4 |
| FB15k-237 | uniform | 100 | 0.50·tetto | MAP-I | 832 | 6656 | 1.000 | 0.530 | 4.4 |
| FB15k-237 | uniform | 100 | 0.50·tetto | FHRR | 448 | 14336 | 1.000 | 0.571 | 4.4 |
| FB15k-237 | uniform | 100 | 0.50·tetto | BSDC | 1152 | 1152 | 1.000 | 0.628 | 4.3 |
| FB15k-237 | uniform | 100 | 0.85·tetto | MAP-B | 2432 | 2432 | 1.000 | 0.851 | 3.2 |
| FB15k-237 | uniform | 100 | 0.85·tetto | MAP-I | 1536 | 12288 | 1.000 | 0.850 | 3.2 |
| FB15k-237 | uniform | 100 | 0.85·tetto | FHRR | 768 | 24576 | 1.000 | 0.850 | 3.2 |
| FB15k-237 | uniform | 100 | 0.85·tetto | BSDC | 1408 | 1408 | 1.000 | 0.864 | 3.0 |
| FB15k-237 | uniform | 100 | pari bit | MAP-I | 304 | 2432 | 1.000 | 0.159 | 3.3 |
| FB15k-237 | uniform | 100 | pari bit | FHRR | 76 | 2432 | 1.000 | 0.069 | 2.3 |
| FB15k-237 | uniform | 100 | pari bit | BSDC | 2432 | 2432 | 1.000 | 0.998 | 0.3 |
| FB15k-237 | uniform | 400 | 0.50·tetto | MAP-B | 6336 | 6336 | 0.999 | 0.502 | 3.1 |
| FB15k-237 | uniform | 400 | 0.50·tetto | MAP-I | 4032 | 40320 | 0.999 | 0.503 | 3.1 |
| FB15k-237 | uniform | 400 | 0.50·tetto | FHRR | 2048 | 65536 | 0.999 | 0.512 | 3.1 |
| FB15k-237 | uniform | 400 | 0.50·tetto | BSDC | 5632 | 5632 | 0.999 | 0.639 | 3.0 |
| FB15k-237 | uniform | 400 | 0.85·tetto | MAP-B | 11520 | 11520 | 0.999 | 0.851 | 2.2 |
| FB15k-237 | uniform | 400 | 0.85·tetto | MAP-I | 7296 | 72960 | 0.999 | 0.849 | 2.2 |
| FB15k-237 | uniform | 400 | 0.85·tetto | FHRR | 3648 | 116736 | 0.999 | 0.849 | 2.2 |
| FB15k-237 | uniform | 400 | 0.85·tetto | BSDC | 6656 | 6656 | 0.999 | 0.869 | 2.1 |
| FB15k-237 | uniform | 400 | pari bit | MAP-I | 1152 | 11520 | 0.999 | 0.081 | 1.7 |
| FB15k-237 | uniform | 400 | pari bit | FHRR | 360 | 11520 | 0.999 | 0.041 | 1.2 |
| FB15k-237 | uniform | 400 | pari bit | BSDC | 11264 | 11264 | 0.999 | 0.998 | 0.2 |
| FB15k-237 | dense | 100 | 0.50·tetto | MAP-B | 896 | 896 | 0.989 | 0.498 | 4.8 |
| FB15k-237 | dense | 100 | 0.50·tetto | MAP-I | 576 | 4608 | 0.989 | 0.504 | 4.8 |
| FB15k-237 | dense | 100 | 0.50·tetto | FHRR | 320 | 10240 | 0.989 | 0.547 | 4.8 |
| FB15k-237 | dense | 100 | 0.50·tetto | BSDC | 896 | 896 | 0.989 | 0.587 | 4.9 |
| FB15k-237 | dense | 100 | 0.85·tetto | MAP-B | 2048 | 2048 | 0.989 | 0.844 | 3.6 |
| FB15k-237 | dense | 100 | 0.85·tetto | MAP-I | 1344 | 10752 | 0.989 | 0.857 | 3.5 |
| FB15k-237 | dense | 100 | 0.85·tetto | FHRR | 704 | 22528 | 0.989 | 0.873 | 3.3 |
| FB15k-237 | dense | 100 | 0.85·tetto | BSDC | 1280 | 1280 | 0.989 | 0.901 | 3.0 |
| FB15k-237 | dense | 100 | pari bit | MAP-I | 256 | 2048 | 0.989 | 0.253 | 4.0 |
| FB15k-237 | dense | 100 | pari bit | FHRR | 64 | 2048 | 0.989 | 0.139 | 3.2 |
| FB15k-237 | dense | 100 | pari bit | BSDC | 2048 | 2048 | 0.989 | 0.986 | 0.8 |
| FB15k-237 | dense | 400 | 0.50·tetto | MAP-B | 4800 | 4800 | 0.998 | 0.504 | 3.1 |
| FB15k-237 | dense | 400 | 0.50·tetto | MAP-I | 3072 | 30720 | 0.998 | 0.508 | 3.1 |
| FB15k-237 | dense | 400 | 0.50·tetto | FHRR | 1536 | 49152 | 0.998 | 0.508 | 3.1 |
| FB15k-237 | dense | 400 | 0.50·tetto | BSDC | 4608 | 4608 | 0.998 | 0.610 | 3.1 |
| FB15k-237 | dense | 400 | 0.85·tetto | MAP-B | 9792 | 9792 | 0.998 | 0.849 | 2.3 |
| FB15k-237 | dense | 400 | 0.85·tetto | MAP-I | 6208 | 62080 | 0.998 | 0.848 | 2.3 |
| FB15k-237 | dense | 400 | 0.85·tetto | FHRR | 3136 | 100352 | 0.998 | 0.852 | 2.3 |
| FB15k-237 | dense | 400 | 0.85·tetto | BSDC | 5632 | 5632 | 0.998 | 0.851 | 2.3 |
| FB15k-237 | dense | 400 | pari bit | MAP-I | 979 | 9790 | 0.998 | 0.155 | 2.1 |
| FB15k-237 | dense | 400 | pari bit | FHRR | 306 | 9792 | 0.998 | 0.096 | 1.7 |
| FB15k-237 | dense | 400 | pari bit | BSDC | 9728 | 9728 | 0.998 | 0.996 | 0.3 |
| WN18RR | uniform | 100 | 0.50·tetto | MAP-B | 1216 | 1216 | 1.000 | 0.510 | 4.4 |
| WN18RR | uniform | 100 | 0.50·tetto | MAP-I | 768 | 6144 | 1.000 | 0.509 | 4.4 |
| WN18RR | uniform | 100 | 0.50·tetto | FHRR | 384 | 12288 | 1.000 | 0.509 | 4.4 |
| WN18RR | uniform | 100 | 0.50·tetto | BSDC | 1024 | 1024 | 1.000 | 0.504 | 4.4 |
| WN18RR | uniform | 100 | 0.85·tetto | MAP-B | 2368 | 2368 | 1.000 | 0.851 | 3.2 |
| WN18RR | uniform | 100 | 0.85·tetto | MAP-I | 1536 | 12288 | 1.000 | 0.861 | 3.1 |
| WN18RR | uniform | 100 | 0.85·tetto | FHRR | 768 | 24576 | 1.000 | 0.861 | 3.1 |
| WN18RR | uniform | 100 | 0.85·tetto | BSDC | 1408 | 1408 | 1.000 | 0.883 | 2.8 |
| WN18RR | uniform | 100 | pari bit | MAP-I | 296 | 2368 | 1.000 | 0.167 | 3.3 |
| WN18RR | uniform | 100 | pari bit | FHRR | 74 | 2368 | 1.000 | 0.075 | 2.3 |
| WN18RR | uniform | 100 | pari bit | BSDC | 2304 | 2304 | 1.000 | 0.998 | 0.4 |
| WN18RR | uniform | 400 | 0.50·tetto | MAP-B | 6336 | 6336 | 0.998 | 0.504 | 3.1 |
| WN18RR | uniform | 400 | 0.50·tetto | MAP-I | 4032 | 40320 | 0.998 | 0.504 | 3.1 |
| WN18RR | uniform | 400 | 0.50·tetto | FHRR | 2048 | 65536 | 0.998 | 0.514 | 3.1 |
| WN18RR | uniform | 400 | 0.50·tetto | BSDC | 5632 | 5632 | 0.998 | 0.649 | 3.0 |
| WN18RR | uniform | 400 | 0.85·tetto | MAP-B | 11456 | 11456 | 0.998 | 0.849 | 2.2 |
| WN18RR | uniform | 400 | 0.85·tetto | MAP-I | 7296 | 72960 | 0.998 | 0.849 | 2.2 |
| WN18RR | uniform | 400 | 0.85·tetto | FHRR | 3648 | 116736 | 0.998 | 0.849 | 2.2 |
| WN18RR | uniform | 400 | 0.85·tetto | BSDC | 6656 | 6656 | 0.998 | 0.873 | 2.1 |
| WN18RR | uniform | 400 | pari bit | MAP-I | 1145 | 11450 | 0.998 | 0.081 | 1.7 |
| WN18RR | uniform | 400 | pari bit | FHRR | 358 | 11456 | 0.998 | 0.041 | 1.2 |
| WN18RR | uniform | 400 | pari bit | BSDC | 11264 | 11264 | 0.998 | 0.998 | 0.2 |
| WN18RR | dense | 100 | 0.50·tetto | MAP-B | 832 | 832 | 0.937 | 0.485 | 4.3 |
| WN18RR | dense | 100 | 0.50·tetto | MAP-I | 512 | 4096 | 0.937 | 0.476 | 4.2 |
| WN18RR | dense | 100 | 0.50·tetto | FHRR | 256 | 8192 | 0.937 | 0.476 | 4.2 |
| WN18RR | dense | 100 | 0.50·tetto | BSDC | 768 | 768 | 0.937 | 0.529 | 4.5 |
| WN18RR | dense | 100 | 0.85·tetto | MAP-B | 2112 | 2112 | 0.937 | 0.801 | 3.4 |
| WN18RR | dense | 100 | 0.85·tetto | MAP-I | 1344 | 10752 | 0.937 | 0.804 | 3.4 |
| WN18RR | dense | 100 | 0.85·tetto | FHRR | 704 | 22528 | 0.937 | 0.817 | 3.3 |
| WN18RR | dense | 100 | 0.85·tetto | BSDC | 1152 | 1152 | 0.937 | 0.850 | 3.0 |
| WN18RR | dense | 100 | pari bit | MAP-I | 264 | 2112 | 0.937 | 0.292 | 3.9 |
| WN18RR | dense | 100 | pari bit | FHRR | 66 | 2112 | 0.937 | 0.164 | 3.3 |
| WN18RR | dense | 100 | pari bit | BSDC | 2048 | 2048 | 0.937 | 0.936 | 1.3 |
| WN18RR | dense | 400 | 0.50·tetto | MAP-B | 4736 | 4736 | 0.950 | 0.478 | 2.7 |
| WN18RR | dense | 400 | 0.50·tetto | MAP-I | 3008 | 30080 | 0.950 | 0.478 | 2.7 |
| WN18RR | dense | 400 | 0.50·tetto | FHRR | 1536 | 49152 | 0.950 | 0.485 | 2.7 |
| WN18RR | dense | 400 | 0.50·tetto | BSDC | 4096 | 4096 | 0.950 | 0.572 | 2.9 |
| WN18RR | dense | 400 | 0.85·tetto | MAP-B | 11008 | 11008 | 0.950 | 0.810 | 2.2 |
| WN18RR | dense | 400 | 0.85·tetto | MAP-I | 6976 | 69760 | 0.950 | 0.809 | 2.2 |
| WN18RR | dense | 400 | 0.85·tetto | FHRR | 3520 | 112640 | 0.950 | 0.812 | 2.2 |
| WN18RR | dense | 400 | 0.85·tetto | BSDC | 5120 | 5120 | 0.950 | 0.813 | 2.2 |
| WN18RR | dense | 400 | pari bit | MAP-I | 1100 | 11000 | 0.950 | 0.198 | 2.1 |
| WN18RR | dense | 400 | pari bit | FHRR | 344 | 11008 | 0.950 | 0.120 | 1.8 |
| WN18RR | dense | 400 | pari bit | BSDC | 10752 | 10752 | 0.950 | 0.950 | 0.8 |

Regolarità previste (non criteri): a pari accuratezza MAP-I chiede ≈ 0.63 volte il D
di MAP-B (il fattore 2/π della binarizzazione) e FHRR la metà del D di MAP-I; in bit,
MAP-B e BSDC sono 5–10× più economici di MAP-I e 10–20× più di FHRR, e a pari bit BSDC
sta al tetto dove MAP-B è a 0.85 e MAP-I/FHRR sotto 0.3.

## Ipotesi e criteri, fissati ora

Gruppo = (famiglia, campione, dataset): 4 celle calib ciascuno (2 N × 2 target).
Errore = misurato − previsto, in punti; per ogni gruppo si riportano errore medio
assoluto (MAE), errore medio con segno e massimo di cella.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — MAP-B (controllo), sottografi uniform | MAE ≤ 3 in entrambi i gruppi | MAE > 5 in un gruppo |
| **H2** — MAP-I, gaussiana FKS + contabilità, uniform | MAE ≤ 3 in entrambi i gruppi | MAE > 5 in un gruppo |
| **H3** — FHRR, gaussiana FKS + contabilità, uniform | MAE ≤ 3 in entrambi i gruppi | MAE > 5 in un gruppo |
| **H4** — BSDC, occupazione binomiale + contabilità, uniform | MAE ≤ 3 in entrambi i gruppi | MAE > 5 in un gruppo |
| **H5 (rischiosa)** — la previsione regge sui sottografi densi con gemelli e alias, per tutte e quattro le famiglie | MAE ≤ 3 in tutti gli 8 gruppi dense | MAE > 5 in almeno un gruppo |
| **H6** — nessun bias oltre l'errore di campionamento, per famiglia | al più 2 delle 16 celle calib con \|z\| > 3 | 5 o più celle con \|z\| > 3 |
| **H7** — a pari bit, l'ordine fra famiglie è quello previsto | in tutte le coppie con differenza prevista ≥ 5 punti (MAP-B misurato nella sua cella 0.85) | più di una coppia invertita |

Il resto è **sostenuto in parte**. A livello di cella l'MDB va da 2.1 a 4.9 punti nel
blocco calib: una singola cella non può mostrare un bias di 3 punti con potenza 0.8 se
il suo MDB è maggiore; per questo i criteri sono sul gruppo (MAE su 4 celle) e H6
li completa in unità di SE. Sotto un modello perfetto il MAE atteso per solo rumore è
≈ 0.8 · SE ≈ 0.6–1.4 punti.

**Previsioni (non criteri).** H1 sostenuta (è il controllo). H2 e H3 sostenute: con
somme reali e D ≥ 256 la gaussiana di FKS dovrebbe bastare. H4 **a rischio**:
l'approssimazione con l'occupazione media ignora la variabilità di π fra memorie, e in
un test di tempo su un grafo sintetico (N = 400, 1 seed, 200 query) BSDC ha misurato
0.855 contro 0.895 previsto. H5 a rischio per BSDC e per i gruppi WN18RR dense; per
FB15k-237 dense il test è debole (alias < 3 %). H6: probabilmente falsificata o in
parte, perché l'SE ignora la correlazione fra query della stessa memoria (vedi limiti):
è un test severo di proposito. H7 sostenuta.

## Cosa ho visto prima

- Le previsioni qui sopra (`--predict`): leggono i dati reali ma non costruiscono
  memorie.
- Lo smoke (`--smoke`, grafo sintetico di 60 entità, N = 40, 3 seed, prima di portarli a 10): MAE per gruppo
  0.7–8.8 punti con poche decine di query per cella (rumore dominante); ordine a pari
  bit 24/24 coppie. Su questi numeri non ho cambiato nulla.
- Un test di tempo su un grafo sintetico (800 entità, N = 400, 1 seed, D del blocco
  calib WN18RR uniform 0.85): misurato/previsto MAP-B 0.845/0.843, MAP-I 0.855/0.842,
  FHRR 0.890/0.842, BSDC 0.855/0.895.

## Tempo stimato

`--predict` 3.7 min; le misure (88 celle × 10 seed, < 1 s ciascuna) circa 5 min:
**≈ 10 min** su questa macchina (12 core, un processo).

## Limiti dichiarati prima

- **Quale FKS.** La parte gaussiana è FKS con la contabilità; se H2/H3 reggono, il
  merito della forma funzionale è di FKS. Se falliscono solo su dense, il difetto è
  nella contabilità (nuova).
- **SE ottimista.** Le query della stessa memoria condividono la traccia e il
  codebook: l'SE Poisson-binomiale è un limite inferiore, e lo z di H6 è gonfiato.
- **Bit.** Le larghezze sono formati di memorizzazione, non entropia: una traccia
  MAP-I compressa costerebbe ≈ ½ log₂(2πeN) bit per componente (≈ 6 per N = 400, non
  10), FHRR con fasi quantizzate a pochi bit costerebbe molto meno di 32. Il codebook
  (M atomi) non è contato per nessuna famiglia. Il blocco pari bit è quindi
  sfavorevole a MAP-I e FHRR per costruzione, e lo dichiaro.
- **BSDC** solo con bundling per OR e binding per shift di blocco: niente somma con
  soglia né CDT di Rachkovskij; L fissato da una regola, non ottimizzato.
- **HRR** con convoluzione circolare e vettori gaussiani **non** è testato: solo FHRR
  (atomi unitari). L'unbinding approssimato di HRR aggiunge rumore che il modello qui
  non contiene.
- Atomi i.i.d., ρ = shift ciclico, encoding simmetrico in s, o (gemelli identici in
  tutte le famiglie); un encoding asimmetrico cambierebbe la contabilità.
- Due soli carichi e due target; D è scelto dalla previsione sugli stessi sottografi
  su cui si misura (lecito: la scelta non usa misure, ma accoppia D e campione).
- Self-loop esclusi: in MAP-B renderebbero ogni soggetto un alias, nelle altre
  famiglie no, e la contabilità divergerebbe.
