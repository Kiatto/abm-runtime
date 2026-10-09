# Preregistrazione 21 — la stima a priori del cleanup su cinque famiglie VSA

**Data: 2026-10-09.** Committata **prima** di misurare, insieme all'harness
[`examples/vsa_families_prereg.py`](../../examples/vsa_families_prereg.py) e alle
previsioni congelate `results/vsa_families_prereg_predictions.json`. Prima dei commit
l'harness è stato eseguito solo con `--smoke` (grafo sintetico generato nello script)
e con `--predict` (legge FB15k-237 e WN18RR, campiona i sottografi, sceglie D e
calcola le previsioni; **non costruisce nessuna memoria** sui dati reali). Stesse
regole di [`fb15k237.md`](fb15k237.md).

**Revisione del disegno prima dei dati** (commit successivo a `9857511`, nessuna misura
sui dati reali in mezzo), dopo una revisione ostile: comparatore FKS ingenuo e ipotesi
di ablazione (H7); bit di MAP-I corretti a ⌈log₂(N+1)⌉; SE a cluster per seed con t a
9 gdl; l'ordine a pari bit diventa descrittivo; BSDC rinominato BSDC-OR, con π
dall'occupazione realizzata, e nuova variante BSDC-S a somma; le celle uniform
dichiarate replica di FKS; validazione di D fuori campione (H9); flussi casuali
separati con `SeedSequence.spawn`; equivalenza MAP-B/`Memory.store` verificata nello
smoke. La versione di `9857511` è superata in tutto: tabelle e ipotesi sono queste.

## Perché

Il modello esatto (`reference/exact.py`) prevede l'accuratezza del cleanup di ABM
(MAP-B) prima di costruire la memoria, e lo ha fatto entro 1–2 punti nelle
preregistrazioni 10, 18 e 19. Qui si chiede se la stessa procedura — *dimensionare una
memoria VSA dalle triple, prima di costruirla* — regge su altre famiglie VSA.

## Cosa è nuovo rispetto a Frady-Kleyko-Sommer, e cosa no

Frady, Kleyko e Sommer (2018, `frady2018sequence`) danno la teoria gaussiana del
recupero per MAP, HRR e FHRR con bundling per somma e argmax sul codebook; Kleyko et
al. (2023) la estendono; Schlegel et al. (2022) confrontano le famiglie empiricamente;
BSDC-OR è un filtro di Bloom partizionato e la sua formula è occupazione alla Bloom.
**Nessuna teoria della capacità è nuova.** È nuovo solo:

1. la **contabilità della struttura di un grafo reale** dentro quelle formule:
   gemelli (s, r, o)/(o, r, s) come un vettore di peso 2 (rumore S2 = Σ w², non N),
   alias (x, r, s) come candidati con il segnale dell'oggetto vero, query con più
   oggetti veri, codebook M = entità + relazioni;
2. il **test** su sottografi di FB15k-237 e WN18RR, e soprattutto su sottografi
   **densi**, dove la contabilità cambia la previsione;
3. la **procedura**: D scelto dalla previsione prima di costruire la memoria, e
   validato fuori campione.

**Le celle uniform sono una replica di FKS, non un test della contabilità**: lì alias
e gemelli sono < 0.5 % e la previsione con contabilità e quella FKS ingenua
differiscono di 0.1–0.2 punti (tabella). Il test della contabilità sono i gruppi dense
(H6) e l'ablazione (H7).

## Famiglie e modelli di previsione

Stesso encoding per tutte: f = s ∘ ρ(r) ∘ o, ρ = shift ciclico (BSDC: dei blocchi),
simmetrico in s, o. Unbinding con s ∘ ρ(r), cleanup sul codebook di M simboli. Corretto
= la risposta è uno degli oggetti veri. Pareggi divisi a caso (MAP-B: regola della
reference, il modello li divide a metà).

| famiglia | atomi | bundling | cleanup | previsione per query |
|---|---|---|---|---|
| **MAP-B** (ABM, controllo) | bipolari, da md5 del nome (gli stessi in tutti i seed) | maggioranza | min Hamming | `exact.predict_queries` (binomiale esatta) |
| **MAP-I** | bipolari | somma intera, non binarizzata | argmax ⟨y, c⟩ | gaussiana FKS, v = 1 |
| **FHRR** | fasori e^{iθ} | somma complessa, salvata in float16 | argmax Re⟨y, c̄⟩ | gaussiana FKS, v = 1/2 |
| **BSDC-OR** (≈ Bloom partizionato) | B blocchi × L, un 1 per blocco | OR (B·L bit) | n. di blocchi colpiti | occupazione realizzata, esatta |
| **BSDC-S** | come BSDC-OR | somma (traccia a conteggi) | somma dei conteggi letti | distribuzione discreta esatta del rumore |

**Gaussiana (MAP-I, FHRR).** Punteggi indipendenti: vero o alias di peso w ~ N(w·D,
v·D·(S2 − w²)); nullo ~ N(0, v·D·S2). P(il massimo è un oggetto vero) per integrazione.

**BSDC-OR.** Un candidato a segnale colpisce tutti i B blocchi; un nullo li colpisce
tutti con π = Π_b K_b/L, K_b = bin occupati del blocco b, con la distribuzione esatta
P(K = k) = S(n,k)·L!/((L−k)!·L^n) (per ricorrenza), n vettori distinti; 2 000 estrazioni
con seme fisso. Accuratezza = E_π[g · E[1/(g + a + X)]], X | π ~ Bin(M − g − a, π).

**BSDC-S.** Il punteggio di un nullo è Σ_b Σ_w w·Bin(n_w, 1/L): pmf esatta per
convoluzione; un candidato di peso w vale w·B più lo stesso rumore senza il proprio
vettore. P(il massimo è un oggetto vero), pareggi divisi in proporzione al rischio
relativo (f/F) dei candidati al massimo: approssimazione.

**Comparatore FKS ingenuo (in ogni cella):** la stessa formula della famiglia con N
fatti distinti di peso 1, un oggetto vero, nessun alias (MAP-B: `exact.cleanup_accuracy(N, D, M)`).

## Impianto

- **Dati:** FB15k-237 e WN18RR, train split, in `data/external/` con gli sha256 di
  `examples/replicate.py`; self-loop esclusi, triple deduplicate.
- **Sottografi:** N ∈ {100, 400}; *uniform* (a caso) e *dense* (BFS non orientata da
  un'entità a caso). **10 seed**. Flussi casuali separati: `SeedSequence([seme, dataset,
  campione, N, seed]).spawn(2)` → campionamento e query; atomi (uno per riga e seed).
  Le celle calib dello stesso (dataset, campione, N) — le 4 di un gruppo e le 5
  famiglie — **condividono i 10 sottografi**: le loro misure sono correlate.
- **Query:** tutte le (s, r) distinte, al più 200 per seed.
- **Regola per D (blocco calib):** per ogni (dataset, campione, N, famiglia) e target
  t ∈ {0.50, 0.85}: il minimo multiplo di 64 (BSDC: di L) con previsione media **sui seed
  0–4** ≥ t × tetto degli alias (tetto = media di g/(g+a) sui seed 0–4), D ≤ 65 536. Le
  previsioni sui 10 seed cadono fra 0.47 e 0.89 (BSDC-OR sale oltre il target fino a 15
  punti per il passo di L).
- **BSDC, L:** potenza di 2 più vicina a N/ln 2: 128 per N = 100, 512 per N = 400.
- **Bit per componente** (solo traccia; codebook non contato, come nelle prereg. 18–19):
  MAP-B e BSDC-OR 1; MAP-I ⌈log₂(N+1)⌉ (T_d ≡ N mod 2, N+1 valori: 7 per N = 100, 9 per
  N = 400); BSDC-S ⌈log₂(N+1)⌉ (conteggi 0..N, stessa larghezza); FHRR 32 (float16 ×2;
  la traccia misurata è davvero quantizzata).
- **Blocco pari bit (descrittivo):** budget = bit di MAP-B al target 0.85; le altre
  famiglie hanno D = budget / bit per componente (BSDC arrotondato a multipli di L).
- **Statistica per cella (pesatura per seed):** previsto = media sui 10 seed della
  previsione media per query; misurato idem; e_k = mis_k − prev_k; **SE a cluster** =
  sd_k(e_k)/√10; t = errore/SE con 9 gdl. **MDE** preregistrato = (t₀.₉₇₅,₉ + t₀.₈₀,₉)·SE
  = 3.15·SE, con l'SE stimato ora dal modello (Poisson-binomiale per seed, query
  indipendenti): è un **limite inferiore** dell'MDE vero.

## Previsioni, calcolate ora (`--predict`), una riga per cella

Struttura (medie sui 10 seed): uniform, alias ≤ 0.4 %, gemelli ≤ 0.1 %, tetto ≥ 0.998.
FB15k-237 dense: alias 3.5 % (N = 100) e 0.7 % (400), gemelli 7.9 % e 0.9 %, tetto
0.985 e 0.996. WN18RR dense: alias 8.5 % e 9.4 %, gemelli 12.1 % e 14.4 %, tetto 0.927
e 0.957.

| dataset | campione | N | blocco | famiglia | D | bit | previsto | FKS ingenuo | MDE (punti) |
|---|---|---|---|---|---|---|---|---|---|
| FB15k-237 | uniform | 100 | 0.50·tetto | MAP-B | 1280 | 1280 | 0.517 | 0.515 | 5.0 |
| FB15k-237 | uniform | 100 | 0.50·tetto | MAP-I | 832 | 5824 | 0.531 | 0.529 | 5.0 |
| FB15k-237 | uniform | 100 | 0.50·tetto | FHRR | 448 | 14336 | 0.571 | 0.569 | 4.9 |
| FB15k-237 | uniform | 100 | 0.50·tetto | BSDC-OR | 1152 | 1152 | 0.630 | 0.629 | 4.8 |
| FB15k-237 | uniform | 100 | 0.50·tetto | BSDC-S | 1152 | 8064 | 0.521 | 0.519 | 5.0 |
| FB15k-237 | uniform | 100 | 0.85·tetto | MAP-B | 2432 | 2432 | 0.851 | 0.850 | 3.6 |
| FB15k-237 | uniform | 100 | 0.85·tetto | MAP-I | 1536 | 10752 | 0.850 | 0.850 | 3.6 |
| FB15k-237 | uniform | 100 | 0.85·tetto | FHRR | 768 | 24576 | 0.850 | 0.850 | 3.6 |
| FB15k-237 | uniform | 100 | 0.85·tetto | BSDC-OR | 1408 | 1408 | 0.864 | 0.864 | 3.4 |
| FB15k-237 | uniform | 100 | 0.85·tetto | BSDC-S | 1920 | 13440 | 0.851 | 0.850 | 3.6 |
| FB15k-237 | uniform | 100 | pari bit | MAP-I | 347 | 2429 | 0.188 | 0.187 | 3.9 |
| FB15k-237 | uniform | 100 | pari bit | FHRR | 76 | 2432 | 0.069 | 0.069 | 2.5 |
| FB15k-237 | uniform | 100 | pari bit | BSDC-OR | 2432 | 2432 | 0.998 | 0.999 | 0.4 |
| FB15k-237 | uniform | 100 | pari bit | BSDC-S | 256 | 1792 | 0.051 | 0.051 | 2.2 |
| FB15k-237 | uniform | 400 | 0.50·tetto | MAP-B | 6272 | 6272 | 0.497 | 0.493 | 3.5 |
| FB15k-237 | uniform | 400 | 0.50·tetto | MAP-I | 4032 | 36288 | 0.504 | 0.499 | 3.5 |
| FB15k-237 | uniform | 400 | 0.50·tetto | FHRR | 2048 | 65536 | 0.513 | 0.509 | 3.5 |
| FB15k-237 | uniform | 400 | 0.50·tetto | BSDC-OR | 5632 | 5632 | 0.641 | 0.640 | 3.4 |
| FB15k-237 | uniform | 400 | 0.50·tetto | BSDC-S | 5632 | 50688 | 0.499 | 0.495 | 3.5 |
| FB15k-237 | uniform | 400 | 0.85·tetto | MAP-B | 11456 | 11456 | 0.848 | 0.847 | 2.5 |
| FB15k-237 | uniform | 400 | 0.85·tetto | MAP-I | 7296 | 65664 | 0.849 | 0.848 | 2.5 |
| FB15k-237 | uniform | 400 | 0.85·tetto | FHRR | 3648 | 116736 | 0.849 | 0.848 | 2.5 |
| FB15k-237 | uniform | 400 | 0.85·tetto | BSDC-OR | 6656 | 6656 | 0.869 | 0.870 | 2.4 |
| FB15k-237 | uniform | 400 | 0.85·tetto | BSDC-S | 9216 | 82944 | 0.856 | 0.855 | 2.5 |
| FB15k-237 | uniform | 400 | pari bit | MAP-I | 1272 | 11448 | 0.095 | 0.093 | 2.1 |
| FB15k-237 | uniform | 400 | pari bit | FHRR | 358 | 11456 | 0.041 | 0.040 | 1.4 |
| FB15k-237 | uniform | 400 | pari bit | BSDC-OR | 11264 | 11264 | 0.998 | 0.999 | 0.3 |
| FB15k-237 | uniform | 400 | pari bit | BSDC-S | 1024 | 9216 | 0.020 | 0.020 | 1.0 |
| FB15k-237 | dense | 100 | 0.50·tetto | MAP-B | 832 | 832 | 0.504 | 0.435 | 5.6 |
| FB15k-237 | dense | 100 | 0.50·tetto | MAP-I | 512 | 3584 | 0.494 | 0.423 | 5.6 |
| FB15k-237 | dense | 100 | 0.50·tetto | FHRR | 256 | 8192 | 0.494 | 0.423 | 5.6 |
| FB15k-237 | dense | 100 | 0.50·tetto | BSDC-OR | 768 | 768 | 0.487 | 0.381 | 5.8 |
| FB15k-237 | dense | 100 | 0.50·tetto | BSDC-S | 896 | 6272 | 0.558 | 0.512 | 5.6 |
| FB15k-237 | dense | 100 | 0.85·tetto | MAP-B | 2048 | 2048 | 0.847 | 0.849 | 4.3 |
| FB15k-237 | dense | 100 | 0.85·tetto | MAP-I | 1280 | 8960 | 0.843 | 0.844 | 4.3 |
| FB15k-237 | dense | 100 | 0.85·tetto | FHRR | 640 | 20480 | 0.843 | 0.844 | 4.3 |
| FB15k-237 | dense | 100 | 0.85·tetto | BSDC-OR | 1152 | 1152 | 0.860 | 0.829 | 4.0 |
| FB15k-237 | dense | 100 | 0.85·tetto | BSDC-S | 1664 | 11648 | 0.850 | 0.862 | 4.2 |
| FB15k-237 | dense | 100 | pari bit | MAP-I | 292 | 2044 | 0.323 | 0.240 | 5.2 |
| FB15k-237 | dense | 100 | pari bit | FHRR | 64 | 2048 | 0.168 | 0.107 | 4.4 |
| FB15k-237 | dense | 100 | pari bit | BSDC-OR | 2048 | 2048 | 0.983 | 0.997 | 1.3 |
| FB15k-237 | dense | 100 | pari bit | BSDC-S | 256 | 1792 | 0.157 | 0.104 | 4.3 |
| FB15k-237 | dense | 400 | 0.50·tetto | MAP-B | 5376 | 5376 | 0.504 | 0.492 | 3.4 |
| FB15k-237 | dense | 400 | 0.50·tetto | MAP-I | 3392 | 30528 | 0.500 | 0.487 | 3.4 |
| FB15k-237 | dense | 400 | 0.50·tetto | FHRR | 1728 | 55296 | 0.510 | 0.498 | 3.4 |
| FB15k-237 | dense | 400 | 0.50·tetto | BSDC-OR | 4608 | 4608 | 0.533 | 0.510 | 3.5 |
| FB15k-237 | dense | 400 | 0.50·tetto | BSDC-S | 5120 | 46080 | 0.537 | 0.531 | 3.4 |
| FB15k-237 | dense | 400 | 0.85·tetto | MAP-B | 10368 | 10368 | 0.849 | 0.850 | 2.5 |
| FB15k-237 | dense | 400 | 0.85·tetto | MAP-I | 6592 | 59328 | 0.849 | 0.850 | 2.5 |
| FB15k-237 | dense | 400 | 0.85·tetto | FHRR | 3328 | 106496 | 0.853 | 0.855 | 2.5 |
| FB15k-237 | dense | 400 | 0.85·tetto | BSDC-OR | 6144 | 6144 | 0.892 | 0.886 | 2.2 |
| FB15k-237 | dense | 400 | 0.85·tetto | BSDC-S | 8704 | 78336 | 0.876 | 0.880 | 2.3 |
| FB15k-237 | dense | 400 | pari bit | MAP-I | 1152 | 10368 | 0.139 | 0.120 | 2.3 |
| FB15k-237 | dense | 400 | pari bit | FHRR | 324 | 10368 | 0.070 | 0.057 | 1.7 |
| FB15k-237 | dense | 400 | pari bit | BSDC-OR | 10240 | 10240 | 0.996 | 0.999 | 0.3 |
| FB15k-237 | dense | 400 | pari bit | BSDC-S | 1024 | 9216 | 0.045 | 0.037 | 1.4 |
| WN18RR | uniform | 100 | 0.50·tetto | MAP-B | 1216 | 1216 | 0.509 | 0.509 | 5.0 |
| WN18RR | uniform | 100 | 0.50·tetto | MAP-I | 768 | 5376 | 0.508 | 0.508 | 5.0 |
| WN18RR | uniform | 100 | 0.50·tetto | FHRR | 384 | 12288 | 0.508 | 0.508 | 5.0 |
| WN18RR | uniform | 100 | 0.50·tetto | BSDC-OR | 1024 | 1024 | 0.508 | 0.508 | 5.0 |
| WN18RR | uniform | 100 | 0.50·tetto | BSDC-S | 1152 | 8064 | 0.545 | 0.545 | 5.0 |
| WN18RR | uniform | 100 | 0.85·tetto | MAP-B | 2368 | 2368 | 0.850 | 0.851 | 3.5 |
| WN18RR | uniform | 100 | 0.85·tetto | MAP-I | 1536 | 10752 | 0.860 | 0.861 | 3.4 |
| WN18RR | uniform | 100 | 0.85·tetto | FHRR | 768 | 24576 | 0.860 | 0.861 | 3.4 |
| WN18RR | uniform | 100 | 0.85·tetto | BSDC-OR | 1408 | 1408 | 0.882 | 0.884 | 3.2 |
| WN18RR | uniform | 100 | 0.85·tetto | BSDC-S | 1920 | 13440 | 0.863 | 0.864 | 3.4 |
| WN18RR | uniform | 100 | pari bit | MAP-I | 338 | 2366 | 0.196 | 0.196 | 4.0 |
| WN18RR | uniform | 100 | pari bit | FHRR | 74 | 2368 | 0.075 | 0.074 | 2.6 |
| WN18RR | uniform | 100 | pari bit | BSDC-OR | 2304 | 2304 | 0.997 | 0.998 | 0.5 |
| WN18RR | uniform | 100 | pari bit | BSDC-S | 256 | 1792 | 0.058 | 0.058 | 2.3 |
| WN18RR | uniform | 400 | 0.50·tetto | MAP-B | 6336 | 6336 | 0.503 | 0.503 | 3.5 |
| WN18RR | uniform | 400 | 0.50·tetto | MAP-I | 4032 | 36288 | 0.503 | 0.503 | 3.5 |
| WN18RR | uniform | 400 | 0.50·tetto | FHRR | 2048 | 65536 | 0.513 | 0.513 | 3.5 |
| WN18RR | uniform | 400 | 0.50·tetto | BSDC-OR | 5632 | 5632 | 0.651 | 0.649 | 3.3 |
| WN18RR | uniform | 400 | 0.50·tetto | BSDC-S | 5632 | 50688 | 0.499 | 0.499 | 3.5 |
| WN18RR | uniform | 400 | 0.85·tetto | MAP-B | 11456 | 11456 | 0.848 | 0.849 | 2.5 |
| WN18RR | uniform | 400 | 0.85·tetto | MAP-I | 7296 | 65664 | 0.849 | 0.850 | 2.5 |
| WN18RR | uniform | 400 | 0.85·tetto | FHRR | 3648 | 116736 | 0.849 | 0.850 | 2.5 |
| WN18RR | uniform | 400 | 0.85·tetto | BSDC-OR | 6656 | 6656 | 0.874 | 0.874 | 2.3 |
| WN18RR | uniform | 400 | 0.85·tetto | BSDC-S | 9216 | 82944 | 0.855 | 0.857 | 2.5 |
| WN18RR | uniform | 400 | pari bit | MAP-I | 1272 | 11448 | 0.096 | 0.095 | 2.1 |
| WN18RR | uniform | 400 | pari bit | FHRR | 358 | 11456 | 0.042 | 0.041 | 1.4 |
| WN18RR | uniform | 400 | pari bit | BSDC-OR | 11264 | 11264 | 0.998 | 0.999 | 0.3 |
| WN18RR | uniform | 400 | pari bit | BSDC-S | 1024 | 9216 | 0.021 | 0.021 | 1.0 |
| WN18RR | dense | 100 | 0.50·tetto | MAP-B | 832 | 832 | 0.478 | 0.444 | 5.0 |
| WN18RR | dense | 100 | 0.50·tetto | MAP-I | 512 | 3584 | 0.469 | 0.432 | 5.0 |
| WN18RR | dense | 100 | 0.50·tetto | FHRR | 256 | 8192 | 0.469 | 0.432 | 5.0 |
| WN18RR | dense | 100 | 0.50·tetto | BSDC-OR | 768 | 768 | 0.507 | 0.401 | 5.2 |
| WN18RR | dense | 100 | 0.50·tetto | BSDC-S | 896 | 6272 | 0.525 | 0.523 | 5.0 |
| WN18RR | dense | 100 | 0.85·tetto | MAP-B | 2048 | 2048 | 0.800 | 0.854 | 4.0 |
| WN18RR | dense | 100 | 0.85·tetto | MAP-I | 1280 | 8960 | 0.796 | 0.849 | 4.1 |
| WN18RR | dense | 100 | 0.85·tetto | FHRR | 640 | 20480 | 0.796 | 0.849 | 4.1 |
| WN18RR | dense | 100 | 0.85·tetto | BSDC-OR | 1152 | 1152 | 0.847 | 0.840 | 3.5 |
| WN18RR | dense | 100 | 0.85·tetto | BSDC-S | 1664 | 11648 | 0.800 | 0.868 | 4.0 |
| WN18RR | dense | 100 | pari bit | MAP-I | 292 | 2044 | 0.310 | 0.247 | 4.6 |
| WN18RR | dense | 100 | pari bit | FHRR | 64 | 2048 | 0.158 | 0.111 | 3.8 |
| WN18RR | dense | 100 | pari bit | BSDC-OR | 2048 | 2048 | 0.945 | 0.997 | 1.5 |
| WN18RR | dense | 100 | pari bit | BSDC-S | 256 | 1792 | 0.147 | 0.109 | 3.7 |
| WN18RR | dense | 400 | 0.50·tetto | MAP-B | 4416 | 4416 | 0.470 | 0.406 | 4.8 |
| WN18RR | dense | 400 | 0.50·tetto | MAP-I | 2816 | 25344 | 0.471 | 0.407 | 4.8 |
| WN18RR | dense | 400 | 0.50·tetto | FHRR | 1408 | 45056 | 0.471 | 0.407 | 4.8 |
| WN18RR | dense | 400 | 0.50·tetto | BSDC-OR | 4096 | 4096 | 0.549 | 0.369 | 5.3 |
| WN18RR | dense | 400 | 0.50·tetto | BSDC-S | 4608 | 41472 | 0.500 | 0.477 | 4.8 |
| WN18RR | dense | 400 | 0.85·tetto | MAP-B | 10688 | 10688 | 0.810 | 0.871 | 3.8 |
| WN18RR | dense | 400 | 0.85·tetto | MAP-I | 6784 | 61056 | 0.809 | 0.870 | 3.8 |
| WN18RR | dense | 400 | 0.85·tetto | FHRR | 3392 | 108544 | 0.809 | 0.870 | 3.8 |
| WN18RR | dense | 400 | 0.85·tetto | BSDC-OR | 5632 | 5632 | 0.861 | 0.824 | 4.2 |
| WN18RR | dense | 400 | 0.85·tetto | BSDC-S | 9216 | 82944 | 0.830 | 0.913 | 3.5 |
| WN18RR | dense | 400 | pari bit | MAP-I | 1187 | 10683 | 0.229 | 0.133 | 4.3 |
| WN18RR | dense | 400 | pari bit | FHRR | 334 | 10688 | 0.129 | 0.065 | 3.8 |
| WN18RR | dense | 400 | pari bit | BSDC-OR | 10240 | 10240 | 0.951 | 0.999 | 1.8 |
| WN18RR | dense | 400 | pari bit | BSDC-S | 1024 | 9216 | 0.078 | 0.041 | 3.2 |

Differenza media prevista |contabilità − FKS ingenuo| nelle celle calib, per gruppo:
uniform 0.1–0.2 punti per ogni famiglia; FB15k-237 dense 1.7–4.2 (MAP-B/I, FHRR 2.1;
BSDC-OR 4.2; BSDC-S 1.7); WN18RR dense 4.4–8.3 (MAP-B 5.3, MAP-I 5.4, FHRR 5.4,
BSDC-OR 8.3, BSDC-S 4.4).

Regolarità previste (non criteri): a pari accuratezza MAP-I chiede ≈ 0.64 volte il D
di MAP-B (2/π) e FHRR la metà di MAP-I; BSDC-S chiede più componenti di BSDC-OR al
target alto. In bit, a pari bit con MAP-B a 0.85, BSDC-OR sta al tetto e MAP-I, FHRR,
BSDC-S sotto 0.3.

## Ipotesi e criteri, fissati ora

Gruppo = (famiglia, campione, dataset): 4 celle calib (2 N × 2 target). Per ogni gruppo
si riportano MAE (punti), massimo di cella, MAE del FKS ingenuo, errore con segno
medio e il suo IC al 95 % **a cluster** (errore per seed mediato sulle 4 celle, sd/√10,
t₉).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** MAP-B (controllo), uniform | MAE ≤ 3 in entrambi i gruppi | MAE > 5 in un gruppo |
| **H2** MAP-I, uniform (replica di FKS) | idem | idem |
| **H3** FHRR, uniform (replica di FKS) | idem | idem |
| **H4** BSDC-OR, uniform | idem | idem |
| **H5** BSDC-S, uniform | idem | idem |
| **H6 (rischiosa)** la previsione con contabilità regge sui sottografi densi | MAE ≤ 3 in tutti i 10 gruppi dense | MAE > 5 in almeno un gruppo |
| **H7 (ablazione)** nei gruppi WN18RR dense la contabilità riduce il MAE rispetto al FKS ingenuo | MAE_ingenuo − MAE ≥ X = 2 punti in tutti e 5 i gruppi | MAE_ingenuo − MAE < 0 in 2 o più gruppi |
| **H8** nessun bias oltre il campionamento, per famiglia | al più 2 delle 16 celle calib con \|t\| > 3.25 (t₉, bilaterale 0.01) | 5 o più celle |
| **H9** D validato fuori campione (scelto sui seed 0–4) | per famiglia, misurato medio sui seed 5–9 ≥ target − 3 punti in ≥ 15 celle su 16 | in ≤ 12 celle su 16 |

Il resto è **sostenuto in parte**. **Descrittivi, non ipotesi:** l'ordine fra famiglie
a pari bit (coppie con differenza prevista ≥ 5 punti, MAP-B misurato nella sua cella
0.85; chiave `descr_equal_bits_order`); l'errore con segno per gruppo con IC a cluster;
il confronto FKS ingenuo nei gruppi FB15k-237 dense e uniform.

**X = 2 punti, perché.** Nei gruppi WN18RR dense la differenza prevista fra le due
previsioni è 4.4–8.3 punti; il referee vede 5–7 punti di divario su un denso sintetico
FHRR; il nostro smoke (sintetico, N = 40) dà 5–8 punti di vantaggio sul MAE. X è circa
metà del divario previsto più piccolo: lascia spazio all'errore proprio del modello con
contabilità senza rendere l'ipotesi vuota. I gruppi FB15k-237 dense (divario 1.7–4.2)
sono troppo poco densi per un criterio.

**Potenza e falsificazioni spurie.** MDE per cella 2.2–5.8 punti (limite inferiore): una
singola cella non rileva 3 punti; per questo i criteri sono sul MAE di gruppo e su H8.
Sotto il modello vero: P(|t₉| > 3.25) = 0.01 per cella, quindi P(≥ 5 su 16) è
trascurabile anche con correlazione; il MAE di 4 celle supera 5 punti per solo rumore
con probabilità ≲ 1 % per gruppo se SE ≤ 1.5 punti, ≈ 10–20 % su 30 gruppi nel caso
peggiore (celle con MDE ≈ 5–6, N = 100). **FWER stimato analiticamente, non simulato su
sintetico** (raccomandazione non applicata, per tempo).

**Previsioni (non criteri).** H1 sostenuta. H2, H3 sostenute. H4 sostenuta. H5 **a
rischio**: allo smoke il modello discreto è pessimista di 2–4 punti (vedi sotto). H6
sostenuta per le famiglie gaussiane e BSDC-OR, a rischio per BSDC-S. H7 sostenuta.
H8 in parte (l'SE a cluster con 10 seed è rumoroso). H9 sostenuta tranne forse BSDC-S.

## Cosa ho visto prima

- Le previsioni qui sopra (`--predict`), che leggono i dati reali ma non costruiscono
  memorie (anche quelle della versione `9857511`, ora superate).
- Lo smoke del primo disegno (N = 40, 3 seed) e di questo (N = 40, 10 seed). In questo,
  MAE per gruppo: MAP-B 0.5–3.3, MAP-I 0.4–2.5, FHRR 0.9–2.3, BSDC-OR 0.5–2.6; MAE FKS
  ingenuo nei dense 6.6–9.5; ordine a pari bit 40/40.
- **BSDC-S: il modello è cambiato dopo lo smoke.** La prima versione, gaussiana
  (media B·W/L, varianza B·S2·p(1−p)), era ottimista di 10–13 punti allo smoke (coda
  destra del rumore poissoniano e pareggi interi). L'ho sostituita con la pmf discreta
  esatta del rumore, prima con il proprio vettore incluso nel rumore e poi tolto: allo
  smoke ora è **pessimista** di 1.7–3.6 punti (MAE). Non l'ho ritoccata oltre per non
  adattarla allo smoke. È la sola scelta di modello fatta dopo aver visto misure,
  sintetiche.
- Un test di tempo del primo disegno su sintetico (N = 400, 1 seed): BSDC 0.855 misurato
  contro 0.895 previsto. Con 1 seed e 200 query era rumore (SE ≈ 2.5 punti), non un
  effetto dell'occupazione media, che il referee stima ≤ 0.3 punti; comunque ora π usa
  l'occupazione realizzata.

## Tempo stimato

`--predict` 7.4 min; misure (110 righe × 10 seed, < 1 s ciascuna) ≈ 8 min: **≈ 15 min**
su questa macchina (12 core, un processo).

## Limiti dichiarati prima

- **Quale FKS.** Se H2/H3 reggono, il merito della forma funzionale è di FKS; la sola
  parte nuova sotto test è la contabilità (H6, H7).
- **SE a cluster con 10 seed**: stima rumorosa della varianza; l'MDE preregistrato usa un
  SE di modello che ignora la correlazione fra query della stessa memoria.
- **Celle correlate**: le 4 celle di un gruppo e le 5 famiglie usano gli stessi 10
  sottografi; i gruppi non sono prove indipendenti.
- **Bit**: formati di memorizzazione, non entropia (MAP-I compressa ≈ ½ log₂(2πeN) bit);
  codebook non contato; FHRR con fasi quantizzate costerebbe meno di 32 bit. Il blocco a
  pari bit è descrittivo e sfavorisce MAP-I, FHRR e BSDC-S per costruzione.
- **BSDC-S**: modello scelto dopo uno smoke sintetico (sopra), pareggi trattati in modo
  approssimato; niente soglia né CDT di Rachkovskij.
- **HRR** con convoluzione circolare e vettori gaussiani **non** è testato: l'unbinding
  approssimato aggiunge rumore di segnale che richiede un modello a parte, non
  validato in tempo. Solo FHRR.
- Atomi i.i.d., ρ = shift ciclico, encoding simmetrico; self-loop esclusi.
- Due carichi e due target; D è scelto sui seed 0–4 e misurato su tutti e 10 (H9 usa
  solo 5–9).
