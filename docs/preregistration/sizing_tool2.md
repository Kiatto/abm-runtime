# Preregistrazione 24 — il dimensionatore su grafi mai visti, ridisegnata per il tempo

**Data: 2026-10-15.** Committata **prima** di costruire qualsiasi memoria per misurare
un'accuratezza sui grafi reali. Harness
[`examples/sizing_tool2_prereg.py`](../../examples/sizing_tool2_prereg.py), che **importa**
l'harness congelato della 23 (`examples/sizing_tool_prereg.py`, non modificato) e ne riusa
caricamento e sha256, campionamento, `struct`, `predict_sub`, `Truth`, `d_min_measured`,
`baseline`, `_CountMake`, `summarize`, `verdict_boot`, `cluster_boot`, `power_cluster`, i
controlli sintetici e tutte le soglie. Previsioni congelate
`results/sizing_tool2_prereg_predictions.json`; tempi `results/sizing_tool2_prereg_timing.json`.

Prima del commit l'harness è stato eseguito solo con `--smoke` (grafi sintetici),
`--predict` (`recommend(verify=False)`: nessuna memoria), `--timing-real` (memorie su
sottografi reali **dedicati**, solo tempi), `--estimate` e `--power` (dalle previsioni).

## Seconda revisione prima dei dati (dopo `9aed475`): ritorno al disegno della 23

Decisione presa **prima di qualsiasi dato** e dichiarata come revisione del disegno: il
budget liberato dal fix dei thread si usa per tornare il più vicino possibile alla 23.
Questa sezione **prevale** su tutte le precedenti (anche sul punto 1 della prima revisione).

**Regola scritta prima di guardare le stime.** Passi in ordine di preferenza: (a) disegno
della 23 pieno (vero su 20 memorie; N = 1600 in tutti i ruoli: base k_in = 4 / k_un = 8,
val k_val = 6 / k_val_un = 30; HRR 256 MiB anche a N = 1600; D minimo 0.85, N ≤ 400, tutti;
baseline 0.85, N = 100, tutti); (b) come (a) con vero su 10; (c) come (b) con HRR a 64 MiB a
N = 1600; (d) il passo 4 precedente. Si sceglie il **primo** la cui stima × 1.5 sta sotto
**20 h** con 10 processi; watchdog, tetto per cella e tetto rigido 30 h invariati. (I passi
0–3 e 5–6 delle versioni precedenti sono tolti dalla griglia: non erano mai stati scelti.)

**Tempi a N = 1600, 1 thread** (`--timing-1600`, k = 90 000, solo tempi, 225 misure, massimo
di 3 ripetizioni, 2171 s; aggiunti a `results/sizing_tool2_prereg_timing_t1.json`).
Secondi per una memoria (costruzione + query su tutte le chiavi), mediana / max:
BSDC-OR 2.1 / 10.5; BSDC-S 3.1 / 14.6; FHRR 14.3 / 90.9; HRR 59.9 / 371; MAP-B 16.1 / 38.3;
MAP-I 23.7 / 86.1 (con BLAS libero erano 9.8–664 / 35–5287).

**Stime (makespan simulato, 10 processi):**

| passo | celle | CPU | makespan | × 1.5 | cella max |
|---|---|---|---|---|---|
| (a) 23 pieno | 5970 | 163 h | 16.8 h | 25.2 h | 2.6 h |
| **(b) vero su 10** | **5970** | **112 h** | **11.6 h** | **17.5 h** | **1.8 h** |
| (c) HRR 64 MiB a 1600 | 5970 | 99 h | 10.4 h | 15.5 h | 0.6 h |
| (d) passo 4 | 2874 | 7.3 h | 0.76 h | 1.13 h | 0.05 h |

**Scelto (b)** (`GRID_STEP = 1`). Fasi (cumulato): F1 2007 celle 4.0 h; F2 1915 celle 10.7 h;
F3 1938 celle (D minimo e baseline) 11.3 h; F4 110 celle 11.6 h; controlli per primi.

**Conseguenze.** L'assert di `select()` vale solo per (d). Le popolazioni tornano quelle
della 23: H3, H5, H6 su **tutti** i sottografi base (N = 100, 400, 1600), H1/H2/H1u/H2u con
le validated a N = 1600. I limiti 1 e 6 e il punto 1 della prima revisione sono superati:
N = 1600 è collaudato. Unica differenza di disegno dalla 23 rimasta: **vero su 10 memorie**
(nota sul rumore della prima revisione) e le modifiche d'esecuzione (fasi, celle, watchdog,
1 thread, sottografi nuovi). Le 249 validated previste a N = 1600 sono nel pool; quelle nei
sottografi scelti entrano.

**Cluster previsti (passo (b)):** 269 sottografi (192 base, 131 val). H1/H2: **70** cluster,
≤ 1177 validated; H1u/H2u: **61**, ≤ 830; H3/H5/H6: 192 base, H5 su 465 coppie, copertura
massima prevista **0.335** (H5 prevista falsificata); H4: 199 sottografi nel perimetro
(118 con validated); H7: 124 (91 con validated).

**Potenza (P sost. / P fals.):** H1, 70 cluster: p = 0.97 → 1.00 (ρ = 0), 0.96 (ρ = 0.2);
p = 0.95 → 0.51 / 0.05; p = 0.92 → 0 / 0.995 (ρ = 0), 0.03 / 0.63 (ρ = 0.2). H2: p = 0.92 →
0.99 / 0 e 0.87 / 0; p = 0.90 → ~0.5. H1u (61): p = 0.97 → 1.00 / 0.95; p = 0.92 → falsificata
0.95 / 0.56. H2u: p = 0.92 → 0.99 / 0.86. Come nella 23.

**Memoria:** la memoria stimata più grande arriva al limite di 1 GiB (MAX_MEM della 23, non
HRR) a N = 1600: fino a ~1.3 GB per processo, ~13 GB con 10 processi su 30 GB. Un OOM è
gestito dal watchdog (cella `timeout`, run che prosegue) e conta nel worst case.

**Smoke** rifatto al passo (b) (180 s): 10/10 controlli superati; ripresa idempotente; tetto 1 s → 24 timeout su 54; tetto rigido scaduto → tutte `not_started`; ValueError → `error`, SIGKILL → `timeout` dal watchdog, 6 celle vere `ok`; modello rotto: H2 0.19 su 72, H3 0.00 su 68.

## Revisione del disegno prima dei dati (dopo `edc1eb1`)

Revisione ostile applicata **prima** di qualsiasi accuratezza sui grafi reali. Questa sezione
prevale sulle sezioni sotto dove le contraddice.

1. **Popolazione di TUTTE le conclusioni = sottografi con N ∈ {100, 400}** (H1, H2, H1u,
   H2u comprese): `select()` esclude N = 1600 da ogni ruolo (asserito). Le **249
   raccomandazioni validated previste a N = 1600** nel pool (di cui **45 HRR**, elencate in
   `power.hrr_validated_1600_excluded` del file delle previsioni) **non sono collaudate**.
   Nessuna conclusione di questa preregistrazione generalizza a N = 1600, il regime in cui
   la verifica corregge di più (nella 23 la quota di D corretti cresceva con N).
2. `run_cell`: `CellTimeout` deriva da `BaseException`; ogni altra eccezione è registrata
   come `error` con il messaggio; un allarme tardivo nel `finally` è intercettato.
3. Una cella validated pianificata con status `absent` (recommend non la restituisce) conta
   come **fallimento** nel worst case ed è elencata fra le non misurate.
4. **Watchdog nel padre**: al più 10 celle in volo (`apply_async`); una cella in volo oltre
   min(inizio + tetto della cella, scadenza rigida) + 10 min è scritta `timeout`
   (`watchdog: true`), il pool si termina e si ricrea, le altre celle in volo (non scritte)
   tornano in testa alla coda e si rifanno. Un worker ucciso (OOM/SIGKILL) non blocca il run.
5. **Thread BLAS = 1** (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) fissati
   prima di importare numpy. Il primo `--timing-real` **non** li fissava (OpenBLAS
   multi-thread in 10 processi su 12 core: sovrascrittura). Ripetuto a 1 thread su N ≤ 400
   (`results/sizing_tool2_prereg_timing_t1.json`, 348 s): le memorie costano **~13 volte
   meno**; la stima del passo 4 scende da 9.9 h a **0.76 h (1.13 h con × 1.5)**. Il primo
   file resta come registro e non entra nella stima. Questo è probabilmente anche parte
   dell'errore di ~50× della 23 (stesso harness, stessi 6 processi senza vincolo di thread).
   La griglia non cambia: i passi 0–3 includono N = 1600, che la revisione esclude e che a
   1 thread non è stato misurato, quindi sono «non stimabili»; il passo 4 è il primo
   stimabile sotto il budget. **Dichiarato:** il budget permetterebbe ora molto di più
   (vero su 20 memorie, più cluster); non lo si usa per non cambiare il disegno dopo la
   revisione.
6. **Smoke rifatto** (passo 4, 1 thread, 4 processi, 176 s): 10/10 controlli superati;
   ripresa idempotente; tetto di 1 s per cella → 24 timeout su 54; tetto rigido scaduto →
   tutte `not_started`; **ValueError provocato → `error` registrato**, **worker ucciso con
   SIGKILL → `timeout` dal watchdog**, le 6 celle vere della stessa fase finite `ok`;
   modello rotto: quota H2 0.19 su 72, sensibilità H3 0.00 su 68 (falliscono, come devono).

Consigliate, applicate:
- **Conteggi della 23 verificabili**: `verify_counts_23()` (eseguita dallo smoke) controlla
  lo sha256 del checkpoint della 23
  (`ba2869f7746e8ef0adb62ed347e2dbaec0d7fdfda44a1262b1bde2a84d292882`) e legge SOLO i campi
  `family`, `tool_memories`, `adjusted`; asserisce che VERIFY_MEMS e P_ADJ ne sono le medie
  arrotondate (12–15 raccomandazioni per famiglia, solo FB15k-237).
- **Il passo 4, aggiunto a posteriori, allarga il perimetro di H4** (da N = 100 del passo 5
  a N ≤ 400): è stato aggiunto vedendo solo tempi.
- **H4/H7 «non testabili»**: con 57 e 51 sottografi nei perimetri e una quota di verificate
  vicina a quella descrittiva della 23 (0.90 per raccomandazione, ~6 celle per sottografo),
  la probabilità di avere meno di 30 cluster è trascurabile (< 0.01) se le celle si
  misurano; il rischio è un D minimo non trovato entro 8× (cluster senza d_min) o una
  baseline non verificata (H7 richiede verificate entrambe): non stimabile senza dati,
  dichiarato. **Un esito «non testabile» di H4/H7 non si ripara con un nuovo run** su
  questi dati: resta l'esito.
- **H5**: la copertura massima possibile dalle previsioni è 0.434 < 0.50, quindi p̂ < 0.50
  è certo: H5 si registra **falsificata** se boot_sup < 0.50 (regola della 23) e comunque
  **mai sostenuta**; non «non testabile» (64 cluster ≥ 30).
- **Rumore del vero a 10 memorie** (≈ 0.7 punti sulla media invece di 0.5): per H1/H2 spinge
  verso il fallimento (contro lo strumento); per H3/H6 nella direzione opposta: più errori
  apparenti vicino a 3 punti, classificati a caso rispetto all'affidabilità, il che
  **diluisce** la sensibilità e il risk ratio (contro lo strumento anch'esso per H6, ma
  gonfia il numero di positivi di H3).

## Cosa resta uguale alla 23

Ipotesi, unità, criteri e soglie: **H1/H2** replica in-distribuzione (4 grafi della
calibrazione), **H1u/H2u/H8u** grafi mai visti (UMLS, Kinship), **H3** sensibilità,
**H5** copertura, **H6** discriminazione, **H4** gonfiamento, **H7** valore contro la
baseline; secondari H1_q, H2_q, worst case con gli esclusi HRR, esito di cluster con
Clopper-Pearson. Bootstrap per cluster (sottografo), B = 10 000, seme 23 023; minimo
**30 cluster**. Dati e sha256, regola del tetto (target ≤ tetto − 0.02), vero primario su
tutte le chiavi, D minimo con il criterio della verifica su 20 memorie, baseline empirica
(8 + 8 semi, da D = 256), controlli sintetici (positivi, 6 avversari, negativo, modello
rotto) con gli stessi criteri. La verifica di `recommend` usa i default dello strumento:
**8 semi**, tol 0.01 (sono il prodotto e non si toccano). Il codice dell'analisi è
`P.summarize` della 23, chiamato sulle righe misurate.

## Cambiamenti rispetto alla 23 (motivati dal solo TEMPO)

La 23 è fallita «non testabile per tempo»: la stima sintetica era sbagliata di ~50 volte e
il tetto si controllava solo a fine sottografo. Cambia:

1. **Sottografi nuovi.** `SeedSequence([24_000_024, grafo, schema, N, k])`, k = 60 000 + j
   (pool come la 23: j < 8 sui grafi di calibrazione, j < 40 sui mai visti, j < 8 a
   N = 1600; 544 sottografi, 8152 righe di previsione). Nessun sottografo della 23 (5 già
   misurati) rientra. Semi per (sottografo, target) come la 23 con la nuova base.
2. **Tempi misurati su sottografi reali** (`--timing-real`): un sottografo per grafo ×
   schema × N con k = 90 000 (mai di valutazione), per ogni famiglia ai D previsti dei tre
   target e a 2× il massimo; costruzione (traccia pigra inclusa), query su Q e su tutte le
   chiavi; **massimo di 3 ripetizioni**; 10 processi in parallelo (il carico del run).
   **Nessuna accuratezza**: le risposte non si confrontano con nessun oggetto, e nei worker
   `sizing._measure` e `Truth.__call__` sono sostituiti da funzioni che falliscono
   (asserito). 747 misure, 6.7 h.
3. **Costo per raccomandazione ridotto**, regole scritte:
   - vero su **10** memorie fresche (23: 20);
   - **niente N = 1600 nelle ipotesi**: le popolazioni di H3, H5, H6 sono i sottografi base
     a N ≤ 400 (nella 23 includevano N = 1600). Con la griglia scelta N = 1600 non si misura
     (`k_1600 = 0`): una sola memoria HRR a N = 1600 costa fino a 88 min, una MAP-B o MAP-I
     fino a 20–23 min (tabella sotto);
   - **HRR**: max_mem 64 MiB (23: 256 MiB) e mai a N = 1600; le escluse sono elencate e
     contano nel worst case come nella 23;
   - **D minimo** (H4): target 0.85, N ≤ 400, **1 sottografo ogni 3** per cella (in ordine
     di priorità); 20 memorie come nella 23;
   - **baseline** (H7): target 0.85, N = 100, **1 sottografo ogni 2** per cella;
   - per cella k_in = 2 (base), k_val = 5, k_un = 4, k_val_un = 25.
4. **Ordine di priorità per fasi** (barriera fra le fasi):
   - **C** controlli sintetici **per primi**: nello smoke costano ~1 h di CPU (~6 min con
     10 processi);
   - **F1** le celle validated dei sottografi «val», grafi di calibrazione e mai visti
     **intercalati** (H1, H2, H1u, H2u): ≥ 30 cluster per gruppo prima di tutto il resto;
   - **F2** tutte le celle dei sottografi base N ≤ 400 (H3, H5, H6, H8u);
   - **F3** D minimo e baseline (H4, H7), solo su celle verificate di F1–F2;
   - **F4** il resto (famiglie non validated dei sottografi extra).
5. **Cella = (sottografo, target, famiglia)**. `recommend(families=[f])` dà lo stesso
   risultato della chiamata congiunta (la divisione semi/query dipende solo da grafo,
   rng_seed, seeds). **Tetto per cella** nel worker (SIGALRM):
   min(max(30 min, 10 × stima della cella), tempo rimasto al tetto rigido); la cella
   abbandonata è scritta come `timeout`, quelle non partite come `not_started`.
   **Checkpoint una riga JSONL per cella** con flush e fsync
   (`results/sizing_tool2_prereg_results.jsonl`); ripresa idempotente (le celle scritte,
   anche timeout, non si rifanno); inizio registrato in
   `results/sizing_tool2_prereg_meta.json`; un passo diverso blocca la ripresa.
6. **Celle non misurate (worst case, decisionale).** Per H1, H2, H1u, H2u si calcola anche
   la quota con ogni cella validated prevista e non misurata (timeout, non partita, errore)
   contata come **fallimento**. Decisione: «sostenuta» solo se è sostenuta sia sulle
   misurate sia nel worst case; «falsificata» se una delle due lo è e l'altra non è
   sostenuta; altrimenti non conclusiva. Il tetto per cella abbandona più facilmente le
   celle che la verifica corregge (più memorie): senza il worst case questo favorirebbe lo
   strumento. Per H3–H7 le non misurate sono escluse ed elencate.
7. **Budget**: la griglia è il primo passo la cui stima × 1.5 sta sotto **20 h** con 10
   processi; **tetto rigido 30 h** (orologio da muro dal primo avvio; il tempo di un crash
   conta).

## Tempi reali misurati (massimo di 3, 10 processi in parallelo) — PRIMA misura, BLAS multi-thread, superata dal punto 5 della revisione

Secondi per UNA memoria (costruzione + query su tutte le chiavi), sui 36 sottografi dei tempi:

| N | BSDC-OR | BSDC-S | FHRR | HRR | MAP-B | MAP-I |
|---|---|---|---|---|---|---|
| 100 (mediana / max) | 0.07 / 0.2 | 0.07 / 0.2 | 0.40 / 0.9 | 0.59 / 4.2 | 0.95 / 2.0 | 0.36 / 1.1 |
| 400 | 0.53 / 2.1 | 0.89 / 3.4 | 2.7 / 13.6 | 15.4 / 228 | 6.8 / 32.6 | 4.6 / 33.5 |
| 1600 | 9.8 / 41 | 13.3 / 35 | 91 / 602 | 664 / 5287 | 206 / 1184 | 221 / 1356 |

Ecco l'errore della 23: una raccomandazione costruisce ~20–60 memorie per la verifica e
20–40 per il vero; a N = 1600 una cella costa ore.

**Modello del costo di una cella** (`cell_cost`, nel codice): interpolazione lineare in D
(estrapolazione lineare oltre il massimo misurato), query scalate per M; rec = memorie
della verifica per famiglia a 1.25·D su Q (medie dei conteggi della 23: MAP-B 62.3, FHRR
51.7, MAP-I 34.7, HRR 32.7, BSDC-S 22.9, BSDC-OR 20.3) + 10 memorie a D e, con la
probabilità di correzione della 23 per famiglia, 10 a 1.5·D, su tutte le chiavi; D minimo e
baseline come nel modello della 23. Makespan simulato con 10 processi, celle in ordine e
barriera fra le fasi. **Dichiarato:** dal checkpoint della 23 ho letto solo conteggi di
memorie, quota di D corretti e secondi (non accuratezze); la quota corretta (39 %) era già
nell'esito pubblicato.

## Griglia scelta (stima con la PRIMA misura; vedi il punto 5 della revisione: a 1 thread il passo 4 stima 0.76 h, 1.13 h con il margine)

| passo | k_in | k_val | k_un | k_val_un | k_1600 | D minimo | baseline | HRR | celle | CPU | makespan | × 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 6 | 8 | 30 | 2 | 0.85, N ≤ 400, tutti | 0.85, N = 100, tutti | 256 MiB, anche 1600 | 5400 | 514 h | 52.8 h | 79.2 h |
| 1 | 3 | 6 | 6 | 30 | 1 | 1 su 2 | tutti | 128 MiB, no 1600 | 4248 | 278 h | 29.8 h | 44.6 h |
| 2 | 3 | 6 | 4 | 25 | 1 | 1 su 3 | tutti | 128 MiB | 3654 | 260 h | 28.0 h | 42.0 h |
| 3 | 2 | 5 | 4 | 25 | 1 | 1 su 3 | 1 su 2 | 64 MiB | 3048 | 243 h | 26.4 h | 39.6 h |
| **4** | **2** | **5** | **4** | **25** | **0** | **0.85, N ≤ 400, 1 su 3** | **0.85, N = 100, 1 su 2** | **64 MiB, no 1600** | **2874** | **95 h** | **9.9 h** | **14.9 h** |
| 5 | 2 | 5 | 4 | 25 | 0 | N = 100, 1 su 3 | 1 su 2 | 64 MiB | 2742 | 76 h | 7.9 h | 11.9 h |
| 6 | 2 | 4 | 4 | 20 | 0 | N = 100, 1 su 3 | 1 su 3 | 64 MiB | 2316 | 69 h | 7.1 h | 10.7 h |

**Scelto il passo 4** (`GRID_STEP = 4`). Il passo 4 è stato **aggiunto dopo** aver visto la
prima stima (che dava i passi 0–3 e 5–6): differisce dal 5 solo per il D minimo a N ≤ 400
(il perimetro di H4 della 23). Nessuna accuratezza era stata vista. Fasi stimate (makespan
cumulato): C ~0.1 h; F1 1550 celle, 4.6 h; F2 582 celle, 7.1 h; F3 648 celle (342 D minimo,
306 baseline), 9.6 h; F4 94 celle, 9.9 h. Cella più costosa stimata 1.2 h (tetto per
cella 12 h).

**Memoria:** la memoria di picco stimata più grande (`families.estimate_bytes`, tutte le
chiavi, fino a 2×D) è 169 MiB; con ~0.3 GB di Python e del sottografo, ~0.5 GB per
processo, ~5 GB con 10 processi (la macchina ha 12 core e 30 GB). I worker ricevono solo i
sottografi; il pool si ricrea ogni 50 celle.

## Cluster previsti per ipotesi (dalle previsioni, passo 4)

101 sottografi «val», 64 base (N ≤ 400), 140 sottografi in tutto, 2874 celle.

| ipotesi | cluster previsti | unità |
|---|---|---|
| H1, H2 | 50 sottografi con validated | ≤ 864 raccomandazioni validated (H1: le verified fra queste) |
| H1u, H2u | 51 | ≤ 686 |
| H8u | i sottografi dei mai visti misurati (≥ 51 + base) | verified |
| H3, H5, H6 | 64 sottografi base, N ≤ 400 | H5: 166 coppie; copertura massima prevista **0.434** < 0.50 |
| H4 | 57 sottografi nel perimetro, 41 con almeno una validated | celle verificate |
| H7 | 51 sottografi nel perimetro, 43 con validated | celle verificate da strumento e baseline |

H5 resta prevista falsificata prima dei dati (come nella 23: 0.337 lì). H4 e H7 possono
scendere sotto 30 cluster se poche celle del perimetro sono verificate: in quel caso
«non testabile», dichiarato ora.

## Potenza (passo 4; `power_cluster` della 23, 400 ripetizioni, bootstrap 1000)

P(sostenuta) / P(falsificata), sulle misurate (il worst case la riduce se ci sono celle
non misurate):

| p vero | 0.99 | 0.98 | 0.97 | 0.95 | 0.92 | 0.90 | 0.85 |
|---|---|---|---|---|---|---|---|
| H1, 50 cluster, ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.48 / 0.06 | 0 / 0.97 | 0 / 1 | 0 / 1 |
| H1, ρ = 0.2 | 1 / 0 | 1 / 0 | 0.93 / 0 | 0.51 / 0.05 | 0.06 / 0.51 | 0 / 0.88 | 0 / 1 |
| H2, ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.98 / 0 | 0.47 / 0.05 | 0 / 0.99 |
| H2, ρ = 0.2 | 1 / 0 | 1 / 0 | 1 / 0 | 0.99 / 0 | 0.83 / 0.003 | 0.55 / 0.02 | 0.02 / 0.68 |
| H1u, 51 cluster, ρ = 0 | 1 / 0 | 1 / 0 | 1.00 / 0 | 0.56 / 0.04 | 0 / 0.94 | 0 / 1.00 | 0 / 1 |
| H1u, ρ = 0.2 | 1 / 0 | 0.98 / 0 | 0.92 / 0 | 0.57 / 0.03 | 0.07 / 0.47 | 0.003 / 0.82 | 0 / 1.00 |
| H2u, ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.96 / 0 | 0.54 / 0.06 | 0 / 0.98 |
| H2u, ρ = 0.2 | 1 / 0 | 1 / 0 | 1 / 0 | 1.00 / 0 | 0.83 / 0.003 | 0.51 / 0.05 | 0.03 / 0.67 |

Quasi identica alla 23 (68 e 61 cluster lì): la potenza è dominata dalla soglia, non dal
numero di cluster sopra 50. Il vero su 10 memorie invece di 20 aumenta l'errore di misura
della media per cella (≈ 0.7 punti invece di 0.5): per H1/H2 le soglie restano quelle (il
rumore di misura spinge verso il fallimento, cioè contro lo strumento). H3, H6, H4, H7:
nessuna potenza in forma chiusa, come nella 23.

## Smoke (sintetico)

Eseguito con l'ultimo passo allora presente (identico al passo 4 su grafi sintetici N = 100:
le differenze riguardano N ≥ 400 e N = 1600), 3 sottografi sintetici + 10 controlli,
4 processi, 1139 s: **tutti i 10 controlli superati**; ripresa idempotente (nessuna riga
nuova); tetto per cella di 1 s → 47 timeout su 54 celle registrati e conteggiati come
fallimenti nel worst case (H1 worst case falsificata, come deve); tetto rigido scaduto →
tutte le celle `not_started`; modello rotto (+5 punti): quota H2 0.19 su 72, sensibilità H3
0.00 su 68 (devono fallire: falliscono). Controlli: 3571 s di CPU → vanno per primi.

## Regola di arresto

Il run (`python examples/sizing_tool2_prereg.py`) esegue C, F1, F2, F3, F4 in ordine;
nessun riepilogo intermedio; il riepilogo si calcola alla fine o al tetto di 30 h sulle
celle scritte, con le non misurate nel worst case. Un cambio di passo o di qualunque
parametro dopo l'avvio è una deviazione da dichiarare. Se il run supera 30 h il codice si
ferma da solo (SIGALRM per cella e `not_started` per le restanti).

## Limiti dichiarati prima

1. **Tutte** le ipotesi (H1, H2, H1u, H2u, H3–H8u) parlano di N ∈ {100, 400}; N = 1600 non è
   collaudato (la 23 lo includeva): 249 validated previste lì restano fuori.
2. HRR escluso a N = 1600 e oltre 64 MiB: le escluse validated contano come fallimenti nel
   worst case della 23 (H1_worst_case) e sono elencate.
3. Vero su 10 memorie: errore di misura maggiore (H3/H6 classificano con più rumore errori
   vicini a 3 punti).
4. Perimetri di H4 (1 sottografo su 3, N ≤ 400) e H7 (1 su 2, N = 100) più stretti.
5. La stima usa conteggi medi di memorie della 23 (solo FB15k-237); è il punto più debole
   del modello del costo, coperto dal margine × 1.5 e dal tetto per cella.
6. Il tetto per cella e il watchdog possono abbandonare preferibilmente celle corrette dalla
   verifica: per questo il worst case (timeout, error, absent, not_started come fallimenti)
   è decisionale per H1/H2/H1u/H2u; e nessuna conclusione vale a N = 1600, dove la
   verifica corregge di più.
7. Il passo 4 è stato aggiunto alla griglia dopo la prima stima dei tempi (solo tempi).

## Hash

sha256 nel messaggio del commit di revisione (harness, documento, previsioni, tempi).
