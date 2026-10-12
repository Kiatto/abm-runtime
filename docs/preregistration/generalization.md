# Preregistrazione 22 — la stima a priori del cleanup generalizza?

**Data: 2026-10-12.** Committata **prima** di misurare, insieme all'harness
[`examples/generalization_prereg.py`](../../examples/generalization_prereg.py), alla
diagnosi sintetica `results/generalization_diagnosis.json` e alle previsioni congelate
`results/generalization_prereg_predictions.json`. Prima del commit l'harness è stato
eseguito solo con `--diagnose`, `--verify`, `--verify-tails`, `--smoke` (grafi sintetici
generati nello script), `--postdict` (risultati già pubblicati della 21, nessuna misura)
e con `--predict` e `--timing` (`--predict` legge CoDEx-M e YAGO3-10, campiona i sottografi
e le catene, sceglie D e calcola le previsioni; **non costruisce nessuna memoria** sui
dati reali; `--timing` misura il tempo su un grafo sintetico). Nessuna misura su
CoDEx-M, YAGO3-10, FB15k-237 o WN18RR è stata fatta per questa preregistrazione. Stesse regole di
[`vsa_families.md`](vsa_families.md) (prereg. 21), di cui riusa il codice importandolo,
senza modificarlo.

## Revisione del disegno prima dei dati (2026-10-12, dopo `53f99a5`)

Revisione ostile del commit `53f99a5`, applicata prima di qualsiasi misura sui grafi reali.
Questa sezione è committata **prima** di lanciare `--predict` su YAGO3-10 (e prima di
rilanciarlo su CoDEx-M): le regole qui sotto non hanno visto previsioni di YAGO3-10.
Tabelle, ipotesi e tempo più sotto sono quelle di `53f99a5` e **vengono sostituite** nel
commit successivo, che riporta le nuove previsioni.

1. **H9, comparatore.** Il comparatore delle catene in `53f99a5` (FKS ingenuo per hop,
   un oggetto, D tarato sulla media a 1 hop) misurava anche un effetto di selezione. Ora
   H9 usa `predict_naive_sig`: conserva la firma della query (numero di oggetti veri;
   sulle catene funzionali è 1), toglie solo gemelli (S2 = N) e alias.
2. **Grafo denso nuovo: YAGO3-10** (repo pubblico `TimDettmers/ConvE` al commit
   `f3c0eb286025410fa5b4c04696c918264163d0ca`, `YAGO3-10.tar.gz`, sha256
   `9a3bdd5fd08c064bd4b9ffdbe972f37ba0b8799f4e9f868ff70f5d2f5331fb92`, in
   `data/external/yago3_10/`, voce in `examples/replicate.py`; l'harness legge
   `train.txt` dall'archivio). Stesso impianto di CoDEx-M (carichi, seed, campioni).
   **Regola di ammissione al test della contabilità** (`admission`, usa solo
   `--predict`, nessuna memoria): un grafo entra nel test della contabilità (H2 come
   test e H5) se il divario previsto medio |contabilità − FKS ingenuo| nelle celle dense
   a 1 hop è ≥ 2 punti in **almeno 3 famiglie**. Altrimenti H2 su quel grafo è una
   replica di FKS. **CoDEx-M non testa la contabilità**: lì H1/H2 sono repliche di FKS
   (più le molteplicità di HRR, che sono un modello nuovo, H4).
   **Tempo e riduzione della griglia** (regola scritta ora): se la stima del run
   (`--predict` di entrambi i grafi + misure stimate con `--timing`) supera 3 h, si
   tolgono nell'ordine (1) N = 3200 di YAGO3-10, (2) N = 3200 di CoDEx-M, (3) N = 1600 di
   YAGO3-10, fermandosi al primo passo sotto le 3 h; la stima dopo ogni passo è la somma
   dei tempi per (grafo, N) misurati da `--predict` e `--timing`. Se anche dopo (3) si
   supera, il run si fa così.
3. **H3 simmetrica**: sostenuta se |errore con segno| ≤ 1.0 punti in entrambi i gruppi
   a 1 hop (per grafo), falsificata se |errore con segno| ≥ 1.5 in almeno uno.
   **Percorso dipendente dall'esito, dichiarato:** la ricerca della correzione di BSDC-S
   è partita dal bias osservato sui dati reali della 21; la diagnosi è stata fatta su
   sintetico, ma la domanda nasceva da quell'esito. **Postdizione** di `bsdc_s_exact`
   sulle 16 celle BSDC-S già misurate della 21 (`--postdict`, nessuna nuova misura,
   risultati pubblicati della 21): riportata sotto, etichettata come postdizione.
4. **Catene, scala per hop:** H6 e H9 confrontano misura^(1/h) con previsione^(1/h) (per
   h = 1 è l'accuratezza), con le stesse soglie in punti del singolo hop (3 e 5): un
   errore di 3 punti per hop. **Escluse da H6 e H9 le celle con MDE per hop > 5 punti**
   (MDE per hop = MDE / (h · p^((h−1)/h)), metodo delta).
5. **MAP-B: atomi salati con il seed** (nome + "#" + sale del sottografo): i seed sono
   indipendenti anche per MAP-B. La previsione (`exact`) non dipende dagli atomi. Lo
   smoke verifica che traccia e risposte dell'Engine coincidano con `abm.Memory.store` /
   `query` sugli stessi nomi salati.
6. **Gradi di libertà = n_seed − 1 per cella** (19 per le celle a 1 hop, 9–19 per le
   catene): MDE, IC e soglie di t usano i quantili di Student di quei gdl.
7. **Provenienza:** le soglie X sono calcolate dall'harness (`thresholds`) **insieme**
   alle previsioni, con la regola scritta (metà del divario previsto, per difetto al
   mezzo punto, minimo 1 punto); push e marche OpenTimestamps li fa il coordinatore dopo
   il commit.
8. **Gruppi incompleti o assenti:** in H6 e H9, G = numero di gruppi catena con almeno
   una cella dopo l'esclusione del punto 4 (e, per H9, con X definito). H6: sostenuta se
   MAE ≤ 3 in tutti i G gruppi, falsificata se > 5 in almeno uno. H9: sostenuta se il
   guadagno ≥ X in almeno ⌈5G/6⌉ gruppi, falsificata se < 0 in almeno ⌈G/4⌉. Con G = 0
   l'ipotesi è **non testabile** (riportata così).

Consigliate, applicate: verifica sintetica di HRR alle code (N = 3200, M ≈ 4000,
`--verify-tails`); MAE pesato per l'inverso della varianza (descrittivo); docstring
corretta sul modello BSDC-S congelato; `hrr_struct` non più ricalcolato a ogni chiamata.
Il SeedSequence ora include l'indice del grafo: i sottografi di CoDEx-M cambiano rispetto
a `53f99a5` (nessuna misura in mezzo).

## Perché

La 21 ha sostenuto H1–H9: la previsione a priori dell'accuratezza del cleanup, con la
contabilità della struttura (gemelli peso 2, alias, risposte multiple, codebook), regge
per MAP-B, MAP-I, FHRR, BSDC-OR, BSDC-S su FB15k-237/WN18RR. Limiti dichiarati: BSDC-S
pessimista di ≈ +1.7–2.1 punti; N ≤ 400; un solo livello di binding; HRR non testato;
due grafi. Qui si attacca ciascun limite: diagnosi del bias di BSDC-S (passo 1), HRR
(passo 2), catene a 2 e 3 hop (passo 3), un grafo nuovo (CoDEx-M, passo 4), carichi
fino a N = 3200 e 20 seed (passo 5).

## Passo 1 — diagnosi del bias di BSDC-S (solo sintetico)

**Ipotesi principale (del revisore):** il rumore di conteggio è correlato fra candidati
che condividono atomi (la stessa traccia, gli stessi bin), e il modello della 21 lo
tratta come indipendente.

**Esperimento** (`--diagnose`, grafo `v21.synthetic(300 entità, 6 relazioni, 3000
fatti)`, seed 22_000_122, 16 sottografi per impianto, 5 memorie per sottografo; uniform
e dense; N ∈ {100, 400}; B ∈ {8, 12, 16} e {12, 18} blocchi). Cinque colonne:
misura; modello della 21; **le ipotesi della 21 campionate esattamente** (rumore
marginale esatto, candidati indipendenti, pareggi divisi a caso come nella misura, MC
300 prove per firma); **modello condizionato alla traccia** (i nulli sono i.i.d. *dato*
l'istogramma realizzato dei B blocchi; media su 400 tracce estratte dal modello
generativo — è la forma esatta della correlazione indotta dalla traccia condivisa, e per
Jensen va nella direzione del bias); **pareggi esatti analitici** (ipotesi della 21,
pareggi esatti, per programmazione dinamica: nessun MC).

| campione | N | B | misura (± SE) | 21 | ipotesi 21, pareggi esatti (MC) | condizionato alla traccia | pareggi esatti, analitico |
|---|---|---|---|---|---|---|---|
| uniform | 100 | 8 | 0.545 ± 0.005 | 0.510 | 0.528 | 0.531 | 0.531 |
| uniform | 100 | 12 | 0.782 ± 0.005 | 0.752 | 0.772 | 0.771 | 0.771 |
| uniform | 100 | 16 | 0.893 ± 0.003 | 0.881 | 0.891 | 0.893 | 0.893 |
| uniform | 400 | 12 | 0.677 ± 0.004 | 0.657 | 0.674 | 0.674 | 0.674 |
| uniform | 400 | 18 | 0.845 ± 0.003 | 0.839 | 0.845 | 0.848 | 0.848 |
| dense | 100 | 8 | 0.595 ± 0.006 | 0.590 | 0.601 | 0.601 | 0.601 |
| dense | 100 | 12 | 0.754 ± 0.006 | 0.732 | 0.741 | 0.743 | 0.743 |
| dense | 100 | 16 | 0.825 ± 0.005 | 0.815 | 0.825 | 0.823 | 0.823 |
| dense | 400 | 12 | 0.650 ± 0.005 | 0.646 | 0.658 | 0.655 | 0.655 |
| dense | 400 | 18 | 0.794 ± 0.004 | 0.786 | 0.793 | 0.793 | 0.793 |

**Esito della diagnosi.**

- **L'ipotesi principale è respinta** su questo sintetico: il modello condizionato alla
  traccia (che contiene tutta la correlazione da traccia condivisa e da collisioni nei
  bin) coincide con il modello a candidati indipendenti entro 0.3 punti in tutti i 10
  impianti. La correlazione esiste ma il suo effetto è sotto la risoluzione.
- **La causa è l'approssimazione dei pareggi** della 21 (quota proporzionale al rischio
  f/F dei candidati al massimo): con i conteggi interi i pareggi al massimo sono
  frequenti, e quella quota sottostima la vittoria dell'oggetto vero. Le stesse
  ipotesi con i pareggi esatti (MC) tolgono quasi tutto il bias: errore con segno medio
  misura − modello da +1.53 (21) a +0.30 punti (analitico) sui 10 impianti; scarti
  −0.6…+1.4 punti (residui più grandi: uniform N = 100, B = 8, +1.4, e dense N = 100,
  B = 12, +1.1, 2–3 SE; non spiegati).
- **Correzione congelata** (derivata, non tarata: nessun parametro): `bsdc_s_exact`,
  le ipotesi della 21 con i pareggi esatti. Per ogni valore x del massimo, programmazione
  dinamica sul numero di veri (t_g) e di alias (t_a) che valgono x, nulli tutti ≤ x con
  probabilità F(x)^n e J ~ Bin(n, f(x)/F(x)) nulli pari a x; quota t_g/(t_g + t_a + J).
  Verificata contro il MC delle stesse ipotesi (0.7399 contro 0.7406 ± 0.003 con
  20 000 prove). Il modello condizionato alla traccia resta nell'harness solo per la
  diagnosi. Le righe BSDC-S riportano anche la previsione della 21 (`pred_v21`).
- Nessun dato di FB15k-237, WN18RR o CoDEx-M è stato usato in questo passo.

## Passo 2 — HRR

Atomi a ~ N(0, 1/D)^D reali; fatto f = s ⊛ ρ(r) ⊛ o (⊛ convoluzione circolare via FFT,
ρ = shift ciclico); traccia T = Σ_j w_j f_j salvata in float32 (32 bit/componente);
unbinding per involuzione y = k* ⊛ T, k = s ⊛ ρ(r); cleanup argmax ⟨y, c⟩ sul codebook.

**Varianza del rumore** (Plate 1995, con le molteplicità degli atomi). In Fourier,
⟨y, c⟩ = (1/D) Σ_ν conj(K_ν) T_ν conj(C_ν), con E|A_ν|² = 1 e componenti complesse
gaussiane circolari (si ignorano ν = 0, D/2). Per il vettore j il termine
Z_ν = conj(S R C)_ν (A₁A₂A₃)_ν ha E|Z|² = Π_a k_a! (k_a = molteplicità dell'atomo a nel
multinsieme {s, r, c} ∪ atomi(f_j), perché E|X|^{2k} = k!; ρ non cambia |R_ν|).

- **Segnale** (multinsiemi uguali: oggetto vero o alias): Z = |S|²|R|²|C|² reale,
  E Z = 1, E Z² = 8; contributo w_j alla media e 2·(8 − 1)·w_j²/D = **14 w_j²/D** alla
  varianza (le D frequenze sono D/2 coppie coniugate).
- **Rumore**: E Z = E Z² = 0, contributo **w_j² Π_a k_a!/D**. Un fatto che non condivide
  atomi con la query e il candidato vale 1/D come in MAP-I; uno che condivide s, r o c
  vale 2/D; i fatti con la stessa chiave (s, r) — gli altri oggetti veri e gli alias —
  valgono 4/D; il fatto del candidato stesso come rumore per gli altri 4/D.
- Termini di vettori diversi scorrelati. Previsione: punteggi gaussiani indipendenti
  con quelle medie e varianze (nulli raggruppati per varianza, relativa 10⁻³), argmax per
  integrazione numerica. Approssimazione dichiarata: si ignora la fluttuazione comune di
  |K_ν|² condivisa dai candidati di una query (ordine √(3/D) sulla varianza).
- **Comparatore ingenuo per HRR:** la gaussiana di MAP-I (v = 1) senza molteplicità.

**Verifica Monte Carlo sintetica** (`--verify`, seed 22_000_222, 16 sottografi, D
scelto per 0.85 × tetto): HRR a 1 hop, previsto contro misurato — uniform N = 100: 0.816
contro 0.817 ± 0.011; N = 400: 0.765 contro 0.768 ± 0.007; dense N = 100: 0.762 contro
0.766 ± 0.010; N = 400: 0.739 contro 0.748 ± 0.008. Nello smoke (N = 40, 80, 20 seed) il
MAE di gruppo è 1.6 (uniform) e 0.5 (dense) punti, quello del comparatore ingenuo 8.9 e
14.2: senza le molteplicità HRR è previsto ottimista di 10–15 punti. Congelato così.
(La verifica ha usato il raggruppamento dei nulli a 10⁻⁴; il congelato usa 10⁻³, per
tempo: effetto ≪ 0.1 punti.)

## Passo 3 — catene a 2 e 3 hop

**Protocollo** di `Memory.chain`: a ogni hop query (nodo, r_i) e cleanup sull'intero
codebook; il nodo restituito è il soggetto dell'hop dopo. **Catene**: percorsi
s₀ −r₁→ e₁ −r₂→ … nel sottografo in cui ogni (e_{i−1}, r_i) ha **un solo** oggetto vero
(percorso funzionale) e i nodi sono distinti; fino a 100 per seed e per h, per
passeggiate a caso (flusso casuale separato). Corretta = nodo finale = e_h (un recupero
per caso dopo un hop sbagliato conta come corretto: probabilità ≲ 1/M).

**Previsione.** Per MAP-I, FHRR, HRR, BSDC-OR, BSDC-S: Π_i p(e_{i−1}, r_i), con p la
previsione a 1 hop della famiglia per quella query (contabilità inclusa). Derivazione
dell'indipendenza: il solo termine condiviso fra l'hop i e l'hop i+1 è ⟨f_i, f_{i+1}⟩
(il fatto dell'altro hop visto come rumore), covarianza w_i w_{i+1}/D su varianze ≈ S2/D:
correlazione ≈ 1/S2 ≤ 1 % a N ≥ 100; i nulli dei due hop leggono chiavi diverse. Per
**MAP-B** il voto di maggioranza introduce la covarianza per bit −ρ² della reference: la
previsione è Π_i p_i × dep_h, dep₂ = both/single² da `exact.two_hop_joint_fast(N, D, M)`,
dep₃ = `exact.chain_accuracy_mc(N, D, M, 3, 20 000)`/single³ (N = vettori distinti, M =
codebook, entrambi medi sui seed della cella: un solo fattore per cella; nella
verifica sintetica invece per sottografo): il modello della dipendenza fra hop esistente (paper §test 9),
applicato come fattore alla previsione con contabilità (approssimazione: il fattore è
calcolato per fatti uniformi).

**Verifica sintetica** (`--verify`, 16 sottografi, D per 0.85 × tetto; tra parentesi il
numero di catene; errore = misura − previsione, in punti, ± SE fra sottografi):

| campione, N | h | MAP-B | MAP-I | FHRR | HRR | BSDC-OR | BSDC-S |
|---|---|---|---|---|---|---|---|
| uniform 400 (1600) | 2 | −1.1 ± 1.4 | +1.2 ± 1.4 | −1.7 ± 1.0 | −0.9 ± 1.1 | −0.1 ± 1.1 | −0.8 ± 2.1 |
| uniform 400 (1600) | 3 | −1.9 ± 1.7 | −2.0 ± 1.5 | −1.2 ± 1.2 | +0.1 ± 1.3 | +1.3 ± 1.3 | −0.5 ± 1.4 |
| dense 400 (1600) | 2 | +2.8 ± 2.3 | +1.4 ± 2.0 | −0.7 ± 2.1 | +3.6 ± 1.9 | −1.8 ± 1.2 | +0.9 ± 2.0 |
| dense 400 (1600) | 3 | +2.4 ± 2.4 | −1.2 ± 1.7 | −1.6 ± 2.5 | +3.9 ± 2.4 | 0.0 ± 1.1 | +0.5 ± 2.1 |
| dense 100 (1304 / 413) | 2 | +3.3 ± 3.5 | +0.3 ± 3.3 | +6.1 ± 3.4 | −1.4 ± 3.3 | −2.2 ± 1.7 | −4.9 ± 2.3 |
| dense 100 (1304 / 413) | 3 | +14.8 ± 8.4 | +4.7 ± 7.5 | +10.4 ± 8.1 | −3.4 ± 6.3 | −3.6 ± 2.9 | +0.3 ± 7.1 |
| uniform 100 (452 / 122) | 2 | −3.3 ± 2.1 | +5.0 ± 2.4 | −2.0 ± 3.0 | −3.1 ± 2.1 | +2.6 ± 3.0 | −1.0 ± 2.7 |
| uniform 100 (452 / 122) | 3 | −4.0 ± 4.6 | +6.7 ± 5.4 | −7.6 ± 6.5 | −5.0 ± 6.1 | +5.0 ± 4.8 | −4.9 ± 6.4 |

Con ≥ 1600 catene gli errori stanno entro ≈ 2 SE e sotto 4 punti; con poche catene
(dense N = 100 a 3 hop, uniform N = 100) l'SE fra sottografi arriva a 5–8 punti e il
test non discrimina. Nessun segno sistematico comune alle famiglie. Congelato così.

## Passo 4 — dati nuovi: CoDEx-M (e YAGO3-10, dalla revisione)

CoDEx-M (Safavi e Koutra, EMNLP 2020), train split, dal repo pubblico
`github.com/tsafavi/codex` al commit `3132e426c2a6b643b70bad679905a3a6270be440`
(`data/triples/codex-m/train.txt`, 185 584 triple), in `data/external/codex/`,
sha256 `d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556`, aggiunto a
`examples/replicate.py` (DATASETS). Letto **solo** per campionare la struttura
(sottografi, query, catene) in `--predict`; nessuna memoria costruita.

## Impianto (passo 5)

- **Famiglie**: MAP-B, MAP-I, FHRR, BSDC-OR come nella 21; BSDC-S con i pareggi esatti
  (passo 1); HRR (passo 2). Stesso encoding f = s ∘ ρ(r) ∘ o per tutte.
- **Sottografi** di CoDEx-M e YAGO3-10: N ∈ {100, 400, 1600} (griglia ridotta, vedi sotto; era {100, 400, 1600, 3200}); *uniform* e *dense* (BFS non
  orientata, `v21.sample`); **20 seed**; flussi separati
  `SeedSequence([22 000 022, grafo, campione, N, seed]).spawn(3)` → campionamento e query,
  atomi, catene. Self-loop esclusi, triple deduplicate.
- **Query**: tutte le (s, r) distinte, al più 200 per seed; catene h ∈ {2, 3}, al più 100
  per seed. **Una cella catena usa i seed con ≥ 10 catene, ed esiste se questi sono ≥
  10** (SE a cluster su quei seed, t con n_seed − 1 gdl; la validazione fuori campione
  e H7 riguardano solo le celle a 1 hop). *Revisione del disegno prima dei dati:* la
  prima regola ("ogni seed ≥ 10 catene") lasciava una sola famiglia di celle catena
  (uniform h = 2, N ≥ 1600); vista la sola struttura (conteggi di catene, nessuna
  misura) è stata sostituita da questa. Nient'altro è cambiato dopo `--predict`.
- **Regola per D** (come la 21): per (campione, N, famiglia, target t ∈ {0.50, 0.85}),
  il minimo multiplo di 64 (BSDC: di L) con previsione media a 1 hop **sui seed 0–9** ≥
  t × tetto (tetto = media di g/(g + a) sui seed 0–9), D ≤ 2¹⁸; altrimenti la cella non
  esiste. Le catene si misurano sulla stessa memoria della cella a 1 hop (stesso D).
- **Bit per componente**: come la 21; HRR 32 (float32).
- **Statistica per cella**: e_k = misurato_k − previsto_k per seed; **SE a cluster** =
  sd(e_k)/√n_seed; t con **n_seed − 1 gdl** (19 a 1 hop). MDE per cella = (t₀.₉₇₅,gdl + t₀.₈₀,gdl)·SE
  (2.954·SE con 19 gdl), con SE di modello (Poisson-binomiale per seed, query indipendenti): limite
  inferiore. MDE di gruppo = 2.954·√(Σ SE²)/n_celle.
- **Fuori campione**: D scelto sui seed 0–9, H9 usa solo i seed 10–19.
- **Comparatore FKS ingenuo** in ogni cella (come la 21; per HRR la gaussiana di MAP-I;
  per BSDC-S `bsdc_s_exact` con N fatti di peso 1 e un oggetto); per le catene il
  prodotto delle previsioni ingenue degli hop, senza fattore di dipendenza.

## Postdizione di BSDC-S sulle celle della 21 (non una nuova misura)

`--postdict`: le 16 celle BSDC-S calib della 21, **già misurate e pubblicate**
(`results/vsa_families_prereg_results.json`), con i sottografi della 21 ricampionati dal
suo codice (il controllo riproduce le sue previsioni) e la previsione ricalcolata con
`bsdc_s_exact`. È una **postdizione**: il modello è stato scelto dopo che quel bias era
noto (vedi revisione, punto 3), quindi non conta come conferma; serve a dire se la
correzione spiega il bias osservato. File: `results/generalization_postdiction_bsdc_s.json`.

| dataset | campione | N | target | D | misurato (21) | previsto 21 | previsto `bsdc_s_exact` |
|---|---|---|---|---|---|---|---|
| fb15k237 | uniform | 100 | 0.5 | 1152 | 0.545 | 0.521 | 0.542 |
| fb15k237 | uniform | 100 | 0.85 | 1920 | 0.867 | 0.851 | 0.867 |
| fb15k237 | uniform | 400 | 0.5 | 5632 | 0.507 | 0.499 | 0.517 |
| fb15k237 | uniform | 400 | 0.85 | 9216 | 0.883 | 0.856 | 0.870 |
| fb15k237 | dense | 100 | 0.5 | 896 | 0.570 | 0.558 | 0.575 |
| fb15k237 | dense | 100 | 0.85 | 1664 | 0.861 | 0.850 | 0.865 |
| fb15k237 | dense | 400 | 0.5 | 5120 | 0.570 | 0.537 | 0.556 |
| fb15k237 | dense | 400 | 0.85 | 8704 | 0.890 | 0.876 | 0.890 |
| wn18rr | uniform | 100 | 0.5 | 1152 | 0.581 | 0.545 | 0.567 |
| wn18rr | uniform | 100 | 0.85 | 1920 | 0.883 | 0.863 | 0.879 |
| wn18rr | uniform | 400 | 0.5 | 5632 | 0.517 | 0.499 | 0.517 |
| wn18rr | uniform | 400 | 0.85 | 9216 | 0.865 | 0.855 | 0.870 |
| wn18rr | dense | 100 | 0.5 | 896 | 0.535 | 0.525 | 0.541 |
| wn18rr | dense | 100 | 0.85 | 1664 | 0.815 | 0.800 | 0.814 |
| wn18rr | dense | 400 | 0.5 | 4608 | 0.517 | 0.500 | 0.510 |
| wn18rr | dense | 400 | 0.85 | 9216 | 0.861 | 0.830 | 0.840 |

| gruppo | MAE 21 | errore con segno 21 | MAE corretto | errore con segno corretto |
|---|---|---|---|---|
| fb15k237 dense | 1.74 | +1.74 | 0.60 | +0.12 |
| fb15k237 uniform | 1.88 | +1.88 | 0.64 | +0.13 |
| wn18rr dense | 1.81 | +1.81 | 0.85 | +0.55 |
| wn18rr uniform | 2.08 | +2.08 | 0.56 | +0.31 |

Il bias della 21 (+1.7…+2.1 punti) scende a +0.1…+0.6 punti; il gruppo WN18RR dense
resta il peggiore (+0.55).

## Previsioni, calcolate ora (`--predict`)

**Ammissione al test della contabilità (regola della revisione, punto 2), esito:**
nessuno dei due grafi è ammesso. Divario medio previsto |contabilità − FKS ingenuo| nelle
celle dense a 1 hop (griglia completa con N = 3200, calcolato prima della riduzione;
identico nell'esito sulla griglia ridotta): **YAGO3-10** MAP-B 0.9, MAP-I 0.9, FHRR 0.9,
BSDC-OR 1.0, BSDC-S 0.7 punti, HRR 18.5 (solo HRR ≥ 2: 1 famiglia su 3 richieste);
**CoDEx-M** (griglia ridotta; esito identico) 0.8, 0.8, 0.8, 1.6, 0.8, HRR 20.6. Il divario di HRR viene dalle
molteplicità (modello nuovo, H4), non da gemelli e alias. Quindi **in questa
preregistrazione la contabilità non è testata**: H2 è una replica di FKS su entrambi i
grafi e H5 è **non testabile**. Il test della contabilità resta quello della 21
(WN18RR dense).

**Struttura** (medie sui 20 seed; alias = quota di query con almeno un alias, gemelli =
quota di vettori di peso 2):

| grafo | campione | N = 100 | 400 | 1600 |
|---|---|---|---|---|
| CoDEx-M | uniform | alias 0.1 %, gemelli 0, M 205 | 0.6 %, 0, 681 | 0.9 %, 0, 2237 |
| CoDEx-M | dense | 0.0 %, 0.4 %, 104 | 0.6 %, 3.4 %, 385 | 0.2 %, 0.8 %, 1465 |
| YAGO3-10 | uniform | 0, 0, 207 | 0.1 %, 0, 776 | 0.3 %, 0, 2870 |
| YAGO3-10 | dense | 1.0 %, 1.0 %, 94 | 0.8 %, 0.5 %, 319 | 1.0 %, 2.6 %, 1250 |

Tetto ≥ 0.992 ovunque.

**Riduzione della griglia (regola della revisione, punto 2), applicata:** stima
iniziale 350 min (`--predict` 47 min YAGO3-10 + 56 min CoDEx-M; misure stimate con
`--timing` 124 + 124 min, dominate da N = 3200: 108 min per grafo). Passo (1), via
N = 3200 di YAGO3-10: 219 min, ancora sopra; passo (2), via N = 3200 di CoDEx-M:
**≈ 80 min** (previsioni 49 min, misure 31 min). Griglia finale: **N ∈ (100, 400, 1600)** su entrambi i grafi. Le righe N = 3200 sono tolte dal file delle previsioni (che
conserva l'esito dell'ammissione sulla griglia completa, `admission_full_grid`).

**Catene presenti** dopo la riduzione (seed con ≥ 10 catene ≥ 10): **solo h = 2**, nelle
celle CoDEx-M uniform N = 1600 (20 seed), CoDEx-M dense N = 400 (10) e 1600 (11),
YAGO3-10 dense N = 1600 (14). **Nessuna cella a 3 hop** (né uniform né dense, su nessuno
dei due grafi): le catene a 3 hop **non sono testate** in questa preregistrazione.
Con i nuovi sottografi (SeedSequence con l'indice del grafo) anche le celle uniform h = 2
a N = 400 di CoDEx-M non raggiungono 10 seed.

Una riga per cella (`results/generalization_prereg_predictions.json`, con le previsioni
per seed, le soglie X e l'esito dell'ammissione). Previsto = media sui seed della cella;
per BSDC-S tra parentesi la previsione del modello della 21. "Firma senza gemelli/alias"
è il comparatore di H9. MDE per hop: *escl.* = esclusa da H6/H9 (> 5 punti).

| grafo | campione | N | target | famiglia | h | seed | D | previsto | FKS ingenuo | firma senza gemelli/alias | MDE cella | MDE per hop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yago3-10 | uniform | 100 | 0.5 | MAP-B | 1 | 20 | 1216 | 0.510 | 0.510 | 0.510 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.5 | MAP-I | 1 | 20 | 768 | 0.508 | 0.508 | 0.508 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.5 | FHRR | 1 | 20 | 384 | 0.508 | 0.508 | 0.508 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.5 | HRR | 1 | 20 | 1024 | 0.520 | 0.663 | 0.663 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.5 | BSDC-OR | 1 | 20 | 1024 | 0.509 | 0.509 | 0.509 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.5 | BSDC-S | 1 | 20 | 1152 | 0.568 (21: 0.545) | 0.568 | 0.568 | 3.3 | 3.3 |
| yago3-10 | uniform | 100 | 0.85 | MAP-B | 1 | 20 | 2368 | 0.851 | 0.851 | 0.851 | 2.4 | 2.4 |
| yago3-10 | uniform | 100 | 0.85 | MAP-I | 1 | 20 | 1536 | 0.861 | 0.861 | 0.861 | 2.3 | 2.3 |
| yago3-10 | uniform | 100 | 0.85 | FHRR | 1 | 20 | 768 | 0.861 | 0.861 | 0.861 | 2.3 | 2.3 |
| yago3-10 | uniform | 100 | 0.85 | HRR | 1 | 20 | 2048 | 0.857 | 0.950 | 0.950 | 2.3 | 2.3 |
| yago3-10 | uniform | 100 | 0.85 | BSDC-OR | 1 | 20 | 1408 | 0.884 | 0.884 | 0.884 | 2.1 | 2.1 |
| yago3-10 | uniform | 100 | 0.85 | BSDC-S | 1 | 20 | 1920 | 0.880 (21: 0.864) | 0.880 | 0.880 | 2.1 | 2.1 |
| yago3-10 | uniform | 400 | 0.5 | MAP-B | 1 | 20 | 6336 | 0.505 | 0.504 | 0.505 | 2.3 | 2.3 |
| yago3-10 | uniform | 400 | 0.5 | MAP-I | 1 | 20 | 4032 | 0.505 | 0.505 | 0.505 | 2.3 | 2.3 |
| yago3-10 | uniform | 400 | 0.5 | FHRR | 1 | 20 | 2048 | 0.515 | 0.514 | 0.515 | 2.3 | 2.3 |
| yago3-10 | uniform | 400 | 0.5 | HRR | 1 | 20 | 4992 | 0.505 | 0.636 | 0.637 | 2.3 | 2.3 |
| yago3-10 | uniform | 400 | 0.5 | BSDC-OR | 1 | 20 | 5632 | 0.652 | 0.652 | 0.652 | 2.2 | 2.2 |
| yago3-10 | uniform | 400 | 0.5 | BSDC-S | 1 | 20 | 5632 | 0.520 (21: 0.502) | 0.519 | 0.520 | 2.3 | 2.3 |
| yago3-10 | uniform | 400 | 0.85 | MAP-B | 1 | 20 | 11456 | 0.850 | 0.850 | 0.850 | 1.7 | 1.7 |
| yago3-10 | uniform | 400 | 0.85 | MAP-I | 1 | 20 | 7296 | 0.851 | 0.851 | 0.851 | 1.7 | 1.7 |
| yago3-10 | uniform | 400 | 0.85 | FHRR | 1 | 20 | 3648 | 0.851 | 0.851 | 0.851 | 1.7 | 1.7 |
| yago3-10 | uniform | 400 | 0.85 | HRR | 1 | 20 | 9280 | 0.854 | 0.939 | 0.939 | 1.6 | 1.6 |
| yago3-10 | uniform | 400 | 0.85 | BSDC-OR | 1 | 20 | 6656 | 0.875 | 0.875 | 0.876 | 1.5 | 1.5 |
| yago3-10 | uniform | 400 | 0.85 | BSDC-S | 1 | 20 | 9216 | 0.873 (21: 0.858) | 0.873 | 0.873 | 1.6 | 1.6 |
| yago3-10 | uniform | 1600 | 0.5 | MAP-B | 1 | 20 | 31296 | 0.501 | 0.500 | 0.501 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.5 | MAP-I | 1 | 20 | 19904 | 0.500 | 0.500 | 0.501 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.5 | FHRR | 1 | 20 | 9984 | 0.503 | 0.502 | 0.503 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.5 | HRR | 1 | 20 | 24448 | 0.501 | 0.641 | 0.642 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.5 | BSDC-OR | 1 | 20 | 26624 | 0.631 | 0.631 | 0.631 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.5 | BSDC-S | 1 | 20 | 28672 | 0.543 (21: 0.527) | 0.542 | 0.543 | 2.3 | 2.3 |
| yago3-10 | uniform | 1600 | 0.85 | MAP-B | 1 | 20 | 53696 | 0.849 | 0.850 | 0.850 | 1.7 | 1.7 |
| yago3-10 | uniform | 1600 | 0.85 | MAP-I | 1 | 20 | 34176 | 0.849 | 0.850 | 0.851 | 1.7 | 1.7 |
| yago3-10 | uniform | 1600 | 0.85 | FHRR | 1 | 20 | 17088 | 0.849 | 0.850 | 0.851 | 1.7 | 1.7 |
| yago3-10 | uniform | 1600 | 0.85 | HRR | 1 | 20 | 42752 | 0.851 | 0.940 | 0.940 | 1.6 | 1.6 |
| yago3-10 | uniform | 1600 | 0.85 | BSDC-OR | 1 | 20 | 30720 | 0.865 | 0.866 | 0.866 | 1.6 | 1.6 |
| yago3-10 | uniform | 1600 | 0.85 | BSDC-S | 1 | 20 | 43008 | 0.867 (21: 0.853) | 0.867 | 0.868 | 1.6 | 1.6 |
| yago3-10 | dense | 100 | 0.5 | MAP-B | 1 | 20 | 960 | 0.525 | 0.498 | 0.528 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.5 | MAP-I | 1 | 20 | 640 | 0.547 | 0.521 | 0.550 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.5 | FHRR | 1 | 20 | 320 | 0.547 | 0.521 | 0.550 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.5 | HRR | 1 | 20 | 896 | 0.502 | 0.684 | 0.705 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.5 | BSDC-OR | 1 | 20 | 896 | 0.587 | 0.562 | 0.580 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.5 | BSDC-S | 1 | 20 | 896 | 0.557 (21: 0.535) | 0.536 | 0.564 | 3.8 | 3.8 |
| yago3-10 | dense | 100 | 0.85 | MAP-B | 1 | 20 | 2048 | 0.851 | 0.849 | 0.860 | 2.7 | 2.7 |
| yago3-10 | dense | 100 | 0.85 | MAP-I | 1 | 20 | 1344 | 0.864 | 0.863 | 0.872 | 2.6 | 2.6 |
| yago3-10 | dense | 100 | 0.85 | FHRR | 1 | 20 | 704 | 0.880 | 0.879 | 0.888 | 2.5 | 2.5 |
| yago3-10 | dense | 100 | 0.85 | HRR | 1 | 20 | 2304 | 0.841 | 0.983 | 0.984 | 2.7 | 2.7 |
| yago3-10 | dense | 100 | 0.85 | BSDC-OR | 1 | 20 | 1280 | 0.906 | 0.902 | 0.907 | 2.2 | 2.2 |
| yago3-10 | dense | 100 | 0.85 | BSDC-S | 1 | 20 | 1664 | 0.878 (21: 0.862) | 0.880 | 0.888 | 2.5 | 2.5 |
| yago3-10 | dense | 400 | 0.5 | MAP-B | 1 | 20 | 5120 | 0.500 | 0.489 | 0.504 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.5 | MAP-I | 1 | 20 | 3264 | 0.501 | 0.491 | 0.505 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.5 | FHRR | 1 | 20 | 1664 | 0.511 | 0.501 | 0.515 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.5 | HRR | 1 | 20 | 5056 | 0.506 | 0.734 | 0.744 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.5 | BSDC-OR | 1 | 20 | 4608 | 0.579 | 0.567 | 0.575 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.5 | BSDC-S | 1 | 20 | 4608 | 0.514 (21: 0.495) | 0.507 | 0.521 | 2.3 | 2.3 |
| yago3-10 | dense | 400 | 0.85 | MAP-B | 1 | 20 | 9984 | 0.846 | 0.847 | 0.853 | 1.7 | 1.7 |
| yago3-10 | dense | 400 | 0.85 | MAP-I | 1 | 20 | 6400 | 0.850 | 0.850 | 0.856 | 1.7 | 1.7 |
| yago3-10 | dense | 400 | 0.85 | FHRR | 1 | 20 | 3200 | 0.850 | 0.850 | 0.856 | 1.7 | 1.7 |
| yago3-10 | dense | 400 | 0.85 | HRR | 1 | 20 | 11648 | 0.849 | 0.990 | 0.990 | 1.6 | 1.6 |
| yago3-10 | dense | 400 | 0.85 | BSDC-OR | 1 | 20 | 6144 | 0.904 | 0.903 | 0.905 | 1.4 | 1.4 |
| yago3-10 | dense | 400 | 0.85 | BSDC-S | 1 | 20 | 8192 | 0.877 (21: 0.862) | 0.880 | 0.884 | 1.5 | 1.5 |
| yago3-10 | dense | 1600 | 0.5 | MAP-B | 1 | 20 | 25920 | 0.491 | 0.470 | 0.488 | 2.3 | 2.3 |
| yago3-10 | dense | 1600 | 0.5 | MAP-B | 2 | 14 | 25920 | 0.189 | 0.229 | 0.229 | 3.8 | 4.3 |
| yago3-10 | dense | 1600 | 0.5 | MAP-I | 1 | 20 | 16512 | 0.491 | 0.471 | 0.489 | 2.3 | 2.3 |
| yago3-10 | dense | 1600 | 0.5 | MAP-I | 2 | 14 | 16512 | 0.189 | 0.229 | 0.229 | 3.8 | 4.3 |
| yago3-10 | dense | 1600 | 0.5 | FHRR | 1 | 20 | 8256 | 0.491 | 0.471 | 0.489 | 2.3 | 2.3 |
| yago3-10 | dense | 1600 | 0.5 | FHRR | 2 | 14 | 8256 | 0.189 | 0.229 | 0.229 | 3.8 | 4.3 |
| yago3-10 | dense | 1600 | 0.5 | HRR | 1 | 20 | 24064 | 0.488 | 0.710 | 0.723 | 2.2 | 2.2 |
| yago3-10 | dense | 1600 | 0.5 | HRR | 2 | 14 | 24064 | 0.217 | 0.513 | 0.513 | 4.0 | 4.3 |
| yago3-10 | dense | 1600 | 0.5 | BSDC-OR | 1 | 20 | 22528 | 0.555 | 0.528 | 0.537 | 2.3 | 2.3 |
| yago3-10 | dense | 1600 | 0.5 | BSDC-OR | 2 | 14 | 22528 | 0.294 | 0.305 | 0.305 | 3.9 | 3.6 |
| yago3-10 | dense | 1600 | 0.5 | BSDC-S | 1 | 20 | 24576 | 0.540 (21: 0.524) | 0.527 | 0.544 | 2.3 | 2.3 |
| yago3-10 | dense | 1600 | 0.5 | BSDC-S | 2 | 14 | 24576 | 0.230 (21: 0.216) | 0.289 | 0.289 | 4.0 | 4.1 |
| yago3-10 | dense | 1600 | 0.85 | MAP-B | 1 | 20 | 48256 | 0.848 | 0.847 | 0.855 | 1.7 | 1.7 |
| yago3-10 | dense | 1600 | 0.85 | MAP-B | 2 | 14 | 48256 | 0.570 | 0.725 | 0.725 | 4.1 | 2.7 |
| yago3-10 | dense | 1600 | 0.85 | MAP-I | 1 | 20 | 30720 | 0.848 | 0.847 | 0.855 | 1.7 | 1.7 |
| yago3-10 | dense | 1600 | 0.85 | MAP-I | 2 | 14 | 30720 | 0.570 | 0.725 | 0.725 | 4.1 | 2.7 |
| yago3-10 | dense | 1600 | 0.85 | FHRR | 1 | 20 | 15360 | 0.848 | 0.847 | 0.855 | 1.7 | 1.7 |
| yago3-10 | dense | 1600 | 0.85 | FHRR | 2 | 14 | 15360 | 0.570 | 0.725 | 0.725 | 4.1 | 2.7 |
| yago3-10 | dense | 1600 | 0.85 | HRR | 1 | 20 | 51200 | 0.842 | 0.986 | 0.987 | 1.6 | 1.6 |
| yago3-10 | dense | 1600 | 0.85 | HRR | 2 | 14 | 51200 | 0.644 | 0.974 | 0.974 | 3.5 | 2.2 |
| yago3-10 | dense | 1600 | 0.85 | BSDC-OR | 1 | 20 | 28672 | 0.895 | 0.891 | 0.893 | 1.4 | 1.4 |
| yago3-10 | dense | 1600 | 0.85 | BSDC-OR | 2 | 14 | 28672 | 0.663 | 0.805 | 0.805 | 3.5 | 2.2 |
| yago3-10 | dense | 1600 | 0.85 | BSDC-S | 1 | 20 | 38912 | 0.865 (21: 0.852) | 0.870 | 0.876 | 1.6 | 1.6 |
| yago3-10 | dense | 1600 | 0.85 | BSDC-S | 2 | 14 | 38912 | 0.594 (21: 0.576) | 0.765 | 0.765 | 4.0 | 2.6 |
| codex-m | uniform | 100 | 0.5 | MAP-B | 1 | 20 | 1216 | 0.511 | 0.511 | 0.511 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.5 | MAP-I | 1 | 20 | 768 | 0.509 | 0.509 | 0.509 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.5 | FHRR | 1 | 20 | 384 | 0.509 | 0.509 | 0.509 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.5 | HRR | 1 | 20 | 960 | 0.527 | 0.628 | 0.629 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.5 | BSDC-OR | 1 | 20 | 1024 | 0.511 | 0.511 | 0.511 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.5 | BSDC-S | 1 | 20 | 1152 | 0.569 (21: 0.547) | 0.569 | 0.569 | 3.3 | 3.3 |
| codex-m | uniform | 100 | 0.85 | MAP-B | 1 | 20 | 2368 | 0.852 | 0.852 | 0.852 | 2.3 | 2.3 |
| codex-m | uniform | 100 | 0.85 | MAP-I | 1 | 20 | 1536 | 0.862 | 0.862 | 0.862 | 2.3 | 2.3 |
| codex-m | uniform | 100 | 0.85 | FHRR | 1 | 20 | 768 | 0.862 | 0.862 | 0.862 | 2.3 | 2.3 |
| codex-m | uniform | 100 | 0.85 | HRR | 1 | 20 | 1856 | 0.849 | 0.926 | 0.926 | 2.3 | 2.3 |
| codex-m | uniform | 100 | 0.85 | BSDC-OR | 1 | 20 | 1408 | 0.885 | 0.885 | 0.885 | 2.1 | 2.1 |
| codex-m | uniform | 100 | 0.85 | BSDC-S | 1 | 20 | 1920 | 0.881 (21: 0.865) | 0.881 | 0.881 | 2.1 | 2.1 |
| codex-m | uniform | 400 | 0.5 | MAP-B | 1 | 20 | 6144 | 0.501 | 0.501 | 0.502 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.5 | MAP-I | 1 | 20 | 3904 | 0.501 | 0.500 | 0.502 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.5 | FHRR | 1 | 20 | 1984 | 0.511 | 0.510 | 0.512 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.5 | HRR | 1 | 20 | 4544 | 0.503 | 0.592 | 0.594 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.5 | BSDC-OR | 1 | 20 | 5120 | 0.518 | 0.518 | 0.519 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.5 | BSDC-S | 1 | 20 | 5632 | 0.539 (21: 0.520) | 0.538 | 0.540 | 2.3 | 2.3 |
| codex-m | uniform | 400 | 0.85 | MAP-B | 1 | 20 | 11264 | 0.849 | 0.850 | 0.851 | 1.7 | 1.7 |
| codex-m | uniform | 400 | 0.85 | MAP-I | 1 | 20 | 7168 | 0.850 | 0.851 | 0.852 | 1.7 | 1.7 |
| codex-m | uniform | 400 | 0.85 | FHRR | 1 | 20 | 3584 | 0.850 | 0.851 | 0.852 | 1.7 | 1.7 |
| codex-m | uniform | 400 | 0.85 | HRR | 1 | 20 | 8576 | 0.853 | 0.921 | 0.921 | 1.6 | 1.6 |
| codex-m | uniform | 400 | 0.85 | BSDC-OR | 1 | 20 | 6656 | 0.888 | 0.890 | 0.890 | 1.5 | 1.5 |
| codex-m | uniform | 400 | 0.85 | BSDC-S | 1 | 20 | 8704 | 0.850 (21: 0.835) | 0.852 | 0.853 | 1.7 | 1.7 |
| codex-m | uniform | 1600 | 0.5 | MAP-B | 1 | 20 | 29952 | 0.499 | 0.496 | 0.500 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | MAP-B | 2 | 20 | 29952 | 0.216 | 0.246 | 0.246 | 3.2 | 3.5 |
| codex-m | uniform | 1600 | 0.5 | MAP-I | 1 | 20 | 19072 | 0.499 | 0.497 | 0.501 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | MAP-I | 2 | 20 | 19072 | 0.216 | 0.247 | 0.247 | 3.2 | 3.5 |
| codex-m | uniform | 1600 | 0.5 | FHRR | 1 | 20 | 9536 | 0.499 | 0.497 | 0.501 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | FHRR | 2 | 20 | 9536 | 0.216 | 0.247 | 0.247 | 3.2 | 3.5 |
| codex-m | uniform | 1600 | 0.5 | HRR | 1 | 20 | 21824 | 0.499 | 0.587 | 0.591 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | HRR | 2 | 20 | 21824 | 0.257 | 0.345 | 0.345 | 3.4 | 3.4 |
| codex-m | uniform | 1600 | 0.5 | BSDC-OR | 1 | 20 | 24576 | 0.530 | 0.530 | 0.532 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | BSDC-OR | 2 | 20 | 24576 | 0.234 | 0.281 | 0.281 | 3.3 | 3.4 |
| codex-m | uniform | 1600 | 0.5 | BSDC-S | 1 | 20 | 26624 | 0.513 (21: 0.497) | 0.511 | 0.514 | 2.3 | 2.3 |
| codex-m | uniform | 1600 | 0.5 | BSDC-S | 2 | 20 | 26624 | 0.227 (21: 0.214) | 0.261 | 0.261 | 3.3 | 3.4 |
| codex-m | uniform | 1600 | 0.85 | MAP-B | 1 | 20 | 51968 | 0.846 | 0.848 | 0.850 | 1.7 | 1.7 |
| codex-m | uniform | 1600 | 0.85 | MAP-B | 2 | 20 | 51968 | 0.587 | 0.719 | 0.719 | 3.5 | 2.3 |
| codex-m | uniform | 1600 | 0.85 | MAP-I | 1 | 20 | 33088 | 0.846 | 0.848 | 0.850 | 1.7 | 1.7 |
| codex-m | uniform | 1600 | 0.85 | MAP-I | 2 | 20 | 33088 | 0.588 | 0.719 | 0.719 | 3.5 | 2.3 |
| codex-m | uniform | 1600 | 0.85 | FHRR | 1 | 20 | 16576 | 0.847 | 0.849 | 0.851 | 1.7 | 1.7 |
| codex-m | uniform | 1600 | 0.85 | FHRR | 2 | 20 | 16576 | 0.589 | 0.721 | 0.721 | 3.5 | 2.3 |
| codex-m | uniform | 1600 | 0.85 | HRR | 1 | 20 | 38656 | 0.847 | 0.915 | 0.916 | 1.6 | 1.6 |
| codex-m | uniform | 1600 | 0.85 | HRR | 2 | 20 | 38656 | 0.639 | 0.837 | 0.837 | 3.3 | 2.1 |
| codex-m | uniform | 1600 | 0.85 | BSDC-OR | 1 | 20 | 30720 | 0.890 | 0.893 | 0.894 | 1.4 | 1.4 |
| codex-m | uniform | 1600 | 0.85 | BSDC-OR | 2 | 20 | 30720 | 0.640 | 0.798 | 0.798 | 3.3 | 2.1 |
| codex-m | uniform | 1600 | 0.85 | BSDC-S | 1 | 20 | 40960 | 0.854 (21: 0.840) | 0.856 | 0.857 | 1.6 | 1.6 |
| codex-m | uniform | 1600 | 0.85 | BSDC-S | 2 | 20 | 40960 | 0.597 (21: 0.579) | 0.732 | 0.732 | 3.5 | 2.3 |
| codex-m | dense | 100 | 0.5 | MAP-B | 1 | 20 | 960 | 0.506 | 0.483 | 0.506 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.5 | MAP-I | 1 | 20 | 640 | 0.529 | 0.507 | 0.529 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.5 | FHRR | 1 | 20 | 320 | 0.529 | 0.507 | 0.529 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.5 | HRR | 1 | 20 | 960 | 0.487 | 0.706 | 0.722 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.5 | BSDC-OR | 1 | 20 | 896 | 0.550 | 0.532 | 0.545 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.5 | BSDC-S | 1 | 20 | 896 | 0.538 (21: 0.516) | 0.519 | 0.540 | 3.5 | 3.5 |
| codex-m | dense | 100 | 0.85 | MAP-B | 1 | 20 | 2048 | 0.849 | 0.841 | 0.851 | 2.5 | 2.5 |
| codex-m | dense | 100 | 0.85 | MAP-I | 1 | 20 | 1344 | 0.862 | 0.856 | 0.864 | 2.4 | 2.4 |
| codex-m | dense | 100 | 0.85 | FHRR | 1 | 20 | 704 | 0.878 | 0.873 | 0.880 | 2.3 | 2.3 |
| codex-m | dense | 100 | 0.85 | HRR | 1 | 20 | 2624 | 0.842 | 0.991 | 0.992 | 2.5 | 2.5 |
| codex-m | dense | 100 | 0.85 | BSDC-OR | 1 | 20 | 1280 | 0.898 | 0.893 | 0.896 | 2.2 | 2.2 |
| codex-m | dense | 100 | 0.85 | BSDC-S | 1 | 20 | 1664 | 0.876 (21: 0.859) | 0.872 | 0.879 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | MAP-B | 1 | 20 | 5504 | 0.511 | 0.504 | 0.513 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | MAP-B | 2 | 10 | 5504 | 0.237 | 0.263 | 0.263 | 8.5 | 8.8 *escl.* |
| codex-m | dense | 400 | 0.5 | MAP-I | 1 | 20 | 3520 | 0.514 | 0.508 | 0.516 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | MAP-I | 2 | 10 | 3520 | 0.241 | 0.266 | 0.266 | 8.6 | 8.7 *escl.* |
| codex-m | dense | 400 | 0.5 | FHRR | 1 | 20 | 1792 | 0.523 | 0.518 | 0.526 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | FHRR | 2 | 10 | 1792 | 0.250 | 0.277 | 0.277 | 8.7 | 8.7 *escl.* |
| codex-m | dense | 400 | 0.5 | HRR | 1 | 20 | 6016 | 0.513 | 0.807 | 0.812 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | HRR | 2 | 10 | 6016 | 0.402 | 0.660 | 0.660 | 9.7 | 7.7 *escl.* |
| codex-m | dense | 400 | 0.5 | BSDC-OR | 1 | 20 | 4608 | 0.543 | 0.509 | 0.514 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | BSDC-OR | 2 | 10 | 4608 | 0.346 | 0.282 | 0.282 | 8.5 | 7.3 *escl.* |
| codex-m | dense | 400 | 0.5 | BSDC-S | 1 | 20 | 5120 | 0.548 (21: 0.530) | 0.551 | 0.559 | 2.3 | 2.3 |
| codex-m | dense | 400 | 0.5 | BSDC-S | 2 | 10 | 5120 | 0.267 (21: 0.249) | 0.315 | 0.315 | 8.8 | 8.6 *escl.* |
| codex-m | dense | 400 | 0.85 | MAP-B | 1 | 20 | 10496 | 0.848 | 0.855 | 0.858 | 1.7 | 1.7 |
| codex-m | dense | 400 | 0.85 | MAP-B | 2 | 10 | 10496 | 0.660 | 0.738 | 0.738 | 9.2 | 5.7 *escl.* |
| codex-m | dense | 400 | 0.85 | MAP-I | 1 | 20 | 6720 | 0.851 | 0.858 | 0.861 | 1.6 | 1.6 |
| codex-m | dense | 400 | 0.85 | MAP-I | 2 | 10 | 6720 | 0.665 | 0.744 | 0.744 | 9.2 | 5.6 *escl.* |
| codex-m | dense | 400 | 0.85 | FHRR | 1 | 20 | 3392 | 0.855 | 0.862 | 0.865 | 1.6 | 1.6 |
| codex-m | dense | 400 | 0.85 | FHRR | 2 | 10 | 3392 | 0.671 | 0.750 | 0.750 | 9.1 | 5.6 *escl.* |
| codex-m | dense | 400 | 0.85 | HRR | 1 | 20 | 14144 | 0.853 | 0.997 | 0.997 | 1.6 | 1.6 |
| codex-m | dense | 400 | 0.85 | HRR | 2 | 10 | 14144 | 0.852 | 0.994 | 0.994 | 4.7 | 2.5 |
| codex-m | dense | 400 | 0.85 | BSDC-OR | 1 | 20 | 6144 | 0.893 | 0.885 | 0.886 | 1.4 | 1.4 |
| codex-m | dense | 400 | 0.85 | BSDC-OR | 2 | 10 | 6144 | 0.767 | 0.794 | 0.794 | 7.7 | 4.4 |
| codex-m | dense | 400 | 0.85 | BSDC-S | 1 | 20 | 8192 | 0.853 (21: 0.837) | 0.865 | 0.868 | 1.6 | 1.6 |
| codex-m | dense | 400 | 0.85 | BSDC-S | 2 | 10 | 8192 | 0.659 (21: 0.635) | 0.757 | 0.757 | 9.2 | 5.7 *escl.* |
| codex-m | dense | 1600 | 0.5 | MAP-B | 1 | 20 | 27840 | 0.498 | 0.495 | 0.504 | 2.3 | 2.3 |
| codex-m | dense | 1600 | 0.5 | MAP-B | 2 | 11 | 27840 | 0.273 | 0.251 | 0.251 | 6.9 | 6.6 *escl.* |
| codex-m | dense | 1600 | 0.5 | MAP-I | 1 | 20 | 17728 | 0.498 | 0.495 | 0.504 | 2.3 | 2.3 |
| codex-m | dense | 1600 | 0.5 | MAP-I | 2 | 11 | 17728 | 0.273 | 0.251 | 0.251 | 6.9 | 6.6 *escl.* |
| codex-m | dense | 1600 | 0.5 | FHRR | 1 | 20 | 8896 | 0.501 | 0.497 | 0.506 | 2.3 | 2.3 |
| codex-m | dense | 1600 | 0.5 | FHRR | 2 | 11 | 8896 | 0.275 | 0.253 | 0.253 | 6.9 | 6.6 *escl.* |
| codex-m | dense | 1600 | 0.5 | HRR | 1 | 20 | 27328 | 0.491 | 0.772 | 0.777 | 2.2 | 2.2 |
| codex-m | dense | 1600 | 0.5 | HRR | 2 | 11 | 27328 | 0.450 | 0.603 | 0.603 | 7.4 | 5.5 *escl.* |
| codex-m | dense | 1600 | 0.5 | BSDC-OR | 1 | 20 | 22528 | 0.496 | 0.476 | 0.481 | 2.3 | 2.3 |
| codex-m | dense | 1600 | 0.5 | BSDC-OR | 2 | 11 | 22528 | 0.266 | 0.244 | 0.244 | 6.5 | 6.3 *escl.* |
| codex-m | dense | 1600 | 0.5 | BSDC-S | 1 | 20 | 24576 | 0.500 (21: 0.484) | 0.502 | 0.511 | 2.3 | 2.3 |
| codex-m | dense | 1600 | 0.5 | BSDC-S | 2 | 11 | 24576 | 0.272 (21: 0.258) | 0.260 | 0.260 | 6.9 | 6.6 *escl.* |
| codex-m | dense | 1600 | 0.85 | MAP-B | 1 | 20 | 49920 | 0.850 | 0.853 | 0.857 | 1.7 | 1.7 |
| codex-m | dense | 1600 | 0.85 | MAP-B | 2 | 11 | 49920 | 0.710 | 0.733 | 0.733 | 6.7 | 4.0 |
| codex-m | dense | 1600 | 0.85 | MAP-I | 1 | 20 | 31808 | 0.851 | 0.854 | 0.857 | 1.7 | 1.7 |
| codex-m | dense | 1600 | 0.85 | MAP-I | 2 | 11 | 31808 | 0.711 | 0.734 | 0.734 | 6.7 | 4.0 |
| codex-m | dense | 1600 | 0.85 | FHRR | 1 | 20 | 15936 | 0.852 | 0.855 | 0.858 | 1.7 | 1.7 |
| codex-m | dense | 1600 | 0.85 | FHRR | 2 | 11 | 15936 | 0.712 | 0.735 | 0.735 | 6.7 | 4.0 |
| codex-m | dense | 1600 | 0.85 | HRR | 1 | 20 | 61056 | 0.847 | 0.996 | 0.996 | 1.6 | 1.6 |
| codex-m | dense | 1600 | 0.85 | HRR | 2 | 11 | 61056 | 0.892 | 0.992 | 0.992 | 2.8 | 1.5 |
| codex-m | dense | 1600 | 0.85 | BSDC-OR | 1 | 20 | 28672 | 0.881 | 0.873 | 0.874 | 1.5 | 1.5 |
| codex-m | dense | 1600 | 0.85 | BSDC-OR | 2 | 11 | 28672 | 0.764 | 0.772 | 0.772 | 6.6 | 3.8 |
| codex-m | dense | 1600 | 0.85 | BSDC-S | 1 | 20 | 40960 | 0.879 (21: 0.866) | 0.885 | 0.888 | 1.5 | 1.5 |
| codex-m | dense | 1600 | 0.85 | BSDC-S | 2 | 11 | 40960 | 0.753 (21: 0.733) | 0.789 | 0.789 | 6.3 | 3.6 |

Per gruppo, sulle celle che restano (a 1 hop tutte; a 2 hop quelle con MDE per hop ≤ 5):
MDE di gruppo = 2.95·√(Σ SE²)/n sulla scala per hop; divario = previsto contro FKS
ingenuo (h = 1) o contro il comparatore con firma (h = 2), scala per hop; X dalla regola
(punto 7), calcolato dall'harness insieme alle previsioni.

| grafo | famiglia | campione | h | celle (dopo esclusione) | MDE di gruppo (punti, scala per hop) | divario previsto | X (ipotesi) |
|---|---|---|---|---|---|---|---|
| codex-m | BSDC-OR | dense | 1 | 6 | 0.9 | 1.6 | — |
| codex-m | BSDC-S | dense | 1 | 6 | 1.0 | 0.8 | — |
| codex-m | FHRR | dense | 1 | 6 | 1.0 | 0.8 | — |
| codex-m | HRR | dense | 1 | 6 | 1.0 | 20.6 | 10.0 (H4) |
| codex-m | MAP-B | dense | 1 | 6 | 1.0 | 0.8 | — |
| codex-m | MAP-I | dense | 1 | 6 | 1.0 | 0.8 | — |
| codex-m | BSDC-OR | dense | 2 | 2 | 2.9 | 1.0 | 1.0 (H9) |
| codex-m | BSDC-S | dense | 2 | 1 | 3.6 | 2.0 | 1.0 (H9) |
| codex-m | FHRR | dense | 2 | 1 | 4.0 | 1.4 | 1.0 (H9) |
| codex-m | HRR | dense | 2 | 2 | 1.5 | 6.3 | 3.0 (H9) |
| codex-m | MAP-B | dense | 2 | 1 | 4.0 | 1.3 | 1.0 (H9) |
| codex-m | MAP-I | dense | 2 | 1 | 4.0 | 1.3 | 1.0 (H9) |
| codex-m | BSDC-OR | uniform | 1 | 6 | 0.9 | 0.1 | — |
| codex-m | BSDC-S | uniform | 1 | 6 | 0.9 | 0.1 | — |
| codex-m | FHRR | uniform | 1 | 6 | 1.0 | 0.1 | — |
| codex-m | HRR | uniform | 1 | 6 | 0.9 | 8.2 | 4.0 (H4) |
| codex-m | MAP-B | uniform | 1 | 6 | 1.0 | 0.1 | — |
| codex-m | MAP-I | uniform | 1 | 6 | 1.0 | 0.1 | — |
| codex-m | BSDC-OR | uniform | 2 | 2 | 2.0 | 6.9 | 3.0 (H9) |
| codex-m | BSDC-S | uniform | 2 | 2 | 2.1 | 5.9 | 2.5 (H9) |
| codex-m | FHRR | uniform | 2 | 2 | 2.1 | 5.6 | 2.5 (H9) |
| codex-m | HRR | uniform | 2 | 2 | 2.0 | 9.8 | 4.5 (H9) |
| codex-m | MAP-B | uniform | 2 | 2 | 2.1 | 5.6 | 2.5 (H9) |
| codex-m | MAP-I | uniform | 2 | 2 | 2.1 | 5.6 | 2.5 (H9) |
| yago3-10 | BSDC-OR | dense | 1 | 6 | 1.0 | 1.2 | — |
| yago3-10 | BSDC-S | dense | 1 | 6 | 1.0 | 0.8 | — |
| yago3-10 | FHRR | dense | 1 | 6 | 1.0 | 1.0 | — |
| yago3-10 | HRR | dense | 1 | 6 | 1.0 | 17.7 | 8.5 (H4) |
| yago3-10 | MAP-B | dense | 1 | 6 | 1.0 | 1.0 | — |
| yago3-10 | MAP-I | dense | 1 | 6 | 1.0 | 1.0 | — |
| yago3-10 | BSDC-OR | dense | 2 | 2 | 2.1 | 4.7 | 2.0 (H9) |
| yago3-10 | BSDC-S | dense | 2 | 2 | 2.4 | 8.1 | 4.0 (H9) |
| yago3-10 | FHRR | dense | 2 | 2 | 2.6 | 7.0 | 3.5 (H9) |
| yago3-10 | HRR | dense | 2 | 2 | 2.4 | 21.8 | 10.5 (H9) |
| yago3-10 | MAP-B | dense | 2 | 2 | 2.6 | 7.0 | 3.5 (H9) |
| yago3-10 | MAP-I | dense | 2 | 2 | 2.6 | 7.0 | 3.5 (H9) |
| yago3-10 | BSDC-OR | uniform | 1 | 6 | 0.9 | 0.0 | — |
| yago3-10 | BSDC-S | uniform | 1 | 6 | 0.9 | 0.0 | — |
| yago3-10 | FHRR | uniform | 1 | 6 | 1.0 | 0.0 | — |
| yago3-10 | HRR | uniform | 1 | 6 | 0.9 | 11.4 | 5.5 (H4) |
| yago3-10 | MAP-B | uniform | 1 | 6 | 1.0 | 0.0 | — |
| yago3-10 | MAP-I | uniform | 1 | 6 | 1.0 | 0.0 | — |

## Ipotesi e criteri finali, fissati ora

Gruppo = (grafo, famiglia, campione, h). Celle a 1 hop: 3 carichi × 2 target = 6 per
gruppo, 20 seed, t con 19 gdl. Celle a 2 hop: i seed con ≥ 10 catene (10–20), t con
n_seed − 1 gdl, errori sulla scala per hop (misura^(1/2) − previsione^(1/2)), escluse le
celle con MDE per hop > 5 punti. Per ogni gruppo: MAE (punti), massimo di cella, MAE del
comparatore, errore con segno e (h = 1) IC al 95 % a cluster.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** replica di FKS, uniform, h = 1: 6 famiglie × 2 grafi = 12 gruppi | MAE ≤ 3 in ogni gruppo | MAE > 5 in un gruppo |
| **H2** replica di FKS sui dense, h = 1 (nessun grafo ammesso al test della contabilità): 12 gruppi | MAE ≤ 3 in tutti | MAE > 5 in almeno uno |
| **H3** BSDC-S corretto, h = 1: 4 gruppi (2 grafi × 2 campioni) | \|errore con segno\| ≤ 1.0 in tutti e 4 | \|errore con segno\| ≥ 1.5 in almeno uno |
| **H4** HRR, h = 1: 4 gruppi | MAE ≤ 3 in tutti e MAE_ingenuo − MAE ≥ X (4.0 CoDEx-M uniform, 10.0 CoDEx-M dense, 5.5 YAGO3-10 uniform, 8.5 YAGO3-10 dense) in tutti | MAE > 5 in uno, o MAE_ingenuo − MAE < 0 in uno |
| **H5** ablazione della contabilità | — | **non testabile** (nessun grafo ammesso): descrittiva |
| **H6 (rischiosa)** catene a 2 hop, previsione a prodotto (MAP-B: con il fattore di dipendenza), scala per hop: G = 18 gruppi (6 famiglie × CoDEx-M uniform, CoDEx-M dense, YAGO3-10 dense) | MAE per hop ≤ 3 in tutti i G gruppi | MAE per hop > 5 in almeno uno |
| **H7** nessun bias oltre il campionamento, per famiglia e grafo (12 celle h = 1) | al più 2 celle con \|t\| > t₀.₉₉₅,₁₉ = 2.861 | 4 o più celle |
| **H8** D fuori campione (seed 10–19), per famiglia e grafo (12 celle h = 1) | misurato medio ≥ target − 3 punti in almeno 11 celle | in 3 o più celle sotto |
| **H9** ablazione sulle catene, comparatore con firma (punto 1), scala per hop: G = 18 | guadagno MAE_comp − MAE ≥ X del gruppo in almeno ⌈5G/6⌉ = 15 gruppi | guadagno < 0 in almeno ⌈G/4⌉ = 5 gruppi |
| **H3 a 3 hop, H6/H9 a 3 hop** | — | **non testabili** (nessuna cella) |

Regola per gruppi che al momento del run risultassero vuoti (punto 8): G si ricalcola
(gruppi con ≥ 1 cella); con G = 0 l'ipotesi è non testabile. Il resto è **sostenuto in
parte**.

**Perché H6 è rischiosa.** Gli hop delle catene passano per nodi con più relazioni; il
fattore di dipendenza di MAP-B è calcolato per fatti uniformi; l'indipendenza fra hop è
verificata solo su sintetico con N ≤ 400; nello smoke a 2 e 3 hop il MAE per hop arriva
a 4.5 punti (MAP-B uniform, 1 cella) e 5.1 (MAP-B dense h = 3); nessuna catena è mai
stata prevista a priori su un KG reale in questo progetto.

**Potenza.** MDE di gruppo 0.9–1.0 punti a 1 hop e 1.5–4.0 a 2 hop (scala per hop,
limite inferiore; tabella sopra): le soglie 3/5 sono sopra l'MDE a 1 hop; a 2 hop nei
gruppi CoDEx-M dense con una sola cella l'MDE (3.6–4.0) supera la soglia di sostegno, e un
errore vero di 3 punti per hop non si distingue dal rumore lì. Le soglie X di H4 e H9
sono state calcolate insieme alle previsioni (punto 7) e non viste prima di scrivere la
regola.

**Descrittivi, non ipotesi:** errore con segno per gruppo; MAE pesato per l'inverso della
varianza (`mae_weighted_descr`); `pred_v21` contro misura per BSDC-S; il comparatore FKS
ingenuo nei gruppi dense h = 1; il fattore di dipendenza di MAP-B contro il prodotto.

**Previsioni (non criteri).** H1, H2, H4, H7, H8 sostenute. H3 sostenuta (la
postdizione dà +0.1…+0.6). H6 a rischio. H9 sostenuta in parte: nei gruppi CoDEx-M dense
il divario previsto è 1.0–2.0 punti per hop e X = 1.0 è vicino al rumore.

## Cosa ho visto prima

- I risultati sintetici dei passi 1–3 (sopra, `results/generalization_diagnosis.json`),
  inclusa la verifica alle code di HRR (`--verify-tails`, uniform N = 3200, M = 3158,
  D = 80 576, 4 sottografi): previsto 0.820, misurato 0.807 ± 0.016.
- La postdizione BSDC-S sulle celle già pubblicate della 21 (sopra).
- Lo smoke (`--smoke`, codice congelato, grafo sintetico, N = 40 e 80, 20 seed, impianto
  intero con misure; l'Engine MAP-B salato riproduce traccia e risposte di
  `abm.Memory.store`/`query` sugli stessi nomi salati): MAE per gruppo h = 1 0.5–2.0
  punti (FKS ingenuo nei dense 5.1–13.2); h = 2 per hop 0.5–4.5; h = 3 per hop 0.9–5.1.
- Le previsioni su CoDEx-M e YAGO3-10 (`--predict`), che leggono i dati ma non
  costruiscono memorie, e `--timing` (sintetico).
- **Scelte dopo aver visto qualcosa:** la correzione di BSDC-S (dopo il bias della 21 e la
  diagnosi sintetica: percorso dipendente dall'esito, dichiarato); la regola delle celle
  catena (dopo i soli conteggi delle catene di CoDEx-M, nessuna misura); la riduzione
  della griglia (regola scritta prima delle previsioni di YAGO3-10, applicata dopo il
  timing). Il numero di seed della diagnosi e della verifica (8 → 16) dopo il primo giro.

## Tempo stimato

Griglia finale (N ≤ 1600, 2 grafi): `--predict` ≈ 49 min (il run lo ripete), misure
stimate con `--timing` ≈ 31 min: **≈ 80 min** su questa macchina (12 core). La stima per
N = 3200 (non nel run) era 108 min di misure per grafo.

## Limiti dichiarati prima

- Due grafi nuovi (CoDEx-M, YAGO3-10), **nessuno dei due denso abbastanza da testare la
  contabilità** (regola di ammissione): qui la contabilità non è testata.
- Griglia ridotta a N ≤ 1600 (regola del tempo); **nessuna catena a 3 hop**; catene a 2
  hop in 4 sole configurazioni (grafo, campione, N).
- **Catene**: solo percorsi funzionali (un oggetto vero per hop); un percorso con più
  rami non è previsto; il fattore di dipendenza di MAP-B è calcolato per fatti uniformi;
  l'indipendenza fra hop è verificata solo su sintetico con N ≤ 400; molte celle catena
  hanno poche catene per seed (SE grandi).
- **HRR**: si ignora la fluttuazione comune |K_ν|² e le frequenze reali; unbinding per
  involuzione (non l'inversa esatta); float32.
- **BSDC-S**: la correzione toglie ≈ 80 % del bias sul sintetico; resta un residuo
  fino a +1.4 punti a B piccolo, non spiegato.
- SE a cluster con 20 seed; celle dello stesso (campione, N) correlate (stessi
  sottografi); gruppi non indipendenti; MDE di modello = limite inferiore.
- Bit per componente convenzionali; codebook non contato.
- D ≤ 2¹⁸: celle oltre il limite assenti (dichiarate in tabella).
