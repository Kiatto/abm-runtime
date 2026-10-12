# Preregistrazione 22 — la stima a priori del cleanup generalizza?

**Data: 2026-10-12.** Committata **prima** di misurare, insieme all'harness
[`examples/generalization_prereg.py`](../../examples/generalization_prereg.py), alla
diagnosi sintetica `results/generalization_diagnosis.json` e alle previsioni congelate
`results/generalization_prereg_predictions.json`. Prima del commit l'harness è stato
eseguito solo con `--diagnose`, `--verify`, `--smoke` (grafi sintetici generati nello
script) e con `--predict` e `--timing` (`--predict` legge CoDEx-M, campiona i sottografi
e le catene, sceglie D e calcola le previsioni; **non costruisce nessuna memoria** sui
dati reali; `--timing` misura il tempo su un grafo sintetico). Nessuna misura su
CoDEx-M, FB15k-237 o WN18RR è stata fatta per questa preregistrazione. Stesse regole di
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

## Passo 4 — dati nuovi: CoDEx-M

CoDEx-M (Safavi e Koutra, EMNLP 2020), train split, dal repo pubblico
`github.com/tsafavi/codex` al commit `3132e426c2a6b643b70bad679905a3a6270be440`
(`data/triples/codex-m/train.txt`, 185 584 triple), in `data/external/codex/`,
sha256 `d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556`, aggiunto a
`examples/replicate.py` (DATASETS). Letto **solo** per campionare la struttura
(sottografi, query, catene) in `--predict`; nessuna memoria costruita.

## Impianto (passo 5)

- **Famiglie**: MAP-B, MAP-I, FHRR, BSDC-OR come nella 21; BSDC-S con i pareggi esatti
  (passo 1); HRR (passo 2). Stesso encoding f = s ∘ ρ(r) ∘ o per tutte.
- **Sottografi** di CoDEx-M: N ∈ {100, 400, 1600, 3200}; *uniform* e *dense* (BFS non
  orientata, `v21.sample`); **20 seed**; flussi separati
  `SeedSequence([22 000 022, campione, N, seed]).spawn(3)` → campionamento e query,
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
  sd(e_k)/√20; t con **19 gdl**. MDE preregistrato per cella = (t₀.₉₇₅,₁₉ + t₀.₈₀,₁₉)·SE
  = 2.954·SE, con SE di modello (Poisson-binomiale per seed, query indipendenti): limite
  inferiore. MDE di gruppo = 2.954·√(Σ SE²)/n_celle.
- **Fuori campione**: D scelto sui seed 0–9, H9 usa solo i seed 10–19.
- **Comparatore FKS ingenuo** in ogni cella (come la 21; per HRR la gaussiana di MAP-I;
  per BSDC-S `bsdc_s_exact` con N fatti di peso 1 e un oggetto); per le catene il
  prodotto delle previsioni ingenue degli hop, senza fattore di dipendenza.

## Previsioni, calcolate ora (`--predict`)

Struttura (medie sui 20 seed). **CoDEx-M è poco "denso" nel senso della contabilità**:
anche i sottografi dense hanno alias ≤ 0.5 % e gemelli ≤ 2.0 % delle query/vettori
(dense: tetto 0.998, 0.999, 0.998, 1.000 per N = 100, 400, 1600, 3200; M = 105, 379,
1486, 2886); uniform: alias 0.2–1.1 %, nessun gemello, tetto 0.994–0.999, M = 204,
675, 2240, 3938. Quindi a 1 hop la previsione con contabilità e quella FKS ingenua
differiscono di 0.5–1.3 punti anche nei dense (tranne HRR, dove il divario viene dalle
molteplicità, non dalla contabilità): **su CoDEx-M l'ablazione della contabilità a 1 hop
non è testabile** (vedi H5), e i gruppi dense a 1 hop sono soprattutto una replica.

**Catene presenti** (seed con ≥ 10 catene, su 20): h = 2 uniform N = 100: 0, 400: 10,
1600: 20, 3200: 20; dense 100: 9, 400: 10, 1600: 11, 3200: 13. h = 3 uniform 400: 1,
1600: 11, 3200: 18; dense: 0, 1, 2, 3. Celle catena esistenti: uniform h = 2 (N = 400,
1600, 3200), uniform h = 3 (N = 1600, 3200), dense h = 2 (N = 400, 1600, 3200), per i
2 target. **Nessuna cella dense a 3 hop.**

Una riga per cella (salvate in `results/generalization_prereg_predictions.json`, con
le previsioni per seed). Previsto = media sui 20 seed; per BSDC-S tra parentesi la
previsione del modello della 21.

| campione | N | target | famiglia | h | D | bit | catene/query | previsto | FKS ingenuo | MDE cella (punti) |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 100 | 0.5 | MAP-B | 1 | 1216 | 1216 | 1997 | 0.512 | 0.511 | 3.3 |
| uniform | 100 | 0.5 | MAP-I | 1 | 768 | 5376 | 1997 | 0.510 | 0.510 | 3.3 |
| uniform | 100 | 0.5 | FHRR | 1 | 384 | 12288 | 1997 | 0.510 | 0.510 | 3.3 |
| uniform | 100 | 0.5 | HRR | 1 | 960 | 30720 | 1997 | 0.526 | 0.629 | 3.3 |
| uniform | 100 | 0.5 | BSDC-OR | 1 | 1024 | 1024 | 1997 | 0.513 | 0.513 | 3.3 |
| uniform | 100 | 0.5 | BSDC-S | 1 | 1152 | 8064 | 1997 | 0.570 (21: 0.547) | 0.570 | 3.3 |
| uniform | 100 | 0.85 | MAP-B | 1 | 2368 | 2368 | 1997 | 0.852 | 0.852 | 2.3 |
| uniform | 100 | 0.85 | MAP-I | 1 | 1536 | 10752 | 1997 | 0.862 | 0.862 | 2.3 |
| uniform | 100 | 0.85 | FHRR | 1 | 768 | 24576 | 1997 | 0.862 | 0.862 | 2.3 |
| uniform | 100 | 0.85 | HRR | 1 | 1856 | 59392 | 1997 | 0.848 | 0.926 | 2.3 |
| uniform | 100 | 0.85 | BSDC-OR | 1 | 1408 | 1408 | 1997 | 0.885 | 0.885 | 2.1 |
| uniform | 100 | 0.85 | BSDC-S | 1 | 1920 | 13440 | 1997 | 0.881 (21: 0.865) | 0.881 | 2.1 |
| uniform | 400 | 0.5 | MAP-B | 1 | 6144 | 6144 | 4000 | 0.502 | 0.502 | 2.3 |
| uniform | 400 | 0.5 | MAP-B | 2 | 6144 | 6144 | 207 | 0.225 | 0.252 | 9.1 |
| uniform | 400 | 0.5 | MAP-I | 1 | 3904 | 35136 | 4000 | 0.502 | 0.501 | 2.3 |
| uniform | 400 | 0.5 | MAP-I | 2 | 3904 | 35136 | 207 | 0.225 | 0.252 | 9.1 |
| uniform | 400 | 0.5 | FHRR | 1 | 1984 | 63488 | 4000 | 0.511 | 0.511 | 2.3 |
| uniform | 400 | 0.5 | FHRR | 2 | 1984 | 63488 | 207 | 0.234 | 0.262 | 9.3 |
| uniform | 400 | 0.5 | HRR | 1 | 4608 | 147456 | 4000 | 0.509 | 0.601 | 2.3 |
| uniform | 400 | 0.5 | HRR | 2 | 4608 | 147456 | 207 | 0.269 | 0.363 | 9.7 |
| uniform | 400 | 0.5 | BSDC-OR | 1 | 5120 | 5120 | 4000 | 0.520 | 0.520 | 2.3 |
| uniform | 400 | 0.5 | BSDC-OR | 2 | 5120 | 5120 | 207 | 0.234 | 0.273 | 9.2 |
| uniform | 400 | 0.5 | BSDC-S | 1 | 5632 | 50688 | 4000 | 0.540 (21: 0.521) | 0.539 | 2.3 |
| uniform | 400 | 0.5 | BSDC-S | 2 | 5632 | 50688 | 207 | 0.258 (21: 0.241) | 0.292 | 9.6 |
| uniform | 400 | 0.85 | MAP-B | 1 | 11264 | 11264 | 4000 | 0.850 | 0.851 | 1.7 |
| uniform | 400 | 0.85 | MAP-B | 2 | 11264 | 11264 | 207 | 0.608 | 0.725 | 10.1 |
| uniform | 400 | 0.85 | MAP-I | 1 | 7168 | 64512 | 4000 | 0.851 | 0.851 | 1.7 |
| uniform | 400 | 0.85 | MAP-I | 2 | 7168 | 64512 | 207 | 0.609 | 0.726 | 10.1 |
| uniform | 400 | 0.85 | FHRR | 1 | 3584 | 114688 | 4000 | 0.851 | 0.851 | 1.7 |
| uniform | 400 | 0.85 | FHRR | 2 | 3584 | 114688 | 207 | 0.609 | 0.726 | 10.1 |
| uniform | 400 | 0.85 | HRR | 1 | 8640 | 276480 | 4000 | 0.855 | 0.923 | 1.6 |
| uniform | 400 | 0.85 | HRR | 2 | 8640 | 276480 | 207 | 0.661 | 0.853 | 9.5 |
| uniform | 400 | 0.85 | BSDC-OR | 1 | 6656 | 6656 | 4000 | 0.889 | 0.891 | 1.5 |
| uniform | 400 | 0.85 | BSDC-OR | 2 | 6656 | 6656 | 207 | 0.654 | 0.794 | 9.5 |
| uniform | 400 | 0.85 | BSDC-S | 1 | 8704 | 78336 | 4000 | 0.852 (21: 0.836) | 0.853 | 1.7 |
| uniform | 400 | 0.85 | BSDC-S | 2 | 8704 | 78336 | 207 | 0.609 (21: 0.589) | 0.728 | 10.1 |
| uniform | 1600 | 0.5 | MAP-B | 1 | 29888 | 29888 | 4000 | 0.498 | 0.495 | 2.3 |
| uniform | 1600 | 0.5 | MAP-B | 2 | 29888 | 29888 | 1660 | 0.213 | 0.245 | 3.0 |
| uniform | 1600 | 0.5 | MAP-B | 3 | 29888 | 29888 | 290 | 0.072 | 0.121 | 5.4 |
| uniform | 1600 | 0.5 | MAP-I | 1 | 19072 | 209792 | 4000 | 0.500 | 0.497 | 2.3 |
| uniform | 1600 | 0.5 | MAP-I | 2 | 19072 | 209792 | 1660 | 0.214 | 0.247 | 3.0 |
| uniform | 1600 | 0.5 | MAP-I | 3 | 19072 | 209792 | 290 | 0.074 | 0.123 | 5.4 |
| uniform | 1600 | 0.5 | FHRR | 1 | 9536 | 305152 | 4000 | 0.500 | 0.497 | 2.3 |
| uniform | 1600 | 0.5 | FHRR | 2 | 9536 | 305152 | 1660 | 0.214 | 0.247 | 3.0 |
| uniform | 1600 | 0.5 | FHRR | 3 | 9536 | 305152 | 290 | 0.074 | 0.123 | 5.4 |
| uniform | 1600 | 0.5 | HRR | 1 | 21760 | 696320 | 4000 | 0.499 | 0.585 | 2.3 |
| uniform | 1600 | 0.5 | HRR | 2 | 21760 | 696320 | 1660 | 0.252 | 0.342 | 3.2 |
| uniform | 1600 | 0.5 | HRR | 3 | 21760 | 696320 | 290 | 0.096 | 0.200 | 6.1 |
| uniform | 1600 | 0.5 | BSDC-OR | 1 | 24576 | 24576 | 4000 | 0.530 | 0.529 | 2.3 |
| uniform | 1600 | 0.5 | BSDC-OR | 2 | 24576 | 24576 | 1660 | 0.231 | 0.280 | 3.1 |
| uniform | 1600 | 0.5 | BSDC-OR | 3 | 24576 | 24576 | 290 | 0.073 | 0.149 | 5.5 |
| uniform | 1600 | 0.5 | BSDC-S | 1 | 26624 | 292864 | 4000 | 0.513 (21: 0.497) | 0.511 | 2.3 |
| uniform | 1600 | 0.5 | BSDC-S | 2 | 26624 | 292864 | 1660 | 0.225 (21: 0.212) | 0.261 | 3.1 |
| uniform | 1600 | 0.5 | BSDC-S | 3 | 26624 | 292864 | 290 | 0.079 (21: 0.073) | 0.133 | 5.6 |
| uniform | 1600 | 0.85 | MAP-B | 1 | 51968 | 51968 | 4000 | 0.847 | 0.848 | 1.7 |
| uniform | 1600 | 0.85 | MAP-B | 2 | 51968 | 51968 | 1660 | 0.578 | 0.718 | 3.4 |
| uniform | 1600 | 0.85 | MAP-B | 3 | 51968 | 51968 | 290 | 0.271 | 0.609 | 8.6 |
| uniform | 1600 | 0.85 | MAP-I | 1 | 33088 | 363968 | 4000 | 0.847 | 0.848 | 1.7 |
| uniform | 1600 | 0.85 | MAP-I | 2 | 33088 | 363968 | 1660 | 0.579 | 0.719 | 3.4 |
| uniform | 1600 | 0.85 | MAP-I | 3 | 33088 | 363968 | 290 | 0.272 | 0.610 | 8.6 |
| uniform | 1600 | 0.85 | FHRR | 1 | 16576 | 530432 | 4000 | 0.848 | 0.849 | 1.7 |
| uniform | 1600 | 0.85 | FHRR | 2 | 16576 | 530432 | 1660 | 0.580 | 0.721 | 3.4 |
| uniform | 1600 | 0.85 | FHRR | 3 | 16576 | 530432 | 290 | 0.273 | 0.612 | 8.6 |
| uniform | 1600 | 0.85 | HRR | 1 | 38592 | 1234944 | 4000 | 0.848 | 0.914 | 1.6 |
| uniform | 1600 | 0.85 | HRR | 2 | 38592 | 1234944 | 1660 | 0.627 | 0.836 | 3.2 |
| uniform | 1600 | 0.85 | HRR | 3 | 38592 | 1234944 | 290 | 0.305 | 0.765 | 8.6 |
| uniform | 1600 | 0.85 | BSDC-OR | 1 | 30720 | 30720 | 4000 | 0.891 | 0.893 | 1.4 |
| uniform | 1600 | 0.85 | BSDC-OR | 2 | 30720 | 30720 | 1660 | 0.628 | 0.797 | 3.2 |
| uniform | 1600 | 0.85 | BSDC-OR | 3 | 30720 | 30720 | 290 | 0.291 | 0.713 | 8.4 |
| uniform | 1600 | 0.85 | BSDC-S | 1 | 40960 | 450560 | 4000 | 0.855 (21: 0.841) | 0.856 | 1.6 |
| uniform | 1600 | 0.85 | BSDC-S | 2 | 40960 | 450560 | 1660 | 0.587 (21: 0.570) | 0.732 | 3.4 |
| uniform | 1600 | 0.85 | BSDC-S | 3 | 40960 | 450560 | 290 | 0.276 (21: 0.266) | 0.627 | 8.6 |
| uniform | 3200 | 0.5 | MAP-B | 1 | 64896 | 64896 | 4000 | 0.497 | 0.493 | 2.3 |
| uniform | 3200 | 0.5 | MAP-B | 2 | 64896 | 64896 | 2000 | 0.216 | 0.243 | 2.7 |
| uniform | 3200 | 0.5 | MAP-B | 3 | 64896 | 64896 | 487 | 0.077 | 0.120 | 4.0 |
| uniform | 3200 | 0.5 | MAP-I | 1 | 41344 | 496128 | 4000 | 0.498 | 0.493 | 2.3 |
| uniform | 3200 | 0.5 | MAP-I | 2 | 41344 | 496128 | 2000 | 0.216 | 0.243 | 2.7 |
| uniform | 3200 | 0.5 | MAP-I | 3 | 41344 | 496128 | 487 | 0.077 | 0.120 | 4.0 |
| uniform | 3200 | 0.5 | FHRR | 1 | 20672 | 661504 | 4000 | 0.498 | 0.493 | 2.3 |
| uniform | 3200 | 0.5 | FHRR | 2 | 20672 | 661504 | 2000 | 0.216 | 0.243 | 2.7 |
| uniform | 3200 | 0.5 | FHRR | 3 | 20672 | 661504 | 487 | 0.077 | 0.120 | 4.0 |
| uniform | 3200 | 0.5 | HRR | 1 | 47168 | 1509376 | 4000 | 0.497 | 0.586 | 2.3 |
| uniform | 3200 | 0.5 | HRR | 2 | 47168 | 1509376 | 2000 | 0.258 | 0.343 | 2.9 |
| uniform | 3200 | 0.5 | HRR | 3 | 47168 | 1509376 | 487 | 0.103 | 0.201 | 4.6 |
| uniform | 3200 | 0.5 | BSDC-OR | 1 | 53248 | 53248 | 4000 | 0.544 | 0.543 | 2.3 |
| uniform | 3200 | 0.5 | BSDC-OR | 2 | 53248 | 53248 | 2000 | 0.253 | 0.295 | 2.8 |
| uniform | 3200 | 0.5 | BSDC-OR | 3 | 53248 | 53248 | 487 | 0.089 | 0.160 | 4.3 |
| uniform | 3200 | 0.5 | BSDC-S | 1 | 57344 | 688128 | 4000 | 0.506 (21: 0.491) | 0.502 | 2.3 |
| uniform | 3200 | 0.5 | BSDC-S | 2 | 57344 | 688128 | 2000 | 0.223 (21: 0.210) | 0.252 | 2.7 |
| uniform | 3200 | 0.5 | BSDC-S | 3 | 57344 | 688128 | 487 | 0.079 (21: 0.073) | 0.126 | 4.1 |
| uniform | 3200 | 0.85 | MAP-B | 1 | 110464 | 110464 | 4000 | 0.845 | 0.846 | 1.7 |
| uniform | 3200 | 0.85 | MAP-B | 2 | 110464 | 110464 | 2000 | 0.605 | 0.716 | 3.0 |
| uniform | 3200 | 0.85 | MAP-B | 3 | 110464 | 110464 | 487 | 0.315 | 0.606 | 6.4 |
| uniform | 3200 | 0.85 | MAP-I | 1 | 70336 | 844032 | 4000 | 0.846 | 0.847 | 1.7 |
| uniform | 3200 | 0.85 | MAP-I | 2 | 70336 | 844032 | 2000 | 0.605 | 0.717 | 3.0 |
| uniform | 3200 | 0.85 | MAP-I | 3 | 70336 | 844032 | 487 | 0.314 | 0.607 | 6.4 |
| uniform | 3200 | 0.85 | FHRR | 1 | 35200 | 1126400 | 4000 | 0.846 | 0.847 | 1.7 |
| uniform | 3200 | 0.85 | FHRR | 2 | 35200 | 1126400 | 2000 | 0.606 | 0.718 | 3.0 |
| uniform | 3200 | 0.85 | FHRR | 3 | 35200 | 1126400 | 487 | 0.314 | 0.608 | 6.4 |
| uniform | 3200 | 0.85 | HRR | 1 | 81920 | 2621440 | 4000 | 0.845 | 0.915 | 1.6 |
| uniform | 3200 | 0.85 | HRR | 2 | 81920 | 2621440 | 2000 | 0.659 | 0.837 | 2.8 |
| uniform | 3200 | 0.85 | HRR | 3 | 81920 | 2621440 | 487 | 0.361 | 0.766 | 6.3 |
| uniform | 3200 | 0.85 | BSDC-OR | 1 | 65536 | 65536 | 4000 | 0.894 | 0.898 | 1.4 |
| uniform | 3200 | 0.85 | BSDC-OR | 2 | 65536 | 65536 | 2000 | 0.672 | 0.806 | 2.7 |
| uniform | 3200 | 0.85 | BSDC-OR | 3 | 65536 | 65536 | 487 | 0.354 | 0.724 | 6.2 |
| uniform | 3200 | 0.85 | BSDC-S | 1 | 90112 | 1081344 | 4000 | 0.872 (21: 0.859) | 0.873 | 1.5 |
| uniform | 3200 | 0.85 | BSDC-S | 2 | 90112 | 1081344 | 2000 | 0.641 (21: 0.624) | 0.763 | 2.8 |
| uniform | 3200 | 0.85 | BSDC-S | 3 | 90112 | 1081344 | 487 | 0.339 (21: 0.326) | 0.666 | 6.3 |
| dense | 100 | 0.5 | MAP-B | 1 | 960 | 960 | 1739 | 0.501 | 0.482 | 3.5 |
| dense | 100 | 0.5 | MAP-I | 1 | 640 | 4480 | 1739 | 0.524 | 0.506 | 3.5 |
| dense | 100 | 0.5 | FHRR | 1 | 320 | 10240 | 1739 | 0.524 | 0.506 | 3.5 |
| dense | 100 | 0.5 | HRR | 1 | 960 | 30720 | 1739 | 0.494 | 0.705 | 3.5 |
| dense | 100 | 0.5 | BSDC-OR | 1 | 896 | 896 | 1739 | 0.546 | 0.529 | 3.5 |
| dense | 100 | 0.5 | BSDC-S | 1 | 896 | 6272 | 1739 | 0.532 (21: 0.510) | 0.517 | 3.5 |
| dense | 100 | 0.85 | MAP-B | 1 | 2112 | 2112 | 1739 | 0.856 | 0.853 | 2.5 |
| dense | 100 | 0.85 | MAP-I | 1 | 1344 | 9408 | 1739 | 0.858 | 0.855 | 2.5 |
| dense | 100 | 0.85 | FHRR | 1 | 704 | 22528 | 1739 | 0.874 | 0.872 | 2.3 |
| dense | 100 | 0.85 | HRR | 1 | 2560 | 81920 | 1739 | 0.842 | 0.990 | 2.5 |
| dense | 100 | 0.85 | BSDC-OR | 1 | 1280 | 1280 | 1739 | 0.896 | 0.892 | 2.2 |
| dense | 100 | 0.85 | BSDC-S | 1 | 1664 | 11648 | 1739 | 0.872 (21: 0.855) | 0.871 | 2.4 |
| dense | 400 | 0.5 | MAP-B | 1 | 5440 | 5440 | 3950 | 0.509 | 0.501 | 2.3 |
| dense | 400 | 0.5 | MAP-B | 2 | 5440 | 5440 | 924 | 0.237 | 0.257 | 4.5 |
| dense | 400 | 0.5 | MAP-I | 1 | 3456 | 31104 | 3950 | 0.508 | 0.500 | 2.3 |
| dense | 400 | 0.5 | MAP-I | 2 | 3456 | 31104 | 924 | 0.237 | 0.257 | 4.5 |
| dense | 400 | 0.5 | FHRR | 1 | 1728 | 55296 | 3950 | 0.508 | 0.500 | 2.3 |
| dense | 400 | 0.5 | FHRR | 2 | 1728 | 55296 | 924 | 0.237 | 0.257 | 4.5 |
| dense | 400 | 0.5 | HRR | 1 | 5568 | 178176 | 3950 | 0.527 | 0.770 | 2.3 |
| dense | 400 | 0.5 | HRR | 2 | 5568 | 178176 | 924 | 0.326 | 0.599 | 5.0 |
| dense | 400 | 0.5 | BSDC-OR | 1 | 4608 | 4608 | 3950 | 0.533 | 0.515 | 2.3 |
| dense | 400 | 0.5 | BSDC-OR | 2 | 4608 | 4608 | 924 | 0.323 | 0.283 | 4.6 |
| dense | 400 | 0.5 | BSDC-S | 1 | 5120 | 46080 | 3950 | 0.556 (21: 0.537) | 0.554 | 2.3 |
| dense | 400 | 0.5 | BSDC-S | 2 | 5120 | 46080 | 924 | 0.281 (21: 0.261) | 0.317 | 4.8 |
| dense | 400 | 0.85 | MAP-B | 1 | 10432 | 10432 | 3950 | 0.852 | 0.854 | 1.7 |
| dense | 400 | 0.85 | MAP-B | 2 | 10432 | 10432 | 924 | 0.687 | 0.734 | 4.9 |
| dense | 400 | 0.85 | MAP-I | 1 | 6656 | 59904 | 3950 | 0.854 | 0.856 | 1.7 |
| dense | 400 | 0.85 | MAP-I | 2 | 6656 | 59904 | 924 | 0.690 | 0.737 | 4.9 |
| dense | 400 | 0.85 | FHRR | 1 | 3328 | 106496 | 3950 | 0.854 | 0.856 | 1.7 |
| dense | 400 | 0.85 | FHRR | 2 | 3328 | 106496 | 924 | 0.690 | 0.737 | 4.9 |
| dense | 400 | 0.85 | HRR | 1 | 13184 | 421888 | 3950 | 0.870 | 0.995 | 1.5 |
| dense | 400 | 0.85 | HRR | 2 | 13184 | 421888 | 924 | 0.846 | 0.990 | 3.2 |
| dense | 400 | 0.85 | BSDC-OR | 1 | 6144 | 6144 | 3950 | 0.891 | 0.886 | 1.5 |
| dense | 400 | 0.85 | BSDC-OR | 2 | 6144 | 6144 | 924 | 0.788 | 0.792 | 4.3 |
| dense | 400 | 0.85 | BSDC-S | 1 | 8192 | 73728 | 3950 | 0.862 (21: 0.846) | 0.867 | 1.6 |
| dense | 400 | 0.85 | BSDC-S | 2 | 8192 | 73728 | 924 | 0.699 (21: 0.673) | 0.756 | 4.8 |
| dense | 1600 | 0.5 | MAP-B | 1 | 28096 | 28096 | 4000 | 0.499 | 0.499 | 2.3 |
| dense | 1600 | 0.5 | MAP-B | 2 | 28096 | 28096 | 1059 | 0.226 | 0.255 | 3.8 |
| dense | 1600 | 0.5 | MAP-I | 1 | 17920 | 197120 | 4000 | 0.500 | 0.500 | 2.3 |
| dense | 1600 | 0.5 | MAP-I | 2 | 17920 | 197120 | 1059 | 0.228 | 0.256 | 3.8 |
| dense | 1600 | 0.5 | FHRR | 1 | 8960 | 286720 | 4000 | 0.500 | 0.500 | 2.3 |
| dense | 1600 | 0.5 | FHRR | 2 | 8960 | 286720 | 1059 | 0.228 | 0.256 | 3.8 |
| dense | 1600 | 0.5 | HRR | 1 | 27072 | 866304 | 4000 | 0.477 | 0.766 | 2.2 |
| dense | 1600 | 0.5 | HRR | 2 | 27072 | 866304 | 1059 | 0.323 | 0.592 | 4.1 |
| dense | 1600 | 0.5 | BSDC-OR | 1 | 22528 | 22528 | 4000 | 0.499 | 0.472 | 2.3 |
| dense | 1600 | 0.5 | BSDC-OR | 2 | 22528 | 22528 | 1059 | 0.293 | 0.239 | 3.8 |
| dense | 1600 | 0.5 | BSDC-S | 1 | 26624 | 292864 | 4000 | 0.557 (21: 0.541) | 0.566 | 2.3 |
| dense | 1600 | 0.5 | BSDC-S | 2 | 26624 | 292864 | 1059 | 0.279 (21: 0.263) | 0.329 | 4.1 |
| dense | 1600 | 0.85 | MAP-B | 1 | 50944 | 50944 | 4000 | 0.855 | 0.862 | 1.6 |
| dense | 1600 | 0.85 | MAP-B | 2 | 50944 | 50944 | 1059 | 0.682 | 0.748 | 4.1 |
| dense | 1600 | 0.85 | MAP-I | 1 | 32448 | 356928 | 4000 | 0.855 | 0.862 | 1.6 |
| dense | 1600 | 0.85 | MAP-I | 2 | 32448 | 356928 | 1059 | 0.683 | 0.748 | 4.1 |
| dense | 1600 | 0.85 | FHRR | 1 | 16256 | 520192 | 4000 | 0.856 | 0.863 | 1.6 |
| dense | 1600 | 0.85 | FHRR | 2 | 16256 | 520192 | 1059 | 0.684 | 0.750 | 4.1 |
| dense | 1600 | 0.85 | HRR | 1 | 58624 | 1875968 | 4000 | 0.830 | 0.994 | 1.7 |
| dense | 1600 | 0.85 | HRR | 2 | 58624 | 1875968 | 1059 | 0.806 | 0.989 | 3.2 |
| dense | 1600 | 0.85 | BSDC-OR | 1 | 28672 | 28672 | 4000 | 0.880 | 0.872 | 1.5 |
| dense | 1600 | 0.85 | BSDC-OR | 2 | 28672 | 28672 | 1059 | 0.765 | 0.769 | 3.7 |
| dense | 1600 | 0.85 | BSDC-S | 1 | 40960 | 450560 | 4000 | 0.873 (21: 0.860) | 0.884 | 1.5 |
| dense | 1600 | 0.85 | BSDC-S | 2 | 40960 | 450560 | 1059 | 0.707 (21: 0.685) | 0.787 | 4.0 |
| dense | 3200 | 0.5 | MAP-B | 1 | 62080 | 62080 | 4000 | 0.499 | 0.495 | 2.3 |
| dense | 3200 | 0.5 | MAP-B | 2 | 62080 | 62080 | 1242 | 0.257 | 0.247 | 3.6 |
| dense | 3200 | 0.5 | MAP-I | 1 | 39488 | 473856 | 4000 | 0.498 | 0.494 | 2.3 |
| dense | 3200 | 0.5 | MAP-I | 2 | 39488 | 473856 | 1242 | 0.256 | 0.247 | 3.6 |
| dense | 3200 | 0.5 | FHRR | 1 | 19776 | 632832 | 4000 | 0.499 | 0.495 | 2.3 |
| dense | 3200 | 0.5 | FHRR | 2 | 19776 | 632832 | 1242 | 0.257 | 0.248 | 3.6 |
| dense | 3200 | 0.5 | HRR | 1 | 62208 | 1990656 | 4000 | 0.490 | 0.798 | 2.2 |
| dense | 3200 | 0.5 | HRR | 2 | 62208 | 1990656 | 1242 | 0.489 | 0.640 | 4.1 |
| dense | 3200 | 0.5 | BSDC-OR | 1 | 53248 | 53248 | 4000 | 0.648 | 0.631 | 2.2 |
| dense | 3200 | 0.5 | BSDC-OR | 2 | 53248 | 53248 | 1242 | 0.418 | 0.406 | 4.1 |
| dense | 3200 | 0.5 | BSDC-S | 1 | 57344 | 688128 | 4000 | 0.540 (21: 0.524) | 0.542 | 2.3 |
| dense | 3200 | 0.5 | BSDC-S | 2 | 57344 | 688128 | 1242 | 0.295 (21: 0.279) | 0.298 | 3.8 |
| dense | 3200 | 0.85 | MAP-B | 1 | 107392 | 107392 | 4000 | 0.849 | 0.850 | 1.7 |
| dense | 3200 | 0.85 | MAP-B | 2 | 107392 | 107392 | 1242 | 0.696 | 0.725 | 3.7 |
| dense | 3200 | 0.85 | MAP-I | 1 | 68352 | 820224 | 4000 | 0.849 | 0.850 | 1.7 |
| dense | 3200 | 0.85 | MAP-I | 2 | 68352 | 820224 | 1242 | 0.696 | 0.725 | 3.7 |
| dense | 3200 | 0.85 | FHRR | 1 | 34176 | 1093632 | 4000 | 0.849 | 0.850 | 1.7 |
| dense | 3200 | 0.85 | FHRR | 2 | 34176 | 1093632 | 1242 | 0.696 | 0.725 | 3.7 |
| dense | 3200 | 0.85 | HRR | 1 | 129024 | 4128768 | 4000 | 0.840 | 0.996 | 1.7 |
| dense | 3200 | 0.85 | HRR | 2 | 129024 | 4128768 | 1242 | 0.916 | 0.992 | 1.9 |
| dense | 3200 | 0.85 | BSDC-OR | 1 | 61440 | 61440 | 4000 | 0.874 | 0.865 | 1.5 |
| dense | 3200 | 0.85 | BSDC-OR | 2 | 61440 | 61440 | 1242 | 0.739 | 0.754 | 3.5 |
| dense | 3200 | 0.85 | BSDC-S | 1 | 86016 | 1032192 | 4000 | 0.863 (21: 0.849) | 0.867 | 1.6 |
| dense | 3200 | 0.85 | BSDC-S | 2 | 86016 | 1032192 | 1242 | 0.717 (21: 0.696) | 0.755 | 3.7 |

Per gruppo (famiglia, campione, h), sulle celle esistenti:

| famiglia | campione | h | celle | MDE gruppo (punti) | divario medio previsto |contab.−ingenuo| |
|---|---|---|---|---|---|
| BSDC-OR | dense | 1 | 8 | 0.8 | 1.3 |
| BSDC-S | dense | 1 | 8 | 0.8 | 0.6 |
| FHRR | dense | 1 | 8 | 0.8 | 0.5 |
| HRR | dense | 1 | 8 | 0.8 | 20.5 |
| MAP-B | dense | 1 | 8 | 0.8 | 0.6 |
| MAP-I | dense | 1 | 8 | 0.8 | 0.5 |
| BSDC-OR | dense | 2 | 6 | 1.6 | 2.1 |
| BSDC-S | dense | 2 | 6 | 1.7 | 4.4 |
| FHRR | dense | 2 | 6 | 1.7 | 3.3 |
| HRR | dense | 2 | 6 | 1.5 | 18.3 |
| MAP-B | dense | 2 | 6 | 1.7 | 3.3 |
| MAP-I | dense | 2 | 6 | 1.7 | 3.3 |
| BSDC-OR | uniform | 1 | 8 | 0.8 | 0.1 |
| BSDC-S | uniform | 1 | 8 | 0.8 | 0.1 |
| FHRR | uniform | 1 | 8 | 0.8 | 0.1 |
| HRR | uniform | 1 | 8 | 0.8 | 8.2 |
| MAP-B | uniform | 1 | 8 | 0.8 | 0.1 |
| MAP-I | uniform | 1 | 8 | 0.8 | 0.1 |
| BSDC-OR | uniform | 2 | 6 | 2.4 | 9.5 |
| BSDC-S | uniform | 2 | 6 | 2.5 | 8.1 |
| FHRR | uniform | 2 | 6 | 2.5 | 7.6 |
| HRR | uniform | 2 | 6 | 2.5 | 14.1 |
| MAP-B | uniform | 2 | 6 | 2.5 | 7.6 |
| MAP-I | uniform | 2 | 6 | 2.5 | 7.6 |
| BSDC-OR | uniform | 3 | 4 | 3.1 | 23.5 |
| BSDC-S | uniform | 3 | 4 | 3.2 | 19.5 |
| FHRR | uniform | 3 | 4 | 3.2 | 18.1 |
| HRR | uniform | 3 | 4 | 3.3 | 26.7 |
| MAP-B | uniform | 3 | 4 | 3.2 | 18.0 |
| MAP-I | uniform | 3 | 4 | 3.2 | 18.1 |

## Ipotesi e criteri, fissati ora

Gruppo = (famiglia, campione, h): le celle dei 4 carichi × 2 target che esistono. Per
ogni gruppo si riportano MAE (punti), massimo di cella, MAE del FKS ingenuo, errore
con segno medio e IC al 95 % a cluster (errore per seed mediato sulle celle, t₁₉).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** replica su dati nuovi: uniform, h = 1, per ognuna delle 6 famiglie | MAE ≤ 3 | MAE > 5 |
| **H2** contabilità sui densi: dense, h = 1 | MAE ≤ 3 in tutti e 6 i gruppi | MAE > 5 in almeno un gruppo |
| **H3** BSDC-S corretto (passo 1): errore con segno nei 2 gruppi h = 1 | \|errore con segno\| ≤ 1.0 in entrambi | errore con segno ≥ +1.5 in entrambi (il bias della 21) |
| **H4** HRR (passo 2), h = 1 | MAE ≤ 3 in entrambi i gruppi e MAE_ingenuo − MAE ≥ X_HRR in entrambi | MAE > 5 in un gruppo, o MAE_ingenuo − MAE < 0 in un gruppo |
| **H5** ablazione FKS ingenuo, dense, h = 1 | MAE_ingenuo − MAE ≥ X in tutte le famiglie ammesse | MAE_ingenuo − MAE < 0 in 2 o più famiglie ammesse (**nessuna ammessa su CoDEx-M: descrittiva**) |
| **H6 (rischiosa)** catene: la previsione a prodotto (MAP-B: con il fattore di dipendenza) regge a 2 e 3 hop | MAE ≤ 3 in tutti i 18 gruppi catena esistenti (uniform h = 2, uniform h = 3, dense h = 2; 6 famiglie) | MAE > 5 in almeno un gruppo catena |
| **H7** nessun bias oltre il campionamento, per famiglia (celle h = 1) | al più 2 celle con \|t₁₉\| > 2.861 (bilaterale 0.01) | 5 o più celle |
| **H8** D fuori campione (seed 10–19), per famiglia | misurato medio ≥ target − 3 punti in tutte le celle h = 1 tranne al più una | in 4 o più celle meno del target − 3 |
| **H9** ablazione sulle catene: il prodotto delle previsioni ingenue degli hop contro quello con contabilità | MAE_ingenuo − MAE ≥ X (del gruppo) in almeno 15 dei 18 gruppi catena | MAE_ingenuo − MAE < 0 in 4 o più gruppi |

Il resto è **sostenuto in parte**.

**Perché H6 è rischiosa.** L'errore relativo di un hop si moltiplica per h; le catene
passano per nodi ad alto grado (più alias e gemelli lungo il percorso); il fattore di
dipendenza di MAP-B è calcolato per fatti uniformi; l'indipendenza fra hop per le altre
famiglie è un'approssimazione derivata ma verificata solo su sintetico con N ≤ 400;
nessuna catena è mai stata prevista a priori su un KG reale in questo progetto. Una
falsificazione direbbe che la contabilità a 1 hop non basta per i percorsi.

**Soglie X, regola scritta prima di vedere le previsioni di CoDEx:** per un gruppo, X è
metà del divario medio previsto |previsto − ingenuo| del gruppo, arrotondato per difetto
al mezzo punto, minimo 1 punto. Una famiglia è **ammessa** a H5 solo se il divario
previsto del suo gruppo dense h = 1 è ≥ 2 punti (sotto, la contabilità e FKS non sono
distinguibili e il gruppo è riportato come descrittivo). X_HRR si calcola con la stessa
regola sui gruppi HRR h = 1. Valori risultanti: nessuna famiglia è ammessa a H5 (divari dense h = 1 previsti 0.5–1.3 punti: H5 diventa
**descrittiva**, non testabile su CoDEx-M); X_HRR = 4.0 (uniform, divario 8.2) e 10.0
(dense, divario 20.5); per H9, X = 1.0 (BSDC-OR dense h = 2, divario 2.1), 1.5 (MAP-B,
MAP-I, FHRR dense h = 2, 3.3), 2.0 (BSDC-S dense h = 2, 4.4), 9.0 (HRR dense h = 2,
18.3); uniform h = 2: 3.5 (MAP-B, MAP-I, FHRR), 4.0 (BSDC-S), 4.5 (BSDC-OR), 7.0 (HRR);
uniform h = 3: 9.0 (MAP-B, MAP-I, FHRR), 9.5 (BSDC-S), 11.5 (BSDC-OR), 13.0 (HRR).

**Potenza.** Le soglie 3/5 punti sono confrontate con l'MDE di gruppo (tabella sopra):
MDE di gruppo 0.8 punti a 1 hop, 1.5–2.5 a 2 hop, 3.1–3.3 a 3 hop (limite inferiore,
SE di modello). A 1 e 2 hop le soglie 3/5 sono ben sopra l'MDE; a 3 hop la soglia di
sostegno (3) è all'MDE: un errore vero di 3 punti a 3 hop si rileva con potenza ≈ 0.8,
uno minore no. Errore assoluto atteso per solo rumore in una cella a 3 hop ≈ 1.7 punti
(SE di cella ≈ 2.2): un MAE di gruppo > 5 per solo rumore è improbabile (< 1 %) ma un
MAE fra 3 e 5 (sostenuta in parte) non lo è. Le celle a 1 hop hanno MDE di cella
2.6–4.6 punti.

**Descrittivi, non ipotesi:** l'errore con segno per gruppo; il confronto fra
`pred_v21` e la misura per BSDC-S (ablazione della correzione dei pareggi); il fattore
di dipendenza di MAP-B contro il prodotto semplice.

**Previsioni (non criteri).** H1, H2 sostenute (H2 è quasi una replica: CoDEx-M ha
pochi alias e gemelli). H3 sostenuta, con rischio a N piccolo (il residuo del passo 1 a
B piccolo). H4 sostenuta. H5 descrittiva. **H6 a rischio**: lo smoke a 3 hop ha già dato
MAE 6–7 per MAP-B e MAP-I su poche catene. H7, H8 sostenute. H9 sostenuta (i divari
previsti sulle catene sono grandi: il prodotto ingenuo ignora che gli hop passano per
nodi con più relazioni e oggetti).

## Cosa ho visto prima

- I risultati sintetici dei passi 1–3 (tabelle sopra, `results/generalization_diagnosis.json`).
- Lo smoke (`--smoke`: grafo sintetico 400 entità, N = 40 e 80, 20 seed, impianto
  intero con misure, codice congelato): MAE per gruppo h = 1 0.7–1.4 punti in tutte le
  famiglie (MAE FKS ingenuo dense 5.9–13.1); h = 2 1.1–3.9; **h = 3 (solo dense, 2
  celle, poche catene) 0.5–6.8: MAP-B 6.8, MAP-I 6.1, BSDC-OR 3.6** — un segnale
  sintetico a favore della falsificazione di H6, con SE grandi. Un t con SE = 0 vale 0 se
  anche l'errore è 0 (0/0 = nessuna differenza), ±∞ altrimenti. L'Engine MAP-B dà le stesse risposte di
  `abm.Memory.query` e la previsione MAP-B per firma coincide con `exact.predict_queries`.
- Le previsioni su CoDEx-M (`--predict`), che leggono i dati ma non costruiscono memorie.
- **Scelte dopo aver visto misure sintetiche:** la correzione di BSDC-S (passo 1) è
  stata scelta dopo la diagnosi, che è il suo scopo; è derivata e senza parametri, e il
  modello condizionato alla traccia, più vicino all'ipotesi del revisore, non è stato
  preferito perché equivalente e più costoso. Il numero di seed della diagnosi e della
  verifica è passato da 8 a 16 dopo il primo giro (SE troppo grandi per le catene).
  Nient'altro è stato cambiato dopo le misure sintetiche.

## Tempo stimato

`--predict` 33 min (1999 s; il run lo ripete, perché ricalcola D e previsioni). Misure,
stimate con `--timing` (una memoria per famiglia e target al D previsto, su un grafo
sintetico con lo stesso N, × 20 seed): 74 min, dominati da HRR a N = 3200 (38–57 s per
seed) e da MAP-B/FHRR a N = 3200. Totale **≈ 1 h 50 min** su questa macchina (12 core,
un processo), sotto le 2 h: **griglia non ridotta**. Il grafo sintetico ha un codebook
un po' più piccolo di CoDEx-M a N = 3200 (il costo cresce con M·D): se il run supera le
3 h, è comunque eseguito per intero (nessuna regola di riduzione dopo la partenza).

## Limiti dichiarati prima

- Un solo grafo nuovo (CoDEx-M, da Wikidata); celle uniform = replica di FKS e della 21.
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
