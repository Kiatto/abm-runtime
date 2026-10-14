# Preregistrazione 23 — il dimensionatore mantiene la promessa su grafi mai visti?

**Data: 2026-10-13.** Prima versione committata in `22970dc` **prima** di misurare; questa
versione la **sostituisce** dopo una revisione ostile del disegno, ancora **prima** di
costruire qualsiasi memoria sui grafi reali. File: harness
[`examples/sizing_tool_prereg.py`](../../examples/sizing_tool_prereg.py), previsioni
congelate `results/sizing_tool_prereg_predictions.json` (rigenerate dopo la revisione).
Prima del commit l'harness è stato eseguito solo con `--smoke` (grafi sintetici generati
nello script), `--predict` (legge i sei grafi, campiona sottografi e query, chiama
`recommend(..., verify=False)` e `predict`: **nessuna memoria costruita** sui dati reali),
`--timing` (memorie solo su grafi sintetici) e `--power` (simulazione dalle previsioni).
Push e marche OpenTimestamps li fa il coordinatore dopo il commit.

## Revisione del disegno prima dei dati (dopo `22970dc`)

Le sezioni «Ipotesi», «Potenza», «Previsioni», «Controlli», «Tempo» di `22970dc` sono
sostituite da quelle sotto. Punto per punto:

1. **Cluster.** L'unità di inferenza è il **sottografo**: tutte le quote hanno un IC
   **bootstrap per cluster** (B = 10 000, seme 23 023, limiti unilaterali al 95 % =
   quantili 5 % e 95 %); secondario, l'esito di cluster ("tutte le unità del sottografo
   riescono") con Clopper-Pearson sul numero di cluster. Minimo **30 cluster** per H1, H2,
   H1u, H2u, H8u, H4, H5, H6, H7. Per arrivarci, oltre ai sottografi base si aggiungono
   (in ordine di k, dalle sole previsioni) sottografi con almeno una raccomandazione
   `validated` prevista: 68 cluster validated sui grafi di calibrazione, 61 sui mai visti.
2. **Grafi mai visti, ipotesi separate.** UMLS e **Kinship** (stesso repo ConvE al commit
   `f3c0eb2`, `kinship.tar.gz`, sha256
   `a3cb9ca67213eb38133aebc12901195e283d844ebd59f1a6cbf2d4fa12ced4e6`, 8544 fatti di
   train; voce in `examples/replicate.py`) hanno ipotesi proprie (H1u, H2u, H8u).
   **H1 e H2 sui quattro grafi della calibrazione si chiamano «replica in-distribuzione»**:
   i sottografi sono nuovi, i grafi no.
3. **Il vero primario è su tutte le chiavi (s, r) del sottografo**; le 200 query della
   verifica sono il secondario (H1_q, H2_q). Anche la previsione per l'errore del modello
   (H3, H6) è su tutte le chiavi.
4. **H3b sostituita** da H5 (copertura), H6 (discriminazione: risk ratio) e H7 (valore
   contro una baseline empirica senza modello); H3 resta.
5. **H4**: il D minimo usa lo **stesso criterio della verifica** (IC_inf 95 % ≥ target −
   tol, t di Student con 19 gdl) su **altre 20 memorie** con semi indipendenti dal vero.
6. **Regola di arresto**: la stima di `--timing` sceglie il passo e non tronca; checkpoint
   JSONL per sottografo nell'ordine di priorità fisso (`select`), nessun riepilogo
   intermedio; ripresa dopo un crash saltando i sottografi fatti; un checkpoint con un
   altro passo blocca la ripresa (cambio di passo = deviazione dichiarata); **tetto rigido
   12 h**: si analizza il prefisso scritto (che è fatto di sottografi interi, in ordine).
7. **Esclusioni HRR**: la stessa funzione (`families.estimate_bytes` con n_queries = |Q|,
   come nella verifica di `recommend`) decide se una memoria si costruisce nella verifica e
   nel vero; gli esclusi sono elencati per chiave nel risultato; H1 e H2 hanno un'analisi
   **worst-case** con gli esclusi contati come fallimenti.
8. **Controlli**: 6 sintetici **avversari** da un generatore con parametri estratti ora;
   un **controllo negativo** (uniforme: deve essere `validated` e avere ragione); un
   controllo **«modello rotto»** nello smoke (previsione +5 punti via monkeypatch di
   `sizing._predict_graph`): H2 e H3 devono fallire. Lo smoke gira con la configurazione
   del passo scelto.

Bug corretti: la memo di `Truth` ora include i semi; `--timing` usa il limite di HRR del
passo (e la stessa funzione di esclusione del run), misura la costruzione separata dalle
query (la traccia si costruisce pigramente alla prima query) e prende il **massimo di 3
ripetizioni**; la griglia si applica con `configure(step)`, la stessa via per run, smoke,
timing e power; i semi di verifica, vero, D minimo e baseline sono derivati da
(sottografo, target). Il conteggio di `22970dc` era **327 validated previste, 9 HRR
escluse dal limite di memoria → 318 misurabili**; qui i conteggi riportano sempre i due
numeri.

**Deviazione dichiarata dal mandato iniziale:** il budget per scegliere il passo passa da
3 h a **10 h** (tetto rigido 12 h). Con i punti 1–3 (≥ 30 cluster, grafi mai visti con K
alto, vero su tutte le chiavi) nessun passo stimato sta sotto le 3 h (minimo 4.0 h, passo 6).

## Perché

Le 21–22 hanno validato i **modelli** su sottografi specifici. `abm.sizing.recommend` è un
**prodotto** costruito e tarato su quelle celle (STRUCT_RANGES, OBSERVED_ERROR, regole di
affidabilità, criterio della verifica). La sua promessa va collaudata su dati che non
l'hanno toccato:

- (a) con `verify=True`, la raccomandazione marcata `validated` e `verified` raggiunge il
  target su memorie nuove;
- (b) senza verifica, le raccomandazioni `validated` raggiungono il target meno una banda
  nella grande maggioranza dei casi;
- (c) le regole di affidabilità segnalano (`caution`/`unvalidated`) i casi in cui il
  modello sbaglia di più di 3 punti, senza segnalare tutto;
- (d) la verifica costa poco: il D verificato non è molto più grande del minimo misurato.

## Dati

| grafo | file | sha256 | ruolo |
|---|---|---|---|
| FB15k-237 | `fb15k237_train.txt` | `6e4c2782…cdfae` | calibrazione (21) |
| WN18RR | `wn18rr_train.csv` | `28f7a0b3…e6a0` | calibrazione (21) |
| CoDEx-M | `codex/codex-m_train.txt` | `d99c3437…6556` | calibrazione (22) |
| YAGO3-10 | `yago3_10/YAGO3-10.tar.gz` | `9a3bdd5f…fb92` | calibrazione (22) |
| **UMLS** | `umls/umls.tar.gz` (train.txt, 5216 fatti) | `26223aa91f0ec8db2324e0f489b9a4425e04d1d0df72ce5cfd62af4b91aee6fd` | mai visto |
| **Kinship** | `kinship/kinship.tar.gz` (train.txt, 8544 fatti) | `a3cb9ca67213eb38133aebc12901195e283d844ebd59f1a6cbf2d4fa12ced4e6` | mai visto |

UMLS e Kinship vengono dal repo pubblico `TimDettmers/ConvE` al commit
`f3c0eb286025410fa5b4c04696c918264163d0ca`; entrambi sono piccoli (135 e 104 entità) e
densi. Nations scartato (1592 fatti, niente N = 1600).

**Sottografi.** `SeedSequence([23_000_023, indice grafo, indice schema, N, k])`,
k = 50 000 + j (mai usati). Pool con previsioni: j < 8 sui grafi di calibrazione, j < 40
sui mai visti (j < 8 a N = 1600). Campionamento uniform/dense copiato da
`examples/vsa_families_prereg.py`; Q = 200 chiavi a caso per la verifica.
**Semi per (sottografo, target)** (`SeedSequence([..., k, 1000 + indice target]).spawn(4)`):
`rng_seed` della verifica (che estrae i suoi semi in [0, 2^31)); vero 20 semi in
[2^31, 2^32); D minimo 20 semi in [2^32, 2^32 + 2^31); baseline 8 + 8 semi da 2^33:
disgiunti per costruzione.

## Impianto

- Griglia: 6 grafi × {uniform, dense} × N ∈ {100, 400, 1600} × target
  {0.70, 0.85, 0.95} (accuratezza assoluta) × 6 famiglie.
- **Regola del tetto:** (sottografo, target) scartato se target > min(tetto sulle 200
  query, tetto su tutte le chiavi) − 0.02.
- **Selezione e priorità** (`select`, solo dalle previsioni): per cella i primi k_in
  (k_un sui mai visti) sottografi sono **base** (campione non selezionato: H3, H5, H6);
  poi, in ordine di j, si aggiungono sottografi **extra** con almeno una `validated`
  prevista (dopo l'esclusione HRR) finché la cella ha k_val (k_val_un) sottografi con
  validated; gli extra entrano solo in H1, H2, H1u, H2u, H8u, H4, H7. Ordine di
  esecuzione: per giro (posizione nella lista della cella), poi grafo, schema, N.
- Per ogni (sottografo, target): `recommend(..., queries=Q, verify=True, seeds=8, tol=0.01,
  rng_seed, max_mem_bytes)`, HRR in una chiamata a parte con il suo limite. Le memorie
  costruite da `recommend` si contano per famiglia (sostituendo `families.make` durante la
  chiamata) per H7.
- **Vero**: 20 memorie fresche `families.make(f, D, seme)`; accuratezza su tutte le chiavi
  (primaria) e sulle Q (secondaria); al D finale e al D del modello.
- **D minimo** (H4, perimetro del passo): il più piccolo D della griglia (64; BSDC: L)
  accettato con IC_inf 95 % (20 memorie, t₁₉ = 2.093) ≥ target − 0.01 su tutte le chiavi;
  ricerca: raddoppi da D finale (fino a 8×), dimezzamenti, bisezione.
- **Baseline empirica** (H7, perimetro del passo): stessa regola di accettazione di
  `recommend` (IC_inf su 8 semi ≥ target − tol, sulle Q) ma **senza modello**: da D = 256
  (BSDC: multiplo di L ≥ 256), raddoppi fino a D_MAX o al limite di memoria, bisezione al
  passo, report su altri 8 semi (media ≥ target − tol). Si contano memorie, secondi, bit,
  e il vero (20 memorie) al suo D.

## Ipotesi e criteri (fissati ora)

Bootstrap per cluster: stima puntuale p̂ sulle unità; boot_inf/boot_sup = quantili 5 %/95 %.
"Sostenuta" e "falsificata" come sotto; altrimenti **non conclusiva**; con meno cluster del
minimo **non testabile**. Unità = raccomandazione (sottografo × target × famiglia).

| | promessa | unità (cluster) | successo | sostenuta | falsificata |
|---|---|---|---|---|---|
| **H1** replica in-distribuzione | (a) | validated ∧ verified, 4 grafi di calibrazione | vero (tutte le chiavi) al D finale ≥ target − 0.01 | p̂ ≥ 0.95 e boot_inf ≥ 0.90 | boot_sup < 0.95 |
| **H2** replica in-distribuzione | (b) | validated, 4 grafi di calibrazione | vero al D del modello ≥ target − 0.02 | p̂ ≥ 0.90 e boot_inf ≥ 0.85 | boot_sup < 0.90 |
| **H1u** mai visti | (a) | come H1 su UMLS + Kinship | come H1 | come H1 | come H1 |
| **H2u** mai visti | (b) | come H2 su UMLS + Kinship | come H2 | come H2 | come H2 |
| **H8u** mai visti | verifica fuori regime | verified (qualunque affidabilità) su UMLS + Kinship | come H1 | come H1 | come H1 |
| **H3** sensibilità | (c) | sottografi base, positivi: \|prev − vero\| > 3 punti (tutte le chiavi, D del modello) | affidabilità ≠ validated | p̂ ≥ 0.80 e boot_inf ≥ 0.60 (min. 10 positivi in ≥ 5 cluster) | boot_sup < 0.80 |
| **H5** copertura | utilità | (sottografo base, target) non scartati | almeno una validated ∧ verified | p̂ ≥ 0.50 e boot_inf ≥ 0.40 | boot_sup < 0.50 |
| **H6** discriminazione | (c) | sottografi base, tutte le misurate | RR = P(err > 3 \| non validated) / P(err > 3 \| validated), +0.5 a numeratori e +1 ai denominatori | RR ≥ 3 e boot_inf ≥ 1.5 | boot_sup < 3 |
| **H4** costo della verifica | (d) | verified nel perimetro del D minimo | gonfiamento D finale / D minimo − 1 | mediana ≤ 0.15 e boot_sup ≤ 0.20 | boot_inf > 0.15 |
| **H7** valore contro la baseline | utilità | verified dallo strumento **e** dalla baseline, perimetro H7 | rapporti memorie (strumento/baseline) e bit | mediana memorie ≤ 0.5 con boot_sup ≤ 0.6 **e** mediana bit ≤ 1.10 con boot_sup ≤ 1.10 | boot_inf memorie > 0.5 **o** boot_inf bit > 1.10 |

Motivazioni delle soglie: H1 come in `22970dc` (la verifica rifiuta un ottimismo ≥ tol con
probabilità ≥ 0.975 per costruzione; atteso p ≈ 0.97–0.99). H2: modello centrato, MAE
osservato ≤ 0.84 punti per le famiglie validate. H5: «lo strumento certifica almeno metà
delle richieste realistiche». **Dalle sole previsioni la copertura massima possibile è
0.337 (454 coppie base)**: H5 è prevista **falsificata** prima dei dati, ed è registrata
così (dice quanto è stretto il dominio in cui lo strumento si dichiara validato). H6: gli
avvisi devono separare, non solo accendersi (H3 da sola è soddisfatta da "tutto
unvalidated"). H7: lo strumento deve costruire al più metà delle memorie della ricerca
cieca a parità di criterio, senza costare più del 10 % di bit. H4 come prima.
Secondari (riportati, non decisionali): H1_q e H2_q sulle 200 query; H1/H2 worst-case con
gli esclusi HRR come fallimenti; esito di cluster con Clopper-Pearson; specificità; errori
solo ottimisti; tasso di `adjusted`, verificate ed errore medio per grafo, famiglia, regime;
successo della baseline sul vero.

## Potenza (dalle previsioni, passo scelto 2)

Simulazione (`power_cluster`, 400 ripetizioni, bootstrap 1000): cluster con le dimensioni
previste (raccomandazioni validated per sottografo), p_c ~ Beta di media p e correlazione
intra-cluster ρ. n per H1 è un limite superiore (le verified sono un sottoinsieme).
P(sostenuta) / P(falsificata):

| p vero | 0.99 | 0.98 | 0.97 | 0.95 | 0.92 | 0.90 | 0.85 |
|---|---|---|---|---|---|---|---|
| H1, 68 cluster (1109 racc.), ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.49 / 0.04 | 0 / 1 | 0 / 1 | 0 / 1 |
| H1, ρ = 0.2 | 1 / 0 | 1.00 / 0 | 0.95 / 0 | 0.57 / 0.02 | 0.01 / 0.59 | 0 / 0.97 | 0 / 1 |
| H2, ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1.00 / 0 | 0.48 / 0.04 | 0 / 1 |
| H2, ρ = 0.2 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.89 / 0.003 | 0.50 / 0.03 | 0.005 / 0.81 |
| H1u, 61 cluster (825 racc.), ρ = 0 | 1 / 0 | 1 / 0 | 1.00 / 0 | 0.58 / 0.03 | 0 / 0.96 | 0 / 1 | 0 / 1 |
| H1u, ρ = 0.2 | 1 / 0 | 1.00 / 0 | 0.95 / 0 | 0.52 / 0.06 | 0.05 / 0.56 | 0.003 / 0.88 | 0 / 1 |
| H2u, ρ = 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.99 / 0 | 0.53 / 0.04 | 0 / 1.00 |
| H2u, ρ = 0.2 | 1 / 0 | 1 / 0 | 1 / 0 | 1.00 / 0 | 0.85 / 0 | 0.55 / 0.04 | 0.02 / 0.71 |

H3: potenza binomiale di `22970dc` (sensibilità 0.9 → P(sost.) 0.87 con 20 positivi, 0.99
con 40), da ridurre per l'effetto cluster; numero di positivi ignoto. H6, H4, H7: nessuna
potenza in forma chiusa (distribuzioni ignote), dichiarato.

## Previsioni dello strumento (congelate, `--predict`) e celle

`results/sizing_tool_prereg_predictions.json`: 544 sottografi (pool), 8102 righe, per ogni
(sottografo, target, famiglia) D, bit, previsione su Q e su tutte le chiavi, affidabilità,
statistiche strutturali; scarti per il tetto; parametri dei controlli avversari; la
selezione del passo scelto (`selection`) e la potenza (`power_grid`). Nessuna famiglia
irraggiungibile entro d_max.

**Run set (passo 2): 266 sottografi (192 base + 74 extra), 3854 raccomandazioni
misurabili (1934 validated, 162 caution, 1758 unvalidated) + 82 HRR escluse dal limite di
256 MiB (34 delle quali validated: 1968 validated previste, 34 escluse → 1934).**
Sottografi per cella (base / extra / con almeno una validated prevista):

| grafo | schema | N | base | extra | con validated |
|---|---|---|---|---|---|
| fb15k237 | uniform | 100 / 400 / 1600 | 4 / 4 / 4 | 4 / 1 / 0 | 5 / 2 / 0 |
| fb15k237 | dense | 100 / 400 / 1600 | 4 / 4 / 4 | 1 / 1 / 2 | 1 / 1 / 4 |
| wn18rr | uniform | 100 / 400 / 1600 | 4 / 4 / 4 | 2 / 1 / 0 | 6 / 3 / 0 |
| wn18rr | dense | 100 / 400 / 1600 | 4 / 4 / 4 | 0 / 2 / 1 | 0 / 2 / 2 |
| codex-m | uniform | 100 / 400 / 1600 | 4 / 4 / 4 | 2 / 2 / 2 | 6 / 6 / 6 |
| codex-m | dense | 100 / 400 / 1600 | 4 / 4 / 4 | 0 / 0 / 1 | 2 / 1 / 2 |
| yago3-10 | uniform | 100 / 400 / 1600 | 4 / 4 / 4 | 2 / 2 / 0 | 6 / 6 / 0 |
| yago3-10 | dense | 100 / 400 / 1600 | 4 / 4 / 4 | 0 / 2 / 0 | 3 / 2 / 2 |
| umls | uniform | 100 / 400 / 1600 | 8 / 8 / 8 | 23 / 0 / 0 | 30 / 0 / 0 |
| umls | dense | 100 / 400 / 1600 | 8 / 8 / 8 | 1 / 0 / 0 | 1 / 0 / 0 |
| kinship | uniform | 100 / 400 / 1600 | 8 / 8 / 8 | 22 / 0 / 0 | 30 / 0 / 0 |
| kinship | dense | 100 / 400 / 1600 | 8 / 8 / 8 | 0 / 0 / 0 | 0 / 0 / 0 |

**Cluster per ipotesi (previsti):** H1/H2 ≤ 68; H1u/H2u ≤ 61 (60 di questi sono UMLS e
Kinship **uniform N = 100**: sui mai visti la promessa (a)/(b) è testabile solo lì);
H8u ≤ 144 sottografi dei mai visti; H3, H5, H6 sui 192 base (454 coppie per H5);
H4 nel perimetro target 0.85, N ≤ 400; H7 nel perimetro target 0.85, N = 100.

**Cosa ho visto prima (dichiarato):** le validated stanno su uniform N ≤ 400 dei grafi di
calibrazione e su uniform N = 100 dei mai visti; il denso, UMLS/Kinship oltre N = 100 e
quasi tutto N = 1600 sono `unvalidated`. La copertura prevista (≤ 0.337) è già sotto la
soglia di H5.

## Controlli sintetici (non entrano nelle ipotesi)

- **Positivi** (casi del revisore, come in `reference/tests/test_sizing.py`): club,
  inverse, three. Superati se MAP-B non è mai validated, le verificate «raggiungono» (vedi
  sotto) e, sul club, l'errore di MAP-B è > 3 punti.
- **Avversari** (6; `adversarial_params`, seme `[23_000_023, 777]`, estratti ora): hub
  Zipf (esponente 1.1–2.0), oggetti multipli per (s, r) (10–40 %), simmetriche (5–30 %),
  multi-relazione (30–60 %), 3–12 relazioni, 150–400 fatti; target standard più uno
  vicino al tetto (tetto − 0.03). Parametri estratti:

| | fatti | entità | relazioni | Zipf | multi-oggetto | simmetriche | multi-relazione |
|---|---|---|---|---|---|---|---|
| adv0 | 291 | 152 | 7 | 1.481 | 0.302 | 0.193 | 0.306 |
| adv1 | 186 | 242 | 4 | 1.618 | 0.269 | 0.122 | 0.369 |
| adv2 | 233 | 331 | 7 | 1.642 | 0.163 | 0.202 | 0.566 |
| adv3 | 286 | 319 | 3 | 1.484 | 0.315 | 0.092 | 0.588 |
| adv4 | 192 | 392 | 8 | 1.767 | 0.232 | 0.181 | 0.433 |
| adv5 | 187 | 145 | 4 | 1.334 | 0.375 | 0.249 | 0.500 |

  Superati se nessuna raccomandazione validated ha |errore| > 3 punti e le verificate
  «raggiungono».
- «Raggiungono» (positivi e avversari): fra le verificate, quota ≥ 0.8 con vero (tutte le
  chiavi) ≥ target − 0.01 **e** nessuna sotto target − 0.03. **Criterio corretto dopo il
  primo smoke** (solo sintetico): la versione «ogni verificata ≥ target − 0.01» falliva su
  inverse, adv2, adv5 per scarti di 1–2 punti (MAP-B 0.936 a t = 0.95; 0.832 a t = 0.85;
  0.680 a t = 0.70; con 6 memorie per il vero), cioè era più severa di H1 stessa.
- **Negativo**: 200 fatti uniformi (5000 entità, 20 relazioni): superato se tutte le
  raccomandazioni sono validated, |errore| ≤ 3 punti e vero al D del modello ≥ target − 0.02.
- **Modello rotto** (solo smoke): `sizing._predict_graph` + 0.05; le quote di H2 e la
  sensibilità di H3 devono stare sotto le soglie (asserito).

Un controllo fallito nel run non cambia le ipotesi, ma va riportato come difetto dello
strumento o dell'harness. 

**Smoke (passo 2, 3 sottografi sintetici uniform N = 100, 6 memorie per il vero, 1232 s):**
tutti i 10 controlli superati. Club: MAP-B ottimista fino a 31 punti; inverse 7.2; three
8.4; avversari |errore| massimo 2.1–3.9 punti, **tutte le raccomandazioni avversarie
risultano unvalidated** (il criterio "nessuna validated con errore > 3" è quindi vacuo su
questi sei: dichiarato); negativo: 18/18 validated, errore massimo 1.9 punti. Modello rotto
(+5 punti): quota H2 0.18 su 72 (falsificata), sensibilità H3 0.00 su 68 positivi. **Visto
prima dei dati, dichiarato:** sullo smoke sintetico H7 risulta falsificata (rapporto
mediano delle memorie 0.73, boot 0.51–0.73; bit 1.00), H4 sostenuta (mediana 0), H1 e H2
sostenute su 3 cluster; le soglie non sono state cambiate.

## Tempo e regola di arresto

`--timing` (memorie sintetiche con lo stesso N e M ≈ mediana reale; costruzione e query
per chiave misurate a parte; massimo di 3 ripetizioni; interpolazione in D; 6 processi;
modello delle memorie per raccomandazione nel docstring di `timing`). Budget per la scelta
del passo **10 h**; si sceglie il **primo** passo sotto il budget:

| passo | k_in | k_val | k_un | k_val_un | D minimo | baseline H7 | HRR max_mem | sottografi | stima |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 6 | 8 | 30 | tutti i target, N ≤ 1600 | 0.85, N ≤ 400 | 1 GiB | 266 | 75.1 h |
| 1 | 4 | 6 | 8 | 30 | 0.85, N ≤ 400 | 0.85, N ≤ 400 | 1 GiB | 266 | 15.3 h |
| **2** | **4** | **6** | **8** | **30** | **0.85, N ≤ 400** | **0.85, N ≤ 100** | **256 MiB** | **266** | **8.6 h** |
| 3 | 3 | 4 | 6 | 30 | 0.85, N ≤ 400 | 0.85, N ≤ 100 | 256 MiB | 222 | 7.5 h |
| 4 | 3 | 4 | 6 | 30 | 0.85, N ≤ 400 | 0.85, N ≤ 100 | 128 MiB | 222 | 5.9 h |
| 5 | 2 | 4 | 4 | 30 | 0.85, N ≤ 400 | 0.85, N ≤ 100 | 128 MiB | 186 | 4.4 h |
| 6 | 2 | 3 | 4 | 30 | 0.85, N = 100 | 0.85, N = 100 | 64 MiB | 178 | 4.0 h |

**Scelto il passo 2 (`GRID_STEP = 2`).** Una prima esecuzione di `--timing`, con la
costruzione pigra contata nelle query (bug), dava stime 2–3 volte più alte: è stata
scartata. **Regola di arresto:** il run esegue i sottografi nell'ordine di `select` e
scrive una riga JSONL per sottografo (`results/sizing_tool_prereg_results.jsonl`) in quell'ordine;
dopo un crash si riprende saltando i sottografi già scritti; il riepilogo si calcola solo
alla fine. Se si superano **12 h** il run si ferma e l'analisi usa il prefisso scritto
(sottografi interi, in ordine di priorità: ogni prefisso copre tutte le celle a giri).
Qualunque cambio di passo dopo l'avvio è una deviazione da dichiarare.
Comando: `python examples/sizing_tool_prereg.py`.

## Limiti dichiarati prima

1. Bootstrap per cluster con 60–70 cluster: gli IC sono approssimati; i cluster extra
   sono scelti in base alle previsioni (validated), quindi H1/H2 parlano dei sottografi
   che lo strumento dichiara validati, non di un campione casuale.
2. Il vero è una misura (20 memorie, errore ≈ 0.5 punti sulla media); errori del modello
   vicini a 3 punti sono classificati con rumore (H3, H6).
3. Sui grafi mai visti la promessa (a)/(b) è testabile solo su uniform N = 100; il resto
   dei mai visti entra solo in H8u, H3, H5, H6.
4. H4 solo a target 0.85 e N ≤ 400, H7 solo a target 0.85 e N = 100; il D minimo assume
   monotonia.
5. HRR oltre 256 MiB non misurato (82 raccomandazioni, quasi tutte a N = 1600).
6. La verifica di `recommend` non riduce mai D: H4 contiene anche il pessimismo del
   modello.
7. `reliability` è una regola strutturale: H3/H6 dicono se anticipa gli errori, non perché.
8. La stima dei tempi è rumorosa; se il run supera 12 h vale la regola di arresto.

## Esito (2026-10-14): non testabile per tempo

Eseguito dopo il push e le marche OpenTimestamps (`45240c1`), con
`python examples/sizing_tool_prereg.py` (passo 2, stimato 8.6 h). Dopo **12 h 10 min**
il checkpoint conteneva **5 sottografi su 266** (tutti FB15k-237: uniform 100/400/1600,
dense 100/400). Il tetto rigido di 12 h era superato, ma il codice lo controlla solo alla
scrittura di un sottografo e quello in corso (FB15k-237 dense 1600) non finiva: il run è
stato fermato a mano e il riepilogo calcolato con `summarize` dell'harness sul prefisso,
come prescrive la regola d'arresto. Risultati in `results/sizing_tool_prereg_results.json`
(prefisso) e `results/sizing_tool_prereg_results.jsonl` (checkpoint).

| ipotesi | cluster | esito |
|---|---|---|
| H1, H2 (replica in-distribuzione) | ≤ 5 (minimo 30) | **non testabile** |
| H1u, H2u, H8u (grafi mai visti) | 0 | **non testabile** |
| H3, H5, H6 | 5 | **non testabile** |
| H4 | 4 | **non testabile** |
| H7 | 2 | **non testabile** |
| controlli sintetici | — | non raggiunti (in coda all'ordine) |

**Descrittivo, non un test** (90 raccomandazioni, 5 sottografi FB15k-237): errore medio del
modello 0.41 punti; la verifica corregge D nel 39 % dei casi (MAP-B 73 %, FHRR 67 %);
verificate 90 %.

### Perché, e cosa cambia

- **La stima dei tempi era sbagliata di circa 50 volte.** `--timing` misurava memorie su
  grafi sintetici; la verifica con 8 semi di selezione, 8 di report, 20 memorie fresche per il
  vero su tutte le chiavi e la ricerca del D minimo su altre 20, ripetuta per 6 famiglie × 3
  target, su sottografi reali densi, costa molto di più. È un errore del disegno, non dei dati.
- **Il tetto rigido non era rigido**: andava controllato anche durante un sottografo.
- Nessuna ipotesi è valutata; nessuna soglia è stata toccata. Un nuovo collaudo richiede una
  nuova preregistrazione, con i tempi misurati su sottografi reali (solo tempo, senza guardare
  l'accuratezza) e un tetto controllato per cella.
