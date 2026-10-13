# Preregistrazione 23 — il dimensionatore mantiene la promessa su grafi mai visti?

**Data: 2026-10-13.** Committata **prima** di misurare, insieme all'harness
[`examples/sizing_tool_prereg.py`](../../examples/sizing_tool_prereg.py) e alle previsioni
congelate `results/sizing_tool_prereg_predictions.json`. Prima del commit l'harness è
stato eseguito solo con:

- `--smoke`: impianto intero (sottografi, `recommend(verify=True)`, memorie fresche, D
  minimo, statistiche, controlli) su un grafo **sintetico** generato nello script e sui
  tre controlli sintetici;
- `--predict`: legge i cinque grafi, campiona i sottografi e le query, chiama
  `recommend(..., verify=False)` (solo il modello: **nessuna memoria costruita** sui dati
  reali) e salva D, previsione e affidabilità per ogni raccomandazione;
- `--timing`: tempo di costruzione+query di memorie su grafi **sintetici** con gli N e i
  D previsti; `--power`: potenza dalle previsioni.

Nessuna memoria è stata costruita sui sottografi di valutazione. Lo strumento collaudato è
quello del commit `ae2c99e` (`abm.sizing`, `abm.families`); l'harness usa solo l'API
pubblica (`recommend`, `predict`, `STEP`, `FAMILIES`; `families.make`, `estimate_bytes`,
`bsdc_length`) più `sizing._Graph` per statistiche **solo descrittive** (regime, quote),
mai usate per decidere. Push e marche OpenTimestamps li fa il coordinatore dopo il commit.

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

| grafo | file | sha256 | fonte |
|---|---|---|---|
| FB15k-237 | `fb15k237_train.txt` | `6e4c2782…cdfae` | come la 21 |
| WN18RR | `wn18rr_train.csv` | `28f7a0b3…e6a0` | come la 21 |
| CoDEx-M | `codex/codex-m_train.txt` | `d99c3437…6556` | come la 22 |
| YAGO3-10 | `yago3_10/YAGO3-10.tar.gz` (train.txt) | `9a3bdd5f…fb92` | come la 22 |
| **UMLS** (nuovo) | `umls/umls.tar.gz` (train.txt) | `26223aa91f0ec8db2324e0f489b9a4425e04d1d0df72ce5cfd62af4b91aee6fd` | repo pubblico `TimDettmers/ConvE` al commit `f3c0eb286025410fa5b4c04696c918264163d0ca`; voce aggiunta in `examples/replicate.py` |

UMLS (135 entità, 46 relazioni, 5216 fatti di train) è piccolo e denso: molte coppie
{s, o} con più relazioni. È scelto apposta: è fuori dal regime di calibrazione e mette alla
prova le regole di affidabilità (c). Kinship e Nations (stesso repo) sono stati scartati
perché più piccoli (Nations: 1592 fatti, niente N = 1600).

**Sottografi nuovi.** `SeedSequence([23_000_023, indice grafo, indice schema, N, k])` con
**k = 50 000, 50 001, …** (mai usati: le 21-22 usano altri SEED_BASE e k = 0–19).
Campionamento uniform (a caso) e dense (BFS), copiato senza modifiche da
`examples/vsa_families_prereg.py`; Q = 200 query (s, r) a caso per sottografo (o tutte se
sono meno). Limite: per FB15k-237, WN18RR, CoDEx-M e YAGO3-10 i grafi sono gli stessi
della calibrazione, solo i sottografi sono nuovi (un sottografo dense può sovrapporsi in
parte a uno usato); **solo UMLS è un grafo mai visto**.

## Impianto

- Griglia: 5 grafi × {uniform, dense} × N ∈ {100, 400, 1600} × K sottografi × target
  ∈ {0.70, 0.85, 0.95} (accuratezza assoluta, la semantica di `recommend`) × le sei
  famiglie. N = 1600 si salta se il grafo ha meno di 2N fatti (non accade).
- **Regola del tetto (scritta):** un (sottografo, target) è scartato se
  target > tetto degli alias − 0.02 (`exact.ceiling` sulle stesse query). Vicino al tetto
  D esplode e "raggiungere il target" non dice nulla sul dimensionatore.
- Per ogni (sottografo, target): `recommend(triples, target, queries=Q, verify=True,
  seeds=8, tol=0.01, rng_seed=…, max_mem_bytes=…)` (HRR in una chiamata a parte con il
  suo limite di memoria, vedi «Tempo»; l'ordinamento non è usato).
- **Il "vero"**: accuratezza media su **20 memorie fresche** `families.make(f, D, seme)`
  con semi ≥ 2^31, disgiunti per costruzione dai semi della verifica (che `recommend`
  estrae in [0, 2^31)), sulle **stesse query Q**; misurato al D finale (`dim`) e al D del
  modello (`dim_predicted`; coincidono se la verifica non ha corretto). Una risposta è
  corretta se è uno degli oggetti veri di (s, r).
- **D minimo misurato** (H4): il più piccolo D sulla griglia della famiglia (passo 64;
  BSDC: L = `bsdc_length(N)`) con media delle **stesse 20 memorie fresche** ≥ target.
  Ricerca: dal D finale, raddoppio se non basta (fino a 8×), poi dimezzamenti fino al primo
  D che non basta, poi bisezione al passo. Assume monotonia; il rumore della media (20
  memorie) entra nella stima (limite dichiarato).
- Errore del modello: `predict(triples, f, dim_predicted, Q)` − vero al `dim_predicted`.

## Ipotesi e criteri (fissati ora)

Unità = una raccomandazione (sottografo × target × famiglia) sui grafi reali. I controlli
sintetici non entrano. Intervalli di Clopper-Pearson **unilaterali al 95 %**
(CP_inf, CP_sup). Con meno unità del minimo indicato un'ipotesi è **non testabile**.
Fra "sostenuta" e "falsificata" l'esito è **non conclusivo** (riportato così).

**H1 — promessa (a).** Unità: raccomandazioni `validated` **e** `verified` con il vero
misurato. Successo: vero al D finale ≥ target − tol (tol = 0.01).
*Sostenuta* se p̂ ≥ 0.95 **e** CP_inf ≥ 0.90; *falsificata* se CP_sup < 0.95. Minimo 30.
Motivazione: la verifica accetta solo se l'IC_inf della selezione ≥ target − tol, che
per costruzione rifiuta un ottimismo ≥ tol con probabilità ≥ 0.975; ci si aspetta
p ≈ 0.97–0.99 (il vero ha a sua volta un errore di ≈ 0.5 punti). 0.95 è la promessa che
si vuole poter scrivere; il pavimento 0.90 evita di sostenerla per caso con n piccolo.
Descrittiva: la stessa quota su tutte le verificate (qualunque affidabilità).

**H2 — promessa (b).** Unità: raccomandazioni `validated` (D del modello, senza
verifica). Successo: vero al `dim_predicted` ≥ target − 0.02. *Sostenuta* se p̂ ≥ 0.90
**e** CP_inf ≥ 0.85; *falsificata* se CP_sup < 0.90. Minimo 30. Motivazione: il modello è
centrato (circa metà delle celle sotto il target), con MAE osservato ≤ 0.84 punti per le
famiglie validate (OBSERVED_ERROR) e coda fino a ~3–4 punti: una banda di 2 punti
dovrebbe coprire ≥ 90 %.

**H3 — promessa (c), sensibilità.** Unità: tutte le raccomandazioni con il vero al
`dim_predicted`. Positivo: |previsione − vero| > 3 punti. Segnalato: affidabilità ≠
`validated`. *Sostenuta* se sensibilità ≥ 0.80 **e** CP_inf ≥ 0.60; *falsificata* se
CP_sup < 0.80. Minimo 10 positivi (sotto: non testabile; se le raccomandazioni validated
non sbagliano mai di 3 punti, non c'è nulla da segnalare). Descrittiva: la stessa sui soli
errori ottimisti (previsione > vero).
**H3b — specificità** (che gli avvisi non siano sempre accesi). Negativi: |errore| ≤ 3
punti. *Sostenuta* se quota `validated` fra i negativi ≥ 0.50 **e** CP_inf ≥ 0.40;
*falsificata* se CP_sup < 0.50. Minimo 30. **Avvertenza dichiarata:** dalle previsioni
(sotto) il 64 % delle raccomandazioni è già `unvalidated` per struttura, quindi H3b rischia
di fallire anche con un modello accurato: è esattamente il costo che va misurato.

**H4 — promessa (d).** Unità: raccomandazioni `verified` (qualunque affidabilità) con D
minimo misurato (target 0.85, N ≤ 400: vedi «Tempo»). Gonfiamento = D finale / D minimo − 1.
*Sostenuta* se la mediana ≤ 0.15 **e** il limite superiore dell'IC 95 % della mediana
(distribution-free, statistiche d'ordine) ≤ 0.20; *falsificata* se il limite inferiore
> 0.15. Minimo 30. Motivazione: la verifica non riduce mai D (eredita il pessimismo del
modello, es. BSDC-S) e chiede IC_inf ≥ target − tol con 8 semi (≈ +0.7 punti di
margine); vicino a 0.85 un punto di accuratezza costa ≈ 5–10 % di D, un blocco BSDC
5–10 %. Descrittivo: lo stesso rapporto per il D del modello.

## Potenza (dalle previsioni, griglia scelta: K = 2, 926 raccomandazioni)

Unità supposte indipendenti (vedi limiti). n per H1 ignoto prima dei dati: ≤ 318 (le
`validated` previste); si riporta n = 318 e n = 159 (metà verificate).

P(sostenuta) / P(falsificata), H1:

| p vero | 0.99 | 0.98 | 0.97 | 0.95 | 0.92 | 0.90 | 0.85 |
|---|---|---|---|---|---|---|---|
| n = 318 | 1.00 / 0 | 1.00 / 0 | 0.97 / 0 | 0.47 / 0.03 | 0.02 / 0.65 | 0 / 0.94 | 0 / 1.00 |
| n = 159 | 1.00 / 0 | 0.99 / 0 | 0.89 / 0 | 0.46 / 0.03 | 0.06 / 0.40 | 0.01 / 0.73 | 0 / 0.99 |

H2 (n = 318): p = 0.95 → 1.00 / 0; 0.92 → 0.89 / 0.001; 0.90 → 0.49 / 0.04;
0.85 → 0.004 / 0.84. (n = 159: 0.99 / 0, 0.80 / 0.004, 0.47 / 0.05, 0.03 / 0.61.)

H3 (n positivi ignoto): sensibilità vera 0.95 → P(sost.) 0.91 (n = 10), 0.98 (20), 1.00
(40); 0.90 → 0.74 / 0.87 / 0.99; 0.70 → P(falsificata) 0.15 / 0.23 / 0.42; 0.50 →
P(falsificata) 0.62 / 0.87 / 0.99. Con meno di 20 positivi H3 ha poca potenza contro
sensibilità ≈ 0.7: dichiarato.

H3b (n negativi 463–926): specificità 0.6 → P(sost.) 1.00; 0.4 → P(falsificata) ≥ 0.996.

H4: con n ≥ 30 l'IC della mediana è fra la 10ª e la 21ª statistica d'ordine; potenza non
calcolata in forma chiusa (la distribuzione del gonfiamento è ignota); candidati previsti
234 (target 0.85, N ≤ 400), di cui le verificate sono un sottoinsieme.

## Previsioni dello strumento (congelate, `--predict`)

`results/sizing_tool_prereg_predictions.json`: per ogni (sottografo, target, famiglia) D,
bit, previsione, affidabilità, numero di avvisi, statistiche strutturali; scarti per il
tetto; potenza (`power_grid` per la griglia scelta). Griglia scelta (K = 2, HRR ≤ 128 MiB):

| grafo | schema | N | target scartati (×sottografi) | racc. | validated | caution | unvalidated |
|---|---|---|---|---|---|---|---|
| fb15k237 | uniform | 100 | — | 36 | 18 | 0 | 18 |
| fb15k237 | uniform | 400 | — | 36 | 18 | 0 | 18 |
| fb15k237 | uniform | 1600 | — | 30 | 0 | 0 | 30 |
| fb15k237 | dense | 100 | — | 36 | 0 | 0 | 36 |
| fb15k237 | dense | 400 | — | 36 | 0 | 0 | 36 |
| fb15k237 | dense | 1600 | — | 31 | 0 | 0 | 31 |
| wn18rr | uniform | 100 | — | 36 | 36 | 0 | 0 |
| wn18rr | uniform | 400 | — | 36 | 18 | 0 | 18 |
| wn18rr | uniform | 1600 | — | 30 | 0 | 0 | 30 |
| wn18rr | dense | 100 | — | 36 | 0 | 0 | 36 |
| wn18rr | dense | 400 | 0.95×2 | 24 | 0 | 0 | 24 |
| wn18rr | dense | 1600 | 0.95×1 | 25 | 0 | 0 | 25 |
| codex-m | uniform | 100 | — | 36 | 36 | 0 | 0 |
| codex-m | uniform | 400 | — | 36 | 36 | 0 | 0 |
| codex-m | uniform | 1600 | — | 30 | 27 | 3 | 0 |
| codex-m | dense | 100 | — | 36 | 0 | 0 | 36 |
| codex-m | dense | 400 | — | 36 | 0 | 0 | 36 |
| codex-m | dense | 1600 | — | 30 | 0 | 0 | 30 |
| yago3-10 | uniform | 100 | — | 36 | 36 | 0 | 0 |
| yago3-10 | uniform | 400 | — | 36 | 36 | 0 | 0 |
| yago3-10 | uniform | 1600 | — | 30 | 0 | 0 | 30 |
| yago3-10 | dense | 100 | 0.95×1 | 30 | 25 | 5 | 0 |
| yago3-10 | dense | 400 | 0.95×1 | 30 | 0 | 0 | 30 |
| yago3-10 | dense | 1600 | — | 30 | 12 | 3 | 15 |
| umls | uniform | 100 | 0.95×2 | 24 | 20 | 4 | 0 |
| umls | uniform | 400 | 0.95×2 | 24 | 0 | 0 | 24 |
| umls | uniform | 1600 | 0.85×1, 0.95×2 | 18 | 0 | 0 | 18 |
| umls | dense | 100 | 0.85×1, 0.95×1 | 24 | 0 | 0 | 24 |
| umls | dense | 400 | 0.95×2 | 24 | 0 | 0 | 24 |
| umls | dense | 1600 | 0.95×2 | 24 | 0 | 0 | 24 |

Totale: 926 raccomandazioni (318 validated, 15 caution, 593 unvalidated), 60 sottografi,
18 (sottografo, target) scartati per il tetto; nessuna famiglia irraggiungibile entro
d_max. Sulla griglia completa (K = 4, prima della riduzione): 1944 raccomandazioni (719 /
43 / 1182). Le 46 raccomandazioni HRR escluse dal limite di memoria sono tutte a N = 1600
(9 validated).

**Cosa ho visto prima (dichiarato):** le `validated` sono concentrate su uniform N ≤ 400 e
su CoDEx-M/YAGO3-10 (i grafi della 22); quasi tutto il denso e tutto UMLS oltre N = 100 è
`unvalidated`. Quindi H1-H2 parlano soprattutto del regime di calibrazione, e il test
fuori regime è H3/H3b (più le misure descrittive su `unvalidated`+verificate).

## Controlli positivi sintetici (non entrano nelle ipotesi)

Gli stessi tre casi dei test (`reference/tests/test_sizing.py`), generati nello script:
**club** (100 giocatori, due relazioni verso lo stesso club, 2 club), **inverse** (100
fatti uniformi + 50 coppie (p, a, q) + (q, b, p)), **three** (80 fatti uniformi + 40
giocatori con tre relazioni verso 3 club); tutte le query; stessi target e stesso impianto.
Il controllo è **superato** se: (i) MAP-B non è mai `validated`; (ii) sul club l'errore
del modello di MAP-B è > 3 punti (ottimista); (iii) ogni raccomandazione verificata ha
vero ≥ target − tol. Nello smoke (6 memorie fresche, non 20) i tre controlli passano:
club MAP-B ottimista di 25–34 punti, la verifica porta D da 4224 a 8576 (t = 0.85) e il
vero diventa 0.872. Un controllo fallito nel run non cambia le ipotesi ma va riportato come
difetto dello strumento o dell'harness.

## Tempo e riduzione della griglia (regola scritta, applicata ora)

Stima di `--timing` (memorie solo sintetiche, stesso N, M ≈ mediana dei sottografi reali,
interpolazione in D; memorie attese per raccomandazione: 16 di verifica + 16 di
correzione e 20 al D finale se non `validated` + 20 al D del modello + D minimo:
(2 + log2(D / 2·passo)) valutazioni da 20 memorie; 6 processi). Budget 3 h. Passi in
ordine, ci si ferma al primo sotto il budget:

| passo | K | D minimo su | HRR max_mem_bytes | stima |
|---|---|---|---|---|
| 0 | 4 | tutti i target, tutti gli N | 1 GiB | 30.8 h |
| 1 | 4 | target 0.85 | 1 GiB | 16.6 h |
| 2 | 4 | target 0.85, N ≤ 400 | 1 GiB | 9.7 h |
| 3 | 3 | target 0.85, N ≤ 400 | 1 GiB | 7.7 h |
| 4 | 2 | target 0.85, N ≤ 400 | 1 GiB | 5.2 h |
| 5 | 2 | target 0.85, N ≤ 400 | 256 MiB | 3.5 h |
| **6** | **2** | **target 0.85, N ≤ 400** | **128 MiB** | **2.98 h** |
| 7 | 2 | target 0.85, N ≤ 400 | 64 MiB | 2.9 h |

**Scelto il passo 6** (`GRID_STEP = 6`). Il costo è dominato da HRR a N = 1600 (fino a
48 s per memoria a D ≈ 97 000). Con il limite di 128 MiB, una raccomandazione HRR il cui
D (del modello o di una correzione) supera il limite non viene costruita: nella verifica
`recommend` la segna non verificata con un avviso; nel vero è "non misurata" ed è esclusa
da tutte le ipotesi (contata e riportata). Le stime di tempo sono rumorose (due
esecuzioni di `--timing`: 21.5 h e 30.8 h al passo 0); se il run supera le 3 h si fa così.
Comando: `python examples/sizing_tool_prereg.py` (scrive
`results/sizing_tool_prereg_results.json`).

## Limiti dichiarati prima

1. **Unità non indipendenti.** Le sei famiglie e i tre target dello stesso sottografo
   condividono triple e query; le potenze e gli IC assumono indipendenza e sono ottimisti.
   Descrittivo (non decisionale): le stesse quote per sottografo.
2. **Il vero è una misura**: 20 memorie, errore ≈ 0.5 punti sulla media; le soglie
   (target − tol, 3 punti) sono confrontate con una stima, non con il valore atteso.
   Errori del modello vicini a 3 punti sono classificati con rumore (H3).
3. Il vero è sulle stesse 200 query della verifica: misura la generalizzazione ai **semi**
   (memorie nuove), non a query nuove dello stesso grafo.
4. Solo UMLS è un grafo nuovo; gli altri quattro sono i grafi di calibrazione con
   sottografi nuovi. Le `validated` stanno quasi tutte nel regime di calibrazione.
5. H4 solo a target 0.85 e N ≤ 400 (riduzione per il tempo); il D minimo assume
   monotonia ed è stimato con le stesse memorie del vero.
6. HRR a N = 1600 è in parte non misurato (limite di memoria scelto per il tempo).
7. La verifica di `recommend` non riduce mai D: H4 misura anche il pessimismo del modello,
   non solo il costo della verifica.
8. `reliability` è una regola strutturale: H3/H3b non dicono se gli avvisi spiegano
   *perché* il modello sbaglia, solo se lo anticipano.
