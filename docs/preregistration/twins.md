# Preregistrazione — gemelli simmetrici e Law VII esatta

**Data: 2026-09-28.** Committato insieme a
[`examples/twins_prereg.py`](../../examples/twins_prereg.py) **prima** di
eseguire, a parte uno smoke test che stampa solo conteggi. Stesse regole di
[`fb15k237.md`](fb15k237.md).

## Da dove viene l'ipotesi, dichiarato per intero

La preregistrazione 2 ([`exact_contract.md`](exact_contract.md)) ha trovato il
modello esatto **pessimista** sui sottografi densi di FB15k-237 a D = 2048
(bias −2.61, fino a −8.4 punti a N = 400). Dopo averlo visto ho cercato la causa,
in modo **esplorativo**, sugli stessi dati:

1. nelle tracce dense a N = 400 i fatti concordano con la traccia più del
   previsto (0.5216 contro 0.5199), nei campioni uniformi no;
2. per l'encoding s ⊕ ρ(r) ⊕ o, **(s, r, o) e (o, r, s) sono lo stesso vettore**: una
   relazione simmetrica nelle due direzioni è un solo fatto di peso 2 (Law VII).
   Nel 12.5% delle triple di FB15k-237 esiste il gemello; nei campioni densi il
   6–14% delle triple lo ha nel campione, in quelli uniformi lo 0%;
3. aggiungendo i pesi al modello esatto (`p_agree_weighted`,
   `cleanup_accuracy_mixed`), l'errore sulle **stesse** celle scende da 2.88 a 1.03
   a D = 2048, e il caso peggiore da −8.4 a −0.7.

Il punto 3 è un adattamento a posteriori su dati già visti, **non una prova**.
Questa preregistrazione lo mette alla prova su dati che non ho visto.

**Non ho visto:** FB15k-237 con i seed 10–19; nessuna misura su WN18RR (ne ho
guardato solo la struttura: 86 835 triple, 40 559 entità, 11 relazioni, il 34.2%
con gemello simmetrico, soprattutto `_derivationally_related_form`); nessuna
misura dei pesi sintetici della parte W.

## Parti

Stessa griglia della preregistrazione 2: D = 2048 con N ∈ {50 … 400}, D = 8192
con N ∈ {200 … 1600}; al più 200 query per cella e seed; reference congelata.

- **T1** — FB15k-237, sottografi densi, seed **10–19**.
- **T2** — WN18RR, sottografi densi e campioni uniformi, seed 0–9.
- **W** — pesi sintetici, D = 2048, 20 seed: sei configurazioni, dall'uniforme a
  2 fatti di peso 14, con accuratezza misurata separatamente su singoli e pesanti.
  Sostituisce `conjecture7_results.json`, il cui script non è mai stato committato.

## Predittori

- **primario**: modello esatto con i pesi dei gemelli, `cleanup_accuracy_mixed`
  con `p_agree_weighted` per ogni candidato;
- confronto: modello esatto senza gemelli (quello della preregistrazione 2) e
  Law IV con k = 0.92;
- in W, anche il modello esatto con N_eff = Σw² al posto di N (la forma della
  Law VII nel paper).

## Criteri, fissati ora

Errori come previsto − misurato, in punti, sulle celle mediate sui seed.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — T1, per ciascuna D | errore medio assoluto ≤ 2 **e** con segno entro ±1.5 | > 5, **o** con segno oltre ±3 |
| **H2** — T2 denso, per ciascuna D | ≤ 3 **e** con segno entro ±2 | > 6, **o** con segno oltre ±4 |
| **H3** — T2 denso: il meccanismo | il modello *senza* gemelli ha bias con segno < −2 su entrambe le D, **e** quello con gemelli ha errore assoluto minore | il modello senza gemelli non è pessimista (bias con segno ≥ 0) su una qualsiasi D |
| **H4** — T2 uniforme, controllo | quota di gemelli < 1%, e modello con gemelli ≤ 3 per ciascuna D | > 6 |
| **H5** — W, Law VII esatta | errore medio assoluto ≤ 2 su singoli **e** pesanti, per ogni configurazione | > 5 in una qualsiasi |

Tutto il resto è **sostenuto in parte**, e va scritto così.

## Limiti dichiarati prima

- Il modello tratta come indipendenti le distanze dei candidati a pari segnale,
  e le distanze nulle: la parte B della preregistrazione 2 stima questo costo
  sotto il punto.
- Un peso w > 2 può nascere solo da duplicati veri; nei due grafi non ce ne
  sono, quindi i pesi reali sono 1 o 2.

---

## Esito — 2026-09-28, eseguito dopo il commit `aebd0dc`

Previsioni e criteri **non modificati**. Risultati:
[`results/twins_prereg_results.json`](../../results/twins_prereg_results.json).
Errori come previsto − misurato, in punti.

| ipotesi | con gemelli (primario) | senza gemelli | Law IV k = 0.92 | quota gemelli | esito |
|---|---|---|---|---|---|
| H1 — FB15k-237 denso, D = 2048 | 1.09, segno −0.81 | 1.33, −1.29 | 2.24 | 5.4% | **sostenuta** |
| H1 — FB15k-237 denso, D = 8192 | 0.68, −0.23 | 0.71, −0.65 | 1.93 | 5.3% | **sostenuta** |
| H2 — WN18RR denso, D = 2048 | 0.57, −0.16 | 2.94, −2.13 | 4.20 | 23.1% | **sostenuta** |
| H2 — WN18RR denso, D = 8192 | 0.60, −0.36 | 5.12, −3.75 | 7.01 | 25.7% | **sostenuta** |
| H3 — il meccanismo | errore minore in entrambe | bias < −2 in entrambe | — | — | **sostenuta** |
| H4 — WN18RR uniforme, D = 2048 | 0.91 | 0.93 | 0.91 | 0.1% | **sostenuta** |
| H4 — WN18RR uniforme, D = 8192 | 1.11 | 1.01 | 0.80 | 0.3% | **sostenuta** |

**H5 — sostenuta.** Law VII esatta, 20 seed per configurazione:

| configurazione | singoli: misurato / esatto / N_eff | pesanti: misurato / esatto |
|---|---|---|
| uniforme | 53.6 / 52.2 / 52.2 | — |
| 10 × w = 3 | 34.6 / 34.0 / 34.3 | 100.0 / 100.0 |
| 10 × w = 5 | 18.7 / 19.0 / 19.6 | 100.0 / 100.0 |
| 5 × w = 8 | 15.0 / 14.8 / 16.0 | 100.0 / 100.0 |
| 20 × w = 4 | 15.2 / 15.8 / 16.1 | 99.8 / 100.0 |
| 2 × w = 14 | 15.2 / 14.8 / **12.5** | 100.0 / 100.0 |

### Cosa dice

- I **gemelli simmetrici** spiegano il pessimismo trovato nella preregistrazione 2,
  su dati che non avevo visto e su un secondo grafo: su WN18RR denso, dove un
  quarto delle triple è gemella, ignorarli costa fino a 8.9 punti; contarli come
  fatti di peso 2 riporta l'errore sotto il punto.
- Il controllo uniforme conferma che il termine non migliora tutto a caso: dove
  i gemelli non ci sono, le due previsioni coincidono.
- La **Law VII esatta** sostituisce lo script perduto di `conjecture7_results.json`
  e risolve il problema aperto 4 del paper (saturazione a pesi estremi): con due
  fatti di peso 14, la forma N_eff = Σw² sbaglia di 2.7 punti, quella esatta di 0.5.
- Nel campione denso FB15k-237 coi seed nuovi i gemelli sono il 5%, meno che coi
  seed 0–9: lì il termine incide poco, ed è coerente.

## Nota del 2026-10-01: self-loop (esplorativa, non preregistrata)

Il 2026-09-30 (commit `2a41f74`) `abm.exact` ha iniziato a modellare i self-loop (s, r, s), che in FB15k-237 sono 1625 (l'audit diceva 0) e in WN18RR 7. Questa preregistrazione è stata calcolata prima. I numeri sopra restano quelli pubblicati: `examples/replicate.py` li riproduce sul commit che li ha registrati. L'harness congelato non gira più sul modello attuale: legge i pesi con
la chiave vecchia e va in crash su un self-loop.

`examples/selfloop_impact.py` rigenera gli stessi campioni e le stesse query, e
ricalcola la previsione con il modello attuale (controllo: sulle celle senza
self-loop coincide con la pubblicata entro 1e-16):

- FB15k-237: 18 celle su 120 contengono self-loop; bias medio −0.52 → −0.52
  punti, errore assoluto medio 2.29 → 2.29; spostamento massimo 0.45 punti.
- WN18RR: 6 celle su 240; bias −0.02 → −0.11, errore assoluto 2.06 → 1.98.
- **Una cella era sbagliata di 17.6 punti senza che lo vedessimo**: WN18RR
  uniforme, D = 8192, N = 200, seed 7 — previsti 97.6%, misurati 80.0%. Il
  modello corretto prevede 80.2%.

Le medie si spostano meno di 0.1 punti; le ipotesi H1–H5 **non** sono state
rivalutate con il modello attuale.
