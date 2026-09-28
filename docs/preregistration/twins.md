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
