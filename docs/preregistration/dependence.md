# Preregistrazione — la dipendenza esatta fra hop, e la proiezione tipizzata

**Data: 2026-09-28.** Committato insieme a
[`examples/dependence_prereg.py`](../../examples/dependence_prereg.py) **prima**
di eseguire, a parte uno smoke test che stampa solo conteggi. Stesse regole di
[`fb15k237.md`](fb15k237.md).

## Perché

La **Law V** (Acc(h) = p^h) assume che i successi di hop diversi sulla stessa
traccia siano indipendenti. Il paper lo misura (φ = +0.014 ± 0.024) ma non lo
dimostra (§9, problema 3). Per ogni bit, con T la traccia e f₁, f₂ i fatti dei
due hop, (T·f₁)(T·f₂) = f₁·f₂ perché T² = 1; ne segue che l'indipendenza è
**falsa**, e di quanto esattamente:

- per bit, la correlazione fra l'accordo della query con f₁ e con f₂ è
  **−ρ²/(1 − ρ²)**, con ρ = 2·p_agree(N) − 1, e non 0;
- sugli eventi, `two_hop_joint` (in
  [`exact_contract.py`](../../bsm/memory/exact_contract.py)) dà la probabilità
  esatta che entrambi gli hop riescano, trattando come indipendenti solo le
  distanze *nulle* dei due hop.

Il vecchio esperimento (D = 1024, N = 90) aveva un errore di ±0.033 sulla
correlazione per bit, contro un effetto previsto di −0.007: non poteva vederlo.
Qui la precisione è scelta per vederlo.

**P3** (guadagno del cleanup tipato) è nel paper con i valori 1.84 / 2.25 / 2.68×,
ma lo script che li ha prodotti **non è mai stato committato**
(`36d0881`, 2026-07-14, contiene solo il JSON). La parte G lo ricostruisce.

**Cosa ho visto prima di scrivere questo file:** solo previsioni del modello,
usate per scegliere D e il numero di coppie, e i vecchi valori del paper.
**Non ho visto** nessuna misura di questa preregistrazione.

## Parti, previsioni e criteri

Tutte le catene sono (a, r₁, b), (b, r₂, c); ogni hop è interrogato con la sua
chiave vera. Il codebook è M = 3N/2 + 2.

### E1 — correlazione per bit (D = 1024, 400 tracce per N)

| N | previsto | Law V |
|---|---|---|
| 10 | −0.0645 | 0 |
| 30 | −0.0213 | 0 |
| 90 | −0.0071 | 0 |

**Sostenuta** se, per ogni N, |misurato − previsto| ≤ 3 SE e |misurato| > 3 SE.
**Falsificata** se, per un qualsiasi N, |misurato − previsto| > 5 SE.

Onestà: dato che (T·f₁)(T·f₂) = f₁·f₂ è un'identità, E1 verifica soprattutto che
ρ misurato coincida con p_agree(N). È il passaggio che porta alla previsione di
E2, e il vecchio esperimento lo dichiarava "≈ 0".

### E2 — correlazione fra i successi (8000 tracce per configurazione)

| N | D | M | coppie | p previsto | P(entrambi) previsto | p² (Law V) | **φ previsto** |
|---|---|---|---|---|---|---|---|
| 10 | 48 | 17 | 40 000 | 0.476 | 0.2189 | 0.2268 | **−0.032** |
| 30 | 240 | 47 | 120 000 | 0.510 | 0.2573 | 0.2601 | **−0.011** |

**Sostenuta** se, per entrambe, |φ misurato − φ previsto| ≤ 3 SE (SE ≈ 1/√coppie),
**e** a N = 10 la Law V è respinta: |φ misurato| > 3 SE.
**Falsificata** se, per una qualsiasi, |φ misurato − φ previsto| > 5 SE.

### G — proiezione tipizzata, ricostruita (D = 1024, 30 seed)

Fatti (sᵢ, r_{i mod 13}, oᵢ); codebook pieno = 2N + 13 + distrattori; codebook
tipato = gli N oggetti. N\* = crossing del 50% su una griglia geometrica ×1.1 fra
0.6 e 1.6 volte la previsione, per interpolazione lineare.

| distrattori | N\* pieno previsto | N\* tipato previsto | **guadagno previsto** | nel paper: previsto Gumbel / misurato |
|---|---|---|---|---|
| 2 000 | 55.6 | 103.3 | **1.86×** | 1.94 / 1.84 |
| 8 000 | 45.6 | 103.3 | **2.26×** | 2.37 / 2.25 |
| 32 000 | 37.5 | 103.3 | **2.75×** | 2.82 / 2.68 |

**Sostenuta** se ogni guadagno misurato è entro il 7% del previsto, e ogni N\*
misurato entro il 7%. **Falsificata** se un guadagno o un N\* si discosta di oltre
il 15%. Il 7% tiene conto del rumore di un crossing stimato con 30 seed.

## Limiti dichiarati prima

- `two_hop_joint` tratta come indipendenti le distanze nulle dei due hop: la
  loro correlazione è di ordine 1/√D, con segno casuale.
- In G i distrattori sono aggiunti al codebook dopo i fatti; conta solo per i
  pareggi. L'harness verifica l'equivalenza con la reference sulle prime query.
- La costruzione di G (13 relazioni a rotazione) non è necessariamente quella,
  perduta, dell'esperimento originale: i valori assoluti di N\* possono differire
  da quelli del paper; il confronto sostanziale è sul guadagno.
