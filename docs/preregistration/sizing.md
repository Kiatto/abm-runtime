# Preregistrazione — il contratto usato per scegliere D prima

**Data: 2026-09-28.** Committato insieme a
[`examples/sizing_prereg.py`](../../examples/sizing_prereg.py) **prima** di
misurare, a parte uno smoke test con N = 30 che stampa solo il numero di query e la
dimensione scelta. Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

Le preregistrazioni precedenti confrontano previsioni e misure. La tesi del paper
però è pratica: *l'accuratezza si può dichiarare prima del deployment*. Questo
test usa il modello come lo userebbe qualcuno: per un sottografo nuovo, scegliere
**la dimensione minima** che promette un'accuratezza obiettivo, e poi verificare
che la promessa regga.

**Un risultato emerso preparandolo, dichiarato qui:** con l'encoding simmetrico
un alias a pari segnale limita l'accuratezza di una query a g/(g + a), qualunque
sia D. Su 5 sottografi densi di WN18RR con N = 300 il tetto medio è 0.934 (l'11.4%
delle query ha un alias), su FB15k-237 0.989. Un contratto deve dichiararlo; per
un obiettivo sopra il tetto la dimensione giusta non esiste. **Ho visto:** solo
questi tetti, calcolati dalla struttura, e le previsioni di un sottografo usato
per stimare i tempi. **Non ho visto** nessuna misura.

## Disegno

- 20 sottografi densi **nuovi** (seed da 5000) per ciascun grafo (FB15k-237,
  WN18RR) e ciascun N ∈ {150, 300}: 80 sottografi.
- Obiettivi T ∈ {0.70, 0.80}, sotto i tetti.
- Per ogni sottografo e obiettivo, due dimensionamenti:
  - **esatto**: la D minima (multiplo di 64, al più 32 768) con accuratezza
    prevista ≥ T dal modello con gemelli e alias;
  - **Law IV**: la stessa scelta con la Law IV, k = 0.92, senza alias né gemelli.
  Poi si misura l'accuratezza alla D scelta, con la reference.
- **Tetto**: per ogni sottografo, previsione e misura a D = 16 384.

## Criteri, fissati ora

Differenze in punti. "Gruppo" = (grafo, N, T), 20 sottografi.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il dimensionamento esatto mantiene la promessa | in ogni gruppo, media di (misurato − T) ≥ −1.5, **e** al più 2 sottografi su 20 sotto T − 6 | media di (misurato − T) < −4 in un qualsiasi gruppo |
| **H2** — conta usare il modello esatto | su WN18RR, la media di \|misurato − T\| con il dimensionamento esatto è minore che con la Law IV, per ogni N e T | è maggiore per un qualsiasi gruppo di WN18RR |
| **H3** — il tetto | su WN18RR, a D = 16 384, misurato ≤ tetto + 2 in almeno 36 sottografi su 40, **e** \|misurato − previsto\| medio ≤ 2 | misurato > tetto + 4 in più di 4 sottografi |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- 200 query per sottografo: il rumore di campionamento su un singolo sottografo è
  circa 3 punti a T = 0.8, da cui la soglia di 6 punti per "mancato di molto".
- La D scelta è un multiplo di 64: la previsione alla D scelta supera T di poco,
  non esattamente.
- Il tetto calcolato assume che fra candidati a pari segnale vinca uno a caso; la
  regola vera della reference dipende dall'ordine di inserimento.

---

## Esito — 2026-09-28, eseguito dopo il commit `c14c4b4`

Previsioni e criteri **non modificati**. Risultati:
[`results/sizing_prereg_results.json`](../../results/sizing_prereg_results.json).
Per gruppo (20 sottografi): media di (misurato − T), media di |misurato − T|, e
sottografi sotto T − 6. Nessun obiettivo è risultato irraggiungibile.

| grafo | N | T | esatto | Law IV |
|---|---|---|---|---|
| FB15k-237 | 150 | 0.70 | +0.73 · 3.81 · 1/20 | +2.43 · 4.65 · 2/20 |
| FB15k-237 | 150 | 0.80 | +0.01 · 3.18 · 2/20 | +0.52 · 3.63 · 2/20 |
| FB15k-237 | 300 | 0.70 | +1.68 · 2.68 · 0/20 | +1.61 · 3.11 · 0/20 |
| FB15k-237 | 300 | 0.80 | +1.10 · 2.55 · 1/20 | +0.25 · 2.36 · 0/20 |
| WN18RR | 150 | 0.70 | +1.17 · 3.06 · 1/20 | **−2.82** · 4.72 · 5/20 |
| WN18RR | 150 | 0.80 | +0.87 · 2.99 · 1/20 | **−5.10** · 6.06 · **9/20** |
| WN18RR | 300 | 0.70 | +0.76 · 2.69 · 1/20 | **−3.20** · 4.45 · 4/20 |
| WN18RR | 300 | 0.80 | +0.51 · 1.85 · 0/20 | **−6.21** · 6.26 · **12/20** |

**H1 — sostenuta.** In ogni gruppo la media di (misurato − T) è ≥ −1.5 (minimo
+0.01), e al più 2 sottografi su 20 sono sotto T − 6.

**H2 — sostenuta.** Su WN18RR il dimensionamento esatto ha |misurato − T| minore
della Law IV in tutti e quattro i gruppi.

**H3 — sostenuta.** Su WN18RR a D = 16 384, 38 sottografi su 40 entro tetto + 2 (1
oltre tetto + 4); |misurato − previsto| medio 0.62 punti; tetto medio 0.946,
misurato medio 0.945.

### Cosa dice

Usato per scegliere la dimensione prima, il modello esatto **mantiene la
promessa**. La Law IV, che è ciò che la reference usa oggi per il contratto, la
mantiene su FB15k-237, dove alias e gemelli sono rari, ma su WN18RR la **manca**:
in media fino a 6.2 punti sotto l'obiettivo, e in 12 sottografi su 20 di oltre 6
punti. E il tetto imposto dagli alias è reale: su WN18RR nessuna dimensione porta
oltre il ~95%, come previsto dai soli fatti.

### Cosa ne segue, per il prodotto

Il contratto che la reference espone (`predicted_accuracy`, Law IV) va sostituito,
per chi dimensiona una memoria su un grafo con relazioni simmetriche, dal modello
esatto, e deve dichiarare il tetto. Oggi il modello esatto è in `bsm/`, non nel
pacchetto pubblicato.

## Nota del 2026-10-01: self-loop (esplorativa, non preregistrata)

Il 2026-09-30 (commit `2a41f74`) `abm.exact` ha iniziato a modellare i self-loop (s, r, s), che in FB15k-237 sono 1625 (l'audit diceva 0) e in WN18RR 7. Questa preregistrazione è stata calcolata prima. I numeri sopra restano quelli pubblicati: `examples/replicate.py` li riproduce sul commit che li ha registrati. L'harness congelato non gira più sul modello attuale (usa la `predict`
della preregistrazione 4).

`examples/selfloop_impact.py` ripete la scelta della D con il modello attuale sui
9 sottografi FB15k-237 che contengono self-loop (18 coppie sottografo × obiettivo;
nessun sottografo WN18RR ne contiene). **Nessuna D cambia**, quindi nessuna
promessa cambia esito. Controllo: sui sottografi senza self-loop la previsione
attuale coincide con la pubblicata entro 1e-16.

## Errata del 2026-10-01: il criterio per sottografo (H1)

L'esito sopra non cambia. Due errori di questa preregistrazione:

- **Il rumore dichiarato è sbagliato.** "200 query per sottografo" è un tetto (`Q_MAX` in `examples/sizing_prereg.py`), non il numero: su FB15k-237 con N = 150 la mediana è 92 query e il minimo 10; altrove la mediana è 135–200 e il minimo 85–93. Con 92 query l'errore standard a T = 0.8 è circa 4.2 punti, non 3.
- **La seconda condizione di H1 era mal tarata.** Se la misura di ogni sottografo è binomiale attorno alla previsione alla D scelta, con il suo numero di query, i sottografi attesi sotto T − 6 sono 0.4–2.2 per gruppo (osservati 0–2), e un modello perfetto supera "al più 2 su 20" in tutti e otto i gruppi con probabilità 0.39 (0.25 se la probabilità di successo è T invece della previsione). Il criterio era quindi più severo del previsto, non più lasco: superarlo non è una prova debole, ma un fallimento avrebbe detto poco. Le prossime soglie per sottografo vanno fissate in unità di errore standard.
