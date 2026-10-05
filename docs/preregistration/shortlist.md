# Preregistrazione 16 — meno opzioni al front-end piccolo

Committata **prima** di misurare qualsiasi cosa con la shortlist, insieme
all'harness `examples/shortlist_prereg.py`. Richiesta di kiatto (2026-10-05):
cercare un modo per migliorare ancora.

## Da dove viene l'idea, dichiarato per intero

Una diagnosi sui dati già pubblicati della preregistrazione 15 (risposte con
anteprima, tutte le 743 domande; esplorativa) ha diviso gli errori per causa:

| | domande |
|---|---|
| relazione sbagliata | 352 |
| entità sbagliata o non collegata | 75 |
| relazione giusta, memoria sbaglia | 32 |
| giusto (più 19 con un'altra relazione che porta allo stesso oggetto) | 265 |

Con l'entità giusta, Gemma sceglie la relazione giusta nel **64%** dei casi con
2–9 candidate e nel **20%** con 10 o più (314 domande, 252 errori). La leva è il
numero di opzioni.

## Impianto

- Un modello di embedding piccolo, **bge-small-en-v1.5** (`bge-small-en-v1.5-f16.gguf`,
  sha256 `f0b2fef9…bae999`), servito da `llama-server --embedding --pooling cls`,
  ordina le relazioni candidate per somiglianza coseno fra la domanda (con il
  prefisso di query di bge) e il percorso della relazione reso in parole
  (`/people/person/place_of_birth` → "people person place of birth"). Si tengono
  le prime **k**, in ordine alfabetico.
- **Scelta di k, sulle 100 domande di audit** (mai usate dai test 12–15 per misurare
  nulla): il k più piccolo in {3, 5, 8, 12} per cui la relazione giusta è nella
  shortlist in almeno il 90% delle domande di audit con entità giusta e almeno 2
  candidate; se nessuno basta, k = 12. Scritto in `results/shortlist_k.json`
  prima della fase con Gemma.
- Gemma 4 E2B sceglie fra le k con il prompt della preregistrazione 15 (anteprima
  della risposta della memoria), memoria a D = 16 384, temperatura 0, contesto 8 192.
- **Confronto** sulle 643 domande di test: le scelte con anteprima e tutte le
  opzioni della preregistrazione 15 (stesso modello, stesse domande).
- Riportata anche, non come ipotesi, la scelta del solo embedding (la prima della
  classifica, senza Gemma).

## Ipotesi e criteri, fissati ora

Bootstrap per cluster sulle 28 memorie, 10 000 estrazioni, seed 20261007.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — relazione giusta, shortlist − tutte le opzioni | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — risposta giusta, shortlist − tutte le opzioni | intervallo tutto sopra 0 | stima ≤ 0 |

Il resto è **sostenuto in parte**.

**Previsione (non criterio).** H1 e H2 sostenute, +8–15 punti sulla relazione:
nelle domande con molte candidate la shortlist porta Gemma nel regime in cui sceglie
bene. Il rischio è il richiamo: se la relazione giusta esce dalla shortlist, la
domanda è persa.

## Cosa ho visto prima

- Le scelte di Gemma con anteprima su tutte le 743 domande e i loro esiti
  (preregistrazione 15), e la diagnosi qui sopra, fatta su quei dati.
- Uno smoke test dell'embedding su una domanda inventata ("where was the poet
  born?": la relazione più simile è place of birth).
- **Nessun** richiamo della shortlist su domande del dataset, nessuna risposta di
  Gemma con la shortlist.

## Limiti dichiarati prima

- La diagnosi che motiva il test è fatta sugli stessi dati (le 643 domande di test
  ne sono la gran parte): l'effetto può essere più piccolo su domande nuove.
- Il testo della relazione è il solo percorso Freebase; descrizioni migliori
  potrebbero aumentare il richiamo.
- La baseline viene da un'esecuzione precedente (non determinismo ≈ 0.3%).
