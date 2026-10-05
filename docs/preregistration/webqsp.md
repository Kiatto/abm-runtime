# Preregistrazione 17 — il front-end a embedding su domande naturali

Committata **prima** di interrogare qualsiasi modello su queste domande, insieme
all'harness `examples/webqsp_prereg.py`.

## Perché

Nella preregistrazione 16 la sola relazione più simile alla domanda (bge-small,
senza modello linguistico) ha dato 0.647 di risposte giuste, più di Gemma sulla sua
shortlist (0.593) e vicino a Qwen3-4B con la stessa memoria (0.681). Non era
un'ipotesi, e c'è un sospetto: in SimpleQuestions le domande sono scritte da
persone che vedevano la tripla, quindi somigliano lessicalmente alla relazione.
Questo test lo verifica su domande che non sono state scritte così.

## Dati

- **WebQuestionsSP** (Yih et al. 2016), archivio originale Microsoft `WebQSP.zip`,
  sha256 `95cb9cd2…71b0ee`: domande prese dalle ricerche Google, annotate dopo.
- Si usano le domande di train e test con catena di **una** relazione, entità di
  partenza in FB15k-237 e almeno una tripla (entità, relazione, risposta) in
  FB15k-237: **515** (308 + 207), 29 relazioni. Esempi: "what did william
  shakespeare do for a living", "what town was martin luther king assassinated in".
- Una risposta è giusta se è una delle risposte annotate (entità) della domanda.

## Pipeline

- Come nel test 11: collegamento per nome (il nome più lungo, fra le entità della
  memoria, contenuto nella domanda), memorie ABM con tutte le triple di FB15k-237
  dei loro soggetti, D = 16 384, reference congelata.
- Memorie da **10** soggetti (seed 20261008): **27** memorie. Con 25 soggetti, come nel
  test 11, sarebbero state 11, troppo poche per il bootstrap per cluster; cambiato
  prima di qualsiasi misura.
- Quattro front-end per la relazione:

| | front-end |
|---|---|
| **E** | solo embedding: la relazione più simile alla domanda (bge-small, come nella 16) |
| **S** | shortlist di 3 (k fissato nella 16, non ritarato) + Gemma 4 E2B con anteprima |
| **G** | Gemma 4 E2B con tutte le opzioni e anteprima (prompt della 15) |
| **Q** | Qwen3-4B con tutte le opzioni (prompt dei test 11–14) |

Modelli, file e sha256 come nelle preregistrazioni 12–16; temperatura 0; server con
contesto 8 192.

## Ipotesi e criteri, fissati ora

Accuratezza delle risposte. Bootstrap per cluster sulle 27 memorie, 10 000
estrazioni, seed 20261008.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — E batte G | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — S batte G (la shortlist regge su domande naturali) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H3** — E non inferiore a Q di più di 5 punti | limite inferiore di E − Q sopra −5 | limite superiore sotto −5 |
| **H4** — E contro S | intervallo tutto sopra 0 | intervallo tutto sotto 0 |

Il resto è **sostenuto in parte** (H1–H3) o **indeciso** (H4).

**Previsioni (non criteri).** H2 sostenuta. H1 sostenuta ma più piccola che nella
16. H3 incerta: su domande naturali, lontane dal nome della relazione, l'embedding
dovrebbe perdere di più di un modello da 4 miliardi. H4: prevedo S ≥ E, al
contrario della 16, per la stessa ragione.

## Cosa ho visto prima

- Il numero di domande utilizzabili, le relazioni coinvolte, tre esempi di domanda
  per split, e il numero e la dimensione delle memorie.
- **Nessuna** risposta di nessun modello, nessun richiamo della shortlist, nessuna
  misura del collegamento su queste domande.

## Limiti dichiarati prima

- Le domande di WebQuestionsSP sono in minuscolo e senza punteggiatura; il
  collegamento per nome può fallire più spesso che nel test 11, allo stesso modo
  per tutti i front-end.
- Le risposte annotate possono essere più ampie delle triple di FB15k-237 (o
  diverse): conta come giusta la risposta della memoria se è fra quelle annotate.
- 515 domande e 27 cluster: differenze di 3–4 punti possono non uscire dal rumore.
- Un solo grafo; i modelli sono piccoli.
