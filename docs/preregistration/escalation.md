# Preregistrazione 12 — il contratto come criterio di escalation

Committata **prima** di interrogare qualsiasi modello sulle domande del dataset,
insieme all'harness `examples/escalation_prereg.py`. Verifica tecnica dell'ipotesi
H6 di `PRODUCT_HYPOTHESES.md` (proposta da kiatto il 2026-10-02). Non è evidenza
per H0: nessun utente è coinvolto.

## La domanda

Un sistema risponde con un modello piccolo e la memoria ABM e, per una parte delle
domande, passa a un percorso più grande e più costoso. **Decidere quali domande
passare con la previsione di `abm.exact` funziona meglio che deciderlo a caso, o
con la confidenza del modello piccolo?**

## Dati e pipeline

- **Domande e memorie:** quelle del test 11 (`human_questions.md`): le 643 domande
  di test di SimpleQuestions v2 la cui tripla è in FB15k-237, 28 memorie da 25
  soggetti, stesso collegamento per nome, stesso prompt.
- **Unica differenza: D = 8192** invece di 16 384, perché la memoria sia un collo
  di bottiglia. Scelta solo con il modello, prima di qualsiasi misura: a D = 8192
  l'accuratezza prevista della memoria sulle coppie vere è 0.556 in media (sulle
  743), deviazione standard 0.30, 10°–90° percentile 0.15–0.99.
- **Percorso piccolo:** Gemma 4 E2B (`gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf`, sha256
  `0a5bbc20…72da28`, lo stesso del test 11) sceglie la relazione; la memoria ABM a
  D = 8192 risponde.
- **Percorso grande:** Qwen3-4B Instruct 2507 (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`,
  sha256 `3605803b…c67e597`) sceglie la relazione con lo stesso prompt; uno store
  esatto restituisce il primo oggetto, in ordine, della coppia (entità collegata,
  relazione scelta).
- Entrambi via `llama-server` di llama.cpp, temperatura 0, ragionamento disattivato,
  massimo 8 token. Una risposta è corretta se è uno degli oggetti della coppia vera.

## I criteri di escalation

Per ogni domanda un punteggio, calcolato **senza** la risposta giusta. Si passano
al percorso grande per prime le domande con il punteggio più basso.

| criterio | punteggio |
|---|---|
| **C** — contratto | `abm.exact.predict_queries` per (entità collegata, relazione scelta dal piccolo) |
| **F** — confidenza del piccolo | probabilità del primo token generato da Gemma (1 se l'entità ha una sola relazione) |
| **CF** — combinato | C × F |
| **R** — a caso | valore atteso della scelta a caso, in forma chiusa |

Le domande in cui il piccolo non ha un'entità collegata vanno sempre al grande, con
ogni criterio, e contano nella quota. Sono 59 su 643 (9.2%), contate dal
collegamento, che è deterministico e non usa modelli; per questo la quota più bassa
è il 20% e non il 10%, dove resterebbero 5 domande da scegliere. 24 domande hanno
un'entità con una sola relazione (F = 1).

**Misura.** Accuratezza del sistema alle quote di escalation 20%, 30%, 40% e 50%;
riassunto: la media sulle quattro quote (AUC4). Differenze fra criteri con
intervallo al 95% da un bootstrap per cluster sulle 28 memorie (10 000 estrazioni,
seed 20261002).

## Ipotesi e criteri, fissati ora

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il contratto batte il caso: AUC4(C) − AUC4(R) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — il contratto contro la confidenza: AUC4(C) − AUC4(F) | intervallo tutto sopra 0 | intervallo tutto sotto 0 |
| **H3** — il contratto aggiunge alla confidenza: AUC4(CF) − AUC4(F) | intervallo tutto sopra 0 | stima ≤ 0 |

Il resto è **sostenuto in parte** (H1, H3) o **indeciso** (H2).

**Conseguenza per H6, fissata ora.** H6 si abbandona se H1 è falsificata, oppure
se H2 e H3 sono entrambe falsificate: il contratto non batte il caso, oppure è
peggio della confidenza del modello piccolo e non le aggiunge nulla.

**Previsioni (non criteri).** H1 sostenuta: il contratto vede gli alias e il
carico, che variano molto fra le domande. H2 incerta: gli errori del front-end
(nel test 11, 0.41 di relazioni giuste sul test) non li vede il contratto, e li
vede in parte la confidenza. H3 sostenuta: i due segnali guardano errori diversi.

## Cosa ho visto prima

- Le scelte di Gemma su queste 643 domande sono deterministiche e le ho già viste
  nel test 11: so che è giusta la relazione nel 41% dei casi, e quali. Non ho visto
  le sue confidenze.
- Nessuna misura della memoria a D = 8192, nessuna risposta di Qwen.
- Lo smoke test usa una domanda inventata.

## Limiti dichiarati prima

- Il "grande" è un modello da 4 miliardi di parametri: misura il principio, non il
  caso con un modello grande vero.
- Il percorso grande risponde con uno store esatto: è più forte della memoria anche
  a parità di front-end, e questo favorisce l'escalation in sé, non un criterio.
- Un solo grafo, mondo chiuso, 28 cluster: l'intervallo per cluster è largo.
- Il contratto prevede l'errore della memoria, non quello del front-end; C usa la
  relazione scelta dal piccolo, che può essere sbagliata.
- Il costo è contato come quota di domande passate, non in tempo o denaro.
