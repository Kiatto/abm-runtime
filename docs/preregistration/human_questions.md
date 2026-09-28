# Preregistrazione — il contratto su domande scritte da persone

**Data: 2026-09-28.** Committato insieme a
[`examples/human_questions_prereg.py`](../../examples/human_questions_prereg.py)
**prima** di interrogare il modello linguistico su qualsiasi domanda del dataset.
Lo smoke test ha usato una domanda inventata («tell me about …»), non del
dataset. Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

Ogni test precedente interroga la memoria con coppie (soggetto, relazione) date.
Il paper lo dichiarava come limite: nessuna domanda scritta da persone. Qui le
domande sono umane, e la pipeline è quella che userebbe qualcuno: un front-end
linguistico traduce la domanda in (soggetto, relazione), la memoria risponde, e il
contratto — emesso prima delle domande di test — prevede l'accuratezza finale.

## Dati

- **SimpleQuestions v2** (Bordes et al., 2015): 108 442 domande scritte da persone,
  ciascuna con la tripla Freebase che la risponde. Archivio originale,
  sha256 `58f65630…6788bf5`.
- Si usano le **743** la cui tripla (s, r, o) è esattamente in FB15k-237
  (stessi identificativi Freebase): 87 relazioni, 676 soggetti.
- Nomi delle entità: `FB15k_mid2name.txt` del mirror `KGraph/FB15k-237`,
  sha256 `4da94b80…6d13d73`; tutti i 676 soggetti hanno un nome.

**Cosa ho visto prima:** il numero di domande con la tripla in FB15k-237, le
relazioni e i soggetti coinvolti, che tutti i soggetti hanno un nome, e tre
esempi di domanda. **Nessuna** risposta del modello linguistico e nessuna misura.

## Pipeline

- **Memorie.** I 676 soggetti, mescolati con seed fisso, in 28 gruppi da 25; ogni
  memoria contiene **tutte** le triple di FB15k-237 dei suoi soggetti (da 26 a
  2 394 triple), a D = 16 384, con la reference congelata. Ogni domanda interroga la
  memoria del suo soggetto.
- **Collegamento.** L'entità della memoria il cui nome (minuscolo, "_" come spazio)
  è il più lungo contenuto nella domanda.
- **Relazione.** Se l'entità collegata ha più relazioni memorizzate, un modello
  linguistico locale sceglie per indice fra quelle, con un prompt fisso
  (nell'harness). Modello: `gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf` (Unsloth), sha256
  `0a5bbc20f91f92da96ab4870fa71b356c45b8500a7b8b9c3e0eb48359b72da28`, servito da
  `llama-server` di llama.cpp (`-c 4096 --parallel 1`), temperatura 0, ragionamento
  disattivato.
- **Risposta.** La memoria risponde a (entità, relazione); la risposta è corretta se
  è uno degli oggetti memorizzati per la coppia (s, r) vera.
- **Audit e test.** Le 743 domande sono mescolate con seed fisso: le prime 100 sono
  l'**audit**, le altre 643 il **test**.

## Il contratto

Emesso dopo l'audit e prima di qualsiasi domanda di test:

    previsto = π̂ × m̄

- π̂ = quota di domande dell'audit in cui collegamento e relazione sono entrambi
  giusti, con intervallo di Wilson al 95%;
- m̄ = media sulle 643 domande di test dell'accuratezza della memoria per la coppia
  (s, r) vera, da `abm.exact` (gemelli e alias). Calcolata ora, solo dalla teoria:
  **m̄ = 0.8019**.

L'intervallo del contratto è quello di π̂ moltiplicato per m̄.

## Criteri, fissati ora

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — il contratto, end-to-end | l'accuratezza misurata sul test cade nell'intervallo del contratto **e** \|misurato − previsto\| ≤ 5 punti | \|misurato − previsto\| > 10 punti |
| **H2** — il livello della memoria | con la coppia (s, r) vera, accuratezza misurata sul test entro 3 SE da m̄ | oltre 5 SE |
| **H3** — la stabilità del front-end | accuratezza del front-end sul test dentro l'intervallo di Wilson dell'audit | fuori di oltre 5 punti |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- Il contratto assume che il successo del front-end e quello della memoria siano
  indipendenti fra le domande; se le domande più difficili da collegare sono anche
  quelle con la memoria più carica, l'ipotesi cade.
- Ignora le risposte giuste arrivate per una strada sbagliata (entità o relazione
  sbagliata che porta comunque a un oggetto corretto).
- Il front-end è volutamente semplice: il test riguarda la previsione della sua
  accuratezza, non la sua qualità.
- Le domande sono solo quelle con la tripla in FB15k-237: un sottoinsieme
  selezionato di SimpleQuestions.
