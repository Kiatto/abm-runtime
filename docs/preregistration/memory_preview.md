# Preregistrazione 15 — il front-end piccolo vede la risposta della memoria

Committata **prima** di interrogare il modello con il prompt nuovo, insieme
all'harness `examples/memory_preview_prereg.py`. Richiesta di kiatto
(2026-10-05): migliorare il front-end, che nei test 12–14 è il collo di bottiglia.

## La domanda

Nei test 11–14 Gemma 4 E2B sceglie la relazione vedendo solo i nomi delle relazioni
memorizzate per l'entità collegata, e sceglie quella giusta nel 41% dei casi.
**Se ogni opzione mostra anche la risposta che la memoria dà per quella relazione,
il modello sceglie meglio?**

    3. /people/person/place_of_birth (stored answer: Seattle)

Il margine osservato non serve a questo: ogni candidata è una relazione davvero
memorizzata per l'entità, quindi ognuna dà una risposta lontana dal rumore. Serve il
contenuto della risposta, che dice al modello di che tipo è.

## Impianto

- Le 743 domande e le 28 memorie dei test 11–14, memoria a **D = 16 384**.
- Stesso collegamento per nome; stesso modello (`gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf`,
  sha256 `0a5bbc20…72da28`), temperatura 0, ragionamento disattivato, 8 token.
- **Prompt nuovo** (nell'harness, `PROMPT2`): quello dei test 11–14 più la frase
  "Each option shows the answer stored for it." e, per ogni opzione, "(stored
  answer: <nome>)", con il nome leggibile dell'oggetto restituito dalla memoria
  (l'identificativo Freebase se manca il nome).
- Contesto del server 8 192 token invece di 4 096, perché le opzioni sono più
  lunghe.
- **Confronto:** le scelte senza anteprima della preregistrazione 14
  (`results/escalation3_small_answers.json`), stesso modello e stesse domande.

## Ipotesi e criteri, fissati ora

Bootstrap per cluster sulle 28 memorie, 10 000 estrazioni, seed 20261006.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — relazione giusta (collegamento e relazione) con anteprima − senza | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — risposta giusta end-to-end con anteprima − senza | intervallo tutto sopra 0 | stima ≤ 0 |

Il resto è **sostenuto in parte**.

**Previsione (non criterio).** H1 e H2 sostenute, con un guadagno di 5–10 punti
sulla relazione: molte relazioni candidate di un'entità hanno risposte di tipo
diverso (un luogo, una persona, una data, un genere).

## Cosa ho visto prima

- Le scelte senza anteprima su tutte le 743 domande (preregistrazioni 12–14) e i
  loro esiti a D = 16 384.
- Nessuna risposta del modello con il prompt nuovo. Lo smoke test usa una domanda
  inventata.

## Limiti dichiarati prima

- La baseline viene da un'esecuzione precedente: fra esecuzioni le scelte di Gemma
  cambiano in circa lo 0.3% delle domande (preregistrazione 14).
- L'anteprima costa una query alla memoria per relazione candidata, e un prompt più
  lungo: il costo non è misurato.
- Se la memoria sbaglia la risposta di una relazione, l'anteprima mostra un nome
  sbagliato; l'effetto netto è quello misurato.
- Un solo modello piccolo, un solo grafo.
