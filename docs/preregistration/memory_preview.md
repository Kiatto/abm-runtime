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

---

## Esito — 2026-10-05, eseguito dopo il commit `fec8212`

Ipotesi e criteri **non modificati**. Risultati:
[`results/memory_preview_prereg_results.json`](../../results/memory_preview_prereg_results.json);
risposte in `results/memory_preview_small_answers.json`. Con l'anteprima Gemma
cambia scelta in 186 domande su 743.

| | senza anteprima | con anteprima | differenza | 95%, cluster | esito |
|---|---|---|---|---|---|
| **H1** — relazione giusta | 0.402 | 0.400 | −0.3 | [−2.6, +2.0] | **falsificata** (stima ≤ 0) |
| **H2** — risposta giusta | 0.354 | 0.382 | **+2.8** | **[+1.0, +4.6]** | **sostenuta** |

La previsione (5–10 punti in più sulla relazione) era sbagliata.

### Da dove viene il guadagno (esplorativo, non preregistrato)

Domanda per domanda (stesso file):

- relazione giusta solo con l'anteprima: 64 domande; in 54 la risposta è giusta;
- relazione giusta solo senza: 66 domande; in 20 la memoria sbagliava comunque, e
  la perdita non costa nulla;
- in 9 domande la risposta diventa giusta con una relazione diversa da quella della
  domanda, che porta allo stesso oggetto.

Vedendo la risposta, il modello non sceglie più spesso la relazione giusta: sposta
le sue scelte verso le opzioni in cui la memoria restituisce una risposta plausibile,
ed evita quelle in cui la memoria sbaglierebbe. Il guadagno è nell'accoppiamento fra
front-end e memoria, non nel front-end da solo.
