# Preregistrazione — composizione grounding × reasoning, senza calibrazione

**Data: 2026-09-28.** Committato insieme a
[`examples/composition_prereg.py`](../../examples/composition_prereg.py) **prima**
di eseguire, a parte uno smoke test che stampa solo conteggi. Stesse regole di
[`fb15k237.md`](fb15k237.md).

## Perché, e cosa non andava

Il paper (§7) presenta la "Resource Composition Law", Acc = E_q[Pg(q)] ·
Pr(N_eff(ε)), come corroborata e senza parametri. Rileggendo il codice:

- in `extraction_robustness.py` **Pr è l'accuratezza misurata a ε = 0**, e la
  previsione per i fatti spuri usa una scala calcolata dalle misure: la legge è
  calibrata sui dati che prevede;
- la forma "raffinata" per i fatti mancanti (la traccia più leggera), descritta
  in `docs/law8_report.md` e da cui viene il "~3%" del paper, **non è
  implementata**: lo script usa (1 − ε)² · Pr, e con 10 seed sbaglia di 10 punti
  sui fatti mancanti;
- lo stress test con errori non i.i.d. (`composition_stress_results.json`, "52%
  contro 30% a ε = 0.4") **non ha uno script**.

## Previsione, a zero parametri

Per ogni catena a due hop (x, r₁, y), (y, r₂, z): se uno dei due fatti è
mancante o corrotto, la catena non può riuscire (previsione 0); se sono intatti,
riesce con `two_hop_joint_fast` al carico N e al codebook M **effettivi** della
traccia, che includono la dipendenza fra hop (preregistrazione 3). La previsione
di cella è la quota di catene intatte per quella probabilità. Nessuna misura
entra nella previsione.

Per confronto, riportato ma non come criterio: la stessa previsione con i due
hop trattati come indipendenti (p²).

**Cosa ho visto prima:** i risultati di `extraction_robustness.py` (3 e 10
seed) e il JSON orfano dello stress test. **Non ho visto** nessuna misura
prodotta da questo harness, i cui seed sono nuovi.

## Parti

Stesso disegno di `extraction_robustness.py`: D = 2048, 60 catene (120 fatti),
10 seed, domande (x, z) via `Memory.chain`.

- **R1** — tipi *missing*, *wrong_relation*, *wrong_entity* (ciascun fatto
  corrotto con probabilità ε) e *spurious* (ε·120 fatti inventati in più),
  ε ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}.
- **R2** — stesso tasso medio d'errore per fatto, tre strutture (errore
  *wrong_entity*), ε ∈ {0.1, 0.2, 0.3, 0.4}: *iid* (ogni fatto con prob. ε),
  *chain* (tutta la catena con prob. ε), *hop2* (solo il secondo fatto, con prob.
  2ε).

## Criteri, fissati ora

Errori come previsto − misurato, in punti, sulle celle mediate sui seed.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — R1, per ciascun tipo | errore medio assoluto ≤ 3 | > 6 per un qualsiasi tipo |
| **H2** — R2, per ciascuna struttura | errore medio assoluto ≤ 3 | > 6 per una qualsiasi |
| **H3** — R2, l'ordine | a ε = 0.4, misurato chain > iid > hop2, come prevedono le quote di catene intatte (1 − ε, (1 − ε)², 1 − 2ε) | l'ordine misurato è diverso |

Tutto il resto è **sostenuto in parte**.

## Limiti dichiarati prima

- La previsione 0 per le catene non intatte ignora la probabilità, di ordine
  1/M, che un cleanup sbagliato arrivi per caso alla risposta giusta.
- Le relazioni sono due (r₁, r₂): nessun alias per costruzione.
