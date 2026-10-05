# Preregistrazione 14 — confidenza esatta del front-end, più domande, coperture corrette

Committata **prima** di interrogare i modelli per questo test, insieme all'harness
`examples/escalation3_prereg.py`. Segue la preregistrazione 13 (`escalation2.md`),
che ha trovato il limite nel front-end piccolo e un difetto di disegno nelle
coperture. Richiesta di kiatto (2026-10-04): procedere con i tre miglioramenti.

## Cosa cambia rispetto alla 13

1. **Confidenza del front-end esatta.** Non la probabilità del primo token, ma la
   distribuzione sulle opzioni: P(opzione k) è il prodotto delle probabilità dei
   token delle sue cifre e del token di fine turno, calcolato con `llama-server`
   (`/apply-template`, poi `/completion` con `n_probs = 20`), esplorando le cifre
   con probabilità cumulata ≥ 0.01. **FM** = scarto fra le prime due opzioni dopo
   aver rinormalizzato sulle opzioni. Serve perché 315 domande su 743 hanno 10 o
   più relazioni candidate, e lì la prima cifra non distingue fra 1 e 10–19.
   Smoke test su una domanda inventata: l'opzione più probabile è quella giusta, e
   il 61% della massa va a risposte che non cominciano con un numero.
2. **Più domande:** tutte le 743 (anche le 100 di audit del test 11). Le 28
   memorie sono le stesse.
3. **Coperture corrette:** le domande forzate sono sempre astensioni; le coperture
   50/70/90% sono quote delle domande **non** forzate.

Resto dell'impianto come nella 13: memoria a D = 16 384, la stessa per il piccolo
(Gemma 4 E2B sceglie la relazione) e per il grande (Qwen3-4B la sceglie); scelte a
temperatura 0 con lo stesso prompt. Le domande forzate (nessuna relazione scelta)
sono 70 per il collegamento, più le risposte illeggibili.

## Criteri

**B — indice per rispondere o astenersi** (solo percorso piccolo; riassunto: media
sulle tre coperture dell'accuratezza delle risposte date):
F1 (probabilità del primo token, come nella 13), **FM**, **CFM** = p̂_mem × FM,
**ZFM** = media dei ranghi di Z e di FM (Z: margine di Hamming osservato, come nella 13).

**A — instradamento** (quote 20/30/40/50% di tutte le domande; riassunto AUC4):
valore **FM**: 1 − FM; **EM**: p̂_mem × (1 − FM); **R**: a caso.

## Ipotesi e criteri, fissati ora

Bootstrap per cluster sulle 28 memorie, 10 000 estrazioni, seed 20261005.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — B: FM batte F1 (la confidenza esatta è migliore) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — B: CFM batte FM (il contratto aggiunge) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H3** — B: ZFM batte FM (il margine osservato aggiunge) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H4** — A: EM batte FM | intervallo tutto sopra 0 | stima ≤ 0 |
| **H5** — A: EM batte R | intervallo tutto sopra 0 | stima ≤ 0 |

Il resto è **sostenuto in parte**.

**Previsioni (non criteri).** H1 e H5 sostenute; H2, H3, H4 in parte: la memoria
spiega circa un quinto dell'errore, e con 28 cluster differenze di un punto non
escono dal rumore.

## Cosa ho visto prima

- Per le 643 domande di test dei test 11–13: le scelte di Gemma e di Qwen, l'esito
  del percorso piccolo e del grande a D = 16 384, p̂_mem, Z e la probabilità del
  primo token (nella preregistrazione 13, in aggregato).
- Per le 100 domande di audit: dal test 11, quante volte Gemma sceglie la relazione
  giusta (in aggregato). Nessuna risposta di Qwen.
- **Nessuna** distribuzione esatta sulle opzioni, su nessuna domanda.

## Limiti dichiarati prima

- La distribuzione è troncata (cifre sotto 0.01 di probabilità cumulata non
  esplorate, top 20 per passo): FM è esatto a meno di quella massa.
- Le 100 domande in più sono la parte nuova del campione; le altre 643 sono già
  state analizzate con indici diversi.
- Stessi limiti della 13: il grande è un modello da 4 miliardi; il costo è la quota
  di chiamate; 28 cluster.

---

## Esito — 2026-10-05, eseguito dopo il commit `e5fb119`

Ipotesi e criteri **non modificati**. Risultati:
[`results/escalation3_prereg_results.json`](../../results/escalation3_prereg_results.json);
risposte in `results/escalation3_small_answers.json` e `escalation3_big_answers.json`.

**Non determinismo.** Le scelte non sono del tutto deterministiche, come i test
12 e 13 avevano dato per scontato: sulle 643 domande comuni Gemma coincide con la
preregistrazione 12 in 641 casi, Qwen in 637 (temperatura 0, stessa GPU; cause
probabili l'ordine delle somme in virgola mobile e la cache del prompt). Le
differenze sono lo 0.3% e l'1%.

743 domande, 85 forzate (70 senza collegamento, 15 risposte illeggibili).
Piccolo da solo **0.354**, grande da solo **0.681**.

**B — indice per rispondere o astenersi** (accuratezza delle risposte date)

| copertura delle non forzate | F1 | FM | CFM | ZFM |
|---|---|---|---|---|
| 50% | 0.517 | 0.550 | 0.544 | 0.581 |
| 70% | 0.447 | 0.447 | 0.466 | 0.475 |
| 90% | 0.405 | 0.410 | 0.417 | 0.429 |

**A — instradamento** (accuratezza del sistema)

| quota | FM | EM | R |
|---|---|---|---|
| 20% | 0.404 | 0.396 | 0.399 |
| 30% | 0.444 | 0.455 | 0.434 |
| 40% | 0.497 | 0.518 | 0.469 |
| 50% | 0.548 | 0.576 | 0.505 |

| ipotesi | stima (punti) | 95%, cluster | esito |
|---|---|---|---|
| **H1** — FM contro F1 | +1.3 | [−0.6, +2.9] | **sostenuta in parte** |
| **H2** — CFM contro FM | +0.7 | [−0.4, +2.2] | **sostenuta in parte** |
| **H3** — ZFM contro FM | +2.6 | [+0.9, +4.1] | **sostenuta** |
| **H4** — EM contro FM | +1.3 | [−0.1, +2.4] | **sostenuta in parte** |
| **H5** — EM contro R | +3.4 | [+2.4, +4.5] | **sostenuta** |

### Cosa dice, e cosa no

- La previsione su H1 era sbagliata: la confidenza esatta sulle opzioni migliora
  quella del primo token solo a copertura 50% (+3.3 punti) e in media non esce dal
  rumore. Il front-end piccolo resta poco calibrato anche misurato bene.
- **Il segnale che regge è il margine osservato della memoria** (H3): aggiunto alla
  confidenza esatta, alza l'accuratezza delle risposte date di 2.6 punti in media,
  da 0.550 a 0.581 a copertura 50%. Il contratto (la previsione, non l'osservazione)
  aggiunge meno e non esce dal rumore (H2).
- Per instradare, il valore EM batte il caso in modo netto (H5), e batte la sola
  confidenza alle quote alte (+2.1 e +2.8 punti al 40% e 50%), non a quelle basse;
  in media resta al limite (H4).
- In tre test (12–14) lo stesso quadro: il contratto e la memoria danno un segnale
  reale ma piccolo, perché la maggior parte dell'errore è del front-end piccolo.
