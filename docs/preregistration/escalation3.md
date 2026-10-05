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
