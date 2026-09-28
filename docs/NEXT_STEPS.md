# NEXT_STEPS — stato del lavoro verso 9/10

Nota di passaggio tra sessioni. Tiene lo stato **corrente**, non la storia: la
storia è nei commit. Aggiornarla alla fine di ogni blocco di lavoro.

**Obiettivo (kiatto, 2026-09-28):** portare il progetto ad almeno 9/10 su una
review ostile, lavorando in autonomia. Vale come riapertura del FREEZE per
questo lavoro.

**Regole che valgono sempre:** ogni affermazione nuova si preregistra prima di
vedere i dati (`docs/preregistration/`); un esito negativo si registra, non si
aggira; i risultati vecchi restano accanto ai nuovi; budget zero.

## Stato al 2026-09-28, sera

Paper `docs/paper.md` **v1.4**, voto della review ostile **7/10**.

| preregistrazione | commit | stato |
|---|---|---|
| `fb15k237.md` — Law IV (k = 0.92) su FB15k-237 uniforme | `fc3aa26` | **fatta**: H1 sostenuta (1.05 punti), H2 hub né sostenuta né falsificata |
| `exact_contract.md` — contratto esatto, 0 parametri: D = 16384, multi-oggetto, FB15k-237 denso | `0856406` | **fatta**: H1, H2, H4 sostenute; H3 sostenuta a 8192 e in parte a 2048 (bias −2.6); H5 indecisa |
| `dependence.md` — Law V violata di una quantità esatta; P3 ricostruito | `e6bb351` | **fatta**: E1, E2, G sostenute; Law V respinta a 6 SE come previsto |
| `twins.md` — gemelli simmetrici (peso 2), Law VII esatta, WN18RR | `aebd0dc` | **fatta**: H1–H5 tutte sostenute |
| `composition.md` — composizione grounding × reasoning senza calibrazione, stress test ricostruito | `32d948c` | in esecuzione / da valutare |

Risultati: `results/exact_prereg_results.json`, `results/dependence_prereg_results.json`.
Valutare **solo** con i criteri scritti nei file di preregistrazione, e
scrivere l'esito in coda allo stesso file, come in `fb15k237.md`.

## Prossimi passi, in ordine

1. **Valutare le due preregistrazioni in corso** e registrarne l'esito.
2. **Paper v1.5**, se reggono: il contratto esatto (`bsm/memory/exact_contract.py`)
   diventa il risultato principale, senza parametri; la Law IV resta come
   approssimazione asintotica che spiega la scala; la Law V diventa "violata di
   −ρ²/(1−ρ²) per bit", con la formula esatta per due hop. Aggiornare abstract
   (≤ 1920 caratteri), tabella delle leggi, §3, §4, §5 (P3), §7, §9, figure.
   Se falliscono, scriverlo, e capire perché prima di qualsiasi altra cosa.
3. **Conformal prediction** per il termine d'audit dei contratti (oggi un IC
   gaussiano su n = 40): garanzia a campione finito, senza ipotesi di
   distribuzione.
4. **Riproducibilità**: quattro JSON in `results/` non hanno lo script che li ha
   prodotti, perso prima del commit: `independence_results.json` e
   `projection_results.json` (sostituiti dalla preregistrazione 3),
   `conjecture7_results.json` (Law VII, il confronto 6.3% contro 28.5%) e
   `composition_stress_results.json` (stress test della composizione). Gli
   ultimi due vanno ricostruiti con una preregistrazione, e il paper deve
   dichiararli non riproducibili finché non lo sono. La Law VII ha anche una
   versione esatta possibile: p_agree con pesi interi, per convoluzione.
5. **Il pessimismo del modello esatto su sottografi densi reali** (fino a 8
   punti a D = 2048, alto carico; stessa direzione sugli hub): capire la causa,
   poi preregistrare.
6. **Pubblicazione**, che resta il vero collo di bottiglia: PyPI (serve
   l'account di kiatto, vedi `PUBLISHING.md`), poi Zenodo, poi l'email a
   Clarkson (autore del Teorema 16 con cui il paper si confronta).

## Cosa resta fuori portata senza kiatto

PyPI e Zenodo richiedono i suoi account; l'endorsement arXiv e i tester
richiedono persone. Il lavoro sul paper non sostituisce nessuna delle due cose.
