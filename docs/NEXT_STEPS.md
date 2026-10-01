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

Paper `docs/paper.md` **v1.8** (undici preregistrazioni, abm.exact nel pacchetto), **audit ostile a più agenti del 2026-09-30: 5.5/10** (il mio 8.5 era troppo generoso). Leggere `docs/audit_2026-09-30.md`: nove bloccanti prima di qualsiasi email. Da integrare nel paper: preregistrazioni 6–9 (due fallite, con le cause), la regola esatta dei pareggi, la figura 10. Problema aperto emerso: recupero fuori percorso con codebook piccoli (nessun modello pulito).

| preregistrazione | commit | stato |
|---|---|---|
| `fb15k237.md` — Law IV (k = 0.92) su FB15k-237 uniforme | `fc3aa26` | **fatta**: H1 sostenuta (1.05 punti), H2 hub né sostenuta né falsificata |
| `exact_contract.md` — contratto esatto, 0 parametri: D = 16384, multi-oggetto, FB15k-237 denso | `0856406` | **fatta**: H1, H2, H4 sostenute; H3 sostenuta a 8192 e in parte a 2048 (bias −2.6); H5 indecisa |
| `dependence.md` — Law V violata di una quantità esatta; P3 ricostruito | `e6bb351` | **fatta**: E1, E2, G sostenute; Law V respinta a 6 SE come previsto |
| `twins.md` — gemelli simmetrici (peso 2), Law VII esatta, WN18RR | `aebd0dc` | **fatta**: H1–H5 tutte sostenute |
| `composition.md` — composizione grounding × reasoning senza calibrazione, stress test ricostruito | `32d948c` | **fatta**: H1–H3 sostenute; missing da 10.2 a 0.75 punti |
| `asymmetric.md` — encoding simmetrico contro asimmetrico | `f4a7dd1` | **fatta**: H1–H4 sostenute; inversione 6/6 |
| `deepchain.md` — dipendenza fra hop nelle catene profonde (h fino a 6) | `2157233` | **FALSIFICATA**: codebook minuscolo, domina il recupero fuori percorso |
| `deepchain2.md` — lo stesso con 1000 distrattori | `b4eec35` | **FALSIFICATA** già al singolo hop: regola dei pareggi (la reference sceglie il primo inserito) |
| `deepchain3.md` — lo stesso con la regola esatta dei pareggi, D = 256 e 320 | `794def8` | **fatta**: H1, H2 sostenute; Law V respinta fino a 8.5 SE a h = 6 |
| `sizing.md` — il contratto usato per scegliere D prima; il tetto degli alias | `c14c4b4` | **fatta**: H1–H3 sostenute; la Law IV manca la promessa su WN18RR (12/20) |
| `human_questions.md` — il contratto su domande scritte da persone (SimpleQuestions ∩ FB15k-237, LLM locale) | `b63f65b` | **fatta**: memoria sostenuta (+0.3 SE); end-to-end in parte (+7.2 punti, dentro l'intervallo) |

Risultati: `results/exact_prereg_results.json`, `results/dependence_prereg_results.json`.
Valutare **solo** con i criteri scritti nei file di preregistrazione, e
scrivere l'esito in coda allo stesso file, come in `fb15k237.md`.

## Stato al 2026-09-30

Tier A dell'audit fatto sul testo (commit `2a41f74`, `9407eab`), paper **v1.9**.
Aperto: (a) modello dei cicli pari su GF(2) — l'indipendente è ottimista fino a
6 punti, il "lineare" (dipendenza solo nel voto) pessimista di 3–4
(`examples/cycles_probe.py`); (b) contare i 4-cicli nei campioni reali e
correlarli al residuo sugli hub; (c) verificare la versione JAIR 2026 di
Clarkson; (d) Tier B e C di `docs/audit_2026-09-30.md`; (e) nuovo audit ostile su v1.9.

## Stato al 2026-10-01

Paper **v1.10**. `replicate.py` copre ora tutte le 11 preregistrazioni più
seed10, Clarkson e ProofWriter, e fa girare ogni harness sul commit che ha
registrato i risultati (`--code published`, default). Esito: 14/14 riprodotte
(12 byte per byte, twins e sizing all'ultima cifra di un float). Test 11 solo
nella parte deterministica (`--memory-only`, senza LLM).

Trovato: **FB15k-237 ha 1625 self-loop** (l'audit diceva 0). Il modello corretto
del 30/9 li tratta; twins, asymmetric e sizing, congelati, non girano più sul
modello attuale. Effetto misurato (`examples/selfloop_impact.py`, esplorativo):
medie di twins spostate < 0.1 punti, ma una cella WN18RR era sbagliata di 17.6
punti e ora no; sizing invariato; test 11 +0.13 punti. Poi ricalcolati anche
asymmetric e le ipotesi: H1–H4 di twins e H2, H4 di asymmetric restano sostenute
con il modello attuale.

`BENCHMARKS.md` fatto (ogni numero con file, script e nome in `replicate.py`;
limiti in poche righe), linkato dal README. Prossimo: la pubblicazione (punto 6),
che richiede kiatto.

## Stato al 2026-10-01, sera

Paper v1.10 rivisto dopo il **secondo audit ostile: 6.0/10** (`docs/audit_2026-10-01.md`).
Fatti: Tier B/C del primo audit; FKS gaussiano (il raffinamento binomiale non
aggiunge nulla di misurabile: il contributo è la contabilità — pareggi, risposte
multiple, alias, gemelli, hop — e la validazione preregistrata); cicli pari
(nessuna associazione con l'errore); residuo sugli hub = gemelli; Tier A 2–4 e
B 5–7, 9, 11, C 12–13, 15–16 del secondo audit. 196 test; replicate copre anche
proofwriter_seeds e scarica solo i dati necessari.

**Bloccante che solo kiatto può sbloccare:** il push (il remoto è indietro: chi
clona non può riprodurre BENCHMARKS.md). Poi tag, e replica da un clone del remoto.

Aperti, non bloccanti per la prima email: test su testo estratto a carico reale
(DocRED, preregistrato; L); tabella a pari memoria contro Bloom/hash (M);
indipendenza delle distanze nulle fra hop a D piccolo (C14); --code current per
i test 4, 6, 10 (C17); un secondo campionatore denso (C18).

## Prossimi passi, in ordine

1. ~~Valutare le preregistrazioni~~ fatto. 2. ~~Paper v1.5~~ fatto: il contratto esatto (`bsm/memory/exact_contract.py`)
   diventa il risultato principale, senza parametri; la Law IV resta come
   approssimazione asintotica che spiega la scala; la Law V diventa "violata di
   −ρ²/(1−ρ²) per bit", con la formula esatta per due hop. Aggiornare abstract
   (≤ 1920 caratteri), tabella delle leggi, §3, §4, §5 (P3), §7, §9, figure.
   Se falliscono, scriverlo, e capire perché prima di qualsiasi altra cosa.
3. **Conformal prediction** per il termine d'audit dei contratti (oggi un IC
   gaussiano su n = 40): garanzia a campione finito, senza ipotesi di
   distribuzione.
4. ~~Riproducibilità dei quattro JSON senza script~~: fatto nel paper v1.9
   ("Lost scripts", sostituiti dai test 3, 4 e 5).
5. **Il pessimismo del modello esatto su sottografi densi reali** (fino a 8
   punti a D = 2048, alto carico; stessa direzione sugli hub): capire la causa,
   poi preregistrare.
6. **Pubblicazione**, che resta il vero collo di bottiglia: PyPI (serve
   l'account di kiatto, vedi `PUBLISHING.md`), poi Zenodo, poi l'email a
   Clarkson (autore del Teorema 16 con cui il paper si confronta).

## Cosa resta fuori portata senza kiatto

PyPI e Zenodo richiedono i suoi account; l'endorsement arXiv e i tester
richiedono persone. Il lavoro sul paper non sostituisce nessuna delle due cose.

## Strumenti aggiunti

- `examples/replicate.py`: riesegue le preregistrazioni da un clone pulito
  (scarica i dati con sha256, scrive in `results/replica/`, confronta con i
  pubblicati).
- `examples/prereg_summary.py`: intervalli bootstrap e pavimento di rumore per
  ogni errore riassuntivo.

## Da fare, emerso dalla preregistrazione 10

~~Portare il modello esatto nel pacchetto~~ fatto: `abm.exact` (`reference/exact.py`),
con `contract_for` e `min_dimension`; il tetto degli alias è dichiarato.

## Per la preregistrazione 11 serve il server locale

    M=$(ls ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/*/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf)
    ~/.unsloth/llama.cpp/build/bin/llama-server -m "$M" --host 127.0.0.1 --port 8765 -c 4096 -t 8 --parallel 1

## Prossimo passo proposto

Preregistrazione 12: un contratto a due livelli che corregga i due difetti emersi
nel test 11 — audit più grande (per esempio 250 domande) e memoria condizionata
alle domande risolte dal front-end nell'audit — su domande nuove (SimpleQuestions
con la tripla in FB15k-237 ma soggetti non ancora usati, o un altro grafo).
