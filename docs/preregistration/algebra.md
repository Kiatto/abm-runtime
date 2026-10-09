# Preregistrazione 19 — l'algebra di ABM contro uno store esatto con join, a pari bit e a pari tempo

**Data: 2026-10-09.** Committata **prima** di misurare, insieme all'harness
[`examples/algebra_prereg.py`](../../examples/algebra_prereg.py). Prima del commit
l'harness è stato eseguito solo con `--smoke` (grafo sintetico generato nello
script, mai FB15k-237 o WN18RR) e con `--predict` (legge i dati e calcola le
previsioni del modello esatto e le accuratezze analitiche degli store; non
costruisce nessuna memoria ABM sui dati reali). Stesse regole di
[`fb15k237.md`](fb15k237.md).

## Perché

La preregistrazione 18 ([`equal_bits.md`](equal_bits.md)) ha mostrato che a pari bit
uno store esatto e un Bloom battono ABM sul recupero (s, r) → o e sull'appartenenza.
Il paper dichiara come valore residuo "l'algebra (composizione, legame)", mai
misurata contro un'alternativa. Qui la si misura, contro l'alternativa ovvia: un
dizionario e un join.

## Impianto (ostile ad ABM)

Grafi: FB15k-237 e WN18RR, train split, gli stessi file e sha256 di
`examples/replicate.py` (già in `data/external/`). Per ogni (dataset, D, N, seed):
N fatti distinti campionati come percorsi a due hop (x, r₁, y), (y, r₂, z), z ∉ {x, y};
le domande sono i percorsi **funzionali nel campione** ((x, r₁) e (y, r₂) hanno un solo
oggetto), al più 200 per cella e seed. Griglia: D = 2048 con N ∈ {100, 200, 400, 800};
D = 8192 con N ∈ {400, 800, 1600, 3200}; 3 seed. 16 celle per compito.

Bit dello store, con E entità e R relazioni nel campione, le = ⌈log₂ E⌉, lr = ⌈log₂ R⌉:

- **store di retrieval (il baseline):** chiavi implicite, come un XOR filter / Bloomier
  per una funzione statica: 1.23 · le bit per fatto (o per percorso). Realizzabile, non
  idealizzato; un ribbon filter farebbe meglio (≈ 1.05). Se D non basta tiene un
  sottoinsieme a caso e perde il resto. Come ABM, su chiavi non memorizzate risponde
  qualcosa di arbitrario.
- **store a chiavi esplicite** (quello della preregistrazione 18): 2·le + lr bit per
  fatto, senza sovraccarico. Solo riportato.

Il vocabolario (nomi → codeword o → indici) è gratis per entrambi: i codeword ABM si
rigenerano dai nomi, lo store ha bisogno degli stessi nomi.

- **A — catena a due hop.** ABM: una traccia di D bit con gli N fatti,
  `Memory.chain(x, [r₁, r₂])` (cleanup per hop). Store: D bit, due lookup (join).
- **B — composizione compilata.** ABM: una traccia di D bit con i P percorsi domandati
  composti (`compile_pairs`, legame XOR che elimina il ponte), una sola cleanup
  (`query_compiled`). Store: D bit, il migliore **previsto** fra tabella dei percorsi
  (s, r₁, r₂) → z e fatti base + join (scelto ora, dalla previsione: in tutte le celle
  è la tabella).
- **C — tempo.** Tempo medio per domanda a due hop: ABM con cleanup vettoriale numpy
  (identica alla reference, verificata sulla prima domanda di ogni cella; più veloce
  della reference, a favore di ABM) contro `dict` Python con due `get`.

A pari tempo non serve una griglia: se C mostra ABM più lento di due ordini di
grandezza, a pari tempo lo store risponde a più domande con accuratezza ≥ di quella a
pari bit.

## Previsioni, calcolate ora (`--predict`)

ABM, catena: prodotto delle accuratezze per hop di `exact.predict_queries` sul
campione (gemelli, alias, codebook = entità + relazioni). ABM, compilata:
`predict_queries` sulle triple sintetiche (x, r₁∘r₂, z), con r₁∘r₂ simmetrico e nullo
per r₁ = r₂ (perché ρ(r₁) ⊕ ρ(r₂) lo è). Store: analitico, P(entrambi i fatti tenuti)
= c(c−1)/(N(N−1)) con c = ⌊D/bit per fatto⌋; tabella = min(1, c/P). Medie sui 3 seed.

| dataset | D | N | Q | A: ABM previsto | A: store join | B: ABM previsto | B: store |
|---|---|---|---|---|---|---|---|
| FB15k-237 | 2048 | 100 | 140 | 0.618 | 1.000 | 0.990 | 1.000 |
| | | 200 | 277 | 0.129 | 0.855 | 0.780 | 1.000 |
| | | 400 | 513 | 0.012 | 0.172 | 0.371 | 0.959 |
| | | 800 | 600 | 0.001 | 0.043 | 0.245 | 0.830 |
| | 8192 | 400 | 520 | 0.456 | 1.000 | 0.986 | 1.000 |
| | | 800 | 600 | 0.060 | 0.693 | 0.956 | 1.000 |
| | | 1600 | 600 | 0.004 | 0.143 | 0.942 | 1.000 |
| | | 3200 | 600 | 0.000 | 0.030 | 0.925 | 1.000 |
| WN18RR | 2048 | 100 | 148 | 0.520 | 1.000 | 0.988 | 1.000 |
| | | 200 | 299 | 0.129 | 0.855 | 0.756 | 1.000 |
| | | 400 | 592 | 0.012 | 0.172 | 0.316 | 0.841 |
| | | 800 | 600 | 0.001 | 0.035 | 0.247 | 0.755 |
| | 8192 | 400 | 591 | 0.401 | 1.000 | 0.973 | 1.000 |
| | | 800 | 600 | 0.056 | 0.572 | 0.957 | 1.000 |
| | | 1600 | 600 | 0.003 | 0.120 | 0.940 | 1.000 |
| | | 3200 | 600 | 0.000 | 0.030 | 0.920 | 1.000 |

Q = domande totali sui 3 seed. Il modello prevede **nessuna cella** in cui l'algebra
dia un vantaggio a pari bit: la catena paga p² dove lo store paga (c/N)², e la
composizione compilata è ancora un recupero a un salto, dove la preregistrazione 18
ha già visto lo store vincere.

## Ipotesi e criteri, fissati ora

Vantaggio = ABM misurato − store misurato (store di retrieval; in B quello scelto
sopra). SE = √(SE_ABM² + SE_store²), SE binomiale sulle Q domande della cella.

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — A, catena: nessun vantaggio di ABM a pari bit | in 16 celle su 16, vantaggio ≤ 2 SE | vantaggio > 2 SE in almeno una cella |
| **H2** — B, composizione compilata: nessun vantaggio di ABM a pari bit | in 16 celle su 16, vantaggio ≤ 2 SE | vantaggio > 2 SE in almeno una cella |
| **H3** — il modello esatto prevede ABM | errore medio assoluto ≤ 3 punti per (dataset, D, compito) | > 6 punti per uno qualsiasi |
| **H4** — C, tempo: lo store con join è più veloce | in ogni cella, t_ABM / t_store ≥ 100 | in almeno una cella t_ABM ≤ t_store |

Il resto è **sostenuto in parte**. Una falsificazione di H1 o H2 è il risultato
interessante per ABM: va riportata cella per cella, senza cambiare baseline dopo.

**Previsioni (non criteri).** H1, H2, H4 sostenute; H3 sostenuta per la catena, a
rischio per la compilata (il modello sulle triple sintetiche è una traduzione nuova,
mai validata; allo smoke test sintetico ha sbagliato di 2.6 punti). Se così, il
"valore residuo dell'algebra" non è un vantaggio di accuratezza né di velocità
rispetto a dict + join a pari bit, e il paper deve dirlo.

## Cosa ho visto prima

- Le previsioni qui sopra (`--predict`), che leggono i dati reali ma non misurano ABM.
- Lo smoke test su grafo sintetico (D = 1024, N ∈ {60, 120}, 2 seed): ABM catena
  0.583 / 0.140 (previsto 0.576 / 0.140), compilata 0.950 / 0.719 (previsto
  0.976 / 0.746), store sempre ≥; tempo × 1 000 circa. Nessuna misura ABM sui dati
  reali.

## Limiti dichiarati prima

- Lo store di retrieval perde un sottoinsieme **a caso** quando è sovraccarico; uno
  store che sapesse quali domande arriveranno farebbe meglio (a favore di ABM).
- La previsione della catena tratta i due hop come indipendenti; la dipendenza
  (preregistrazioni 8–9) vale meno di un punto a due hop.
- La compilata memorizza solo i percorsi domandati, per ABM e per lo store: misura
  la composizione, non la scoperta di percorsi.
- Il tempo è Python + numpy su una CPU, non il runtime bitpacked; il divario O(M·D)
  contro O(1) per hop è strutturale, il fattore esatto no.
- Non si misurano query composte più ricche (congiunzioni, analogie, legame di
  ruoli); se il valore dell'algebra sta lì, questo test non lo vede.
- Sotto ~0.01 di accuratezza (N grandi a D = 2048) ABM e store sono entrambi al
  pavimento: quelle celle non discriminano.

## Esito (2026-10-09)

Eseguito una volta, dopo il commit `e041f9a` della preregistrazione, con
`python examples/algebra_prereg.py` (≈3 min); risultati in
`results/algebra_prereg_results.json`.

| ipotesi | risultato | esito |
|---|---|---|
| **H1** — catena, nessun vantaggio ABM a pari bit | 0 celle su 16 con ABM > store + 2 SE | **sostenuta** |
| **H2** — compilata, nessun vantaggio ABM a pari bit | 0 celle su 16 | **sostenuta** |
| **H3** — il modello esatto prevede ABM | errore medio 0.32–2.01 punti per (dataset, D, compito); massimo 2.01 (WN18RR, D = 2048, catena) | **sostenuta** |
| **H4** — tempo | t_ABM / t_store ≥ 2600 in ogni cella | **sostenuta** |

### Cosa dice

- Nemmeno l'algebra salva ABM a pari bit: con la catena (unbinding a due hop) ABM
  sta sotto store + join in ogni cella; la composizione compilata si avvicina
  (0.92–0.99 a D = 8192) ma la tabella dei percorsi a pari bit resta a 1.0.
- Il modello esatto ha previsto anche questo entro 2 punti, compresa la nuova
  traduzione della composizione compilata (la parte più a rischio).
- In Python ABM è da 2600 a 230 000 volte più lento di un `dict` per domanda a due
  hop. Il tempo dello store è sotto la risoluzione stampata (0.0 s nel log): il
  rapporto è indicativo dell'ordine di grandezza, non una misura fine.
- Cosa resta a ABM: l'accuratezza prevedibile prima di costruire la memoria. Non la
  densità (test 18), non l'algebra su catene e composizioni (test 19). Restano non
  misurate le query più ricche (congiunzioni, analogie, ruoli) e il runtime bitpacked.
