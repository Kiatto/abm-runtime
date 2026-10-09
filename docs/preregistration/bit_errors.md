# Preregistrazione 20 — ABM contro store esatti con codice correttore, sotto errori di bit

**Data: 2026-10-09.** Committata **prima** di misurare, insieme all'harness
[`examples/bit_errors_prereg.py`](../../examples/bit_errors_prereg.py). Prima del
commit l'harness è stato eseguito solo con `--smoke` (grafo sintetico generato nello
script, mai FB15k-237 o WN18RR) e con `--predict` (legge i dati reali, costruisce le
strutture degli store per sapere quante chiavi tengono e ne simula il rumore per la
previsione dei codici di Hamming; **non** costruisce nessuna memoria ABM sui dati
reali). Stesse regole di [`fb15k237.md`](fb15k237.md).

## Perché

I test 18 ([`equal_bits.md`](equal_bits.md)) e 19 ([`algebra.md`](algebra.md)) hanno
mostrato che a pari bit store esatto, Bloom e dict + join battono ABM ovunque, su
memoria **perfetta**. Resta non testato il perimetro hardware: memorie che sbagliano
bit durante la conservazione (memorie analogiche, in-memory computing, celle
multilivello). L'argomento classico per le rappresentazioni distribuite è che
degradano con grazia; uno store esatto, invece, perde la chiave intera a un solo bit
sbagliato, a meno di spendere bit in un codice correttore. Qui si misura chi vince, a
pari bit totali, quando il codice lo sceglie l'avversario al meglio.

## Impianto (ostile ad ABM)

Grafi: FB15k-237 e WN18RR, train split, stessi file e sha256 di
`examples/replicate.py` (già in `data/external/`; self-loop esclusi). Per ogni
(dataset, D, N, seed): N fatti distinti campionati uniformemente; domande = chiavi
(s, r) distinte del campione, al più 300 per seed; giusta se la risposta è uno degli
oggetti veri di (s, r) nel campione.

Griglia: D = 2048 con N ∈ {50, 100, 200, 400}; D = 8192 con N ∈ {200, 400, 800, 1600};
ε ∈ {0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.2}; 5 seed. 16 serie × 7 ε = **112 celle**.

**Modello d'errore (principale):** ogni bit conservato si inverte indipendentemente con
probabilità ε, una volta per memoria (rumore di conservazione, non di lettura); tutte
le domande leggono la stessa memoria rumorosa.

- **ABM:** una traccia di D bit (reference congelata), maschera di flip sulla traccia.
  Il codebook (item memory) è **senza rumore**: si rigenera dai nomi (vedi limiti).
- **Store (a), senza protezione:** funzione statica a chiavi implicite, XOR filter a 3
  segmenti costruito per peeling: ⌈1.23·c⌉ celle (multiplo di 3) di
  le = ⌈log₂ #oggetti⌉ bit per c chiavi; lettura = XOR di 3 celle. Stesso 1.23 del
  test 19. Se il peeling fallisce con 20 semi di hash, c scende di 1. Se D non basta,
  tiene un sottoinsieme a caso delle chiavi; una chiave non tenuta è contata sbagliata.
  Il flip colpisce tutti i D bit (o meno) occupati.
- **Store (b), con codice:** le stesse celle serializzate in bit passano per un codice
  che spende parte di D: ripetizione r ∈ {3, 5, 7, 9, 11, 15, 21} (copie interlacciate
  a distanza L, decodifica a maggioranza) oppure Hamming esteso SECDED (2^m, 2^m−m−1),
  m ∈ {3, …, 8} (corregge 1 errore per blocco, con 2 rilevati lascia i dati come sono).
  Il codice riduce c: c = il massimo che entra in D bit codificati.
- **(c) Bloom:** non pertinente: risponde all'appartenenza, non a (s, r) → o.
  Non incluso.

**Regola di scelta dell'avversario, fissata ora.** Per ogni (dataset, D, N, seed, ε)
si sceglie fra `none`, `rep*`, `ham*` il codice con l'accuratezza **prevista** più
alta (pari: il primo nell'ordine `CODES`); il compromesso ridondanza/capacità è quindi
ottimizzato a priori per ogni ε, carico e D. Previsione:
- none e ripetizione, **esatta**: una lettura è giusta se, per ognuno degli le bit,
  il numero di errori fra le 3 celle è pari: P = (1 − o₃(e))^le con
  o₃(e) = (1 − (1−2e)³)/2, e = ε (none) o e = P(Bin(r, ε) > r/2) (ripetizione);
  accuratezza = media sulle domande di P·[chiave tenuta].
- Hamming: il limite analitico P(ogni blocco toccato ha ≤ 1 errore) è lasco (allo
  smoke sottostima fino a 90 punti); per la scelta si usa un **Monte Carlo del solo
  store** (20 rumori da un flusso di numeri casuali separato da quello della misura).
  Il limite è riportato.
Il vantaggio di ABM si misura contro il miglior avversario così scelto; si misura e
riporta sempre anche `none`.

**Descrittivo, non realizzato:** lo store con un codice ideale alla capacità del
canale binario simmetrico, c = ⌊D·(1 − h(ε)) / (1.23·le)⌋. È il tetto di qualunque
codice (BCH, LDPC, polar), non un avversario misurato.

**Descrittivo, raffiche:** a ε ∈ {1e-2, 3e-2}, flip a raffiche di 16 bit consecutivi
con la stessa frequenza media di bit invertiti, su ABM e sul codice scelto per l'iid.
Nessuna ipotesi.

## Modello esatto di ABM con rumore

Nel modello esatto un candidato a segnale ha p = P(un bit della traccia concorda con
il fatto) (`exact.p_agree_weighted`, gemelli e alias inclusi). Un flip indipendente
con probabilità ε dà

  p′ = p(1 − ε) + (1 − p)ε = ε + p(1 − 2ε),

mentre un codeword nullo resta Binomial(D, ½) (un bit uniforme XOR un flip
indipendente resta uniforme). Quindi basta sostituire p con p′ in
`cleanup_accuracy_mixed`; il resto del modello è quello della v1.1.0
(`abm_predict` nell'harness). Il segnale si contrae di (1 − 2ε): ABM non ha soglia, la
sua accuratezza scende con continuità; lo store senza codice perde ogni chiave con
probabilità ≈ 3·le·ε.

Verifica allo smoke (sintetico, D = 1024, N ∈ {40, 120}, 3 seed): p′ contro Monte
Carlo sui bit 0.5800/0.5795 e 0.5300/0.5302; accuratezza ABM prevista contro misurata,
errore medio 2.6 punti, massimo 7.2 (N = 40, ε = 0.2: 0.491 contro 0.563 ± 0.045,
1.6 SE). Lo store `none` e `rep*` previsto contro simulato concorda entro il rumore
di campionamento.

## Previsioni, calcolate ora (`--predict`)

ABM previsto / miglior avversario previsto, media sui 5 seed. In **grassetto** le
celle in cui il modello prevede ABM sopra l'avversario. MDE = bias minimo rilevabile
per cella (differenza vera che dà vantaggio > 2 SE con potenza 80%:
2.84·SE della differenza, SE binomiale sulle Q domande delle 5 seed).

| dataset | D | N | MDE | ε=0 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fb15k237 | 2048 | 50 | 0.127 | 0.986 / 1.000 | 0.986 / 1.000 | 0.986 / 1.000 | 0.983 / 1.000 | 0.975 / 0.995 | **0.910 / 0.858** | **0.661 / 0.435** |
| fb15k237 | 2048 | 100 | 0.090 | 0.773 / 1.000 | 0.772 / 1.000 | 0.771 / 1.000 | 0.752 / 0.990 | 0.708 / 0.910 | **0.531 / 0.439** | **0.275 / 0.160** |
| fb15k237 | 2048 | 200 | 0.064 | 0.337 / 1.000 | 0.337 / 1.000 | 0.335 / 0.994 | 0.320 / 0.833 | 0.286 / 0.576 | 0.183 / 0.187 | **0.083 / 0.066** |
| fb15k237 | 2048 | 400 | 0.052 | 0.098 / 0.465 | 0.098 / 0.463 | 0.097 / 0.452 | 0.092 / 0.367 | 0.081 / 0.251 | 0.051 / 0.076 | 0.024 / 0.028 |
| fb15k237 | 8192 | 200 | 0.064 | 0.974 / 1.000 | 0.974 / 1.000 | 0.973 / 1.000 | 0.968 / 0.993 | **0.952 / 0.939** | **0.844 / 0.679** | **0.524 / 0.287** |
| fb15k237 | 8192 | 400 | 0.052 | 0.662 / 1.000 | 0.662 / 1.000 | 0.660 / 0.999 | 0.637 / 0.969 | 0.585 / 0.831 | **0.397 / 0.307** | **0.172 / 0.126** |
| fb15k237 | 8192 | 800 | 0.052 | 0.233 / 0.868 | 0.233 / 0.865 | 0.232 / 0.842 | 0.218 / 0.676 | 0.191 / 0.459 | 0.112 / 0.140 | 0.044 / 0.059 |
| fb15k237 | 8192 | 1600 | 0.052 | 0.057 / 0.411 | 0.057 / 0.410 | 0.057 / 0.399 | 0.053 / 0.327 | 0.046 / 0.228 | 0.027 / 0.060 | 0.011 / 0.027 |
| wn18rr | 2048 | 50 | 0.127 | 0.990 / 1.000 | 0.990 / 1.000 | 0.990 / 1.000 | 0.987 / 1.000 | 0.980 / 0.995 | **0.922 / 0.858** | **0.688 / 0.435** |
| wn18rr | 2048 | 100 | 0.090 | 0.785 / 1.000 | 0.785 / 1.000 | 0.783 / 1.000 | 0.765 / 0.987 | 0.722 / 0.900 | **0.549 / 0.439** | **0.293 / 0.160** |
| wn18rr | 2048 | 200 | 0.064 | 0.350 / 1.000 | 0.350 / 1.000 | 0.348 / 0.992 | 0.332 / 0.828 | 0.298 / 0.572 | **0.193 / 0.192** | **0.089 / 0.067** |
| wn18rr | 2048 | 400 | 0.052 | 0.100 / 0.459 | 0.100 / 0.457 | 0.099 / 0.447 | 0.094 / 0.362 | 0.083 / 0.249 | 0.052 / 0.072 | 0.024 / 0.026 |
| wn18rr | 8192 | 200 | 0.064 | 0.975 / 1.000 | 0.975 / 1.000 | 0.974 / 1.000 | 0.969 / 0.993 | **0.954 / 0.939** | **0.851 / 0.676** | **0.537 / 0.285** |
| wn18rr | 8192 | 400 | 0.052 | 0.661 / 1.000 | 0.661 / 1.000 | 0.659 / 1.000 | 0.636 / 0.968 | 0.584 / 0.818 | **0.397 / 0.301** | **0.172 / 0.123** |
| wn18rr | 8192 | 800 | 0.052 | 0.224 / 0.849 | 0.224 / 0.847 | 0.223 / 0.824 | 0.210 / 0.660 | 0.183 / 0.454 | 0.107 / 0.141 | 0.042 / 0.059 |
| wn18rr | 8192 | 1600 | 0.052 | 0.051 / 0.389 | 0.051 / 0.387 | 0.051 / 0.376 | 0.047 / 0.304 | 0.041 / 0.213 | 0.024 / 0.059 | 0.010 / 0.027 |

Codice scelto dalla regola (fra parentesi lo store a codice ideale, non realizzato):

| dataset | D | N | ε=0 | 1e-4 | 1e-3 | 1e-2 | 3e-2 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| fb15k237 | 2048 | 50 | none (1.000) | ham3 (1.000) | ham3,ham4 (1.000) | rep5 (1.000) | rep5 (1.000) | rep5 (1.000) | rep7 (1.000) |
| fb15k237 | 2048 | 100 | none (1.000) | ham3 (1.000) | ham3,ham4 (1.000) | ham3,ham4 (1.000) | ham3 (1.000) | rep3 (1.000) | rep7,rep9 (0.663) |
| fb15k237 | 2048 | 200 | none (1.000) | ham8 (1.000) | ham8 (1.000) | ham6,ham7 (0.958) | ham4,ham5 (0.838) | ham3,rep3 (0.552) | rep7,rep9 (0.286) |
| fb15k237 | 2048 | 400 | none (0.473) | none (0.470) | ham8,none (0.465) | ham6,ham7,none (0.435) | ham4 (0.381) | ham3,rep3 (0.251) | rep15,rep9 (0.130) |
| fb15k237 | 8192 | 200 | none (1.000) | ham3 (1.000) | ham3 (1.000) | ham3,rep3 (1.000) | rep3 (1.000) | rep5 (1.000) | rep11 (1.000) |
| fb15k237 | 8192 | 400 | none (1.000) | ham4 (1.000) | ham4,ham5 (1.000) | ham4 (1.000) | ham3 (1.000) | ham3,rep3,rep5 (0.997) | rep11,rep9 (0.524) |
| fb15k237 | 8192 | 800 | none (0.864) | none (0.863) | none (0.854) | ham5,ham6 (0.794) | ham4 (0.695) | rep3,rep5 (0.458) | rep11,rep9 (0.240) |
| fb15k237 | 8192 | 1600 | none (0.407) | none (0.407) | ham8,none (0.403) | ham5,ham6 (0.374) | ham4 (0.328) | ham3,ham4,rep5,rep9 (0.216) | rep11,rep15,rep9 (0.113) |
| wn18rr | 2048 | 50 | none (1.000) | ham3 (1.000) | ham3 (1.000) | rep5 (1.000) | rep5 (1.000) | rep5 (1.000) | rep7 (1.000) |
| wn18rr | 2048 | 100 | none (1.000) | ham3 (1.000) | ham3 (1.000) | ham3 (1.000) | ham3 (1.000) | rep3 (1.000) | rep9 (0.663) |
| wn18rr | 2048 | 200 | none (1.000) | ham8 (1.000) | ham8 (1.000) | ham6 (0.962) | ham4 (0.841) | ham3 (0.554) | rep7 (0.287) |
| wn18rr | 2048 | 400 | none (0.466) | none (0.463) | ham8,none (0.458) | ham6,ham7 (0.428) | ham4,ham5 (0.375) | rep3,rep5 (0.247) | rep9 (0.128) |
| wn18rr | 8192 | 200 | none (1.000) | ham3 (1.000) | ham3 (1.000) | rep3 (1.000) | rep3 (1.000) | rep5 (1.000) | rep11 (1.000) |
| wn18rr | 8192 | 400 | none (1.000) | ham4 (1.000) | ham4 (1.000) | ham4 (1.000) | ham3 (1.000) | ham3,rep3,rep5 (0.986) | rep11,rep9 (0.516) |
| wn18rr | 8192 | 800 | none (0.847) | none (0.846) | none (0.837) | ham6,none (0.779) | ham4 (0.682) | rep3,rep5 (0.449) | rep11,rep15,rep9 (0.235) |
| wn18rr | 8192 | 1600 | none (0.387) | none (0.386) | none (0.382) | ham5,ham6 (0.356) | ham4 (0.311) | ham3,rep3,rep5 (0.205) | rep11,rep15,rep9 (0.107) |

(Più codici in una cella: la regola ha scelto codici diversi in seed diversi.)

**ε\* previsto** (primo ε della griglia con ABM previsto > miglior avversario previsto):

| serie | FB15k-237 | WN18RR |
|---|---|---|
| D = 2048, N = 50 | 0.1 | 0.1 |
| D = 2048, N = 100 | 0.1 | 0.1 |
| D = 2048, N = 200 | 0.2 | 0.1 (margine 0.1 punti: di fatto 0.1–0.2) |
| D = 2048, N = 400 | mai | mai |
| D = 8192, N = 200 | 0.03 (margine 1.3 punti, sotto l'MDE) | 0.03 (1.5 punti) |
| D = 8192, N = 400 | 0.1 | 0.1 |
| D = 8192, N = 800 | mai | mai |
| D = 8192, N = 1600 | mai | mai |

Lettura: il modello prevede che ABM **vinca** contro gli store con codici semplici a
carichi bassi (dove ABM a ε = 0 è sopra ~0.6) e rumore alto (ε ≥ 0.1, a volte 0.03),
e **perda** a carichi alti, dove lo store anche rumoroso tiene più chiavi di quante
ABM ne recuperi. 21 celle su 112 previste a favore di ABM; in 12 (6 per dataset) il margine previsto
supera l'MDE: D = 2048, N = 50, ε = 0.2; N = 100, ε = 0.1 (FB15k-237 al limite: 9.2
contro 9.0 punti) e 0.2; D = 8192, N = 200, ε = 0.1 e 0.2; N = 400, ε = 0.1. Il modello prevede anche che **lo store a codice ideale
batta ABM in ogni cella**: il vantaggio previsto è contro codici realizzabili
semplici (ripetizione, Hamming), non contro la teoria dei codici.

## Ipotesi e criteri, fissati ora

Vantaggio = ABM misurato − miglior avversario misurato (scelto con la regola sopra).
SE di ciascuno = il massimo fra SE binomiale sulle Q domande della cella e SD fra i 5
seed / √5 (il rumore è unico per memoria: le domande non sono indipendenti);
SE del vantaggio = √(SE_ABM² + SE_avv²).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — esiste un ε in cui ABM batte il miglior avversario (direzionale) | almeno una cella con vantaggio > 2 SE **e** ABM > avversario in tutti i 5 seed | nessuna cella così |
| **H2** — il modello esatto prevede ABM sotto rumore | errore medio assoluto ≤ 3 punti in ogni gruppo (dataset × D, 28 celle) | > 5 punti in un gruppo |
| **H3** — ε\* previsto vs osservato | ε\* osservato (primo ε con ABM misurato > avversario misurato; "mai" se nessuno) entro un passo di griglia da quello previsto in ≥ 12 serie su 16 | in < 8 serie su 16 |
| **H4** — il modello dell'avversario prevede lo store scelto | errore medio assoluto ≤ 3 punti per gruppo | > 5 punti in un gruppo |
| **H0** — controllo: a ε = 0 ABM perde ovunque (repliche dei test 18–19) | ABM < avversario in 16 serie su 16 | ABM > avversario + 2 SE in una serie |

Il resto è **sostenuto in parte**. Si riportano sempre: il massimo errore di cella di
H2; quante celle di H1 superano anche 3.5 SE (≈ Bonferroni one-sided su 112 celle);
il numero atteso di falsi positivi a 2 SE senza correzione (≈ 2.5 su 112, ragione del
requisito "in tutti i seed").

**Previsioni (non criteri).** H1 sostenuta (12 celle con margine previsto sopra l'MDE), H0
sostenuta, H2 sostenuta ma a rischio a ε = 0.2 (allo smoke l'errore massimo è stato
7.2 punti lì), H3 sostenuta, H4 sostenuta. Se H1 è sostenuta, la conclusione
ammessa è stretta: a pari bit, sotto rumore iid ε ≳ 0.03–0.1 e a carico basso, ABM
batte store esatti protetti da ripetizione o Hamming; **non** batte uno store con un
codice vicino alla capacità, che il modello prevede superiore ovunque.

## Bias minimo rilevabile

Per cella, l'MDE nella tabella delle previsioni: 12.7 punti a N = 50 (Q ≈ 250),
9.0 a N = 100, 6.4 a N = 200, 5.2 sopra (Q = 1500). Le vittorie previste a ε = 0.03
(1.3–1.5 punti) e a D = 2048, N = 200 o 400 (≤ 2 punti) **non** sono rilevabili con
questa potenza: un esito nullo lì non le smentisce.

## Tempo stimato

`--predict` ha richiesto 37 s. Il run completo aggiunge, per ognuna delle 160
memorie, la traccia ABM e 9 cleanup vettoriali (7 ε + 2 raffiche): stima 3–8 minuti
su una CPU.

## Cosa ho visto prima

- Le previsioni qui sopra (`--predict`), che leggono i dati reali ma non misurano ABM.
- Lo smoke test sintetico (D = 1024, N ∈ {40, 120}, 3 seed): ABM batte l'avversario a
  ε = 0.2 (+0.36 ± 0.07 a N = 40; +0.06 ± 0.02 a N = 120), lo perde a ε ≤ 0.03.
  Nessuna misura ABM sui dati reali.
- Durante lo sviluppo la regola di scelta usava il limite analitico di Hamming; allo
  smoke era lasco (fino a 90 punti) e penalizzava l'avversario: sostituito con il
  Monte Carlo dello store **prima** di eseguire `--predict` e prima di ogni dato reale.

## Limiti dichiarati prima

- **Codici semplici.** L'avversario usa ripetizione e Hamming SECDED, non BCH, LDPC o
  polar. Un codice moderno si avvicina allo store ideale riportato, che il modello
  prevede sopra ABM in ogni cella. Un'eventuale vittoria di ABM va letta come "contro
  i codici implementati qui", e il divario con il tetto va riportato insieme.
- **Codebook senza errori.** I codeword ABM si rigenerano dai nomi; lo store ha
  bisogno degli stessi nomi (anch'essi senza errori). Errori nell'item memory non sono
  modellati: in un dispositivo reale anche quella sarebbe in memoria.
- **Rumore iid e unico per memoria.** Nessun errore di lettura, nessuna deriva nel
  tempo, nessuna correlazione spaziale tranne il descrittivo a raffiche. Le raffiche
  non entrano in nessuna ipotesi; l'Hamming senza interlacciamento le soffre per
  costruzione, ABM no (i bit della traccia sono scambiabili).
- **Store a funzione statica senza verifica.** Su una chiave persa o corrotta risponde
  un valore arbitrario (≈ 1/#oggetti di essere giusto, ignorato nella previsione, non
  nella misura). Uno store che rilevasse l'errore (CRC) potrebbe astenersi, ma qui
  conta solo l'accuratezza.
- **Campione uniforme di fatti**, non sottografi connessi: pochi gemelli e alias. Il
  modello esatto li gestisce comunque.
- **Pochi seed per il rumore.** Il rumore è un'estrazione per memoria; la SE usa il
  massimo fra binomiale e varianza fra seed, ma con 5 seed quest'ultima è grezza.
- **Nessuna combinazione di codici:** la regola sceglie il codice per (cella, seed, ε),
  ma non combina codici (es. ripetizione + Hamming); questo può lasciare qualche punto
  all'avversario, a favore di ABM.
