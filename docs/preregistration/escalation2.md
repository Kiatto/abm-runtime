# Preregistrazione 13 — instradamento e indice di confidenza a parità di memoria

Committata **prima** di calcolare qualsiasi misura di questo test, insieme
all'harness `examples/escalation2_prereg.py`. Segue la preregistrazione 12
(`escalation.md`), dove il percorso grande vinceva quasi sempre perché rispondeva
con uno store esatto. Richiesta di kiatto (2026-10-04): verificare se si può
migliorare l'indice di confidenza e l'instradamento.

## Impianto

- Le stesse 643 domande di test e le stesse 28 memorie del test 11, a **D = 16 384**,
  per **entrambi** i percorsi. Piccolo: Gemma 4 E2B sceglie la relazione, la
  memoria risponde. Grande: Qwen3-4B sceglie la relazione, **la stessa memoria**
  risponde. L'escalation spende una chiamata al modello grande e può correggere
  solo l'errore del front-end, non quello della memoria.
- **Nessun modello viene interrogato di nuovo.** Le scelte di Gemma e di Qwen (e
  la confidenza di Gemma, probabilità del primo token) sono deterministiche e già
  registrate dalla preregistrazione 12. Rieseguirle darebbe gli stessi output.
- Domande forzate (il piccolo non ha una relazione): vanno al grande in A, sono le
  prime astensioni in B.

## Parte A — instradamento

Si passano al grande, alle quote 20/30/40/50%, le domande con il **valore** più alto:

| criterio | valore |
|---|---|
| **F** | 1 − confidenza del piccolo |
| **E** | p̂_mem × (1 − confidenza del piccolo), p̂_mem = `abm.exact` per entità collegata e relazione scelta dal piccolo |
| **R** | a caso, valore atteso |

L'idea di E: passare conviene se la memoria risponderà bene **e** il piccolo è
insicuro; se la memoria sbaglierà comunque, la chiamata al grande è sprecata.
Misura: media dell'accuratezza sulle quattro quote (AUC4).

## Parte B — indice di confidenza (rispondere o astenersi)

Solo percorso piccolo. Si risponde alle domande con il punteggio più alto, a
copertura 50/70/90%, e si misura l'accuratezza delle risposte date; riassunto: la
media sulle tre coperture.

| indice | punteggio |
|---|---|
| **F** | confidenza del piccolo |
| **CF** | p̂_mem × F |
| **ZF** | media dei ranghi di Z e di F; Z = (D/2 − d)/(√D/2), d la distanza di Hamming fra la query e il codeword restituito: un segnale **osservato** sulla singola risposta, non previsto |

## Ipotesi e criteri, fissati ora

Intervalli al 95% da un bootstrap per cluster sulle 28 memorie (10 000 estrazioni,
seed 20261004).

| ipotesi | sostenuta | falsificata |
|---|---|---|
| **H1** — A: E batte F | intervallo tutto sopra 0 | stima ≤ 0 |
| **H2** — A: E batte R | intervallo tutto sopra 0 | stima ≤ 0 |
| **H3** — B: ZF batte CF (il margine osservato migliora l'indice) | intervallo tutto sopra 0 | stima ≤ 0 |
| **H4** — B: CF batte F (il contratto migliora la sola confidenza) | intervallo tutto sopra 0 | stima ≤ 0 |

Il resto è **sostenuto in parte**.

**Previsioni (non criteri).** H2 e H4 sostenute. H1 incerta: E evita di spendere
chiamate sulle domande che la memoria sbaglierebbe comunque, ma a D = 16 384 la
memoria è giusta l'80% delle volte e c'è poco da evitare. H3 sostenuta: la distanza
osservata separa le risposte vere dagli alias meglio della previsione media.

## Cosa ho visto prima

- Dal test 11: a D = 16 384 il percorso piccolo è giusto nel 36.1% delle domande e
  la memoria, data la coppia vera, nell'80.7%; so per ogni domanda se Gemma ha
  scelto la relazione giusta.
- Dalla preregistrazione 12: l'accuratezza aggregata di Qwen con lo store esatto
  (0.871), quindi approssimativamente quella del suo front-end.
- **Non** ho guardato le confidenze di Gemma domanda per domanda, né p̂_mem a
  D = 16 384 per le relazioni scelte, né alcuna distanza di Hamming.

## Limiti dichiarati prima

- Il "grande" è Qwen3-4B, non un modello grande vero; il costo è la quota di
  chiamate, non tempo o denaro.
- Coperture e quote sono poche e fisse; 28 cluster.
- ZF combina per ranghi, senza pesi stimati: una combinazione appresa potrebbe fare
  meglio, ma andrebbe stimata su dati separati.

---

## Esito — 2026-10-04, eseguito alle 9:20 dopo il commit `cf65a56`

Ipotesi e criteri **non modificati**. Risultati:
[`results/escalation2_prereg_results.json`](../../results/escalation2_prereg_results.json).

Accuratezza dei percorsi da soli, stessa memoria: **piccolo 0.364**, **grande
0.695**. Il grande ha ragione dove il piccolo sbaglia in 225 domande, il contrario
in 12: anche a parità di memoria il front-end di Qwen è molto migliore di quello di
Gemma.

**A — instradamento** (accuratezza del sistema)

| quota | F | E | R |
|---|---|---|---|
| 20% | 0.420 | 0.415 | 0.413 |
| 30% | 0.462 | 0.462 | 0.448 |
| 40% | 0.502 | 0.518 | 0.483 |
| 50% | 0.547 | 0.575 | 0.519 |

**B — indice di confidenza** (accuratezza delle risposte date)

| copertura | F | CF | ZF |
|---|---|---|---|
| 50% | 0.491 | 0.503 | 0.531 |
| 70% | 0.436 | 0.458 | 0.458 |
| 90% | 0.404 | 0.404 | 0.404 |

| ipotesi | stima (punti) | 95%, cluster | esito |
|---|---|---|---|
| **H1** — E contro F | +1.0 | [0.0, +2.0] | **sostenuta in parte**: il limite inferiore è 0, non sopra |
| **H2** — E contro R | +2.7 | [+1.5, +3.8] | **sostenuta** |
| **H3** — ZF contro CF | +0.9 | [−0.3, +2.1] | **sostenuta in parte** |
| **H4** — CF contro F | +1.2 | [−0.2, +2.5] | **sostenuta in parte** |

**Difetto del disegno, non previsto.** Al 90% di copertura si escludono 64 domande
e le forzate sono 68: le escluse sono tutte forzate con ogni indice, e i tre
indici coincidono per costruzione. Quel punto non può distinguere nulla e dimezza
le differenze medie di H3 e H4. Non ricalcolo: le medie sulle sole coperture 50% e
70% sarebbero un'analisi dopo aver visto i dati.

### Cosa dice, e cosa no

- Il criterio di valore E instrada meglio del caso, e meglio della sola confidenza
  alle quote alte (+1.6 e +2.8 punti al 40% e 50%), non a quelle basse. In media il
  vantaggio sulla confidenza non esce dal rumore.
- Il margine osservato e il contratto spostano l'accuratezza delle risposte date
  nella direzione prevista (al 50% di copertura: 0.491 → 0.503 → 0.531), ma con 28
  cluster e un punto di copertura inutile nessuno dei due esce dal rumore.
- Il limite vero è il front-end piccolo: Gemma sceglie la relazione giusta solo nel
  41% dei casi, e la confidenza del suo primo token ne coglie poco. Gli indici
  della memoria possono migliorare solo la parte d'errore che è della memoria, il
  20% a D = 16 384.
