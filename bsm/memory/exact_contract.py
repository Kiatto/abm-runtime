"""exact_contract.py — accuratezza di cleanup calcolata a D finito, senza costanti.

La Law IV (reference/abm.py) predice l'accuratezza con due approssimazioni
asintotiche: il segnale di un membro è gaussiano con z = sqrt(2D/(pi N)), e il
minimo di M distanze nulle segue una Gumbel del secondo ordine. Le due insieme
lasciano un errore sistematico, che la costante k = 0.92 ± 0.03 misurava.

Qui le due approssimazioni sono sostituite dalle distribuzioni esatte che
seguono dagli assiomi A1–A2:

- un bit della query concorda con il codeword di un oggetto memorizzato con
  probabilità esatta p_agree(N): è la probabilità che la maggioranza di N fatti
  ±1 indipendenti concordi con uno di essi (pareggi divisi a metà, come il
  tie-break della reference). La distanza dal codeword giusto è quindi
  Binomial(D, 1 - p_agree(N));
- la distanza da un codeword non coinvolto è Binomial(D, 1/2).

Nessun parametro. Un'unica ipotesi non esatta, dichiarata: quando una query ha
più candidati a pari segnale (più oggetti veri, o alias per la simmetria s/o),
le loro distanze sono trattate come indipendenti. Tra codeword diversi il
legame passa solo attraverso i bit in cui i loro fatti coincidono, ed è piccolo;
va comunque misurato, non assunto (vedi docs/preregistration/exact_contract.md).

Solo numpy, come il resto del runtime.
"""
from __future__ import annotations

import numpy as np


def _log_binom_pmf(dim: int, p: float) -> np.ndarray:
    """log P(X = d), d = 0..dim, per X ~ Binomial(dim, p), senza overflow."""
    d = np.arange(dim + 1, dtype=np.float64)
    logc = np.concatenate(([0.0], np.cumsum(np.log(np.arange(dim, 0, -1, dtype=np.float64))
                                            - np.log(np.arange(1, dim + 1, dtype=np.float64)))))
    with np.errstate(divide="ignore"):
        lp, lq = np.log(p), np.log1p(-p)
    return logc + d * lp + (dim - d) * lq


def binom_pmf(dim: int, p: float) -> np.ndarray:
    return np.exp(_log_binom_pmf(dim, p))


def p_agree(n_facts: int) -> float:
    """P(un bit della maggioranza di n fatti concorda con uno di essi), esatta.

    Condizionando sul bit del fatto (+1), gli altri n-1 sono Rademacher
    indipendenti; la maggioranza è +1 se la somma è > 0, e un pareggio (solo per
    n pari) vale 1/2.
    """
    if n_facts < 1:
        raise ValueError("serve almeno un fatto")
    m = n_facts - 1
    k = np.arange(m + 1)
    pk = binom_pmf(m, 0.5)
    total = 1 + 2 * k - m
    return float(np.sum(pk * np.where(total > 0, 1.0, np.where(total == 0, 0.5, 0.0))))


def cleanup_accuracy(n_facts: int, dim: int, codebook: int,
                     correct: int = 1, aliases: int = 0) -> float:
    """P(il cleanup restituisce uno dei `correct` oggetti veri).

    `codebook` è il numero totale di codeword su cui compete la query. Dei
    candidati, `correct + aliases` hanno il segnale di un fatto memorizzato; gli
    altri sono nulli. Per simmetria, se vince un candidato a segnale, è uno degli
    oggetti veri con probabilità correct / (correct + aliases).
    """
    g = correct + aliases
    if g < 1 or codebook < g:
        raise ValueError("servono 1 <= correct + aliases <= codebook")
    q = 1.0 - p_agree(n_facts)
    sig = binom_pmf(dim, q)
    null = binom_pmf(dim, 0.5)
    sig_cdf = np.cumsum(sig)
    null_sf = 1.0 - np.cumsum(null)          # P(null > d)
    null_sf = np.clip(null_sf, 0.0, 1.0)
    # P(min dei g segnali == d)
    above = np.clip(1.0 - sig_cdf, 0.0, 1.0)  # P(segnale > d)
    above_prev = np.concatenate(([1.0], above[:-1]))
    p_min = above_prev ** g - above ** g
    n_null = codebook - g
    # vittoria stretta sulle nulle, più pareggi con una nulla divisi a metà
    win = null_sf ** n_null
    if n_null > 0:
        win = win + 0.5 * n_null * null * null_sf ** (n_null - 1)
    p_signal_wins = float(np.sum(p_min * np.clip(win, 0.0, 1.0)))
    return p_signal_wins * correct / g


def capacity(dim: int, codebook_of_n=lambda n: 2 * n + 11,
             lo: int = 2, hi: int = 1_000_000) -> float:
    """N* esatto: il carico a cui l'accuratezza di cleanup scende al 50%.

    `codebook_of_n` dà il codebook in funzione del carico, perché nei
    benchmark del paper cresce con N (M = 2N + 11).
    """
    for _ in range(60):
        mid = (lo + hi) / 2
        n = max(int(round(mid)), 1)
        if cleanup_accuracy(n, dim, codebook_of_n(n)) > 0.5:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Due hop sulla stessa traccia: la dipendenza esatta, al posto dell'indipendenza
# ---------------------------------------------------------------------------
#
# La Law V (Acc(h) = p^h) assume che i successi di hop diversi sulla stessa
# traccia siano indipendenti. Non lo sono. Per ogni bit, con T la traccia e f1, f2
# i fatti interrogati dai due hop,
#
#     (T·f1)·(T·f2) = f1·f2      perché T² = 1,
#
# e f1·f2 è un bit uniforme indipendente dalla maggioranza. Quindi l'accordo della
# query con i due bersagli è correlato: Cov = -rho², con rho = 2·p_agree(N) - 1.
# La distribuzione congiunta per bit è esatta, e dà quella esatta delle due
# distanze; le distanze nulle dei due hop si trattano come indipendenti (la loro
# correlazione è una somma di segni casuali, di ordine 1/sqrt(D)).


def _agree_given(n_facts: int, same: bool) -> float:
    """P(la maggioranza concorda con f1), dato che f2 = f1 (same) o f2 = -f1."""
    m = n_facts - 2                      # gli altri fatti
    k = np.arange(m + 1)
    pk = binom_pmf(m, 0.5) if m > 0 else np.array([1.0])
    rest = 2 * k - m
    total = rest + (2 if same else 0)    # f1 (+1) e f2 (+1 o -1)
    return float(np.sum(pk * np.where(total > 0, 1.0, np.where(total == 0, 0.5, 0.0))))


def bit_correlation(n_facts: int) -> float:
    """Correlazione esatta, per bit, fra l'accordo della query con due fatti."""
    rho = 2 * p_agree(n_facts) - 1
    return -rho * rho / (1 - rho * rho)


def two_hop_joint(n_facts: int, dim: int, codebook: int):
    """(p, P(entrambi i hop corretti)) per due fatti della stessa traccia.

    Restituisce anche p², cioè la previsione della Law V, per il confronto.
    """
    if n_facts < 2:
        raise ValueError("servono almeno due fatti")
    u = _agree_given(n_facts, same=True)     # f1 = f2: la traccia concorda con entrambi o con nessuno
    v = _agree_given(n_facts, same=False)    # f1 = -f2: concorda con esattamente uno
    # per bit, P(accordo1, accordo2): (+,+), (+,-), (-,+), (-,-)
    p_pp, p_mm = 0.5 * u, 0.5 * (1 - u)
    p_pm, p_mp = 0.5 * v, 0.5 * (1 - v)
    # distanza = numero di bit in disaccordo; distribuzione congiunta (d1, d2)
    # per convoluzione sui D bit, nella base (x, y) = (disaccordo1, disaccordo2)
    joint = np.zeros((dim + 1, dim + 1))
    joint[0, 0] = 1.0
    step = {(0, 0): p_pp, (0, 1): p_pm, (1, 0): p_mp, (1, 1): p_mm}
    for _ in range(dim):
        new = np.zeros_like(joint)
        for (dx, dy), pr in step.items():
            new[dx:, dy:] += pr * joint[:dim + 1 - dx, :dim + 1 - dy]
        joint = new
    null = binom_pmf(dim, 0.5)
    null_sf = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)
    n_null = codebook - 1
    win = null_sf ** n_null
    if n_null > 0:
        win = win + 0.5 * n_null * null * null_sf ** (n_null - 1)
    win = np.clip(win, 0.0, 1.0)
    both = float(win @ joint @ win)
    p = float(np.sum(joint.sum(axis=1) * win))
    return p, both, p * p
