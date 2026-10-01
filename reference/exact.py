"""abm.exact — accuratezza di cleanup calcolata a D finito, senza costanti.

(Fino alla v1.6 del paper: bsm/memory/exact_contract.py, che ora rinvia qui.)

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

from functools import lru_cache

import numpy as np


def _log_binom_pmf(dim: int, p: float) -> np.ndarray:
    """log P(X = d), d = 0..dim, per X ~ Binomial(dim, p), senza overflow."""
    d = np.arange(dim + 1, dtype=np.float64)
    logc = np.concatenate(([0.0], np.cumsum(np.log(np.arange(dim, 0, -1, dtype=np.float64))
                                            - np.log(np.arange(1, dim + 1, dtype=np.float64)))))
    if p <= 0.0 or p >= 1.0:                 # distribuzione degenere: evita 0·log 0 = nan
        out = np.full(dim + 1, -np.inf)
        out[0 if p <= 0.0 else dim] = 0.0
        return out
    return logc + d * np.log(p) + (dim - d) * np.log1p(-p)


def binom_pmf(dim: int, p: float) -> np.ndarray:
    return np.exp(_log_binom_pmf(dim, p))


def p_agree(n_facts: int) -> float:
    """P(un bit della maggioranza di n fatti concorda con uno di essi), esatta.

    Condizionando sul bit del fatto (+1), gli altri n-1 sono Rademacher
    indipendenti; la maggioranza è +1 se la somma è > 0, e un pareggio (solo per
    n pari) vale 1/2.
    """
    if n_facts < 1:
        raise ValueError("p_agree needs at least one stored fact (n_facts >= 1)")
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
    if correct < 0 or aliases < 0 or codebook < g:
        raise ValueError("cleanup_accuracy needs correct, aliases >= 0 and "
                         "correct + aliases <= codebook")
    if correct == 0:                      # unanswerable or alias-only: never correct
        return 0.0
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
        raise ValueError("two_hop_joint needs at least two stored facts")
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


# ---------------------------------------------------------------------------
# Fatti con peso: la Law VII in forma esatta, e i gemelli simmetrici
# ---------------------------------------------------------------------------
#
# Un fatto scritto w volte pesa w nel voto di maggioranza (Law VII). Succede anche
# senza volerlo: per l'encoding s ⊕ ρ(r) ⊕ o, i fatti (s, r, o) e (o, r, s) sono lo
# STESSO vettore, quindi una relazione simmetrica memorizzata nelle due direzioni
# produce un fatto di peso 2. La Law VII approssima con N_eff = Σw²; qui la
# distribuzione della somma pesata degli altri fatti è calcolata esattamente.


def _weighted_sum_pmf(weights) -> tuple:
    """pmf di Σ w_j x_j con x_j Rademacher indipendenti; restituisce (offset, pmf)."""
    from collections import Counter
    pmf = np.array([1.0])
    offset = 0                      # pmf[i] = P(somma = i + offset)
    for w, count in Counter(int(w) for w in weights).items():
        # c fatti di peso w: somma = w·(2B - c), B ~ Binomial(c, 1/2)
        b = binom_pmf(count, 0.5)
        part = np.zeros(2 * w * count + 1)
        part[::2 * w] = b          # valori -w·c, -w·c + 2w, ..., +w·c
        pmf = np.convolve(pmf, part)
        offset -= w * count
    return offset, pmf


def p_agree_weighted(query_weight: int, other_weights) -> float:
    """P(la maggioranza concorda con un fatto di peso `query_weight`), esatta."""
    offset, pmf = _weighted_sum_pmf(other_weights)
    totals = np.arange(len(pmf)) + offset + query_weight
    return float(np.sum(pmf * np.where(totals > 0, 1.0, np.where(totals == 0, 0.5, 0.0))))


def cleanup_accuracy_mixed(dim: int, codebook: int, correct_p, alias_p=()) -> float:
    """Come cleanup_accuracy, ma ogni candidato a segnale ha la sua p di accordo.

    `correct_p` e `alias_p` sono le probabilità di accordo per bit dei codeword
    degli oggetti veri e degli alias. I pareggi fra un oggetto vero e un alias
    si dividono a metà.
    """
    correct_p, alias_p = list(correct_p), list(alias_p)
    n_null = codebook - len(correct_p) - len(alias_p)
    if n_null < 0:
        raise ValueError(f"codebook ({codebook}) is smaller than the number of "
                         f"signal candidates ({len(correct_p) + len(alias_p)})")
    if not correct_p:              # unanswerable or alias-only: never correct
        return 0.0

    def all_above(ps):             # P(tutte le distanze > d), per d = 0..dim
        out = np.ones(dim + 1)
        for p in ps:
            out = out * _signal_sf(dim, float(p))
        return out
    fc, fa = all_above(correct_p), all_above(alias_p)
    fc_prev = np.concatenate(([1.0], fc[:-1]))
    fa_prev = np.concatenate(([1.0], fa[:-1]))
    p_min_correct = fc_prev - fc                      # min degli oggetti veri == d
    beats_alias = fa + 0.5 * (fa_prev - fa)           # alias tutti > d, o pari a metà
    return float(np.sum(p_min_correct * beats_alias * _null_win(dim, n_null)))


@lru_cache(maxsize=256)
def _null_win_cached(dim: int, n_null: int) -> np.ndarray:
    """P(il candidato batte n_null codeword nulli | distanza d), pareggi a metà."""
    null = binom_pmf(dim, 0.5)
    null_sf = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)
    win = null_sf ** n_null
    if n_null > 0:
        win = win + 0.5 * n_null * null * null_sf ** (n_null - 1)
    win = np.clip(win, 0.0, 1.0)
    win.setflags(write=False)
    return win


def _null_win(dim: int, n_null: int) -> np.ndarray:
    return _null_win_cached(int(dim), int(n_null))


@lru_cache(maxsize=4096)
def _signal_sf_cached(dim: int, p: float) -> np.ndarray:
    """P(distanza di un candidato con accordo p > d), d = 0..dim."""
    out = np.clip(1.0 - np.cumsum(binom_pmf(dim, 1.0 - p)), 0.0, 1.0)
    out.setflags(write=False)
    return out


def _signal_sf(dim: int, p: float) -> np.ndarray:
    return _signal_sf_cached(int(dim), float(p))


def fact_key(s, r, o):
    """Chiave del vettore di un fatto. (s, r, o) e (o, r, s) sono lo stesso vettore;
    un self-loop (s, r, s) vale c_s ⊕ ρ(c_r) ⊕ c_s = ρ(c_r), quindi TUTTI i self-loop
    di una relazione sono lo stesso vettore, qualunque sia s."""
    return ("__self__", r) if s == o else (frozenset((s, o)), r)


def fact_weights(triples):
    """Peso di ogni vettore distinto (vedi fact_key)."""
    from collections import Counter
    return Counter(fact_key(s, r, o) for s, r, o in triples)


def two_hop_joint_fast(n_facts: int, dim: int, codebook: int, width: float = 8.0):
    """Come two_hop_joint, in tempo circa lineare in D invece che cubico.

    Condizionando sul numero k di bit in cui f1 = f2 (k ~ Binomial(D, 1/2)): sui k
    bit uguali i due hop sono in disaccordo insieme (a di essi, a ~ Bin(k, 1 - u));
    sui D - k bit diversi esattamente uno dei due lo è (b per il primo,
    b ~ Bin(D - k, 1 - v), e D - k - b per il secondo). Quindi d1 = a + b e
    d2 = a + (D - k - b). Le tre binomiali si troncano a `width` deviazioni
    standard; la massa scartata è sotto 1e-12.
    """
    u = _agree_given(n_facts, same=True)
    v = _agree_given(n_facts, same=False)
    null = binom_pmf(dim, 0.5)
    null_sf = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)
    n_null = codebook - 1
    win = null_sf ** n_null
    if n_null > 0:
        win = win + 0.5 * n_null * null * null_sf ** (n_null - 1)
    win = np.clip(win, 0.0, 1.0)

    def window(n, p):
        mu, sd = n * p, np.sqrt(max(n * p * (1 - p), 1e-12))
        lo, hi = max(0, int(mu - width * sd) - 1), min(n, int(mu + width * sd) + 2)
        return lo, hi

    pk = binom_pmf(dim, 0.5)
    klo, khi = window(dim, 0.5)
    both = single = 0.0
    for k in range(klo, khi + 1):
        alo, ahi = window(k, 1 - u)
        blo, bhi = window(dim - k, 1 - v)
        pa = binom_pmf(k, 1 - u)[alo:ahi + 1]
        pb = binom_pmf(dim - k, 1 - v)[blo:bhi + 1]
        a = np.arange(alo, ahi + 1)[:, None]
        b = np.arange(blo, bhi + 1)[None, :]
        d1, d2 = a + b, a + (dim - k - b)
        w = pa[:, None] * pb[None, :]
        both += pk[k] * float(np.sum(w * win[d1] * win[d2]))
        single += pk[k] * float(np.sum(w * win[d1]))
    return single, both, single * single


# ---------------------------------------------------------------------------
# Catene di h hop: il modello per bit, valutato per Monte Carlo
# ---------------------------------------------------------------------------
#
# Per h > 2 la congiunta esatta delle h distanze ha 2^h esiti per bit e diventa
# costosa. Il modello per bit però resta esatto: i bit dei fatti della catena
# sono ±1 uniformi e indipendenti (se le entità della catena sono distinte), gli
# altri N - h fatti sommano a una binomiale, la traccia è il segno del totale.
# Qui lo si campiona direttamente: nessuna memoria, nessun codeword, solo il
# modello. L'unica ipotesi resta l'indipendenza delle distanze nulle fra hop.


def chain_accuracy_mc(n_facts: int, dim: int, codebook: int, hops: int,
                      trials: int = 20000, seed: int = 0, wins=None) -> float:
    """P(tutti gli h hop di una catena riescono), dal modello per bit.

    `wins`, se dato, è una lista di h vettori P(vince | distanza d), uno per hop:
    serve per la regola esatta dei pareggi (win_ordered), in cui la posizione del
    bersaglio nel codebook conta. Senza, i pareggi si dividono a metà.
    """
    if n_facts < hops:
        raise ValueError("chain_accuracy_mc needs n_facts >= hops")
    rng = np.random.RandomState(seed)
    null = binom_pmf(dim, 0.5)
    null_sf = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)
    n_null = codebook - 1
    win = null_sf ** n_null
    if n_null > 0:
        win = win + 0.5 * n_null * null * null_sf ** (n_null - 1)
    win = np.clip(win, 0.0, 1.0)
    win_mat = np.stack(wins) if wins is not None else np.tile(win, (hops, 1))
    others = n_facts - hops
    total = 0.0
    batch = max(1, min(trials, 2_000_000 // max(dim * hops, 1)))
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        f = rng.choice(np.array([-1, 1], dtype=np.int8), size=(b, dim, hops))
        rest = 2 * rng.binomial(others, 0.5, size=(b, dim)) - others
        s = f.sum(axis=2).astype(np.int64) + rest
        ties = s == 0
        t = np.where(s > 0, 1, -1)
        t[ties] = rng.choice([-1, 1], size=int(ties.sum()))
        d = (f != t[:, :, None]).sum(axis=1)            # (b, hops): distanze
        total += float(np.prod(win_mat[np.arange(hops)[None, :], d], axis=1).sum())
        done += b
    return total / trials


# ---------------------------------------------------------------------------
# La regola dei pareggi della reference, esatta
# ---------------------------------------------------------------------------
#
# ItemMemory.cleanup restituisce il PRIMO codeword a distanza minima, in ordine
# di inserimento. Per un bersaglio con n_before codeword nulli inseriti prima e
# n_after inseriti dopo, vince se e solo se i primi sono tutti a distanza
# strettamente maggiore e i secondi almeno uguale:
#     P(vince | d) = P(nullo > d)^n_before · P(nullo >= d)^n_after,
# senza approssimazioni, date le distanze nulle indipendenti. La divisione a metà
# usata altrove in questo modulo è la sua media su posizioni casuali; con molti
# codeword inseriti dopo il bersaglio, come i distrattori, lo scarto arriva a
# quasi 2 punti (docs/preregistration/deepchain2.md).


def win_ordered(dim: int, n_before: int, n_after: int) -> np.ndarray:
    null = binom_pmf(dim, 0.5)
    gt = np.clip(1.0 - np.cumsum(null), 0.0, 1.0)      # P(nullo > d)
    ge = np.clip(gt + null, 0.0, 1.0)                    # P(nullo >= d)
    return gt ** n_before * ge ** n_after


def cleanup_accuracy_ordered(n_facts: int, dim: int, n_before: int, n_after: int) -> float:
    """Accuratezza di cleanup con la regola dei pareggi della reference."""
    sig = binom_pmf(dim, 1.0 - p_agree(n_facts))
    return float(np.sum(sig * win_ordered(dim, n_before, n_after)))


# ---------------------------------------------------------------------------
# API d'uso: il contratto di un insieme di triple, prima di memorizzarle
# ---------------------------------------------------------------------------


def _structure(triples):
    from collections import defaultdict
    objects, into = defaultdict(set), defaultdict(set)
    for s, r, o in triples:
        objects[(s, r)].add(o)
        into[(o, r)].add(s)          # (x, r, s) memorizzato: x è un alias per (s, r)
    codebook = len({x for t in triples for x in t})
    return objects, into, codebook


def predict_queries(triples, dim: int, queries=None, codebook=None):
    """Accuratezza prevista di ogni query (s, r), per l'encoding della reference.

    Conta i gemelli simmetrici come un fatto di peso 2, gli alias come candidati a
    pari segnale, più oggetti veri come più bersagli. Divide i pareggi a metà.
    `codebook`, se dato, sostituisce il numero di entità distinte nelle triple
    (per esempio quando l'item memory contiene anche distrattori). Una query senza
    oggetto vero memorizzato (non rispondibile, o solo alias) vale 0.0.
    """
    objects, into, m = _structure(triples)
    if codebook is not None:
        m = int(codebook)
    queries = list(objects) if queries is None else list(queries)
    weights = fact_weights(triples)
    all_w = list(weights.values())
    p_by_w = {}

    def p_of(vec):
        w = weights[vec]
        if w not in p_by_w:
            others = list(all_w)
            others.remove(w)
            p_by_w[w] = p_agree_weighted(w, others)
        return p_by_w[w]

    self_rels = {r for s, r, o in triples if s == o}
    out = []
    for s, r in queries:
        good, bad = objects[(s, r)], into[(s, r)] - objects[(s, r)]
        cp = [p_of(fact_key(s, r, o)) for o in good]
        ap = [p_of(fact_key(x, r, s)) for x in bad]
        # un self-loop su r, ρ(c_r), dà a OGNI query (x, r) il candidato x stesso
        if r in self_rels and s not in good and s not in bad:
            ap.append(p_of(("__self__", r)))
        out.append(cleanup_accuracy_mixed(dim, m, cp, ap))
    return out


def ceiling(triples, queries=None) -> float:
    """Il tetto di accuratezza imposto dagli alias, qualunque sia D: media di g/(g+a)."""
    objects, into, _m = _structure(triples)
    queries = list(objects) if queries is None else list(queries)
    self_rels = {r for s, r, o in triples if s == o}

    def cap(q):
        s, r = q
        g, a = len(objects[q]), len(into[q] - objects[q])
        if r in self_rels and s not in objects[q] and s not in into[q]:
            a += 1
        return g / (g + a) if g else 0.0
    if not queries:
        raise ValueError("no queries: the triples contain no (subject, relation) pair")
    return float(np.mean([cap(q) for q in queries]))


def _queries_or_raise(objects, queries):
    queries = list(objects) if queries is None else list(queries)
    if not queries:
        raise ValueError("no queries: the triples contain no (subject, relation) pair")
    return queries


def contract_for(triples, dim: int, queries=None, codebook=None) -> dict:
    """Il contratto di queste triple a dimensione `dim`, calcolato prima di memorizzarle.

    Restituisce l'accuratezza prevista (media sulle query), il tetto imposto dagli
    alias, la quota di fatti con gemello simmetrico e di query con alias.
    """
    objects, into, m = _structure(triples)
    if codebook is not None:
        m = int(codebook)
    queries = _queries_or_raise(objects, queries)
    weights = fact_weights(triples)
    return {
        "dim": dim,
        "facts": len(triples),
        "codebook": m,
        "expected_accuracy": float(np.mean(predict_queries(triples, dim, queries, m))),
        "ceiling": ceiling(triples, queries),
        "twin_share": sum(c for c in weights.values() if c > 1) / max(len(triples), 1),
        "alias_share": float(np.mean([len(into[q] - objects[q]) > 0 for q in queries])),
    }


def min_dimension(triples, target: float, step: int = 64, d_max: int = 1 << 16,
                  queries=None, codebook=None):
    """La dimensione minima (multiplo di `step`) con accuratezza prevista >= target.

    Restituisce None se il target supera ciò che si ottiene a `d_max`, per esempio
    perché sta sopra il tetto degli alias. `queries` e `codebook` come in
    contract_for.
    """
    objects, _into, _m = _structure(triples)
    queries = _queries_or_raise(objects, queries)

    def acc(d):
        return float(np.mean(predict_queries(triples, d, queries, codebook)))
    if acc(d_max) < target:
        return None
    lo, hi = 0, d_max // step
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if acc(step * mid) >= target:
            hi = mid
        else:
            lo = mid
    return step * hi
