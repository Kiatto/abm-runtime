"""Sei famiglie VSA dietro un'interfaccia sola: la base del dimensionatore `abm size`.

Porta nel pacchetto, ripulito, il codice di misura degli harness congelati
`examples/vsa_families_prereg.py` (prereg. 21: `measure`, MAP-B, MAP-I, FHRR, BSDC-OR,
BSDC-S) ed `examples/generalization_prereg.py` (prereg. 22: HRR, atomi salati di MAP-B).
Gli harness non sono importati: restano congelati, e i test confrontano le due versioni.

    from abm import families
    mem = families.make("FHRR", 2048, seed=3)
    mem.store([("a", "r", "b"), ("b", "r", "c")])
    mem.query("a", "r")            # ("b", punteggio)
    mem.query_many([("a", "r")])   # vettorizzata
    mem.bits()                     # bit della traccia
    families.estimate_bytes("HRR", 2048, n_facts=2, n_symbols=3)   # memoria di picco stimata

Encoding comune (come negli harness): f = s ∘ ρ(r) ∘ o, ρ = shift ciclico di 1; la traccia
è il bundling dei fatti; la query (s, r) slega s ∘ ρ(r) e fa il cleanup sul codebook di
TUTTI i simboli memorizzati (entità e relazioni insieme, M = |codebook|).

Opzioni di `make` (solo keyword; i default riproducono gli harness e `abm.Memory`)
-----------------------------------------------------------------------------------
- `atoms="sorted"` (default) | `"per_symbol"`: vedi «Atomi» sotto.
- `ties=None` | `"first"` | `"random"`: vedi «Pareggi». Per MAP-B il default è `"first"`;
  per le altre famiglie è ammesso solo `"random"` (o None).
- `strict=False` | `True`: solo BSDC, vedi «BSDC».

Convenzioni
-----------
- La memoria è funzione pura di (famiglia, dim, seed, opzioni, multinsieme delle triple): la
  traccia e (per le famiglie casuali) gli atomi si ricostruiscono pigramente alla prima query
  dopo uno `store`. Le triple duplicate contano come peso (un fatto memorizzato due volte vale
  2, come `abm.Memory.store`); i self-loop (s == o) sono ammessi come in `measure` degli
  harness (che li ricevono già filtrati dal campionamento, non dal codice di misura).
- Codebook. MAP-B: ordine di inserimento (s, r, o per tripla), come `abm.Memory`.
  Le altre: ordine lessicografico dei nomi, come `Cell.symbols` degli harness.
- Atomi, `atoms="sorted"` (default, equivalenza con gli harness): per le famiglie non MAP-B
  gli atomi sono estratti in blocco da `numpy.random.default_rng(seed)` nell'ordine
  lessicografico, quindi aggiungere un simbolo (anche con un solo fatto nuovo) cambia TUTTI
  gli atomi: la memoria resta deterministica, ma non è stabile all'aggiunta di fatti.
  MAP-I: ±1 da `rng.integers(2, (M, D))`; FHRR: e^{iθ}, θ = `rng.uniform(0, 2π, (M, D))`,
  complex64; HRR: `rng.standard_normal((M, D)) / √D` in float32 (la prereg. 22 usa un
  generatore per simbolo: distribuzione identica, valori diversi); BSDC: indici
  `rng.integers(L, (M, B))`. (L'estrazione avviene a blocchi di righe dallo stesso flusso:
  valori identici all'estrazione in un colpo, verificato nei test.)
- Atomi, `atoms="per_symbol"`: ogni atomo viene da un generatore proprio,
  `default_rng([seed, md5(nome)[:8]])`, con la stessa distribuzione: l'atomo di un simbolo
  non dipende dagli altri simboli, quindi è stabile all'aggiunta di fatti. Eccezione BSDC:
  l'alfabeto L dipende da N, quindi gli indici cambiano quando L cambia (restano stabili a L
  fisso). Le risposte NON coincidono con gli harness (atomi diversi, stessa distribuzione).
- Atomi MAP-B (sempre per simbolo; `atoms` non ha effetto): `abm.random_hv(nome, D)` (md5 del
  nome, minuscolo come in abm); con seed ≠ 0 il nome è salato, `f"{nome}#{seed}"`, così gli
  atomi sono indipendenti fra seed (come il sale della prereg. 22). Con seed = 0 la memoria
  coincide con `abm.Memory` (stessa traccia, stesse risposte, stessa confidenza; asserito nei
  test).
- Pareggi. MAP-B, `ties="first"` (default): il PRIMO simbolo del codebook (ordine di
  inserimento) a distanza minima, cioè argmax, come `ItemMemory.cleanup` di abm e la reference
  delle preregistrazioni. È una scelta diversa dalle altre famiglie e introduce una
  preferenza sistematica per i simboli inseriti prima: va tenuta per l'equivalenza con
  `abm.Memory`. `ties="random"` (solo MAP-B) usa il pareggio uniforme delle altre famiglie.
  Tutte le altre: uniforme fra i pari merito (`_pick` degli harness), ma con un generatore
  derivato da (seed, s, r) invece che dal flusso condiviso dell'harness: la query è così una
  funzione pura e `query_many` == `query` ripetuta. Su query senza pareggi le risposte
  coincidono con l'harness a parità di atomi; con pareggi coincide la distribuzione.
- Simboli mai memorizzati: `query_many(pairs, unknown="raise")` (default) solleva KeyError
  col simbolo e la posizione della query; `unknown="skip"` restituisce None al posto di
  quelle risposte. Differenza con `abm.Memory`: quella codifica un simbolo nuovo col suo
  vettore deterministico e risponde comunque (con confidenza bassa); qui il cleanup è sul
  codebook dei simboli memorizzati e, per le famiglie casuali, un simbolo nuovo non ha un
  atomo definito nella modalità "sorted" (cambierebbe tutti gli altri), quindi si rifiuta.
- BSDC: D = B·L con L = potenza di 2 più vicina a N/ln 2 (`bsdc_length` della prereg. 21),
  N = fatti memorizzati contati col peso. L dipende dal carico: alla costruzione della
  traccia B = ⌈D_richiesto / L⌉ e `dim` diventa il D effettivo B·L (arrotondato per eccesso
  al multiplo valido); `dim_requested` conserva quello chiesto. Quando i due differiscono si
  emette un `UserWarning`; con `strict=True` si solleva invece ValueError.
- `bits()` (convenzione di `bits_per_component` delle prereg. 21-22): bit della sola traccia,
  N = fatti memorizzati CONTATI COL PESO (i duplicati contano, come nella traccia).
  MAP-B D; BSDC-OR B·L; MAP-I e BSDC-S D·⌈log2(N+1)⌉ (somma intera / conteggi con N+1
  valori); FHRR D·32 (float16 reale + float16 immaginaria); HRR D·32 (float32).
- `codebook_bits()`: costo del codebook, escluso da `bits()`. Convenzione: gli atomi sono
  rigenerabili da (seed, nomi) e non si contano; si conta l'indice simbolo → riga,
  M·⌈log2 M⌉ bit (0 se M ≤ 1). I nomi stessi (stringhe) non sono contati.
- Il secondo elemento della risposta è il punteggio della famiglia, non confrontabile fra
  famiglie: MAP-B `abm.confidence` (come `abm.Memory.query`); MAP-I prodotto scalare intero;
  FHRR Re⟨y, c̄⟩; HRR prodotto scalare; BSDC-OR blocchi colpiti; BSDC-S somma dei conteggi.
- Memoria. Costruzione e query lavorano a blocchi di righe (≈ `_CHUNK_BYTES` di temporanei).
  Restano in memoria solo gli atomi e la traccia: per HRR gli spettri degli atomi non sono
  conservati ma ricalcolati (in complex128, valori identici) per i soli simboli di ciascun
  blocco. `estimate_bytes` stima il picco di costruzione + query.
- Cache: il modulo non ha cache globali; ogni memoria tiene solo i propri array.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np

from .abm import confidence, random_hv

FAMILIES = ("MAP-B", "MAP-I", "FHRR", "HRR", "BSDC-OR", "BSDC-S")
FHRR_BITS = 32
HRR_BITS = 32
_CHUNK_BYTES = 32 << 20         # temporanei per blocco (costruzione e query)

__all__ = ["FAMILIES", "Memory", "make", "bsdc_length", "estimate_bytes"]


class Memory(Protocol):
    family: str
    dim: int

    def store(self, triples: Iterable[Tuple[str, str, str]]) -> None: ...
    def query(self, s: str, r: str) -> Tuple[str, float]: ...
    def query_many(self, pairs: Iterable[Tuple[str, str]],
                   unknown: str = "raise") -> List[Optional[Tuple[str, float]]]: ...
    def bits(self) -> int: ...
    def codebook_bits(self) -> int: ...
    def codebook_size(self) -> int: ...


def bsdc_length(n: int) -> int:
    """L: potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà); prereg. 21."""
    return int(2 ** round(math.log2(max(n, 1) / math.log(2))))


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} deve essere un intero ≥ {minimum}, ricevuto {value!r}")
    return int(value)


def make(family: str, dim: int, seed: int = 0, *, atoms: str = "sorted",
         ties: Optional[str] = None, strict: bool = False) -> "Memory":
    """Una memoria vuota della famiglia. ValueError su argomenti non validi.

    atoms: "sorted" (default, come gli harness) | "per_symbol" (atomo per simbolo, stabile
    all'aggiunta di fatti). ties: MAP-B "first" (default, come abm.Memory) | "random"; le
    altre famiglie solo "random". strict: per BSDC solleva se dim va arrotondato.
    """
    if family not in FAMILIES:
        raise ValueError(f"famiglia sconosciuta {family!r}; valide: {FAMILIES}")
    dim = _check_int("dim", dim, 1)
    seed = _check_int("seed", seed, 0)
    if atoms not in ("sorted", "per_symbol"):
        raise ValueError(f"atoms deve essere 'sorted' o 'per_symbol', ricevuto {atoms!r}")
    if ties is None:
        ties = "first" if family == "MAP-B" else "random"
    if ties not in ("first", "random"):
        raise ValueError(f"ties deve essere 'first' o 'random', ricevuto {ties!r}")
    if ties == "first" and family != "MAP-B":
        raise ValueError("ties='first' esiste solo per MAP-B; le altre famiglie usano 'random'")
    return _Memory(family, dim, seed, atoms, ties, bool(strict))


def _h64(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:8], "little")


def _rows(nbytes_per_row: float) -> int:
    return max(1, int(_CHUNK_BYTES // max(nbytes_per_row, 1)))


# byte per riga di query (temporanei vivi insieme, per famiglia); usati sia per dimensionare
# i blocchi sia da estimate_bytes
def _query_row_bytes(family, d, m, b):
    if family == "MAP-B":
        return 12 * d + 8 * m
    if family == "MAP-I":
        return 32 * d + 16 * m
    if family == "FHRR":
        return 32 * d + 16 * m
    if family == "HRR":
        return 64 * d + 8 * m
    return 32 * m * b + 16 * m                       # BSDC


def estimate_bytes(family: str, dim: int, n_facts: int, n_symbols: int,
                   n_queries: Optional[int] = None) -> int:
    """Memoria di picco stimata (byte) per costruzione + query di una memoria della famiglia.

    Conta gli array numpy (atomi, traccia, temporanei a blocchi di costruzione e query), non
    l'interprete né le liste Python di triple/risposte. n_queries: numero di query passate
    a una sola `query_many` (default n_facts). Validata nei test contro il picco misurato
    (tracemalloc) entro un fattore 1.5 su configurazioni dove gli array dominano.
    """
    if family not in FAMILIES:
        raise ValueError(f"famiglia sconosciuta {family!r}; valide: {FAMILIES}")
    d = _check_int("dim", dim, 1)
    n = _check_int("n_facts", n_facts, 1)
    m = _check_int("n_symbols", n_symbols, 1)
    q = n if n_queries is None else _check_int("n_queries", n_queries, 0)
    # temporanei effettivi per riga (misurati con tracemalloc) entro i blocchi del codice
    gen = min(m, _rows(32 * d))                              # righe per blocco di atomi
    b = 1
    if family.startswith("BSDC"):
        L = bsdc_length(n)
        b = -(-d // L)
        d = b * L
        persist = 16 * m * b + 8 * d
        build = 32 * n * b + 16 * d
        q_row = 16 * m * b + 8 * m
    elif family == "MAP-B":
        persist = 5 * m * d + d
        build = m * d + min(n, _rows(24 * d)) * 16 * d       # lista di atomi prima dello stack
        q_row = 6 * d + 4 * m
    elif family == "MAP-I":
        persist = 8 * m * d + 8 * d
        build = max(gen * 24 * d, min(n, _rows(32 * d)) * 24 * d)
        q_row = 24 * d + 12 * m
    elif family == "FHRR":
        persist = 8 * m * d + 8 * d
        build = gen * 40 * d
        q_row = 24 * d + 12 * m
    else:                                                    # HRR
        persist = 4 * m * d + 12 * d
        build = max(gen * 21 * d, min(n, _rows(48 * d)) * 32 * d)
        q_row = 24 * d + 4 * m
    query = min(q, _rows(_query_row_bytes(family, d, m, b))) * q_row
    return int(persist + max(build, query))


class _Memory:
    def __init__(self, family, dim, seed, atoms="sorted", ties=None, strict=False):
        self.family, self.dim, self.dim_requested, self.seed = family, dim, dim, seed
        self.atoms_mode = atoms
        self.ties = ties or ("first" if family == "MAP-B" else "random")
        self.strict = strict
        self._triples: List[Tuple[str, str, str]] = []
        self._order: dict = {}                     # simbolo -> ordine di inserimento
        self._dirty = True
        self.length = self.blocks = None           # solo BSDC

    # -- scrittura -------------------------------------------------------------

    def store(self, triples):
        new = []
        for t in triples:
            if not isinstance(t, (tuple, list)) or len(t) != 3 \
                    or not all(isinstance(x, str) for x in t):
                raise ValueError(f"tripla non valida {t!r}: servono tre stringhe (s, r, o)")
            new.append(tuple(t))
        for t in new:
            self._triples.append(t)
            for x in t:
                self._order.setdefault(x, len(self._order))
        if new:
            self._dirty = True

    def n_facts(self) -> int:
        """Fatti memorizzati contati col peso (duplicati inclusi)."""
        return len(self._triples)

    def codebook_size(self) -> int:
        return len(self._order)

    def bits(self) -> int:
        """Bit della traccia; N = fatti col peso (duplicati inclusi). Vedi docstring del modulo."""
        self._build()
        n, f = len(self._triples), self.family
        if f in ("MAP-B", "BSDC-OR"):
            return self.dim
        if f in ("FHRR", "HRR"):
            return self.dim * (FHRR_BITS if f == "FHRR" else HRR_BITS)
        return self.dim * math.ceil(math.log2(n + 1))

    def codebook_bits(self) -> int:
        """Bit dell'indice del codebook: M·⌈log2 M⌉ (atomi rigenerabili, non contati)."""
        m = len(self._order)
        return m * math.ceil(math.log2(m)) if m > 1 else 0

    # -- costruzione pigra -----------------------------------------------------

    def _atom_rows(self, width, draw):
        """Matrice (M, width) degli atomi, generata a blocchi di righe. draw(rng, k) -> (k, width)."""
        m = len(self.sym)
        out = None
        if self.atoms_mode == "per_symbol":
            for i, x in enumerate(self.sym):
                row = draw(np.random.default_rng([self.seed, _h64(x)]), 1)
                if out is None:
                    out = np.empty((m, width), dtype=row.dtype)
                out[i] = row[0]
            return out
        rng = np.random.default_rng(self.seed)
        step = _rows(32 * width)
        for a in range(0, m, step):
            blk = draw(rng, min(step, m - a))
            if out is None:
                out = np.empty((m, width), dtype=blk.dtype)
            out[a:a + len(blk)] = blk
        return out

    def _build(self):
        if not self._dirty:
            return
        self._dirty = False
        f, d = self.family, self.dim_requested
        if f == "MAP-B":
            self.sym = sorted(self._order, key=self._order.get)
        else:
            self.sym = sorted(self._order)
        self.idx = {x: i for i, x in enumerate(self.sym)}
        if not self._triples:
            return
        n = len(self._triples)
        T = np.array([[self.idx[x] for x in t] for t in self._triples], dtype=np.int64)
        if f == "MAP-B":
            salt = (lambda x: x) if self.seed == 0 else (lambda x: f"{x}#{self.seed}")
            self.atoms = np.stack([random_hv(salt(x), d) for x in self.sym])
            acc = np.zeros(d, dtype=np.int64)
            step = _rows(24 * d)
            for a in range(0, n, step):
                t = T[a:a + step]
                acc += (self.atoms[t[:, 0]].astype(np.int64)
                        * np.roll(self.atoms[t[:, 1]], 1, axis=1) * self.atoms[t[:, 2]]).sum(axis=0)
            if n % 2 == 0:
                acc = acc * 2 + random_hv("__tie__", d)
            self.trace = np.where(acc >= 0, 1, -1).astype(np.int8)
            self._codes = self.atoms.astype(np.float32)
            return
        if f.startswith("BSDC"):
            L = bsdc_length(n)
            B = -(-d // L)
            if B * L != d:
                msg = (f"{f}: dim={d} non è multiplo di L={L} (N={n}); "
                       f"dim effettivo {B * L} = {B}·{L}")
                if self.strict:
                    self._dirty = True
                    raise ValueError(msg + " (strict=True)")
                warnings.warn(msg, UserWarning, stacklevel=3)
            self.length, self.blocks, self.dim = L, B, B * L
            self.atoms = self._atom_rows(B, lambda g, k: g.integers(L, size=(k, B)))
            self.rho = np.roll(self.atoms, 1, axis=1)
            mem = np.zeros((B, L), dtype=np.int64)
            pos = (self.atoms[T[:, 0]] + self.rho[T[:, 1]] + self.atoms[T[:, 2]]) % L
            np.add.at(mem, (np.broadcast_to(np.arange(B), pos.shape), pos), 1)
            self.trace = (mem > 0).astype(np.int64) if f == "BSDC-OR" else mem
            return
        if f == "MAP-I":
            self.atoms = self._atom_rows(
                d, lambda g, k: (2 * g.integers(2, size=(k, d)) - 1).astype(np.float64))
            acc = np.zeros(d, dtype=np.float64)
            step = _rows(32 * d)
            for a in range(0, n, step):            # somme intere: esatte in qualunque ordine
                t = T[a:a + step]
                acc += (self.atoms[t[:, 0]] * np.roll(self.atoms[t[:, 1]], 1, axis=1)
                        * self.atoms[t[:, 2]]).sum(axis=0)
            self.trace = acc
        elif f == "FHRR":
            self.atoms = self._atom_rows(
                d, lambda g, k: np.exp(1j * g.uniform(0, 2 * np.pi, size=(k, d))).astype(np.complex64))
            acc = np.zeros(d, dtype=np.complex128)
            for s, r, o in T:                       # stesso ordine di somma dell'harness
                acc += self.atoms[s] * np.roll(self.atoms[r], 1) * self.atoms[o]
            self.trace = (acc.real.astype(np.float16).astype(np.float32)
                          + 1j * acc.imag.astype(np.float16).astype(np.float32))
        else:                                       # HRR
            sq = math.sqrt(d)
            self.atoms = self._atom_rows(
                d, lambda g, k: (g.standard_normal((k, d)) / sq).astype(np.float32))
            tr = np.zeros(d // 2 + 1, np.complex128)
            step = _rows(48 * d)
            for a in range(0, n, step):            # spettri dei soli simboli del blocco
                t = T[a:a + step]
                Fs = np.fft.rfft(self.atoms[t[:, 0]].astype(np.float64), axis=1)
                Fr = np.fft.rfft(np.roll(self.atoms[t[:, 1]], 1, axis=1).astype(np.float64), axis=1)
                Fo = np.fft.rfft(self.atoms[t[:, 2]].astype(np.float64), axis=1)
                for k in range(len(t)):            # stesso ordine di somma dell'harness
                    tr += Fs[k] * Fr[k] * Fo[k]
                del Fs, Fr, Fo
            self.trace = np.fft.irfft(tr, n=d).astype(np.float32)   # traccia salvata, float32
            self._Ft = np.fft.rfft(self.trace.astype(np.float64))

    # -- lettura ---------------------------------------------------------------

    def _ids(self, pairs, unknown="raise"):
        self._build()
        if not self._triples:
            raise ValueError("query su una memoria vuota: memorizza almeno un fatto")
        ss, rr, keep = [], [], []
        for k, p in enumerate(pairs):
            if not isinstance(p, (tuple, list)) or len(p) != 2 \
                    or not all(isinstance(x, str) for x in p):
                raise ValueError(f"query non valida {p!r}: servono due stringhe (s, r)")
            missing = [x for x in p if x not in self.idx]
            if missing:
                if unknown == "skip":
                    continue
                raise KeyError(
                    f"simbolo mai memorizzato {missing[0]!r} nella query #{k} {tuple(p)!r} "
                    f"({self.family}: il cleanup è sul codebook dei {len(self.sym)} simboli "
                    "memorizzati; usa unknown='skip' per ottenere None)")
            keep.append(k)
            ss.append(self.idx[p[0]])
            rr.append(self.idx[p[1]])
        return np.array(ss, dtype=np.int64), np.array(rr, dtype=np.int64), keep

    def _scores(self, s, r):
        """Punteggi (len(s), M) del cleanup; più alto = migliore."""
        f = self.family
        if f.startswith("BSDC"):
            pos = (self.atoms[None, :, :] + (self.atoms[s] + self.rho[r])[:, None, :]) % self.length
            return self.trace[np.arange(self.blocks), pos].sum(axis=2)
        if f == "HRR":
            K = (np.fft.rfft(self.atoms[s].astype(np.float64), axis=1)
                 * np.fft.rfft(np.roll(self.atoms[r], 1, axis=1).astype(np.float64), axis=1))
            K = np.conj(K, out=K)
            K *= self._Ft[None, :]
            Y = np.fft.irfft(K, n=self.dim).astype(np.float32)
            del K
            return Y @ self.atoms.T
        if f == "FHRR":
            # Re(Y · conj(A)ᵀ) = Re(conj(Y) · Aᵀ): niente copia coniugata del codebook
            Y = self.trace[None, :] * np.conj(self.atoms[s]) * np.conj(np.roll(self.atoms[r], 1, axis=1))
            return (np.conj(Y) @ self.atoms.T).real
        if f == "MAP-I":
            Y = self.trace[None, :] * self.atoms[s] * np.roll(self.atoms[r], 1, axis=1)
            return Y @ self.atoms.T
        Y = (self.trace[None, :].astype(np.int16) * self.atoms[s]
             * np.roll(self.atoms[r], 1, axis=1)).astype(np.float32)
        return Y @ self._codes.T                    # MAP-B: D − 2·Hamming

    def _tie_rng(self, s, r):
        h = hashlib.md5(f"{s}\x00{r}".encode()).digest()
        return np.random.default_rng([self.seed, int.from_bytes(h[:8], "little")])

    def query_many(self, pairs, unknown: str = "raise") -> List[Optional[Tuple[str, float]]]:
        """Risposte (simbolo, punteggio) nell'ordine delle query.

        unknown="raise" (default): KeyError se una query contiene un simbolo mai memorizzato.
        unknown="skip": None al posto della risposta di quelle query.
        """
        if unknown not in ("raise", "skip"):
            raise ValueError(f"unknown deve essere 'raise' o 'skip', ricevuto {unknown!r}")
        pairs = list(pairs)
        s, r, keep = self._ids(pairs, unknown)
        out: List[Optional[Tuple[str, float]]] = [None] * len(pairs)
        if not keep:
            return out
        m = len(self.sym)
        chunk = _rows(_query_row_bytes(self.family, self.dim, m, self.blocks or 1))
        for a in range(0, len(s), chunk):
            sc = self._scores(s[a:a + chunk], r[a:a + chunk])
            best = sc.max(axis=1)
            for k, row in enumerate(sc):
                j = keep[a + k]
                if self.family == "MAP-B" and self.ties == "first":
                    i = int(np.argmax(row))         # primo a distanza minima
                else:
                    top = np.flatnonzero(row == best[k])
                    if len(top) == 1:
                        i = int(top[0])
                    else:
                        q = pairs[j]
                        i = int(top[self._tie_rng(q[0], q[1]).integers(len(top))])
                if self.family == "MAP-B":
                    dist = int(round((self.dim - float(row[i])) / 2))
                    out[j] = (self.sym[i], confidence(dist, self.dim))
                else:
                    out[j] = (self.sym[i], float(row[i]))
        return out

    def query(self, s: str, r: str) -> Tuple[str, float]:
        return self.query_many([(s, r)])[0]

    def __repr__(self):
        return (f"families.make({self.family!r}, {self.dim_requested}, seed={self.seed})"
                f" [D={self.dim}, fatti={len(self._triples)}, M={len(self._order)}]")
