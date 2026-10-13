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

Encoding comune (come negli harness): f = s ∘ ρ(r) ∘ o, ρ = shift ciclico di 1; la traccia
è il bundling dei fatti; la query (s, r) slega s ∘ ρ(r) e fa il cleanup sul codebook di
TUTTI i simboli memorizzati (entità e relazioni insieme, M = |codebook|).

Convenzioni
-----------
- La memoria è funzione pura di (famiglia, dim, seed, multinsieme delle triple): la traccia
  e (per le famiglie casuali) gli atomi si ricostruiscono pigramente alla prima query dopo
  uno `store`. Le triple duplicate contano come peso (un fatto memorizzato due volte vale
  2, come `abm.Memory.store`); i self-loop (s == o) sono ammessi come in `measure` degli
  harness (che li ricevono già filtrati dal campionamento, non dal codice di misura).
- Codebook. MAP-B: ordine di inserimento (s, r, o per tripla), come `abm.Memory`.
  Le altre: ordine lessicografico dei nomi, come `Cell.symbols` degli harness; gli atomi sono
  estratti in blocco da `numpy.random.default_rng(seed)` in quell'ordine, quindi aggiungere
  un simbolo cambia tutti gli atomi (la memoria resta deterministica).
- Atomi. MAP-B: `abm.random_hv(nome, D)` (md5 del nome, minuscolo come in abm); con seed ≠ 0
  il nome è salato, `f"{nome}#{seed}"`, così gli atomi sono indipendenti fra seed (come il
  sale della prereg. 22). Con seed = 0 la memoria coincide con `abm.Memory` (stessa traccia,
  stesse risposte, stessa confidenza; asserito nei test). MAP-I: ±1 da
  `rng.integers(2, (M, D))`; FHRR: e^{iθ}, θ = `rng.uniform(0, 2π, (M, D))`, complex64;
  HRR: `rng.standard_normal((M, D)) / √D` in float32 (la prereg. 22 usa un generatore per
  simbolo: distribuzione identica, valori diversi); BSDC: indici `rng.integers(L, (M, B))`.
- Pareggi, quelli degli harness. MAP-B: il primo del codebook a distanza minima
  (`ItemMemory.cleanup`). Tutte le altre: uniforme fra i pari merito (`_pick`), ma con un
  generatore derivato da (seed, s, r) invece che dal flusso condiviso dell'harness: la query
  è così una funzione pura e `query_many` == `query` ripetuta. Su query senza pareggi le
  risposte coincidono con l'harness a parità di atomi; con pareggi coincide la distribuzione.
- Simboli mai memorizzati: KeyError (gli harness non li interrogano mai). Nota: per MAP-B
  `abm.Memory` li codificherebbe col loro vettore; qui si rifiutano per tutte le famiglie.
- BSDC: D = B·L con L = potenza di 2 più vicina a N/ln 2 (`bsdc_length` della prereg. 21),
  N = fatti memorizzati contati col peso. L dipende dal carico: alla costruzione della
  traccia B = ⌈D_richiesto / L⌉ e `dim` diventa il D effettivo B·L (arrotondato per eccesso
  al multiplo valido); `dim_requested` conserva quello chiesto.
- `bits()` (convenzione di `bits_per_component` delle prereg. 21-22, N = fatti col peso):
  MAP-B D; BSDC-OR B·L; MAP-I e BSDC-S D·⌈log2(N+1)⌉ (somma intera / conteggi con N+1
  valori); FHRR D·32 (float16 reale + float16 immaginaria); HRR D·32 (float32).
- Il secondo elemento della risposta è il punteggio della famiglia, non confrontabile fra
  famiglie: MAP-B `abm.confidence` (come `abm.Memory.query`); MAP-I prodotto scalare intero;
  FHRR Re⟨y, c̄⟩; HRR prodotto scalare; BSDC-OR blocchi colpiti; BSDC-S somma dei conteggi.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Protocol, Sequence, Tuple

import numpy as np

from .abm import confidence, random_hv

FAMILIES = ("MAP-B", "MAP-I", "FHRR", "HRR", "BSDC-OR", "BSDC-S")
FHRR_BITS = 32
HRR_BITS = 32
_CHUNK_ELEMS = 1 << 24          # elementi intermedi per blocco di query (memoria limitata)

__all__ = ["FAMILIES", "Memory", "make", "bsdc_length"]


class Memory(Protocol):
    family: str
    dim: int

    def store(self, triples: Iterable[Tuple[str, str, str]]) -> None: ...
    def query(self, s: str, r: str) -> Tuple[str, float]: ...
    def query_many(self, pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, float]]: ...
    def bits(self) -> int: ...
    def codebook_size(self) -> int: ...


def bsdc_length(n: int) -> int:
    """L: potenza di 2 più vicina a N/ln 2 (traccia OR piena circa a metà); prereg. 21."""
    return int(2 ** round(math.log2(max(n, 1) / math.log(2))))


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} deve essere un intero ≥ {minimum}, ricevuto {value!r}")
    return int(value)


def make(family: str, dim: int, seed: int = 0) -> "Memory":
    """Una memoria vuota della famiglia. ValueError su famiglia, dim o seed non validi."""
    if family not in FAMILIES:
        raise ValueError(f"famiglia sconosciuta {family!r}; valide: {FAMILIES}")
    dim = _check_int("dim", dim, 1)
    seed = _check_int("seed", seed, 0)
    return _Memory(family, dim, seed)


class _Memory:
    def __init__(self, family: str, dim: int, seed: int):
        self.family, self.dim, self.dim_requested, self.seed = family, dim, dim, seed
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
        return len(self._triples)

    def codebook_size(self) -> int:
        return len(self._order)

    def bits(self) -> int:
        self._build()
        n, f = len(self._triples), self.family
        if f in ("MAP-B", "BSDC-OR"):
            return self.dim
        if f in ("FHRR", "HRR"):
            return self.dim * (FHRR_BITS if f == "FHRR" else HRR_BITS)
        return self.dim * math.ceil(math.log2(n + 1))

    # -- costruzione pigra -----------------------------------------------------

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
        m, n = len(self.sym), len(self._triples)
        T = np.array([[self.idx[x] for x in t] for t in self._triples], dtype=np.int64)
        if f == "MAP-B":
            salt = (lambda x: x) if self.seed == 0 else (lambda x: f"{x}#{self.seed}")
            self.atoms = np.stack([random_hv(salt(x), d) for x in self.sym])
            facts = (self.atoms[T[:, 0]].astype(np.int64)
                     * np.roll(self.atoms[T[:, 1]], 1, axis=1) * self.atoms[T[:, 2]])
            acc = facts.sum(axis=0)
            if n % 2 == 0:
                acc = acc * 2 + random_hv("__tie__", d)
            self.trace = np.where(acc >= 0, 1, -1).astype(np.int8)
            self._codes = self.atoms.astype(np.float32)
            return
        rng = np.random.default_rng(self.seed)
        if f.startswith("BSDC"):
            L = bsdc_length(n)
            B = -(-d // L)
            self.length, self.blocks, self.dim = L, B, B * L
            self.atoms = rng.integers(L, size=(m, B))
            self.rho = np.roll(self.atoms, 1, axis=1)
            mem = np.zeros((B, L), dtype=np.int64)
            pos = (self.atoms[T[:, 0]] + self.rho[T[:, 1]] + self.atoms[T[:, 2]]) % L
            np.add.at(mem, (np.broadcast_to(np.arange(B), pos.shape), pos), 1)
            self.trace = (mem > 0).astype(np.int64) if f == "BSDC-OR" else mem
            return
        if f == "MAP-I":
            self.atoms = (2 * rng.integers(2, size=(m, d)) - 1).astype(np.float64)
            rho = np.roll(self.atoms, 1, axis=1)
            self.trace = (self.atoms[T[:, 0]] * rho[T[:, 1]] * self.atoms[T[:, 2]]).sum(axis=0)
            self._codes = self.atoms
        elif f == "FHRR":
            theta = rng.uniform(0, 2 * np.pi, size=(m, d))
            self.atoms = np.exp(1j * theta).astype(np.complex64)
            rho = np.roll(self.atoms, 1, axis=1)
            acc = np.zeros(d, dtype=np.complex128)
            for s, r, o in T:                       # stesso ordine di somma dell'harness
                acc += self.atoms[s] * rho[r] * self.atoms[o]
            self.trace = (acc.real.astype(np.float16).astype(np.float32)
                          + 1j * acc.imag.astype(np.float16).astype(np.float32))
            self._codes = np.conj(self.atoms)
        else:                                       # HRR
            self.atoms = (rng.standard_normal((m, d)) / math.sqrt(d)).astype(np.float32)
            F = np.fft.rfft(self.atoms.astype(np.float64), axis=1)
            Fr = np.fft.rfft(np.roll(self.atoms, 1, axis=1).astype(np.float64), axis=1)
            tr = np.zeros(d // 2 + 1, np.complex128)
            for s, r, o in T:
                tr += F[s] * Fr[r] * F[o]
            self.trace = np.fft.irfft(tr, n=d).astype(np.float32)   # traccia salvata, float32
            self._Ft = np.fft.rfft(self.trace.astype(np.float64))
            self._F, self._Fr = F, Fr
            self._codes = self.atoms

    # -- lettura ---------------------------------------------------------------

    def _ids(self, pairs):
        self._build()
        if not self._triples:
            raise ValueError("query su una memoria vuota: memorizza almeno un fatto")
        ss, rr = [], []
        for p in pairs:
            if not isinstance(p, (tuple, list)) or len(p) != 2 \
                    or not all(isinstance(x, str) for x in p):
                raise ValueError(f"query non valida {p!r}: servono due stringhe (s, r)")
            for x in p:
                if x not in self.idx:
                    raise KeyError(f"simbolo mai memorizzato: {x!r}")
            ss.append(self.idx[p[0]])
            rr.append(self.idx[p[1]])
        return np.array(ss, dtype=np.int64), np.array(rr, dtype=np.int64)

    def _scores(self, s, r):
        """Punteggi (len(s), M) del cleanup; più alto = migliore."""
        f = self.family
        if f.startswith("BSDC"):
            pos = (self.atoms[None, :, :] + (self.atoms[s] + self.rho[r])[:, None, :]) % self.length
            return self.trace[np.arange(self.blocks), pos].sum(axis=2)
        if f == "HRR":
            K = self._F[s] * self._Fr[r]
            Y = np.fft.irfft(np.conj(K) * self._Ft[None, :], n=self.dim).astype(np.float32)
            return Y @ self._codes.T
        if f == "FHRR":
            Y = self.trace[None, :] * np.conj(self.atoms[s]) * np.conj(np.roll(self.atoms[r], 1, axis=1))
            return (Y @ self._codes.T).real
        if f == "MAP-I":
            Y = self.trace[None, :] * self.atoms[s] * np.roll(self.atoms[r], 1, axis=1)
            return Y @ self._codes.T
        Y = (self.trace[None, :].astype(np.int16) * self.atoms[s]
             * np.roll(self.atoms[r], 1, axis=1)).astype(np.float32)
        return Y @ self._codes.T                    # MAP-B: D − 2·Hamming

    def _tie_rng(self, s, r):
        h = hashlib.md5(f"{s}\x00{r}".encode()).digest()
        return np.random.default_rng([self.seed, int.from_bytes(h[:8], "little")])

    def query_many(self, pairs) -> List[Tuple[str, float]]:
        pairs = list(pairs)
        s, r = self._ids(pairs)
        m = len(self.sym)
        per = m * self.blocks if self.family.startswith("BSDC") else 2 * self.dim + m
        chunk = max(1, _CHUNK_ELEMS // max(per, 1))
        out = []
        for a in range(0, len(s), chunk):
            sc = self._scores(s[a:a + chunk], r[a:a + chunk])
            best = sc.max(axis=1)
            for k, row in enumerate(sc):
                if self.family == "MAP-B":
                    i = int(np.argmax(row))         # primo a distanza minima
                    dist = int(round((self.dim - float(row[i])) / 2))
                    out.append((self.sym[i], confidence(dist, self.dim)))
                    continue
                top = np.flatnonzero(row == best[k])
                if len(top) == 1:
                    i = int(top[0])
                else:
                    q = pairs[a + k]
                    i = int(top[self._tie_rng(q[0], q[1]).integers(len(top))])
                out.append((self.sym[i], float(row[i])))
        return out

    def query(self, s: str, r: str) -> Tuple[str, float]:
        return self.query_many([(s, r)])[0]

    def __repr__(self):
        return (f"families.make({self.family!r}, {self.dim_requested}, seed={self.seed})"
                f" [D={self.dim}, fatti={len(self._triples)}, M={len(self._order)}]")
