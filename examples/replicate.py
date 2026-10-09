"""replicate.py — rieseguire le preregistrazioni da un clone pulito, con un comando.

Nessuna delle preregistrazioni del paper è stata replicata da qualcun altro.
Questo script non lo sostituisce, ma toglie ogni ostacolo pratico a farlo:

1. scarica i dati pubblici che servono ai target scelti (FB15k-237 con i nomi
   delle entità, WN18RR, ProofWriter, SimpleQuestions v2; nessuno per i target
   sintetici) e verifica i loro sha256;
2. riesegue gli harness delle preregistrazioni scelte, scrivendo i risultati in
   una cartella a parte (results/replica/), senza toccare quelli pubblicati;
   Di default ogni harness gira sul codice del commit che ha registrato i
   risultati (--code published); --code current usa il codice di adesso.
   Dopo il 2026-09-30 il modello esatto tratta i self-loop (s, r, s), che in
   FB15k-237 sono 1625: le preregistrazioni che lo usano danno numeri un po'
   diversi con il codice attuale.
3. confronta, per ogni preregistrazione, i numeri riassuntivi della replica con
   quelli pubblicati.

Gli harness sono deterministici (seed fissati): una replica sulla stessa
piattaforma deve riprodurre i numeri pubblicati; su un'altra piattaforma deve
riprodurli se i codeword sono gli stessi bit, cosa che la CI verifica su Linux,
macOS arm64 e Windows.

Il test 11 (human_questions) interroga un modello linguistico locale: qui se ne
riesegue solo la parte deterministica (previsione di abm.exact e risposta della
memoria alla relazione vera, con --memory-only), confrontata con i campi
corrispondenti del file pubblicato. Il test 19 (algebra) misura anche tempi:
t_*_s e time_ratio non sono deterministici e sono esclusi dal confronto.
ProofWriter richiede pyarrow.

Uso, dalla root del repo:
    python examples/replicate.py --list
    python examples/replicate.py fb15k237 exact_contract ...
    python examples/replicate.py --all          # tutte, ore di calcolo
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "external"
REPLICA = ROOT / "results" / "replica"

DATASETS = {
    "fb15k237_train.txt": (
        "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data/train.txt",
        "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae"),
    "wn18rr_train.csv": (
        "https://huggingface.co/datasets/VLyb/WN18RR/resolve/main/train.csv",
        "28f7a0b3e13d6c0b2884ed2ceef4a18087e203b2f0cdb7dc946508453688e6a0"),
    "proofwriter_val.parquet": (
        "https://huggingface.co/datasets/tasksource/proofwriter/resolve/main/data/"
        "validation-00000-of-00001-8f79b25dd5b0f2c3.parquet",
        "34281c119af707a0f13878a1938c3522750852869ae897a5a0ee3a5b97cc0ff4"),
    "fb15k_mid2name.txt": (
        "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data/FB15k_mid2name.txt",
        "4da94b8059a83bc7e08c832f573d34221d38e85030576e332a0d0e9726d13d73"),
    # archivio originale di Bordes et al. (2015), il link della pagina del dataset
    "SimpleQuestions_v2.tgz": (
        "https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz?dl=1",
        "58f65630895de4f9712eeb33458ca20538972436fd48bf5913df4765e6788bf5"),
}

# preregistrazione -> (harness, file di risultati, tempo indicativo su 12 core)
# oppure (harness, argomenti, file prodotto, file pubblicato, campi confrontati, tempo):
# con campi = None si confronta il file intero
PREREGS = {
    "fb15k237": ("fb15k237_prereg.py", "fb15k237_prereg_results.json", "~10 min"),
    "exact_contract": ("exact_prereg.py", "exact_prereg_results.json", "~1 h"),
    "dependence": ("dependence_prereg.py", "dependence_prereg_results.json", "~30 min"),
    "twins": ("twins_prereg.py", "twins_prereg_results.json", "~40 min"),
    "composition": ("composition_prereg.py", "composition_prereg_results.json", "~10 min"),
    "asymmetric": ("asymmetric_prereg.py", "asymmetric_prereg_results.json", "~20 min"),
    "deepchain": ("deepchain_prereg.py", "deepchain_prereg_results.json", "~10 min"),
    "deepchain2": ("deepchain2_prereg.py", "deepchain2_prereg_results.json", "~10 min"),
    "deepchain3": ("deepchain3_prereg.py", "deepchain3_prereg_results.json", "~25 min"),
    "sizing": ("sizing_prereg.py", "sizing_prereg_results.json", "~30 min"),
    "human_questions": ("human_questions_prereg.py", ["--memory-only"],
                        "human_questions_memory_check.json", "human_questions_prereg_results.json",
                        ("memory_pred", "memory_given_gold", "memory_ok_given_gold"), "~1 min"),
    "seed10": ("capacity_seed10.py", "capacity_seed10_rerun.json", "~3 min"),
    "clarkson": ("clarkson_comparison.py", "clarkson_comparison_results.json", "~2 min"),
    # il file pubblicato usa 150 problemi per profondità (il default dello script è 100)
    "proofwriter": ("proofwriter_eval.py", ["150"], "proofwriter_results.json",
                    "proofwriter_results.json", None, "~1 min"),
    # Fig. 3: solo la parte "proofwriter" di seed10_results.json (law4 è il target seed10)
    "proofwriter_seeds": ("proofwriter_seeds.py", [], "seed10_proofwriter_check.json",
                          "seed10_results.json", ("proofwriter",), "~1 min"),
    # test 12–14: solo l'analisi, dalle risposte dei modelli committate (gli LLM
    # non vengono interrogati; le loro scelte sono in results/escalation*_answers.json)
    "escalation": ("escalation_prereg.py", ["--stage", "analyze"], "escalation_prereg_results.json",
                   "escalation_prereg_results.json", None, "~1 min"),
    "escalation2": ("escalation2_prereg.py", [], "escalation2_prereg_results.json",
                    "escalation2_prereg_results.json", None, "~1 min"),
    "escalation3": ("escalation3_prereg.py", ["--stage", "analyze"], "escalation3_prereg_results.json",
                    "escalation3_prereg_results.json", None, "~2 min"),
    # test 18–19: sintetico (18) e FB15k-237/WN18RR (19), nessun LLM
    "equal_bits": ("equal_bits_prereg.py", "equal_bits_prereg_results.json", "~1 min"),
    "algebra": ("algebra_prereg.py", "algebra_prereg_results.json", "~9 min"),
}

# campi non deterministici, esclusi dal confronto: nel test 19 i tempi misurati
# (t_*_s per cella, time_ratio per cella e riepilogo) dipendono dalla macchina
# e dal carico; tutto il resto (accuratezze, previsioni, capacità) è confrontato
def _no_timing(d):
    if isinstance(d, dict):
        return {k: _no_timing(v) for k, v in d.items()
                if not ((k.startswith("t_") and k.endswith("_s")) or k == "time_ratio")}
    if isinstance(d, list):
        return [_no_timing(x) for x in d]
    return d


NONDET = {"algebra": _no_timing}

# i dati che ogni harness legge: si scaricano solo quelli dei target scelti
FB, WN, PW = "fb15k237_train.txt", "wn18rr_train.csv", "proofwriter_val.parquet"
NEEDS = {
    "fb15k237": {FB}, "exact_contract": {FB}, "twins": {FB, WN}, "asymmetric": {FB, WN},
    "sizing": {FB, WN}, "algebra": {FB, WN},
    "human_questions": {FB, "fb15k_mid2name.txt", "SimpleQuestions_v2.tgz"},
    "proofwriter": {PW}, "proofwriter_seeds": {PW},
    **{k: {FB, "fb15k_mid2name.txt", "SimpleQuestions_v2.tgz"}
       for k in ("escalation", "escalation2", "escalation3")},
}


# il commit in cui ogni file di risultati è stato registrato: con --code published
# (il default) l'harness gira sul codice di quel commit, non su quello attuale
COMMITS = {
    "fb15k237": "eebc030", "exact_contract": "fedd815", "dependence": "7340fb9",
    "twins": "95ab8e0", "composition": "a8db06e", "asymmetric": "359bdd1",
    "deepchain": "84532ed", "deepchain2": "64db145", "deepchain3": "dd07d49",
    "sizing": "df18e43", "human_questions": "f7ea046", "seed10": "9407eab",
    "clarkson": "28ac0da", "proofwriter": "13538cc",
    # seed10_results.json (13538cc) non ha un commit che lo scriva: lo script è
    # posteriore e gira sul codice di ca77196, dove proofwriter_eval legge data/
    "proofwriter_seeds": "ca77196",
    "escalation": "f447da8", "escalation2": "b46c28e", "escalation3": "5c9ba4b",
    "equal_bits": "728c88d", "algebra": "4fc9e8a",
}
# harness copiati dal codice attuale sull'albero di quel commit: human_questions
# per --memory-only (la pipeline è la stessa, il modello resta quello del commit);
# proofwriter perché a 13538cc leggeva il parquet da uno scratchpad privato e
# scriveva nella cartella corrente (cambiano solo i due percorsi)
OVERLAY = {"human_questions", "proofwriter", "proofwriter_seeds"}


def spec(name):
    e = PREREGS[name]
    if len(e) == 3:
        return e[0], [], e[1], e[1], None, e[2]
    return e


def fields(name, d):
    """I campi confrontati, letti dal file prodotto o da quello pubblicato."""
    if "per_question" in d:                      # file pubblicato del test 11
        return {"memory_pred": d["test"]["memory_pred"],
                "memory_given_gold": d["test"]["memory_given_gold"],
                "memory_ok_given_gold": [q["memory_ok_given_gold"] for q in d["per_question"]]}
    return {k: d[k] for k in spec(name)[4]}


def close(a, b, rel=1e-12):
    """Uguali a meno dell'arrotondamento dei float (ordine delle somme, BLAS)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(close, a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b


def fetch(names):
    DATA.mkdir(parents=True, exist_ok=True)
    need = set().union(*(NEEDS.get(n, set()) for n in names))
    for name, (url, sha) in DATASETS.items():
        if name not in need:
            continue
        path = DATA / name
        if not path.exists():
            print(f"scarico {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got}, atteso {sha}")
        print(f"ok  {name}")
    if "SimpleQuestions_v2.tgz" in need and not (DATA / "SimpleQuestions_v2").is_dir():
        import tarfile
        with tarfile.open(DATA / "SimpleQuestions_v2.tgz") as t:
            try:
                t.extractall(DATA, filter="data")
            except TypeError:                    # Python senza i filtri di tarfile
                t.extractall(DATA)


def tree_at(commit, tmp):
    """L'albero del repo al commit indicato, con data/ collegata a quella vera."""
    tree = Path(tmp) / commit
    tree.mkdir()
    archive = subprocess.run(["git", "archive", commit, "examples", "reference", "bsm", "results"],
                             check=True, cwd=ROOT, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    (tree / "data").symlink_to(ROOT / "data")
    return tree


def run(name, code, tmp):
    harness, extra, result, pub, keys, _t = spec(name)
    REPLICA.mkdir(parents=True, exist_ok=True)
    if code == "published":
        tree = tree_at(COMMITS[name], tmp)
        if name in OVERLAY:                      # opzione aggiunta dopo: il modello resta quello
            shutil.copy(ROOT / "examples" / harness, tree / "examples" / harness)
        subprocess.run([sys.executable, str(tree / "examples" / harness), *extra],
                       check=True, cwd=tree)
        shutil.copy(tree / "results" / result, REPLICA / result)
    else:
        published = ROOT / "results" / result
        backup = REPLICA / f"_published_{result}"
        # gli harness scrivono in results/: si salva il pubblicato e lo si ripristina
        if published.exists():
            shutil.copy(published, backup)
        try:
            subprocess.run([sys.executable, str(ROOT / "examples" / harness), *extra],
                           check=True, cwd=ROOT)
            shutil.copy(published, REPLICA / result)
        finally:
            if backup.exists():
                shutil.copy(backup, published)
                backup.unlink()
        if pub != result:                        # file solo di controllo, non pubblicato
            published.unlink(missing_ok=True)
    replica = json.loads((REPLICA / result).read_text())
    reference = json.loads((ROOT / "results" / pub).read_text())
    if keys is not None:
        replica, reference = fields(name, replica), fields(name, reference)
    if name in NONDET:
        replica, reference = NONDET[name](replica), NONDET[name](reference)
    if replica == reference:
        verdict = "IDENTICA"
    elif close(replica, reference):
        verdict = "IDENTICA a meno di 1e-12 relativo"
    else:
        verdict = "DIVERSA"
    print(f"{name}: replica {verdict} dai risultati pubblicati ({REPLICA / result})", flush=True)
    return verdict != "DIVERSA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--code", choices=("published", "current"), default="published",
                    help="published: il codice del commit che ha prodotto i risultati; "
                         "current: il codice di adesso (se il modello è cambiato, può dare "
                         "numeri diversi da quelli pubblicati)")
    args = ap.parse_args()
    if args.list:
        for k in PREREGS:
            h, extra, *_r, t = spec(k)
            print(f"{k:15} {' '.join([h, *extra]):40} {t}")
        return
    names = list(PREREGS) if args.all else args.names
    if not names:
        ap.error("indica almeno una preregistrazione, o --all, o --list")
    fetch(names)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results = {n: run(n, args.code, tmp) for n in names}
    print("\nriepilogo:", ", ".join(f"{n}={'ok' if s else 'DIVERSA'}" for n, s in results.items()))


if __name__ == "__main__":
    main()
