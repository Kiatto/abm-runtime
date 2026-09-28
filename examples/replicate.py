"""replicate.py — rieseguire le preregistrazioni da un clone pulito, con un comando.

Nessuna delle preregistrazioni del paper è stata replicata da qualcun altro.
Questo script non lo sostituisce, ma toglie ogni ostacolo pratico a farlo:

1. scarica i dati pubblici (FB15k-237, WN18RR, ProofWriter) da Hugging Face e
   verifica i loro sha256;
2. riesegue gli harness delle preregistrazioni scelte, scrivendo i risultati in
   una cartella a parte (results/replica/), senza toccare quelli pubblicati;
3. confronta, per ogni preregistrazione, i numeri riassuntivi della replica con
   quelli pubblicati.

Gli harness sono deterministici (seed fissati): una replica sulla stessa
piattaforma deve riprodurre i numeri pubblicati; su un'altra piattaforma deve
riprodurli se i codeword sono gli stessi bit, cosa che la CI verifica su Linux,
macOS arm64 e Windows.

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
}

# preregistrazione -> (harness, file di risultati, tempo indicativo su 12 core)
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
}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in DATASETS.items():
        path = DATA / name
        if not path.exists():
            print(f"scarico {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got}, atteso {sha}")
        print(f"ok  {name}")


def run(name):
    harness, result, _t = PREREGS[name]
    published = ROOT / "results" / result
    backup = REPLICA / f"_published_{result}"
    REPLICA.mkdir(parents=True, exist_ok=True)
    # gli harness scrivono in results/: si salva il pubblicato e lo si ripristina
    if published.exists():
        shutil.copy(published, backup)
    try:
        subprocess.run([sys.executable, str(ROOT / "examples" / harness)], check=True, cwd=ROOT)
        shutil.copy(published, REPLICA / result)
    finally:
        if backup.exists():
            shutil.copy(backup, published)
            backup.unlink()
    same = json.loads((REPLICA / result).read_text()) == json.loads(published.read_text())
    print(f"{name}: replica {'IDENTICA' if same else 'DIVERSA'} dai risultati pubblicati "
          f"({REPLICA / result})")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (h, r, t) in PREREGS.items():
            print(f"{k:15} {h:25} {t}")
        return
    names = list(PREREGS) if args.all else args.names
    if not names:
        ap.error("indica almeno una preregistrazione, o --all, o --list")
    fetch()
    results = {n: run(n) for n in names}
    print("\nriepilogo:", ", ".join(f"{n}={'ok' if s else 'DIVERSA'}" for n, s in results.items()))


if __name__ == "__main__":
    main()
