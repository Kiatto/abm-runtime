# Pubblicare su PyPI

Stato al 2026-09-28: **`abm-runtime` non è su PyPI** (risponde 404). Il nome è
libero, e finché resta libero chiunque può registrarlo.

La pubblicazione usa il **Trusted Publishing**: nessun token da creare o
custodire. PyPI si fida del workflow [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
di questo repo, e di nient'altro. Gratuito.

## Una volta sola (serve il tuo account, ~5 minuti)

1. Crea un account su <https://pypi.org> e attiva la 2FA (PyPI la richiede).
2. Vai su <https://pypi.org/manage/account/publishing/> e aggiungi un
   **pending publisher** con questi valori esatti:

   | campo | valore |
   |---|---|
   | PyPI Project Name | `abm-runtime` |
   | Owner | `Kiatto` |
   | Repository name | `abm-runtime` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

3. Su GitHub: **Settings → Environments → New environment**, nome `pypi`.

## Ogni rilascio

Su GitHub: **Releases → Draft a new release**, tag `v1.0.1` (o la versione in
`pyproject.toml`), **Publish release**. Il workflow:

1. costruisce sdist e wheel;
2. verifica i metadati (`twine check`);
3. installa il wheel in un ambiente vergine ed esegue il quickstart del README
   (`abm demo` e gli import): **se il comando che il README promette non
   funziona, non pubblica**;
4. carica su PyPI.

Da quel momento `pip install abm-runtime` funziona per chiunque.

## Cosa contiene il pacchetto

Solo `abm/` (il runtime di riferimento, `reference/`). **Non contiene `bsm/`**:
il runtime bitpacked, gli esperimenti e i test di `bsm/` si usano da un clone del
repo. Import name: `abm`.
