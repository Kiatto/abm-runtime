#!/usr/bin/env bash
# Rende verificabile da terzi che una preregistrazione esisteva prima del run.
#
#   tools/stamp_prereg.sh docs/preregistration/<nome>.md [examples/<harness>.py ...]
#
# 1. rifiuta file con modifiche non committate;
# 2. rifiuta se il commit non è già su origin (il push è il primo timestamp, lato GitHub);
# 3. crea una marca OpenTimestamps (<file>.ots) per ogni file: l'hash SHA-256 viene
#    ancorato a Bitcoin dai calendari pubblici, gratis; esce solo l'hash, non il testo.
# Verifica, più tardi:  ots upgrade <file>.ots && ots verify <file>.ots
set -euo pipefail
[ $# -ge 1 ] || { echo "uso: $0 file [file...]" >&2; exit 2; }
git fetch -q origin
for f in "$@"; do
    git diff --quiet HEAD -- "$f" || { echo "$f ha modifiche non committate" >&2; exit 1; }
    c=$(git log -1 --format=%H -- "$f")
    git merge-base --is-ancestor "$c" origin/master \
        || { echo "$f: il commit ${c:0:7} non è su origin/master, fai prima il push" >&2; exit 1; }
done
uv tool run --from opentimestamps-client ots stamp "$@"
for f in "$@"; do echo "$f  sha256=$(sha256sum "$f" | cut -d' ' -f1)  commit=$(git log -1 --format=%h -- "$f")"; done
