#!/usr/bin/env bash
# Generate pinned requirements files from pyproject.toml extras.
# Output: requirements/{dev,data,reports}.txt
# These are CI lockfiles for repeatable builds, not user constraints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/requirements"
mkdir -p "$OUT"

lock_extra() {
    local extra="$1"
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap "rm -rf $tmpdir" RETURN

    python -m venv "$tmpdir/venv"
    "$tmpdir/venv/bin/python" -m pip install -q --upgrade pip
    "$tmpdir/venv/bin/pip" install -q -e "$REPO_ROOT[$extra]"
    "$tmpdir/venv/bin/pip" freeze --exclude-editable > "$OUT/$extra.txt"
    echo "Locked $extra -> $OUT/$extra.txt ($(wc -l < "$OUT/$extra.txt") packages)"
}

for extra in dev data reports; do
    lock_extra "$extra"
done
