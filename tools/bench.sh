#!/usr/bin/env bash
# okbench eval helper — run ON the 5090 server.
#
#   bench.sh <op> <variant> [device]
#
# Reads forge/kernels/<op>/<variant>.cu, deploys it into the OpenKernels
# submission tree, runs okbench (validate + compile + correctness + timing),
# saves runs/<op>__<variant>.json, prints a per-shape + geomean summary.
#
# The deploy->okbench->parse logic lives ONCE, in anvil's okeval module
# (anvil/anvil/okeval.py); this is just a thin wrapper so forge and anvil
# can't drift. anvil is on the same venv, so we add it to PYTHONPATH.
set -euo pipefail

OP="${1:?usage: bench.sh <op> <variant> [device]}"
VARIANT="${2:?usage: bench.sh <op> <variant> [device]}"
DEV="${3:-6}"   # gucheng's lane on the shared box is GPU 6/7, not 0

FORGE=/nvme/share/gucheng/forge
REPO=/nvme/share/gucheng/OpenKernels
ANVIL=/nvme/share/gucheng/anvil
PY="$ANVIL/.venv/bin/python"
export PATH=/usr/local/cuda-13.0/bin:$PATH      # nvcc (non-interactive shells lack it)
export PYTHONPATH="$ANVIL${PYTHONPATH:+:$PYTHONPATH}"

SRC="$FORGE/kernels/$OP/$VARIANT.cu"
OUT="$FORGE/runs/${OP}__${VARIANT}.json"
[ -f "$SRC" ] || { echo "no kernel source: $SRC"; exit 1; }
mkdir -p "$FORGE/runs"

exec "$PY" -m anvil.okeval --repo "$REPO" --op "$OP" --variant "$VARIANT" \
  --src "$SRC" --device "$DEV" --out "$OUT"
