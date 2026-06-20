#!/usr/bin/env bash
# okbench eval helper — run ON the 5090 server.
#
#   bench.sh <op> <variant> [device]
#
# Reads forge/kernels/<op>/<variant>.cu, deploys it into the OpenKernels
# submission tree, runs okbench (validate + compile + correctness + timing),
# saves runs/<op>__<variant>.json, prints a per-shape + geomean summary.
#
# The deploy->okbench->parse logic lives in tools/okeval.py (forge's own copy,
# self-contained: stdlib + yaml, no anvil import). anvil/anvil/okeval.py is an
# identical sibling -- keep the two in sync if you touch either.
set -euo pipefail

OP="${1:?usage: bench.sh <op> <variant> [device]}"
VARIANT="${2:?usage: bench.sh <op> <variant> [device]}"
DEV="${3:-6}"   # gucheng's lane on the shared box is GPU 6/7, not 0

FORGE=/nvme/share/gucheng/forge
REPO=/nvme/share/gucheng/OpenKernels
PY=/nvme/share/gucheng/anvil/.venv/bin/python   # just an interpreter w/ okbench+yaml
export PATH=/usr/local/cuda-13.0/bin:$PATH      # nvcc (non-interactive shells lack it)

SRC="$FORGE/kernels/$OP/$VARIANT.cu"
OUT="$FORGE/runs/${OP}__${VARIANT}.json"
[ -f "$SRC" ] || { echo "no kernel source: $SRC"; exit 1; }
mkdir -p "$FORGE/runs"

exec "$PY" "$FORGE/tools/okeval.py" --repo "$REPO" --op "$OP" --variant "$VARIANT" \
  --src "$SRC" --device "$DEV" --out "$OUT"
