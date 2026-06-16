#!/usr/bin/env bash
# Architecture-B eval helper — run ON the 5090 server.
#
#   bench.sh <variant> [device]
#
# Reads  /nvme/share/gucheng/forge/kernels/<variant>.cu
# Deploys it into the OpenKernels submission tree, runs okbench
# (compile + correctness + timing), saves runs/<variant>.json, prints a summary.
set -euo pipefail

VARIANT="${1:?usage: bench.sh <variant> [device]}"
DEV="${2:-0}"

FORGE=/nvme/share/gucheng/forge
REPO=/nvme/share/gucheng/OpenKernels
PY=/nvme/share/gucheng/anvil/.venv/bin/python
export PATH=/usr/local/cuda-13.0/bin:$PATH      # nvcc (non-interactive shells lack it)

SRC="$FORGE/kernels/$VARIANT.cu"
VDIR="$REPO/submissions/5090/gemm_bf16_nt/$VARIANT"
OUT="$FORGE/runs/$VARIANT.json"

[ -f "$SRC" ] || { echo "no kernel source: $SRC"; exit 1; }
mkdir -p "$VDIR" "$FORGE/runs"
cp "$SRC" "$VDIR/kernel.cu"
cat > "$VDIR/metadata.yaml" <<EOF
author: gucheng
op: gemm_bf16_nt
variant: $VARIANT
status: draft
entry_symbol: openkernels_launch_gemm_bf16_nt
pure_cuda: true
arch:
  - sm120a
features:
  - bf16
EOF

cd "$REPO"
"$PY" -m okbench.cli bench-gemm-bf16 --op gemm_bf16_nt --variant "$VARIANT" \
  --hardware 5090 --platform sm120_rtx5090 --arch sm_120a \
  --runner-id gucheng_5090_dev --status community_reported --suite required_5 \
  --device "$DEV" --output "$OUT" >/dev/null

"$PY" - "$OUT" <<'PY'
import json, sys, statistics
d = json.load(open(sys.argv[1]))
sp = []
for s in d.get("shapes", []):
    por = s.get("pure_over_reference"); spd = (1.0/por) if por else None
    if spd: sp.append(spd)
    ok = s.get("correct")
    if spd:
        print(f"  {s['name']:26s} correct={ok}  {s.get('pure_median_ms'):8.3f}ms  {spd:.4f}x")
    else:
        print(f"  {s['name']:26s} correct={ok}")
if sp:
    print("  -> geomean %.4fx   %.1f TFLOPS" %
          (statistics.geometric_mean(sp), d.get("score", {}).get("geomean_tflops", 0)))
PY
