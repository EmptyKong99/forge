#!/usr/bin/env bash
# okbench eval helper — run ON the 5090 server.
#
#   bench.sh <op> <variant> [device]
#
# Reads  forge/kernels/<op>/<variant>.cu
# Deploys it into the OpenKernels submission tree, runs okbench
# (compile + correctness + timing), saves runs/<op>__<variant>.json, prints a summary.
set -euo pipefail

OP="${1:?usage: bench.sh <op> <variant> [device]}"
VARIANT="${2:?usage: bench.sh <op> <variant> [device]}"
DEV="${3:-0}"

FORGE=/nvme/share/gucheng/forge
REPO=/nvme/share/gucheng/OpenKernels
PY=/nvme/share/gucheng/anvil/.venv/bin/python
export PATH=/usr/local/cuda-13.0/bin:$PATH      # nvcc (non-interactive shells lack it)

# op -> okbench bench subcommand
case "$OP" in
  gemm_bf16_nt)                   CMD=bench-gemm-bf16 ;;
  gemm_fp8_nt_scaled_bf16)        CMD=bench-gemm-fp8 ;;
  flash_attention_bf16_fwd_bhsd)  CMD=bench-fa ;;
  linear_attention_bf16_kda_no_state) CMD=bench-linear-kda ;;
  *) echo "unknown op: $OP"; exit 1 ;;
esac

SRC="$FORGE/kernels/$OP/$VARIANT.cu"
VDIR="$REPO/submissions/5090/$OP/$VARIANT"
OUT="$FORGE/runs/${OP}__${VARIANT}.json"

[ -f "$SRC" ] || { echo "no kernel source: $SRC"; exit 1; }
mkdir -p "$VDIR" "$FORGE/runs"
cp "$SRC" "$VDIR/kernel.cu"
cat > "$VDIR/metadata.yaml" <<EOF
author: gucheng
op: $OP
variant: $VARIANT
status: draft
entry_symbol: openkernels_launch_$OP
pure_cuda: true
arch:
  - sm120a
features:
  - bf16
EOF

cd "$REPO"
"$PY" -m okbench.cli "$CMD" --op "$OP" --variant "$VARIANT" \
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
