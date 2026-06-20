"""okbench eval for forge — deploy -> okbench -> parse, in one self-contained file.

`tools/bench.sh` runs this as a script (`python tools/okeval.py ...`) and prints
`format_summary()`. Self-contained on purpose: stdlib + yaml only, no anvil import,
so forge doesn't depend on the anvil package.

NOTE: identical sibling of `anvil/anvil/okeval.py`. Kept in sync by hand (the
price of staying as two separate repos); merging the repos later removes the dup.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

# op name -> the okbench CLI subcommand that runs its stable-ABI benchmark.
# THE single source of truth for this mapping (anvil op.py + forge bench.sh).
OKBENCH_BENCH_CMD = {
    "gemm_bf16_nt": "bench-gemm-bf16",
    "gemm_fp8_nt_scaled_bf16": "bench-gemm-fp8",
    "flash_attention_bf16_fwd_bhsd": "bench-fa",
    "linear_attention_bf16_kda_no_state": "bench-linear-kda",
}


def bench_cmd(op: str) -> str:
    try:
        return OKBENCH_BENCH_CMD[op]
    except KeyError:
        raise NotImplementedError(
            f"op {op!r} not wired into okbench eval yet; supported: "
            f"{list(OKBENCH_BENCH_CMD)}"
        )


@dataclass
class EvalOutcome:
    """Plain verdict from a single eval. Anvil maps this onto its EvalResult."""
    stage: str                  # "validate" | "compile" | "bench"
    ok: bool                    # okbench ran and produced result JSON
    result: dict | None = None  # parsed okbench result JSON (when ok)
    error: str = ""             # trimmed validator / nvcc / launch output
    out_json: Path | None = None


# --- submission deploy ------------------------------------------------------

def _entry_symbol(repo: Path, op: str) -> str:
    """Authoritative entry symbol from the op spec, with a sane fallback."""
    op_yaml = repo / "ops" / op / "op.yaml"
    if op_yaml.exists():
        sym = yaml.safe_load(op_yaml.read_text()).get("entry_symbol")
        if sym:
            return str(sym)
    return f"openkernels_launch_{op}"


def write_submission(repo: Path, hardware: str, op: str, variant: str,
                     kernel_src: str, *, author: str, arch: str,
                     entry_symbol: str | None = None, notes: str = "") -> Path:
    vdir = repo / "submissions" / hardware / op / variant
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "kernel.cu").write_text(kernel_src)
    metadata = {
        "author": author,
        "op": op,
        "variant": variant,
        "status": "draft",
        "entry_symbol": entry_symbol or _entry_symbol(repo, op),
        "pure_cuda": True,
        "arch": [arch.replace("sm_", "sm")],
        "features": ["bf16"],
    }
    if notes:
        metadata["notes"] = notes[:200]
    (vdir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    return vdir


# --- okbench invocation -----------------------------------------------------

def _okbench(python: str, repo: Path, *args: str,
             timeout: int) -> subprocess.CompletedProcess:
    cmd = [python, "-m", "okbench.cli", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                          timeout=timeout)


def _unwrap_okbench_compile(text: str) -> str | None:
    """okbench raises `RuntimeError(json.dumps({...,"stderr":<nvcc>,...}))` on a
    compile failure, so the real nvcc output arrives JSON-escaped and buried under
    okbench's own python traceback. Pull the compiler stderr/stdout back out (un-
    escaped) and drop the traceback. Returns None if `text` isn't that shape."""
    marker = "RuntimeError: "
    i = text.rfind(marker)
    if i < 0:
        return None
    try:
        obj = json.loads(text[i + len(marker):])
    except (ValueError, TypeError):
        return None
    parts = [s.strip() for s in (obj.get("stderr"), obj.get("stdout")) if s and s.strip()]
    if not parts:
        return None
    cmd = (obj.get("plan") or {}).get("shell_command")
    out = "\n".join(parts)
    return out + (f"\n\n(compile command: {cmd})" if cmd else "")


def trim_error(stdout: str, stderr: str, limit: int) -> str:
    """Build the error string fed BACK to the LLM — the thing it has to fix.

    Old behaviour (tail-only `[-N:]`) was wrong: it handed the model okbench's
    python traceback / "N errors detected" tail instead of the actual nvcc
    `error:` line. Fix, in order:
      0. if this is okbench's compile RuntimeError, unwrap the embedded nvcc
         stderr (the traceback itself is noise) — see `_unwrap_okbench_compile`;
      1. ALWAYS hoist every real compiler `error:` / `ptxas` line to the top — not
         gated on length: most compile errors are short but still lead with the
         traceback / warnings, so the model needs the cause up front regardless;
      2. then the (head+tail trimmed) full output for context.
    No compiler-error lines (e.g. okbench crashed before nvcc) -> step 2 alone
    surfaces that traceback, which is then the genuine error.
    """
    text = (stdout + "\n" + stderr).strip()
    text = _unwrap_okbench_compile(text) or text

    err_lines = [ln for ln in text.splitlines()
                 if "error:" in ln.lower() or ln.lstrip().lower().startswith("ptxas")]
    head_block = ""
    if err_lines:
        key = "\n".join(err_lines[:40])[: limit // 2]
        head_block = "KEY COMPILER ERRORS:\n" + key + "\n\n"

    budget = max(800, limit - len(head_block))
    if len(text) > budget:
        head = budget * 2 // 3
        text = text[:head] + "\n\n...[middle trimmed]...\n\n" + text[-(budget - head):]
    return head_block + text


def evaluate(repo: Path, op: str, variant: str, kernel_src: str, *,
             out_json: Path, hardware: str = "5090",
             platform: str = "sm120_rtx5090", arch: str = "sm_120a",
             author: str = "gucheng", suite: str = "required_5",
             device: int = 0, python: str | None = None,
             timeout: int = 1800, validate: bool = True,
             notes: str = "") -> EvalOutcome:
    """Deploy the kernel, run okbench (validate gate + bench), parse the JSON.

    `out_json` MUST be absolute: okbench resolves --output against its cwd
    (the OpenKernels repo), so a relative path lands under the repo and we'd
    misread a working kernel as a compile failure.
    """
    repo = Path(repo).expanduser().resolve()
    python = python or sys.executable
    out_json = Path(out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    write_submission(repo, hardware, op, variant, kernel_src,
                     author=author, arch=arch, notes=notes)

    # 1. validate (cheap gate: forbidden tokens, missing symbol, pure_cuda)
    if validate:
        v = _okbench(python, repo, "validate", "--op", op,
                     "--hardware", hardware, "--variant", variant,
                     timeout=timeout)
        if v.returncode != 0:
            return EvalOutcome("validate", False,
                               error=trim_error(v.stdout, v.stderr, 4000))

    # 2. compile + correctness + timing through the stable ABI
    b = _okbench(
        python, repo, bench_cmd(op),
        "--op", op, "--variant", variant,
        "--hardware", hardware, "--platform", platform, "--arch", arch,
        "--runner-id", f"{author}_{hardware}_dev",
        "--status", "community_reported", "--suite", suite,
        "--device", str(device), "--output", str(out_json),
        timeout=timeout,
    )
    if b.returncode != 0 or not out_json.exists():
        # nvcc compile error or a launch crash both land here
        return EvalOutcome("compile", False,
                           error=trim_error(b.stdout, b.stderr, 8000),
                           out_json=out_json)

    return EvalOutcome("bench", True, result=json.loads(out_json.read_text()),
                       out_json=out_json)


# --- summary (shared by the CLI; mirrors EvalResult parsing) ----------------

def format_summary(result: dict) -> str:
    lines = []
    speedups = []
    for s in result.get("shapes", []):
        por = s.get("pure_over_reference")
        spd = (1.0 / por) if por else None
        ok = s.get("correct")
        if spd:
            speedups.append(spd)
            ms = s.get("pure_median_ms")
            lines.append(f"  {s['name']:26s} correct={ok}  {ms:8.3f}ms  {spd:.4f}x")
        else:
            lines.append(f"  {s['name']:26s} correct={ok}")
    if speedups:
        geo = statistics.geometric_mean(speedups)
        tflops = (result.get("score") or {}).get("geomean_tflops", 0)
        lines.append(f"  -> geomean {geo:.4f}x   {tflops:.1f} TFLOPS")
    return "\n".join(lines)


# --- CLI (forge bench.sh shells out to this) --------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="anvil.okeval",
        description="Deploy a kernel.cu into OpenKernels and run okbench.")
    p.add_argument("--repo", required=True, help="OpenKernels repo path")
    p.add_argument("--op", required=True)
    p.add_argument("--variant", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", help="path to the kernel .cu source")
    src.add_argument("--src-text", help="kernel source as a literal string")
    p.add_argument("--out", help="where to write okbench result JSON "
                                 "(default: a temp file)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hardware", default="5090")
    p.add_argument("--platform", default="sm120_rtx5090")
    p.add_argument("--arch", default="sm_120a")
    p.add_argument("--author", default="gucheng")
    p.add_argument("--suite", default="required_5")
    p.add_argument("--python", default=None,
                   help="python running okbench (default: this one)")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args(argv)

    kernel_src = (Path(args.src).read_text() if args.src else args.src_text)
    out_json = Path(args.out) if args.out else (
        Path(tempfile.gettempdir()) / f"{args.op}__{args.variant}.json")

    outcome = evaluate(
        args.repo, args.op, args.variant, kernel_src, out_json=out_json,
        hardware=args.hardware, platform=args.platform, arch=args.arch,
        author=args.author, suite=args.suite, device=args.device,
        python=args.python, validate=not args.no_validate,
    )
    if not outcome.ok:
        print(f"[{outcome.stage} failed]\n{outcome.error}", file=sys.stderr)
        return 1
    print(format_summary(outcome.result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
