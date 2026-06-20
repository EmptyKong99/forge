# Menu / PTX: asynchronous copy + barrier model

**Type:** menu (breadth). `cp.async` is **verified-in-use** (v4/v8, → see
`../facts/cp-async.md`); `cp.async.bulk`, `mbarrier`, and TMA are **UNVERIFIED**
pointers (sm_90+, not reachable on the sm_120 target). Source: PTX ISA §9.7.

## Two async-copy generations

| Family | Min SM | Granularity | On sm_120? |
|---|---|---|---|
| `cp.async` (non-bulk) | sm_80 | per-thread, 4/8/16B global→shared | **yes** — this is the GEMM pipeline primitive |
| `cp.async.bulk` (TMA) | sm_90 | whole-tile, descriptor-driven, up to 256B | **no** (Hopper+) |

## `cp.async` (sm_80, the one forge uses)

- `cp.async.ca.shared.global` — copy through all cache levels.
- `cp.async.cg.shared.global` — **cache-global-only**; for 16B copies the L1 hint
  matters. The GEMM uses 16B (`int4` = 8 bf16) per thread.
- **Completion model (group-based):**
  - `cp.async.commit_group` — seal all `cp.async` issued so far into a group.
  - `cp.async.wait_group N` — block until all but the most recent **N** groups
    have landed. (v8: `wait_group 1` keeps 1 tile in flight = double-buffer.)
  - `cp.async.wait_all` — drain everything.
- In CUDA C this is `<cuda_pipeline.h>`: `__pipeline_memcpy_async`,
  `__pipeline_commit`, `__pipeline_wait_prior(N)`.

## `mbarrier` (sm_80 core; transaction-tracking sm_90+) — UNVERIFIED

Async barrier object in shared memory. The transaction-count flavor
(`mbarrier.expect_tx` / `complete_tx` / `try_wait`) is how `cp.async.bulk`/TMA
signal completion — i.e. it's the sm_90 pairing for bulk copy, **not needed for
sm_80 `cp.async`** (which uses the simpler commit/wait-group model above).
- `mbarrier.init`, `mbarrier.arrive[.nowait]`, `mbarrier.arrive_drop`,
  `mbarrier.test_wait` / `try_wait` (phase-based).

## `cp.async.bulk` / TMA (sm_90+) — UNVERIFIED, off-target

Whole-tile descriptor-driven copy; pairs with `mbarrier` transactions and `wgmma`.
This is the Hopper async pipeline. Listed for completeness; **not reachable on the
RTX 5090 (sm_120)** — do not reach for it on this hardware.

## Practical reading (ties to a heuristic)
Pipeline *depth* (how many groups you keep in flight via `wait_group N`) is exactly
the occupancy tradeoff in `[[pipeline-depth-vs-occupancy]]`. The async *model* here
is the menu; *how deep to go* is the heuristic.

## Cross-refs
- fact (verified): `../facts/cp-async.md`
- heuristic: `../heuristics/pipeline-depth-vs-occupancy.md`
