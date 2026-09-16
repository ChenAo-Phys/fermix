# fermix — developer / agent guide

Batched fp32 `slogdet` / `slogpf` / `det` / `pf` on CUDA GPUs with Pallas (Triton) kernels, generic XLA fallback
elsewhere, singular-safe gradients. Read this before touching the kernels; the internal design notes are in `CLAUDE.local.md`,
which is untracked and local to each checkout (`docs/` is reserved for the user-facing documentation).

## Layout

| file | contents |
|---|---|
| `src/fermix/api.py` | public functions, dtype/device check and `FermixFallbackWarning`, jitted dispatch |
| `src/fermix/_diff.py` | `custom_jvp` cores for the four functions; `_det_gradT` / `_pf_gradT` (singular-compatible gradients); `lax.platform_dependent` kernel/generic switch |
| `src/fermix/_lu.py` | blocked LU kernels (inner panel, inter-panel update, U-row, trailing GEMM) and `_lu_core` (with `factors=True` export) |
| `src/fermix/_pf.py` | blocked Parlett–Reid kernels (pair-step panel, lower-triangle mirrored update) and `_pf_core` |
| `src/fermix/_inverse.py` | packed-LU assembly + diagonal-block kernels, sub-block GEMM kernel `_bgemm`, block-recursive inverse, `_permT`, `_lu_parts`; cuSOLVER reference path |
| `src/fermix/_fallback.py` | generic XLA `slogdet`, batched masked Parlett–Reid `slogpf`, `_lu_parts_generic` |
| `src/fermix/_poly.py` | explicit polynomials (det n ≤ 4, pf n ≤ 6) |
| `src/fermix/_common.py` | constants, row-tile layout cost model, register triangular inverses, padding helpers |
| `tests/test_fermix.py` | pytest suite (CUDA GPU: the kernels, ~3 min; any other backend: the generic path, ~30 s) |
| `.github/workflows/` | CI: black + pyright (`lint.yml`), pytest on CPU / generic path (`tests.yml`) |
| `benchmarks/bench.py` | forward + gradient timings vs `jnp.linalg.slogdet` / `lrux.slogpf` |
| `CLAUDE.local.md` | untracked local notes: kernel status, measured performance/accuracy, dead ends, Pallas/Triton/XLA gotchas |

Origin: the kernels were developed as `Hubbard_Next_Neighbor_SC/project/fast_linalg/fastslog.py` (frozen); develop here.

## Workflow

- Test: `cd ~/fermix && CUDA_VISIBLE_DEVICES=<id> XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -q`.
  On the shared DGX pick the GPU with free memory (`~/.claude/scripts/local_status.sh`; CUDA ids skip the display GPU).
- Format and types: `python -m black .` and `python -m pyright` must both be clean
  (same as CI). Style: black at 88 columns; when a line has to be split, introduce a named intermediate
  instead of nesting parentheses.
- Timing on the shared DGX is ±15–30 % noisy (power cap, other users): only interleaved same-run A/B minima are
  comparable; record forward numbers from the same run as the numbers you compare against.
- Any kernel change: re-run the full suite at small batch **and** check a B = 4096 case — cross-program hazards in
  in-place grid kernels only show up at large batch.
- `CLAUDE.local.md` is gitignored, so a fresh clone has no notes: keep it up to date as you work (it is the only
  record of measured numbers, dead ends and gotchas), and never put anything there that the repo needs to ship.
  It is loaded into every agent session, so keep it dense — facts and numbers, no narrative. The long-form
  development record is in git history: `git show c640eeb:.agents/notes/development-history.md`.
- Compile time matters (one kernel set per block; ~15 s at n = 256, ~65 s at n = 1024): never unroll per-block work into
  one giant kernel body; use a `(B, nb)` grid with `program_id`-derived offsets.
