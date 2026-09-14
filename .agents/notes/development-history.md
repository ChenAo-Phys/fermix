# Development history (from Hubbard_Next_Neighbor_SC/project/fast_linalg/SUMMARY.md, 2026-09-12..14)

File names below (`fastslog.py`, `test_fastslog.py`, `dev/...`) refer to that project folder; the code now lives in
`src/fermix/` (see AGENTS.md for the mapping).

Goal (2026-09-12): forward-only, batched (B ~ 4096), n ~ 128–256, fp32, faster than
`jnp.linalg.slogdet` (cuSOLVER batched LU) and `lrux.slogpf` (Householder, vmapped).

## Result (local A100-80GB, B=4096)

Interleaved min-of-4 timings (`dev/final_ab.py`, other GPUs of the DGX were loaded; absolute numbers move
±15% with the power cap, relative numbers within a run are reliable):

| n | `slogdet` ours | `jnp.linalg.slogdet` | `slogpf` ours (v6, symmetric update) | previous `slogpf` (v4) | `lrux.slogpf` |
|---|---|---|---|---|---|
| 128 | 2.7 ms | 11.8 ms (4.4x) | 2.6 ms | 2.9 ms | 90 ms (35x) |
| 256 | 15.0 ms | 61 ms (4.1x) | 13.5 ms | 17.0 ms | 671 ms (50x) |

Earlier single-run table (`bench_fastslog.py`, pre-v6 slogpf): n=64: 0.97 / 2.75 / 0.99 / 15.1 ms;
n=192: 10.4 / 32.3 / 11.3 / 313 ms (slogdet / jnp / slogpf-v4 / lrux).

Accuracy (vs float64 NumPy, Gaussian matrices, `test_fastslog.py`): signs exact; rms |Δlog| at n=128:
ours 1.4e-4 vs jnp-f32 4.1e-4; at n=256: ours 8.9e-4 vs jnp-f32 5.8e-4 (same fp32 class). slogpf signs match a
float64 Parlett–Reid reference; 2·log|pf| agrees with log|det| to ~5e-4 rms.

### slogpf vs slogdet cost

Parlett–Reid (pair steps, rank-2 updates) has the same 2n³/3 flops as LU when the full trailing matrix is
updated and n³/3 when only one triangle is updated. v6 exploits the skew symmetry in the trailing update
(lower-triangle tiles only, mirrored stores), which makes slogpf slightly faster than slogdet at n=256.
lrux is 30–50x slower because Householder tridiagonalisation costs 4n³/3 and its vmapped fori_loop is
memory-bound, not because pf is inherently harder.

### Near-singular / singular inputs (`test_fastslog.py near`)

A = U·diag(1,…,1,ε)·Vᵀ and S = Q·(⊕[[0,d],[−d,0]])·Qᵀ with d = (1,…,1,ε), n = 128/256, B = 64:

| ε | max Δlog slogdet (jnp f32) | max Δlog slogpf vs f64 Parlett–Reid | signs |
|---|---|---|---|
| 1e-2 | 3e-5 – 7e-5 (1e-5 – 2e-5) | 2e-5 – 3e-5 | exact |
| 1e-3 | 1.4e-4 – 1.9e-4 (1.4e-4 – 1.7e-4) | 1.4e-4 – 1.7e-4 | exact |
| 1e-4 | 1.3e-3 – 1.9e-3 (1.0e-3 – 1.2e-3) | 1.2e-3 – 1.5e-3 | exact |
| 1e-5 | 1.5e-2 – 1.8e-2 (1.4e-2 – 1.5e-2) | 1.3e-2 – 1.9e-2 | exact |
| 0 | finite tiny pivot or −inf, never NaN | finite tiny pivot, never NaN | ∈ {−1,0,1} |

Errors grow as u·cond as expected for fp32 and match jnp's fp32 LU. Zero pivots are guarded in both kernels
(reciprocal replaced by 0 ⇒ sign 0, log −inf, no NaN propagation).

## Arbitrary n up to 1024 (2026-09-13)

Sweep with ~1 GiB of fp32 input per size (`dev/sweep.py`, GPU 2 of the DGX while the other GPUs were loaded;
the shared power cap makes single sweep rows jump by up to 30%, so the interleaved min-of-20 A/B numbers in the
last column block are the reliable ones; jnp/lrux were measured in the same sweep):

| n | B | slogdet ours | jnp | speedup | slogpf ours | lrux | speedup | compile det / pf | A/B min det / pf (`dev/ab_chunk.py`, `dev/ab_pf.py`) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 65536 | 12.6 ms | 30.0 ms | 2.4x | 13.6 ms | 182 ms | 13x | 4 s / 3 s | |
| 96 | 29127 | 12.8 | 29.9 | 2.3x | 13.8 | 303 | 22x | 7 / 8 | |
| 128 | 16384 | 12.8 | 43.9 | 3.4x | 14.6 | 363 | 25x | 6 / 4 | |
| 192 | 7281 | 15.2 | 50.0 | 3.3x | 14.6 | 561 | 38x | 10 / 9 | |
| 256 | 4096 | 17.4 | 63.0 | 3.6x | 14.7 | 670 | 45x | 11 / 10 | 14.2 / 13.6 |
| 320 | 2621 | 18.9 | 84.1 | 4.5x | 17.8 | 835 | 47x | 14 / 11 | |
| 384 | 1820 | 20.0 | 85.2 | 4.3x | 21.3 | 907 | 43x | 15 / 10 | 15.7 / 17.2 |
| 448 | 1337 | 19.7 | 99.2 | 5.0x | 19.3 | 1012 | 52x | 23 / 15 | |
| 512 | 1024 | 18.1 | 108 | 5.9x | 24.7 | 1140 | 46x | 18 / 14 | 17.0 / 17.9 |
| 640 | 655 | 26.8 | 123 | 4.6x | 22.3 | – | – | 22 / 14 | |
| 768 | 455 | 29.8 | 144 | 4.8x | 26.9 | – | – | 31 / 13 | – / 25.5 |
| 896 | 334 | 29.2 | 183 | 6.3x | 29.1 | – | – | 39 / 27 | |
| 1000 | 268 | 27.0 | 225 | 8.3x | 29.1 | – | – | 41 / 15 | – / 29.3 |
| 1024 | 256 | 32.2 | 226 | 7.0x | 28.5 | – | – | 39 / 14 | 26.0 / 28.6 |

Time per GiB of input is flat (13-30 ms for both) from n=64 to 1024 and non-power-of-2 sizes sit on the same curve
as their neighbours. Before this round n=1024 took 75 / 214 ms (slogdet / slogpf) and n=384 56 / 83 ms.
jnp is 61-63 ms at n=256 in quiet moments (71-83 ms in loaded ones): compare within a run only.

Accuracy at large n (24 Gaussian matrices, vs float64): slogdet rms |Δlog| 1.3e-4 / 5.6e-4 / 1.5e-3 at
n=256 / 576 / 1024 (jnp fp32: 6.4e-5 / 2.8e-4 / 5.5e-4, i.e. we are 2-3x above cuSOLVER but in the same fp32
class; 3xTF32 == IEEE), signs exact; slogpf 2·log|pf| vs log|det|: ~1e-4 / 5e-3 / 1e-3 rms (the n=512 batch
contains an ill-conditioned matrix; identical for IEEE).

Changes that made large / non-power-of-2 sizes efficient (all in `fastslog.py`):

1. **Row-tile layout chosen per block by a cost model** (`_layout`). Candidates: exact power-of-2 chunks of the
   m active rows (e.g. 352 = 256+64+32, a list of register tiles whose argmax / row-extraction / bookkeeping are
   combined), one tile padded upward into the already-factored rows above r0, or largest chunk + padded rest.
   Cost = rows + per_chunk x (chunks-1) with per_chunk = 64 rows for LU and 512 for pf (the pf pair step has
   several more reductions per chunk, and a 32-row tail chunk on 8 warps is pure overhead). Pure exact chunking
   made slogpf 12-19% slower at n=512/1024; the hybrid matches the padded version there and removes the up-to-2x
   padding waste elsewhere (slogdet -8% at n=384). The embedding is now just padding n to a multiple of the block.
2. **Panel warps scale with the tile height**: 1 warp up to 256 rows, 4 up to 512, 8 above. The 1-warp tile
   spilled badly at 512-1024 rows (n=1024: slogdet 51 -> 39 ms, slogpf 68 -> 38 ms).
3. **Outer block 64 for n > 256 (slogdet)**: inner panels stay 8 wide (panel cost unchanged), the inter-panel
   update gets one more level (K=32 via 16x16 block substitution), the trailing GEMM has K=64 and half the C
   traffic. 5-14% faster than b=32 for n >= 256. slogpf keeps b=32 (its panel corrections grow with the number
   of inner blocks; the update is already the smaller part).
4. **Exactly skew-symmetric diagonal tiles** in the pf update (lower triangle authoritative) so that the mirrored
   stores agree bit-for-bit. Without it 3xTF32 lost ~10x accuracy at n=512; with it 3xTF32 == IEEE.

Measured limits / rejected at large n: K=128 GEMM tiles (Triton dot collapses to 0.5 TFLOPs); runtime block
offsets (`pl.multiple_of` hinted) cost ~5% and did not reduce compile time because XLA compiles every Pallas
call separately; sharing kernels through `lax.fori_loop` over blocks cuts compile time 2.5x but XLA copies the
(B,n,n) loop state each iteration (3-5x slower); one refinement step after the explicit-inverse triangular
solves changes nothing (so the ~2-3x larger rms vs cuSOLVER at n >= 256 is not from the inverses).

Compile time (first call, per n) grows with the number of blocks: ~20 s at n=256, ~30 s at 512, ~65 s at 1024.

## Tiny n and backward (2026-09-14)

**Tiny matrices use explicit polynomials** instead of the kernels: det for n ≤ 4 (2×2 minors, Sarrus, Laplace expansion
in complementary 2×2 minors at n = 4), pf for even n ≤ 6 (perfect-matching sum: 1 / 3 / 15 terms). n = 0 gives
(1, 0). Errors follow the per-matrix bound |Δlog| ≲ 3u·#terms·∏‖row_i‖/|det| (checked on 2000 Gaussian matrices per
size; signs exact wherever the bound is < 0.5). Their gradient is the polynomial's own derivative (the adjugate, always
finite) divided by the guarded det/pf.

**Both functions are differentiable** (`jax.custom_jvp`, so `grad`, `jvp`, `vmap(grad)` and batch dims all work):
d log|det A| = tr(A⁻¹ dA), d log|pf S| = ½ tr(S⁻¹ dS) with S the skew-symmetrised input (the gradient w.r.t. a general
input `a` is ½ S⁻ᵀ, skew-symmetric), d sign = 0 (as in jax). The reverse-mode cotangent of A is logabs̄ · A⁻ᵀ.

**Singular inputs.** Neither `jnp.linalg.slogdet` nor `lrux.slogpf` is singular-safe in the backward (both return
inf/NaN for a rank-deficient matrix; only `jnp.linalg.det` has the cofactor trick, and that cannot be transported to the
(sign, log) representation because exp(−∞) = 0 kills every finite cotangent). We therefore define: matrices with an
exact zero pivot (forward sign 0, log −∞) get an all-zero gradient; near-singular matrices get the exact, large
gradient; any inverse that overflows is zeroed too. Structured singularities produce exact zero pivots (zero row/column,
zero matrix) or roundoff-level pivots (duplicate rows: our panel multiplies by 1/pivot instead of dividing, and the
blocked update does not mirror the pivot row's arithmetic) — in the latter case the forward returns a finite tiny log
and the gradient is its consistent, finite ~1e6–1e9 value. Tests: `test_fastslog.py <gpu> back` (zero row/column/matrix,
duplicate rows/columns, rank-deficient, mixed batches; gradients vs float64 A⁻ᵀ for n = 1…256).

**How A⁻ᵀ is formed without a second factorisation.** The forward LU kernels leave everything needed in their buffers
(`_lu_core(..., factors=True)`): block k's L panel stays in ping-pong buffer k%2 (rows in the block's pre-pivot order),
U_k's trailing row block sits in buffer (k+1)%2, the per-block pivot rows and compaction maps compose into the final row
permutation g_k = S_k[g_{k+1}], and a snapshot of each block's b panel columns (taken before the block; an
`optimization_barrier` keeps XLA from fusing that slice into its consumer, which had cost a 1 GiB copy per block) gives
the raw pivot rows from which U_kk = L_kk⁻¹·A_raw is rebuilt (the pivot rows' stored values right of their pivot column
are stale). Then:

1. `_assemble_kernel` gathers the packed LU (B,N,N) in final row order (one row-gather per column block + select
   between the two buffers); `_diag_kernel` (one program per matrix and block) rebuilds each diagonal block in 16×16
   pieces (register substitution + tensor-core dots) and emits the 32×32 leaf inverses of L and U.
2. `_inv_unit_lower` / `_solve_upper`: block-recursive L⁻¹ and U⁻¹(L⁻¹) on `_bgemm`, a sub-block GEMM kernel that
   addresses (row, col) windows of the (B,N,N) operands directly and accumulates in place (128×64 / 64×64 tiles, K-loop
   pipelined with 3 stages, 3xTF32 or IEEE per `prec`): 1.67 N³ flops, no slice copies, no concatenations.
3. `_permT_kernel`: A⁻ᵀ = Pᵀ Zᵀ as a transpose + row-permuted store, restricted to the leading n×n block, zeroed for
   flagged matrices.

`slogpf` reuses the same LU path on the skew matrix for S⁻ᵀ (pivoting handles the zero diagonal), so its backward costs
one pf forward + one slogdet-with-inverse. `GRAD_INVERSE = "cusolver"` switches to the plain cuSOLVER LU + cuBLAS trsm
inverse (same guards) for reference.

Timings (B = 4096, A100, GPU shared with other users' jobs — compare within a row; `dev/test_backward.py --time`,
end-to-end `jax.grad` of sum(logabs), same run as the forward numbers):

| n | slogdet forward | grad(slogdet) | `jax.grad(jnp.linalg.slogdet)` | slogpf forward | grad(slogpf) |
|---|---|---|---|---|---|
| 128 | 3.7 ms | 7.8 ms | 26.9 ms | 3.7 ms | 8.5 ms |
| 256 | 18.2 ms | 29.9 ms | 105.6 ms | 19.5 ms | 38.0 ms |

Breakdown at n = 256 (`dev/test_factors.py --time`): snapshots free, assembly + diagonal kernels ~4 ms, inverse
~15 ms (of which ~1.6 ms the transpose/permute store and ~1 ms the isfinite guard); the cuSOLVER-inverse route costs
103 ms on top of the forward. So the backward is 1.6–2.2× the forward (vs ~7× for the cuSOLVER route); jax's own gradient is 3.5× slower
than ours. Compile: grad(slogdet) 19 s at n = 256 (forward
14 s); the first version of the assembly kernel unrolled the diagonal rebuild for every block and took 4 min to compile —
splitting it into the (B, nb)-grid `_diag_kernel` fixed that. Identical or offset-differing `_bgemm` calls add < 1 s.

Dead ends this round: XLA-level block recursion (`jnp.concatenate` of the quadrants: 15 GiB of concat traffic at
n = 256, 25 ms), in-place `.at[].set`/`.add` on the recursion buffers (XLA rewrites them into pad + add over the whole
array: 25 GiB), `jnp.take_along_axis` with a broadcast index for row gathers (element-wise gather, 3× slower than
`vmap(lambda m, i: m[i])`), cuBLAS batched `triangular_solve` for 32×32 / 64×64 leaves (5 / 12 ms at n = 256) and for
U_kk (2.5 ms), 64-leaf recursion (fewer nodes but the trsm leaves dominate), 8-warp or 128×128 GEMM tiles.

## Files

- `fastslog.py` — the module: `slogdet(a, prec="tf32x3"|"ieee")`, `slogpf(a, prec=..., skew_symmetrize=True)`.
  fp32 only, any batch dims, any n (padded internally, no extra flops for non-power-of-2 n), differentiable
  (custom JVP; singular-safe), explicit polynomials for det n ≤ 4 / pf n ≤ 6.
- `test_fastslog.py` — correctness suite (sizes 0…1024 incl. odd sizes, both block sizes, batch dims, singular,
  pf²=det, B=4096 stats, near-singular sweep, tiny-n polynomials, gradients incl. singular inputs and the packed-LU
  export; `python test_fastslog.py <gpu> near|back` runs only that section).
- `bench_fastslog.py` — the table above.
- `dev/` — development history (lu_v1…v13, pf_v1…v6, micro-benchmarks, `final_ab.py` interleaved bench, nsys summariser,
  large-n work: `sweep.py`, `gemm_large.py`, `fastslog_static_r0.py` (pre-chunk version), `fastslog_grouped.py`
  (rejected fori_loop sharing), `fastslog_refine.py` (rejected refinement), `test_chunked.py`).
  pf_v5 (mirror store with gaps + scalar φ lookups) was *slower* than v4; v6 fixed both.

## Design (what finally worked)

Blocked right-looking LU with partial pivoting (outer block b=32), split into 4 Pallas kernels per block,
ping-pong between two (B,n,n) buffers; rows stay in "tile order", pivoting is tracked with virtual positions
and the compaction is folded into the trailing update by gather-loads (no physical row swaps):

1. `_lu_inner_kernel` (1 warp / matrix, one call per 8-column inner panel): left-looking factorisation
   of a register-resident (mp×8) tile — column = A'−L·u, argmax, row extraction, parity bookkeeping —
   the odd panels apply the previous panel's update in a prologue (8 rank-1 sweeps); epilogue stores the
   8×8 / 16×16 unit-lower inverses into the dead diagonal block of the other buffer.
2. `_lu_update_kernel` (4 warps): panels 2,3 ← panels 0,1 with one K=16 tensor-core dot per 64-row chunk;
   pivot rows are excluded from the store (cross-program read-after-write hazard otherwise).
3. `_lu_urow_kernel` (4 warps): U row block by 16×16 block forward substitution (3 dots).
4. `_lu_gemm_kernel` (4 warps, 64×64 tiles): Q[compact rows] = P[src rows] − L[src rows]·U (3xTF32).

`slogpf`: blocked Parlett–Reid on the skew matrix with the same skeleton: pair steps (a = lowest
unassigned row, p = argmax of the corrected column), tau/w vectors kept in a (mp×8) register tile per inner
block and stored as G (n×b) and Gᵀ (b×n); corrections from earlier inner blocks read Gᵀ rows + an 8-element
gather. Trailing update (v6): the matrix is kept with *rows physically permuted by the previous block's
src map and columns compact* (X_k[φ_k(u), v] = M'[u, v]); the panel reads row a as X_k[φ_k(a), :] (φ kept
register-resident). The update kernel runs over compact (i, j) tiles, computes only tiles with i0+tm−1 ≥ j0
(2-D gather loads of M' and Hᵀ, one K=32 3xTF32 dot) and stores each tile twice: X_{k+1}[src_j, i] = −M''[i, j]
(transposed) and X_{k+1}[src_i, j] = M''[i, j] (mirror) — both are row-gather stores with contiguous columns.
This removes the old column-compaction pass and ~35% of the GEMM work (62% of tiles computed at 64×64;
the diagonal band is computed fully). Parity = stable-partition inversions + sort of the assigned rows.

## Time breakdown at n=256 (nsys, per call type, ms)

slogdet: inner panels 7.9, GEMM 6.4, inter-panel update 1.5, U-row 1.2 (+0.3 misc) ≈ 17.7 in-kernel.
slogpf v6: panel 12.4, symmetric update 5.1 (v4 was pass1 4.5 + pass2 3.5). The pf panel (70%) is now the
only big item; its per-pair cost is ~2 LU columns plus the corrections from earlier inner blocks.

## Why not faster — measured limits (see `dev/micro*.py`, memory note `pallas-triton-kernel-gotchas`)

- Pivoted panel step is instruction-bound on where/select sweeps over the (rows×8) register tile:
  6.8 µs per column over 4096 matrices at 128 rows / 1 warp; cross-warp reductions double it; strided
  column access or ref-resident panels are 2.5–4x worse; 1-D "vector leaf" formulations were not faster.
  1-warp kernels sit at 255 registers (8 CTAs/SM), so every phase is latency-bound (~3x its instruction
  estimate). Pallas gives no layout / maxnreg control.
- Trailing GEMM (rank-32 update, read+write C each block) runs at ~0.8–1.1 TB/s effective; Triton dots reach
  14 (IEEE) / 17.7 (3xTF32) TFLOPs only in proper GEMM grids.
- Left-looking / Crout variants were ruled out: the row compaction would need its own pass.
- Things tried and rejected: fused single kernel (spills), 16-wide inner panels, per-panel update launches
  with padded 16-windows, Newton–Schulz inverses at 1 warp, larger GEMM tiles / 8 warps, ref-resident panels,
  pf update tiles over block-index columns with gapped mirror scatter (pf_v5), 8-warp pf update kernel (2x slower).

## Caveats

- fp32 only; `prec="tf32x3"` uses tensor cores in the block updates (dropped term ~2⁻²², fp32-level);
  `prec="ieee"` is exact fp32 and ~5% slower.
- Exactly singular matrices give a tiny pivot or −inf (as any fp32 LU); an exact zero pivot yields sign 0 / −inf without NaN,
  and a zero gradient (see the backward section); roundoff-level pivots give large finite gradients.
- The gradient path needs ~5 extra (B,n,n) fp32 buffers (packed LU, L⁻¹/Z, A⁻ᵀ, snapshots) on top of the forward's.
- Memory: two (B,n,n) work buffers (+ the input copy unless donated); slogpf adds (B,n,32)+(B,32,n)+3×(B,n) int.
  At n=1024 use B ≲ 1024 on an 80 GB GPU (3 × 4 MB per matrix).
- First call compiles ~20 s at n=256 and ~65 s at n=1024 (one set of kernels per block); fixed n per compile.
- Not tested on Hopper (H100/H200): the Triton tile configs (64×64, 4 warps) may need retuning.
