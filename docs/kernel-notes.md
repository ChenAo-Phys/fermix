# Kernel notes: status, design, performance, dead ends

(Consolidated from the development memory notes of 2026-09-12..14. Numbers: A100-80GB, B = 4096, fp32; the DGX was
shared with other users, so only same-run comparisons are reliable.)

## Status

- `slogdet`: blocked right-looking LU with partial pivoting; outer block 32 (n ≤ 256) or 64; 8-column register-resident
  inner panels (1 warp per matrix up to 256 active rows, 4 up to 512, 8 above); 4-warp tensor-core kernels for the
  inter-panel update (K = 16 / 32), the U row block and the trailing GEMM (64×64 tiles, K = block); ping-pong
  (B,n,n) buffers; virtual pivots (rows never move; compaction folded into gather loads). Rows are held as power-of-2
  chunks chosen per block by a cost model (`_layout`; LU prefers exact chunks, cost 64 rows per extra chunk).
  ~4× faster than `jnp.linalg.slogdet` at n = 128–256 (2.7 / 14 ms), 7–10× at n = 1024.
- `slogpf`: blocked Parlett–Reid (pair steps: a = lowest unassigned row, p = argmax), tau/w vectors kept in register
  tiles and stored as G (n×32) / Gᵀ; the rank-2 updates of a block form one K = 32 GEMM over the lower-triangle tiles
  with mirrored stores (skew symmetry), exactly skew diagonal tiles; outer block always 32; layout cost 512 rows per
  extra chunk (pf prefers one padded tile). 35–50× faster than `lrux.slogpf` (2.6 / 13.6 ms at n = 128 / 256).
- Time per GiB of input is flat (13–30 ms) from n = 64 to 1024; non-power-of-2 n sit on the same curve.
- Accuracy: signs exact; rms |Δlog| 1e-4…1e-3 (same fp32 class as cuSOLVER, 2–3× above it at n ≥ 256); 3xTF32 == IEEE.
- Gradients: `custom_jvp` on all four functions. A⁻ᵀ is rebuilt from the forward LU's own buffers (`_lu_core(factors=True)`
  → `_packed_lu` assembly + diagonal kernels → block-recursive inverse on the `_bgemm` sub-block GEMM kernel → `_permT`).
  Backward ≈ 1.6–2.2× forward (n = 256: 30 vs 18 ms; `jax.grad(jnp.linalg.slogdet)` 106 ms; a cuSOLVER-based inverse
  would be 7× the forward). `slogpf`'s backward = pf forward + that LU path on the skew matrix.
- Singular inputs: exact zero pivots (zero row/column/matrix) → `slogdet`/`slogpf` gradient 0, `det`/`pf` gradient from
  the adjugate / null-space construction (see `_diff.py` docstrings); duplicate rows give roundoff-level pivots (the
  panel multiplies by 1/pivot, the blocked update does not mirror the pivot row's arithmetic) → large finite gradients.
- Fallback (non-fp32 or non-CUDA): generic XLA path with a warning; kernel/generic branch also chosen at lowering time
  via `lax.platform_dependent`.

## Time breakdown (n = 256)

slogdet forward: inner panels 7.9, GEMM 6.4, inter-panel update 1.5, U-row 1.2 ms (nsys). slogpf: panel 12.4 (70 %),
symmetric update 5.1. Gradient overhead: snapshots free, assembly + diagonal kernels ~4 ms, inverse ~15 ms
(GEMMs at 8–14 TFLOPs, transpose/permute store 1.6, isfinite guard 1).

## Why not faster (measured limits)

- The pivoted panel step is instruction/latency-bound on where/select sweeps over the (rows×8) register tile
  (6.8 µs per column over 4096 matrices at 128 rows / 1 warp); cross-warp reductions double it; strided column access or
  ref-resident panels are 2.5–4× worse; 1-warp kernels sit at 255 registers, every extra phase costs ~3× its
  instruction estimate. Pallas gives no layout / maxnreg control.
- The trailing GEMM (rank-32/64 update, C read + written per block) runs at ~1 TB/s effective.
- The pf panel is the dominant remaining cost (per pair ≈ 2 LU columns + corrections from earlier inner blocks).
- Possible levers if revisited: Mosaic-GPU (Hopper), or the panel in raw Triton/CUDA with a thread-per-row layout.

## Dead ends (do not retry without a new idea)

fused single kernel (spills); 16-wide inner panels; per-panel update launches; Newton–Schulz inverses at 1 warp;
larger GEMM tiles / 8 warps; ref-resident panels; pf update tiles over block-index columns with gapped mirror scatter;
8-warp pf update (2× slower); K = 128 GEMM tiles (Triton dot collapses); runtime block offsets (5 % slower, no compile
gain); sharing kernels through `lax.fori_loop` over blocks (XLA copies the (B,n,n) loop state, 3–5× slower);
refinement steps after the explicit-inverse solves (no accuracy gain); pure exact chunking for pf (12–19 % slower);
XLA-level block recursion for the inverse (concatenates 15 GiB, `.at[].set` rewritten to pad+add 25 GiB); cuBLAS
batched trsm for 32×32/64×64 leaves (5–12 ms); 64-leaf recursion; unrolling the diagonal rebuild into the assembly
kernel (4 min compile).
