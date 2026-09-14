# Pallas / Triton / XLA gotchas (jax 0.11, A100)

Hard-won facts from writing these kernels. Start any new kernel from this list; profile with nsys, not ncu.

## Correctness

- **Global-memory RAW/WAW hazards across threads are not ordered**: storing a tile then re-loading the same addresses
  with a different layout silently reads stale data. Put `plgpu.debug_barrier()` after every store phase whose data is
  re-read in the kernel.
- **Cross-program hazards in in-place grid kernels**: if program A gathers rows that program B writes (same
  `pallas_call`, aliased buffer), results are wrong only at large batch/grid (nondeterministic). Mask such rows out of
  the store or write to a second (ping-pong) buffer. Always re-test at full batch.
- `pl.load`/`pl.store` are gone; use `plgpu.load(ref.at[...], mask=, other=)` / `plgpu.store(...)`. Static
  out-of-bounds `pl.ds` slices are rejected at trace time; a traced start bypasses the check (masked loads are safe).
- Int-array indexing: leading int index + slice → shape (len, w) fine. Slice + trailing int index (column gather) is
  inconsistent between abstract eval and lowering → use a full 2-D int-array index instead.
- `jnp.where(c, -1.0, 1.0)` on scalars can lower to an f32*i1 multiply and crash; use `lax.select(c, f32(-1), f32(1))`.
  `j % b` in-kernel also crashed (i32*i1); use `j & (b-1)`.
- `if traced_cond:` inside a kernel fails at trace time; use `@pl.when(cond)` (closures over static loop variables are fine).
- `jnp.stack` of more than two arrays is unsupported in Triton (use `functools.reduce(jnp.minimum, ...)`).
- Dot precision: tuple/HIGHEST → IEEE fp32 FMA; DEFAULT/HIGH → TF32; `DotAlgorithmPreset.TF32_TF32_F32_X3` → 3xTF32
  (fp32-level accuracy). For pf, keep the diagonal tiles exactly skew or 3xTF32 loses ~10× accuracy at n = 512.

## Performance

- Triton IEEE fp32 dot is 3.4 TFLOPs when many small dots sit in one program, 14 TFLOPs (3xTF32: 17.7) in a proper GEMM
  grid (tile per program, `lax.fori_loop` over K). Structure, not the FMA path, is the bottleneck. cuBLAS via XLA:
  HIGHEST 10.7, TF32_X3 21 TFLOPs.
- Sub-block GEMM tiles: 128×64 (M % 128 == 0) / 64×64, K chunk 32, 4 warps, K loop as `lax.fori_loop` with
  `num_stages=3` (12–14 TFLOPs); stages = 4 or 8 warps are slower; in-place accumulate variants are C-traffic bound.
  Rank-K update tiles: K = 128 collapses (0.5 TFLOPs), K = 64 is the max.
- Strided column loads/stores are very expensive (each element its own 32 B sector). Register-resident tiles with
  where+reduce sweeps beat ref-resident panels 2.5×. Cross-warp reductions cost ~2× vs 1-warp shuffles.
- 1-warp kernels use 255 registers (8 CTAs/SM); every extra phase in the same kernel costs ~3× its instruction estimate
  (latency-bound) — split phases into separate calls. Tall register tiles: scale warps with rows (1 warp ≤ 256 rows,
  4 at 512, 8 at 1024) or the kernel spills.
- Static `jnp.zeros` scratch costs a memset pass (0.75 ms/GiB); allocate scratch as a non-aliased output of the first
  kernel that writes it.
- cuBLAS batched `triangular_solve` on tiny matrices is slow (B·8 systems of 32×32 with identity RHS: 2.5 ms; 64×64:
  6 ms); do small triangular inverses in registers (16×16 substitution + 16×16 dots) inside a kernel.

## XLA interplay

- **Slices of a buffer that is later donated to a `pallas_call`** (`input_output_aliases`): XLA fuses the slice into its
  downstream consumer, the read lands after the in-place kernel, and XLA inserts a full copy of the buffer (1 GiB per
  block). `lax.optimization_barrier((slice, buf))` before the kernel forces the slice first.
- **XLA-level block recursion is traffic-bound**: `jnp.concatenate` of quadrants re-copies every level; `.at[].set/.add`
  with static indices gets rewritten into pad + add over the whole array. Write a sub-block GEMM kernel that addresses
  (row, col) windows of the operands directly and accumulates in place.
- `jnp.take_along_axis` with a broadcast index is an element-wise gather (3× slower than `vmap(lambda m, i: m[i])`);
  `vmap(lambda m, i: m[i, c0:c1])` gathers full rows first (slice after gather).
- Identical (or offset-differing) `pallas_call`s dedupe at compile time (10 extra calls < 1 s), but a kernel that unrolls a
  big body per block took 4 min to compile — give per-block work its own kernel with a `(B, nb)` grid.
- Sharing kernels through `lax.fori_loop` over blocks cuts compile time but XLA copies the carried (B,n,n) buffers each
  iteration (3–5× slower). Runtime block offsets (`pl.multiple_of` hinted) cost ~5 %.
- Profiling: ncu counters are locked on the DGX; use `nsys profile -t cuda --cuda-graph-trace=node` (XLA runs Pallas
  kernels inside CUDA graphs; without the flag they are invisible) and the sqlite export (`registersPerThread` = 255 means
  spilling).
