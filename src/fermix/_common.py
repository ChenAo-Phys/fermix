"""Shared constants and helpers for the fermix kernels (per-architecture and per-dtype
launch parameters, row-tile layout cost model, small register-tile triangular inverses,
block specs, input preparation and padding)."""

import dataclasses
import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_batching import custom_vmap
from jax.experimental import pallas as pl
from ._field import Field, where, vsum, recip, iszero

BLOCK = 32
INNER = 8


@dataclasses.dataclass(frozen=True)
class Tune:
    """Kernel launch parameters (tile sizes, warps, block-size switch, row-chunk cost
    model) for one GPU architecture and one dtype kind. The defaults are the A100
    float32 values; ``TUNES[arch][kind]`` holds the tables (the 64-bit / complex kinds
    start from the float32 table with the register-sensitive tile heights scaled by
    the value's register cost, see ``_derive``) and ``OVERRIDE`` lets a tuning sweep
    replace them (it is read at trace time)."""

    # outer LU block: lu_block_small for n <= lu_block_switch, lu_block_large above
    # (fewer inter-panel launches vs half the trailing-update traffic). The gradient's
    # LU uses the same switch on purpose, so that its (sign, logabs) -- the value the
    # custom_jvp returns -- is bit-identical to a plain forward call, even though its
    # own optimum is the small block for longer (on H200 64-blocks at n = 192 / 256
    # gain 5 / 13 % forward but lose 11 / 5 % in the gradient).
    lu_block_small: int = 32
    lu_block_large: int = 64
    lu_block_switch: int = 256
    # a third, 128-column block (16 inner panels, one more K = 64 level in the
    # inter-panel update, the trailing GEMM as two K = 64 dots) for n above
    # lu_block_switch_huge: at large n and a batch that fills the GPU the trailing
    # rank-b update is bound by its C read + write per block (H200, n = 4096, B = 64:
    # 76 % of slogdet's kernel time at 34 TFLOPs / 2.1 TB/s), which b = 128 halves
    lu_block_huge: int = 128
    lu_block_switch_huge: int = 2048
    # unroll the 8 column steps of the LU inner panel (Python loop) for n up to this;
    # above it a lax.fori_loop (smaller kernel body: +2-4 % on H200 for n >= 192, -5 %
    # at n = 128)
    lu_unroll_max_n: int = 1 << 30
    # extra rows equivalent of one more register row-chunk in the LU panel step, and
    # the same for the pf pair step (more reductions per step; small tail chunks on 8
    # warps are pure overhead)
    lu_chunk_cost: int = 64
    pf_chunk_cost: int = 512
    # pf panel: split register row-chunks taller than this (power of 2; smaller
    # chunks halve the reduction temporaries of a step at the price of more partial
    # reductions), only for blocks with more active rows than pf_split_above (0:
    # always). Used by complex128 on Ampere, whose pf panel spills from 768 rows:
    # 256-row chunks are 1.5x there, but a lone 512-row block is 0.91x when split.
    # (The LU panel never gains from splitting: 0.64-0.95x wherever measured.)
    pf_max_chunk: int = 1 << 30
    pf_split_above: int = 0
    # panel kernels: 1 warp per matrix up to panel_rows_1w active rows (warp-shuffle
    # reductions), 2 warps up to panel_rows_2w (half the registers per thread, so
    # twice the resident warps, but cross-warp reductions), 4 warps up to
    # panel_rows_4w, 8 up to panel_rows_8w and 16 above (otherwise the tiles spill);
    # pf_* are the same for the Parlett-Reid panel
    panel_rows_1w: int = 256
    panel_rows_2w: int = 256
    panel_rows_4w: int = 512
    panel_rows_8w: int = 1 << 30
    pf_panel_rows_1w: int = 256
    pf_panel_rows_2w: int = 256
    pf_panel_rows_4w: int = 512
    pf_panel_rows_8w: int = 1 << 30
    # inter-panel update: row chunk and warps
    lu_upd_ch: int = 64
    lu_upd_warps: int = 4
    # U-row block: column tile and warps
    lu_urow_tn: int = 64
    lu_urow_warps: int = 4
    # trailing GEMM tile and warps; lu_gemm_nt > 1 makes each program loop over that
    # many consecutive row tiles of one column strip (U strip loaded once, more memory
    # traffic in flight per resident CTA); lu_gemm_stages is Triton's num_stages for
    # that loop (measured neutral on H200)
    lu_gemm_tm: int = 64
    lu_gemm_tn: int = 64
    lu_gemm_warps: int = 4
    lu_gemm_nt: int = 1
    lu_gemm_stages: int = 1
    # pf mirrored rank-2 update: tile and warps; pf_upd_nt > 1 makes each program loop
    # over that many consecutive row tiles of one column strip (the gathered H^T strip
    # is then built once per program)
    pf_tm: int = 64
    pf_tn: int = 64
    pf_upd_warps: int = 4
    pf_upd_nt: int = 1
    # sub-block GEMM of the inverse (gradients): tall tile height used when the block
    # height is a multiple of it, square tile otherwise, K chunk, pipelining stages,
    # warps
    inv_tm_big: int = 128
    inv_tile: int = 64
    inv_tk: int = 32
    inv_stages: int = 3
    inv_warps: int = 4
    # warps of the packed-LU diagonal kernel (register-resident 16x16 block
    # substitution; 1 warp holds the float32 tiles, wider values need more warps)
    diag_warps: int = 1
    # slogdet uses the generic LU path (cuSOLVER's batched getrf, the call
    # jnp.linalg.slogdet makes, plus XLA triangular solves for the gradient) for
    # n <= this: a single 32-block costs the kernels 5 launches (~0.17 ms at
    # B = 4096 on an H200 whatever n), which cuSOLVER beats up to n = 32; from two
    # blocks on the kernels win. The forward and the gradient share the choice. det
    # has its own limit, det_generic_max_n below.
    lu_generic_max_n: int = 32
    # det uses the generic LU the same way for n <= this ("small" mode in _diff): the
    # forward and the regular gradient come from cuSOLVER's batched LU, and only a batch
    # with an exact zero pivot (where that LU is not valid, see api._det_mode) reruns
    # the gradient through the kernels. Measured on the A100 (2026-09-18): the det
    # crossovers coincide with slogdet's, and fermix's generic det gradient beats
    # jnp.linalg.det's cofactor solve by 10-40 %; the Hopper values follow its slogdet
    # crossovers (det itself unmeasured there).
    det_generic_max_n: int = 32
    # ... and again above this n (slogdet and det alike): at large n the wide kinds'
    # register panels spill, and single-matrix cuSOLVER -- which jax runs per matrix
    # at these sizes, batched or vmapped -- beats the kernels even with a batch that
    # fills the GPU (2026-09-23 sweeps, `benchmark_data.md` §17: e.g. c128 n = 4096
    # 0.55x on an H200 / 0.50x on an A100-80GB, f64 n = 8192 0.53x / n = 6144 0.62x).
    # Chosen from n alone like every dispatch (B is 1 at trace time under vmap).
    lu_generic_above_n: int = 1 << 30
    # latency mode for n above this: matrices of that size come in small batches (a
    # few dozen 8192^2 float32 matrices fill an 80 GB GPU), so one panel program per
    # matrix sits alone on its SM and the panels' *latency* per column is the cost,
    # not their throughput. The panels then run on one power-of-2 register tile per
    # block (a chunked layout costs 1.4x at B = 1, measured on the A100 at n = 2048;
    # rows past n are masked), with rolled column steps (the unrolled body spills at
    # 8 warps: rolled is 1.25x at n = 1024, B = 1), warps from the lat_* ladder below
    # and dynamic block offsets, so that all blocks of one tile class share one
    # compiled kernel (the compile time of n = 2048 was 4 min with a kernel per block)
    latency_min_n: int = 1024
    # latency mode warps: 4 up to lat_rows_4w tile rows, 8 up to lat_rows_8w, 16 up to
    # lat_rows_16w, 32 above (the LU inner panel and the pf pair-step panel)
    lat_rows_4w: int = 512
    lat_rows_8w: int = 2048
    lat_rows_16w: int = 4096
    pf_lat_rows_4w: int = 256
    pf_lat_rows_8w: int = 1024
    pf_lat_rows_16w: int = 2048


def _derive(base, regs, **over):
    """Starting table for a kind whose values cost ``regs`` float32 registers each (2 for
    float64 / complex64, 4 for complex128): the register-resident row tiles of the
    panels shrink by that factor (a tile of the same height would spill), the inverse's
    tall GEMM tile as well, the diagonal kernel gets ``regs`` warps; ``over`` are the
    measured per-kind values that replace the guesses."""
    if regs == 1:
        return dataclasses.replace(base, **over)
    big = 1 << 30

    def rows(v):
        return v if v >= big else max(64, v // regs)

    def lat(v):
        return rows(v) if regs >= 4 else v

    t = dataclasses.replace(
        base,
        panel_rows_1w=rows(base.panel_rows_1w),
        panel_rows_2w=rows(base.panel_rows_2w),
        panel_rows_4w=rows(base.panel_rows_4w),
        panel_rows_8w=rows(base.panel_rows_8w),
        pf_panel_rows_1w=rows(base.pf_panel_rows_1w),
        pf_panel_rows_2w=rows(base.pf_panel_rows_2w),
        pf_panel_rows_4w=rows(base.pf_panel_rows_4w),
        pf_panel_rows_8w=rows(base.pf_panel_rows_8w),
        # latency-mode ladders (Perlmutter A100 measurements, B = 1): complex128 needs
        # the register-scaled ladders (its 2048-row tile on 8 warps is 256 registers of
        # W per thread: slogdet n = 2048 127 ms vs 77 with 32 warps, n = 3072 219 vs 194),
        # whereas float64 / complex64 are best on the float32 ladders (32 warps on
        # their 4096-row tile is 0.85x, and the pf panel prefers 16 warps at 2048 rows)
        lat_rows_4w=lat(base.lat_rows_4w),
        lat_rows_8w=lat(base.lat_rows_8w),
        lat_rows_16w=lat(base.lat_rows_16w),
        pf_lat_rows_4w=lat(base.pf_lat_rows_4w),
        pf_lat_rows_8w=lat(base.pf_lat_rows_8w),
        pf_lat_rows_16w=lat(base.pf_lat_rows_16w),
        inv_tm_big=max(64, base.inv_tm_big // regs),
        diag_warps=regs,
    )
    return dataclasses.replace(t, **over)


REGS = {"f32": 1, "f64": 2, "c64": 2, "c128": 4}

# Per-kind values measured on the H200 (2026-09-17, tune.py sweeps at n = 128 / 256 /
# 512, interleaved ratios vs the derived table; details in CLAUDE.local.md):
# - every 64-bit / complex kind: the *rolled* panel column loop (the unrolled body
#   spills with 2-4x the registers: +22-28 % at n = 128, +12-20 % at n >= 256) and
#   diag_warps = 1 (+4-7 %);
# - float64 / complex64 LU panels: 1 warp up to 128 rows, 2 up to 256, 4 up to 512
#   (+3 % at n = 256, +7 % at 512 over the halved tiles); complex128: 4 warps for
#   129-256 rows instead of 8 (+18 % at n = 256);
# - float64 / complex128 pf update tiles 32x32 (+5-10 % / +19-30 %), complex64 keeps
#   64x64 (32x32 is 0.91x there);
# - complex128 trailing GEMM 32x64 tiles (+4 / +6 / +15 % at n = 128 / 256 / 512) and
#   32x32 inverse GEMM tiles with K = 16 (the 64x64 complex128 tiles spill: 1.9-2.0x,
#   then K = 16 +3-5 %);
# - Gauss' 3-multiplication complex dot is slower everywhere (0.6-0.99x) and off;
# - small n (benchmarks/small_n.py): cuSOLVER's batched LU beats the launch-bound
#   single-block kernels for n <= 32 (f32), 40 (f64), 48 (c64 / c128) in the forward
#   (the gradient's crossover is a little lower, but it shares the forward's path).
KIND_OVERRIDES = {
    "f64": dict(
        lu_generic_above_n=6144,
        lu_generic_max_n=40,
        det_generic_max_n=40,
        lu_unroll_max_n=0,
        lu_chunk_cost=32,
        panel_rows_1w=128,
        panel_rows_2w=256,
        panel_rows_4w=512,
        pf_tm=32,
        pf_tn=32,
        diag_warps=1,
    ),
    # the complex kinds take the 128-block only above n = 6144: with two K = 64 complex
    # dots per trailing-GEMM tile c64 slogdet at n = 3072 / 4096 read 234 / 280 ms vs
    # 181 / 224 with 64-blocks on the H200 (2026-09-23; c128 415 / 548 vs 401 / 525),
    # while at n = 8192 it is 588 vs 752 (c128 1316 vs 1531)
    "c64": dict(
        lu_generic_above_n=6144,
        lu_generic_max_n=48,
        det_generic_max_n=48,
        lu_unroll_max_n=0,
        lu_block_switch_huge=6144,
        panel_rows_1w=128,
        panel_rows_2w=256,
        panel_rows_4w=512,
        diag_warps=1,
    ),
    "c128": dict(
        lu_generic_above_n=3072,
        lu_generic_max_n=48,
        det_generic_max_n=48,
        lu_unroll_max_n=0,
        lu_block_switch_huge=6144,
        lu_gemm_tm=32,
        lu_gemm_tn=64,
        panel_rows_1w=64,
        panel_rows_2w=64,
        panel_rows_4w=256,
        pf_tm=32,
        pf_tn=32,
        inv_tm_big=64,
        inv_tile=32,
        inv_tk=16,
        diag_warps=1,
    ),
}


# Ampere (A100-80GB, Rusty workergpu072, 2026-09-18, `benchmarks/small_n.py`, B = 4096 and
# 32768, min of 30): cuSOLVER's batched LU beats the one-block kernels up to n = 32 for
# float32 and float64 in both slogdet and det (forward and gradient; the H200 value 40
# for float64 is a loss here: n = 40 kernels 1.7 ms vs generic 5.0), so det takes the
# generic LU there too ("small" mode, api._det_mode). Complex kinds: see CLAUDE.local.md.
# Ampere sweeps of the wide kinds (2026-09-19, quiet A100s: tune.py --what det/pf/grad,
# n = 128 / 256 / 512, interleaved ratios vs the table; CLAUDE.local.md):
# - the pf panels want 2 warps earlier than the derived tables say: f64 / c64 2 warps
#   from 65 rows (1.34 / 1.12 at n = 128, 1.06 / 1.03 at 256, ~1.0 at 512), c128 from
#   33 rows (1.06 / 1.10 at n = 128 / 256);
# - c64 pf update tiles 32x32 (1.11-1.13 at every n; the H200 kept 64x64 for c64);
# - f64 and c128 trailing GEMM as the 4-tile loop (1.04-1.05 / 1.06-1.09), and f64
#   64-blocks from n = 192 (1.08 at n = 256; the gradient shares the switch);
# - pf update multi-tile loop pf_upd_nt = 8 for the wide kinds (against the new base:
#   f64 1.01 / 1.08 / 1.05, c64 1.01 / 1.02 / 1.04, c128 1.00 / 1.03 / 1.03; bit-identical);
# - the c128 inverse (grad) knobs are all <= 1.0: table kept.
AMPERE_KIND_OVERRIDES = {
    # pf update 8-tile loop: 1.15x at n = 4096 B = 16 on the local A100 (2026-09-23,
    # interleaved, bit-identical); the wide kinds had it since 2026-09-19
    "f32": dict(pf_upd_nt=8),
    "f64": dict(
        KIND_OVERRIDES["f64"],
        lu_generic_above_n=4096,
        lu_block_switch_huge=2048,
        pf_upd_nt=8,
        lu_generic_max_n=32,
        det_generic_max_n=32,
        lu_gemm_nt=4,
        lu_block_switch=128,
        pf_panel_rows_1w=64,
        pf_panel_rows_2w=128,
        pf_panel_rows_4w=256,
    ),
    # the 64x64 complex64 trailing-GEMM tile spills on the A100: with it slogdet took
    # 9.8 ms at n = 48 (B = 4096) vs 1.96 with 32x64 tiles, and at n = 128 / 256 the
    # 64x64 tile is 17x / 25x slower (tune.py 2026-09-18); 32x64 as for c128. With the
    # fix the c64 crossover is 48 (forward 2.0 vs 2.0 ms at n = 48, gradient 4.3 vs 3.7)
    "c64": dict(
        KIND_OVERRIDES["c64"],
        lu_block_switch_huge=1 << 30,
        pf_upd_nt=8,
        lu_gemm_tm=32,
        lu_gemm_tn=64,
        pf_tm=32,
        pf_tn=32,
        pf_panel_rows_1w=64,
        pf_panel_rows_2w=128,
        pf_panel_rows_4w=256,
    ),
    "c128": dict(
        KIND_OVERRIDES["c128"],
        lu_block_switch_huge=1 << 30,
        pf_upd_nt=8,
        lu_gemm_nt=4,
        pf_panel_rows_1w=32,
        pf_panel_rows_2w=64,
        pf_panel_rows_4w=128,
        # pf panel blocks taller than 512 rows spill on the A100: 256-row chunks are
        # 1.50x at n = 768 (128-row ones 1.25x), while the 512-row block is best whole
        # (0.91x when split)
        pf_max_chunk=256,
        pf_split_above=512,
    ),
}


def _table(f32, overrides=KIND_OVERRIDES):
    return {
        kind: _derive(f32, regs, **overrides.get(kind, {}))
        for kind, regs in REGS.items()
    }


# Hopper values measured on an H200 (2026-09-16, jax 0.11.1): 64-blocks already win at
# n = 192 / 256 (+3 / +7 %) in the forward (the gradient loses 11 / 5 % there but
# shares the switch, see lu_block_switch), the 4-tile trailing-GEMM loop gives +3 %
# (n = 160) to +15 % (n = 1024), 2-warp pf panels for 129-256 rows +16 % at n = 256,
# rolled panel steps above n = 128 +2-4 %; tile sizes, warps, chunk costs and the
# inverse GEMM settings are the A100 values (nothing else in the sweeps beat them).
# The per-kind overrides above were measured on the H200 too and are applied to the
# Ampere table untested (they address register pressure, not the architecture).
# See CLAUDE.local.md for the measurements.
TUNES = {
    # Ampere: no 128-block for the 3xTF32 kinds -- on an A100-80GB the two K = 64
    # float32 dot slices per trailing-GEMM tile hit a cliff (slogdet n = 4096 B = 32
    # 2355 ms vs 209 with 64-blocks, 2026-09-23); float64 keeps it above n = 2048
    "ampere": _table(Tune(lu_block_switch_huge=1 << 30), AMPERE_KIND_OVERRIDES),
    "hopper": _table(
        Tune(
            lu_block_switch=128,
            lu_unroll_max_n=128,
            lu_gemm_nt=4,
            pf_panel_rows_1w=128,
            pf_panel_rows_2w=256,
            # H200, n = 4096, B = 64 (2026-09-23): the 8-tile loop of the pf update
            # is 1.34x (4 tiles 1.28x); at n <= 512 it had measured 0.99-1.03
            pf_upd_nt=8,
            # with the 128-block (two K = 64 dots per trailing-GEMM tile) two pipeline
            # stages are 1.12x at n = 4096 B = 64; neutral (<= 0.02) at n <= 1024
            lu_gemm_stages=2,
        )
    ),
}
# a tuning sweep sets this to bypass the architecture table
OVERRIDE: Tune | None = None


def _arch():
    """'hopper' for compute capability >= 9 (H100 / H200, and newer parts until they
    get their own table), 'ampere' for anything else (A100 and older, non-GPU)."""
    try:
        cc = jax.devices()[0].compute_capability
        major = int(str(cc).split(".")[0])
    except Exception:
        return "ampere"
    return "hopper" if major >= 9 else "ampere"


def _tune(kind="f32"):
    return OVERRIDE if OVERRIDE is not None else TUNES[_arch()][kind]


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _chunks(m):
    """Descending power-of-2 decomposition of m (a multiple of 32) into row chunks:
    [(offset, size), ...]."""
    out, off, rem = [], 0, m
    while rem:
        h = 1 << (rem.bit_length() - 1)
        out.append((off, h))
        off += h
        rem -= h
    return out


def _split_chunks(layout, max_chunk):
    """Split chunks taller than max_chunk (a power of 2) into max_chunk-row pieces."""
    out = []
    for off, h in layout:
        while h > max_chunk:
            out.append((off, max_chunk))
            off += max_chunk
            h -= max_chunk
        out.append((off, h))
    return out


def _layout(m, r0, per_chunk, max_chunk=1 << 30, split_above=0):
    """Row-tile layout for a block with m active rows at r0: list of (offset relative to
    r0, height). Candidates: exact power-of-2 chunks; one padded tile; largest chunk +
    padded remainder. Padding rows sit *above* r0 (dead, already factored rows ->
    harmless to read/write) so they need r0 >= pad. Cost is
    rows + per_chunk * (#chunks-1); chunks taller than max_chunk are split afterwards,
    but only for blocks with more than split_above active rows (0: always).
    """
    if split_above and m <= split_above:
        max_chunk = 1 << 30
    cands = [_chunks(m)]
    mp = _next_pow2(m)
    if mp > m and r0 >= mp - m:
        cands.append([(-(mp - m), mp)])
    h1 = 1 << (m.bit_length() - 1)
    rest = m - h1
    if rest and rest & (rest - 1):
        h2 = _next_pow2(rest)
        pad = h2 - rest
        if r0 >= pad:
            cands.append([(-pad, h1), (h1 - pad, h2)])
    cost = lambda L: sum(h for _, h in L) + per_chunk * (len(L) - 1)
    return _split_chunks(min(cands, key=cost), max_chunk)


def isum(x, axis=None):
    """Integer sum kept in int32 (jnp.sum of int32 / bool is int64 under
    jax_enable_x64, which breaks fori_loop carries and index arithmetic in kernels)."""
    return jnp.sum(x, axis=axis, dtype=jnp.int32)


@custom_vmap
def batch_any(x):
    """``jnp.any(x)`` as the predicate of a lax.cond that decides for the whole batch
    (the rare singular branches of the det / pf gradients). Frameworks vmap a
    per-sample function, so inside the vmap every call has B = 1 and the predicate is
    batched; JAX then turns the cond into a select that evaluates *both* branches on
    every call (measured on an H200: the det gradient 2.5x slower at n = 8, the pf
    gradient up to 2x). This reduction's vmap rule reduces over the mapped axis as well,
    so the predicate stays unbatched and the cond stays a cond. Results are unchanged:
    the branches agree wherever both are valid (they mask per member), the branch is
    only chosen for the vmapped batch as a whole -- exactly what a batched call does."""
    return jnp.any(x)


@batch_any.def_vmap
def _batch_any_vmap(axis_size, in_batched, x):
    del axis_size, in_batched
    return jnp.any(x), False


def _argmax_chunks(cands, offs):
    """Global (first-occurrence) argmax over a list of 1-D candidate vectors with row
    offsets."""
    # lax.argmax with an explicit int32 index type: jnp.argmax yields int64 under
    # jax_enable_x64, which the Triton lowering rejects
    best_v = jnp.max(cands[0], axis=0)
    best_p = lax.argmax(cands[0], 0, jnp.int32) + offs[0]
    for c, off in zip(cands[1:], offs[1:]):
        mv = jnp.max(c, axis=0)
        ip = lax.argmax(c, 0, jnp.int32) + off
        best_p = lax.select(mv > best_v, ip, best_p)
        best_v = jnp.maximum(mv, best_v)
    return best_p


def _unit_lower_inv(L, bsz, fld, rolled=False):
    """Inverse of the unit lower-triangular bsz x bsz register tile L (strict lower part
    used) by forward substitution; ``rolled`` runs the bsz - 1 steps as a lax.fori_loop
    (same arithmetic, a 15x smaller kernel body at bsz = 16: the diagonal kernel's
    compile time)."""
    ib = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 1)
    Ls = where(ib > jb, L, 0.0)
    X = fld.eye(bsz)

    def step(i, X):
        lrow = vsum(where(ib == i, Ls, 0.0), axis=0)
        acc = vsum(lrow[:, None] * X, axis=0)
        return where(ib == i, X - acc[None, :], X)

    if rolled:
        return lax.fori_loop(1, bsz, step, X)
    for i in range(1, bsz):
        X = step(i, X)
    return X


def _upper_inv(U, bsz, fld):
    """Inverse of an upper-triangular bsz x bsz register tile by back substitution (zero
    diagonal entries -> 1); the steps run as a lax.fori_loop (as _unit_lower_inv with
    ``rolled``)."""
    ib = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 1)
    ci = lax.broadcasted_iota(jnp.int32, (bsz,), 0)
    Us = where(ib <= jb, U, 0.0)
    d = vsum(where(ib == jb, U, 0.0), axis=1)
    d = where(iszero(d), 1.0, d)
    X = fld.zeros((bsz, bsz))

    def step(i, X):
        urow = vsum(where(ib == i, Us, 0.0), axis=0)
        acc = vsum(urow[:, None] * X, axis=0)
        di = vsum(where(ci == i, d, 0.0))
        row = (where(ci == i, 1.0, 0.0) - acc) * recip(di)
        return where(ib == i, row[None, :], X)

    return lax.fori_loop(0, bsz, lambda t, X: step(bsz - 1 - t, X), X)


def _full(n):
    return pl.BlockSpec((None, n, n), lambda *idx: (idx[0], 0, 0))


def _vec(n):
    return pl.BlockSpec((None, n), lambda *idx: (idx[0], 0))


def _embed(A, n, pad_block, fld, b=BLOCK):
    """Pad the parts A of a (B, n, n) batch to N = ceil_b(n) with `pad_block(m, dtype)`
    (identity / skew identity) in the trailing rows/cols of the real part."""
    N = -(-n // b) * b
    if N == n:
        return A, N // b
    out = []
    for c, Ac in enumerate(A):
        Ap = jnp.zeros((Ac.shape[0], N, N), fld.real).at[:, :n, :n].set(Ac)
        if c == 0:
            Ap = Ap.at[:, n:, n:].set(pad_block(N - n, fld.real))
        out.append(Ap)
    return tuple(out), N // b


def _lat_layout(m, r0, N):
    """Latency-mode row tile of a block with m = N - r0 active rows: one power-of-2
    tile of mp = next_pow2(m) rows as (absolute start row, height). It ends at N when
    the matrix has room for it above r0 (the dead rows above r0 hold finished U rows
    and are harmless to read and to rewrite unchanged); otherwise it starts at row 0
    and overhangs N, and the kernels mask the rows past N. The tile position depends
    only on the class mp, so blocks of one class share a kernel."""
    mp = _next_pow2(m)
    return ((max(N - mp, 0), mp),)


def _lat_warps(mp, t, pf=False):
    """Latency-mode warps for a tile of mp rows (see Tune.lat_rows_*)."""
    if pf:
        ladder = (t.pf_lat_rows_4w, t.pf_lat_rows_8w, t.pf_lat_rows_16w)
    else:
        ladder = (t.lat_rows_4w, t.lat_rows_8w, t.lat_rows_16w)
    for limit, warps in zip(ladder, (4, 8, 16)):
        if mp <= limit:
            return warps
    return 32


def _grid_class(x):
    """x rounded up to 3 significant bits (a multiple of 2^(bitlength - 3)): the
    grid size of a latency-mode kernel, so that the blocks fall into a few classes
    that share one compiled kernel while at most 1/8 of the programs are empty."""
    if x <= 4:
        return x
    unit = 1 << (x.bit_length() - 3)
    return -(-x // unit) * unit


def _blk_spec():
    """BlockSpec of the (1,) int32 block-offset input of a latency-mode kernel."""
    return pl.BlockSpec((1,), lambda *idx: (0,))


def _split_blk(r0, rest):
    """Static r0: return it and ``rest`` unchanged. Dynamic (r0 is None): the first
    ref of ``rest`` holds the block offset; read it (hinted as a multiple of the inner
    width, the finest granularity of any block offset)."""
    if r0 is not None:
        return r0, rest
    r0 = pl.multiple_of(rest[0][0], INNER)
    return r0, rest[1:]


class _Blk:
    """The block offset of one call: ``r0`` static (an int; ``blk`` None) in the
    throughput regime, or ``r0`` None with ``blk`` = the (1,) int32 array holding it
    and ``m`` = the tile class the grids are sized for (latency mode: kernels differ
    only in their operands, so XLA compiles one per class)."""

    def __init__(self, r0, lat, mp):
        self.value = r0
        self.lat = lat
        self.r0 = None if lat else r0
        self.blk = jnp.full((1,), r0, jnp.int32) if lat else None
        self.mp = mp  # rows of the panel tile (latency mode) or n - r0

    def ins(self):
        """The extra kernel input of the latency mode (appended after the regular
        inputs, before the outputs)."""
        return [(self.blk, _blk_spec())] if self.lat else []

    def grid(self, x):
        """A grid axis of x programs, rounded to its class in the latency mode."""
        return _grid_class(x) if self.lat else x


def _panel_warps(m, t, pf=False):
    """1 warp per matrix up to panel_rows_1w active rows (warp-shuffle reductions),
    2 up to panel_rows_2w; wider panels need more warps to avoid register spills
    (<= panel_rows_4w -> 4, <= panel_rows_8w -> 8, above -> 16). pf=True reads the
    pf_panel_rows_* fields."""
    if pf:
        rows = (t.pf_panel_rows_1w, t.pf_panel_rows_2w, t.pf_panel_rows_4w)
        r8 = t.pf_panel_rows_8w
    else:
        rows = (t.panel_rows_1w, t.panel_rows_2w, t.panel_rows_4w)
        r8 = t.panel_rows_8w
    for limit, warps in zip(rows, (1, 2, 4)):
        if m <= limit:
            return warps
    return 8 if m <= r8 else 16


def _lu_block(n, t):
    """Outer block size, the same in the forward and in the gradient's LU:
    lu_block_small up to n = lu_block_switch (fewer inter-panel launches),
    lu_block_large above (halves the trailing-update traffic) unless its coarser
    padding of n costs more than it gains: the extra padded rows are charged at 3x
    their share of n (O(n^3) work) against a ~10 % gain, so the large block is used
    only when it pads at most n / 30 rows more than the small one (n = 160 padded to
    192 measured 29 % slower than 32-blocks on H200)."""
    small, large, huge = t.lu_block_small, t.lu_block_large, t.lu_block_huge
    if n <= t.lu_block_switch or large == small:
        return small
    extra = -(-n // large) * large - (-(-n // small) * small)
    b = large if extra * 30 < n else small
    if n > t.lu_block_switch_huge and huge > b:
        extra = -(-n // huge) * huge - (-(-n // b) * b)
        if extra * 30 < n:
            b = huge
    return b


def _eye_pad(m, dt):
    return jnp.eye(m, dtype=dt)


def _skew_pad(m, dt):
    """Direct sum of m/2 blocks [[0, 1], [-1, 0]]: skew-symmetric with pf = +1 exactly
    (note pf([[0, I], [-I, 0]]) = (-1)^{h(h-1)/2}, so that form cannot be used as
    neutral padding)."""
    eye = jnp.eye(m // 2, dtype=dt)
    pair = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dt)
    return jnp.kron(eye, pair)


__all__ = [
    "BLOCK",
    "INNER",
    "Tune",
    "TUNES",
    "OVERRIDE",
    "REGS",
    "KIND_OVERRIDES",
    "AMPERE_KIND_OVERRIDES",
    "Field",
    "_arch",
    "_tune",
    "_next_pow2",
    "_layout",
    "_lat_layout",
    "_lat_warps",
    "_grid_class",
    "_Blk",
    "_blk_spec",
    "_split_blk",
    "isum",
    "_argmax_chunks",
    "_unit_lower_inv",
    "_upper_inv",
    "_full",
    "_vec",
    "_embed",
    "_panel_warps",
    "_lu_block",
    "_eye_pad",
    "_skew_pad",
]
