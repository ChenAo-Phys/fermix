"""Shared constants and helpers for the fermix kernels (per-architecture and per-dtype
launch parameters, row-tile layout cost model, small register-tile triangular inverses,
block specs, input preparation and padding)."""

import dataclasses
import jax
import jax.numpy as jnp
from jax import lax
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
    # unroll the 8 column steps of the LU inner panel (Python loop) for n up to this;
    # above it a lax.fori_loop (smaller kernel body: +2-4 % on H200 for n >= 192, -5 %
    # at n = 128)
    lu_unroll_max_n: int = 1 << 30
    # extra rows equivalent of one more register row-chunk in the LU panel step, and
    # the same for the pf pair step (more reductions per step; small tail chunks on 8
    # warps are pure overhead)
    lu_chunk_cost: int = 64
    pf_chunk_cost: int = 512
    # split register row-chunks taller than this (power of 2; smaller chunks halve the
    # reduction temporaries of a step at the price of more partial reductions)
    lu_max_chunk: int = 1 << 30
    pf_max_chunk: int = 1 << 30
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
    # run the diagonal kernel's 16x16 triangular substitutions as lax.fori_loops
    # instead of unrolled code: bit-identical results, runtime 0.97-1.08x (H200), and
    # the packed-LU stage compiles in 4-14 s instead of 14 s (f32) / 128 s (c64) /
    # 259 s (c128) at n = 256
    diag_rolled: bool = True
    # compute the U blocks of a block row in a fori_loop over the column block (and
    # re-read them for the leaf inverses) instead of 20 unrolled dot chains: same
    # arithmetic, the diagonal kernel's compile time halves again
    diag_loop_d: bool = True
    # complex dots as Gauss' three real dots instead of four (fewer tensor-core
    # passes, one extra rounding); no effect on real kinds
    cplx_dot3: bool = False
    # slogdet uses the generic LU path (cuSOLVER's batched getrf, the call
    # jnp.linalg.slogdet makes, plus XLA triangular solves for the gradient) for
    # n <= this: a single 32-block costs the kernels 5 launches (~0.17 ms at
    # B = 4096 on an H200 whatever n), which cuSOLVER beats up to n = 32; from two
    # blocks on the kernels win. The forward and the gradient share the choice. det
    # is exempt (its singular-input gradient needs an exact LU, see api._lu_kernels).
    lu_generic_max_n: int = 32
    # det uses the generic LU the same way for n <= this ("small" mode in _diff): the
    # forward and the regular gradient come from cuSOLVER's batched LU, and only a batch
    # with an exact zero pivot (where that LU is not valid, see api._det_mode) reruns
    # the gradient through the kernels. 0 = kernels at every n (until measured).
    det_generic_max_n: int = 0


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
        lu_generic_max_n=40,
        lu_unroll_max_n=0,
        lu_chunk_cost=32,
        panel_rows_1w=128,
        panel_rows_2w=256,
        panel_rows_4w=512,
        pf_tm=32,
        pf_tn=32,
        diag_warps=1,
    ),
    "c64": dict(
        lu_generic_max_n=48,
        lu_unroll_max_n=0,
        panel_rows_1w=128,
        panel_rows_2w=256,
        panel_rows_4w=512,
        diag_warps=1,
    ),
    "c128": dict(
        lu_generic_max_n=48,
        lu_unroll_max_n=0,
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
    "ampere": _table(Tune()),
    "hopper": _table(
        Tune(
            lu_block_switch=128,
            lu_unroll_max_n=128,
            lu_gemm_nt=4,
            pf_panel_rows_1w=128,
            pf_panel_rows_2w=256,
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


def _layout(m, r0, per_chunk, max_chunk=1 << 30):
    """Row-tile layout for a block with m active rows at r0: list of (offset relative to
    r0, height). Candidates: exact power-of-2 chunks; one padded tile; largest chunk +
    padded remainder. Padding rows sit *above* r0 (dead, already factored rows ->
    harmless to read/write) so they need r0 >= pad. Cost is
    rows + per_chunk * (#chunks-1); chunks taller than max_chunk are split afterwards.
    """
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


def _upper_inv(U, bsz, fld, rolled=False):
    """Inverse of an upper-triangular bsz x bsz register tile by back substitution (zero
    diagonal entries -> 1); ``rolled`` as in _unit_lower_inv."""
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

    if rolled:
        return lax.fori_loop(0, bsz, lambda t, X: step(bsz - 1 - t, X), X)
    for i in reversed(range(bsz)):
        X = step(i, X)
    return X


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
    small, large = t.lu_block_small, t.lu_block_large
    if n <= t.lu_block_switch or large == small:
        return small
    extra = -(-n // large) * large - (-(-n // small) * small)
    return large if extra * 30 < n else small


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
    "Field",
    "_arch",
    "_tune",
    "_next_pow2",
    "_layout",
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
