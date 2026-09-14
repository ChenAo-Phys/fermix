"""Shared constants and helpers for the fermix kernels (row-tile layout cost model, dot precision, small
register-tile triangular inverses, block specs, input preparation and padding)."""
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

f32 = jnp.float32
PREC = {"ieee": lax.Precision.HIGHEST, "tf32x3": lax.DotAlgorithmPreset.TF32_TF32_F32_X3}
BLOCK = 32
INNER = 8
LU_CHUNK_COST = 64      # extra rows equivalent of one more register row-chunk in the LU panel step
PF_CHUNK_COST = 512     # same for the pf pair step (more reductions per step; small tail chunks on 8 warps are pure overhead)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _chunks(m):
    """Descending power-of-2 decomposition of m (a multiple of 32) into row chunks: [(offset, size), ...]."""
    out, off, rem = [], 0, m
    while rem:
        h = 1 << (rem.bit_length() - 1)
        out.append((off, h))
        off += h
        rem -= h
    return out


def _layout(m, r0, per_chunk):
    """Row-tile layout for a block with m active rows at r0: list of (offset relative to r0, height). Candidates:
    exact power-of-2 chunks; one padded tile; largest chunk + padded remainder. Padding rows sit *above* r0 (dead,
    already factored rows -> harmless to read/write) so they need r0 >= pad. Cost = rows + per_chunk * (#chunks-1)."""
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
    return min(cands, key=lambda L: sum(h for _, h in L) + per_chunk * (len(L) - 1))


def _argmax_chunks(cands, offs):
    """Global (first-occurrence) argmax over a list of 1-D candidate vectors with row offsets."""
    best_v = best_p = None
    for c, off in zip(cands, offs):
        mv = jnp.max(c, axis=0)
        ip = jnp.argmax(c, axis=0).astype(jnp.int32) + off
        if best_v is None:
            best_v, best_p = mv, ip
        else:
            best_p = lax.select(mv > best_v, ip, best_p)
            best_v = jnp.maximum(mv, best_v)
    return best_p

def _dot(a, b, prec):
    return jnp.dot(a, b, precision=PREC[prec], preferred_element_type=f32)

def _unit_lower_inv(L, bsz, static=True):
    ib = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 1)
    Ls = jnp.where(ib > jb, L, 0.0)
    X = jnp.where(ib == jb, 1.0, 0.0).astype(f32)
    def body(i, X):
        lrow = jnp.sum(jnp.where(ib == i, Ls, 0.0), axis=0)
        acc = jnp.sum(lrow[:, None] * X, axis=0)
        return jnp.where(ib == i, X - acc[None, :], X)
    if static:
        for i in range(1, bsz):
            X = body(i, X)
        return X
    return lax.fori_loop(1, bsz, body, X)


def _full(n):
    return pl.BlockSpec((None, n, n), lambda *idx: (idx[0], 0, 0))


def _vec(n):
    return pl.BlockSpec((None, n), lambda *idx: (idx[0], 0))


def _prep(a):
    a = jnp.asarray(a)
    if a.dtype != jnp.float32:
        raise TypeError(f"fastslog kernels are fp32-only, got {a.dtype}")
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected (..., n, n), got {a.shape}")
    batch = a.shape[:-2]
    n = a.shape[-1]
    return a.reshape((int(np.prod(batch)), n, n)), batch, n


def _embed(A, n, pad_block, b=BLOCK):
    """Pad A (B,n,n) to N = ceil_b(n) with `pad_block` (identity / skew identity) in the trailing rows/cols."""
    N = -(-n // b) * b
    if N == n:
        return A, N // b
    Ap = jnp.zeros((A.shape[0], N, N), f32)
    Ap = Ap.at[:, :n, :n].set(A).at[:, n:, n:].set(pad_block(N - n))
    return Ap, N // b


def _panel_warps(m):
    """1 warp per matrix up to 256 active rows (warp-shuffle reductions); wider panels need more warps to avoid
    register spills (<=512 rows -> 4, above -> 8)."""
    return 1 if m <= 256 else (4 if m <= 512 else 8)


def _lu_block(n):
    """Outer block size: 32 up to n=256 (fewer inter-panel launches), 64 above (halves trailing-update traffic)."""
    return 32 if n <= 256 else 64


def _skew_pad(m):
    """Direct sum of m/2 blocks [[0, 1], [-1, 0]]: skew-symmetric with pf = +1 exactly
    (note pf([[0, I], [-I, 0]]) = (-1)^{h(h-1)/2}, so that form cannot be used as neutral padding)."""
    return jnp.kron(jnp.eye(m // 2, dtype=f32), jnp.array([[0.0, 1.0], [-1.0, 0.0]], f32))
