"""Generic XLA implementations (any backend; float32, float64, complex64, complex128)
used when the Pallas kernels do not apply: LU-based slogdet, a masked batched
Parlett-Reid slogpf, and the LU parts (packed factors, permutation, zero-pivot mask,
U~-based inverse) consumed by the singular-safe gradients in _diff.

Conventions shared with the kernels and with jnp.linalg.slogdet: ``sign`` has the input
dtype -- exactly +-1 / 0 for real inputs, the unit complex number prod(d / |d|) times
the permutation parity for complex ones -- and ``logabs`` has the real dtype of the
input (``jnp.finfo(dtype).dtype``)."""

import jax
import jax.numpy as jnp
from jax import lax

_tri_solve = lax.linalg.triangular_solve


def _pairwise(x, op, fill):
    """Reduce the last axis with an explicit pairwise tree of ``op`` (padding with
    ``fill``). XLA's reduce ops pick their own summation order per program, so the
    same float reduction in the forward and in the jvp's program can differ in the last
    bit; explicit elementwise ops are never reassociated, which keeps (sign, logabs)
    bit-identical between the two (the forward/backward rule of AGENTS.md)."""
    n = x.shape[-1]
    while n > 1:
        h = (n + 1) // 2
        pad = 2 * h - n
        if pad:
            x = jnp.concatenate([x, jnp.full(x.shape[:-1] + (pad,), fill, x.dtype)], -1)
        x = op(x[..., :h], x[..., h:])
        n = h
    return x[..., 0]


def _slog_from_lu(lu, piv):
    """(sign, logabs) of the matrix P L U from the diagonal of U and the LAPACK pivots.
    An exact zero pivot gives (0, -inf). The float reductions are explicit pairwise
    trees (see _pairwise); the integer ones are exact in any order."""
    d = jnp.diagonal(lu, axis1=-2, axis2=-1)
    iota = jnp.arange(lu.shape[-1], dtype=piv.dtype)
    swaps = jnp.sum(piv != iota, axis=-1)
    zero = jnp.any(d == 0, axis=-1)
    dsafe = jnp.where(d == 0, 1, d)
    if jnp.iscomplexobj(lu):
        # unit phases of the pivots; the parity only flips the sign (exact)
        sign = _pairwise(dsafe / jnp.abs(dsafe), jnp.multiply, 1)
        sign = jnp.where(swaps % 2 == 1, -sign, sign)
    else:
        parity = swaps + jnp.sum(d < 0, axis=-1)
        sign = jnp.where(parity % 2 == 1, -1.0, 1.0).astype(lu.dtype)
    sign = jnp.where(zero, 0, sign)
    logsum = _pairwise(jnp.log(jnp.abs(dsafe)), jnp.add, 0)
    logabs = jnp.where(zero, -jnp.inf, logsum)
    return sign, logabs.astype(jnp.finfo(lu.dtype).dtype)


def _slogdet_generic(A):
    lu, piv, _ = lax.linalg.lu(A)
    return _slog_from_lu(lu, piv)


def _lu_parts_generic(A, N, zero_singular=True):
    """(sign, logabs, invT, LU, g0, zero) with the same conventions as
    _inverse._lu_parts: A[g0] = L U, invT = A^-T formed with zero pivots replaced by 1
    (finite for singular A; no conjugation, so it is the holomorphic derivative of
    log det for complex A).

    N >= n is the padded size the kernel path's outputs have (a multiple of its LU
    block). The factorisation and the solves always run on the unpadded n x n system
    and the (B, N, N) outputs are assembled from them by exact identity embedding
    ([[A, 0], [0, I]] = [[L, 0], [0, I]] [[U, 0], [0, I]] in the same row order), so
    nothing here depends on N and (sign, logabs) comes from the very lu(A) the forward
    _slogdet_generic uses -- the value under differentiation is then bit-identical to a
    plain forward call."""
    n = A.shape[-1]
    dt = A.dtype
    lu, piv, perm = lax.linalg.lu(A)
    sign, logabs = _slog_from_lu(lu, piv)
    zero = jnp.diagonal(lu, axis1=-2, axis2=-1) == 0
    Ut = lu + zero.astype(dt)[..., None] * jnp.eye(n, dtype=dt)
    X = jax.nn.one_hot(perm, n, dtype=dt)  # (P I)[c, :] = e_perm[c]
    X = _tri_solve(lu, X, left_side=True, lower=True, unit_diagonal=True)
    X = _tri_solve(Ut, X, left_side=True, lower=False)  # U~^-1 L^-1 P
    bad = ~jnp.all(jnp.isfinite(X), axis=(-1, -2))
    if zero_singular:
        bad = bad | jnp.any(zero, axis=-1)
    invT = jnp.where(bad[:, None, None], 0, jnp.swapaxes(X, -1, -2))
    if N == n:
        return sign, logabs, invT, lu, perm, zero
    B, m = A.shape[0], N - n
    LU = jnp.zeros((B, N, N), dt).at[:, :n, :n].set(lu)
    LU = LU.at[:, n:, n:].set(jnp.eye(m, dtype=dt))
    tail = jnp.broadcast_to(n + jnp.arange(m, dtype=perm.dtype), (B, m))
    g0 = jnp.concatenate([perm, tail], axis=1)
    zero = jnp.concatenate([zero, jnp.zeros((B, m), bool)], axis=1)
    return sign, logabs, invT, LU, g0, zero


# ----------------------------------------------- pf / slogpf gradient pieces
def _pivot_index(d, n):
    """Index K of the pivot of smallest magnitude among the real pairs (the padding
    pairs of the kernel path, s >= n / 2, have d = 1 and are never chosen)."""
    a = jnp.abs(d)
    h = d.shape[1]
    if n // 2 < h:
        a = a.at[:, n // 2 :].set(jnp.inf)
    return jnp.argmin(a, axis=1)


def _pair_scale(d, K):
    """d~: the pivots with pivot K and every exact zero replaced by 1 (the well
    conditioned scaling of the factor, see _pfinv)."""
    isK = jnp.arange(d.shape[1])[None, :] == K[:, None]
    return jnp.where(isK | (d == 0), 1, d)


def _pf_adj_blocks(d, Y, R, sgnP, K, adjugate):
    """pf(S) S^-1 (adjugate=True) or S^-1 (False) in the factorisation's row order
    from the pivots d (B, h), Y = L~^-1, R = Y^T D~^-1 Y, the permutation sign sgnP and
    the index K of the smallest pivot (see _pfinv): with w = R e_{a_K}, v = Y^T e_{a_K}
    (a_K = 2K) and D_K = prod_{s != K} d_s

        -[a1 R + a2 (w v^T - v w^T)],   (a1, a2) = sgnP D_K (d_K, 1 - d_K)   [adjugate]
                                        (a1, a2) = (1, (1 - d_K) / d_K)      [inverse]

    (1 / 0 -> 0: the caller zeroes singular members). Returns (PT, pf, d_K) with
    pf = sgnP prod d."""
    h = d.shape[1]
    isK = jnp.arange(h)[None, :] == K[:, None]
    dK = jnp.take_along_axis(d, K[:, None], axis=1)[:, 0]
    dsafe = jnp.where(d == 0, 1, d)
    logD = jnp.sum(jnp.where(isK, 0, jnp.log(jnp.abs(dsafe))), axis=1)
    phase = jnp.prod(jnp.where(isK, 1, jnp.sign(d)), axis=1)  # 0 if another d is 0
    D = phase * jnp.exp(logD)
    pf = sgnP * D * dK
    if adjugate:
        a1, a2 = pf, sgnP * D * (1 - dK)
    else:
        nz = dK != 0
        a1 = jnp.ones_like(dK)
        a2 = jnp.where(nz, (1 - dK) / jnp.where(nz, dK, 1), 0)
    aK = (2 * K)[:, None, None]
    w = jnp.take_along_axis(R, aK, axis=2)[:, :, 0]
    v = jnp.take_along_axis(Y, aK, axis=1)[:, 0, :]
    outer = lambda x, y: x[:, :, None] * y[:, None, :]
    PT = a1[:, None, None] * R + a2[:, None, None] * (outer(w, v) - outer(v, w))
    return -PT, pf, dK


def _pf_parts_generic(S, adjugate):
    """(sign, logabs, G, pf) with the conventions of _pfinv._pf_parts, from the
    masked Parlett-Reid of _slogpf_generic extended to keep the factor L (the w / tau
    vectors of the pair steps, rows permuted along), the reduced matrix T (its pair
    entries are the pivots) and the permutation. (sign, logabs) run the very same
    scalar updates as _slogpf_generic, so they are bit-identical to it."""
    B, n, _ = S.shape
    dt = S.dtype
    ar = jnp.arange(n)
    idx = ar[None, :]

    def step(k, carry):
        A, L, perm, sign, log, par = carry
        c = 2 * k
        col = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)
        cand = jnp.where(idx > c, jnp.abs(col), -1.0)
        kp = jnp.argmax(cand, axis=1)
        moved = jnp.where(idx == kp[:, None], c + 1, idx)
        p = jnp.where(idx == c + 1, kp[:, None], moved)
        A = jnp.take_along_axis(A, p[:, :, None], axis=1)
        A = jnp.take_along_axis(A, p[:, None, :], axis=2)
        L = jnp.take_along_axis(L, p[:, :, None], axis=1)
        perm = jnp.take_along_axis(perm, p, axis=1)
        swapped = kp != c + 1
        sign = jnp.where(swapped, -sign, sign)
        colc = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)
        colc1 = lax.dynamic_index_in_dim(A, c + 1, axis=2, keepdims=False)
        d = lax.dynamic_index_in_dim(colc1, c, axis=1, keepdims=False)
        inv = jnp.where(d == 0, 0.0, 1.0 / jnp.where(d == 0, 1.0, d))
        mask = (idx > c + 1).astype(dt)
        tau = -colc * inv[:, None] * mask
        w = colc1 * mask
        A = A + tau[:, :, None] * w[:, None, :] - w[:, :, None] * tau[:, None, :]
        L = jnp.where(idx[None, :] == c, w[:, :, None], L)  # column a: M[U, p]
        L = jnp.where(idx[None, :] == c + 1, tau[:, :, None], L)  # column p: tau
        sign, log = sign * jnp.sign(d), log + jnp.log(jnp.abs(d))
        return A, L, perm, sign, log, par ^ swapped

    perm0 = jnp.broadcast_to(ar, (B, n))
    init = (S, jnp.zeros_like(S), perm0, jnp.ones(B, dt))
    init += (jnp.zeros(B, jnp.finfo(dt).dtype), jnp.zeros(B, bool))
    T, L, perm, sign, log, par = lax.fori_loop(0, n // 2, step, init)
    h = n // 2
    hp = lax.Precision.HIGHEST
    d = jnp.diagonal(T[:, 0::2, 1::2], axis1=1, axis2=2)  # T[2s, 2s+1]: the pivots
    K = _pivot_index(d, n)
    dt_ = _pair_scale(d, K)
    # L~: the a columns scaled by 1/d~, unit diagonal
    scale = jnp.stack([1 / dt_, jnp.ones_like(dt_)], axis=2).reshape(B, n)
    L = L * scale[:, None, :] + jnp.eye(n, dtype=dt)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=dt), (B, n, n))
    Y = _tri_solve(L, eye, left_side=True, lower=True, unit_diagonal=True)
    Yr = Y.reshape(B, h, 2, n)
    DY = jnp.stack([Yr[:, :, 1], -Yr[:, :, 0]], axis=2) / dt_[:, :, None, None]
    R = jnp.matmul(jnp.swapaxes(Y, 1, 2), DY.reshape(B, n, n), precision=hp)
    sgnP = jnp.where(par, -1.0, 1.0).astype(dt)
    PT, pf, dK = _pf_adj_blocks(d, Y, R, sgnP, K, adjugate)
    iperm = jnp.argsort(perm, axis=1)  # G_S[perm[f], perm[c]] = PT[f, c]
    G = jnp.take_along_axis(PT, iperm[:, :, None], axis=1)
    G = jnp.take_along_axis(G, iperm[:, None, :], axis=2)
    if not adjugate:
        bad = (dK == 0) | ~jnp.all(jnp.isfinite(G), axis=(1, 2))
        G = jnp.where(bad[:, None, None], 0, G)
    return sign, log, G, pf


def _slogpf_generic(S):
    """Batched Parlett-Reid tridiagonalisation with partial pivoting (pair steps, rank-2
    updates) written with masks so that every step has static shapes; same pivoting
    convention as the kernels. S is skew-symmetric (S^T = -S, also for complex S);
    the pivot is the entry of largest magnitude, it contributes d / |d| to the sign
    (exactly +-1 for real S) and log|d| to logabs."""
    B, n, _ = S.shape
    dt = S.dtype
    ar = jnp.arange(n)
    idx = ar[None, :]

    def step(k, carry):
        A, sign, log = carry
        c = 2 * k
        col = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)  # (B, n)
        cand = jnp.where(idx > c, jnp.abs(col), -1.0)
        kp = jnp.argmax(cand, axis=1)  # pivot row in [c+1, n)
        moved = jnp.where(idx == kp[:, None], c + 1, idx)
        p = jnp.where(idx == c + 1, kp[:, None], moved)
        A = jnp.take_along_axis(A, p[:, :, None], axis=1)
        A = jnp.take_along_axis(A, p[:, None, :], axis=2)
        sign = jnp.where(kp != c + 1, -sign, sign)
        colc = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)
        colc1 = lax.dynamic_index_in_dim(A, c + 1, axis=2, keepdims=False)
        d = lax.dynamic_index_in_dim(colc1, c, axis=1, keepdims=False)  # A[c, c+1]
        inv = jnp.where(d == 0, 0.0, 1.0 / jnp.where(d == 0, 1.0, d))
        mask = (idx > c + 1).astype(dt)
        tau = -colc * inv[:, None] * mask
        w = colc1 * mask
        A = A + tau[:, :, None] * w[:, None, :] - w[:, :, None] * tau[:, None, :]
        return A, sign * jnp.sign(d), log + jnp.log(jnp.abs(d))

    init = (S, jnp.ones(B, dt), jnp.zeros(B, jnp.finfo(dt).dtype))
    _, sign, log = lax.fori_loop(0, n // 2, step, init)
    return sign, log
