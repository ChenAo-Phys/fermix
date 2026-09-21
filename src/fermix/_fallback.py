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
from ._common import batch_any

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


def _pf_value(d, sgnP):
    """sgnP prod(d) through exp-log (0 if a pivot is exactly zero); pairwise trees, see
    _pairwise."""
    dsafe = jnp.where(d == 0, 1, d)
    mag = jnp.exp(_pairwise(jnp.log(jnp.abs(dsafe)), jnp.add, 0))
    return sgnP * _pairwise(jnp.sign(d), jnp.multiply, 1) * mag


def _second_zero(d, K):
    """(has2, is2): whether a member has an exact zero pivot other than K, and the mask
    of the first such pivot. The adjugate formula treats one zero pivot (K) exactly;
    with a second one the matrix can still have rank n-2 (a null row paired by the
    pivoting with a regular row: the pair entry is 0 but the partner's column is not),
    and since pf(S) S^-1 is affine in each pair entry of the factorisation, the exact
    value is the mean of the formula evaluated with that pivot lifted to +1 and to -1
    (both regular). Three or more zero pivots mean rank <= n-4 and a zero adjugate,
    which the lifted evaluations also give."""
    others = (d == 0) & (jnp.arange(d.shape[1])[None, :] != K[:, None])
    has2 = jnp.any(others, axis=1)
    first = jnp.argmax(others, axis=1)
    is2 = has2[:, None] & (jnp.arange(d.shape[1])[None, :] == first[:, None])
    return has2, is2


def _lifted_mean(d, K, evaluate, adjugate):
    """G = evaluate(d) (the adjugate formula for one exact zero pivot at most), or, for
    a batch with a member holding a second exact zero pivot, the mean of the lifted
    evaluations (see _second_zero); only the adjugate needs it (S^-1 of a singular
    member is zero-guarded)."""
    if not adjugate:
        return evaluate(d)
    has2, is2 = _second_zero(d, K)

    def lifted(_):
        return 0.5 * (evaluate(jnp.where(is2, 1, d)) + evaluate(jnp.where(is2, -1, d)))

    return lax.cond(batch_any(has2), lifted, lambda _: evaluate(d), None)


def _pf_adj_coeffs(d, sgnP, K, adjugate):
    """(a1, a2, d_K) of the adjugate formula (see _pf_adj_blocks): sgnP D_K (d_K, 1 - d_K)
    for the adjugate, (1, (1 - d_K) / d_K) for the inverse (1 / 0 -> 0)."""
    h = d.shape[1]
    isK = jnp.arange(h)[None, :] == K[:, None]
    dK = jnp.take_along_axis(d, K[:, None], axis=1)[:, 0]
    dsafe = jnp.where(d == 0, 1, d)
    logD = _pairwise(jnp.where(isK, 0, jnp.log(jnp.abs(dsafe))), jnp.add, 0)
    # 0 if another d is 0
    phase = _pairwise(jnp.where(isK, 1, jnp.sign(d)), jnp.multiply, 1)
    D = phase * jnp.exp(logD)
    if adjugate:
        return sgnP * D * dK, sgnP * D * (1 - dK), dK
    nz = dK != 0
    a2 = jnp.where(nz, (1 - dK) / jnp.where(nz, dK, 1), 0)
    return jnp.ones_like(dK), a2, dK


def _pf_adj_blocks(d, Y, R, sgnP, K, adjugate):
    """pf(S) S^-1 (adjugate=True) or S^-1 (False) in the factorisation's row order
    from the pivots d (B, h), Y = L~^-1, R = Y^T D~^-1 Y, the permutation sign sgnP and
    the index K of the smallest pivot (see _pfinv): with w = R e_{a_K}, v = Y^T e_{a_K}
    (a_K = 2K) and D_K = prod_{s != K} d_s

        -[a1 R + a2 (w v^T - v w^T)],   (a1, a2) = sgnP D_K (d_K, 1 - d_K)   [adjugate]
                                        (a1, a2) = (1, (1 - d_K) / d_K)      [inverse]

    (1 / 0 -> 0: the caller zeroes singular members). Returns (PT, pf, d_K) with
    pf = sgnP prod d."""
    a1, a2, _ = _pf_adj_coeffs(d, sgnP, K, adjugate)
    aK = (2 * K)[:, None, None]
    w = jnp.take_along_axis(R, aK, axis=2)[:, :, 0]
    v = jnp.take_along_axis(Y, aK, axis=1)[:, 0, :]
    outer = lambda x, y: x[:, :, None] * y[:, None, :]
    PT = a1[:, None, None] * R + a2[:, None, None] * (outer(w, v) - outer(v, w))
    return -PT


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
    T, L0, perm, sign, log, par = lax.fori_loop(0, n // 2, step, init)
    h = n // 2
    hp = lax.Precision.HIGHEST
    d = jnp.diagonal(T[:, 0::2, 1::2], axis1=1, axis2=2)  # T[2s, 2s+1]: the pivots
    K = _pivot_index(d, n)
    sgnP = jnp.where(par, -1.0, 1.0).astype(dt)
    eye = jnp.broadcast_to(jnp.eye(n, dtype=dt), (B, n, n))
    iperm = jnp.argsort(perm, axis=1)  # G_S[perm[f], perm[c]] = PT[f, c]

    def evaluate(dd):
        dt_ = _pair_scale(dd, K)
        # L~: the a columns scaled by 1/d~, unit diagonal
        scale = jnp.stack([1 / dt_, jnp.ones_like(dt_)], axis=2).reshape(B, n)
        L = L0 * scale[:, None, :] + jnp.eye(n, dtype=dt)
        Y = _tri_solve(L, eye, left_side=True, lower=True, unit_diagonal=True)
        Yr = Y.reshape(B, h, 2, n)
        DY = jnp.stack([Yr[:, :, 1], -Yr[:, :, 0]], axis=2) / dt_[:, :, None, None]
        R = jnp.matmul(jnp.swapaxes(Y, 1, 2), DY.reshape(B, n, n), precision=hp)
        PT = _pf_adj_blocks(dd, Y, R, sgnP, K, adjugate)
        G = jnp.take_along_axis(PT, iperm[:, :, None], axis=1)
        return jnp.take_along_axis(G, iperm[:, None, :], axis=2)

    G = _lifted_mean(d, K, evaluate, adjugate)
    if not adjugate:
        dK = jnp.take_along_axis(d, K[:, None], axis=1)[:, 0]
        bad = (dK == 0) | ~jnp.all(jnp.isfinite(G), axis=(1, 2))
        G = jnp.where(bad[:, None, None], 0, G)
    return sign, log, G, _pf_value(d, sgnP)


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
