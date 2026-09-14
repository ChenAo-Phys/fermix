"""Generic XLA implementations (any backend, float32 or float64) used when the Pallas kernels do not apply:
LU-based slogdet, a masked batched Parlett-Reid slogpf, and the LU parts (packed factors, permutation, zero-pivot
mask, U~-based inverse) consumed by the singular-safe gradients in _diff."""
import jax
import jax.numpy as jnp
from jax import lax


def _slog_from_lu(lu, piv):
    d = jnp.diagonal(lu, axis1=-2, axis2=-1)
    iota = jnp.arange(lu.shape[-1], dtype=piv.dtype)
    parity = jnp.sum(piv != iota, axis=-1) + jnp.sum(d < 0, axis=-1)
    sign = jnp.where(parity % 2 == 1, -1.0, 1.0).astype(lu.dtype)
    zero = jnp.any(d == 0, axis=-1)
    sign = jnp.where(zero, 0.0, sign)
    logabs = jnp.where(zero, -jnp.inf, jnp.sum(jnp.log(jnp.abs(jnp.where(d == 0, 1.0, d))), axis=-1))
    return sign, logabs.astype(lu.dtype)


def _slogdet_generic(A):
    lu, piv, _ = lax.linalg.lu(A)
    return _slog_from_lu(lu, piv)


def _lu_parts_generic(A, N, zero_singular=True):
    """(sign, logabs, invT, LU, g0, zero) with the same conventions as _inverse._lu_parts, from lax.linalg.lu on A
    padded with the identity to N x N (N >= n, matching the kernels' block padding): A[g0] = L U, invT = A^-T (leading
    n x n) formed with zero pivots replaced by 1 (finite for singular A)."""
    n = A.shape[-1]
    dt = A.dtype
    if N > n:
        Ap = jnp.zeros((A.shape[0], N, N), dt)
        A = Ap.at[:, :n, :n].set(A).at[:, n:, n:].set(jnp.eye(N - n, dtype=dt))
    lu, piv, perm = lax.linalg.lu(A)
    sign, logabs = _slog_from_lu(lu, piv)
    zero = jnp.diagonal(lu, axis1=-2, axis2=-1) == 0
    Ut = lu + zero.astype(dt)[..., None] * jnp.eye(N, dtype=dt)
    X = jax.nn.one_hot(perm, N, dtype=dt)                                       # (P I)[c, :] = e_perm[c]
    X = lax.linalg.triangular_solve(lu, X, left_side=True, lower=True, unit_diagonal=True)
    X = lax.linalg.triangular_solve(Ut, X, left_side=True, lower=False)         # U~^-1 L^-1 P
    bad = ~jnp.all(jnp.isfinite(X), axis=(-1, -2))
    if zero_singular:
        bad = bad | jnp.any(zero, axis=-1)
    invT = jnp.where(bad[:, None, None], 0.0, jnp.swapaxes(X, -1, -2))[:, :n, :n]
    return sign, logabs, invT, lu, perm, zero


def _slogpf_generic(S):
    """Batched Parlett-Reid tridiagonalisation with partial pivoting (pair steps, rank-2 updates) written with masks
    so that every step has static shapes; same pivoting convention as the kernels."""
    B, n, _ = S.shape
    dt = S.dtype
    ar = jnp.arange(n)

    def step(k, carry):
        A, sign, log = carry
        c = 2 * k
        col = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)                       # (B, n)
        kp = jnp.argmax(jnp.where(ar[None, :] > c, jnp.abs(col), -1.0), axis=1)          # pivot row in [c+1, n)
        p = jnp.where(ar[None, :] == c + 1, kp[:, None], jnp.where(ar[None, :] == kp[:, None], c + 1, ar[None, :]))
        A = jnp.take_along_axis(A, p[:, :, None], axis=1)
        A = jnp.take_along_axis(A, p[:, None, :], axis=2)
        sign = sign * jnp.where(kp != c + 1, -1.0, 1.0)
        colc = lax.dynamic_index_in_dim(A, c, axis=2, keepdims=False)
        colc1 = lax.dynamic_index_in_dim(A, c + 1, axis=2, keepdims=False)
        d = lax.dynamic_index_in_dim(colc1, c, axis=1, keepdims=False)                     # A[c, c+1]
        inv = jnp.where(d == 0, 0.0, 1.0 / jnp.where(d == 0, 1.0, d))
        mask = (ar[None, :] > c + 1).astype(dt)
        tau = -colc * inv[:, None] * mask
        w = colc1 * mask
        A = A + tau[:, :, None] * w[:, None, :] - w[:, :, None] * tau[:, None, :]
        return A, sign * jnp.sign(d), log + jnp.log(jnp.abs(d))

    _, sign, log = lax.fori_loop(0, n // 2, step, (S, jnp.ones(B, dt), jnp.zeros(B, dt)))
    return sign, log
