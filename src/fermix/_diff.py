"""Differentiable cores: custom JVPs for slogdet / slogpf (log-derivatives) and det / pf (adjugate-type
derivatives that stay finite for singular inputs)."""
import functools
import jax
import jax.numpy as jnp
from jax import lax
from . import _inverse
from ._common import f32
from ._inverse import _inverse_cusolver, _lu_parts
from ._lu import _lu_core
from ._pf import _pf_core
from ._poly import DET_POLY_MAX, PF_POLY_MAX, _det_poly, _pf_poly, _safe_recip, _sign_log


def _slogdet_core(A, prec, unroll_steps, block):
    n = A.shape[-1]
    if n <= DET_POLY_MAX:
        return _sign_log(_det_poly(A))
    return _lu_core(A, n, prec, unroll_steps, block)


def _slogpf_core(S, prec, upd_warps):
    n = S.shape[-1]
    if n <= PF_POLY_MAX:
        return _sign_log(_pf_poly(S))
    return _pf_core(S, n, prec, upd_warps=upd_warps)


# ============================================================================= slogdet / slogpf
@functools.lru_cache(maxsize=None)
def _slogdet_diff(n, prec, unroll_steps, block):
    """custom_jvp slogdet on (B, n, n): d log|det A| = tr(A^-1 dA), d sign = 0; zero gradient for a zero pivot."""
    @jax.custom_jvp
    def f(A):
        return _slogdet_core(A, prec, unroll_steps, block)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        if n <= DET_POLY_MAX:
            d, vjp = jax.vjp(_det_poly, A)
            adjT, = vjp(jnp.ones_like(d))                     # d det / dA = adj(A)^T: a polynomial, finite for any A
            sign, logabs = _sign_log(d)
            dot = jnp.sum(adjT * dA, axis=(-1, -2)) * _safe_recip(d)
        elif _inverse.GRAD_INVERSE == "lu":
            sign, logabs, invT, _, _, _ = _lu_parts(A, n, prec, unroll_steps, block)
            dot = jnp.sum(invT * dA, axis=(-1, -2))
        else:
            sign, logabs = f(A)
            dot = jnp.sum(jnp.swapaxes(_inverse_cusolver(A), -1, -2) * dA, axis=(-1, -2))
        return (sign, logabs), (jnp.zeros_like(sign), dot)

    return f


@functools.lru_cache(maxsize=None)
def _slogpf_diff(n, prec, upd_warps):
    """custom_jvp slogpf on skew-symmetric (B, n, n): d log|pf S| = tr(S^-1 dS) / 2, d sign = 0."""
    @jax.custom_jvp
    def f(S):
        return _slogpf_core(S, prec, upd_warps)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        if n <= PF_POLY_MAX:
            p, vjp = jax.vjp(_pf_poly, S)
            G, = vjp(jnp.ones_like(p))                        # d pf / d s_ij on the strict upper triangle
            G = G - jnp.swapaxes(G, -1, -2)                   # skew extension: d pf = sum(G * dS) / 2 for skew dS
            sign, logabs = _sign_log(p)
            dot = 0.5 * jnp.sum(G * dS, axis=(-1, -2)) * _safe_recip(p)
        else:
            sign, logabs = f(S)
            if _inverse.GRAD_INVERSE == "lu":
                _, _, invT, _, _, _ = _lu_parts(S, n, prec, True, None)     # LU of the skew matrix (pivoting handles it)
            else:
                invT = jnp.swapaxes(_inverse_cusolver(S), -1, -2)
            dot = 0.5 * jnp.sum(invT * dS, axis=(-1, -2))
        return (sign, logabs), (jnp.zeros_like(sign), dot)

    return f


# ============================================================================= det / pf: singular-compatible gradients
def _perm_sign(g0):
    """(-1)^(number of inversions) of the row permutation g0 (B, N), as f32."""
    N = g0.shape[1]
    i = jnp.arange(N)
    inv = jnp.sum((g0[:, :, None] > g0[:, None, :]) & (i[:, None] < i[None, :]), axis=(1, 2))
    return 1.0 - 2.0 * (inv % 2).astype(f32)


def _det_gradT(sign, logabs, invT, LU, g0, zero, n):
    """d det / dA = adj(A)^T from the LU parts. Regular: det * A^-T. With zero pivots at K = {k_1 <= ... <= k_m}
    (U~ = U with those set to 1, d_K = product of the other pivots):
        adj(A) = sgn(P) d_K c_K (U~^-1 e_k1)(e_km^T U~^-1 L^-1 P),  c_K = prod_i (U~^-1)[k_i, k_i+1]
    (Woodbury expansion of det(U(eps)) U(eps)^-1), exact for m <= 2; m >= 3 (rank n-1 with three exact zero pivots)
    returns 0 like the rank <= n-2 case."""
    B, N, _ = LU.shape
    m = jnp.sum(zero, axis=1)
    det = sign * jnp.exp(logabs)
    regular = det[:, None, None] * invT

    def singular(_):
        d = jnp.diagonal(LU, axis1=1, axis2=2)
        idx = jnp.arange(N)
        k1 = jnp.minimum(jnp.min(jnp.where(zero, idx, N), axis=1), N - 1)
        k2 = jnp.maximum(jnp.max(jnp.where(zero, idx, -1), axis=1), 0)
        dsafe = jnp.where(zero, 1.0, d)
        dK = jnp.prod(jnp.sign(dsafe), axis=1) * jnp.exp(jnp.sum(jnp.log(jnp.abs(dsafe)), axis=1))
        Ut = jnp.triu(LU) + zero.astype(f32)[:, :, None] * jnp.eye(N, dtype=f32)
        E = jnp.stack([jax.nn.one_hot(k1, N, dtype=f32), jax.nn.one_hot(k2, N, dtype=f32)], axis=2)
        UE = lax.linalg.triangular_solve(Ut, E, left_side=True, lower=False)              # U~^-1 e_k1, U~^-1 e_k2
        u1 = UE[:, :n, 0]
        c = jnp.take_along_axis(UE[:, :, 1], k1[:, None], axis=1)[:, 0]                  # (U~^-1)[k1, k2]
        s = _perm_sign(g0) * dK * jnp.where(m >= 2, c, 1.0)
        s = jnp.where(m >= 3, 0.0, s)
        col = jnp.take_along_axis(invT, k2[:, None, None], axis=2)[:, :, 0]               # (U~^-1 L^-1 P)^T e_k2
        G = s[:, None, None] * col[:, :, None] * u1[:, None, :]
        return jnp.where((m >= 1)[:, None, None], G, regular)

    return lax.cond(jnp.any(m >= 1), singular, lambda _: regular, None)


def _pf_gradT(sign, logabs, invT, LU, g0, zero, S, n, prec, upd_warps):
    """d pf / dS (skew-symmetric, Frobenius pairing) = pf S^-T / 2 for regular S. For singular S (a zero pivot in the
    LU of S or pf exactly 0): the two smallest pivots k1, k2 give a basis x_i = U~^-1 e_ki of the null space, the
    gradient is c0 (x2 x1^T - x1 x2^T) / 2, and c0 is fixed by one entry, d pf / d s_ij = (-1)^(i+j+1) pf(S without
    rows/cols i, j), evaluated with a forward slogpf on that minor. Three or more zero pivots (rank <= n-4) give 0."""
    B, N, _ = LU.shape
    pf = sign * jnp.exp(logabs)
    m = jnp.sum(zero, axis=1)
    sing = (m >= 1) | (sign == 0)
    regular = 0.5 * pf[:, None, None] * invT

    def singular(_):
        d = jnp.abs(jnp.diagonal(LU, axis1=1, axis2=2))
        if N > n:
            d = d.at[:, n:].set(jnp.inf)                                                 # never pick padding pivots
        order = jnp.argsort(d, axis=1)
        k1 = jnp.minimum(order[:, 0], order[:, 1])
        k2 = jnp.maximum(order[:, 0], order[:, 1])
        E = jnp.stack([jax.nn.one_hot(k1, N, dtype=f32), jax.nn.one_hot(k2, N, dtype=f32)], axis=2)     # (B, N, 2)
        mask = E[:, :, 0] + E[:, :, 1]
        newdiag = jnp.where(mask > 0, 1.0, jnp.diagonal(LU, axis1=1, axis2=2))
        Ut = jnp.triu(LU, 1) + newdiag[:, :, None] * jnp.eye(N, dtype=f32)
        X = lax.linalg.triangular_solve(Ut, E, left_side=True, lower=False)              # null vectors of U (and S)
        x1, x2 = X[:, :n, 0], X[:, :n, 1]
        W = x2[:, :, None] * x1[:, None, :] - x1[:, :, None] * x2[:, None, :]             # (B, n, n), skew
        flat = jnp.argmax(jnp.abs(jnp.triu(W, 1)).reshape(B, -1), axis=1)
        i, j = flat // n, flat % n
        Wij = jnp.take_along_axis(W.reshape(B, -1), flat[:, None], axis=1)[:, 0]
        ar = jnp.arange(n)
        drop = (ar[None, :] == i[:, None]) | (ar[None, :] == j[:, None])
        keep = jnp.argsort(jnp.where(drop, n, ar[None, :]), axis=1)[:, :n - 2]           # sorted complement
        Sm = jax.vmap(lambda s, k: s[k][:, k])(S, keep)                                    # (B, n-2, n-2)
        sm, lm = _slogpf_core(Sm, prec, upd_warps)
        Dij = jnp.where((i + j) % 2 == 0, -1.0, 1.0) * sm * jnp.exp(lm)                  # (-1)^(i+j+1) pf(minor)
        c0 = jnp.where(Wij != 0, Dij / jnp.where(Wij != 0, Wij, 1.0), 0.0)
        G = 0.5 * c0[:, None, None] * W
        G = jnp.where((m >= 3)[:, None, None], 0.0, G)
        return jnp.where(sing[:, None, None], G, regular)

    return lax.cond(jnp.any(sing), singular, lambda _: regular, None)


@functools.lru_cache(maxsize=None)
def _det_diff(n, prec, unroll_steps, block):
    """custom_jvp det on (B, n, n): d det = tr(adj(A) dA), finite for singular A."""
    @jax.custom_jvp
    def f(A):
        sign, logabs = _slogdet_core(A, prec, unroll_steps, block)
        return sign * jnp.exp(logabs)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        if n <= DET_POLY_MAX:
            d, vjp = jax.vjp(_det_poly, A)
            G, = vjp(jnp.ones_like(d))
        else:
            sign, logabs, invT, LU, g0, zero = _lu_parts(A, n, prec, unroll_steps, block, zero_singular=False)
            d = sign * jnp.exp(logabs)
            G = _det_gradT(sign, logabs, invT, LU, g0, zero, n)
        return d, jnp.sum(G * dA, axis=(-1, -2))

    return f


@functools.lru_cache(maxsize=None)
def _pf_diff(n, prec, upd_warps):
    """custom_jvp pf on skew-symmetric (B, n, n): d pf = sum(G dS) with G = pf S^-T / 2 continued to singular S."""
    @jax.custom_jvp
    def f(S):
        sign, logabs = _slogpf_core(S, prec, upd_warps)
        return sign * jnp.exp(logabs)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        if n <= PF_POLY_MAX:
            p, vjp = jax.vjp(_pf_poly, S)
            G, = vjp(jnp.ones_like(p))
            G = 0.5 * (G - jnp.swapaxes(G, -1, -2))
        else:
            sign, logabs = _slogpf_core(S, prec, upd_warps)
            p = sign * jnp.exp(logabs)
            _, _, invT, LU, g0, zero = _lu_parts(S, n, prec, True, None, zero_singular=False)
            G = _pf_gradT(sign, logabs, invT, LU, g0, zero, S, n, prec, upd_warps)
        return p, jnp.sum(G * dS, axis=(-1, -2))

    return f
