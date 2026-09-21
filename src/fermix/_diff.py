"""Differentiable cores: custom JVPs for slogdet / slogpf (log-derivatives) and det / pf
(adjugate-type derivatives that stay finite for singular inputs). ``fast`` selects the
Pallas kernels (CUDA); otherwise the generic XLA implementations in _fallback are used
with the same derivative rules. Complex inputs follow jnp.linalg's conventions: the
sign is a unit complex number whose tangent is i Im(tr(A^-1 dA)) sign, logabs is real
with tangent Re(tr(A^-1 dA))."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from . import _inverse
from ._fallback import (
    _lu_parts_generic,
    _pairwise,
    _pf_parts_generic,
    _slogdet_generic,
    _slogpf_generic,
)
from ._field import Field
from ._common import _lu_block, _tune, batch_any, isum
from ._inverse import _lu_parts
from ._lu import _lu_core
from ._pf import _pf_core
from ._pfinv import _pf_parts
from ._poly import (
    DET_POLY_MAX,
    PF_POLY_MAX,
    _det_poly,
    _pf_poly,
    _safe_recip,
    _sign_log,
)


def _on_cuda(x, kernels, generic):
    """Run ``kernels`` when lowering for CUDA and ``generic`` elsewhere (a committed CPU
    array reaching a traced call would otherwise hit the Pallas lowering); both must
    return identical shapes/dtypes."""
    return lax.platform_dependent(x, cuda=kernels, default=generic)


def _slogdet_core(A, fld, prec, unroll_steps, block, fast):
    n = A.shape[-1]
    if n <= DET_POLY_MAX:
        return _sign_log(_det_poly(A))
    if fast:
        kernels = lambda A: _lu_core(fld.split(A), n, fld, prec, unroll_steps, block)
        return _on_cuda(A, kernels, _slogdet_generic)
    return _slogdet_generic(A)


def _slogpf_core(S, fld, prec, upd_warps, fast):
    n = S.shape[-1]
    if n <= PF_POLY_MAX:
        return _sign_log(_pf_poly(S))
    if fast:
        kernels = lambda S: _pf_core(fld.split(S), n, fld, prec, upd_warps=upd_warps)
        return _on_cuda(S, kernels, _slogpf_generic)
    return _slogpf_generic(S)


def _parts(A, n, fld, prec, unroll_steps, block, fast, zero_singular=True):
    """(sign, logabs, invT, LU, g0, zero) with LU as parts (see _field), from the
    kernels or the generic path."""

    def generic(A, N):
        sign, logabs, invT, LU, g0, zero = _lu_parts_generic(A, N, zero_singular)
        return sign, logabs, invT, fld.split(LU), g0, zero

    if fast and _inverse.GRAD_INVERSE == "lu":
        b = _lu_block(n, _tune(fld.kind)) if block is None else block
        N = -(-n // b) * b  # the kernels pad to the block size
        kernels = lambda A: _lu_parts(
            fld.split(A), n, fld, prec, unroll_steps, block, zero_singular
        )
        return _on_cuda(A, kernels, lambda A: generic(A, N))
    return generic(A, n)


def _pf_grad_parts(S, n, fld, prec, upd_warps, fast, adjugate):
    """(sign, logabs, G, pf) from the pf kernels' own factors (_pfinv._pf_parts) or
    the generic path, G = pf S^-1 (adjugate) or S^-1; (sign, logabs) are the
    forward's, bit-identical to a plain call, so the jvp rules return them as the
    primal (one Parlett-Reid run serves value and gradient)."""
    generic = lambda S: _pf_parts_generic(S, adjugate)
    if fast:
        kernels = lambda S: _pf_parts(fld.split(S), n, fld, prec, upd_warps, adjugate)
        return _on_cuda(S, kernels, generic)
    return generic(S)


def _log_tangents(sign, dot, fld):
    """Tangents of (sign, logabs) from dot = tr(A^-1 dA) (jnp.linalg's convention:
    the phase moves with Im dot, the real sign never moves)."""
    if fld.cplx:
        return (dot - jnp.real(dot)) * sign, jnp.real(dot)
    return jnp.zeros_like(sign), dot


# ============================================================== slogdet / slogpf
@functools.lru_cache(maxsize=None)
def _slogdet_diff(n, dtype, prec, unroll_steps, block, fast):
    """custom_jvp slogdet on (B, n, n): d log|det A| = Re tr(A^-1 dA), d sign = 0
    (real) / i Im tr(A^-1 dA) sign (complex); zero gradient for a zero pivot."""
    fld = Field(dtype)

    @jax.custom_jvp
    def f(A):
        return _slogdet_core(A, fld, prec, unroll_steps, block, fast)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        if n <= DET_POLY_MAX:
            d, vjp = jax.vjp(_det_poly, A)
            # d det / dA = adj(A)^T: a polynomial, finite for any A
            (adjT,) = vjp(jnp.ones_like(d))
            sign, logabs = _sign_log(d)
            dot = jnp.sum(adjT * dA, axis=(-1, -2)) * _safe_recip(d)
        else:
            parts = _parts(A, n, fld, prec, unroll_steps, block, fast)
            sign, logabs, invT = parts[:3]
            dot = jnp.sum(invT * dA, axis=(-1, -2))
        return (sign, logabs), _log_tangents(sign, dot, fld)

    return f


@functools.lru_cache(maxsize=None)
def _slogpf_diff(n, dtype, prec, upd_warps, fast):
    """custom_jvp slogpf on skew-symmetric (B, n, n): d log|pf S| = tr(S^-1 dS) / 2
    (real part; the phase of the complex sign moves with the imaginary part)."""
    fld = Field(dtype)

    @jax.custom_jvp
    def f(S):
        return _slogpf_core(S, fld, prec, upd_warps, fast)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        if n <= PF_POLY_MAX:
            p, vjp = jax.vjp(_pf_poly, S)
            # d pf / d s_ij on the strict upper triangle
            (G,) = vjp(jnp.ones_like(p))
            # skew extension: d pf = sum(G * dS) / 2 for skew dS
            G = G - jnp.swapaxes(G, -1, -2)
            sign, logabs = _sign_log(p)
            dot = 0.5 * jnp.sum(G * dS, axis=(-1, -2)) * _safe_recip(p)
        else:
            parts = _pf_grad_parts(S, n, fld, prec, upd_warps, fast, False)
            sign, logabs, Sinv = parts[:3]
            dot = -0.5 * jnp.sum(Sinv * dS, axis=(-1, -2))  # S^-T = -S^-1
        return (sign, logabs), _log_tangents(sign, dot, fld)

    return f


# ===================================== det / pf: singular-compatible gradients
def _perm_sign(g0, dt):
    """sgn(P) = (-1)^(N - number of cycles) of the row permutation g0 (B, N). The
    cycles are counted by pointer jumping: in ceil(log2 N) rounds of two (B, N)
    gathers every element's label becomes the smallest index of its cycle (lab covers
    2^r consecutive orbit elements after r rounds, p = g^(2^r)), and each cycle has
    exactly one element labelled by itself. The former inversion count -- a reduce
    over the (B, N, N) comparison table -- was the gradient's largest XLA fusion and
    made XLA's GPU compile blow up with B and N (n = 256, B = 4096: the det gradient
    never finished compiling, > 20 min in that one reduce fusion; n = 128: 27 s)."""
    B, N = g0.shape
    idx = jnp.broadcast_to(jnp.arange(N, dtype=g0.dtype), (B, N))
    lab, p = idx, g0
    for _ in range(max(N - 1, 0).bit_length()):  # ceil(log2 N) doubling rounds
        lab = jnp.minimum(lab, jnp.take_along_axis(lab, p, axis=1))
        p = jnp.take_along_axis(p, p, axis=1)
    ncyc = isum(lab == idx, axis=1)
    return (1.0 - 2.0 * ((N - ncyc) % 2)).astype(dt)


def _diag_zero(LU):
    """Zero-pivot mask of a packed LU given as parts."""
    zero = jnp.ones(LU[0].shape[:2], bool)
    for Lc in LU:
        zero = zero & (jnp.diagonal(Lc, axis1=1, axis2=2) == 0)
    return zero


def _diag(LU, fld):
    """Diagonal of a packed LU given as parts: a (B, N) array of the field."""
    return fld.join(tuple(jnp.diagonal(c, axis1=1, axis2=2) for c in LU))


def _lu_det(d, g0, fld):
    """sgn(P) prod(d) for the pivots d (B, N) of a packed LU with P A = L U (P[c] =
    g0[c]): the determinant of the very factors the inverse was formed from, 0 if a
    pivot is exactly zero.

    The adjugate det(A) A^-1 is a polynomial in A and stays accurate for a numerically
    singular A only if the round-off-level pivot u_kk of such a matrix cancels exactly
    between det and the 1/u_kk inside A^-1. The forward's (sign, logabs) does not
    qualify: the packed LU recomputes its diagonal blocks (U_kk = L_kk^-1 A_raw), so
    its tiny pivot differs from the panel's by O(1) relative, and det_fwd * A^-T was
    off by that ratio (measured 2026-09-18: median 20 % / max 30x at n = 8..16 for
    rank n-1 float32 inputs; with this det 1e-6, like jnp.linalg.det's cofactor
    solve). The reductions are explicit pairwise trees (_fallback._pairwise), so the
    value does not depend on the program's reduction order (batched vs vmapped)."""
    dsafe = jnp.where(d == 0, 1, d)
    logsum = _pairwise(jnp.log(jnp.abs(dsafe)), jnp.add, 0)
    phase = _pairwise(jnp.sign(dsafe), jnp.multiply, 1)  # d / |d| for complex d
    det = _perm_sign(g0, fld.dtype) * phase * jnp.exp(logsum)
    return jnp.where(jnp.any(d == 0, axis=1), 0, det)


def _det_gradT(invT, LU, g0, zero, n, fld):
    """d det / dA = adj(A)^T from the LU parts. Regular: det * A^-T with det from the
    packed LU's own pivots (_lu_det). With zero pivots at K = {k_1 <= ... <= k_m}
    (U~ = U with those set to 1, d_K = product of the other pivots):

        adj(A) = sgn(P) d_K c_K (U~^-1 e_k1)(e_km^T U~^-1 L^-1 P)
        c_K = prod_i (U~^-1)[k_i, k_i+1]

    (Woodbury expansion of det(U(eps)) U(eps)^-1), exact for m <= 2; m >= 3 (rank n-1
    with three exact zero pivots) returns 0 like the rank <= n-2 case."""
    B, N = LU[0].shape[:2]
    dt = fld.dtype
    m = jnp.sum(zero, axis=1)
    d = _diag(LU, fld)
    det = _lu_det(d, g0, fld)
    regular = det[:, None, None] * invT

    def singular(_):
        LUj = fld.join(LU)
        idx = jnp.arange(N)
        k1 = jnp.minimum(jnp.min(jnp.where(zero, idx, N), axis=1), N - 1)
        k2 = jnp.maximum(jnp.max(jnp.where(zero, idx, -1), axis=1), 0)
        dsafe = jnp.where(zero, 1.0, d)
        logsum = jnp.sum(jnp.log(jnp.abs(dsafe)), axis=1)
        dK = jnp.prod(jnp.sign(dsafe), axis=1) * jnp.exp(logsum)
        Ut = jnp.triu(LUj) + zero.astype(dt)[:, :, None] * jnp.eye(N, dtype=dt)
        e1 = jax.nn.one_hot(k1, N, dtype=dt)
        e2 = jax.nn.one_hot(k2, N, dtype=dt)
        E = jnp.stack([e1, e2], axis=2)
        # U~^-1 e_k1, U~^-1 e_k2
        UE = lax.linalg.triangular_solve(Ut, E, left_side=True, lower=False)
        u1 = UE[:, :n, 0]
        # (U~^-1)[k1, k2]
        c = jnp.take_along_axis(UE[:, :, 1], k1[:, None], axis=1)[:, 0]
        s = _perm_sign(g0, dt) * dK * jnp.where(m >= 2, c, 1.0)
        s = jnp.where(m >= 3, 0.0, s)
        # (U~^-1 L^-1 P)^T e_k2
        col = jnp.take_along_axis(invT, k2[:, None, None], axis=2)[:, :, 0]
        G = s[:, None, None] * col[:, :, None] * u1[:, None, :]
        return jnp.where((m >= 1)[:, None, None], G, regular)

    return lax.cond(batch_any(m >= 1), singular, lambda _: regular, None)


def _det_small_gradT(A, n, fld, prec, unroll_steps, block):
    """det's gradient in the "small" mode (api._det_mode): (det, G) with det from the
    generic LU (the forward's own value) and G = adj(A)^T from the same LU, except
    for a batch with an exact zero pivot, where cuSOLVER's batched getrf is not a
    valid factorisation and the kernels' LU is used for G instead (the primal is the
    same exact 0 either way)."""
    b = _lu_block(n, _tune(fld.kind)) if block is None else block
    N = -(-n // b) * b
    sign, logabs, invT, LU, g0, zero = _lu_parts_generic(A, N, False)
    d = sign * jnp.exp(logabs)
    G_gen = _det_gradT(invT, fld.split(LU), g0, zero, n, fld)

    def kernels(A):
        parts = _lu_parts(fld.split(A), n, fld, prec, unroll_steps, block, False)
        return _det_gradT(*parts[2:], n, fld)

    exact = lambda A: _on_cuda(A, kernels, lambda A: G_gen)
    G = lax.cond(batch_any(zero), exact, lambda A: G_gen, A)
    return d, G


@functools.lru_cache(maxsize=None)
def _det_diff(n, dtype, prec, unroll_steps, block, fast):
    """custom_jvp det on (B, n, n): d det = tr(adj(A) dA), finite for singular A.
    fast: True (kernels), False (generic) or "small" (generic LU with the kernel
    fallback for exact zero pivots, see _det_small_gradT)."""
    fld = Field(dtype)
    small = fast == "small"

    @jax.custom_jvp
    def f(A):
        if n <= DET_POLY_MAX:
            # the polynomial itself, not sign * exp(log|.|): one rounding pair less,
            # and bit-identical to the value the jvp rule returns
            return _det_poly(A)
        sign, logabs = _slogdet_core(A, fld, prec, unroll_steps, block, fast is True)
        return sign * jnp.exp(logabs)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        if n <= DET_POLY_MAX:
            d, vjp = jax.vjp(_det_poly, A)
            (G,) = vjp(jnp.ones_like(d))
        elif small:
            d, G = _det_small_gradT(A, n, fld, prec, unroll_steps, block)
        else:
            parts = _parts(A, n, fld, prec, unroll_steps, block, fast, False)
            sign, logabs, invT, LU, g0, zero = parts
            d = sign * jnp.exp(logabs)  # the primal: the forward's own value
            G = _det_gradT(invT, LU, g0, zero, n, fld)
        return d, jnp.sum(G * dA, axis=(-1, -2))

    return f


@functools.lru_cache(maxsize=None)
def _pf_diff(n, dtype, prec, upd_warps, fast):
    """custom_jvp pf on skew-symmetric (B, n, n): d pf = sum(G dS) with G = pf S^-T / 2
    continued to singular S (the Pfaffian adjugate, see _pfinv)."""
    fld = Field(dtype)

    @jax.custom_jvp
    def f(S):
        if n <= PF_POLY_MAX:
            return _pf_poly(S)  # as in _det_diff: the jvp returns this same value
        sign, logabs = _slogpf_core(S, fld, prec, upd_warps, fast)
        return sign * jnp.exp(logabs)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        if n <= PF_POLY_MAX:
            p, vjp = jax.vjp(_pf_poly, S)
            (G,) = vjp(jnp.ones_like(p))
            G = 0.5 * (G - jnp.swapaxes(G, -1, -2))
        else:
            parts = _pf_grad_parts(S, n, fld, prec, upd_warps, fast, True)
            sign, logabs, adj = parts[:3]
            p = sign * jnp.exp(logabs)  # the primal: the forward's own value
            # d pf / dS = pf S^-T / 2 = -pf S^-1 / 2 (skew), continued to singular S
            G = -0.5 * adj
        return p, jnp.sum(G * dS, axis=(-1, -2))

    return f
