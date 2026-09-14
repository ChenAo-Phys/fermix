"""Public API: slogdet, slogpf, det, pf (batched fp32, GPU, differentiable)."""
import functools
import jax
import jax.numpy as jnp
from ._common import f32, _prep
from ._diff import _det_diff, _pf_diff, _slogdet_diff, _slogpf_diff


@functools.partial(jax.jit, static_argnames=("prec", "unroll_steps", "block"))
def slogdet(a, *, prec="tf32x3", unroll_steps=True, block=None):
    """Sign and log|det| of a batch of fp32 matrices ``a`` of shape (..., n, n). Returns (sign, logabsdet).

    prec: "tf32x3" (default; tensor cores, fp32-level accuracy) or "ieee" (exact fp32 FMA) for the block GEMM
    updates; the pivoted panel factorisation is always exact fp32. block: outer LU block size 32/64 (default 32 for
    n <= 256, else 64). n <= 4 uses the explicit polynomial instead of the kernels.

    Differentiable: d logabsdet = tr(A^-1 dA), d sign = 0. Exactly singular matrices (a zero pivot) get a zero
    gradient instead of inf/NaN; near-singular ones get the exact (large) gradient.
    """
    A, batch, n = _prep(a)
    sign, logabs = _slogdet_diff(n, prec, unroll_steps, block)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=("prec", "skew_symmetrize", "upd_warps"))
def slogpf(a, *, prec="tf32x3", skew_symmetrize=True, upd_warps=4):
    """Sign and log|pf| of a batch of fp32 skew-symmetric matrices (even n; odd n gives (0, -inf)).

    The input is skew-symmetrised as (a - a^T)/2 unless ``skew_symmetrize=False`` (then it must be exactly
    skew-symmetric). n <= 6 uses the explicit polynomial instead of the kernels.

    Differentiable: d logabspf = tr(S^-1 dS) / 2 with S the skew-symmetrised input (gradient w.r.t. ``a`` is
    S^-T / 2, skew-symmetric); d sign = 0. Exactly singular matrices get a zero gradient instead of inf/NaN.
    """
    A, batch, n = _prep(a)
    if n % 2 == 1:
        return jnp.zeros(batch, f32), jnp.full(batch, -jnp.inf, f32)
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    sign, logabs = _slogpf_diff(n, prec, upd_warps)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=("prec", "unroll_steps", "block"))
def det(a, *, prec="tf32x3", unroll_steps=True, block=None):
    """Determinant of a batch of fp32 matrices (same options as :func:`slogdet`). Overflows fp32 for large, badly
    scaled matrices -- prefer slogdet then.

    Differentiable with d det = tr(adj(A) dA): the adjugate is formed from the LU factors so the gradient is finite and
    correct for singular A of rank n-1 (rank-1 gradient) and zero for rank <= n-2 (exact for up to two zero pivots).
    """
    A, batch, n = _prep(a)
    return _det_diff(n, prec, unroll_steps, block)(A).reshape(batch)


@functools.partial(jax.jit, static_argnames=("prec", "skew_symmetrize", "upd_warps"))
def pf(a, *, prec="tf32x3", skew_symmetrize=True, upd_warps=4):
    """Pfaffian of a batch of fp32 skew-symmetric matrices (same options as :func:`slogpf`; odd n gives 0).

    Differentiable with d pf = sum(G dS), G = pf S^-T / 2 continued to singular S: for rank n-2 the gradient is the
    finite rank-2 matrix built from the null space and one minor pfaffian; rank <= n-4 gives 0.
    """
    A, batch, n = _prep(a)
    if n % 2 == 1:
        return jnp.zeros(batch, f32)
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    return _pf_diff(n, prec, upd_warps)(A).reshape(batch)


__all__ = ["slogdet", "slogpf", "det", "pf"]
