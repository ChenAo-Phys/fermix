"""Public API: slogdet, slogpf, det, pf. Batched, differentiable; Pallas kernels on CUDA
GPUs for float32, a generic XLA implementation (with a warning) elsewhere."""

import functools
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from ._diff import _det_diff, _pf_diff, _slogdet_diff, _slogpf_diff

_DET_STATIC = ("prec", "unroll_steps", "block", "fast")
_PF_STATIC = ("prec", "skew_symmetrize", "upd_warps", "fast")


class FermixFallbackWarning(UserWarning):
    """The fast Pallas kernels do not apply (dtype or device); the generic XLA
    implementation is used."""


def _prep(a):
    a = jnp.asarray(a)
    if a.dtype not in (jnp.float32, jnp.float64):
        supported = "float32 (kernels) and float64 (generic path)"
        raise TypeError(f"fermix supports {supported}, got {a.dtype}")
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected (..., n, n), got {a.shape}")
    return a


def _use_kernels(a):
    """True when the Pallas kernels apply (float32 on a CUDA GPU); otherwise warn and
    return False. For traced inputs (inside jit/grad) only the default device is
    visible; a committed CPU array reaching such a call still runs the generic path (the
    kernel branch is selected at lowering time), just without the warning -- use
    ``jax.default_device`` to make the intent explicit."""
    reasons = []
    if a.dtype != jnp.float32:
        reasons.append(f"dtype {a.dtype.name} (the kernels need float32)")
    if isinstance(a, Tracer) or not isinstance(a, jax.Array):
        dev = jax.config.jax_default_device  # traced: only the default device is known
        platforms = {dev.platform if dev is not None else jax.default_backend()}
    else:
        platforms = {d.platform for d in a.devices()}
    if platforms != {"gpu"}:
        plat = "/".join(sorted(platforms))
        reasons.append(f"device platform {plat} (the kernels need a CUDA GPU)")
    if reasons:
        msg = (
            f"fermix: {'; '.join(reasons)} -- falling back to the generic XLA "
            "implementation (LU / Parlett-Reid in jax.numpy; same results and "
            "singular-safe gradients, but much slower)."
        )
        warnings.warn(msg, FermixFallbackWarning, stacklevel=3)
        return False
    return True


def _flat(a):
    batch = a.shape[:-2]
    n = a.shape[-1]
    return a.reshape((int(np.prod(batch)), n, n)), batch, n


@functools.partial(jax.jit, static_argnames=_DET_STATIC)
def _slogdet(a, prec, unroll_steps, block, fast):
    A, batch, n = _flat(a)
    sign, logabs = _slogdet_diff(n, prec, unroll_steps, block, fast)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=_PF_STATIC)
def _slogpf(a, prec, skew_symmetrize, upd_warps, fast):
    A, batch, n = _flat(a)
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype), jnp.full(batch, -jnp.inf, a.dtype)
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    sign, logabs = _slogpf_diff(n, prec, upd_warps, fast)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=_DET_STATIC)
def _det(a, prec, unroll_steps, block, fast):
    A, batch, n = _flat(a)
    return _det_diff(n, prec, unroll_steps, block, fast)(A).reshape(batch)


@functools.partial(jax.jit, static_argnames=_PF_STATIC)
def _pf(a, prec, skew_symmetrize, upd_warps, fast):
    A, batch, n = _flat(a)
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype)
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    return _pf_diff(n, prec, upd_warps, fast)(A).reshape(batch)


def slogdet(a, *, prec="tf32x3", unroll_steps=True, block=None):
    """Sign and log|det| of a batch of matrices ``a`` of shape (..., n, n). Returns
    (sign, logabsdet).

    float32 on a CUDA GPU runs the Pallas kernels; other dtypes (float64) or devices
    fall back to a generic jax.numpy LU with a :class:`FermixFallbackWarning`. prec:
    "tf32x3" (default; tensor cores, fp32-level accuracy) or "ieee" (exact fp32 FMA) for
    the kernels' block GEMM updates. block: outer LU block size 32/64 (default 32 for
    n <= 256, else 64). n <= 4 uses the explicit polynomial.

    Differentiable: d logabsdet = tr(A^-1 dA), d sign = 0. Exactly singular matrices (a
    zero pivot) get a zero gradient instead of inf/NaN; near-singular ones get the exact
    (large) gradient.
    """
    a = _prep(a)
    return _slogdet(a, prec, unroll_steps, block, _use_kernels(a))


def slogpf(a, *, prec="tf32x3", skew_symmetrize=True, upd_warps=4):
    """Sign and log|pf| of a batch of skew-symmetric matrices (even n; odd n gives
    (0, -inf)).

    The input is skew-symmetrised as (a - a^T)/2 unless ``skew_symmetrize=False`` (then
    it must be exactly skew-symmetric). n <= 6 uses the explicit polynomial. Kernels for
    float32 on CUDA, generic Parlett-Reid (with a :class:`FermixFallbackWarning`)
    otherwise.

    Differentiable: d logabspf = tr(S^-1 dS) / 2 with S the skew-symmetrised input
    (gradient w.r.t. ``a`` is S^-T / 2, skew-symmetric); d sign = 0. Exactly singular
    matrices get a zero gradient instead of inf/NaN.
    """
    a = _prep(a)
    return _slogpf(a, prec, skew_symmetrize, upd_warps, _use_kernels(a))


def det(a, *, prec="tf32x3", unroll_steps=True, block=None):
    """Determinant of a batch of matrices (same options and fallback as
    :func:`slogdet`). Overflows float32 for large, badly scaled matrices -- prefer
    slogdet then.

    Differentiable with d det = tr(adj(A) dA): the adjugate is formed from the LU
    factors so the gradient is finite and correct for singular A of rank n-1 (rank-1
    gradient) and zero for rank <= n-2 (exact for up to two zero pivots).
    """
    a = _prep(a)
    return _det(a, prec, unroll_steps, block, _use_kernels(a))


def pf(a, *, prec="tf32x3", skew_symmetrize=True, upd_warps=4):
    """Pfaffian of a batch of skew-symmetric matrices (same options and fallback as
    :func:`slogpf`; odd n gives 0).

    Differentiable with d pf = sum(G dS), G = pf S^-T / 2 continued to singular S: for
    rank n-2 the gradient is the finite rank-2 matrix built from the null space and one
    minor pfaffian; rank <= n-4 gives 0.
    """
    a = _prep(a)
    return _pf(a, prec, skew_symmetrize, upd_warps, _use_kernels(a))


__all__ = ["slogdet", "slogpf", "det", "pf", "FermixFallbackWarning"]
