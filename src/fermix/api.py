"""Public API: slogdet, slogpf, det, pf. Batched, differentiable; Pallas kernels on CUDA
GPUs for float32 / float64 / complex64 / complex128, a generic XLA implementation (with
a warning) elsewhere."""

import functools
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from ._common import _tune
from ._diff import _det_diff, _pf_diff, _slogdet_diff, _slogpf_diff
from ._field import KINDS, Field

_DET_STATIC = ("prec", "unroll_steps", "block", "fast")
_PF_STATIC = ("prec", "skew_symmetrize", "upd_warps", "fast")


class FermixFallbackWarning(UserWarning):
    """The fast Pallas kernels do not apply (device); the generic XLA implementation
    is used."""


def _prep(a):
    a = jnp.asarray(a)
    if a.dtype not in KINDS:
        supported = "float32, float64, complex64 and complex128"
        raise TypeError(f"fermix supports {supported}, got {a.dtype}")
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected (..., n, n), got {a.shape}")
    return a


def _use_kernels(a):
    """True when the Pallas kernels apply (a CUDA GPU); otherwise warn and return
    False. For traced inputs (inside jit/grad) only the default device is visible; a
    committed CPU array reaching such a call still runs the generic path (the kernel
    branch is selected at lowering time), just without the warning -- use
    ``jax.default_device`` to make the intent explicit."""
    if isinstance(a, Tracer) or not isinstance(a, jax.Array):
        dev = jax.config.jax_default_device  # traced: only the default device is known
        platforms = {dev.platform if dev is not None else jax.default_backend()}
    else:
        platforms = {d.platform for d in a.devices()}
    if platforms != {"gpu"}:
        plat = "/".join(sorted(platforms))
        msg = (
            f"fermix: device platform {plat} (the kernels need a CUDA GPU) -- falling "
            "back to the generic XLA implementation (LU / Parlett-Reid in jax.numpy; "
            "same results and singular-safe gradients, but much slower)."
        )
        warnings.warn(msg, FermixFallbackWarning, stacklevel=3)
        return False
    return True


def _lu_kernels(a):
    """Kernels for slogdet: a CUDA GPU and n inside the architecture table's range
    (outside it -- n <= lu_generic_max_n, where the single-block kernels are
    launch-bound, or n > lu_generic_above_n, where the wide kinds' register panels
    spill -- cuSOLVER's LU through jax.numpy is faster and the generic path is used
    deliberately, without the fallback warning). det has its own dispatch, _det_mode."""
    fast = _use_kernels(a)  # warns off-GPU whatever n
    return fast and not _generic_by_size(a)


def _generic_by_size(a):
    """True where the table sends a CUDA call to the generic cuSOLVER-based path
    deliberately: n <= lu_generic_max_n (launch-bound single-block kernels) or
    n > lu_generic_above_n (large n, where the wide kinds' register panels spill)."""
    t = _tune(Field(a.dtype).kind)
    n = a.shape[-1]
    return n <= t.lu_generic_max_n or n > t.lu_generic_above_n


def _det_mode(a):
    """det's path: True (kernels), False (generic: no CUDA GPU, with the warning) or
    "small" (CUDA, n <= the table's det_generic_max_n or n > its lu_generic_above_n,
    despite the name also at large n: the forward and the regular
    gradient run cuSOLVER's batched LU as slogdet's small-n path does, and the kernels
    are only launched for a batch that contains an exact zero pivot, because cuSOLVER's
    batched getrf is not a valid factorisation there (P A != L U) while det's adjugate
    gradient needs one)."""
    fast = _use_kernels(a)  # warns off-GPU whatever n
    t = _tune(Field(a.dtype).kind)
    n = a.shape[-1]
    if fast and (n <= t.det_generic_max_n or n > t.lu_generic_above_n):
        return "small"
    return fast


def _check_block(block):
    """The outer LU block of slogdet / det: None (the architecture table) or one of
    the three block sizes the kernels are written for (4 / 8 / 16 inner panels)."""
    if block is not None and block not in (32, 64, 128):
        raise ValueError(f"block must be None, 32, 64 or 128, got {block!r}")
    return block


def _flat(a):
    batch = a.shape[:-2]
    n = a.shape[-1]
    return a.reshape((int(np.prod(batch)), n, n)), batch, n


def _real(a):
    return jnp.finfo(a.dtype).dtype


@functools.partial(jax.jit, static_argnames=_DET_STATIC)
def _slogdet(a, prec, unroll_steps, block, fast):
    A, batch, n = _flat(a)
    prec = Field(a.dtype).check_prec(prec)
    sign, logabs = _slogdet_diff(n, a.dtype, prec, unroll_steps, block, fast)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=_PF_STATIC)
def _slogpf(a, prec, skew_symmetrize, upd_warps, fast):
    A, batch, n = _flat(a)
    prec = Field(a.dtype).check_prec(prec)
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype), jnp.full(batch, -jnp.inf, _real(a))
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    sign, logabs = _slogpf_diff(n, a.dtype, prec, upd_warps, fast)(A)
    return sign.reshape(batch), logabs.reshape(batch)


@functools.partial(jax.jit, static_argnames=_DET_STATIC)
def _det(a, prec, unroll_steps, block, fast):
    A, batch, n = _flat(a)
    prec = Field(a.dtype).check_prec(prec)
    return _det_diff(n, a.dtype, prec, unroll_steps, block, fast)(A).reshape(batch)


@functools.partial(jax.jit, static_argnames=_PF_STATIC)
def _pf(a, prec, skew_symmetrize, upd_warps, fast):
    A, batch, n = _flat(a)
    prec = Field(a.dtype).check_prec(prec)
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype)
    if skew_symmetrize:
        A = (A - jnp.swapaxes(A, -1, -2)) * 0.5
    return _pf_diff(n, a.dtype, prec, upd_warps, fast)(A).reshape(batch)


def slogdet(a, *, prec=None, unroll_steps=None, block=None):
    """Sign and log|det| of a batch of matrices ``a`` of shape (..., n, n). Returns
    (sign, logabsdet): sign has the dtype of ``a`` (exactly +-1 / 0 for real dtypes, a
    unit complex number for complex ones), logabsdet the corresponding real dtype --
    the conventions of ``jnp.linalg.slogdet``.

    float32, float64, complex64 and complex128 on a CUDA GPU run the Pallas kernels
    (for n <= 32, where a single-block factorisation is launch-bound, and for the
    64-bit / complex dtypes at large n -- above 3072 for complex128, 4096-6144 for
    float64 / complex64 depending on the GPU -- cuSOLVER's LU through jax.numpy is
    used instead: the same generic path other devices fall back to, there with a
    :class:`FermixFallbackWarning`, with the same singular-safe gradients); other dtypes
    raise TypeError. prec: the block-GEMM algorithm, "tf32x3" (tensor cores,
    fp32-level accuracy; the default for float32 / complex64) or "ieee" (exact fp32
    FMA; the only choice, and the default, for float64 / complex128, whose updates
    run in IEEE fp64). block: outer LU block size 32 / 64 / 128 (default: the
    architecture table, 32 for small n, 64 above a switch and 128 above n = 2048,
    each when it does not pad much more). unroll_steps: unroll the panel's column steps (default: the
    architecture table). n <= 4 uses the explicit polynomial.

    Differentiable: d logabsdet = Re tr(A^-1 dA); d sign = 0 for real dtypes and
    i Im(tr(A^-1 dA)) sign for complex ones. Exactly singular matrices (a zero pivot)
    get a zero gradient instead of inf/NaN; near-singular ones get the exact (large)
    gradient.
    """
    a = _prep(a)
    return _slogdet(a, prec, unroll_steps, _check_block(block), _lu_kernels(a))


def slogpf(a, *, prec=None, skew_symmetrize=True, upd_warps=None):
    """Sign and log|pf| of a batch of skew-symmetric matrices (S^T = -S, also for complex
    dtypes; even n; odd n gives (0, -inf)). Dtypes, devices and ``prec`` as in
    :func:`slogdet`.

    The input is skew-symmetrised as (a - a^T)/2 unless ``skew_symmetrize=False`` (then
    it must be exactly skew-symmetric). n <= 6 uses the explicit polynomial. upd_warps:
    warps of the rank-2 update kernel (None: the architecture default).

    Differentiable: d logabspf = Re tr(S^-1 dS) / 2 with S the skew-symmetrised input
    (gradient w.r.t. ``a`` is S^-T / 2, skew-symmetric); d sign = 0 (real) or
    i Im(tr(S^-1 dS) / 2) sign (complex). S^-1 is formed from the forward's own
    Parlett-Reid factors (no second factorisation). Exactly singular matrices get a
    zero gradient instead of inf/NaN.
    """
    a = _prep(a)
    return _slogpf(a, prec, skew_symmetrize, upd_warps, _use_kernels(a))


def det(a, *, prec=None, unroll_steps=None, block=None):
    """Determinant of a batch of matrices (same dtypes, options and fallback as
    :func:`slogdet`). Overflows the 32-bit dtypes for large, badly scaled matrices --
    prefer slogdet then.

    Differentiable with d det = tr(adj(A) dA): the adjugate is formed from one LU
    (det from the same pivots as A^-1, so a round-off-level pivot cancels exactly and
    numerically singular A get the adjugate to working precision, like
    jnp.linalg.det), finite and correct for exactly singular A of rank n-1 (rank-1
    gradient) and zero for rank <= n-2 (exact for up to two zero pivots). That needs
    an exact factorisation of singular inputs, which cuSOLVER's batched LU does not
    provide: where slogdet takes the generic cuSOLVER path (small and very large n,
    see there) det uses it for the forward and the regular gradient too, but reruns
    the gradient through the kernels for a batch containing an exact zero pivot.
    """
    a = _prep(a)
    return _det(a, prec, unroll_steps, _check_block(block), _det_mode(a))


def pf(a, *, prec=None, skew_symmetrize=True, upd_warps=None):
    """Pfaffian of a batch of skew-symmetric matrices (same options and fallback as
    :func:`slogpf`; odd n gives 0).

    Differentiable with d pf = sum(G dS), G = pf S^-T / 2 (the Pfaffian adjugate) formed
    from the forward's own Parlett-Reid factors with the smallest pivot isolated, so it
    is accurate to working precision for numerically singular S and finite for exactly
    singular S: rank n-2 gives the rank-2 adjugate, rank <= n-4 gives 0. Like pf itself
    the adjugate overflows the 32-bit dtypes for large, badly scaled matrices -- use
    :func:`slogpf` (whose gradient is S^-T / 2) there.
    """
    a = _prep(a)
    return _pf(a, prec, skew_symmetrize, upd_warps, _use_kernels(a))


__all__ = ["slogdet", "slogpf", "det", "pf", "FermixFallbackWarning"]
