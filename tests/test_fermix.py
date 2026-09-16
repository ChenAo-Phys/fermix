"""Compact correctness tests for fermix. On a CUDA GPU they exercise the Pallas/Triton
kernels; on any other backend the very same tests cover the generic XLA fallback (which
fermix selects automatically, with a FermixFallbackWarning)."""

import math
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from fermix import FermixFallbackWarning, det, pf, slogdet, slogpf

rng = np.random.default_rng(0)


def pf_ref64(S):
    """Parlett-Reid with partial pivoting in float64."""
    A = np.array(S, dtype=np.float64)
    n = A.shape[0]
    sign, log = 1.0, 0.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.argmax(np.abs(A[k + 1 :, k]))
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            sign = -sign
        d = A[k, k + 1]
        if d == 0.0:
            return 0.0, -np.inf
        sign *= np.sign(d)
        log += np.log(abs(d))
        if k + 2 < n:
            tau = -A[k + 2 :, k] / d
            w = A[k + 2 :, k + 1]
            A[k + 2 :, k + 2 :] += np.outer(tau, w) - np.outer(w, tau)
    return sign, log


def pf64(S):
    s, l = pf_ref64(S)
    return s * math.exp(l) if np.isfinite(l) else 0.0


def skew(A):
    return A - np.swapaxes(A, -1, -2)


def relerr(x, ref):
    return float(np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-30))


def adjT64(A):
    """d det / dA = adj(A)^T by cofactors (small n)."""
    n = A.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M = np.delete(np.delete(A, i, 0), j, 1)
            G[i, j] = (-1) ** (i + j) * np.linalg.det(M)
    return G


def pf_gradT64(S):
    """d pf / dS with the skew extension: G = (D - D^T)/2, D_ij = (-1)^(i+j+1) pf(S
    without rows/cols i, j)."""
    n = S.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            keep = [k for k in range(n) if k not in (i, j)]
            D[i, j] = (-1) ** (i + j + 1) * pf64(S[np.ix_(keep, keep)])
    return 0.5 * (D - D.T)


# ------------------------------------------------------------ forward
@pytest.mark.parametrize("n", [3, 8, 40, 96])
def test_slogdet_matches_numpy(n):
    A = rng.standard_normal((5, n, n)).astype(np.float32)
    s, l = map(np.asarray, slogdet(jnp.asarray(A)))
    s64, l64 = np.linalg.slogdet(A.astype(np.float64))
    assert np.array_equal(s, s64)
    assert np.max(np.abs(l - l64)) < 2e-3 * max(1.0, n / 64)


@pytest.mark.parametrize("n", [4, 6, 34, 64])
def test_slogpf_matches_reference(n):
    S = skew(rng.standard_normal((5, n, n))).astype(np.float32)
    s, l = map(np.asarray, slogpf(jnp.asarray(S)))
    ref = [pf_ref64(S[i]) for i in range(5)]
    assert np.array_equal(s, [r[0] for r in ref])
    assert np.max(np.abs(l - [r[1] for r in ref])) < 2e-3


def test_det_pf_values_batch_dims_and_odd_n():
    A = rng.standard_normal((2, 3, 5, 5)).astype(np.float32)
    d = np.asarray(det(jnp.asarray(A)))
    assert d.shape == (2, 3)
    assert np.allclose(d, np.linalg.det(A.astype(np.float64)), rtol=1e-4, atol=1e-5)
    S = skew(rng.standard_normal((4, 12, 12))).astype(np.float32)
    p = np.asarray(pf(jnp.asarray(S)))
    assert np.allclose(p, [pf64(S[i]) for i in range(4)], rtol=1e-4, atol=1e-5)
    assert np.all(np.asarray(pf(jnp.zeros((3, 7, 7), jnp.float32))) == 0)
    s, l = slogpf(jnp.zeros((3, 7, 7), jnp.float32))
    assert np.all(np.asarray(s) == 0) and np.all(np.isneginf(np.asarray(l)))


def test_rejects_unsupported_dtype():
    with pytest.raises(TypeError):
        slogdet(jnp.eye(4, dtype=jnp.float16))


# ------------------------------------------------------------ gradients, regular
def test_slogdet_grad_and_jvp():
    n = 48
    A = (rng.standard_normal((4, n, n)) + 3 * np.eye(n)).astype(np.float32)
    g = np.asarray(jax.grad(lambda a: slogdet(a)[1].sum())(jnp.asarray(A)))
    ref = np.swapaxes(np.linalg.inv(A.astype(np.float64)), -1, -2)
    assert relerr(g, ref) < 1e-4
    dA = rng.standard_normal(A.shape).astype(np.float32)
    (_, _), (ds, dl) = jax.jvp(slogdet, (jnp.asarray(A),), (jnp.asarray(dA),))
    assert np.all(np.asarray(ds) == 0)
    ref_dl = np.einsum("bij,bji->b", np.linalg.inv(A.astype(np.float64)), dA)
    assert np.allclose(np.asarray(dl), ref_dl, rtol=1e-4, atol=1e-4)


def test_slogpf_grad():
    n = 32
    Anon = rng.standard_normal((4, n, n)).astype(np.float32)
    g = np.asarray(jax.grad(lambda a: slogpf(a)[1].sum())(jnp.asarray(Anon)))
    S = 0.5 * skew(Anon.astype(np.float64))
    assert relerr(g, 0.5 * np.swapaxes(np.linalg.inv(S), -1, -2)) < 1e-4


def test_det_pf_grad_regular():
    n = 12
    A = (rng.standard_normal((3, n, n)) + 2 * np.eye(n)).astype(np.float32)
    g = np.asarray(jax.grad(lambda a: det(a).sum())(jnp.asarray(A)))
    ref = np.stack(
        [np.linalg.det(a) * np.linalg.inv(a).T for a in A.astype(np.float64)]
    )
    assert relerr(g, ref) < 1e-4
    S = skew(rng.standard_normal((3, n, n))).astype(np.float32)
    g = np.asarray(
        jax.grad(lambda a: pf(a, skew_symmetrize=False).sum())(jnp.asarray(S))
    )
    ref = np.stack([0.5 * pf64(s) * np.linalg.inv(s).T for s in S.astype(np.float64)])
    assert relerr(g, ref) < 1e-4


# ------------------------------------------------------------ gradients, singular
def test_slogdet_slogpf_singular_grad_is_zero_and_finite():
    n = 40
    A = rng.standard_normal((3, n, n)).astype(np.float32)
    A[0, :, 5] = 0.0  # zero column -> exact zero pivot
    A[1] = 0.0
    g = np.asarray(jax.grad(lambda a: slogdet(a)[1].sum())(jnp.asarray(A)))
    assert np.all(np.isfinite(g)) and np.all(g[:2] == 0)
    assert relerr(g[2], np.linalg.inv(A[2].astype(np.float64)).T) < 1e-4
    S = skew(rng.standard_normal((3, n, n))).astype(np.float32)
    S[0, 7, :] = 0.0
    S[0, :, 7] = 0.0
    g = np.asarray(
        jax.grad(lambda a: slogpf(a, skew_symmetrize=False)[1].sum())(jnp.asarray(S))
    )
    assert np.all(np.isfinite(g)) and np.all(g[0] == 0)


@pytest.mark.parametrize("n", [3, 8, 40])
def test_det_grad_singular(n):
    """Zero column (zero pivot in the middle), zero row (zero pivot at the end), zero
    matrix, two zero columns (rank n-2), zero column 0 plus zero row 1 (rank n-1 with
    two zero pivots), and a regular member, vs float64 cofactors."""
    B = 6
    A = (rng.standard_normal((B, n, n)) + 2 * np.eye(n)).astype(np.float32)
    A[0, :, 1] = 0.0
    A[1, 2 % n] = 0.0
    A[2] = 0.0
    A[3, :, 0] = 0.0
    A[3, :, 1] = 0.0
    A[4, :, 0] = 0.0
    A[4, 1] = 0.0
    g = np.asarray(jax.grad(lambda a: det(a).sum())(jnp.asarray(A)))
    assert np.all(np.isfinite(g))
    ref = np.stack([adjT64(a) for a in A.astype(np.float64)])
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.max(np.abs(g - ref)) < 1e-4 * scale
    assert np.all(g[2] == 0) and np.all(g[3] == 0)
    for k in (0, 1, 4):  # rank n-1 members: nonzero, correct gradients
        assert np.abs(ref[k]).max() > 0 and relerr(g[k], ref[k]) < 1e-4


@pytest.mark.parametrize("n", [4, 8, 14])
def test_pf_grad_singular(n):
    """Zero row/column (rank n-2), zero matrix, plus a regular member, vs float64
    minor pfaffians."""
    B = 3
    S = skew(rng.standard_normal((B, n, n))).astype(np.float32)
    S[0, 3 % n, :] = 0.0
    S[0, :, 3 % n] = 0.0
    S[1] = 0.0
    g = np.asarray(
        jax.grad(lambda a: pf(a, skew_symmetrize=False).sum())(jnp.asarray(S))
    )
    assert np.all(np.isfinite(g))
    ref = np.stack([pf_gradT64(s) for s in S.astype(np.float64)])
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.max(np.abs(g - ref)) < 1e-4 * scale
    assert np.all(g[1] == 0) and np.abs(ref[0]).max() > 0


# ------------------------------------------------ generic fallback (CPU / float64)
def test_cpu_fallback_warns_and_matches():
    cpu = jax.devices("cpu")[0]
    n = 12
    A = (rng.standard_normal((4, n, n)) + 2 * np.eye(n)).astype(np.float32)
    A[0, :, 3] = 0.0  # exact zero pivot
    Ad = jax.device_put(jnp.asarray(A), cpu)
    with pytest.warns(FermixFallbackWarning):
        s, l = map(np.asarray, slogdet(Ad))
    s64, l64 = np.linalg.slogdet(A.astype(np.float64))
    assert np.array_equal(s, s64)
    assert np.allclose(l[1:], l64[1:], atol=1e-4)
    assert np.isneginf(l[0])
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = np.asarray(jax.grad(lambda a: det(a).sum())(Ad))
    ref = np.stack([adjT64(a) for a in A.astype(np.float64)])
    assert np.all(np.isfinite(g)) and np.max(np.abs(g - ref)) < 1e-4 * np.abs(ref).max()
    S = skew(rng.standard_normal((3, n, n))).astype(np.float32)
    S[0, 2, :] = 0.0
    S[0, :, 2] = 0.0
    Sd = jax.device_put(jnp.asarray(S), cpu)
    with pytest.warns(FermixFallbackWarning):
        s, l = map(np.asarray, slogpf(Sd, skew_symmetrize=False))
    ref = [pf_ref64(S[i]) for i in range(3)]
    assert np.array_equal(s, [r[0] for r in ref])
    assert np.allclose(l[1:], [r[1] for r in ref[1:]], atol=1e-4)
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = np.asarray(jax.grad(lambda a: pf(a, skew_symmetrize=False).sum())(Sd))
    ref = np.stack([pf_gradT64(x) for x in S.astype(np.float64)])
    assert np.all(np.isfinite(g)) and np.max(np.abs(g - ref)) < 1e-4 * np.abs(ref).max()
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = np.asarray(
            jax.grad(lambda a: slogpf(a, skew_symmetrize=False)[1].sum())(Sd)
        )
    assert np.all(g[0] == 0)
    assert relerr(g[1], 0.5 * np.linalg.inv(S[1].astype(np.float64)).T) < 1e-4
    # a committed CPU array reaching a traced call without default_device: no warning,
    # but the generic path runs. The two calls pad the LU differently (block size vs n),
    # so entries at the float32 round-off level need a loose atol.
    g2 = np.asarray(jax.grad(lambda a: slogpf(a, skew_symmetrize=False)[1].sum())(Sd))
    assert np.allclose(g2, g, rtol=1e-5, atol=1e-5)


def test_float64_fallback():
    jax.config.update("jax_enable_x64", True)
    try:
        n = 20
        A = rng.standard_normal((3, n, n)) + 2 * np.eye(n)
        with pytest.warns(FermixFallbackWarning):
            s, l = slogdet(jnp.asarray(A))
        assert l.dtype == jnp.float64
        s64, l64 = np.linalg.slogdet(A)
        assert np.array_equal(np.asarray(s), s64)
        assert np.allclose(np.asarray(l), l64, atol=1e-10)
        with pytest.warns(FermixFallbackWarning):
            g = np.asarray(jax.grad(lambda a: slogdet(a)[1].sum())(jnp.asarray(A)))
        assert relerr(g, np.swapaxes(np.linalg.inv(A), -1, -2)) < 1e-10
        S = skew(rng.standard_normal((2, n, n)))
        with pytest.warns(FermixFallbackWarning):
            p = np.asarray(pf(jnp.asarray(S)))
        assert np.allclose(p, [pf64(S[i]) for i in range(2)], rtol=1e-10)
    finally:
        jax.config.update("jax_enable_x64", False)
