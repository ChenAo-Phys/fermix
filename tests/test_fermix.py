"""Compact correctness tests for fermix over its four dtypes (float32, float64,
complex64, complex128). On a CUDA GPU they exercise the Pallas/Triton kernels; on any
other backend the very same tests cover the generic XLA fallback (which fermix selects
automatically, with a FermixFallbackWarning)."""

import math
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fermix import FermixFallbackWarning, det, pf, slogdet, slogpf

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
IDS = ["f32", "f64", "c64", "c128"]
dtypes = pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
complex_dtypes = pytest.mark.parametrize("dtype", DTYPES[2:], ids=IDS[2:])

rng = np.random.default_rng(0)


# ------------------------------------------------------------ helpers
def is_complex(dtype):
    return np.issubdtype(dtype, np.complexfloating)


def ref64(x):
    """The same matrix in the 64-bit dtype of its class (float64 / complex128)."""
    x = np.asarray(x)
    return x.astype(np.complex128 if is_complex(x.dtype) else np.float64)


def randn(shape, dtype, gen=None):
    """Standard normal entries (real and imaginary parts for complex dtypes)."""
    gen = rng if gen is None else gen
    z = gen.standard_normal(shape)
    if is_complex(dtype):
        z = z + 1j * gen.standard_normal(shape)
    return z.astype(dtype)


def shifted(shape, dtype, shift, gen=None):
    """Well-conditioned batch: standard normal plus shift * I."""
    return (randn(shape, dtype, gen) + shift * np.eye(shape[-1])).astype(dtype)


def skew(A):
    return A - np.swapaxes(A, -1, -2)


def relerr(x, ref):
    return float(np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-30))


def tol(dtype, t32, t64):
    """A tolerance by precision class (np.finfo(complex64).bits == 32)."""
    return t32 if np.finfo(dtype).bits == 32 else t64


def log_tol(dtype, n=1):
    """|delta log| of a forward: fp32 LU round-off grows with n; the 64-bit dtypes run
    IEEE fp64 block updates."""
    return tol(dtype, 2e-3, 1e-10) * max(1.0, n / 64)


def rel_tol(dtype):
    """Relative error of gradients / values on well-conditioned inputs."""
    return tol(dtype, 1e-4, 1e-10)


def assert_sign(s, ref, dtype, atol):
    """Real dtypes: exactly the reference (+-1 / 0). Complex: zero where the reference
    is zero, otherwise a unit number (to round-off) within atol of the reference -- its
    phase carries the same LU round-off as logabs, so atol is the log tolerance."""
    s, ref = np.asarray(s), np.asarray(ref)
    if not is_complex(dtype):
        assert np.array_equal(s, ref)
        return
    assert np.max(np.abs(s - ref)) < atol
    nz = ref != 0
    assert np.all(s[~nz] == 0)
    assert np.all(np.abs(np.abs(s[nz]) - 1.0) < tol(dtype, 1e-5, 1e-12))


def grad_real_sum(f, x):
    """jax.grad of sum Re f(x). For the real-valued slogdet / slogpf logabs this is the
    plain gradient; for the complex-valued det / pf (holomorphic in the entries) JAX's
    convention makes it the holomorphic derivative, adj(A)^T for det -- the same array
    jax.grad(..., holomorphic=True) returns, without a dtype-dependent flag."""
    return np.asarray(jax.grad(lambda a: jnp.real(f(a)).sum())(x))


def pf_ref64(S):
    """Parlett-Reid with partial pivoting in float64 / complex128: pivot on the
    magnitude, the pivot d contributes d / |d| to the sign (exactly +-1 when real)."""
    A = np.array(ref64(S))
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
        sign *= d / abs(d)
        log += math.log(abs(d))
        if k + 2 < n:
            tau = -A[k + 2 :, k] / d
            w = A[k + 2 :, k + 1]
            A[k + 2 :, k + 2 :] += np.outer(tau, w) - np.outer(w, tau)
    return sign, log


def pf64(S):
    s, l = pf_ref64(S)
    return s * math.exp(l) if np.isfinite(l) else 0.0


def adjT64(A):
    """d det / dA = adj(A)^T by cofactors (small n; no conjugation for complex A)."""
    n = A.shape[0]
    G = np.zeros((n, n), A.dtype)
    for i in range(n):
        for j in range(n):
            M = np.delete(np.delete(A, i, 0), j, 1)
            G[i, j] = (-1) ** (i + j) * np.linalg.det(M)
    return G


def pf_gradT64(S):
    """d pf / dS with the skew extension: G = (D - D^T)/2, D_ij = (-1)^(i+j+1) pf(S
    without rows/cols i, j)."""
    n = S.shape[0]
    D = np.zeros((n, n), S.dtype)
    for i in range(n):
        for j in range(i + 1, n):
            keep = [k for k in range(n) if k not in (i, j)]
            D[i, j] = (-1) ** (i + j + 1) * pf64(S[np.ix_(keep, keep)])
    return 0.5 * (D - D.T)


# ------------------------------------------------------------ forward
@pytest.mark.parametrize("n", [3, 8, 40, 96])
@dtypes
def test_slogdet_matches_numpy(n, dtype):
    A = randn((5, n, n), dtype)
    s, l = map(np.asarray, slogdet(jnp.asarray(A)))
    s64, l64 = np.linalg.slogdet(ref64(A))
    assert_sign(s, s64, dtype, log_tol(dtype, n))
    assert np.max(np.abs(l - l64)) < log_tol(dtype, n)


@pytest.mark.parametrize("n", [4, 6, 34, 64])
@dtypes
def test_slogpf_matches_reference(n, dtype):
    S = skew(randn((5, n, n), dtype))
    s, l = map(np.asarray, slogpf(jnp.asarray(S)))
    ref = [pf_ref64(S[i]) for i in range(5)]
    assert_sign(s, [r[0] for r in ref], dtype, log_tol(dtype))
    assert np.max(np.abs(l - [r[1] for r in ref])) < log_tol(dtype)


@dtypes
def test_n192_forward_and_grad(dtype):
    """n = 192 reaches the paths the small cases skip: 64-blocks and the multi-tile
    trailing-GEMM loop on Hopper, 2-warp pf panels. The gradient goes through jax.vjp
    so that its primal -- the jvp rule's own LU -- is also checked bit-identical to the
    forward at this n, the one where a forward-only block switch would differ (64 vs
    32 on Hopper; see test_grad_primal_is_bit_identical_to_forward)."""
    n = 192
    A = randn((3, n, n), dtype)
    s, l = map(np.asarray, slogdet(jnp.asarray(A)))
    s64, l64 = np.linalg.slogdet(ref64(A))
    assert_sign(s, s64, dtype, log_tol(dtype, n))
    assert np.max(np.abs(l - l64)) < log_tol(dtype, n)
    S = skew(randn((3, n, n), dtype))
    s, l = map(np.asarray, slogpf(jnp.asarray(S)))
    ref = [pf_ref64(S[i]) for i in range(3)]
    assert_sign(s, [r[0] for r in ref], dtype, log_tol(dtype, n))
    assert np.max(np.abs(l - [r[1] for r in ref])) < log_tol(dtype, n)
    W = (randn((3, n, n), dtype) / np.sqrt(n) + 2 * np.eye(n)).astype(dtype)
    x = jnp.asarray(W)
    primal, vjp_fn = jax.vjp(slogdet, x)
    for direct, under_vjp in zip(slogdet(x), primal):
        assert np.array_equal(np.asarray(direct), np.asarray(under_vjp))
    (g,) = vjp_fn((jnp.zeros_like(primal[0]), jnp.ones_like(primal[1])))
    ref_g = np.swapaxes(np.linalg.inv(ref64(W)), -1, -2)
    assert relerr(np.asarray(g), ref_g) < rel_tol(dtype)


@dtypes
def test_det_pf_values_batch_dims_and_odd_n(dtype):
    rt = rel_tol(dtype)
    A = randn((2, 3, 5, 5), dtype)
    d = np.asarray(det(jnp.asarray(A)))
    assert d.shape == (2, 3) and d.dtype == dtype
    assert np.allclose(d, np.linalg.det(ref64(A)), rtol=rt, atol=rt / 10)
    S = skew(randn((4, 12, 12), dtype))
    p = np.asarray(pf(jnp.asarray(S)))
    assert np.allclose(p, [pf64(S[i]) for i in range(4)], rtol=rt, atol=rt / 10)
    odd = jnp.zeros((3, 7, 7), dtype)
    assert np.all(np.asarray(pf(odd)) == 0)
    s, l = slogpf(odd)
    assert np.all(np.asarray(s) == 0) and np.all(np.isneginf(np.asarray(l)))


@dtypes
def test_output_dtypes(dtype):
    """sign / det / pf carry the input dtype and logabs the real dtype on every path:
    the polynomials (n = 4), the kernels or the generic path (n = 12), odd n."""
    real = np.finfo(dtype).dtype
    for n in (4, 12):
        A = jnp.asarray(randn((2, n, n), dtype))
        s, l = slogdet(A)
        assert s.dtype == dtype and l.dtype == real
        assert det(A).dtype == dtype
        s, l = slogpf(A)
        assert s.dtype == dtype and l.dtype == real
        assert pf(A).dtype == dtype
    odd = jnp.zeros((2, 7, 7), dtype)
    s, l = slogpf(odd)
    assert s.dtype == dtype and l.dtype == real
    assert pf(odd).dtype == dtype


@pytest.mark.parametrize("bad", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_rejects_unsupported_dtype(bad):
    with pytest.raises(TypeError):
        slogdet(jnp.eye(4, dtype=bad))


@dtypes
def test_prec_option(dtype):
    """prec="ieee" is valid for every dtype; "tf32x3" (tensor-core fp32 block updates)
    only for the 32-bit ones -- the 64-bit dtypes run theirs in IEEE fp64."""
    n = 8
    A = randn((2, n, n), dtype)
    S = skew(A)
    s, l = map(np.asarray, slogdet(jnp.asarray(A), prec="ieee"))
    s64, l64 = np.linalg.slogdet(ref64(A))
    assert_sign(s, s64, dtype, log_tol(dtype))
    assert np.max(np.abs(l - l64)) < log_tol(dtype)
    s, l = map(np.asarray, slogpf(jnp.asarray(S), prec="ieee"))
    ref = [pf_ref64(S[i]) for i in range(2)]
    assert_sign(s, [r[0] for r in ref], dtype, log_tol(dtype))
    assert np.max(np.abs(l - [r[1] for r in ref])) < log_tol(dtype)
    with pytest.raises(ValueError):
        det(jnp.asarray(A), prec="fast")
    if np.finfo(dtype).bits == 64:
        with pytest.raises(ValueError):
            slogdet(jnp.asarray(A), prec="tf32x3")
        with pytest.raises(ValueError):
            pf(jnp.asarray(S), prec="tf32x3")
    else:
        s, l = map(np.asarray, slogdet(jnp.asarray(A), prec="tf32x3"))
        assert_sign(s, s64, dtype, log_tol(dtype))
        assert np.max(np.abs(l - l64)) < log_tol(dtype)


# ------------------------------------------------------------ gradients, regular
@dtypes
def test_slogdet_grad_and_jvp(dtype):
    n = 48
    rt = rel_tol(dtype)
    A = shifted((4, n, n), dtype, 3)
    g = grad_real_sum(lambda a: slogdet(a)[1], jnp.asarray(A))
    inv = np.linalg.inv(ref64(A))
    assert relerr(g, np.swapaxes(inv, -1, -2)) < rt
    dA = randn(A.shape, dtype)
    (s, _), (ds, dl) = jax.jvp(slogdet, (jnp.asarray(A),), (jnp.asarray(dA),))
    tr = np.einsum("bij,bji->b", inv, ref64(dA))
    assert np.allclose(np.asarray(dl), tr.real, rtol=rt, atol=rt)
    if is_complex(dtype):
        # the phase moves with Im tr(A^-1 dA): jnp.linalg.slogdet's convention
        ref_ds = 1j * tr.imag * np.asarray(s)
        assert np.allclose(np.asarray(ds), ref_ds, rtol=rt, atol=rt)
    else:
        assert np.all(np.asarray(ds) == 0)


@complex_dtypes
def test_complex_conventions_match_jnp_linalg(dtype):
    """Complex derivatives follow jnp.linalg: under jax.jvp the sign tangent is
    i Im(tr(A^-1 dA)) sign and the logabs tangent Re(tr(A^-1 dA)); jax.grad of a real
    loss w.r.t. the complex input is A^-T without conjugation; det's gradient is the
    holomorphic adj(A)^T. slogpf: the same with tr(S^-1 dS) / 2, S the skew-symmetrised
    input."""
    n = 16
    rt = rel_tol(dtype)
    A = shifted((3, n, n), dtype, 3)
    dA = randn(A.shape, dtype)
    x, dx = jnp.asarray(A), jnp.asarray(dA)
    (s, l), (ds, dl) = jax.jvp(slogdet, (x,), (dx,))
    (s_j, l_j), (ds_j, dl_j) = jax.jvp(jnp.linalg.slogdet, (x,), (dx,))
    assert ds.dtype == dtype and dl.dtype == np.finfo(dtype).dtype
    for ours, theirs in ((s, s_j), (l, l_j), (ds, ds_j), (dl, dl_j)):
        assert np.allclose(np.asarray(ours), np.asarray(theirs), rtol=rt, atol=rt)
    g = grad_real_sum(lambda a: slogdet(a)[1], x)
    assert relerr(g, grad_real_sum(lambda a: jnp.linalg.slogdet(a)[1], x)) < rt
    inv = np.linalg.inv(ref64(A))
    assert relerr(g, np.swapaxes(inv, -1, -2)) < rt
    assert relerr(grad_real_sum(det, x), grad_real_sum(jnp.linalg.det, x)) < rt
    S = 0.5 * skew(ref64(A))
    Sinv = np.linalg.inv(S)
    (s, _), (ds, dl) = jax.jvp(slogpf, (x,), (dx,))
    tr = 0.5 * np.einsum("bij,bji->b", Sinv, ref64(dA))
    assert np.allclose(np.asarray(dl), tr.real, rtol=rt, atol=rt)
    assert np.allclose(np.asarray(ds), 1j * tr.imag * np.asarray(s), rtol=rt, atol=rt)
    g = grad_real_sum(lambda a: slogpf(a)[1], x)
    assert relerr(g, 0.5 * np.swapaxes(Sinv, -1, -2)) < rt


@dtypes
def test_slogpf_grad(dtype):
    n = 32
    Anon = randn((4, n, n), dtype)
    g = grad_real_sum(lambda a: slogpf(a)[1], jnp.asarray(Anon))
    S = 0.5 * skew(ref64(Anon))
    assert relerr(g, 0.5 * np.swapaxes(np.linalg.inv(S), -1, -2)) < rel_tol(dtype)


@pytest.mark.parametrize("n", [16, 40])
@dtypes
def test_grad_primal_is_bit_identical_to_forward(n, dtype):
    """slogdet's custom_jvp returns the (sign, logabs) of the gradient's own LU, so that
    LU has to use the same block size as the forward; the value seen under
    differentiation is then bit-identical to a plain call. n = 16 is the small-n range
    where slogdet takes the generic (cuSOLVER) path on a GPU, n = 40 the kernels;
    test_n192_forward_and_grad repeats the check at n = 192, where a forward-only block
    switch would differ (64 vs 32 on Hopper)."""
    gen = np.random.default_rng(11)  # not the shared rng: keeps the other tests' data
    A = shifted((3, n, n), dtype, 3, gen)
    dA = jnp.asarray(randn(A.shape, dtype, gen))
    for f, a in ((slogdet, A), (slogpf, skew(A))):
        x = jnp.asarray(a)
        primal, _ = jax.jvp(f, (x,), (dA,))
        for direct, under_jvp in zip(jax.tree.leaves(f(x)), jax.tree.leaves(primal)):
            assert np.array_equal(np.asarray(direct), np.asarray(under_jvp))


@pytest.mark.parametrize("n", [4, 6])
@dtypes
def test_tiny_n_grad_primal_is_bit_identical_to_forward(n, dtype):
    """The same for the polynomial path: det / pf return the polynomial itself, exactly
    what their jvp rule returns (sign * exp(log|poly|) used to differ by 1-2 ulp).
    n = 6 is past the det polynomial, so slogdet / det go through the kernels there."""
    gen = np.random.default_rng(12)
    A = randn((3, n, n), dtype, gen)
    dA = jnp.asarray(randn(A.shape, dtype, gen))
    for f, a in ((slogdet, A), (det, A), (slogpf, skew(A)), (pf, skew(A))):
        x = jnp.asarray(a)
        primal, _ = jax.jvp(f, (x,), (dA,))
        for direct, under_jvp in zip(jax.tree.leaves(f(x)), jax.tree.leaves(primal)):
            assert np.array_equal(np.asarray(direct), np.asarray(under_jvp))


@dtypes
def test_generic_branch_grad_primal_matches_forward(dtype):
    """A committed CPU array inside a traced call is lowered to the generic path while
    the call was traced as "fast": there the gradient's LU parts carry the kernels'
    padded size N > n. The padding is an exact identity embedding of the unpadded
    factorisation, so it must not move the value the jvp returns."""
    n = 12  # the gradient's parts are assembled at N = 32
    cpu = jax.devices("cpu")[0]
    gen = np.random.default_rng(13)
    A = shifted((3, n, n), dtype, 2, gen)
    dA = jax.device_put(jnp.asarray(randn(A.shape, dtype, gen)), cpu)
    for f, a in ((slogdet, A), (slogpf, skew(A))):
        x = jax.device_put(jnp.asarray(a), cpu)
        primal, _ = jax.jvp(f, (x,), (dA,))
        with pytest.warns(FermixFallbackWarning):
            direct = f(x)
        for d, pr in zip(jax.tree.leaves(direct), jax.tree.leaves(primal)):
            assert np.array_equal(np.asarray(d), np.asarray(pr))


@dtypes
def test_det_pf_grad_regular(dtype):
    n = 12
    A = shifted((3, n, n), dtype, 2)
    g = grad_real_sum(det, jnp.asarray(A))
    ref = np.stack([np.linalg.det(a) * np.linalg.inv(a).T for a in ref64(A)])
    assert relerr(g, ref) < rel_tol(dtype)
    S = skew(randn((3, n, n), dtype))
    g = grad_real_sum(lambda a: pf(a, skew_symmetrize=False), jnp.asarray(S))
    ref = np.stack([0.5 * pf64(s) * np.linalg.inv(s).T for s in ref64(S)])
    assert relerr(g, ref) < rel_tol(dtype)


# ------------------------------------------------------------ gradients, singular
@dtypes
def test_slogdet_slogpf_singular_grad_is_zero_and_finite(dtype):
    n = 40
    A = randn((3, n, n), dtype)
    A[0, :, 5] = 0.0  # zero column -> exact zero pivot
    A[1] = 0.0
    g = grad_real_sum(lambda a: slogdet(a)[1], jnp.asarray(A))
    assert np.all(np.isfinite(g)) and np.all(g[:2] == 0)
    assert relerr(g[2], np.linalg.inv(ref64(A[2])).T) < rel_tol(dtype)
    S = skew(randn((3, n, n), dtype))
    S[0, 7, :] = 0.0
    S[0, :, 7] = 0.0
    loss = lambda a: slogpf(a, skew_symmetrize=False)[1]
    g = grad_real_sum(loss, jnp.asarray(S))
    assert np.all(np.isfinite(g)) and np.all(g[0] == 0)


@pytest.mark.parametrize("n", [3, 8, 40])
@dtypes
def test_det_grad_singular(n, dtype):
    """Zero column (zero pivot in the middle), zero row (zero pivot at the end), zero
    matrix, two zero columns (rank n-2), zero column 0 plus zero row 1 (rank n-1 with
    two zero pivots), and a regular member, vs 64-bit cofactors."""
    B = 6
    A = shifted((B, n, n), dtype, 2)
    A[0, :, 1] = 0.0
    A[1, 2 % n] = 0.0
    A[2] = 0.0
    A[3, :, 0] = 0.0
    A[3, :, 1] = 0.0
    A[4, :, 0] = 0.0
    A[4, 1] = 0.0
    g = grad_real_sum(det, jnp.asarray(A))
    assert np.all(np.isfinite(g))
    ref = np.stack([adjT64(a) for a in ref64(A)])
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.max(np.abs(g - ref)) < rel_tol(dtype) * scale
    assert np.all(g[2] == 0) and np.all(g[3] == 0)
    for k in (0, 1, 4):  # rank n-1 members: nonzero, correct gradients
        assert np.abs(ref[k]).max() > 0 and relerr(g[k], ref[k]) < rel_tol(dtype)


def rank_deficient(shape, dtype, gen=None):
    """Numerically singular batch: the last row is a random combination of the others,
    formed in 64-bit and rounded to ``dtype`` (rank n-1 up to round-off, so the LU's
    last pivot is round-off noise but the adjugate is a well-conditioned polynomial)."""
    gen = rng if gen is None else gen
    A = randn(shape, np.complex128 if is_complex(dtype) else np.float64, gen)
    c = randn(shape[:-2] + (shape[-1] - 1,), A.dtype, gen)
    A[..., -1, :] = np.einsum("...i,...ij->...j", c, A[..., :-1, :])
    return A.astype(dtype)


@pytest.mark.parametrize("n", [8, 24])
@dtypes
def test_det_grad_near_singular(n, dtype):
    """Numerically rank n-1 inputs (a round-off pivot, no singular branch): d det/dA
    must still be the adjugate to working precision -- det and A^-1 have to come from
    the same pivots, or their O(1)-relative round-off in the tiny pivot does not cancel
    (the forward's det times the packed LU's inverse was 20 %..30x off)."""
    A = rank_deficient((4, n, n), dtype)
    g = grad_real_sum(det, jnp.asarray(A))
    ref = np.stack([adjT64(a) for a in ref64(A)])
    for k in range(len(A)):
        assert relerr(g[k], ref[k]) < tol(dtype, 1e-4, 1e-9)


@pytest.mark.parametrize("n", [4, 8, 14])
@dtypes
def test_pf_grad_singular(n, dtype):
    """Zero row/column (rank n-2), zero matrix, plus a regular member, vs 64-bit minor
    pfaffians."""
    B = 3
    S = skew(randn((B, n, n), dtype))
    S[0, 3 % n, :] = 0.0
    S[0, :, 3 % n] = 0.0
    S[1] = 0.0
    g = grad_real_sum(lambda a: pf(a, skew_symmetrize=False), jnp.asarray(S))
    assert np.all(np.isfinite(g))
    ref = np.stack([pf_gradT64(s) for s in ref64(S)])
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.max(np.abs(g - ref)) < rel_tol(dtype) * scale
    assert np.all(g[1] == 0) and np.abs(ref[0]).max() > 0


def skew_rank_deficient(shape, dtype, gen=None):
    """Numerically rank n-2 skew batch: X J X^T with X of n-2 columns, rounded to
    ``dtype`` (its two smallest Parlett-Reid pivots are round-off noise)."""
    gen = rng if gen is None else gen
    n = shape[-1]
    X = randn(
        shape[:-1] + (n - 2,), np.complex128 if is_complex(dtype) else np.float64, gen
    )
    J = np.kron(np.eye((n - 2) // 2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    S = (X @ J @ np.swapaxes(X, -1, -2)).astype(dtype)
    return S - np.swapaxes(S, -1, -2)


@pytest.mark.parametrize("n", [8, 24])
@dtypes
def test_pf_grad_near_singular(n, dtype):
    """Numerically rank n-2 inputs: d pf/dS must be the Pfaffian adjugate to working
    precision. The LU-based pf * S^-T was O(1) wrong here (the LU's backward error is
    not skew); the gradient is now formed from the forward's own Parlett-Reid factors
    with the smallest pivot isolated (see _pfinv)."""
    S = skew_rank_deficient((4, n, n), dtype)
    g = grad_real_sum(lambda a: pf(a, skew_symmetrize=False), jnp.asarray(S))
    ref = np.stack([pf_gradT64(s) for s in ref64(S)])
    for k in range(len(S)):
        assert relerr(g[k], ref[k]) < tol(dtype, 1e-4, 1e-9)


# ------------------------------------------------------------ generic fallback (CPU)
@dtypes
def test_cpu_fallback_warns_and_matches(dtype):
    cpu = jax.devices("cpu")[0]
    n = 12
    rt = rel_tol(dtype)
    A = shifted((4, n, n), dtype, 2)
    A[0, :, 3] = 0.0  # exact zero pivot
    Ad = jax.device_put(jnp.asarray(A), cpu)
    with pytest.warns(FermixFallbackWarning):
        s, l = map(np.asarray, slogdet(Ad))
    s64, l64 = np.linalg.slogdet(ref64(A))
    assert_sign(s, s64, dtype, log_tol(dtype))
    assert np.allclose(l[1:], l64[1:], atol=log_tol(dtype))
    assert np.isneginf(l[0])
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = grad_real_sum(det, Ad)
    ref = np.stack([adjT64(a) for a in ref64(A)])
    assert np.all(np.isfinite(g)) and np.max(np.abs(g - ref)) < rt * np.abs(ref).max()
    S = skew(randn((3, n, n), dtype))
    S[0, 2, :] = 0.0
    S[0, :, 2] = 0.0
    Sd = jax.device_put(jnp.asarray(S), cpu)
    with pytest.warns(FermixFallbackWarning):
        s, l = map(np.asarray, slogpf(Sd, skew_symmetrize=False))
    ref_pf = [pf_ref64(S[i]) for i in range(3)]
    assert_sign(s, [r[0] for r in ref_pf], dtype, log_tol(dtype))
    assert np.allclose(l[1:], [r[1] for r in ref_pf[1:]], atol=log_tol(dtype))
    pf_loss = lambda a: pf(a, skew_symmetrize=False)
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = grad_real_sum(pf_loss, Sd)
    ref = np.stack([pf_gradT64(x) for x in ref64(S)])
    assert np.all(np.isfinite(g)) and np.max(np.abs(g - ref)) < rt * np.abs(ref).max()
    log_loss = lambda a: slogpf(a, skew_symmetrize=False)[1]
    with pytest.warns(FermixFallbackWarning), jax.default_device(cpu):
        g = grad_real_sum(log_loss, Sd)
    assert np.all(g[0] == 0)
    assert relerr(g[1], 0.5 * np.linalg.inv(ref64(S[1])).T) < rt
    # a committed CPU array reaching a traced call without default_device: no warning,
    # but the generic path runs. It assembles the LU parts at the kernels' padded size
    # from the same unpadded factorisation, so the gradient is bit-identical.
    g2 = grad_real_sum(log_loss, Sd)
    assert np.array_equal(g2, g)
