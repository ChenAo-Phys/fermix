"""Explicit polynomial det (n <= 4) and pf (even n <= 6), plus small guarded helpers."""
import jax.numpy as jnp


# ============================================================================= tiny n: explicit polynomials
DET_POLY_MAX = 4    # det by explicit expansion for n <= 4 (Laplace expansion in 2x2 minors at n = 4)
PF_POLY_MAX = 6     # pf by the perfect-matching sum for n <= 6 (15 terms at n = 6)


def _det_poly(A):
    """det of (..., n, n) for n <= 4 as an explicit polynomial in the entries."""
    n = A.shape[-1]
    if n == 0:
        return jnp.ones(A.shape[:-2], A.dtype)
    a = lambda i, j: A[..., i, j]
    m = lambda r, s, i, j: a(r, i) * a(s, j) - a(r, j) * a(s, i)          # 2x2 minor: rows (r, s), cols (i, j)
    if n == 1:
        return a(0, 0)
    if n == 2:
        return m(0, 1, 0, 1)
    if n == 3:
        return a(0, 0) * m(1, 2, 1, 2) - a(0, 1) * m(1, 2, 0, 2) + a(0, 2) * m(1, 2, 0, 1)
    if n == 4:      # Laplace expansion along rows (0, 1): signed products of complementary 2x2 minors
        return (m(0, 1, 0, 1) * m(2, 3, 2, 3) - m(0, 1, 0, 2) * m(2, 3, 1, 3) + m(0, 1, 0, 3) * m(2, 3, 1, 2)
                + m(0, 1, 1, 2) * m(2, 3, 0, 3) - m(0, 1, 1, 3) * m(2, 3, 0, 2) + m(0, 1, 2, 3) * m(2, 3, 0, 1))
    raise ValueError(f"_det_poly: n={n} > {DET_POLY_MAX}")


def _pf_poly(S):
    """pf of skew-symmetric (..., n, n) for even n <= 6 from the strict upper triangle (perfect-matching sum)."""
    n = S.shape[-1]
    if n == 0:
        return jnp.ones(S.shape[:-2], S.dtype)
    s = lambda i, j: S[..., i, j]
    pf4 = lambda a, b, c, d: s(a, b) * s(c, d) - s(a, c) * s(b, d) + s(a, d) * s(b, c)
    if n == 2:
        return s(0, 1)
    if n == 4:
        return pf4(0, 1, 2, 3)
    if n == 6:      # expansion along row 0: sum_j (-1)^(j+1) s_0j pf(S without rows/cols 0, j)
        return (s(0, 1) * pf4(2, 3, 4, 5) - s(0, 2) * pf4(1, 3, 4, 5) + s(0, 3) * pf4(1, 2, 4, 5)
                - s(0, 4) * pf4(1, 2, 3, 5) + s(0, 5) * pf4(1, 2, 3, 4))
    raise ValueError(f"_pf_poly: n={n} > {PF_POLY_MAX}")


def _sign_log(x):
    return jnp.sign(x), jnp.log(jnp.abs(x))


def _safe_recip(x):
    """1/x with 1/0 -> 0: the log-derivative of an exactly singular matrix is undefined, return a finite 0."""
    nz = x != 0
    return jnp.where(nz, 1.0 / jnp.where(nz, x, 1.0), 0.0)
