"""Gradients of pf / slogpf from the Parlett-Reid factors of the forward -- no second
factorisation. The pair steps are a block LDL^T with 2x2 skew pivots,

    P S P^T = L D L^T,   D = diag_s(d_s J),   J = [[0, 1], [-1, 0]],

L unit lower triangular with, for pair s (rows a_s = 2s, p_s = 2s + 1 in the final
order, U_s = the rows assigned later), L[U_s, a_s] = M_s[U_s, p_s] / d_s and
L[U_s, p_s] = tau_s = -M_s[U_s, a_s] / d_s -- the panel's w (= -column p) and tau
vectors -- so that pf(S) = sgn(P) prod d_s and S^-1 = P^T L^-T D^-1 L^-1 P.

The gradient of pf needs the Pfaffian adjugate pf(S) S^-1: a polynomial in S, finite and
well conditioned for singular S, which the plain product is not -- the smallest pivot
d_K of a numerically rank-deficient S is round-off noise, S^-1 ~ 1/d_K and pf ~ d_K, and
any inconsistency between the two (a second factorisation, such as an LU of S, whose
backward error is not skew) leaves an O(1) relative error (measured 2026-09-18: the
LU-based pf * S^-T was 100 % off on rank n-2 float32 inputs). Here 1/d_K never appears
in a factor: with L~ = L whose column a_K is scaled by d_K (i.e. M_K[U_K, p_K] instead
of M_K[U_K, p_K] / d_K; finite), D~ = D with d_K -> 1 and the block lower-triangular
Lambda~ = L~ D~ the factorisation reads P S P^T = Lambda~ D~'' Lambda~^T with a *finite*
D~'' whose K block is J^-T, and Lambda~ differs from the well-conditioned
Lambda^ = L~ D~ (K block J) by one entry: Lambda~ = Lambda^ - (1 - d_K) e_a e_p^T.
Sherman-Morrison on that entry ((Lambda^-1)[p, a] = 1) gives, with Y = L~^-1,
R = Y^T D~^-1 Y (= Lambda^-T D~ Lambda^-1), w = R e_{a_K} and v = Y^T e_{a_K}
(the a_K row of Y; the u^T D~ u term of the expansion vanishes because D~ is skew),

    S^-1 = -[R + c (w v^T - v w^T)],   c = (1 - d_K) / d_K,
    pf(S) S^-1 = -D_K [d_K R + (1 - d_K)(w v^T - v w^T)],   D_K = prod_{s != K} d_s,

exact for any d_K and finite at d_K = 0 (the rank-2 Pfaffian adjugate). A second exact
zero pivot does not necessarily mean a zero adjugate (a null row paired with a regular
row leaves rank n-2): pf(S) S^-1 is affine in each pair entry, so that pivot is lifted
to +1 and -1 and the two exact evaluations averaged (`_fallback._lifted_mean`, a rare
lax.cond branch). Everything is formed from well-conditioned pieces and d_K enters
linearly. Cost: L~^-1 by the block-recursive unit-lower inverse (N^3 / 3), one N^3
sub-block GEMM for R, elementwise work and the permutation scatter -- less than the
LU-based route it replaces (LU kernels + 1.67 N^3 inverse)."""

import functools
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._field import where, dot, ld, st, mst, _pcall
from ._common import _unit_lower_inv, _full, _vec, _tune
from ._fallback import (
    _lifted_mean,
    _pair_scale,
    _pf_adj_coeffs,
    _pf_value,
    _pivot_index,
)
from ._inverse import LEAF, _bgemm, _inv_unit_lower
from ._pf import _pf_core_factors


# ------------------------------------------------ 32x32 unit-lower leaf inverses
def _leaf32(L_ref, r0, fld, rolled):
    """Inverse of the unit lower-triangular 32x32 diagonal block at r0 of the packed
    L_ref as 16x16 pieces (X10 = -X11 L10 X00)."""
    s = LEAF // 2
    L00 = ld(L_ref, (pl.ds(r0, s), pl.ds(r0, s)))
    L10 = ld(L_ref, (pl.ds(r0 + s, s), pl.ds(r0, s)))
    L11 = ld(L_ref, (pl.ds(r0 + s, s), pl.ds(r0 + s, s)))
    X00 = _unit_lower_inv(L00, s, fld, rolled)
    X11 = _unit_lower_inv(L11, s, fld, rolled)
    X10 = -dot(X11, dot(L10, X00, "ieee"), "ieee")
    return X00, X10, X11


def _store_leaf(leaf_ref, t, X00, X10, X11, fld):
    s = LEAF // 2
    st(leaf_ref, (t, pl.ds(0, s), pl.ds(0, s)), X00)
    st(leaf_ref, (t, pl.ds(s, s), pl.ds(0, s)), X10)
    st(leaf_ref, (t, pl.ds(s, s), pl.ds(s, s)), X11)
    st(leaf_ref, (t, pl.ds(0, s), pl.ds(s, s)), fld.zeros((s, s)))


# ------------------------------------------------------------- L assembly
def _pf_assemble_kernel(*refs, fld, N, b, nb, rolled):
    """Row tile i (final rows [i b, (i+1) b)) of L~ and the 32x32 leaf inverse of its
    diagonal block, from the panel buffers. Block k's pair t (local index; final
    columns r0 + 2t for its row a, r0 + 2t + 1 for p) has tau_t in gbuf_k column
    (t // 4) 8 + t % 4 and w_t (= -column p) in column (t // 4) 8 + 4 + t % 4, both
    indexed by the block's compact row index idx[k N + f] of the final row f:

        L~[f, r0 + 2t]     = -w_t[f] / d~_t     (f > r0 + 2t + 1)
        L~[f, r0 + 2t + 1] = tau_t[f]           (f > r0 + 2t + 1)

    with 1/d~ (pivot K and exact zeros replaced by 1) given per pair in dinv."""
    gbufs = refs[:nb]
    idx_ref, dinv_ref = refs[nb], refs[nb + 1]
    L_ref, leaf_ref = refs[nb + 2 :]
    h2 = b // 2
    i = pl.program_id(1)
    i0 = pl.multiple_of(i * b, b)
    cb = lax.broadcasted_iota(jnp.int32, (b,), 0)
    rows_f = (i0 + cb)[:, None]
    half = cb // 2  # the pair of final column c of the block
    tcol = (half // 4) * 8 + (half & 3)
    odd = (cb & 1) == 1
    lcol = jnp.where(odd, tcol, tcol + 4)  # tau for the p column, w for the a column
    one = fld.rscalar(1.0)
    for k in range(nb):
        r0 = k * b
        idx_all = idx_ref[pl.ds(k * N + i0, b)]
        val = ld(gbufs[k], (idx_all[:, None], lcol[None, :]))
        dinv = ld(dinv_ref, (k * h2 + half,))  # 1/d~ of each column's pair
        fac = where(odd, one, -dinv)
        colf = (r0 + cb)[None, :]
        below = rows_f > jnp.where(odd, colf, colf + 1)  # strictly below the pair
        Lt = where(below, val * fac[None, :], 0.0)
        Lt = where(rows_f == colf, 1.0, Lt)
        st(L_ref, (pl.ds(i0, b), pl.ds(r0, b)), Lt)
    # the leaf inverse re-reads the diagonal block this program stored
    plgpu.debug_barrier()
    _store_leaf(leaf_ref, i, *_leaf32(L_ref, i0, fld, rolled), fld)


def _pf_maps(srcs, b, N):
    """Composed row maps from the per-block src (final-or-next index -> block index):
    g0 (B, N), final position -> original row (P[f, g0[f]] = 1), and the flat gather
    index (B, nb N) with idx[k N + f] = block-k compact index of final row f (f itself
    where block k has no such row)."""
    B = srcs[0].shape[0]
    nb = N // b
    ar = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32), (B, N))
    g = ar
    gs = [ar] * nb
    for k in reversed(range(nb)):
        r0 = k * b
        Sk = jnp.concatenate([ar[:, :r0], srcs[k][:, r0:]], axis=1)
        g = jnp.take_along_axis(Sk, g, axis=1)
        gs[k] = g
    rows_k = lambda k: jnp.concatenate([ar[:, : k * b], gs[k][:, k * b :]], axis=1)
    idx = jnp.concatenate([rows_k(k) for k in range(nb)], axis=1)
    return gs[0], idx


# ---------------- adjugate assembly + P^T (.) P (2-D scatter), one pass, 4 warps
def _adj_perm_kernel(r_ref, w_ref, v_ref, a_ref, g_ref, out_ref, *, n, tm):
    """out[g0[f], g0[c]] = -(a1 R + a2 (w v^T - v w^T))[f, c] on the leading n x n
    block: the adjugate formula of _fallback._pf_adj_blocks fused with the permutation
    scatter (a single read of R). a_ref holds (a1, a2) per matrix."""
    i0 = pl.program_id(1) * tm
    c0 = pl.program_id(2) * tm
    tile = ld(r_ref, (pl.ds(i0, tm), pl.ds(c0, tm)))
    wi = ld(w_ref, (pl.ds(i0, tm),))
    wc = ld(w_ref, (pl.ds(c0, tm),))
    vi = ld(v_ref, (pl.ds(i0, tm),))
    vc = ld(v_ref, (pl.ds(c0, tm),))
    a1 = ld(a_ref, (0,))
    a2 = ld(a_ref, (1,))
    rank2 = wi[:, None] * vc[None, :] - vi[:, None] * wc[None, :]
    val = -(a1 * tile + a2 * rank2)
    gi = g_ref[pl.ds(i0, tm)]
    gc = g_ref[pl.ds(c0, tm)]
    mask = (gi < n)[:, None] & (gc < n)[None, :]
    mst(out_ref, (gi[:, None], gc[None, :]), val, mask=mask)


def _adj_perm(R, w, v, a12, g0, n, fld):
    """The (B, n, n) array of the field with out[g0[f], g0[c]] = PT[f, c],
    PT = -(a1 R + a2 (w v^T - v w^T)) (the padding rows g0 >= n are dropped); R, w, v
    as parts of (B, N, N) / (B, N), a12 the (B, 2) coefficients as parts."""
    B, N, _ = R[0].shape
    tm = 64 if N % 64 == 0 else 32
    kern = functools.partial(_adj_perm_kernel, n=n, tm=tm)
    out_spec = pl.BlockSpec((None, n, n), lambda *idx: (idx[0], 0, 0))
    outs = [(fld.structs((B, n, n)), out_spec)]
    ins = [(R, _full(N)), (w, _vec(N)), (v, _vec(N)), (a12, _vec(2)), (g0, _vec(N))]
    return _pcall(kern, ins, outs, (B, N // tm, N // tm), num_warps=4)[0]


# ------------------------------------------------------------------ driver
def _pf_parts(S, n, fld, prec, upd_warps, adjugate):
    """One run of the pf kernels (with factors) on the skew (B, n, n) batch given as
    parts: (sign, logabs, G, pf) with G = pf(S) S^-1 (adjugate=True; finite for
    singular S, see the module docstring) or S^-1 (adjugate=False; zero for an exactly
    singular S or an overflowing inverse) as a (B, n, n) array of the field, and
    pf = sgn(P) prod d_s the Pfaffian of the factors (0 for singular S). (sign, logabs)
    are the forward's own, bit-identical to a plain call."""
    sign, logabs, fac = _pf_core_factors(S, n, fld, prec, upd_warps)
    gbufs, srcs, ds, pars, b, N = fac
    B = S[0].shape[0]
    nb, h = N // b, N // 2
    t = _tune(fld.kind)
    g0, idx = _pf_maps(srcs, b, N)
    d = fld.join(
        tuple(jnp.concatenate([dk[c] for dk in ds], axis=1) for c in range(fld.k))
    )
    K = _pivot_index(d, n)
    parity = jnp.sum(jnp.concatenate(pars, axis=1), axis=1) % 2
    sgnP = jnp.where(parity == 1, -1.0, 1.0).astype(fld.dtype)
    gspec = pl.BlockSpec((None, N, b), lambda bi, i: (bi, 0, 0))
    leaf_spec = pl.BlockSpec((None, nb, LEAF, LEAF), lambda bi, i: (bi, 0, 0, 0))
    kern = functools.partial(
        _pf_assemble_kernel, fld=fld, N=N, b=b, nb=nb, rolled=t.diag_rolled
    )
    at0 = dict(ra=0, ca=0, rb=0, cb=0, rc=0, cc=0)
    dims = dict(M=N, Nn=N, K=N, prec=prec, fld=fld)

    def evaluate(dd):
        """The formula for the pivots dd (d, or d with a second zero lifted)."""
        dt_ = _pair_scale(dd, K)  # d~: pivot K and exact zeros -> 1
        dinv = fld.split(1 / dt_)
        ins = [(g, gspec) for g in gbufs] + [(idx, _vec(nb * N)), (dinv, _vec(h))]
        outs = [
            (fld.structs((B, N, N)), _full(N)),
            (fld.structs((B, nb, LEAF, LEAF)), leaf_spec),
        ]
        L, Lleaf = _pcall(kern, ins, outs, (B, nb), num_warps=t.diag_warps)
        Y = _inv_unit_lower(L, Lleaf, prec, fld)  # L~^-1
        Yj = fld.join(Y)
        # D~^-1 Y: the rows of each pair (a, p) -> (Y[p], -Y[a]) / d~
        Yr = Yj.reshape(B, h, 2, N)
        DY = jnp.stack([Yr[:, :, 1], -Yr[:, :, 0]], axis=2) / dt_[:, :, None, None]
        DY = fld.split(DY.reshape(B, N, N))
        # R = Y^T D~^-1 Y is skew: lower tiles + mirror when the table says so
        R = _bgemm((Y, DY), 0, 1, None, ta=True, skew=t.pf_skew_r, **at0, **dims)
        # w = R e_{a_K}, v = Y^T e_{a_K} (a_K = 2K); the assembly of
        # -(a1 R + a2 (w v^T - v w^T)) is fused with the permutation scatter
        a1, a2, _ = _pf_adj_coeffs(dd, sgnP, K, adjugate)
        aK = (2 * K)[:, None]
        w = tuple(jnp.take_along_axis(Rc, aK[:, :, None], axis=2)[:, :, 0] for Rc in R)
        v = tuple(jnp.take_along_axis(Yc, aK[:, None, :], axis=1)[:, 0, :] for Yc in Y)
        a12 = fld.split(jnp.stack([a1, a2], axis=1))
        return fld.join(_adj_perm(R, w, v, a12, g0, n, fld))

    G = _lifted_mean(d, K, evaluate, adjugate)
    if not adjugate:
        dK = jnp.take_along_axis(d, K[:, None], axis=1)[:, 0]
        bad = (dK == 0) | ~jnp.all(jnp.isfinite(G), axis=(1, 2))
        G = jnp.where(bad[:, None, None], 0, G)
    return sign, logabs, G, _pf_value(d, sgnP)
