"""A^-T for the derivatives: packed-LU assembly from the forward kernel buffers,
block-recursive triangular inverse on a sub-block GEMM kernel, transpose + row-permuted
store. Matrix buffers are parts (see _field)."""

import functools
from typing import Any
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._field import where, dot, ld, st, mld, mst, _pcall
from ._common import _next_pow2, _unit_lower_inv, _upper_inv, _full, _vec, _tune
from ._lu import _lu_core


def _off_spec(k):
    """BlockSpec of a (k,) int32 vector of sub-block offsets (latency mode: the
    offsets are operands, so all GEMMs of one shape share a compiled kernel)."""
    return pl.BlockSpec((k,), lambda *idx: (0,))


# ========================================== inverse from our own LU factors
def _rows(M, idx):
    """M[b, idx[b, :], :] for a batch: (B, N, C), (B, R) -> (B, R, C) (row gather)."""
    return jax.vmap(lambda m, i: m[i])(M, idx)


# ------------------------------------------------- packed-LU assembly (4 warps)
# leaf size of the block-recursive inverse (32x32 leaf inverses come out of the
# assembly kernel)
LEAF = 32


def _assemble_kernel(p0_ref, p1_ref, idx_ref, out_ref, *, N, b, nb, tm, rolled):
    """One row tile [i0, i0+tm) of the packed LU in final pivoted row order. For column
    block k, rows >= r0 are gathered from buffer k%2 by the block's composed row map
    (the rows [r0, r0+b) then hold the pivot rows' panel content, fixed up by
    _diag_kernel), rows < r0 (U rows of earlier blocks) come from the buffer holding
    their U row block. ``rolled`` runs the blocks as a lax.fori_loop over pairs
    (even block from buffer 0, odd from buffer 1) instead of nb unrolled copies:
    the same copies, a kernel body independent of nb (latency mode, nb up to 128).
    """
    i0 = pl.program_id(1) * tm
    rows = i0 + lax.broadcasted_iota(jnp.int32, (tm,), 0)
    bufs = (p0_ref, p1_ref)

    def block(k, parity):
        r0 = k * b
        if not isinstance(r0, int):
            r0 = pl.multiple_of(r0, b)
        cols = pl.ds(r0, b)
        val = ld(bufs[parity], (idx_ref[k, pl.ds(i0, tm)], cols))
        if not isinstance(r0, int) or r0 > 0:
            use_other = (rows < r0) & ((((rows // b) + 1) & 1) != parity)
            oth = mld(bufs[1 - parity], (pl.ds(i0, tm), cols), mask=use_other[:, None])
            val = where(use_other[:, None], oth, val)
        st(out_ref, (pl.ds(i0, tm), cols), val)

    if not rolled:
        for k in range(nb):
            block(k, k & 1)
        return

    def pair(t, carry):
        block(2 * t, 0)
        block(2 * t + 1, 1)
        return carry

    lax.fori_loop(0, nb // 2, pair, 0)
    if nb % 2:
        block(nb - 1, (nb - 1) & 1)


def _diag_kernel(lu_ref, araw_ref, out_ref, linv_ref, uinv_ref, *, fld, b):
    """Diagonal block k (one program per matrix and block, r0 = k b): the block
    currently holds the pivot rows' panel content (correct L_kk strictly below the
    diagonal, stale above). Rebuild it in 16x16 pieces as L_kk + U_kk with U_kk =
    L_kk^-1 A_raw (block substitution + tensor-core dots) and write the 32x32 leaf
    inverses of L_kk / U_kk for the block-recursive inverse (zero pivots treated as 1
    there only).

    The 16x16 triangular substitutions run as fori_loops and the U blocks of one block
    row in a fori_loop over the column block (the leaf inverses then re-read the U
    blocks the loop stored): the same arithmetic as unrolled code in a much smaller
    kernel body (the packed-LU stage compiled in 127 s for complex64 at n = 256
    unrolled, 4 s this way; results are bit-identical)."""
    del out_ref  # aliased to lu_ref
    s = 16
    q = b // s
    r0 = pl.multiple_of(pl.program_id(1) * b, b)
    ib = lax.broadcasted_iota(jnp.int32, (s, s), 0)
    jb = lax.broadcasted_iota(jnp.int32, (s, s), 1)
    L, X = {}, {}
    for a in range(q):
        for c in range(a + 1):
            L[a, c] = ld(lu_ref, (pl.ds(r0 + a * s, s), pl.ds(r0 + c * s, s)))
    # X = L_kk^-1 by 16x16 block forward substitution
    for a in range(q):
        X[a, a] = _unit_lower_inv(L[a, a], s, fld, rolled=True)
        for c in range(a):
            acc = dot(L[a, c], X[c, c], "ieee")
            for j in range(c + 1, a):
                acc = acc + dot(L[a, j], X[j, c], "ieee")
            X[a, c] = -dot(X[a, a], acc, "ieee")

    def u_block(a, d):
        """U[a, d] = sum_{c <= a} X[a, c] A_raw[c, d] (d may be traced)."""
        acc = dot(X[a, 0], ld(araw_ref, (pl.ds(0, s), pl.ds(d * s, s))), "ieee")
        for c in range(1, a + 1):
            araw_blk = ld(araw_ref, (pl.ds(c * s, s), pl.ds(d * s, s)))
            acc = acc + dot(X[a, c], araw_blk, "ieee")
        return acc

    def diag_tile(a, d, Ublk):
        """L strictly below the diagonal, U on and above it, for the block d == a."""
        return where((d == a) & (ib > jb), L[a, a], Ublk)

    for a in range(q):

        def body(d, carry, a=a):
            col = pl.multiple_of(d * s, s)
            blk = diag_tile(a, d, u_block(a, d))
            st(lu_ref, (pl.ds(r0 + a * s, s), pl.ds(r0 + col, s)), blk)
            return carry

        lax.fori_loop(a, q, body, 0)
    zero16 = fld.zeros((s, s))
    # 32x32 leaves = 16-block pairs (2t, 2t+1); _upper_inv reads only the upper
    # triangle of its argument, so a re-read diagonal block (L below) is fine
    for t in range(q // 2):
        a0, a1 = 2 * t, 2 * t + 1
        st(linv_ref, (t, pl.ds(0, s), pl.ds(0, s)), X[a0, a0])
        st(linv_ref, (t, pl.ds(s, s), pl.ds(s, s)), X[a1, a1])
        st(linv_ref, (t, pl.ds(s, s), pl.ds(0, s)), X[a1, a0])
        st(linv_ref, (t, pl.ds(0, s), pl.ds(s, s)), zero16)

    def u_leaves(t, U00, U01, U11):
        Y00 = _upper_inv(U00, s, fld)
        Y11 = _upper_inv(U11, s, fld)
        Y01 = -dot(dot(Y00, U01, "ieee"), Y11, "ieee")
        st(uinv_ref, (t, pl.ds(0, s), pl.ds(0, s)), Y00)
        st(uinv_ref, (t, pl.ds(s, s), pl.ds(s, s)), Y11)
        st(uinv_ref, (t, pl.ds(0, s), pl.ds(s, s)), Y01)
        st(uinv_ref, (t, pl.ds(s, s), pl.ds(0, s)), zero16)

    # the leaf inverses read back the U blocks the loops stored (a store phase
    # re-read in the same kernel needs the barrier), one fori_loop over the pairs
    plgpu.debug_barrier()

    def leaf_body(t, carry):
        row0 = pl.multiple_of(r0 + t * (2 * s), 2 * s)
        U00 = ld(lu_ref, (pl.ds(row0, s), pl.ds(row0, s)))
        U01 = ld(lu_ref, (pl.ds(row0, s), pl.ds(row0 + s, s)))
        U11 = ld(lu_ref, (pl.ds(row0 + s, s), pl.ds(row0 + s, s)))
        u_leaves(t, U00, U01, U11)
        return carry

    lax.fori_loop(0, q // 2, leaf_body, 0)


def _packed_lu(fac, fld, lat=False):
    """Packed LU (unit L strictly below, U on/above the diagonal) in final pivoted row
    order from the forward kernels' buffers, g0 with (P A)[c] = A[g0[c]], and the 32x32
    leaf inverses (B, N/32, 32, 32) of L and U; the matrices as parts. lat: latency
    mode (the assembly kernel loops over the blocks instead of unrolling them).

    Block k's panel (columns [r0, r0+b)) stays in buffer k%2 with rows in that block's
    pre-pivot order; the pivot rows carry correct L entries left of their pivot column
    but stale values right of it, so U_kk is recomputed as L_kk^-1 (raw pivot rows) from
    the snapshot taken before the block. U_k's trailing row block was written by the
    urow kernel into buffer (k+1)%2. Row maps compose as g_k = S_k[g_{k+1}] with
    S_k = (identity | pivots | src_k)."""
    bufs, pivs, srcs, snaps, b, N = fac
    B = bufs[0][0].shape[0]
    nb = N // b
    ar = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32), (B, N))
    g = ar
    gs = [None] * nb
    for k in reversed(range(nb)):
        r0 = k * b
        Sk = jnp.concatenate([ar[:, :r0], pivs[k], srcs[k][:, r0 + b :]], axis=1)
        g = jnp.take_along_axis(Sk, g, axis=1)
        gs[k] = g
    # (B, nb, b, b) per component
    araw = tuple(
        jnp.stack([_rows(snaps[k][c], pivs[k] - k * b) for k in range(nb)], axis=1)
        for c in range(fld.k)
    )
    nleaf = N // LEAF
    rows_k = lambda k: jnp.concatenate([ar[:, : k * b], gs[k][:, k * b :]], axis=1)
    idx = jnp.stack([rows_k(k) for k in range(nb)], axis=1)  # (B, nb, N)
    tm = 64 if N % 64 == 0 else 32
    idx_spec = pl.BlockSpec((None, nb, N), lambda bi, i: (bi, 0, 0))
    ins = [(bufs[0], _full(N)), (bufs[1], _full(N)), (idx, idx_spec)]
    outs = [(fld.structs((B, N, N)), _full(N))]
    (LU,) = _pcall(
        functools.partial(_assemble_kernel, N=N, b=b, nb=nb, tm=tm, rolled=lat),
        ins,
        outs,
        (B, N // tm),
        num_warps=4,
    )
    q2 = b // (2 * 16)
    leaf_spec = pl.BlockSpec((None, q2, LEAF, LEAF), lambda bi, k: (bi, k, 0, 0))
    araw_spec = pl.BlockSpec((None, None, b, b), lambda bi, k: (bi, k, 0, 0))
    leaf = fld.structs((B, nleaf, LEAF, LEAF))
    ins = [(LU, _full(N)), (araw, araw_spec)]
    outs = [(fld.structs((B, N, N)), _full(N)), (leaf, leaf_spec), (leaf, leaf_spec)]
    t = _tune(fld.kind)
    LU, Lleaf, Uleaf = _pcall(
        functools.partial(_diag_kernel, fld=fld, b=b),
        ins,
        outs,
        (B, nb),
        aliases={0: 0},
        # the 128-block holds (b/16)^2 = 64 register tiles of 16x16 instead of 16:
        # spread them over 4x the warps
        num_warps=t.diag_warps * max(1, (b // 64) ** 2),
    )
    return LU, gs[0], Lleaf, Uleaf


def _split(N):
    h = _next_pow2(N) // 2
    return h if h < N else N // 2


# ---------------------------------------------------- sub-block GEMM (4 warps)
def _bgemm_kernel(
    *refs,
    fld,
    ia,
    ib,
    ra,
    ca,
    rb,
    cb,
    rc,
    cc,
    M,
    Nn,
    K,
    tm,
    tn,
    tk,
    alpha,
    beta,
    prec,
    ta,
    tb,
    skew,
):
    """out[rc:rc+M, cc:cc+Nn] = beta * out[...] + alpha * A[ra:ra+M, ca:ca+K]
    @ B[rb:rb+K, cb:cb+Nn] on 2-D sub-blocks of the batched operands refs[ia], refs[ib]
    (out = refs[-1], possibly aliased to one of them). Edge tiles are masked. With
    ta / tb the operand is the transpose of the stored block A[ra:ra+K, ca:ca+M] /
    B[rb:rb+Nn, cb:cb+K] (tiles are transposed in registers). skew: the product is
    known to be skew-symmetric (square, tm == tn, beta = 0): only the tiles on and
    below the diagonal are computed, the ones above are their negated transposes
    (edge tiles, when M is not a multiple of tm, are masked in both stores)."""
    a_ref, b_ref, out_ref = refs[ia], refs[ib], refs[-1]
    if ra is None:
        # latency mode: the six offsets come as an operand (multiples of the leaf)
        off_ref = refs[-2]
        ra, ca, rb, cb, rc, cc = (pl.multiple_of(off_ref[j], LEAF) for j in range(6))
    geom = dict(ra=ra, ca=ca, rb=rb, cb=cb, K=K, tm=tm, tn=tn, tk=tk, prec=prec)
    _bgemm_tile = _bgemm_tile_factory(fld, ta=ta, tb=tb, **geom)
    i0 = pl.program_id(1) * tm
    j0 = pl.program_id(2) * tn
    full = M % tm == 0 and Nn % tn == 0
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    rvalid = (i0 + ri) < M
    cvalid = (j0 + cj) < Nn
    if skew:
        assert tm == tn and beta == 0.0

        def lower():
            acc = _bgemm_tile(a_ref, b_ref, i0, j0, rvalid, cvalid, full)
            if alpha != 1.0:
                acc = alpha * acc
            out_idx = (pl.ds(rc + i0, tm), pl.ds(cc + j0, tn))
            mir_idx = (pl.ds(rc + j0, tn), pl.ds(cc + i0, tm))
            if full:
                st(out_ref, out_idx, acc)
                pl.when(j0 < i0)(lambda: st(out_ref, mir_idx, -acc.T))
            else:
                # edge tiles (M not a multiple of tm): the mirror of a masked
                # (rows, cols) tile is masked on (cols, rows)
                m2 = rvalid[:, None] & cvalid[None, :]
                mst(out_ref, out_idx, acc, mask=m2)
                pl.when(j0 < i0)(lambda: mst(out_ref, mir_idx, -acc.T, mask=m2.T))

        pl.when(j0 <= i0)(lower)
        return

    acc = _bgemm_tile(a_ref, b_ref, i0, j0, rvalid, cvalid, full)
    out_idx = (pl.ds(rc + i0, tm), pl.ds(cc + j0, tn))
    if full:
        if beta != 0.0:
            acc = beta * ld(out_ref, out_idx) + alpha * acc
        elif alpha != 1.0:
            acc = alpha * acc
        st(out_ref, out_idx, acc)
    else:
        m2 = rvalid[:, None] & cvalid[None, :]
        if beta != 0.0:
            C = mld(out_ref, out_idx, mask=m2)
            acc = beta * C + alpha * acc
        elif alpha != 1.0:
            acc = alpha * acc
        mst(out_ref, out_idx, acc, mask=m2)


def _bgemm_tile_factory(fld, ra, ca, rb, cb, K, tm, tn, tk, prec, ta, tb):
    """The K loop of one output tile of _bgemm_kernel (closure over the static
    geometry), returning acc = A_blk[i0:i0+tm, :] @ B_blk[:, j0:j0+tn]."""

    def tile(a_ref, b_ref, i0, j0, rvalid, cvalid, full):
        def body(t, acc):
            kk = pl.multiple_of(t * tk, tk)
            if ta:
                a_idx = (pl.ds(ra + kk, tk), pl.ds(ca + i0, tm))
                a_mask = rvalid[None, :]
            else:
                a_idx = (pl.ds(ra + i0, tm), pl.ds(ca + kk, tk))
                a_mask = rvalid[:, None]
            if tb:
                b_idx = (pl.ds(rb + j0, tn), pl.ds(cb + kk, tk))
                b_mask = cvalid[:, None]
            else:
                b_idx = (pl.ds(rb + kk, tk), pl.ds(cb + j0, tn))
                b_mask = cvalid[None, :]
            if full:
                Ab = ld(a_ref, a_idx)
                Bb = ld(b_ref, b_idx)
            else:
                Ab = mld(a_ref, a_idx, mask=a_mask)
                Bb = mld(b_ref, b_idx, mask=b_mask)
            if ta:
                Ab = Ab.T
            if tb:
                Bb = Bb.T
            return acc + dot(Ab, Bb, prec)

        return lax.fori_loop(0, K // tk, body, fld.zeros((tm, tn)))

    return tile


def _bgemm(
    arrays,
    ia,
    ib,
    ic,
    *,
    fld,
    ra,
    ca,
    rb,
    cb,
    rc,
    cc,
    M,
    Nn,
    K,
    prec,
    alpha=1.0,
    beta=0.0,
    ta: Any = False,  # bool; Any because callers unpack mixed-type **dims dicts
    tb: Any = False,
    skew: Any = False,
    dyn: Any = False,
):
    """Batched sub-block GEMM on (B, ., .) parts: out[rc:rc+M, cc:cc+Nn] = beta*out +
    alpha * A_blk @ B_blk with A = arrays[ia][ra:ra+M, ca:ca+K],
    B = arrays[ib][rb:rb+K, cb:cb+Nn]. ic = index of the array updated in place (aliased
    output; it may also be ia/ib as long as the read and written blocks do not overlap
    across programs), or None for a fresh (B, M, Nn) output. No slice copies: the
    kernels address the sub-blocks directly. ta / tb: use the transpose of the stored
    block arrays[ia][ra:ra+K, ca:ca+M] / arrays[ib][rb:rb+Nn, cb:cb+K] instead. skew: a
    skew-symmetric square product (fresh output), computed on and below the diagonal
    only and mirrored (half the tensor-core work). dyn: pass the offsets as an operand
    (latency mode), so that the GEMMs of one shape compile once."""
    Bn = arrays[0][0].shape[0]
    t = _tune(fld.kind)
    offsets = dict(ra=ra, ca=ca, rb=rb, cb=cb, rc=rc, cc=cc)
    extra = []
    if dyn:
        assert all(v % LEAF == 0 for v in offsets.values())
        extra = [(jnp.array(list(offsets.values()), jnp.int32), _off_spec(6))]
        offsets = dict.fromkeys(offsets, None)
    if skew:
        assert M == Nn and ic is None and beta == 0.0
        tm = tn = min(t.inv_tile, _next_pow2(M))
    else:
        # tall tiles (128x64 on A100) measured best for the big nodes
        tm = t.inv_tm_big if M % t.inv_tm_big == 0 else min(t.inv_tile, _next_pow2(M))
        tn = min(t.inv_tile, _next_pow2(Nn))
    tk = min(t.inv_tk, K)
    shapes = dict(M=M, Nn=Nn, K=K, tm=tm, tn=tn, tk=tk)
    scale = dict(alpha=alpha, beta=beta, prec=prec, ta=ta, tb=tb, skew=skew)
    kern = functools.partial(
        _bgemm_kernel, fld=fld, ia=ia, ib=ib, **shapes, **offsets, **scale
    )

    def spec(shp):
        block = (None,) + tuple(shp[1:])
        index = lambda *idx: (idx[0],) + (0,) * (len(shp) - 1)
        return pl.BlockSpec(block, index)

    oshape = (Bn, M, Nn) if ic is None else arrays[ic][0].shape
    ins = [(a, spec(a[0].shape)) for a in arrays] + extra
    outs = [(fld.structs(oshape), spec(oshape))]
    grid = (Bn, -(-M // tm), -(-Nn // tn))
    aliases = None if ic is None else {ic: 0}
    return _pcall(
        kern,
        ins,
        outs,
        grid,
        aliases=aliases,
        num_warps=t.inv_warps,
        num_stages=t.inv_stages,
    )[0]


def _inv_unit_lower(LU, Lleaf, prec, fld, leaf=LEAF, dyn=False):
    """L^-1 (B, N, N) of the unit lower-triangular L packed in LU: block-diagonal leaf
    inverses Lleaf (B, k, s, s), then per node X21 = -X22 L21 X11 written in place (two
    sub-block GEMMs). Only strictly-lower blocks of LU are read. dyn: GEMM offsets as
    operands (latency mode; the nodes of one level then share their kernels)."""
    B, N, _ = LU[0].shape
    k = N // leaf
    eye = jnp.eye(k, dtype=fld.real)
    X = tuple(jnp.einsum("bkij,kl->bkilj", Lc, eye).reshape(B, N, N) for Lc in Lleaf)

    def rec(X, i0, m):
        if m == leaf:
            return X
        h = _split(m)
        X = rec(X, i0, h)
        X = rec(X, i0 + h, m - h)
        at_t = dict(ra=i0 + h, ca=i0 + h, rb=i0 + h, cb=i0, rc=0, cc=0)
        dims_t = dict(M=m - h, K=m - h, Nn=h, prec=prec, fld=fld, dyn=dyn)
        T = _bgemm((X, LU), 0, 1, None, **at_t, **dims_t)
        at_x = dict(ra=0, ca=0, rb=i0, cb=i0, rc=i0 + h, cc=i0)
        dims = dict(M=m - h, K=h, Nn=h, alpha=-1.0, prec=prec, fld=fld, dyn=dyn)
        return _bgemm((T, X), 0, 1, 1, **at_x, **dims)

    return rec(X, 0, N)


def _solve_upper(LU, Y, Uleaf, prec, fld, leaf=LEAF, dyn=False):
    """U X = Y in place on the row blocks of Y (B, N, R) for the upper-triangular U
    packed in LU; Uleaf[:, i] = U_ii^-1. dyn as in _inv_unit_lower."""
    N = LU[0].shape[-1]
    R = Y[0].shape[-1]

    def rec(Y, i0, m):
        if m == leaf:
            Uinv = tuple(Uc[:, i0 // leaf] for Uc in Uleaf)
            at = dict(ra=0, ca=0, rb=i0, cb=0, rc=i0, cc=0)
            dims = dict(M=leaf, K=leaf, Nn=R, prec=prec, fld=fld, dyn=dyn)
            return _bgemm((Uinv, Y), 0, 1, 1, **at, **dims)
        h = _split(m)
        Y = rec(Y, i0 + h, m - h)
        at = dict(ra=i0, ca=i0 + h, rb=i0 + h, cb=0, rc=i0, cc=0)
        dims = dict(M=h, K=m - h, Nn=R, alpha=-1.0, beta=1.0, prec=prec, fld=fld)
        Y = _bgemm((LU, Y), 0, 1, 1, **at, **dims, dyn=dyn)
        return rec(Y, i0, h)

    return rec(Y, 0, N)


# ----------------------------- P^T Z^T with the singular guard (4 warps)
def _permT_kernel(z_ref, g0_ref, bad_ref, out_ref, *, n, tm):
    """out[g0[c], i] = Z[i, c] for g0[c] < n and i < n, i.e. out = (Z P)^T = P^T Z^T
    with P[c, g0[c]] = 1, restricted to the leading n x n block; all zeros for matrices
    flagged bad."""
    i0 = pl.program_id(1) * tm
    c0 = pl.program_id(2) * tm
    tile = ld(z_ref, (pl.ds(i0, tm), pl.ds(c0, tm)))  # rows i, cols c
    g = g0_ref[pl.ds(c0, tm)]
    ivalid = (i0 + lax.broadcasted_iota(jnp.int32, (tm,), 0)) < n
    val = where(bad_ref[0] != 0, 0.0, tile.T)
    mask = (g < n)[:, None] & ivalid[None, :]
    mst(out_ref, (g, pl.ds(i0, tm)), val, mask=mask)


def _permT(Z, g0, bad, n, fld):
    B, N, _ = Z[0].shape
    tm = 64 if N % 64 == 0 else 32
    kern = functools.partial(_permT_kernel, n=n, tm=tm)
    ins = [(Z, _full(N)), (g0, _vec(N)), (bad, _vec(1))]
    out_spec = pl.BlockSpec((None, n, n), lambda *idx: (idx[0], 0, 0))
    outs = [(fld.structs((B, n, n)), out_spec)]
    return _pcall(kern, ins, outs, (B, N // tm, N // tm), num_warps=4)[0]


def _lu_parts(A, n, fld, prec, unroll_steps, block, zero_singular=True):
    """One run of the LU kernels on a (B, n, n) batch given as parts: (sign, logabs,
    invT, LU, g0, zero).

    invT (a jnp array of the field) is A^-T restricted to the leading n x n block,
    formed with zero pivots replaced by 1 in U ("U-tilde"), so it is finite for
    singular A; overflowing inverses are zeroed and, with zero_singular=True, so are
    those of matrices with a zero pivot. LU (parts) is the packed factorisation
    (B, N, N) in final pivoted row order (P A = L U, (P A)[c] = A[g0[c]]), zero (B, N)
    the zero-pivot mask. A^-1 = U^-1 L^-1 P is formed as Z = U^-1 (L^-1) by block
    recursion on sub-block GEMM kernels (3xTF32 / ieee per ``prec``; 32x32 leaf
    inverses from the assembly kernel), then transposed and row-permuted by _permT.
    block=None takes the architecture table's choice, the same as the forward
    (_lu_block), so that (sign, logabs) matches it bit-for-bit."""
    sign, logabs, fac = _lu_core(A, n, fld, prec, unroll_steps, block, factors=True)
    lat = n > _tune(fld.kind).latency_min_n  # as in _lu_core
    LU, g0, Lleaf, Uleaf = _packed_lu(fac, fld, lat)
    zero = jnp.ones(LU[0].shape[:2], bool)
    for Lc in LU:
        zero = zero & (jnp.diagonal(Lc, axis1=1, axis2=2) == 0)
    Linv = _inv_unit_lower(LU, Lleaf, prec, fld, dyn=lat)
    Z = _solve_upper(LU, Linv, Uleaf, prec, fld, dyn=lat)  # U~^-1 L^-1
    bad = jnp.zeros(LU[0].shape[0], bool)
    for Zc in Z:
        bad = bad | ~jnp.all(jnp.isfinite(Zc), axis=(1, 2))
    if zero_singular:
        bad = bad | jnp.any(zero, axis=1)
    bad_i32 = bad.astype(jnp.int32)[:, None]
    invT = fld.join(_permT(Z, g0, bad_i32, n, fld))  # P^T Z^T, leading n x n
    return sign, logabs, invT, LU, g0, zero
