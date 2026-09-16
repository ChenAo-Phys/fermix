"""A^-T for the derivatives: packed-LU assembly from the forward kernel buffers,
block-recursive triangular inverse on a sub-block GEMM kernel, transpose + row-permuted
store; plus a cuSOLVER reference path."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._common import f32, _next_pow2, _dot, _unit_lower_inv, _full, _vec
from ._lu import _lu_core

_tri_solve = lax.linalg.triangular_solve


# ================================================= inverse for the derivative
def _inverse_cusolver(A):
    """A^{-1} of (B, n, n) by cuSOLVER pivoted LU + two cuBLAS triangular solves with
    the permuted identity. Zero pivots are replaced by 1 before the solves and the
    affected matrices get an all-zero inverse, so an exactly singular input yields
    finite zeros instead of inf/NaN."""
    n = A.shape[-1]
    lu, _, perm = lax.linalg.lu(A)
    zero = jnp.diagonal(lu, axis1=-2, axis2=-1) == 0
    singular = jnp.any(zero, axis=-1)
    lu = lu + zero.astype(A.dtype)[..., None] * jnp.eye(n, dtype=A.dtype)
    x = jax.nn.one_hot(perm, n, dtype=A.dtype)  # (P I)[c, :] = e_perm[c]
    x = _tri_solve(lu, x, left_side=True, lower=True, unit_diagonal=True)
    x = _tri_solve(lu, x, left_side=True, lower=False)
    return _zero_bad(x, singular)


def _zero_bad(x, singular):
    """Zero the inverse of matrices flagged singular or whose inverse overflowed
    (subnormal pivots)."""
    bad = singular | ~jnp.all(jnp.isfinite(x), axis=(-1, -2))
    return jnp.where(bad[..., None, None], 0.0, x)


# ========================================== inverse from our own LU factors
def _rows(M, idx):
    """M[b, idx[b, :], :] for a batch: (B, N, C), (B, R) -> (B, R, C) (row gather)."""
    return jax.vmap(lambda m, i: m[i])(M, idx)


def _upper_inv(U, bsz):
    """Inverse of an upper-triangular bsz x bsz register tile by back substitution (zero
    diagonal entries -> 1)."""
    ib = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bsz, bsz), 1)
    ci = lax.broadcasted_iota(jnp.int32, (bsz,), 0)
    Us = jnp.where(ib <= jb, U, 0.0)
    d = jnp.sum(jnp.where(ib == jb, U, 0.0), axis=1)
    d = jnp.where(d == 0.0, 1.0, d)
    X = jnp.zeros((bsz, bsz), f32)
    for i in reversed(range(bsz)):
        urow = jnp.sum(jnp.where(ib == i, Us, 0.0), axis=0)
        acc = jnp.sum(urow[:, None] * X, axis=0)
        di = jnp.sum(jnp.where(ci == i, d, 0.0))
        row = (jnp.where(ci == i, 1.0, 0.0) - acc) / di
        X = jnp.where(ib == i, row[None, :], X)
    return X


# ------------------------------------------------- packed-LU assembly (4 warps)
# leaf size of the block-recursive inverse (32x32 leaf inverses come out of the
# assembly kernel)
LEAF = 32


def _assemble_kernel(p0_ref, p1_ref, idx_ref, out_ref, *, N, b, nb, tm):
    """One row tile [i0, i0+tm) of the packed LU in final pivoted row order. For column
    block k, rows >= r0 are gathered from buffer k%2 by the block's composed row map
    (the rows [r0, r0+b) then hold the pivot rows' panel content, fixed up by
    _diag_kernel), rows < r0 (U rows of earlier blocks) come from the buffer holding
    their U row block.
    """
    i0 = pl.program_id(1) * tm
    rows = i0 + lax.broadcasted_iota(jnp.int32, (tm,), 0)
    bufs = (p0_ref, p1_ref)
    for k in range(nb):
        r0 = k * b
        cols = pl.ds(r0, b)
        val = bufs[k & 1][idx_ref[k, pl.ds(i0, tm)], cols]
        if r0 > 0:
            use_other = (rows < r0) & ((((rows // b) + 1) & 1) != (k & 1))
            other_blk = bufs[(k + 1) & 1].at[pl.ds(i0, tm), cols]
            oth = plgpu.load(other_blk, mask=use_other[:, None], other=0.0)
            val = jnp.where(use_other[:, None], oth, val)
        out_ref[pl.ds(i0, tm), cols] = val


def _diag_kernel(lu_ref, araw_ref, out_ref, linv_ref, uinv_ref, *, b):
    """Diagonal block k (one program per matrix and block, r0 = k b): the block
    currently holds the pivot rows' panel content (correct L_kk strictly below the
    diagonal, stale above). Rebuild it in 16x16 pieces as L_kk + U_kk with U_kk =
    L_kk^-1 A_raw (block substitution + tensor-core dots) and write the 32x32 leaf
    inverses of L_kk / U_kk for the block-recursive inverse (zero pivots treated as 1
    there only)."""
    del out_ref  # aliased to lu_ref
    s = 16
    q = b // s
    r0 = pl.multiple_of(pl.program_id(1) * b, b)
    ib = lax.broadcasted_iota(jnp.int32, (s, s), 0)
    jb = lax.broadcasted_iota(jnp.int32, (s, s), 1)
    L, X, U = {}, {}, {}
    for a in range(q):
        for c in range(a + 1):
            L[a, c] = lu_ref[pl.ds(r0 + a * s, s), pl.ds(r0 + c * s, s)]
    # X = L_kk^-1 by 16x16 block forward substitution
    for a in range(q):
        X[a, a] = _unit_lower_inv(L[a, a], s)
        for c in range(a):
            acc = _dot(L[a, c], X[c, c], "ieee")
            for j in range(c + 1, a):
                acc = acc + _dot(L[a, j], X[j, c], "ieee")
            X[a, c] = -_dot(X[a, a], acc, "ieee")
    # U_kk = X A_raw (upper blocks only)
    for a in range(q):
        for d in range(a, q):
            acc = _dot(X[a, 0], araw_ref[pl.ds(0, s), pl.ds(d * s, s)], "ieee")
            for c in range(1, a + 1):
                araw_blk = araw_ref[pl.ds(c * s, s), pl.ds(d * s, s)]
                acc = acc + _dot(X[a, c], araw_blk, "ieee")
            U[a, d] = acc
    for a in range(q):
        for d in range(a, q):
            blk = jnp.where(ib > jb, L[a, a], U[a, a]) if d == a else U[a, d]
            lu_ref[pl.ds(r0 + a * s, s), pl.ds(r0 + d * s, s)] = blk
    zero16 = jnp.zeros((s, s), f32)
    # 32x32 leaves = 16-block pairs (2t, 2t+1)
    for t in range(q // 2):
        a0, a1 = 2 * t, 2 * t + 1
        linv_ref[t, pl.ds(0, s), pl.ds(0, s)] = X[a0, a0]
        linv_ref[t, pl.ds(s, s), pl.ds(s, s)] = X[a1, a1]
        linv_ref[t, pl.ds(s, s), pl.ds(0, s)] = X[a1, a0]
        linv_ref[t, pl.ds(0, s), pl.ds(s, s)] = zero16
        Y00 = _upper_inv(U[a0, a0], s)
        Y11 = _upper_inv(U[a1, a1], s)
        Y01 = -_dot(_dot(Y00, U[a0, a1], "ieee"), Y11, "ieee")
        uinv_ref[t, pl.ds(0, s), pl.ds(0, s)] = Y00
        uinv_ref[t, pl.ds(s, s), pl.ds(s, s)] = Y11
        uinv_ref[t, pl.ds(0, s), pl.ds(s, s)] = Y01
        uinv_ref[t, pl.ds(s, s), pl.ds(0, s)] = zero16


def _packed_lu(fac):
    """Packed LU (unit L strictly below, U on/above the diagonal) in final pivoted row
    order from the forward kernels' buffers, g0 with (P A)[c] = A[g0[c]], and the 32x32
    leaf inverses (B, N/32, 32, 32) of L and U.

    Block k's panel (columns [r0, r0+b)) stays in buffer k%2 with rows in that block's
    pre-pivot order; the pivot rows carry correct L entries left of their pivot column
    but stale values right of it, so U_kk is recomputed as L_kk^-1 (raw pivot rows) from
    the snapshot taken before the block. U_k's trailing row block was written by the
    urow kernel into buffer (k+1)%2. Row maps compose as g_k = S_k[g_{k+1}] with
    S_k = (identity | pivots | src_k)."""
    bufs, pivs, srcs, snaps, b, N = fac
    B = bufs[0].shape[0]
    nb = N // b
    ar = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32), (B, N))
    g = ar
    gs = [None] * nb
    for k in reversed(range(nb)):
        r0 = k * b
        Sk = jnp.concatenate([ar[:, :r0], pivs[k], srcs[k][:, r0 + b :]], axis=1)
        g = jnp.take_along_axis(Sk, g, axis=1)
        gs[k] = g
    rows_k = lambda k: jnp.concatenate([ar[:, : k * b], gs[k][:, k * b :]], axis=1)
    idx = jnp.stack([rows_k(k) for k in range(nb)], axis=1)  # (B, nb, N)
    # (B, nb, b, b)
    araw = jnp.stack([_rows(snaps[k], pivs[k] - k * b) for k in range(nb)], axis=1)
    tm = 64 if N % 64 == 0 else 32
    nleaf = N // LEAF
    idx_spec = pl.BlockSpec((None, nb, N), lambda bi, i: (bi, 0, 0))
    LU = pl.pallas_call(
        functools.partial(_assemble_kernel, N=N, b=b, nb=nb, tm=tm),
        grid=(B, N // tm),
        in_specs=[_full(N), _full(N), idx_spec],
        out_specs=_full(N),
        out_shape=jax.ShapeDtypeStruct((B, N, N), f32),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
    )(bufs[0], bufs[1], idx)
    q2 = b // (2 * 16)
    leaf_spec = pl.BlockSpec((None, q2, LEAF, LEAF), lambda bi, k: (bi, k, 0, 0))
    araw_spec = pl.BlockSpec((None, None, b, b), lambda bi, k: (bi, k, 0, 0))
    mat = jax.ShapeDtypeStruct((B, N, N), f32)
    leaf_shape = jax.ShapeDtypeStruct((B, nleaf, LEAF, LEAF), f32)
    LU, Lleaf, Uleaf = pl.pallas_call(
        functools.partial(_diag_kernel, b=b),
        grid=(B, nb),
        in_specs=[_full(N), araw_spec],
        out_specs=[_full(N), leaf_spec, leaf_spec],
        out_shape=[mat, leaf_shape, leaf_shape],
        input_output_aliases={0: 0},
        compiler_params=plgpu.CompilerParams(num_warps=1, num_stages=1),
    )(LU, araw)
    return LU, gs[0], Lleaf, Uleaf


def _split(N):
    h = _next_pow2(N) // 2
    return h if h < N else N // 2


# ---------------------------------------------------- sub-block GEMM (4 warps)
def _bgemm_kernel(
    *refs, ia, ib, ra, ca, rb, cb, rc, cc, M, Nn, K, tm, tn, tk, alpha, beta, prec
):
    """out[rc:rc+M, cc:cc+Nn] = beta * out[...] + alpha * A[ra:ra+M, ca:ca+K]
    @ B[rb:rb+K, cb:cb+Nn] on 2-D sub-blocks of the batched operands refs[ia], refs[ib]
    (out = refs[-1], possibly aliased to one of them). Edge tiles are masked."""
    a_ref, b_ref, out_ref = refs[ia], refs[ib], refs[-1]
    i0 = pl.program_id(1) * tm
    j0 = pl.program_id(2) * tn
    full = M % tm == 0 and Nn % tn == 0
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    rvalid = (i0 + ri) < M
    cvalid = (j0 + cj) < Nn

    def body(t, acc):
        kk = pl.multiple_of(t * tk, tk)
        a_blk = a_ref.at[pl.ds(ra + i0, tm), pl.ds(ca + kk, tk)]
        b_blk = b_ref.at[pl.ds(rb + kk, tk), pl.ds(cb + j0, tn)]
        if full:
            Ab = a_ref[pl.ds(ra + i0, tm), pl.ds(ca + kk, tk)]
            Bb = b_ref[pl.ds(rb + kk, tk), pl.ds(cb + j0, tn)]
        else:
            Ab = plgpu.load(a_blk, mask=rvalid[:, None], other=0.0)
            Bb = plgpu.load(b_blk, mask=cvalid[None, :], other=0.0)
        return acc + _dot(Ab, Bb, prec)

    acc = lax.fori_loop(0, K // tk, body, jnp.zeros((tm, tn), f32))
    out_blk = out_ref.at[pl.ds(rc + i0, tm), pl.ds(cc + j0, tn)]
    if full:
        if beta != 0.0:
            acc = beta * out_ref[pl.ds(rc + i0, tm), pl.ds(cc + j0, tn)] + alpha * acc
        elif alpha != 1.0:
            acc = alpha * acc
        out_ref[pl.ds(rc + i0, tm), pl.ds(cc + j0, tn)] = acc
    else:
        m2 = rvalid[:, None] & cvalid[None, :]
        if beta != 0.0:
            C = plgpu.load(out_blk, mask=m2, other=0.0)
            acc = beta * C + alpha * acc
        elif alpha != 1.0:
            acc = alpha * acc
        plgpu.store(out_blk, acc, mask=m2)


def _bgemm(
    arrays, ia, ib, ic, *, ra, ca, rb, cb, rc, cc, M, Nn, K, prec, alpha=1.0, beta=0.0
):
    """Batched sub-block GEMM on (B, ., .) arrays: out[rc:rc+M, cc:cc+Nn] = beta*out +
    alpha * A_blk @ B_blk with A = arrays[ia][ra:ra+M, ca:ca+K],
    B = arrays[ib][rb:rb+K, cb:cb+Nn]. ic = index of the array updated in place (aliased
    output; it may also be ia/ib as long as the read and written blocks do not overlap
    across programs), or None for a fresh (B, M, Nn) output. No slice copies: the
    kernels address the sub-blocks directly."""
    Bn = arrays[0].shape[0]
    # 128x64 tiles measured best for the big nodes
    tm = 128 if M % 128 == 0 else min(64, _next_pow2(M))
    tn = min(64, _next_pow2(Nn))
    tk = 32
    shapes = dict(M=M, Nn=Nn, K=K, tm=tm, tn=tn, tk=tk)
    offsets = dict(ra=ra, ca=ca, rb=rb, cb=cb, rc=rc, cc=cc)
    scale = dict(alpha=alpha, beta=beta, prec=prec)
    kern = functools.partial(_bgemm_kernel, ia=ia, ib=ib, **shapes, **offsets, **scale)

    def spec(shp):
        block = (None,) + tuple(shp[1:])
        index = lambda *idx: (idx[0],) + (0,) * (len(shp) - 1)
        return pl.BlockSpec(block, index)

    oshape = (Bn, M, Nn) if ic is None else arrays[ic].shape
    return pl.pallas_call(
        kern,
        grid=(Bn, -(-M // tm), -(-Nn // tn)),
        in_specs=[spec(a.shape) for a in arrays],
        out_specs=spec(oshape),
        out_shape=jax.ShapeDtypeStruct(oshape, f32),
        input_output_aliases={} if ic is None else {ic: 0},
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=GEMM_STAGES),
    )(*arrays)


def _inv_unit_lower(LU, Lleaf, prec, leaf=LEAF):
    """L^-1 (B, N, N) of the unit lower-triangular L packed in LU: block-diagonal leaf
    inverses Lleaf (B, k, s, s), then per node X21 = -X22 L21 X11 written in place (two
    sub-block GEMMs). Only strictly-lower blocks of LU are read."""
    B, N, _ = LU.shape
    k = N // leaf
    X = jnp.einsum("bkij,kl->bkilj", Lleaf, jnp.eye(k, dtype=f32)).reshape(B, N, N)

    def rec(X, i0, m):
        if m == leaf:
            return X
        h = _split(m)
        X = rec(X, i0, h)
        X = rec(X, i0 + h, m - h)
        at_t = dict(ra=i0 + h, ca=i0 + h, rb=i0 + h, cb=i0, rc=0, cc=0)
        T = _bgemm((X, LU), 0, 1, None, M=m - h, K=m - h, Nn=h, prec=prec, **at_t)
        at_x = dict(ra=0, ca=0, rb=i0, cb=i0, rc=i0 + h, cc=i0)
        dims = dict(M=m - h, K=h, Nn=h, alpha=-1.0, prec=prec)
        return _bgemm((T, X), 0, 1, 1, **at_x, **dims)

    return rec(X, 0, N)


def _solve_upper(LU, Y, Uleaf, prec, leaf=LEAF):
    """U X = Y in place on the row blocks of Y (B, N, R) for the upper-triangular U
    packed in LU; Uleaf[:, i] = U_ii^-1."""
    N = LU.shape[-1]
    R = Y.shape[-1]

    def rec(Y, i0, m):
        if m == leaf:
            Uinv = Uleaf[:, i0 // leaf]
            at = dict(ra=0, ca=0, rb=i0, cb=0, rc=i0, cc=0)
            return _bgemm((Uinv, Y), 0, 1, 1, M=leaf, K=leaf, Nn=R, prec=prec, **at)
        h = _split(m)
        Y = rec(Y, i0 + h, m - h)
        at = dict(ra=i0, ca=i0 + h, rb=i0 + h, cb=0, rc=i0, cc=0)
        dims = dict(M=h, K=m - h, Nn=R, alpha=-1.0, beta=1.0, prec=prec)
        Y = _bgemm((LU, Y), 0, 1, 1, **at, **dims)
        return rec(Y, i0, h)

    return rec(Y, 0, N)


GEMM_STAGES = 3  # software-pipelining depth of the sub-block GEMM K loop


# ----------------------------- P^T Z^T with the singular guard (4 warps)
def _permT_kernel(z_ref, g0_ref, bad_ref, out_ref, *, n, tm):
    """out[g0[c], i] = Z[i, c] for g0[c] < n and i < n, i.e. out = (Z P)^T = P^T Z^T
    with P[c, g0[c]] = 1, restricted to the leading n x n block; all zeros for matrices
    flagged bad."""
    i0 = pl.program_id(1) * tm
    c0 = pl.program_id(2) * tm
    tile = z_ref[pl.ds(i0, tm), pl.ds(c0, tm)]  # rows i, cols c
    g = g0_ref[pl.ds(c0, tm)]
    ivalid = (i0 + lax.broadcasted_iota(jnp.int32, (tm,), 0)) < n
    val = jnp.where(bad_ref[0] != 0, 0.0, tile.T)
    mask = (g < n)[:, None] & ivalid[None, :]
    plgpu.store(out_ref.at[g, pl.ds(i0, tm)], val, mask=mask)


def _permT(Z, g0, bad, n):
    B, N, _ = Z.shape
    tm = 64 if N % 64 == 0 else 32
    kern = functools.partial(_permT_kernel, n=n, tm=tm)
    return pl.pallas_call(
        kern,
        grid=(B, N // tm, N // tm),
        in_specs=[_full(N), _vec(N), _vec(1)],
        out_specs=pl.BlockSpec((None, n, n), lambda *idx: (idx[0], 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, n, n), f32),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
    )(Z, g0, bad)


def _lu_parts(A, n, prec, unroll_steps, block, zero_singular=True):
    """One run of the LU kernels on a (B, n, n) batch: (sign, logabs, invT, LU, g0,
    zero).

    invT is A^-T restricted to the leading n x n block, formed with zero pivots replaced
    by 1 in U ("U-tilde"), so it is finite for singular A; overflowing inverses are
    zeroed and, with zero_singular=True, so are those of matrices with a zero pivot. LU
    is the packed factorisation (B, N, N) in final pivoted row order (P A = L U,
    (P A)[c] = A[g0[c]]), zero (B, N) the zero-pivot mask. A^-1 = U^-1 L^-1 P is formed
    as Z = U^-1 (L^-1) by block recursion on sub-block GEMM kernels (3xTF32 / ieee per
    ``prec``; 32x32 leaf inverses from the assembly kernel), then transposed and
    row-permuted by _permT."""
    sign, logabs, fac = _lu_core(A, n, prec, unroll_steps, block, factors=True)
    LU, g0, Lleaf, Uleaf = _packed_lu(fac)
    zero = jnp.diagonal(LU, axis1=1, axis2=2) == 0
    Linv = _inv_unit_lower(LU, Lleaf, prec)
    Z = _solve_upper(LU, Linv, Uleaf, prec)  # U~^-1 L^-1
    bad = ~jnp.all(jnp.isfinite(Z), axis=(1, 2))
    if zero_singular:
        bad = bad | jnp.any(zero, axis=1)
    bad_i32 = bad.astype(jnp.int32)[:, None]
    invT = _permT(Z, g0, bad_i32, n)  # P^T Z^T, leading n x n
    return sign, logabs, invT, LU, g0, zero


# "lu": our LU kernels + GEMM recursion (default); "cusolver": cuSOLVER LU + cuBLAS trsm
GRAD_INVERSE = "lu"
