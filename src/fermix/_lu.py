"""Blocked LU with partial pivoting (Pallas Triton kernels) for slogdet: 1-warp
register-resident inner panels, 4-warp tensor-core inter-panel update / U-row / trailing
GEMM kernels, ping-pong buffers, virtual pivots. With factors=True, _lu_core also
returns what _inverse needs to rebuild the packed LU. All matrix buffers are *parts*
(tuples of real component arrays, see _field): one component for float32 / float64,
(re, im) for complex64 / complex128."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._field import (
    where,
    vsum,
    vadd,
    abs1,
    mag,
    recip,
    unit,
    dot,
    ld,
    st,
    mld,
    mst,
    read_sl,
    write_sl,
    init_sl,
    _pcall,
)
from ._common import (
    INNER,
    _next_pow2,
    _layout,
    _lat_layout,
    _lat_warps,
    _Blk,
    _split_blk,
    isum,
    _argmax_chunks,
    _unit_lower_inv,
    _full,
    _vec,
    _embed,
    _eye_pad,
    _panel_warps,
    _lu_block,
    _tune,
)


# ----------------------------------------------------------- inner panel (1 warp)
def _lu_inner_kernel(
    p_ref,
    q_ref,
    pos_ref,
    sl_ref,
    piv_ref,
    *rest,
    fld,
    n,
    r0,
    i,
    b,
    bi,
    layout,
    unroll_steps=False,
):
    """Inner panel i (columns [c0, c0+8)) of the block starting at r0; the m = n - r0
    active rows are held as a list of power-of-2 register chunks given as (absolute
    start row, height). Pivoting is virtual (pos), rows are never moved. Rows above
    r0 inside a chunk are dead (finished rows, read and rewritten unchanged); rows
    past n (a latency-mode tile overhanging the matrix) are masked. r0 is a static
    int, or None for the latency mode, where the block offset is read from the first
    ref of ``rest`` so that the blocks of one tile class share one compiled kernel."""
    r0, rest = _split_blk(r0, rest)
    dyn = not isinstance(r0, int)
    src_ref = rest[5]  # outputs: po, qo, poso, slo, pivo (aliased), src
    del rest
    chunks = list(layout)
    nin = b // bi
    c0 = r0 + i * bi
    ci = lax.broadcasted_iota(jnp.int32, (bi,), 0)
    ci2 = ci[None, :]
    ib = lax.broadcasted_iota(jnp.int32, (bi, bi), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bi, bi), 1)
    offs = [start - r0 for start, h in chunks]
    # active-local row index (negative above r0)
    heights = [h for _, h in chunks]
    trs = [
        lax.broadcasted_iota(jnp.int32, (h,), 0) + off for off, h in zip(offs, heights)
    ]
    rows = lambda start, h: pl.ds(start, h)
    # static in-bounds masks of the chunks overhanging the matrix (None: no overhang)
    inbs = [
        (
            (start + lax.broadcasted_iota(jnp.int32, (h,), 0)) < n
            if start + h > n
            else None
        )
        for start, h in chunks
    ]
    one = fld.rscalar(1.0)
    minus = fld.rscalar(-1.0)

    def load_rows(ref, c, cols):
        start, h = chunks[c]
        if inbs[c] is None:
            return ld(ref, (rows(start, h), cols))
        return mld(ref, (rows(start, h), cols), mask=inbs[c][:, None])

    if i == 0:
        poss = list(trs)
    else:
        poss = []
        for c, (start, h) in enumerate(chunks):
            if inbs[c] is None:
                poss.append(pos_ref[rows(start, h)])
            else:
                poss.append(
                    plgpu.load(pos_ref.at[rows(start, h)], mask=inbs[c], other=-1)
                )
    # unassigned active rows (a dead row above r0 keeps its negative pos from step
    # 0, a row past n is masked out)
    unas = [pos >= i * bi for pos in poss]
    unas = [un if inb is None else un & inb for un, inb in zip(unas, inbs)]
    sign, logabs = read_sl(sl_ref, fld.k)
    Ws = [load_rows(p_ref, c, pl.ds(c0, bi)) for c in range(len(chunks))]
    if i % 2 == 1:
        # prologue: W -= L_{i-1} @ (Linv_{i-1} @ A'[piv_{i-1}, cols_i])
        cp = c0 - bi
        piv_prev = piv_ref[pl.ds((i - 1) * bi, bi)]
        qp = r0 + (i - 1) * bi
        Linv_prev = ld(q_ref, (pl.ds(qp, bi), pl.ds(qp, bi)))
        Ap = ld(p_ref, (piv_prev, pl.ds(c0, bi)))
        U = vsum(Linv_prev[:, :, None] * Ap[None, :, :], axis=1)  # (bi, bi)
        for c, (start, h) in enumerate(chunks):
            Lp = load_rows(p_ref, c, pl.ds(cp, bi))
            for t in range(bi):
                lt = vsum(where(ci2 == t, Lp, 0.0), axis=1)
                ut = vsum(where(ib == t, U, 0.0), axis=0)
                Ws[c] = Ws[c] - lt[:, None] * ut[None, :]

    def step(j, carry):
        Ws, Ut, piv_i, poss, unas, sign, logabs = carry
        jg = i * bi + j
        u = vsum(where((jb == j) & (ib < j), Ut, 0.0), axis=1)
        v = where(ci == j, 1.0, where(ci < j, -u, 0.0))
        cols = [vsum(W * v[None, :], axis=1) for W in Ws]
        cands = [jnp.where(un, abs1(col), -1.0) for un, col in zip(unas, cols)]
        p = _argmax_chunks(cands, offs)
        rowp_parts = [
            vsum(where(tr[:, None] == p, W, 0.0), axis=0) for tr, W in zip(trs, Ws)
        ]
        rowp = vadd(rowp_parts)
        urow = rowp - vsum(Ut * rowp[:, None], axis=0)
        pivot = vsum(where(ci == j, urow, 0.0))
        Ut = where(ib == j, where(jb >= j, urow[None, :], 0.0), Ut)
        piv_i = jnp.where(ci == j, p, piv_i)
        q = sum(isum(jnp.where(tr == p, pos, 0)) for tr, pos in zip(trs, poss))
        poss = [
            jnp.where(tr == p, jg, jnp.where(pos == jg, q, pos))
            for tr, pos in zip(trs, poss)
        ]
        unas = [un & (tr != p) for tr, un in zip(trs, unas)]
        sign = sign * lax.select(q != jg, minus, one) * unit(pivot)
        logabs = logabs + jnp.log(mag(pivot))
        # zero pivot: L column 0, log -> -inf, sign 0
        inv = recip(pivot)
        Ws = [
            where((ci2 == j) & un[:, None], (col * inv)[:, None], W)
            for W, col, un in zip(Ws, cols, unas)
        ]
        return Ws, Ut, piv_i, poss, unas, sign, logabs

    Ut0 = fld.zeros((bi, bi))
    piv0 = jnp.zeros((bi,), jnp.int32)
    carry = (Ws, Ut0, piv0, poss, unas, sign, logabs)
    if unroll_steps:
        for j in range(bi):
            carry = step(j, carry)
    else:
        carry = lax.fori_loop(0, bi, step, carry)
    Ws, Ut, piv_i, poss, unas, sign, logabs = carry
    for c, (start, h) in enumerate(chunks):
        off, inb = offs[c], inbs[c]
        if dyn or off < 0 or inb is not None:
            # padded tile: keep the dead rows above r0 intact (they hold the previous
            # block's U rows) and stay inside the matrix
            alive = trs[c] >= 0
            if inb is not None:
                alive = alive & inb
            mst(p_ref, (rows(start, h), pl.ds(c0, bi)), Ws[c], mask=alive[:, None])
        else:
            st(p_ref, (rows(start, h), pl.ds(c0, bi)), Ws[c])
        # pos is written for the dead rows too (their negative step-0 value keeps
        # them unassigned in the later inner panels)
        if inb is None:
            pos_ref[rows(start, h)] = poss[c]
        else:
            plgpu.store(pos_ref.at[rows(start, h)], poss[c], mask=inb)
    plgpu.debug_barrier()
    write_sl(sl_ref, sign, logabs)
    piv_ref[pl.ds(i * bi, bi)] = r0 + piv_i  # global pivot rows
    if i % 2 == 0:
        # 8x8 unit-lower inverse of L_ii (pivot rows) -> Q diag block (used by the next
        # inner panel's prologue)
        Lii = ld(p_ref, (r0 + piv_i, pl.ds(c0, bi)))
        qc = r0 + i * bi
        st(q_ref, (pl.ds(qc, bi), pl.ds(qc, bi)), _unit_lower_inv(Lii, bi, fld, dyn))
    else:
        # 16x16 unit-lower inverse of the L block of panels (i-1, i) -> Q diag (update
        # kernel + urow kernel)
        plgpu.debug_barrier()
        cp = c0 - bi
        piv16 = piv_ref[pl.ds((i - 1) * bi, 2 * bi)]
        L16 = ld(p_ref, (piv16, pl.ds(cp, 2 * bi)))
        Linv16 = _unit_lower_inv(L16, 2 * bi, fld, dyn)
        qp = r0 + (i - 1) * bi
        st(q_ref, (pl.ds(qp, 2 * bi), pl.ds(qp, 2 * bi)), Linv16)
    if i == nin - 1:
        for tr, pos, un in zip(trs, poss, unas):
            plgpu.store(src_ref.at[r0 + pos], r0 + tr, mask=un)


# ----------------------------------------------------- inter-panel update (4 warps)
def _lu_update_kernel(
    p_ref, q_ref, piv_ref, *rest, n, r0, b, bi, ch, prec, lo, mid, hi
):
    """Right-looking update of inner panels [mid, hi) by the finished panels [lo, mid)
    (units of bi=8 columns):
    P[rows, target] -= L[rows, source] @ (L_ss^{-1} @ P[piv_source, target]).
    K = 8(mid-lo) in {16, 32}. r0 static or read from a ref (latency mode, where the
    grid is a class size and the programs past the last row do nothing)."""
    r0, rest = _split_blk(r0, rest)
    del rest  # the aliased output
    rc = r0 + pl.program_id(1) * ch
    args = (p_ref, q_ref, piv_ref, rc, n, r0, bi, prec, lo, mid, hi, ch)
    if not isinstance(r0, int):
        pl.when(rc < n)(lambda: _lu_update_body(*args))
        return
    _lu_update_body(*args)


def _lu_update_body(p_ref, q_ref, piv_ref, rc, n, r0, bi, prec, lo, mid, hi, ch):
    rows = rc + lax.broadcasted_iota(jnp.int32, (ch,), 0)
    rvalid = rows < n
    rmask = rvalid[:, None]
    K = (mid - lo) * bi
    Wd = (hi - mid) * bi
    c_lo = r0 + lo * bi
    c_mid = r0 + mid * bi
    piv_src = piv_ref[pl.ds(lo * bi, K)]
    if K == 16:
        Linv = ld(q_ref, (pl.ds(c_lo, 16), pl.ds(c_lo, 16)))
        A = ld(p_ref, (piv_src, pl.ds(c_mid, Wd)))
        U = dot(Linv, A, prec)  # (16, Wd)
        Wl = mld(p_ref, (pl.ds(rc, ch), pl.ds(c_lo, 16)), mask=rmask)
        upd = dot(Wl, U, prec)
    elif K == 32:  # block forward substitution with the two 16x16 inverse blocks
        pivA = piv_ref[pl.ds(lo * bi, 16)]
        pivB = piv_ref[pl.ds(lo * bi + 16, 16)]
        Linv_a = ld(q_ref, (pl.ds(c_lo, 16), pl.ds(c_lo, 16)))
        Linv_b = ld(q_ref, (pl.ds(c_lo + 16, 16), pl.ds(c_lo + 16, 16)))
        L_ba = ld(p_ref, (pivB, pl.ds(c_lo, 16)))
        Ua = dot(Linv_a, ld(p_ref, (pivA, pl.ds(c_mid, Wd))), prec)
        Rb = ld(p_ref, (pivB, pl.ds(c_mid, Wd))) - dot(L_ba, Ua, prec)
        Ub = dot(Linv_b, Rb, prec)
        Wa = mld(p_ref, (pl.ds(rc, ch), pl.ds(c_lo, 16)), mask=rmask)
        Wb = mld(p_ref, (pl.ds(rc, ch), pl.ds(c_lo + 16, 16)), mask=rmask)
        upd = dot(Wa, Ua, prec) + dot(Wb, Ub, prec)
    else:  # K = 64 (the 128-block): block forward substitution over K / 16 sub-blocks
        nsub = K // 16
        Us = []
        for s_ in range(nsub):
            piv_s = piv_ref[pl.ds(lo * bi + 16 * s_, 16)]
            R = ld(p_ref, (piv_s, pl.ds(c_mid, Wd)))
            for t_ in range(s_):
                L_st = ld(p_ref, (piv_s, pl.ds(c_lo + 16 * t_, 16)))
                R = R - dot(L_st, Us[t_], prec)
            Linv_s = ld(q_ref, (pl.ds(c_lo + 16 * s_, 16), pl.ds(c_lo + 16 * s_, 16)))
            Us.append(dot(Linv_s, R, prec))
        Ws = [
            mld(p_ref, (pl.ds(rc, ch), pl.ds(c_lo + 16 * s_, 16)), mask=rmask)
            for s_ in range(nsub)
        ]
        upd = dot(Ws[0], Us[0], prec)
        for s_ in range(1, nsub):
            upd = upd + dot(Ws[s_], Us[s_], prec)
    T = mld(p_ref, (pl.ds(rc, ch), pl.ds(c_mid, Wd)), mask=rmask)
    # never write the pivot rows: other programs gather A from them (cross-program RAW
    # hazard otherwise)
    hits = isum(rows[:, None] == piv_src[None, :], axis=1)
    keep = (hits == 0) & rvalid
    mst(p_ref, (pl.ds(rc, ch), pl.ds(c_mid, Wd)), T - upd, mask=keep[:, None])


# ------------------------------------------------------------------ urow (4 warps)
def _lu_urow_kernel(p_ref, piv_ref, q_ref, *rest, n, r0, b, tn, prec):
    r0, rest = _split_blk(r0, rest)
    del rest  # the aliased output
    c0 = r0 + b + pl.program_id(1) * tn
    args = (p_ref, piv_ref, q_ref, c0, n, r0, b, tn, prec)
    if not isinstance(r0, int):
        pl.when(c0 < n)(lambda: _lu_urow_body(*args))
        return
    _lu_urow_body(*args)


def _lu_urow_body(p_ref, piv_ref, q_ref, c0, n, r0, b, tn, prec):
    """U_k = L_kk^-1 A[pivot rows, cols] by 16-row block forward substitution: the
    pivot rows are gathered through piv and U goes to q_ref rows [r0, r0+b)."""
    bu = 16
    nblk = b // bu
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    cvalid = (c0 + cj) < n
    cmask = cvalid[None, :]
    U_list = []
    for i in range(nblk):
        rows_i = piv_ref[pl.ds(i * bu, bu)]
        q_i = pl.ds(r0 + i * bu, bu)
        R = mld(p_ref, (rows_i, pl.ds(c0, tn)), mask=cmask)
        for s_ in range(i):
            Lis = ld(p_ref, (rows_i, pl.ds(r0 + s_ * bu, bu)))
            R = R - dot(Lis, U_list[s_], prec)
        Linv = ld(q_ref, (q_i, q_i))
        Ui = dot(Linv, R, prec)
        U_list.append(Ui)
        mst(q_ref, (q_i, pl.ds(c0, tn)), Ui, mask=cmask)


# ------------------------------------------------------------------ gemm (4 warps)
def _lu_gemm_kernel(p_ref, q_ref, src_ref, *rest, n, r0, b, tm, tn, nt, prec):
    """Trailing update C -= L U of nt consecutive tm x tn tiles of one column strip per
    program (rows gathered through src); nt > 1 runs them in a lax.fori_loop with the U
    strip loaded once, which lifts the occupancy-limited single-tile programs (122
    registers of gather pointers, 4 CTAs/SM on H200) to +7-15 % at n >= 256; extra
    num_stages measured nothing on top. r0 static or from a ref (latency mode: class
    grid, empty programs return)."""
    r0, rest = _split_blk(r0, rest)
    del rest  # the aliased output
    i_base = pl.program_id(1) * (tm * nt)
    c0 = pl.program_id(2) * tn
    s0 = r0 + b
    mrem = n - s0
    args = (p_ref, q_ref, src_ref, i_base, c0, s0, mrem, n, r0, b, tm, tn, nt, prec)
    if not isinstance(r0, int):
        pl.when((i_base < mrem) & (c0 < mrem))(lambda: _lu_gemm_body(*args))
        return
    _lu_gemm_body(*args)


def _lu_gemm_body(
    p_ref, q_ref, src_ref, i_base, c0, s0, mrem, n, r0, b, tm, tn, nt, prec
):
    """C rows gathered from p_ref through src, updated by L U (U in q_ref) and
    written compacted into q_ref."""
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    cvalid = (c0 + cj) < mrem
    cols = pl.ds(s0 + c0, tn)
    # K slices of at most 64 columns (Triton's dot collapses above K = 64, and the
    # lowering has no register-tile slicing): one slice for b <= 64, two for the
    # 128-block; the U strip is loaded once per program
    kw = min(b, 64)
    ks = list(range(0, b, kw))
    Us = [mld(q_ref, (pl.ds(r0 + k0, kw), cols), mask=cvalid[None, :]) for k0 in ks]

    def tile(i0):
        rvalid = (i0 + ri) < mrem
        mask = rvalid[:, None] & cvalid[None, :]
        src = plgpu.load(src_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
        C = mld(p_ref, (src, cols), mask=mask)
        Lt0 = ld(p_ref, (src, pl.ds(r0, kw)))
        acc = dot(Lt0, Us[0], prec)
        for k0, U in zip(ks[1:], Us[1:]):
            Lt = ld(p_ref, (src, pl.ds(r0 + k0, kw)))
            acc = acc + dot(Lt, U, prec)
        C = C - acc
        mst(q_ref, (pl.ds(s0 + i0, tm), cols), C, mask=mask)

    if nt == 1:
        tile(i_base)
    else:

        def body(t, carry):
            tile(pl.multiple_of(i_base + t * tm, tm))
            return carry

        # dynamic trip count: skip the fully masked tiles past the last row
        ntile = jnp.minimum(nt, (mrem - i_base + tm - 1) // tm)
        lax.fori_loop(0, ntile, body, 0)


# ----------------------------------------------------------------------- calls


def _inner_call(
    P,
    Q,
    pos,
    sl,
    piv,
    *,
    fld,
    blk,
    i,
    b,
    bi,
    layout,
    num_warps,
    unroll_steps,
    fresh=False,
):
    """fresh=True: Q, pos, sl, piv, src are allocated as uninitialized outputs (first
    call only)."""
    B, n, _ = P[0].shape
    kern = functools.partial(
        _lu_inner_kernel,
        fld=fld,
        n=n,
        r0=blk.r0,
        i=i,
        b=b,
        bi=bi,
        layout=layout,
        unroll_steps=unroll_steps,
    )
    mat = fld.structs((B, n, n))
    idx_n = jax.ShapeDtypeStruct((B, n), jnp.int32)
    idx_b = jax.ShapeDtypeStruct((B, b), jnp.int32)
    sl_shape = jax.ShapeDtypeStruct((B, fld.k + 1), fld.real)
    outs = [
        (mat, _full(n)),
        (mat, _full(n)),
        (idx_n, _vec(n)),
        (sl_shape, _vec(fld.k + 1)),
        (idx_b, _vec(b)),
        (idx_n, _vec(n)),
    ]
    if fresh:
        return _pcall(
            functools.partial(_fresh_wrapper, kern, fld=fld, lat=blk.lat),
            [(P, _full(n))] + blk.ins(),
            outs,
            (B,),
            aliases={0: 0},
            num_warps=num_warps,
        )
    ins = [
        (P, _full(n)),
        (Q, _full(n)),
        (pos, _vec(n)),
        (sl, _vec(fld.k + 1)),
        (piv, _vec(b)),
    ] + blk.ins()
    aliases = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    return _pcall(kern, ins, outs, (B,), aliases=aliases, num_warps=num_warps)


def _fresh_wrapper(kern, p_ref, *rest, fld, lat):
    """First inner call: initialise sign/log and call the kernel with the outputs acting
    as the (uninitialised) state."""
    blk = rest[:1] if lat else ()
    po_ref, qo_ref, poso_ref, slo_ref, pivo_ref, src_ref = rest[1:] if lat else rest
    init_sl(slo_ref, fld)
    state = (qo_ref, poso_ref, slo_ref, pivo_ref)
    kern(p_ref, *state, *blk, po_ref, *state, src_ref)


def _update_call(P, Q, piv, *, fld, blk, b, bi, ch, prec, num_warps, lo, mid, hi):
    B, n, _ = P[0].shape
    m = blk.mp
    ch = min(ch, _next_pow2(m))
    kern = functools.partial(
        _lu_update_kernel,
        n=n,
        r0=blk.r0,
        b=b,
        bi=bi,
        ch=ch,
        prec=prec,
        lo=lo,
        mid=mid,
        hi=hi,
    )
    ins = [(P, _full(n)), (Q, _full(n)), (piv, _vec(b))] + blk.ins()
    outs = [(fld.structs((B, n, n)), _full(n))]
    grid = (B, blk.grid(-(-m // ch)))
    return _pcall(kern, ins, outs, grid, aliases={0: 0}, num_warps=num_warps)[0]


def _urow_call(P, piv, Q, *, fld, blk, b, tn, prec, num_warps):
    """The U_k rows of the block into Q."""
    B, n, _ = P[0].shape
    nct = -(-(blk.mp - b) // tn)
    kw = dict(n=n, r0=blk.r0, b=b, tn=tn, prec=prec)
    kern = functools.partial(_lu_urow_kernel, **kw)
    ins = [(P, _full(n)), (piv, _vec(b)), (Q, _full(n))] + blk.ins()
    outs = [(fld.structs((B, n, n)), _full(n))]
    grid = (B, blk.grid(nct))
    return _pcall(kern, ins, outs, grid, aliases={2: 0}, num_warps=num_warps)[0]


def _gemm_call(P, Q, src, *, fld, blk, b, tm, tn, prec, num_warps, nt=1, stages=1):
    B, n, _ = P[0].shape
    mrem = blk.mp - b
    nrt = -(-mrem // tm)
    if not blk.lat:
        nt = min(nt, nrt)  # latency mode: the dynamic trip count handles short strips
    shapes = dict(n=n, r0=blk.r0, b=b, tm=tm, tn=tn, nt=nt, prec=prec)
    kern = functools.partial(_lu_gemm_kernel, **shapes)
    ins = [(P, _full(n)), (Q, _full(n)), (src, _vec(n))] + blk.ins()
    outs = [(fld.structs((B, n, n)), _full(n))]
    grid = (B, blk.grid(-(-nrt // nt)), blk.grid(-(-mrem // tn)))
    return _pcall(
        kern, ins, outs, grid, aliases={1: 0}, num_warps=num_warps, num_stages=stages
    )[0]


def _update_schedule(nin):
    """Inter-panel updates of the recursive right-looking block: {i: [(lo, mid, hi),
    ...]} in units of inner panels, applied after inner panel i -- a group of g panels
    factors its first half, updates its second half by it (K = 4 g columns) and
    factors the second half; consecutive panels (g = 2) are handled by the odd
    panel's prologue. nin = 8 gives (0, 2, 4) after panel 1, (0, 4, 8) after 3,
    (4, 6, 8) after 5; nin = 16 adds the K = 64 level (0, 8, 16) after panel 7."""
    sched: dict = {}

    def rec(a, g):
        if g <= 2:
            return
        h = g // 2
        rec(a, h)
        sched.setdefault(a + h - 1, []).append((a, a + h, a + g))
        rec(a + h, h)

    rec(0, nin)
    return sched


def _lu_core(A, n, fld, prec, unroll_steps, block=None, factors=False) -> tuple:
    """Blocked LU of the (B, n, n) batch given as parts; returns (sign, logabs) as
    values (sign a CVal for complex fields) and, with factors=True, also the raw
    material for the packed LU (see _packed_lu): the two ping-pong buffers, per-block
    pivot rows / compaction maps and a snapshot of each block's panel columns taken
    before its factorisation (the pivot rows' U part is recomputed from it).
    unroll_steps=None / block=None take the architecture table's choice."""
    t = _tune(fld.kind)
    b = _lu_block(n, t) if block is None else block
    lat = n > t.latency_min_n  # see Tune.latency_min_n
    if unroll_steps is None:
        unroll_steps = n <= t.lu_unroll_max_n and not lat
    bi = INNER
    nin = b // bi
    assert nin in (4, 8, 16)
    sched = _update_schedule(nin)
    P, nb = _embed(A, n, _eye_pad, fld, b)
    N = P[0].shape[1]
    base_kw = dict(fld=fld, b=b, bi=bi, unroll_steps=unroll_steps)
    Q = pos = sl = piv = src = None
    pivs, srcs, snaps = [], [], []
    for k in range(nb):
        r0 = k * b
        last = k == nb - 1
        m = N - r0
        if lat:
            layout = _lat_layout(m, r0, N)
            pw = _lat_warps(layout[0][1], t)
        else:
            rel = _layout(m, r0, t.lu_chunk_cost)
            layout = tuple((r0 + off, h) for off, h in rel)
            pw = _panel_warps(sum(h for _, h in layout), t)
        blk = _Blk(r0, lat, layout[0][1] if lat else m)
        panel_kw = dict(blk=blk, layout=layout, num_warps=pw, **base_kw)
        upd_kw = dict(fld=fld, blk=blk, b=b, bi=bi, ch=t.lu_upd_ch, prec=prec)
        upd_kw.update(num_warps=t.lu_upd_warps)
        if factors:
            # the barrier forces the slice to be taken before the in-place panel kernels
            # (otherwise XLA fuses it into its downstream gather and has to copy the
            # whole buffer once per block)
            snap = tuple(Pc[:, r0:, r0 : r0 + b] for Pc in P)
            snap, P = lax.optimization_barrier((snap, P))
            snaps.append(snap)
        for i in range(nin):
            fresh = k == 0 and i == 0
            P, Q, pos, sl, piv, src = _inner_call(
                P, Q, pos, sl, piv, i=i, fresh=fresh, **panel_kw
            )
            for lo, mid, hi in sched.get(i, ()):
                P = _update_call(P, Q, piv, lo=lo, mid=mid, hi=hi, **upd_kw)
        if factors:
            pivs.append(piv)
            srcs.append(src)
        if not last:
            urow_kw = dict(fld=fld, blk=blk, b=b, tn=t.lu_urow_tn, prec=prec)
            urow_kw.update(num_warps=t.lu_urow_warps)
            gemm_kw = dict(fld=fld, blk=blk, b=b, tm=t.lu_gemm_tm, tn=t.lu_gemm_tn)
            gemm_kw.update(prec=prec, nt=t.lu_gemm_nt, stages=t.lu_gemm_stages)
            Q = _urow_call(P, piv, Q, **urow_kw)
            Q = _gemm_call(P, Q, src, num_warps=t.lu_gemm_warps, **gemm_kw)
            P, Q = Q, P
    assert sl is not None  # nb >= 1: the panel kernels always ran
    sign = _sl_sign(sl, fld)
    logabs = sl[:, fld.k]
    if factors:
        # bufs[k % 2] holds the panel of block k
        bufs = (P, Q) if (nb - 1) % 2 == 0 else (Q, P)
        return sign, logabs, (bufs, pivs, srcs, snaps, b, N)
    return sign, logabs


def _sl_sign(sl, fld):
    """The sign column(s) of the (B, k + 1) accumulator as a jnp array of the field
    (complex: |sign| renormalised to 1, the product of the pivots' phases drifts by
    ~n ulp)."""
    if fld.cplx:
        s = lax.complex(sl[:, 0], sl[:, 1])
        return s / jnp.where(s == 0, 1.0, jnp.abs(s))
    return sl[:, 0]
