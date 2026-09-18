"""Blocked Parlett-Reid tridiagonalisation with partial pivoting (Pallas Triton kernels)
for slogpf: pair steps in register tiles, rank-2 updates as one K=32 GEMM over the
lower-triangle tiles with mirrored stores. Matrix buffers are parts (see _field)."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._field import (
    parts,
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
    mst,
    read_sl,
    write_sl,
    init_sl,
    _pcall,
)
from ._common import (
    BLOCK,
    INNER,
    _layout,
    isum,
    _argmax_chunks,
    _full,
    _vec,
    _embed,
    _panel_warps,
    _skew_pad,
    _tune,
)
from ._lu import _sl_sign

# ============================================================ pfaffian kernels
# Layout invariant per block k (u, v in the block's compact numbering [r0, n)):
#     X_k[phi_k(u), v] = M'_k[u, v]     rows physically permuted by phi_k
#                                       (= src_{k-1}), columns compact.
# The panel reads row a as X_k[phi_k(a), :]; the update computes the lower-triangle
# tiles of M''[i, j] = M_upd[src_k(i), src_k(j)] and stores X_{k+1}[src_j, i] = -M''[i,
# j] and X_{k+1}[src_i, j] = M''[i, j] (skew symmetry), so no separate column-compaction
# pass is needed.


def _pf_panel_kernel(
    p_ref,
    phys_ref,
    sl_ref,
    po_ref,
    slo_ref,
    src_ref,
    physn_ref,
    posn_ref,
    gbuf_ref,
    gbufT_ref,
    d_ref=None,
    par_ref=None,
    *,
    fld,
    n,
    r0,
    b,
    bi,
    layout,
    identity_phys=False,
):
    """Parlett-Reid block step on the m = n - r0 active indices (chunked register
    tiles). Row a of the current matrix is X[phi(a), :]; columns are compact. tau/w
    vectors are kept in G (m x 8 per inner block) and stored as rows of gbuf/gbufT for
    the update kernel and later inner blocks.

    With ``d_ref`` / ``par_ref`` (the factors export for the gradient) the kernel also
    stores each pair's pivot d (B, b/2), the block's permutation parity (B, 1) and the
    pair rows' final positions into src (src[r0 + rank] = block index of the row), so
    that P S P^T = L T L^T can be assembled from gbuf (see _pfinv); the arithmetic of
    the factorisation is untouched, so (sign, logabs) stay bit-identical."""
    del po_ref
    factors = d_ref is not None
    m = n - r0
    chunks = list(layout)
    nin = b // bi
    hi = bi // 2
    ci = lax.broadcasted_iota(jnp.int32, (bi,), 0)
    ci2 = ci[None, :]
    ib = lax.broadcasted_iota(jnp.int32, (bi, bi), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bi, bi), 1)
    upper = (ib < hi) & (jb == ib + hi)
    lower = (ib >= hi) & (jb == ib - hi)
    Pm = jnp.where(upper, -1.0, jnp.where(lower, 1.0, 0.0)).astype(fld.real)
    offs = [off for off, h in chunks]
    trs = [lax.broadcasted_iota(jnp.int32, (h,), 0) + off for off, h in chunks]
    rows = lambda off, h: pl.ds(r0 + off, h)
    one = fld.rscalar(1.0)
    minus = fld.rscalar(-1.0)

    # rows above r0 (negative index) are dead padding
    actives = [tr >= 0 for tr in trs]
    unas = list(actives)
    ranks = [jnp.full((h,), -1, jnp.int32) for off, h in chunks]
    sign, logabs = read_sl(sl_ref, fld.k)
    if identity_phys:
        physv = [r0 + tr for tr in trs]
    else:
        physv = [phys_ref[rows(off, h)] for off, h in chunks]

    def phys_row(a):
        if identity_phys:
            return r0 + a
        return sum(isum(jnp.where(tr == a, pv, 0)) for tr, pv in zip(trs, physv))

    def corrected_col(a, Gs, i):
        """updated column a of the trailing matrix (all active rows, chunked): -row_a +
        corrections of this block's earlier pairs (register G) and of earlier inner
        blocks (gbufT rows)."""
        pr = phys_row(a)
        g_parts = [
            vsum(where(tr[:, None] == a, G, 0.0), axis=0) for tr, G in zip(trs, Gs)
        ]
        g_a = vadd(g_parts)
        h_a = vsum(Pm * g_a[None, :], axis=1)
        hps = []
        for ip in range(i):
            g_p = ld(gbufT_ref, (pl.ds(ip * bi, bi), r0 + a))
            hps.append(vsum(Pm * g_p[None, :], axis=1))
        cols = []
        for c, (off, h) in enumerate(chunks):
            row_a = ld(p_ref, (pr, rows(off, h)))
            col = -row_a + vsum(Gs[c] * h_a[None, :], axis=1)
            for ip in range(i):
                Gt = ld(gbufT_ref, (pl.ds(ip * bi, bi), rows(off, h)))
                col = col + vsum(Gt * hps[ip][:, None], axis=0)
            cols.append(col)
        return cols

    for i in range(nin):

        def step(s_, carry):
            Gs, unas, ranks, sign, logabs = carry
            sg = i * hi + s_
            mins = [jnp.min(jnp.where(un, tr, m)) for tr, un in zip(trs, unas)]
            a = functools.reduce(jnp.minimum, mins)
            col_a = corrected_col(a, Gs, i)
            cands = [
                jnp.where(un & (tr != a), abs1(col), -1.0)
                for tr, un, col in zip(trs, unas, col_a)
            ]
            p = _argmax_chunks(cands, offs)
            # M[a, p]
            d = -vadd([vsum(where(tr == p, col, 0.0)) for tr, col in zip(trs, col_a)])
            col_p = corrected_col(p, Gs, i)
            newmask = [un & (tr != a) & (tr != p) for tr, un in zip(trs, unas)]
            inv = recip(d)  # zero pivot guard

            def pair_cols(G, nm, ca, cp):
                """columns s_ and hi+s_ of G: the tau and w vectors of this pair
                step."""
                tau = where(nm, -(ca * inv), 0.0)[:, None]
                w = -where(nm, cp, 0.0)[:, None]
                return where(ci2 == s_, tau, where(ci2 == hi + s_, w, G))

            Gs = [pair_cols(*z) for z in zip(Gs, newmask, col_a, col_p)]
            ranks = [
                jnp.where(tr == a, 2 * sg, jnp.where(tr == p, 2 * sg + 1, rk))
                for tr, rk in zip(trs, ranks)
            ]
            sign = sign * unit(d)
            logabs = logabs + jnp.log(mag(d))
            if d_ref is not None:
                for r, comp in zip(d_ref, parts(d)):
                    plgpu.store(r.at[pl.ds(sg, 1)], comp[None])
            return Gs, newmask, ranks, sign, logabs

        G0 = [fld.zeros((h, bi)) for off, h in chunks]
        carry = (G0, unas, ranks, sign, logabs)
        Gs, unas, ranks, sign, logabs = lax.fori_loop(0, hi, step, carry)
        for c, (off, h) in enumerate(chunks):
            st(gbuf_ref, (rows(off, h), pl.ds(i * bi, bi)), Gs[c])
            st(gbufT_ref, (pl.ds(i * bi, bi), rows(off, h)), Gs[c].T)
        plgpu.debug_barrier()

    # ---- parity of the permutation
    # [assigned rows in assignment order, unassigned rows in index order]
    cb = lax.broadcasted_iota(jnp.int32, (b,), 0)
    inv1 = 0
    ordb = jnp.zeros((b,), jnp.int32)
    cnt_a = 0
    cnt_u = 0
    s0 = r0 + b
    for c, (off, h) in enumerate(chunks):
        assigned = actives[c] & (~unas[c])
        rank_a = jnp.cumsum(assigned.astype(jnp.int32)) - 1 + cnt_a
        rank_u = jnp.cumsum(unas[c].astype(jnp.int32)) - 1 + cnt_u
        inv1 = inv1 + isum(jnp.where(assigned, trs[c] - rank_a, 0))
        hit = (rank_a[None, :] == cb[:, None]) & assigned[None, :]
        ordb = ordb + isum(jnp.where(hit, ranks[c][None, :], 0), axis=1)
        cnt_a = cnt_a + isum(assigned)
        cnt_u = cnt_u + isum(unas[c])
        # src: compact-next index -> this block's index; physn: compact-next -> physical
        # row of X_k; posn: inverse map
        plgpu.store(src_ref.at[s0 + rank_u], r0 + trs[c], mask=unas[c])
        plgpu.store(physn_ref.at[s0 + rank_u], physv[c], mask=unas[c])
        posn_ref[rows(off, h)] = jnp.where(unas[c], s0 + rank_u, -1)
        if factors:
            # final position r0 + rank of the pair rows (a: 2 sg, p: 2 sg + 1)
            plgpu.store(src_ref.at[r0 + ranks[c]], r0 + trs[c], mask=assigned)
    pair = (cb[:, None] < cb[None, :]) & (ordb[:, None] > ordb[None, :])
    inv2 = isum(pair)
    odd = ((inv1 + inv2) % 2) == 1
    sign = sign * lax.select(odd, minus, one)
    write_sl(slo_ref, sign, logabs)
    if par_ref is not None:
        par_ref[0] = lax.select(odd, jnp.int32(1), jnp.int32(0))


def _pf_update_kernel(
    p_ref,
    gbuf_ref,
    gbufT_ref,
    src_ref,
    physn_ref,
    posn_ref,
    q_ref,
    out_ref,
    *,
    fld,
    n,
    r0,
    b,
    bi,
    tm,
    tn,
    nt,
    prec,
    three,
    skip,
):
    """Lower-triangle tiles (compact-next i >= j) of M'' = M_upd[src_i, src_j]; stored
    twice into X_{k+1}: X_{k+1}[src_j, i] = -M''[i, j] (transposed) and
    X_{k+1}[src_i, j] = M''[i, j] (mirror). Each program handles nt consecutive row
    tiles of one column strip (a lax.fori_loop for nt > 1), so the gathered H^T strip
    is built once per program; with skip=True the row tiles start at the diagonal
    tile."""
    del out_ref, posn_ref
    hi = bi // 2
    s0 = r0 + b
    mrem = n - s0
    j0 = pl.program_id(2) * tn
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    cvalid = (j0 + cj) < mrem
    i_prog = pl.program_id(1) * (tm * nt)
    if skip:
        i_start = pl.multiple_of((j0 // tm) * tm + i_prog, tm)
    else:
        i_start = pl.multiple_of(i_prog, tm)

    def body():
        srcj = plgpu.load(src_ref.at[pl.ds(s0 + j0, tn)], mask=cvalid, other=n - 1)
        cb = lax.broadcasted_iota(jnp.int32, (b,), 0)
        within = cb % bi
        perm = jnp.where(within < hi, cb + hi, cb - hi)
        sgn = jnp.where(within < hi, -1.0, 1.0).astype(fld.real)
        # (b, tn) gather
        HcT = sgn[:, None] * ld(gbufT_ref, (perm[:, None], srcj[None, :]))

        def tile(i0):
            rvalid = (i0 + ri) < mrem
            srci = plgpu.load(src_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
            phys_blk = physn_ref.at[pl.ds(s0 + i0, tm)]
            phys = plgpu.load(phys_blk, mask=rvalid, other=n - 1)
            Gs = ld(gbuf_ref, (srci, slice(None)))  # (tm, b)
            C = ld(p_ref, (phys[:, None], srcj[None, :]))  # (tm, tn) 2-D gather
            out = C + dot(Gs, HcT, prec, three)  # M''[i, j]
            if tm == tn:
                # diagonal tiles: make the tile exactly skew-symmetric (lower part
                # authoritative) so the primary and mirror stores agree bit-for-bit
                # and the stored matrix stays exactly skew (the panel reads columns
                # as rows)
                skew = where(ri[:, None] >= cj[None, :], out, -out.T)
                out = where(i0 == j0, skew, out)
            mask = rvalid[:, None] & cvalid[None, :]
            maskT = cvalid[:, None] & rvalid[None, :]
            mst(q_ref, (srcj, pl.ds(s0 + i0, tm)), -out.T, mask=maskT)
            mst(q_ref, (srci, pl.ds(s0 + j0, tn)), out, mask=mask)

        if nt == 1:
            tile(i_start)
        else:

            def loop(t, carry):
                tile(pl.multiple_of(i_start + t * tm, tm))
                return carry

            # dynamic trip count: skip the fully masked tiles past the last row
            ntile = jnp.minimum(nt, (mrem - i_start + tm - 1) // tm)
            lax.fori_loop(0, ntile, loop, 0)

    if skip:
        pl.when(i_start < mrem)(body)
    else:
        body()


def _pf_panel_call(
    P,
    phys,
    sl,
    gbuf,
    gbufT,
    *,
    fld,
    r0,
    b,
    bi,
    layout,
    num_warps,
    fresh,
    factors=False,
):
    """factors=True appends the pivots (B, b/2) and the parity (B, 1) to the outputs
    and completes src for the pair rows (see _pf_panel_kernel)."""
    B, n, _ = P[0].shape
    kern = functools.partial(
        _pf_panel_kernel,
        fld=fld,
        n=n,
        r0=r0,
        b=b,
        bi=bi,
        layout=layout,
        identity_phys=fresh,
    )
    gbuf_spec = pl.BlockSpec((None, n, b), lambda *i: (i[0], 0, 0))
    gbufT_spec = pl.BlockSpec((None, b, n), lambda *i: (i[0], 0, 0))
    mat = fld.structs((B, n, n))
    sl_shape = jax.ShapeDtypeStruct((B, fld.k + 1), fld.real)
    idx_n = jax.ShapeDtypeStruct((B, n), jnp.int32)
    outs = [
        (mat, _full(n)),
        (sl_shape, _vec(fld.k + 1)),
        (idx_n, _vec(n)),
        (idx_n, _vec(n)),
        (idx_n, _vec(n)),
        (fld.structs((B, n, b)), gbuf_spec),
        (fld.structs((B, b, n)), gbufT_spec),
    ]
    if factors:
        par = jax.ShapeDtypeStruct((B, 1), jnp.int32)
        outs += [(fld.structs((B, b // 2)), _vec(b // 2)), (par, _vec(1))]
    if fresh:

        def wrapper(p_ref, po_ref, slo_ref, src_ref, physn_ref, posn_ref, *rest):
            init_sl(slo_ref, fld)
            outs = (src_ref, physn_ref, posn_ref)
            kern(p_ref, src_ref, slo_ref, po_ref, slo_ref, *outs, *rest)

        ins = [(P, _full(n))]
        return _pcall(wrapper, ins, outs, (B,), aliases={0: 0}, num_warps=num_warps)
    ins = [(P, _full(n)), (phys, _vec(n)), (sl, _vec(fld.k + 1))]
    aliases = {0: 0, 2: 1}
    return _pcall(kern, ins, outs, (B,), aliases=aliases, num_warps=num_warps)


def _pf_update_call(
    P,
    gbuf,
    gbufT,
    src,
    physn,
    posn,
    Q,
    *,
    fld,
    r0,
    b,
    bi,
    tm,
    tn,
    prec,
    three,
    num_warps,
    fresh,
    skip,
    nt=1,
):
    B, n, _ = P[0].shape
    nrt = -(-(n - r0 - b) // tm)
    nct = -(-(n - r0 - b) // tn)
    nt = min(nt, nrt)
    nrt = -(-nrt // nt)
    shapes = dict(n=n, r0=r0, b=b, bi=bi, tm=tm, tn=tn, nt=nt, prec=prec)
    shapes.update(three=three, skip=skip)
    kern = functools.partial(_pf_update_kernel, fld=fld, **shapes)
    gbuf_spec = pl.BlockSpec((None, n, b), lambda *i: (i[0], 0, 0))
    gbufT_spec = pl.BlockSpec((None, b, n), lambda *i: (i[0], 0, 0))
    ins = [
        (P, _full(n)),
        (gbuf, gbuf_spec),
        (gbufT, gbufT_spec),
        (src, _vec(n)),
        (physn, _vec(n)),
        (posn, _vec(n)),
    ]
    outs = [(fld.structs((B, n, n)), _full(n))]
    grid = (B, nrt, nct)
    if fresh:

        def wrapper(p_ref, gbuf_ref, gbufT_ref, src_ref, physn_ref, posn_ref, out_ref):
            ins = (p_ref, gbuf_ref, gbufT_ref, src_ref, physn_ref, posn_ref)
            kern(*ins, out_ref, out_ref)  # no Q yet: q_ref is the output itself

        return _pcall(wrapper, ins, outs, grid, num_warps=num_warps)[0]
    ins.append((Q, _full(n)))
    return _pcall(kern, ins, outs, grid, aliases={6: 0}, num_warps=num_warps)[0]


def _pf_core(A, n, fld, prec, tm=None, tn=None, upd_warps=None):
    """Blocked Parlett-Reid of the skew-symmetric (B, n, n) batch given as parts;
    returns (sign, logabs) as jnp arrays (sign complex for complex fields). None for
    tm / tn / upd_warps selects the architecture table (_common.Tune)."""
    sign, logabs, _ = _pf_run(A, n, fld, prec, tm, tn, upd_warps, False)
    return sign, logabs


def _pf_core_factors(A, n, fld, prec, upd_warps=None):
    """_pf_core plus the raw material of the block LDL^T P S P^T = L D L^T for the
    gradient (see _pfinv._pf_parts): per block the tau / w buffer gbuf (B, N, b), the
    row map src (final-or-next index -> block index), the pivots (B, b/2) and the
    permutation parity (B, 1), plus b and N. The same arithmetic as the plain forward
    (its kernels only store more), so (sign, logabs) are bit-identical to it."""
    sign, logabs, fac = _pf_run(A, n, fld, prec, None, None, upd_warps, True)
    assert fac is not None
    return sign, logabs, fac


def _pf_run(A, n, fld, prec, tm, tn, upd_warps, factors):
    t = _tune(fld.kind)
    tm = t.pf_tm if tm is None else tm
    tn = t.pf_tn if tn is None else tn
    upd_warps = t.pf_upd_warps if upd_warps is None else upd_warps
    nt = t.pf_upd_nt
    b, bi = BLOCK, INNER
    P, nb = _embed(A, n, _skew_pad, fld)
    N = P[0].shape[1]
    Q = sl = gbuf = gbufT = phys = None
    gbufs, srcs, ds, pars = [], [], [], []
    for k in range(nb):
        r0 = k * b
        last = k == nb - 1
        layout = tuple(_layout(N - r0, r0, t.pf_chunk_cost, t.pf_max_chunk))
        pw = _panel_warps(sum(h for _, h in layout), t, pf=True)
        panel_kw = dict(fld=fld, r0=r0, b=b, bi=bi, layout=layout, num_warps=pw)
        upd_kw = dict(fld=fld, r0=r0, b=b, bi=bi, tm=tm, tn=tn, prec=prec)
        upd_kw.update(three=t.cplx_dot3, num_warps=upd_warps, nt=nt)
        outs = _pf_panel_call(
            P, phys, sl, gbuf, gbufT, fresh=(k == 0), factors=factors, **panel_kw
        )
        P, sl, src, physn, posn, gbuf, gbufT = outs[:7]
        if factors:
            gbufs.append(gbuf)
            srcs.append(src)
            ds.append(outs[7])
            pars.append(outs[8])
        if not last:
            upd_args = (P, gbuf, gbufT, src, physn, posn, Q)
            Q = _pf_update_call(*upd_args, fresh=(k == 0), skip=True, **upd_kw)
            P, Q = Q, P
            phys = src  # phi_{k+1} = src_k
    assert sl is not None  # nb >= 1: the panel kernel always ran
    sign, logabs = _sl_sign(sl, fld), sl[:, fld.k]
    if factors:
        return sign, logabs, (gbufs, srcs, ds, pars, b, N)
    return sign, logabs, None
