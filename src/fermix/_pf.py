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
    mld,
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
    _lat_layout,
    _lat_warps,
    _Blk,
    _split_blk,
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
    *rest,
    fld,
    n,
    r0,
    b,
    bi,
    layout,
    identity_phys=False,
    factors=False,
    use_gt=True,
):
    """Parlett-Reid block step on the m = n - r0 active indices (chunked register
    tiles given as (absolute start row, height); rows above r0 are dead, rows past n
    -- a latency-mode tile overhanging the matrix -- are masked). Row a of the
    current matrix is X[phi(a), :]; columns are compact. tau/w vectors are kept in G
    (m x 8 per inner block) and stored as rows of gbuf/gbufT for the update kernel and
    later inner blocks. r0 is static or, in the latency mode, read from the first ref
    of ``rest`` (the outputs follow: po, slo, src, physn, posn, gbuf, gbufT and, with
    factors, d and par).

    With ``factors`` (the export for the gradient) the kernel also stores each pair's
    pivot d (B, b/2), the block's permutation parity (B, 1) and the pair rows' final
    positions into src (src[r0 + rank] = block index of the row), so that
    P S P^T = L T L^T can be assembled from gbuf (see _pfinv); the arithmetic of the
    factorisation is untouched, so (sign, logabs) stay bit-identical.

    use_gt=False (latency mode) drops the transposed copy gbufT: its store is a
    register-tile transpose through shared memory, 256 KB for an 8192-row tile (the
    A100 has 164 KB), and the corrections read the (h, 8) gbuf tiles instead (the same
    8-term sums, so the value class is unchanged)."""
    r0, rest = _split_blk(r0, rest)
    slo_ref, src_ref, physn_ref, posn_ref, gbuf_ref = rest[1:6]
    gbufT_ref = rest[6] if use_gt else None
    no = 7 if use_gt else 6
    d_ref, par_ref = rest[no : no + 2] if factors else (None, None)
    del rest
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
    offs = [start - r0 for start, h in chunks]
    iotas = [lax.broadcasted_iota(jnp.int32, (h,), 0) for start, h in chunks]
    trs = [io + off for io, off in zip(iotas, offs)]
    rows = lambda start, h: pl.ds(start, h)
    # static in-bounds masks of the chunks overhanging the matrix (None: no overhang)
    inbs = [
        (start + io) < n if start + h > n else None
        for (start, h), io in zip(chunks, iotas)
    ]
    one = fld.rscalar(1.0)
    minus = fld.rscalar(-1.0)

    def masked(x, c):
        return x if inbs[c] is None else x & inbs[c]

    # rows above r0 (negative index) are dead padding, rows past n do not exist
    actives = [masked(tr >= 0, c) for c, tr in enumerate(trs)]
    unas = list(actives)
    ranks = [jnp.full((h,), -1, jnp.int32) for off, h in chunks]
    sign, logabs = read_sl(sl_ref, fld.k)

    def phys_row(a):
        """Physical row of the active index a: a scalar gather from the (read-only)
        phys map instead of a masked reduction over the tile (two fewer cross-warp
        reductions per pair step; the map itself is only loaded after the steps, for
        the physn scatter, so it holds no registers during them)."""
        if identity_phys:
            return r0 + a
        return phys_ref[r0 + a]

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
            if gbufT_ref is not None:
                g_p = ld(gbufT_ref, (pl.ds(ip * bi, bi), r0 + a))
            else:
                g_p = ld(gbuf_ref, (r0 + a, pl.ds(ip * bi, bi)))
            hps.append(vsum(Pm * g_p[None, :], axis=1))
        cols = []
        for c, (start, h) in enumerate(chunks):
            inb = inbs[c]
            if inb is None:
                row_a = ld(p_ref, (pr, rows(start, h)))
            else:
                row_a = mld(p_ref, (pr, rows(start, h)), mask=inb)
            col = -row_a + vsum(Gs[c] * h_a[None, :], axis=1)
            for ip in range(i):
                if gbufT_ref is not None:
                    gt_idx = (pl.ds(ip * bi, bi), rows(start, h))
                    if inb is None:
                        Gt = ld(gbufT_ref, gt_idx)
                    else:
                        Gt = mld(gbufT_ref, gt_idx, mask=inb[None, :])
                    col = col + vsum(Gt * hps[ip][:, None], axis=0)
                else:
                    g_idx = (rows(start, h), pl.ds(ip * bi, bi))
                    if inb is None:
                        Gi = ld(gbuf_ref, g_idx)
                    else:
                        Gi = mld(gbuf_ref, g_idx, mask=inb[:, None])
                    col = col + vsum(Gi * hps[ip][None, :], axis=1)
            cols.append(col)
        return cols

    for i in range(nin):

        def step(s_, carry):
            Gs, unas, ranks, sign, logabs = carry
            sg = i * hi + s_
            # a = the lowest unassigned row, p = the largest corrected entry of its
            # column among the other unassigned rows
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
            sign = sign * unit(d)
            logabs = logabs + jnp.log(mag(d))
            if d_ref is not None:
                for r, comp in zip(d_ref, parts(d)):
                    plgpu.store(r.at[pl.ds(sg, 1)], comp[None])
            ranks = [
                jnp.where(tr == a, 2 * sg, jnp.where(tr == p, 2 * sg + 1, rk))
                for tr, rk in zip(trs, ranks)
            ]
            return Gs, newmask, ranks, sign, logabs

        G0 = [fld.zeros((h, bi)) for off, h in chunks]
        carry = (G0, unas, ranks, sign, logabs)
        Gs, unas, ranks, sign, logabs = lax.fori_loop(0, hi, step, carry)
        for c, (start, h) in enumerate(chunks):
            g_idx = (rows(start, h), pl.ds(i * bi, bi))
            gt_idx = (pl.ds(i * bi, bi), rows(start, h))
            inb = inbs[c]
            if inb is None:
                st(gbuf_ref, g_idx, Gs[c])
                if gbufT_ref is not None:
                    st(gbufT_ref, gt_idx, Gs[c].T)
            else:
                mst(gbuf_ref, g_idx, Gs[c], mask=inb[:, None])
                if gbufT_ref is not None:
                    mst(gbufT_ref, gt_idx, Gs[c].T, mask=inb[None, :])
        plgpu.debug_barrier()

    if identity_phys:
        physv = [r0 + tr for tr in trs]
    else:
        physv = []
        for c, (start, h) in enumerate(chunks):
            if inbs[c] is None:
                physv.append(phys_ref[rows(start, h)])
            else:
                physv.append(plgpu.load(phys_ref.at[rows(start, h)], mask=inbs[c]))
    # ---- parity of the permutation
    # [assigned rows in assignment order, unassigned rows in index order]
    cb = lax.broadcasted_iota(jnp.int32, (b,), 0)
    inv1 = 0
    ordb = jnp.zeros((b,), jnp.int32)
    cnt_a = 0
    cnt_u = 0
    s0 = r0 + b
    for c, (start, h) in enumerate(chunks):
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
        posn = jnp.where(unas[c], s0 + rank_u, -1)
        if inbs[c] is None:
            posn_ref[rows(start, h)] = posn
        else:
            plgpu.store(posn_ref.at[rows(start, h)], posn, mask=inbs[c])
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
    *rest,
    fld,
    n,
    r0,
    b,
    bi,
    tm,
    tn,
    nt,
    prec,
    skip,
    fresh=False,
    use_gt=True,
):
    """Lower-triangle tiles (compact-next i >= j) of M'' = M_upd[src_i, src_j]; stored
    twice into X_{k+1}: X_{k+1}[src_j, i] = -M''[i, j] (transposed) and
    X_{k+1}[src_i, j] = M''[i, j] (mirror). Each program handles nt consecutive row
    tiles of one column strip (a lax.fori_loop for nt > 1), so the gathered H^T strip
    is built once per program; with skip=True the row tiles start at the diagonal
    tile. ``rest`` = [gbufT (use_gt)] src, physn, posn, [block offset ref (latency
    mode)] [Q (unless fresh)] out; fresh writes into the output itself (no Q yet). r0
    static or from the ref; without gbufT the H^T strip is gathered from gbuf."""
    gbufT_ref = rest[0] if use_gt else None
    rest = rest[1:] if use_gt else rest
    src_ref, physn_ref, rest = rest[0], rest[1], rest[3:]  # posn unused here
    r0, rest = _split_blk(r0, rest)
    dyn = not isinstance(r0, int)
    q_ref = rest[-1] if fresh else rest[0]
    del rest
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
        # (b, tn) gather of H^T (from the transposed copy, or from gbuf itself)
        if gbufT_ref is not None:
            HcT = sgn[:, None] * ld(gbufT_ref, (perm[:, None], srcj[None, :]))
        else:
            HcT = sgn[:, None] * ld(gbuf_ref, (srcj[None, :], perm[:, None]))

        def tile(i0):
            rvalid = (i0 + ri) < mrem
            srci = plgpu.load(src_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
            Gs = ld(gbuf_ref, (srci, slice(None)))  # (tm, b)
            phys_blk = physn_ref.at[pl.ds(s0 + i0, tm)]
            phys = plgpu.load(phys_blk, mask=rvalid, other=n - 1)
            C = ld(p_ref, (phys[:, None], srcj[None, :]))  # (tm, tn) 2-D gather
            out = C + dot(Gs, HcT, prec)  # M''[i, j]
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
        live = i_start < mrem
        if dyn:  # class grid: strips past the last column are empty
            live = live & (j0 < mrem)
        pl.when(live)(body)
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
    blk,
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
        r0=blk.r0,
        b=b,
        bi=bi,
        layout=layout,
        identity_phys=fresh,
        factors=factors,
        use_gt=not blk.lat,
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
    ]
    if not blk.lat:  # the transposed copy (see _pf_panel_kernel.use_gt)
        outs.append((fld.structs((B, b, n)), gbufT_spec))
    if factors:
        par = jax.ShapeDtypeStruct((B, 1), jnp.int32)
        outs += [(fld.structs((B, b // 2)), _vec(b // 2)), (par, _vec(1))]
    if fresh:

        def wrapper(p_ref, *rest):
            # rest = [blk] po, slo, src, physn, posn, gbuf, gbufT [, d, par]
            nb_ = 1 if blk.lat else 0
            blk_refs, po_ref, slo_ref, src_ref = rest[:nb_], *rest[nb_ : nb_ + 3]
            init_sl(slo_ref, fld)
            kern(p_ref, src_ref, slo_ref, *blk_refs, po_ref, slo_ref, *rest[nb_ + 2 :])

        wrapper.__name__ = "_pf_panel_fresh"
        ins = [(P, _full(n))] + blk.ins()
        return _pcall(wrapper, ins, outs, (B,), aliases={0: 0}, num_warps=num_warps)
    ins = [(P, _full(n)), (phys, _vec(n)), (sl, _vec(fld.k + 1))] + blk.ins()
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
    blk,
    b,
    bi,
    tm,
    tn,
    prec,
    num_warps,
    fresh,
    skip,
    nt=1,
):
    B, n, _ = P[0].shape
    mrem = blk.mp - b
    nrt = -(-mrem // tm)
    nct = -(-mrem // tn)
    if not blk.lat:
        nt = min(nt, nrt)  # latency mode: the dynamic trip count handles short strips
    nrt = -(-nrt // nt)
    shapes = dict(n=n, r0=blk.r0, b=b, bi=bi, tm=tm, tn=tn, nt=nt, prec=prec)
    shapes.update(skip=skip, fresh=fresh, use_gt=gbufT is not None)
    kern = functools.partial(_pf_update_kernel, fld=fld, **shapes)
    gbuf_spec = pl.BlockSpec((None, n, b), lambda *i: (i[0], 0, 0))
    gbufT_spec = pl.BlockSpec((None, b, n), lambda *i: (i[0], 0, 0))
    ins = [(P, _full(n)), (gbuf, gbuf_spec)]
    if gbufT is not None:
        ins.append((gbufT, gbufT_spec))
    ins += [(src, _vec(n)), (physn, _vec(n)), (posn, _vec(n))] + blk.ins()
    outs = [(fld.structs((B, n, n)), _full(n))]
    grid = (B, blk.grid(nrt), blk.grid(nct))
    if fresh:  # no Q yet: q_ref is the output itself
        return _pcall(kern, ins, outs, grid, num_warps=num_warps)[0]
    ins.append((Q, _full(n)))
    q_index = len(ins) - 1
    return _pcall(kern, ins, outs, grid, aliases={q_index: 0}, num_warps=num_warps)[0]


def _pf_core(A, n, fld, prec, upd_warps=None):
    """Blocked Parlett-Reid of the skew-symmetric (B, n, n) batch given as parts;
    returns (sign, logabs) as jnp arrays (sign complex for complex fields).
    upd_warps=None selects the architecture table (_common.Tune)."""
    sign, logabs, _ = _pf_run(A, n, fld, prec, upd_warps, False)
    return sign, logabs


def _pf_core_factors(A, n, fld, prec, upd_warps=None):
    """_pf_core plus the raw material of the block LDL^T P S P^T = L D L^T for the
    gradient (see _pfinv._pf_parts): per block the tau / w buffer gbuf (B, N, b), the
    row map src (final-or-next index -> block index), the pivots (B, b/2) and the
    permutation parity (B, 1), plus b and N. The same arithmetic as the plain forward
    (its kernels only store more), so (sign, logabs) are bit-identical to it."""
    sign, logabs, fac = _pf_run(A, n, fld, prec, upd_warps, True)
    assert fac is not None
    return sign, logabs, fac


def _pf_run(A, n, fld, prec, upd_warps, factors):
    t = _tune(fld.kind)
    tm, tn = t.pf_tm, t.pf_tn
    upd_warps = t.pf_upd_warps if upd_warps is None else upd_warps
    nt = t.pf_upd_nt
    b, bi = BLOCK, INNER
    lat = n > t.latency_min_n  # see Tune.latency_min_n
    P, nb = _embed(A, n, _skew_pad, fld)
    N = P[0].shape[1]
    Q = sl = gbuf = gbufT = phys = None
    gbufs, srcs, ds, pars = [], [], [], []
    for k in range(nb):
        r0 = k * b
        last = k == nb - 1
        m = N - r0
        if lat:
            layout = _lat_layout(m, r0, N)
            pw = _lat_warps(layout[0][1], t, pf=True)
        else:
            rel = _layout(m, r0, t.pf_chunk_cost, t.pf_max_chunk, t.pf_split_above)
            layout = tuple((r0 + off, h) for off, h in rel)
            pw = _panel_warps(sum(h for _, h in layout), t, pf=True)
        blk = _Blk(r0, lat, layout[0][1] if lat else m)
        panel_kw = dict(fld=fld, b=b, bi=bi, layout=layout, num_warps=pw)
        upd_kw = dict(fld=fld, b=b, bi=bi, tm=tm, tn=tn, prec=prec)
        upd_kw.update(num_warps=upd_warps, nt=nt)
        outs = _pf_panel_call(
            P,
            phys,
            sl,
            gbuf,
            gbufT,
            blk=blk,
            fresh=(k == 0),
            factors=factors,
            **panel_kw,
        )
        P, sl, src, physn, posn, gbuf = outs[:6]
        gbufT = None if lat else outs[6]  # no transposed copy in the latency mode
        no = 6 if lat else 7
        if factors:
            gbufs.append(gbuf)
            srcs.append(src)
            ds.append(outs[no])
            pars.append(outs[no + 1])
        if not last:
            upd_args = (P, gbuf, gbufT, src, physn, posn, Q)
            Q = _pf_update_call(*upd_args, blk=blk, fresh=(k == 0), skip=True, **upd_kw)
            P, Q = Q, P
            phys = src  # phi_{k+1} = src_k
    assert sl is not None  # nb >= 1: the panel kernel always ran
    sign, logabs = _sl_sign(sl, fld), sl[:, fld.k]
    if factors:
        return sign, logabs, (gbufs, srcs, ds, pars, b, N)
    return sign, logabs, None
