"""Blocked Parlett-Reid tridiagonalisation with partial pivoting (Pallas Triton kernels) for slogpf: pair steps
in register tiles, rank-2 updates as one K=32 GEMM over the lower-triangle tiles with mirrored stores."""
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from ._common import (f32, BLOCK, INNER, PF_CHUNK_COST, _layout, _argmax_chunks, _dot, _full, _vec, _embed,
                      _panel_warps, _skew_pad)


# ============================================================================= pfaffian kernels
# Layout invariant per block k (u, v in the block's compact numbering [r0, n)):
#     X_k[phi_k(u), v] = M'_k[u, v]      rows physically permuted by phi_k (= src_{k-1}), columns compact.
# The panel reads row a as X_k[phi_k(a), :]; the update computes the lower-triangle tiles of
# M''[i, j] = M_upd[src_k(i), src_k(j)] and stores X_{k+1}[src_j, i] = -M''[i, j] and X_{k+1}[src_i, j] = M''[i, j]
# (skew symmetry), so no separate column-compaction pass is needed.

def _pf_panel_kernel(p_ref, phys_ref, sl_ref, po_ref, slo_ref, src_ref, physn_ref, posn_ref, gbuf_ref, gbufT_ref,
                     *, n, r0, b, bi, layout, identity_phys=False):
    """Parlett-Reid block step on the m = n - r0 active indices (chunked register tiles). Row a of the current matrix
    is X[phi(a), :]; columns are compact. tau/w vectors are kept in G (m x 8 per inner block) and stored as rows of
    gbuf/gbufT for the update kernel and later inner blocks."""
    del po_ref
    m = n - r0
    chunks = list(layout)
    nin = b // bi
    hi = bi // 2
    ci = lax.broadcasted_iota(jnp.int32, (bi,), 0)
    ci2 = ci[None, :]
    ib = lax.broadcasted_iota(jnp.int32, (bi, bi), 0)
    jb = lax.broadcasted_iota(jnp.int32, (bi, bi), 1)
    Pm = jnp.where((ib < hi) & (jb == ib + hi), -1.0, jnp.where((ib >= hi) & (jb == ib - hi), 1.0, 0.0)).astype(f32)
    offs = [off for off, h in chunks]
    trs = [lax.broadcasted_iota(jnp.int32, (h,), 0) + off for off, h in chunks]
    rows = lambda off, h: pl.ds(r0 + off, h)

    actives = [tr >= 0 for tr in trs]                     # rows above r0 (negative index) are dead padding
    unas = list(actives)
    ranks = [jnp.full((h,), -1, jnp.int32) for off, h in chunks]
    sign = sl_ref[0]
    logabs = sl_ref[1]
    physv = [(r0 + tr) if identity_phys else phys_ref[rows(off, h)] for tr, (off, h) in zip(trs, chunks)]

    def phys_row(a):
        if identity_phys:
            return r0 + a
        return sum(jnp.sum(jnp.where(tr == a, pv, 0)) for tr, pv in zip(trs, physv))

    def corrected_col(a, Gs, i):
        """updated column a of the trailing matrix (all active rows, chunked): -row_a + corrections of this block's
        earlier pairs (register G) and of earlier inner blocks (gbufT rows)."""
        pr = phys_row(a)
        g_a = sum(jnp.sum(jnp.where(tr[:, None] == a, G, 0.0), axis=0) for tr, G in zip(trs, Gs))
        h_a = jnp.sum(Pm * g_a[None, :], axis=1)
        hps = []
        for ip in range(i):
            g_p = gbufT_ref[pl.ds(ip * bi, bi), r0 + a]
            hps.append(jnp.sum(Pm * g_p[None, :], axis=1))
        cols = []
        for c, (off, h) in enumerate(chunks):
            row_a = p_ref[pr, rows(off, h)]
            col = -row_a + jnp.sum(Gs[c] * h_a[None, :], axis=1)
            for ip in range(i):
                Gt = gbufT_ref[pl.ds(ip * bi, bi), rows(off, h)]
                col = col + jnp.sum(Gt * hps[ip][:, None], axis=0)
            cols.append(col)
        return cols

    for i in range(nin):
        def step(s_, carry):
            Gs, unas, ranks, sign, logabs = carry
            sg = i * hi + s_
            a = functools.reduce(jnp.minimum, [jnp.min(jnp.where(un, tr, m)) for tr, un in zip(trs, unas)])
            col_a = corrected_col(a, Gs, i)
            cands = [jnp.where(un & (tr != a), jnp.abs(col), -1.0) for tr, un, col in zip(trs, unas, col_a)]
            p = _argmax_chunks(cands, offs)
            d = -sum(jnp.sum(jnp.where(tr == p, col, 0.0)) for tr, col in zip(trs, col_a))        # M[a, p]
            col_p = corrected_col(p, Gs, i)
            newmask = [un & (tr != a) & (tr != p) for tr, un in zip(trs, unas)]
            inv = lax.select(d == 0.0, f32(0.0), 1.0 / d)                                          # zero pivot guard
            Gs = [jnp.where(ci2 == s_, jnp.where(nm, -ca * inv, 0.0)[:, None],
                            jnp.where(ci2 == hi + s_, -jnp.where(nm, cp, 0.0)[:, None], G))
                  for G, nm, ca, cp in zip(Gs, newmask, col_a, col_p)]
            ranks = [jnp.where(tr == a, 2 * sg, jnp.where(tr == p, 2 * sg + 1, rk)) for tr, rk in zip(trs, ranks)]
            sign = sign * jnp.sign(d)
            logabs = logabs + jnp.log(jnp.abs(d))
            return Gs, newmask, ranks, sign, logabs

        carry = ([jnp.zeros((h, bi), f32) for off, h in chunks], unas, ranks, sign, logabs)
        Gs, unas, ranks, sign, logabs = lax.fori_loop(0, hi, step, carry)
        for c, (off, h) in enumerate(chunks):
            gbuf_ref[rows(off, h), pl.ds(i * bi, bi)] = Gs[c]
            gbufT_ref[pl.ds(i * bi, bi), rows(off, h)] = Gs[c].T
        plgpu.debug_barrier()

    # ---- parity of the permutation [assigned rows in assignment order, unassigned rows in index order]
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
        inv1 = inv1 + jnp.sum(jnp.where(assigned, trs[c] - rank_a, 0))
        ordb = ordb + jnp.sum(jnp.where((rank_a[None, :] == cb[:, None]) & assigned[None, :], ranks[c][None, :], 0), axis=1)
        cnt_a = cnt_a + jnp.sum(assigned.astype(jnp.int32))
        cnt_u = cnt_u + jnp.sum(unas[c].astype(jnp.int32))
        # src: compact-next index -> this block's index; physn: compact-next -> physical row of X_k; posn: inverse map
        plgpu.store(src_ref.at[s0 + rank_u], r0 + trs[c], mask=unas[c])
        plgpu.store(physn_ref.at[s0 + rank_u], physv[c], mask=unas[c])
        posn_ref[rows(off, h)] = jnp.where(unas[c], s0 + rank_u, -1)
    inv2 = jnp.sum(jnp.where((cb[:, None] < cb[None, :]) & (ordb[:, None] > ordb[None, :]), 1, 0))
    sign = sign * lax.select(((inv1 + inv2) % 2) == 1, f32(-1.0), f32(1.0))
    slo_ref[0] = sign
    slo_ref[1] = logabs


def _pf_update_kernel(p_ref, gbuf_ref, gbufT_ref, src_ref, physn_ref, posn_ref, q_ref, out_ref, *, n, r0, b, bi, tm, tn, prec, skip):
    """Lower-triangle tiles (compact-next i >= j) of M'' = M_upd[src_i, src_j]; stored twice into X_{k+1}:
    X_{k+1}[src_j, i] = -M''[i, j] (transposed) and X_{k+1}[src_i, j] = M''[i, j] (mirror)."""
    del out_ref, posn_ref
    hi = bi // 2
    s0 = r0 + b
    mrem = n - s0
    i0 = pl.program_id(1) * tm
    j0 = pl.program_id(2) * tn
    ri = lax.broadcasted_iota(jnp.int32, (tm,), 0)
    cj = lax.broadcasted_iota(jnp.int32, (tn,), 0)
    rvalid = (i0 + ri) < mrem
    cvalid = (j0 + cj) < mrem

    def body():
        srci = plgpu.load(src_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
        srcj = plgpu.load(src_ref.at[pl.ds(s0 + j0, tn)], mask=cvalid, other=n - 1)
        phys = plgpu.load(physn_ref.at[pl.ds(s0 + i0, tm)], mask=rvalid, other=n - 1)
        cb = lax.broadcasted_iota(jnp.int32, (b,), 0)
        within = cb % bi
        perm = jnp.where(within < hi, cb + hi, cb - hi)
        sgn = jnp.where(within < hi, -1.0, 1.0).astype(f32)
        Gs = gbuf_ref[srci, :]                                                                         # (tm, b)
        HcT = sgn[:, None] * gbufT_ref[perm[:, None], srcj[None, :]]                                  # (b, tn) 2-D gather
        C = p_ref[phys[:, None], srcj[None, :]]                                                        # (tm, tn) 2-D gather
        out = C + _dot(Gs, HcT, prec)                                                                   # M''[i, j]
        if tm == tn:
            # diagonal tiles: make the tile exactly skew-symmetric (lower part authoritative) so the primary and mirror
            # stores agree bit-for-bit and the stored matrix stays exactly skew (the panel reads columns as rows)
            skew = jnp.where(ri[:, None] >= cj[None, :], out, -out.T)
            out = jnp.where(i0 == j0, skew, out)
        mask = rvalid[:, None] & cvalid[None, :]
        plgpu.store(q_ref.at[srcj, pl.ds(s0 + i0, tm)], -out.T, mask=cvalid[:, None] & rvalid[None, :])
        plgpu.store(q_ref.at[srci, pl.ds(s0 + j0, tn)], out, mask=mask)

    if skip:
        pl.when(i0 + tm - 1 >= j0)(body)
    else:
        body()


def _pf_panel_call(P, phys, sl, gbuf, gbufT, *, r0, b, bi, layout, num_warps, fresh):
    B, n, _ = P.shape
    kern = functools.partial(_pf_panel_kernel, n=n, r0=r0, b=b, bi=bi, layout=layout, identity_phys=fresh)
    out_specs = [_full(n), _vec(2), _vec(n), _vec(n), _vec(n), pl.BlockSpec((None, n, b), lambda i: (i, 0, 0)),
                 pl.BlockSpec((None, b, n), lambda i: (i, 0, 0))]
    out_shape = [jax.ShapeDtypeStruct((B, n, n), f32), jax.ShapeDtypeStruct((B, 2), f32),
                 jax.ShapeDtypeStruct((B, n), jnp.int32), jax.ShapeDtypeStruct((B, n), jnp.int32),
                 jax.ShapeDtypeStruct((B, n), jnp.int32),
                 jax.ShapeDtypeStruct((B, n, b), f32), jax.ShapeDtypeStruct((B, b, n), f32)]
    if fresh:
        def wrapper(p_ref, po_ref, slo_ref, src_ref, physn_ref, posn_ref, gbuf_ref, gbufT_ref):
            slo_ref[0] = f32(1.0)
            slo_ref[1] = f32(0.0)
            kern(p_ref, src_ref, slo_ref, po_ref, slo_ref, src_ref, physn_ref, posn_ref, gbuf_ref, gbufT_ref)
        return pl.pallas_call(wrapper, grid=(B,), in_specs=[_full(n)], out_specs=out_specs, out_shape=out_shape,
                              input_output_aliases={0: 0},
                              compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1))(P)
    return pl.pallas_call(kern, grid=(B,), in_specs=[_full(n), _vec(n), _vec(2)], out_specs=out_specs, out_shape=out_shape,
                          input_output_aliases={0: 0, 2: 1},
                          compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1))(P, phys, sl)


def _pf_update_call(P, gbuf, gbufT, src, physn, posn, Q, *, r0, b, bi, tm, tn, prec, num_warps, fresh, skip):
    B, n, _ = P.shape
    nrt = -(-(n - r0 - b) // tm)
    nct = -(-(n - r0 - b) // tn)
    kern = functools.partial(_pf_update_kernel, n=n, r0=r0, b=b, bi=bi, tm=tm, tn=tn, prec=prec, skip=skip)
    in_specs = [_full(n), pl.BlockSpec((None, n, b), lambda *i: (i[0], 0, 0)), pl.BlockSpec((None, b, n), lambda *i: (i[0], 0, 0)),
                _vec(n), _vec(n), _vec(n)]
    if fresh:
        def wrapper(p_ref, gbuf_ref, gbufT_ref, src_ref, physn_ref, posn_ref, out_ref):
            kern(p_ref, gbuf_ref, gbufT_ref, src_ref, physn_ref, posn_ref, out_ref, out_ref)
        return pl.pallas_call(wrapper, grid=(B, nrt, nct), in_specs=in_specs, out_specs=_full(n),
                              out_shape=jax.ShapeDtypeStruct((B, n, n), f32),
                              compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1))(P, gbuf, gbufT, src, physn, posn)
    return pl.pallas_call(kern, grid=(B, nrt, nct), in_specs=in_specs + [_full(n)], out_specs=_full(n),
                          out_shape=jax.ShapeDtypeStruct((B, n, n), f32), input_output_aliases={6: 0},
                          compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1))(P, gbuf, gbufT, src, physn, posn, Q)


def _pf_core(A, n, prec, tm=64, tn=64, upd_warps=4):
    b, bi = BLOCK, INNER
    P, nb = _embed(A, n, _skew_pad)
    N = P.shape[1]
    Q = sl = gbuf = gbufT = phys = None
    for k in range(nb):
        r0 = k * b
        last = k == nb - 1
        layout = tuple(_layout(N - r0, r0, PF_CHUNK_COST))
        pw = _panel_warps(sum(h for _, h in layout))
        P, sl, src, physn, posn, gbuf, gbufT = _pf_panel_call(P, phys, sl, gbuf, gbufT, r0=r0, b=b, bi=bi, layout=layout, num_warps=pw, fresh=(k == 0))
        if not last:
            Q = _pf_update_call(P, gbuf, gbufT, src, physn, posn, Q, r0=r0, b=b, bi=bi, tm=tm, tn=tn, prec=prec, num_warps=upd_warps,
                                fresh=(k == 0), skip=True)
            P, Q = Q, P
            phys = src            # phi_{k+1} = src_k
    return sl[:, 0], sl[:, 1]
